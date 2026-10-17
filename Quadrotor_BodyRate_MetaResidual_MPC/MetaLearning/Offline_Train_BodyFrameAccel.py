"""Meta-train body-frame acceleration residuals with latent-only adaptation.

The shared network is optimized offline.  For each task, only a six-dimensional
latent ``z`` is updated on a recent support window; the outer loss evaluates
that adapted latent on a later, disjoint query window.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bodyrate_common import (  # noqa: E402
    COMMAND_MODE_BODY_RATE_THRUST,
    CONTROL_COLS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CTRL_FREQ,
    DEFAULT_DATASET_PATH,
    MODEL_INPUT_DIM,
    RESIDUAL_OUTPUT_DIM,
    STATE_COLS,
    BodyFrameAccelerationResidualMLP,
)


TARGET_COLS = ("res_bx_ddot", "res_by_ddot", "res_bz_ddot")
DT = 1.0 / DEFAULT_CTRL_FREQ
VELOCITY_IDXS = (1, 3, 5)
POSITION_IDXS = (0, 2, 4)


def load_frame(path: Path) -> pd.DataFrame:
    if not path.is_dir():
        return pd.read_csv(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest.get("summary", {})
    if summary.get("complete_task_count") != summary.get("expected_task_count"):
        raise RuntimeError("Dataset shard set is incomplete; resume collection before training.")
    shards = sorted(path.glob("shard-*/data.csv"))
    if not shards:
        raise FileNotFoundError(f"No data.csv shards under {path}")
    return pd.concat((pd.read_csv(shard) for shard in shards), ignore_index=True)


def load_tasks(path: Path):
    frame = load_frame(path)
    required = {"task_id", "episode_id", "time", *STATE_COLS, *CONTROL_COLS, *TARGET_COLS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(
            "Dataset is not body-frame-acceleration format. Recollect it with "
            f"DataCollection_BodyRate.py; missing columns: {missing}"
        )
    tasks = {}
    all_features, all_targets = [], []
    for task_id, task_frame in frame.groupby("task_id", sort=False):
        rollouts = []
        for _, episode in task_frame.groupby("episode_id", sort=False):
            episode = episode.sort_values("time").reset_index(drop=True)
            if len(episode) < 3:
                continue
            states = torch.tensor(episode[STATE_COLS].to_numpy(np.float32), dtype=torch.float32)
            actions = torch.tensor(episode[CONTROL_COLS].to_numpy(np.float32), dtype=torch.float32)
            targets = torch.tensor(episode[list(TARGET_COLS)].to_numpy(np.float32), dtype=torch.float32)
            # Every row records the pre-transition state/action.  The final
            # row has no following row inside this episode.
            features = torch.cat((states[:-1], actions[:-1]), dim=-1)
            rollouts.append({"features": features, "targets": targets[:-1], "states": states[:-1], "actions": actions[:-1], "next_states": states[1:]})
            all_features.append(features)
            all_targets.append(targets[:-1])
        if rollouts:
            tasks[int(task_id)] = rollouts
    if len(tasks) < 2:
        raise RuntimeError("At least two complete dynamics tasks are required for a task-level meta split.")
    features, targets = torch.cat(all_features), torch.cat(all_targets)
    return tasks, features.mean(0), features.std(0, unbiased=False).clamp_min(1e-6), targets.mean(0), targets.std(0, unbiased=False).clamp_min(1e-6)


def body_to_world_torch(state):
    phi, theta, psi = state[..., 6], state[..., 7], state[..., 8]
    cp, sp, ct, st, cy, sy = torch.cos(phi), torch.sin(phi), torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
    return torch.stack((
        torch.stack((ct * cy, -cp * sy + sp * st * cy, sp * sy + cp * st * cy), -1),
        torch.stack((ct * sy, cp * cy + sp * st * sy, -sp * cy + cp * st * sy), -1),
        torch.stack((-st, sp * ct, cp * ct), -1),
    ), -2)


def nominal_derivative(state, action):
    phi, theta, psi = state[..., 6], state[..., 7], state[..., 8]
    p, q, r = state[..., 9], state[..., 10], state[..., 11]
    thrust = action[..., 3]
    m, g = 0.027 * 0.66, 9.8
    cp, sp, ct, st, cy, sy = torch.cos(phi), torch.sin(phi), torch.cos(theta).clamp(min=1e-3), torch.sin(theta), torch.cos(psi), torch.sin(psi)
    acceleration = torch.stack(((sp * sy + cp * st * cy) * thrust / m, (-sp * cy + cp * st * sy) * thrust / m, cp * ct * thrust / m - g), -1)
    euler = torch.stack((p + q * sp * st / ct + r * cp * st / ct, q * cp - r * sp, q * sp / ct + r * cp / ct), -1)
    tau = state.new_tensor((0.08, 0.08, 0.12))
    rate_dot = (action[..., :3] - state[..., 9:12]) / tau
    result = torch.zeros_like(state)
    result[..., POSITION_IDXS] = state[..., VELOCITY_IDXS]
    result[..., VELOCITY_IDXS] = acceleration
    result[..., 6:9] = euler
    result[..., 9:12] = rate_dot
    return result


def velocity_support_loss(model, features, next_states, z, residual_targets=None, latent_weight=1e-3):
    state, action = features[..., :12], features[..., 12:]
    body_residual = model.forward_with_context(features, z)
    world_residual = torch.bmm(body_to_world_torch(state), body_residual.unsqueeze(-1)).squeeze(-1)
    predicted_velocity = state[..., VELOCITY_IDXS] + DT * (nominal_derivative(state, action)[..., VELOCITY_IDXS] + world_residual)
    velocity_loss = F.mse_loss((predicted_velocity - next_states[..., VELOCITY_IDXS]) / 0.25, torch.zeros_like(predicted_velocity))
    label_loss = body_residual.new_tensor(0.0) if residual_targets is None else F.smooth_l1_loss(body_residual, residual_targets, beta=1.0)
    return velocity_loss + 0.10 * label_loss + latent_weight * (z - model.context_init).pow(2).mean(), velocity_loss, label_loss


def rollout_query_loss(model, window, z):
    predicted = window["states"][0:1]
    pos_losses, vel_losses = [], []
    for action, target in zip(window["actions"], window["next_states"]):
        action = action.unsqueeze(0)
        features = torch.cat((predicted, action), dim=-1)
        body_residual = model.forward_with_context(features, z)
        derivative = nominal_derivative(predicted, action)
        derivative[..., VELOCITY_IDXS] += torch.bmm(body_to_world_torch(predicted), body_residual.unsqueeze(-1)).squeeze(-1)
        predicted = predicted + DT * derivative
        predicted[..., 6:9] = torch.remainder(predicted[..., 6:9] + np.pi, 2 * np.pi) - np.pi
        pos_losses.append(((predicted[..., POSITION_IDXS] - target[POSITION_IDXS]) / 0.05).pow(2).mean())
        vel_losses.append(((predicted[..., VELOCITY_IDXS] - target[VELOCITY_IDXS]) / 0.25).pow(2).mean())
    return torch.stack(pos_losses).mean() + torch.stack(vel_losses).mean()


def support_then_query(model, rollout, support_window, query_horizon, rng, create_graph):
    max_start = rollout["features"].shape[0] - support_window - query_horizon
    if max_start < 0:
        return None
    start = int(rng.integers(max_start + 1))
    support = {name: value[start:start + support_window] for name, value in rollout.items()}
    q0 = start + support_window
    query = {name: value[q0:q0 + query_horizon] for name, value in rollout.items()}
    z = model.context_init.unsqueeze(0).expand(1, -1)
    for _ in range(2):
        support_loss, _, _ = velocity_support_loss(model, support["features"], support["next_states"], z, support["targets"])
        gradient, = torch.autograd.grad(support_loss, z, create_graph=create_graph)
        z = z - 0.05 * gradient
    query_support, query_velocity, query_label = velocity_support_loss(model, query["features"], query["next_states"], z, query["targets"], latent_weight=0.0)
    return query_support + 0.25 * rollout_query_loss(model, query, z), query_velocity, query_label


def evaluate(model, tasks, task_ids, support_window, query_horizon, rng, create_graph):
    losses, velocities, labels = [], [], []
    for task_id in task_ids:
        candidates = [r for r in tasks[task_id] if r["features"].shape[0] >= support_window + query_horizon]
        if not candidates:
            continue
        result = support_then_query(model, candidates[int(rng.integers(len(candidates)))], support_window, query_horizon, rng, create_graph)
        if result is not None:
            losses.append(result[0]); velocities.append(result[1]); labels.append(result[2])
    if not losses:
        return None
    return torch.stack(losses).mean(), torch.stack(velocities).mean(), torch.stack(labels).mean()


def main():
    parser = argparse.ArgumentParser(description="Meta-train a body-frame acceleration residual with latent-only online adaptation.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--meta-batch-size", type=int, default=24)
    parser.add_argument("--support-window", type=int, default=32)
    parser.add_argument("--query-horizon", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tasks, input_mean, input_scale, target_mean, target_scale = load_tasks(args.csv_path)
    ids = np.array(sorted(tasks))
    rng = np.random.default_rng(args.seed); rng.shuffle(ids)
    split = max(1, int(round(0.2 * len(ids))))
    val_ids, train_ids = ids[:split].tolist(), ids[split:].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BodyFrameAccelerationResidualMLP(context_dim=args.latent_dim).to(device)
    model.set_normalization(input_mean, input_scale, target_mean, target_scale)
    # Data tensors stay small; moving them once avoids accidentally detaching
    # the inner update while reconstructing support/query batches.
    for rollouts in tasks.values():
        for rollout in rollouts:
            for name, value in rollout.items(): rollout[name] = value.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_state, best_val, history = None, float("inf"), []
    train_rng = np.random.default_rng(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        batch = train_rng.choice(train_ids, min(args.meta_batch_size, len(train_ids)), replace=False).tolist()
        result = evaluate(model, tasks, batch, args.support_window, args.query_horizon, train_rng, create_graph=True)
        if result is None: raise RuntimeError("No rollout is long enough for support + subsequent query windows.")
        result[0].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            model.eval()
            val_result = evaluate(model, tasks, val_ids, args.support_window, args.query_horizon, np.random.default_rng(args.seed), create_graph=False)
            val_loss = float(val_result[0].detach()) if val_result else float(result[0].detach())
            history.append({"epoch": epoch, "train_loss": float(result[0].detach()), "val_loss": val_loss, "query_velocity": float(result[1].detach()), "query_label": float(result[2].detach())})
            if val_loss < best_val: best_val, best_state = val_loss, deepcopy(model.state_dict())
            print(f"epoch={epoch:04d} train={history[-1]['train_loss']:.5f} val={val_loss:.5f} velocity={history[-1]['query_velocity']:.5f}")
    model.load_state_dict(best_state or model.state_dict()); model.reset_context()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "latent_only_meta_residual", "architecture": "bodyframe_accel_latent_v1",
        "command_mode": COMMAND_MODE_BODY_RATE_THRUST, "model_state_dict": model.state_dict(),
        "input_dim": MODEL_INPUT_DIM, "output_dim": RESIDUAL_OUTPUT_DIM, "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers, "context_dim": args.latent_dim, "modulation_scale": model.modulation_scale,
        "input_mean": input_mean.tolist(), "input_scale": input_scale.tolist(), "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
        "support_window": args.support_window, "query_horizon": args.query_horizon,
        "online_context_refinement_steps": 2, "online_context_refinement_lr": 0.05,
        "online_context_ema_decay": 0.0, "online_context_warmup_updates": 0,
        "loss": "velocity_prediction + short_multistep_query", "best_val_loss": best_val,
    }, args.save_path)
    pd.DataFrame(history).to_csv(args.save_path.with_name(f"{args.save_path.stem}_loss_history.csv"), index=False)
    print(f"Saved latent-only body-frame checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
