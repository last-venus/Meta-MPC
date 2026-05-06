"""Train an amortized context-meta residual model for body-rate/thrust MPC."""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bodyrate_common import (  # noqa: E402
    COMMAND_MODE_BODY_RATE_THRUST,
    CONTEXT_COLS,
    CONTROL_COLS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CTRL_FREQ,
    DEFAULT_DATASET_PATH,
    DEFAULT_RATE_TIME_CONSTANTS,
    MODEL_INPUT_DIM,
    RESIDUAL_OUTPUT_DIM,
    STATE_COLS,
    NormalizedContextResidualMLP,
    NormalizedSupportSetEncoder,
)


DEFAULT_EPOCHS = 600
DEFAULT_META_BATCH_SIZE_CAP = 24
DEFAULT_SUPPORT_WINDOW = 32
DEFAULT_QUERY_HORIZON = 16
DEFAULT_VAL_INTERVAL = 25
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SPLIT_SEED = 43
DEFAULT_LOG_INTERVAL = 25

INPUT_COLS = [*STATE_COLS, *CONTROL_COLS]
TARGET_COLS = ["res_x_ddot", "res_y_ddot", "res_z_ddot", "res_p_dot", "res_q_dot", "res_r_dot"]
DATA_DT = 1.0 / DEFAULT_CTRL_FREQ

MODEL_CONFIG = {
    "hidden_dim": 128,
    "num_layers": 3,
    "context_encoder_hidden_dim": 96,
    "context_encoder_num_layers": 2,
    "context_injection": "adapter",
    "modulation_scale": 0.20,
}
TRAINING_CONFIG = {
    "meta_lr": 5e-4,
    "grad_clip_norm": 5.0,
    "online_context_ema_decay": 0.65,
    "online_context_warmup_updates": 1,
}
LOSS_CONFIG = {
    "query_residual_weight": 1.0,
    "query_state_weight": 0.25,
    "context_weight": 0.08,
    "consistency_weight": 0.08,
    "contrastive_weight": 0.05,
    "contrastive_temperature": 0.35,
    "position_weight": 1.4,
    "velocity_weight": 0.25,
    "attitude_weight": 0.25,
    "rate_weight": 0.08,
    "huber_beta": 1.0,
}
FILTER_CONFIG = {
    "max_linear_residual": 25.0,
    "max_rate_residual": 120.0,
    "max_body_rate_cmd": 2.98,
    "min_rollout_len": 96,
}


def wrap_angle_torch(angle):
    return torch.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def wrap_state_angles_torch(state):
    wrapped = state.clone()
    wrapped[..., 6:9] = wrap_angle_torch(wrapped[..., 6:9])
    return wrapped


def safe_cosine(theta, eps=1e-3):
    sign = torch.where(theta.cos() >= 0.0, torch.ones_like(theta), -torch.ones_like(theta))
    return sign * torch.clamp(theta.cos().abs(), min=eps)


def nominal_bodyrate_dynamics_torch(state, action):
    x_dot, y_dot, z_dot = state[..., 1], state[..., 3], state[..., 5]
    phi, theta, psi = state[..., 6], state[..., 7], state[..., 8]
    p_body, q_body, r_body = state[..., 9], state[..., 10], state[..., 11]
    p_cmd, q_cmd, r_cmd, thrust = action[..., 0], action[..., 1], action[..., 2], action[..., 3]
    m = 0.027 * 0.66
    g = 9.8
    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    c_theta, s_theta = safe_cosine(theta), torch.sin(theta)
    c_psi, s_psi = torch.cos(psi), torch.sin(psi)
    tan_theta = s_theta / c_theta
    pos_ddot_x = (s_phi * s_psi + c_phi * s_theta * c_psi) * thrust / m
    pos_ddot_y = (-s_phi * c_psi + c_phi * s_theta * s_psi) * thrust / m
    pos_ddot_z = c_phi * c_theta * thrust / m - g
    euler_rates = torch.stack(
        [
            p_body + q_body * s_phi * tan_theta + r_body * c_phi * tan_theta,
            q_body * c_phi - r_body * s_phi,
            q_body * s_phi / c_theta + r_body * c_phi / c_theta,
        ],
        dim=-1,
    )
    tau = torch.as_tensor(DEFAULT_RATE_TIME_CONSTANTS, dtype=state.dtype, device=state.device)
    body_rates_dot = torch.stack([(p_cmd - p_body) / tau[0], (q_cmd - q_body) / tau[1], (r_cmd - r_body) / tau[2]], dim=-1)
    return torch.stack(
        [
            x_dot,
            pos_ddot_x,
            y_dot,
            pos_ddot_y,
            z_dot,
            pos_ddot_z,
            euler_rates[..., 0],
            euler_rates[..., 1],
            euler_rates[..., 2],
            body_rates_dot[..., 0],
            body_rates_dot[..., 1],
            body_rates_dot[..., 2],
        ],
        dim=-1,
    )


def residual_to_state_derivative(residual):
    zeros = torch.zeros_like(residual[..., 0])
    return torch.stack([zeros, residual[..., 0], zeros, residual[..., 1], zeros, residual[..., 2], zeros, zeros, zeros, residual[..., 3], residual[..., 4], residual[..., 5]], dim=-1)


def rollout_step_torch(state, action, residual, dt):
    return wrap_state_angles_torch(state + dt * (nominal_bodyrate_dynamics_torch(state, action) + residual_to_state_derivative(residual)))


def one_step_state_loss(pred_next_state, target_next_state):
    pos_loss = (pred_next_state[:, [0, 2, 4]] - target_next_state[:, [0, 2, 4]]).pow(2).mean()
    vel_loss = (pred_next_state[:, [1, 3, 5]] - target_next_state[:, [1, 3, 5]]).pow(2).mean()
    att_loss = wrap_angle_torch(pred_next_state[:, 6:9] - target_next_state[:, 6:9]).pow(2).mean()
    rate_loss = (pred_next_state[:, 9:12] - target_next_state[:, 9:12]).pow(2).mean()
    return pos_loss, vel_loss, att_loss, rate_loss


def symmetric_contrastive_loss(context_a_batch, context_b_batch, temperature):
    if context_a_batch.size(0) <= 1:
        return context_a_batch.new_tensor(0.0)
    logits = F.normalize(context_a_batch, dim=-1) @ F.normalize(context_b_batch, dim=-1).transpose(0, 1) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def split_task_ids(task_ids, val_ratio=DEFAULT_VAL_RATIO, seed=DEFAULT_SPLIT_SEED):
    task_ids = np.array(sorted(task_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(task_ids)
    n_val = min(max(1, int(round(len(task_ids) * val_ratio))), len(task_ids) - 1)
    return set(task_ids[n_val:].tolist()), set(task_ids[:n_val].tolist())


def load_task_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"task_id", "episode_id", "reference_id", "time", *INPUT_COLS, *TARGET_COLS, *CONTEXT_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing dataset columns: {missing}")
    context_by_task = df.groupby("task_id", sort=False)[CONTEXT_COLS].first()
    context_center = torch.tensor(context_by_task.mean(axis=0).to_numpy(dtype=np.float32), dtype=torch.float32)
    context_scale = torch.tensor(context_by_task.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32), dtype=torch.float32)
    context_scale = torch.clamp(context_scale, min=1e-6)

    task_data = {}
    kept_samples = 0
    raw_samples = 0
    for task_id, task_df in df.groupby("task_id", sort=False):
        context_raw = torch.tensor(task_df[CONTEXT_COLS].iloc[0].to_numpy(dtype=np.float32), dtype=torch.float32)
        context = (context_raw - context_center) / context_scale
        rollouts = []
        for _, episode_df in task_df.groupby("episode_id", sort=False):
            episode_df = episode_df.sort_values("time").reset_index(drop=True)
            if len(episode_df) < 2:
                continue
            states = episode_df[STATE_COLS].to_numpy(dtype=np.float32, copy=True)
            targets = episode_df[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)[:-1]
            actions = episode_df[CONTROL_COLS].to_numpy(dtype=np.float32, copy=True)[:-1]
            raw_samples += len(targets)
            linear_ok = np.max(np.abs(targets[:, :3]), axis=1) <= FILTER_CONFIG["max_linear_residual"]
            rate_ok = np.max(np.abs(targets[:, 3:]), axis=1) <= FILTER_CONFIG["max_rate_residual"]
            cmd_ok = np.max(np.abs(actions[:, :3]), axis=1) <= FILTER_CONFIG["max_body_rate_cmd"]
            valid_mask = linear_ok & rate_ok & cmd_ok
            if int(valid_mask.sum()) < FILTER_CONFIG["min_rollout_len"]:
                continue
            kept_samples += int(valid_mask.sum())
            rollouts.append(
                {
                    "reference_id": int(episode_df["reference_id"].iloc[0]),
                    "features": torch.from_numpy(episode_df[INPUT_COLS].to_numpy(dtype=np.float32, copy=True)[:-1][valid_mask]),
                    "targets": torch.from_numpy(targets[valid_mask]),
                    "states": torch.from_numpy(states[:-1][valid_mask]),
                    "actions": torch.from_numpy(actions[valid_mask]),
                    "next_states": torch.from_numpy(states[1:][valid_mask]),
                }
            )
        if rollouts:
            task_data[int(task_id)] = {"context": context, "context_raw": context_raw, "rollouts": rollouts}
    if not task_data:
        raise RuntimeError("No valid rollouts remain after filtering. Loosen FILTER_CONFIG or collect cleaner data.")
    kept_features = torch.cat([rollout["features"] for task in task_data.values() for rollout in task["rollouts"]], dim=0)
    kept_targets = torch.cat([rollout["targets"] for task in task_data.values() for rollout in task["rollouts"]], dim=0)
    input_mean = kept_features.mean(dim=0)
    input_scale = kept_features.std(dim=0, unbiased=False).clamp(min=1e-6)
    target_mean = kept_targets.mean(dim=0)
    target_scale = kept_targets.std(dim=0, unbiased=False).clamp(min=1e-6)
    print(f"Filtered training samples: kept {kept_samples}/{raw_samples} ({100.0 * kept_samples / max(1, raw_samples):.1f}%).")
    return task_data, context_center, context_scale, input_mean, input_scale, target_mean, target_scale


def has_valid_rollout(task_rollouts, support_window, query_horizon):
    return any(r["features"].size(0) >= support_window for r in task_rollouts) and any(r["features"].size(0) >= query_horizon for r in task_rollouts)


def sample_window(rollout, window_size, rng, excluded_start=None):
    max_start = rollout["features"].size(0) - window_size
    if max_start < 0:
        raise ValueError("Rollout is shorter than requested window.")
    if max_start == 0:
        return 0, window_size
    start = int(rng.integers(0, max_start + 1))
    if excluded_start is not None and max_start >= 1 and start == excluded_start:
        start = (start + 1 + int(rng.integers(0, max_start))) % (max_start + 1)
    return start, start + window_size


def sample_support_pair_query(task_rollouts, support_window, query_horizon, rng=None):
    rng = rng or np.random.default_rng()
    support_candidates = [r for r in task_rollouts if r["features"].size(0) >= support_window]
    query_candidates = [r for r in task_rollouts if r["features"].size(0) >= query_horizon]
    if not support_candidates or not query_candidates:
        return None
    support_rollout = support_candidates[int(rng.integers(0, len(support_candidates)))]
    alt_candidates = [r for r in support_candidates if r["reference_id"] != support_rollout["reference_id"]] or support_candidates
    alt_support_rollout = alt_candidates[int(rng.integers(0, len(alt_candidates)))]
    query_candidates = [r for r in query_candidates if r["reference_id"] != support_rollout["reference_id"]] or query_candidates
    query_rollout = query_candidates[int(rng.integers(0, len(query_candidates)))]
    s0, s1 = sample_window(support_rollout, support_window, rng)
    a0, a1 = sample_window(alt_support_rollout, support_window, rng, excluded_start=s0 if alt_support_rollout is support_rollout else None)
    q0, q1 = sample_window(query_rollout, query_horizon, rng)
    return (
        support_rollout["features"][s0:s1],
        support_rollout["targets"][s0:s1],
        alt_support_rollout["features"][a0:a1],
        alt_support_rollout["targets"][a0:a1],
        {k: v[q0:q1] for k, v in query_rollout.items() if k != "reference_id"},
    )


def evaluate_meta_batch(residual_model, context_encoder, task_data, task_ids, support_window, query_horizon, dt, device, loss_cfg, deterministic=False):
    task_losses = []
    components = {"query_residual": [], "query_state": [], "context": [], "consistency": []}
    context_a_batch, context_b_batch = [], []
    for task_id in task_ids:
        task_entry = task_data[task_id]
        if not has_valid_rollout(task_entry["rollouts"], support_window, query_horizon):
            continue
        rng = np.random.default_rng(DEFAULT_SPLIT_SEED + int(task_id)) if deterministic else None
        sampled = sample_support_pair_query(task_entry["rollouts"], support_window, query_horizon, rng)
        if sampled is None:
            continue
        x_support, y_support, x_support_alt, y_support_alt, query_batch = sampled
        x_support, y_support = x_support.to(device), y_support.to(device)
        x_support_alt, y_support_alt = x_support_alt.to(device), y_support_alt.to(device)
        query_batch = {name: value.to(device) for name, value in query_batch.items()}
        task_context = task_entry["context"].to(device)
        context_a = context_encoder(x_support, y_support)
        context_b = context_encoder(x_support_alt, y_support_alt)
        query_pred = residual_model.forward_with_context(query_batch["features"], context_a)
        query_residual_loss = F.smooth_l1_loss(query_pred, query_batch["targets"], beta=loss_cfg["huber_beta"])
        pos_loss, vel_loss, att_loss, rate_loss = one_step_state_loss(
            rollout_step_torch(query_batch["states"], query_batch["actions"], query_pred, dt),
            query_batch["next_states"],
        )
        query_state_loss = (
            loss_cfg["position_weight"] * pos_loss
            + loss_cfg["velocity_weight"] * vel_loss
            + loss_cfg["attitude_weight"] * att_loss
            + loss_cfg["rate_weight"] * rate_loss
        )
        context_loss = 0.5 * (F.mse_loss(context_a.squeeze(0), task_context) + F.mse_loss(context_b.squeeze(0), task_context))
        consistency_loss = F.mse_loss(context_a, context_b)
        task_loss = (
            loss_cfg["query_residual_weight"] * query_residual_loss
            + loss_cfg["query_state_weight"] * query_state_loss
            + loss_cfg["context_weight"] * context_loss
            + loss_cfg["consistency_weight"] * consistency_loss
        )
        task_losses.append(task_loss)
        components["query_residual"].append(float(query_residual_loss.detach().item()))
        components["query_state"].append(float(query_state_loss.detach().item()))
        components["context"].append(float(context_loss.detach().item()))
        components["consistency"].append(float(consistency_loss.detach().item()))
        context_a_batch.append(context_a.squeeze(0))
        context_b_batch.append(context_b.squeeze(0))
    if not task_losses:
        return None
    contrastive_loss = symmetric_contrastive_loss(torch.stack(context_a_batch, dim=0), torch.stack(context_b_batch, dim=0), loss_cfg["contrastive_temperature"])
    mean_task_loss = torch.stack(task_losses).mean()
    loss_tensor = mean_task_loss + loss_cfg["contrastive_weight"] * contrastive_loss
    components = {name: float(np.mean(values)) for name, values in components.items() if values}
    components["contrastive"] = float(contrastive_loss.detach().item())
    return {"loss_tensor": loss_tensor, "loss_value": float(loss_tensor.detach().item()), "components": components}


def save_artifacts(
    save_path,
    residual_model,
    context_encoder,
    best_residual_state,
    best_encoder_state,
    train_losses,
    epochs,
    support_window,
    query_horizon,
    best_val_loss,
    best_val_epoch,
    context_center,
    context_scale,
    input_mean,
    input_scale,
    target_mean,
    target_scale,
):
    with torch.no_grad():
        residual_model.reset_context()
    payload = {
        "model_type": "amortized_context_meta",
        "command_mode": COMMAND_MODE_BODY_RATE_THRUST,
        "uses_io_normalization": True,
        "model_state_dict": best_residual_state if best_residual_state is not None else residual_model.state_dict(),
        "context_encoder_state_dict": best_encoder_state if best_encoder_state is not None else context_encoder.state_dict(),
        "input_dim": MODEL_INPUT_DIM,
        "output_dim": RESIDUAL_OUTPUT_DIM,
        "hidden_dim": MODEL_CONFIG["hidden_dim"],
        "num_layers": MODEL_CONFIG["num_layers"],
        "context_dim": len(CONTEXT_COLS),
        "context_cols": CONTEXT_COLS,
        "context_center": context_center.tolist(),
        "context_scale": context_scale.tolist(),
        "input_mean": input_mean.tolist(),
        "input_scale": input_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "context_injection": MODEL_CONFIG["context_injection"],
        "modulation_scale": MODEL_CONFIG["modulation_scale"],
        "context_encoder_hidden_dim": MODEL_CONFIG["context_encoder_hidden_dim"],
        "context_encoder_num_layers": MODEL_CONFIG["context_encoder_num_layers"],
        "support_window": support_window,
        "query_horizon": query_horizon,
        "online_context_ema_decay": TRAINING_CONFIG["online_context_ema_decay"],
        "online_context_warmup_updates": TRAINING_CONFIG["online_context_warmup_updates"],
        "loss_cfg": LOSS_CONFIG,
        "best_val_loss": None if best_residual_state is None else best_val_loss,
        "best_val_epoch": None if best_residual_state is None else best_val_epoch,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    torch.save({**payload, "model_state_dict": residual_model.state_dict(), "context_encoder_state_dict": context_encoder.state_dict(), "final_train_loss": train_losses[-1] if train_losses else None, "final_epoch": epochs}, save_path.with_name(f"{save_path.stem}_final{save_path.suffix}"))
    print(f"Saved checkpoint to {save_path}")


def save_loss_plot(save_path, train_losses, val_epochs, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train")
    if val_losses:
        plt.plot(val_epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description="Train body-rate/thrust context-meta residual model.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--support-window", type=int, default=DEFAULT_SUPPORT_WINDOW)
    parser.add_argument("--query-horizon", type=int, default=DEFAULT_QUERY_HORIZON)
    parser.add_argument("--val-interval", type=int, default=DEFAULT_VAL_INTERVAL)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")
    task_data, context_center, context_scale, input_mean, input_scale, target_mean, target_scale = load_task_data(args.csv_path)
    train_ids, val_ids = split_task_ids(task_data.keys())
    train_data = {tid: task_data[tid] for tid in train_ids}
    val_data = {tid: task_data[tid] for tid in val_ids}
    print(f"Loaded {len(task_data)} tasks. Train={len(train_data)} Val={len(val_data)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_batch_size = min(DEFAULT_META_BATCH_SIZE_CAP, len(train_data)) if args.meta_batch_size is None else min(args.meta_batch_size, len(train_data))
    residual_model = NormalizedContextResidualMLP(
        input_dim=MODEL_INPUT_DIM,
        output_dim=RESIDUAL_OUTPUT_DIM,
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        context_dim=len(CONTEXT_COLS),
        context_injection=MODEL_CONFIG["context_injection"],
        modulation_scale=MODEL_CONFIG["modulation_scale"],
    ).to(device)
    residual_model.set_normalization(input_mean, input_scale, target_mean, target_scale)
    context_encoder = NormalizedSupportSetEncoder(
        feature_dim=MODEL_INPUT_DIM,
        target_dim=RESIDUAL_OUTPUT_DIM,
        context_dim=len(CONTEXT_COLS),
        hidden_dim=MODEL_CONFIG["context_encoder_hidden_dim"],
        num_layers=MODEL_CONFIG["context_encoder_num_layers"],
    ).to(device)
    context_encoder.set_normalization(input_mean, input_scale, target_mean, target_scale)
    parameters = list(residual_model.parameters()) + list(context_encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=TRAINING_CONFIG["meta_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    train_losses, val_epochs, val_losses = [], [], []
    best_val_loss, best_val_epoch = float("inf"), -1
    best_residual_state, best_encoder_state = None, None
    train_id_list = list(train_data.keys())
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        task_ids = np.random.choice(train_id_list, meta_batch_size, replace=False)
        train_result = evaluate_meta_batch(residual_model, context_encoder, train_data, task_ids, args.support_window, args.query_horizon, DATA_DT, device, LOSS_CONFIG)
        if train_result is None:
            continue
        train_result["loss_tensor"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=TRAINING_CONFIG["grad_clip_norm"])
        optimizer.step()
        scheduler.step()
        train_losses.append(train_result["loss_value"])
        current_val_loss = None
        if ((epoch + 1) % args.val_interval == 0) or epoch == 0 or epoch == args.epochs - 1:
            residual_model.eval()
            context_encoder.eval()
            val_result = evaluate_meta_batch(residual_model, context_encoder, val_data, list(val_data.keys()), args.support_window, args.query_horizon, DATA_DT, device, LOSS_CONFIG, deterministic=True) if val_data else None
            residual_model.train()
            context_encoder.train()
            if val_result is not None:
                current_val_loss = val_result["loss_value"]
                val_epochs.append(epoch + 1)
                val_losses.append(current_val_loss)
                if current_val_loss < best_val_loss:
                    best_val_loss, best_val_epoch = current_val_loss, epoch + 1
                    with torch.no_grad():
                        residual_model.reset_context()
                    best_residual_state = deepcopy(residual_model.state_dict())
                    best_encoder_state = deepcopy(context_encoder.state_dict())
        if epoch % DEFAULT_LOG_INTERVAL == 0 or epoch == args.epochs - 1:
            comps = train_result["components"]
            msg = f"[Epoch {epoch + 1:05d}] train={train_result['loss_value']:.6f} q_res={comps['query_residual']:.5f} q_state={comps['query_state']:.5f}"
            if current_val_loss is not None:
                msg += f" val={current_val_loss:.6f}"
            if best_residual_state is not None:
                msg += f" best={best_val_loss:.6f}@{best_val_epoch}"
            print(msg)
    save_loss_plot(args.save_path, train_losses, val_epochs, val_losses)
    save_artifacts(
        args.save_path,
        residual_model,
        context_encoder,
        best_residual_state,
        best_encoder_state,
        train_losses,
        args.epochs,
        args.support_window,
        args.query_horizon,
        best_val_loss,
        best_val_epoch,
        context_center,
        context_scale,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
    )


if __name__ == "__main__":
    main()
