"""Train a fast context-encoder meta-residual model for 3D quadrotor tracking."""

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


TRACKING_DIR = Path(__file__).resolve().parents[1]
if str(TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKING_DIR))

from quadrotor3D_common import (  # noqa: E402
    BASE_PARAMS,
    DEFAULT_META_CHECKPOINT_PATH,
    DEFAULT_META_DATASET_PATH,
    ContextResidualMLP,
    NOMINAL_RATIOS,
    SupportSetEncoder,
)


torch.manual_seed(43)

STATE_COLS = ["x", "x_dot", "y", "y_dot", "z", "z_dot", "phi", "theta", "psi", "p", "q", "r"]
ACTION_COLS = ["u1", "u2", "u3", "u4"]
INPUT_COLS = [*STATE_COLS, *ACTION_COLS]
TARGET_COLS = [
    "res_x_ddot",
    "res_y_ddot",
    "res_z_ddot",
    "res_p_dot",
    "res_q_dot",
    "res_r_dot",
]
CONTEXT_COLS = ["mass_ratio", "ixx_ratio", "iyy_ratio", "izz_ratio"]

INPUT_DIM = len(INPUT_COLS)
OUTPUT_DIM = len(TARGET_COLS)
CONTEXT_DIM = len(CONTEXT_COLS)
DATA_DT = 0.02
GRAVITY = 9.8
ARM_LENGTH = 0.0397
TORQUE_RATIO = 7.94e-12 / 3.16e-10
NOMINAL_MASS = BASE_PARAMS["M"] * NOMINAL_RATIOS["M"]
NOMINAL_IXX = BASE_PARAMS["Ixx"] * NOMINAL_RATIOS["Ixx"]
NOMINAL_IYY = BASE_PARAMS["Iyy"] * NOMINAL_RATIOS["Iyy"]
NOMINAL_IZZ = BASE_PARAMS["Izz"] * NOMINAL_RATIOS["Izz"]
CONTEXT_CENTER = torch.tensor([1.25, 1.025, 1.025, 1.025], dtype=torch.float32)
CONTEXT_SCALE = torch.tensor([0.50, 0.225, 0.225, 0.225], dtype=torch.float32)


def normalize_context(context_tensor: torch.Tensor) -> torch.Tensor:
    return (context_tensor - CONTEXT_CENTER.to(context_tensor.device)) / CONTEXT_SCALE.to(context_tensor.device)


def load_task_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "reference_id" not in df.columns:
        df["reference_id"] = 0

    required_cols = {"task_id", "episode_id", "reference_id", "time", *INPUT_COLS, *TARGET_COLS, *CONTEXT_COLS}
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise KeyError(f"Missing dataset columns: {missing_cols}")

    task_data = {}
    for task_id, task_df in df.groupby("task_id", sort=False):
        episode_rollouts = []
        task_context = torch.tensor(task_df[CONTEXT_COLS].iloc[0].to_numpy(dtype=np.float32), dtype=torch.float32)
        task_context = normalize_context(task_context)

        for _, episode_df in task_df.groupby("episode_id", sort=False):
            episode_df = episode_df.sort_values("time").reset_index(drop=True)
            if len(episode_df) < 2:
                continue

            features = episode_df[INPUT_COLS].to_numpy(dtype=np.float32, copy=True)
            residuals = episode_df[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)
            states = episode_df[STATE_COLS].to_numpy(dtype=np.float32, copy=True)
            actions = episode_df[ACTION_COLS].to_numpy(dtype=np.float32, copy=True)

            episode_rollouts.append(
                {
                    "reference_id": int(episode_df["reference_id"].iloc[0]),
                    "features": torch.from_numpy(features[:-1]),
                    "targets": torch.from_numpy(residuals[:-1]),
                    "states": torch.from_numpy(states[:-1]),
                    "actions": torch.from_numpy(actions[:-1]),
                    "next_states": torch.from_numpy(states[1:]),
                }
            )

        if episode_rollouts:
            task_data[int(task_id)] = {
                "context": task_context,
                "rollouts": episode_rollouts,
            }
    return task_data


def split_task_data(task_data, val_ratio=0.2, seed=43):
    task_ids = np.array(sorted(task_data.keys()))
    if len(task_ids) <= 1:
        return task_data, {}

    rng = np.random.default_rng(seed)
    rng.shuffle(task_ids)
    n_val = max(1, int(round(len(task_ids) * val_ratio)))
    n_val = min(n_val, len(task_ids) - 1)
    val_ids = set(task_ids[:n_val].tolist())
    train_task_data = {task_id: task_data[task_id] for task_id in task_data if task_id not in val_ids}
    val_task_data = {task_id: task_data[task_id] for task_id in task_data if task_id in val_ids}
    return train_task_data, val_task_data


def has_valid_rollout(task_rollouts, support_window, query_horizon):
    has_support = any(rollout["features"].size(0) >= support_window for rollout in task_rollouts)
    has_query = any(rollout["features"].size(0) >= query_horizon for rollout in task_rollouts)
    return has_support and has_query


def sample_support_query_rollouts(task_rollouts, support_window, query_horizon, rng=None):
    rng = rng or np.random.default_rng()
    support_candidates = [rollout for rollout in task_rollouts if rollout["features"].size(0) >= support_window]
    query_candidates = [rollout for rollout in task_rollouts if rollout["features"].size(0) >= query_horizon]
    if not support_candidates or not query_candidates:
        return None

    support_rollout = support_candidates[int(rng.integers(0, len(support_candidates)))]
    preferred_query_candidates = [
        rollout for rollout in query_candidates if rollout["reference_id"] != support_rollout["reference_id"]
    ]
    if preferred_query_candidates:
        query_candidates = preferred_query_candidates
    query_rollout = query_candidates[int(rng.integers(0, len(query_candidates)))]

    support_start = int(rng.integers(0, support_rollout["features"].size(0) - support_window + 1))
    query_start = int(rng.integers(0, query_rollout["features"].size(0) - query_horizon + 1))
    support_end = support_start + support_window
    query_end = query_start + query_horizon

    return (
        support_rollout["features"][support_start:support_end],
        support_rollout["targets"][support_start:support_end],
        {
            "features": query_rollout["features"][query_start:query_end],
            "targets": query_rollout["targets"][query_start:query_end],
            "states": query_rollout["states"][query_start:query_end],
            "actions": query_rollout["actions"][query_start:query_end],
            "next_states": query_rollout["next_states"][query_start:query_end],
        },
    )


def wrap_angle_torch(angle):
    return torch.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def wrap_state_angles_torch(state):
    wrapped = state.clone()
    wrapped[..., 6:9] = wrap_angle_torch(wrapped[..., 6:9])
    return wrapped


def safe_cosine(theta, eps=1e-3):
    sign = torch.where(theta.cos() >= 0.0, torch.ones_like(theta), -torch.ones_like(theta))
    return sign * torch.clamp(theta.cos().abs(), min=eps)


def nominal_dynamics_torch(state, action):
    x_dot = state[..., 1]
    y_dot = state[..., 3]
    z_dot = state[..., 5]
    phi = state[..., 6]
    theta = state[..., 7]
    psi = state[..., 8]
    p_body = state[..., 9]
    q_body = state[..., 10]
    r_body = state[..., 11]

    f1 = action[..., 0]
    f2 = action[..., 1]
    f3 = action[..., 2]
    f4 = action[..., 3]

    c_phi = torch.cos(phi)
    s_phi = torch.sin(phi)
    c_theta = safe_cosine(theta)
    s_theta = torch.sin(theta)
    c_psi = torch.cos(psi)
    s_psi = torch.sin(psi)
    tan_theta = s_theta / c_theta

    total_thrust = f1 + f2 + f3 + f4
    pos_ddot_x = (s_phi * s_psi + c_phi * s_theta * c_psi) * total_thrust / NOMINAL_MASS
    pos_ddot_y = (-s_phi * c_psi + c_phi * s_theta * s_psi) * total_thrust / NOMINAL_MASS
    pos_ddot_z = c_phi * c_theta * total_thrust / NOMINAL_MASS - GRAVITY

    tau_x = ARM_LENGTH / np.sqrt(2.0) * (f1 + f2 - f3 - f4)
    tau_y = ARM_LENGTH / np.sqrt(2.0) * (-f1 + f2 + f3 - f4)
    tau_z = TORQUE_RATIO * (-f1 + f2 - f3 + f4)

    j_rates = torch.stack(
        [
            NOMINAL_IXX * p_body,
            NOMINAL_IYY * q_body,
            NOMINAL_IZZ * r_body,
        ],
        dim=-1,
    )
    body_rates = torch.stack([p_body, q_body, r_body], dim=-1)
    coriolis = torch.cross(body_rates, j_rates, dim=-1)
    body_rates_dot = torch.stack(
        [
            (tau_x - coriolis[..., 0]) / NOMINAL_IXX,
            (tau_y - coriolis[..., 1]) / NOMINAL_IYY,
            (tau_z - coriolis[..., 2]) / NOMINAL_IZZ,
        ],
        dim=-1,
    )

    euler_rates = torch.stack(
        [
            p_body + q_body * s_phi * tan_theta + r_body * c_phi * tan_theta,
            q_body * c_phi - r_body * s_phi,
            q_body * s_phi / c_theta + r_body * c_phi / c_theta,
        ],
        dim=-1,
    )

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
    return torch.stack(
        [
            zeros,
            residual[..., 0],
            zeros,
            residual[..., 1],
            zeros,
            residual[..., 2],
            zeros,
            zeros,
            zeros,
            residual[..., 3],
            residual[..., 4],
            residual[..., 5],
        ],
        dim=-1,
    )


def rollout_step_torch(state, action, residual, dt):
    state_dot = nominal_dynamics_torch(state, action) + residual_to_state_derivative(residual)
    return wrap_state_angles_torch(state + dt * state_dot)


def one_step_state_loss(pred_next_state, target_next_state):
    pos_loss = (pred_next_state[:, [0, 2, 4]] - target_next_state[:, [0, 2, 4]]).pow(2).mean()
    vel_loss = (pred_next_state[:, [1, 3, 5]] - target_next_state[:, [1, 3, 5]]).pow(2).mean()
    att_loss = wrap_angle_torch(pred_next_state[:, 6:9] - target_next_state[:, 6:9]).pow(2).mean()
    rate_loss = (pred_next_state[:, 9:12] - target_next_state[:, 9:12]).pow(2).mean()
    return pos_loss, vel_loss, att_loss, rate_loss


def evaluate_meta_loss(residual_model, context_encoder, val_task_data, support_window, query_horizon, dt, device, loss_cfg):
    if not val_task_data:
        return None, None

    losses = []
    components = {
        "query_residual": [],
        "query_state": [],
        "context": [],
    }

    for task_id, task_entry in val_task_data.items():
        task_rollouts = task_entry["rollouts"]
        task_context = task_entry["context"].to(device)
        if not has_valid_rollout(task_rollouts, support_window, query_horizon):
            continue

        rng = np.random.default_rng(43 + int(task_id))
        sampled = sample_support_query_rollouts(task_rollouts, support_window, query_horizon, rng=rng)
        if sampled is None:
            continue

        x_support, y_support, query_batch = sampled
        x_support = x_support.to(device)
        y_support = y_support.to(device)
        query_batch = {name: value.to(device) for name, value in query_batch.items()}

        context = context_encoder(x_support, y_support)
        query_pred = residual_model.forward_with_context(query_batch["features"], context)
        query_residual_loss = F.mse_loss(query_pred, query_batch["targets"])

        pred_next_state = rollout_step_torch(query_batch["states"], query_batch["actions"], query_pred, dt=dt)
        pos_loss, vel_loss, att_loss, rate_loss = one_step_state_loss(pred_next_state, query_batch["next_states"])
        query_state_loss = (
            loss_cfg["position_weight"] * pos_loss
            + loss_cfg["velocity_weight"] * vel_loss
            + loss_cfg["attitude_weight"] * att_loss
            + loss_cfg["rate_weight"] * rate_loss
        )
        context_loss = F.mse_loss(context.squeeze(0), task_context)
        total_loss = (
            loss_cfg["query_residual_weight"] * query_residual_loss
            + loss_cfg["query_state_weight"] * query_state_loss
            + loss_cfg["context_weight"] * context_loss
        )

        losses.append(total_loss.detach().item())
        components["query_residual"].append(query_residual_loss.detach().item())
        components["query_state"].append(query_state_loss.detach().item())
        components["context"].append(context_loss.detach().item())

    if not losses:
        return None, None
    mean_components = {name: float(np.mean(values)) for name, values in components.items() if values}
    return float(np.mean(losses)), mean_components


def main():
    csv_path = DEFAULT_META_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Meta-dataset not found: {csv_path}")

    task_data = load_task_data(csv_path)
    train_task_data, val_task_data = split_task_data(task_data, val_ratio=0.2, seed=43)
    print(f"Loaded data for {len(task_data)} dynamics tasks.")
    print(f"Train tasks: {len(train_task_data)} | Val tasks: {len(val_task_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 700
    meta_batch_size = min(24, len(train_task_data))
    support_window = 32
    query_horizon = 16
    val_interval = 25
    hidden_dim = 128
    num_layers = 3
    context_encoder_hidden_dim = 96
    context_encoder_num_layers = 2
    meta_lr = 5e-4
    rollout_dt = DATA_DT
    loss_cfg = {
        "query_residual_weight": 1.0,
        "query_state_weight": 0.35,
        "context_weight": 0.15,
        "position_weight": 1.6,
        "velocity_weight": 0.20,
        "attitude_weight": 0.35,
        "rate_weight": 0.08,
    }

    residual_model = ContextResidualMLP(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        context_dim=CONTEXT_DIM,
    ).to(device)
    context_encoder = SupportSetEncoder(
        feature_dim=INPUT_DIM,
        target_dim=OUTPUT_DIM,
        context_dim=CONTEXT_DIM,
        hidden_dim=context_encoder_hidden_dim,
        num_layers=context_encoder_num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(residual_model.parameters()) + list(context_encoder.parameters()),
        lr=meta_lr,
    )

    train_losses = []
    val_epochs = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = -1
    best_residual_state = None
    best_encoder_state = None

    print(
        f"Fast context-meta config: residual_hidden={hidden_dim}, encoder_hidden={context_encoder_hidden_dim}, "
        f"context_dim={CONTEXT_DIM}, support={support_window}, query={query_horizon}, epochs={epochs}"
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        meta_loss = 0.0
        n_tasks_used = 0
        train_components = {
            "query_residual": [],
            "query_state": [],
            "context": [],
        }
        task_ids = np.random.choice(list(train_task_data.keys()), meta_batch_size, replace=False)

        for task_id in task_ids:
            task_entry = train_task_data[task_id]
            task_rollouts = task_entry["rollouts"]
            task_context = task_entry["context"].to(device)
            if not has_valid_rollout(task_rollouts, support_window, query_horizon):
                continue

            sampled = sample_support_query_rollouts(task_rollouts, support_window, query_horizon)
            if sampled is None:
                continue

            x_support, y_support, query_batch = sampled
            x_support = x_support.to(device)
            y_support = y_support.to(device)
            query_batch = {name: value.to(device) for name, value in query_batch.items()}

            context = context_encoder(x_support, y_support)
            query_pred = residual_model.forward_with_context(query_batch["features"], context)
            query_residual_loss = F.mse_loss(query_pred, query_batch["targets"])

            pred_next_state = rollout_step_torch(query_batch["states"], query_batch["actions"], query_pred, dt=rollout_dt)
            pos_loss, vel_loss, att_loss, rate_loss = one_step_state_loss(pred_next_state, query_batch["next_states"])
            query_state_loss = (
                loss_cfg["position_weight"] * pos_loss
                + loss_cfg["velocity_weight"] * vel_loss
                + loss_cfg["attitude_weight"] * att_loss
                + loss_cfg["rate_weight"] * rate_loss
            )
            context_loss = F.mse_loss(context.squeeze(0), task_context)
            task_loss = (
                loss_cfg["query_residual_weight"] * query_residual_loss
                + loss_cfg["query_state_weight"] * query_state_loss
                + loss_cfg["context_weight"] * context_loss
            )

            meta_loss = meta_loss + task_loss
            n_tasks_used += 1
            train_components["query_residual"].append(query_residual_loss.detach().item())
            train_components["query_state"].append(query_state_loss.detach().item())
            train_components["context"].append(context_loss.detach().item())

        if n_tasks_used == 0:
            continue

        meta_loss = meta_loss / n_tasks_used
        current_meta_loss = meta_loss.item()
        current_train_components = {name: float(np.mean(values)) for name, values in train_components.items() if values}

        meta_loss.backward()
        optimizer.step()
        train_losses.append(current_meta_loss)

        current_val_loss = None
        current_val_components = None
        if ((epoch + 1) % val_interval == 0) or epoch == 0 or epoch == epochs - 1:
            residual_model.eval()
            context_encoder.eval()
            current_val_loss, current_val_components = evaluate_meta_loss(
                residual_model,
                context_encoder,
                val_task_data,
                support_window=support_window,
                query_horizon=query_horizon,
                dt=rollout_dt,
                device=device,
                loss_cfg=loss_cfg,
            )
            residual_model.train()
            context_encoder.train()
            if current_val_loss is not None:
                val_epochs.append(epoch + 1)
                val_losses.append(current_val_loss)
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_val_epoch = epoch + 1
                    with torch.no_grad():
                        residual_model.reset_context()
                    best_residual_state = deepcopy(residual_model.state_dict())
                    best_encoder_state = deepcopy(context_encoder.state_dict())

        if epoch % 25 == 0 or epoch == epochs - 1:
            log_msg = f"[Epoch {epoch + 1:05d}] Train Loss: {current_meta_loss:.6f}"
            if current_train_components:
                log_msg += (
                    f" | q_res={current_train_components['query_residual']:.5f}"
                    f" q_state={current_train_components['query_state']:.5f}"
                    f" ctx={current_train_components['context']:.5f}"
                )
            if current_val_loss is not None:
                log_msg += f" | Val Loss: {current_val_loss:.6f}"
            if current_val_components:
                log_msg += (
                    f" | val_q_res={current_val_components['query_residual']:.5f}"
                    f" val_q_state={current_val_components['query_state']:.5f}"
                    f" val_ctx={current_val_components['context']:.5f}"
                )
            if best_residual_state is not None:
                log_msg += f" | Best Val: {best_val_loss:.6f} @ epoch {best_val_epoch}"
            print(log_msg)

    with torch.no_grad():
        residual_model.reset_context()
    effective_residual_state = best_residual_state if best_residual_state is not None else residual_model.state_dict()
    effective_encoder_state = best_encoder_state if best_encoder_state is not None else context_encoder.state_dict()

    save_path = DEFAULT_META_CHECKPOINT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = save_path.with_name(f"{save_path.stem}_final{save_path.suffix}")

    payload = {
        "model_type": "amortized_context_meta",
        "model_state_dict": effective_residual_state,
        "context_encoder_state_dict": effective_encoder_state,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "context_dim": CONTEXT_DIM,
        "context_encoder_hidden_dim": context_encoder_hidden_dim,
        "context_encoder_num_layers": context_encoder_num_layers,
        "support_window": support_window,
        "query_horizon": query_horizon,
        "inner_steps": 1,
        "inner_lr": 0.0,
        "rollout_dt": rollout_dt,
        "loss_cfg": loss_cfg,
        "context_center": CONTEXT_CENTER.tolist(),
        "context_scale": CONTEXT_SCALE.tolist(),
        "best_val_loss": None if best_residual_state is None else best_val_loss,
        "best_val_epoch": None if best_residual_state is None else best_val_epoch,
    }
    torch.save(payload, save_path)
    torch.save(
        {
            **payload,
            "model_state_dict": residual_model.state_dict(),
            "context_encoder_state_dict": context_encoder.state_dict(),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_epoch": epochs,
        },
        final_path,
    )
    print(f"Saved checkpoint to {save_path}")
    print(f"Saved final checkpoint to {final_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train")
    if val_losses:
        plt.plot(val_epochs, val_losses, label="val", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss")
    plt.title(f"Fast Context-Meta Loss (Quadrotor3D, {epochs} epochs)")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plot_path = save_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    main()
