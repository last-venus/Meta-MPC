"""Train a faster and more discriminative context-meta residual model."""

import argparse
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


def sample_window(rollout, window_size, rng, excluded_start=None):
    max_start = rollout["features"].size(0) - window_size
    if max_start < 0:
        raise ValueError("Rollout is shorter than the requested window.")
    if max_start == 0:
        return 0, window_size

    start = int(rng.integers(0, max_start + 1))
    if excluded_start is not None and max_start >= 1 and start == excluded_start:
        start = (start + 1 + int(rng.integers(0, max_start))) % (max_start + 1)
    return start, start + window_size


def sample_support_pair_query(task_rollouts, support_window, query_horizon, rng=None):
    rng = rng or np.random.default_rng()
    support_candidates = [rollout for rollout in task_rollouts if rollout["features"].size(0) >= support_window]
    query_candidates = [rollout for rollout in task_rollouts if rollout["features"].size(0) >= query_horizon]
    if not support_candidates or not query_candidates:
        return None

    support_rollout = support_candidates[int(rng.integers(0, len(support_candidates)))]
    alternate_support_candidates = [
        rollout for rollout in support_candidates if rollout["reference_id"] != support_rollout["reference_id"]
    ]
    if not alternate_support_candidates:
        alternate_support_candidates = support_candidates
    alternate_support_rollout = alternate_support_candidates[int(rng.integers(0, len(alternate_support_candidates)))]

    preferred_query_candidates = [
        rollout for rollout in query_candidates if rollout["reference_id"] != support_rollout["reference_id"]
    ]
    if preferred_query_candidates:
        query_candidates = preferred_query_candidates
    query_rollout = query_candidates[int(rng.integers(0, len(query_candidates)))]

    support_start, support_end = sample_window(support_rollout, support_window, rng)
    alt_support_start, alt_support_end = sample_window(
        alternate_support_rollout,
        support_window,
        rng,
        excluded_start=support_start if alternate_support_rollout is support_rollout else None,
    )
    query_start, query_end = sample_window(query_rollout, query_horizon, rng)

    return (
        support_rollout["features"][support_start:support_end],
        support_rollout["targets"][support_start:support_end],
        alternate_support_rollout["features"][alt_support_start:alt_support_end],
        alternate_support_rollout["targets"][alt_support_start:alt_support_end],
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


def symmetric_contrastive_loss(context_a_batch, context_b_batch, temperature):
    if context_a_batch.size(0) <= 1:
        return context_a_batch.new_tensor(0.0)

    norm_a = F.normalize(context_a_batch, dim=-1)
    norm_b = F.normalize(context_b_batch, dim=-1)
    logits_ab = norm_a @ norm_b.transpose(0, 1) / temperature
    labels = torch.arange(logits_ab.size(0), device=logits_ab.device)
    return 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ab.transpose(0, 1), labels))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the quadrotor amortized-context meta model.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_META_DATASET_PATH, help="Path to the meta dataset CSV.")
    parser.add_argument("--save-path", type=Path, default=DEFAULT_META_CHECKPOINT_PATH, help="Path to save the best checkpoint.")
    parser.add_argument("--epochs", type=int, default=550, help="Number of meta-training epochs.")
    parser.add_argument("--meta-batch-size", type=int, default=None, help="Meta-batch size. Defaults to min(24, n_train_tasks).")
    parser.add_argument("--support-window", type=int, default=32, help="Support-set window size.")
    parser.add_argument("--query-horizon", type=int, default=16, help="Query horizon length.")
    parser.add_argument("--val-interval", type=int, default=25, help="Validation interval in epochs.")
    return parser.parse_args()


def evaluate_meta_loss(residual_model, context_encoder, val_task_data, support_window, query_horizon, dt, device, loss_cfg):
    if not val_task_data:
        return None, None

    task_losses = []
    components = {
        "query_residual": [],
        "query_state": [],
        "context": [],
        "consistency": [],
    }
    context_a_batch = []
    context_b_batch = []

    for task_id, task_entry in val_task_data.items():
        task_rollouts = task_entry["rollouts"]
        task_context = task_entry["context"].to(device)
        if not has_valid_rollout(task_rollouts, support_window, query_horizon):
            continue

        rng = np.random.default_rng(43 + int(task_id))
        sampled = sample_support_pair_query(task_rollouts, support_window, query_horizon, rng=rng)
        if sampled is None:
            continue

        x_support, y_support, x_support_alt, y_support_alt, query_batch = sampled
        x_support = x_support.to(device)
        y_support = y_support.to(device)
        x_support_alt = x_support_alt.to(device)
        y_support_alt = y_support_alt.to(device)
        query_batch = {name: value.to(device) for name, value in query_batch.items()}

        context_a = context_encoder(x_support, y_support)
        context_b = context_encoder(x_support_alt, y_support_alt)
        query_pred = residual_model.forward_with_context(query_batch["features"], context_a)
        query_residual_loss = F.mse_loss(query_pred, query_batch["targets"])

        pred_next_state = rollout_step_torch(query_batch["states"], query_batch["actions"], query_pred, dt=dt)
        pos_loss, vel_loss, att_loss, rate_loss = one_step_state_loss(pred_next_state, query_batch["next_states"])
        query_state_loss = (
            loss_cfg["position_weight"] * pos_loss
            + loss_cfg["velocity_weight"] * vel_loss
            + loss_cfg["attitude_weight"] * att_loss
            + loss_cfg["rate_weight"] * rate_loss
        )
        context_loss = 0.5 * (
            F.mse_loss(context_a.squeeze(0), task_context) + F.mse_loss(context_b.squeeze(0), task_context)
        )
        consistency_loss = F.mse_loss(context_a, context_b)
        total_loss = (
            loss_cfg["query_residual_weight"] * query_residual_loss
            + loss_cfg["query_state_weight"] * query_state_loss
            + loss_cfg["context_weight"] * context_loss
            + loss_cfg["consistency_weight"] * consistency_loss
        )

        task_losses.append(total_loss.detach().item())
        components["query_residual"].append(query_residual_loss.detach().item())
        components["query_state"].append(query_state_loss.detach().item())
        components["context"].append(context_loss.detach().item())
        components["consistency"].append(consistency_loss.detach().item())
        context_a_batch.append(context_a.squeeze(0))
        context_b_batch.append(context_b.squeeze(0))

    if not task_losses:
        return None, None
    contrastive_loss = symmetric_contrastive_loss(
        torch.stack(context_a_batch, dim=0),
        torch.stack(context_b_batch, dim=0),
        temperature=loss_cfg["contrastive_temperature"],
    )
    mean_components = {name: float(np.mean(values)) for name, values in components.items() if values}
    mean_components["contrastive"] = float(contrastive_loss.detach().item())
    total_loss = float(np.mean(task_losses) + loss_cfg["contrastive_weight"] * contrastive_loss.detach().item())
    return total_loss, mean_components


def main():
    args = parse_args()
    csv_path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Meta-dataset not found: {csv_path}")

    task_data = load_task_data(csv_path)
    train_task_data, val_task_data = split_task_data(task_data, val_ratio=0.2, seed=43)
    print(f"Loaded data for {len(task_data)} dynamics tasks.")
    print(f"Train tasks: {len(train_task_data)} | Val tasks: {len(val_task_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = args.epochs
    meta_batch_size = min(24, len(train_task_data)) if args.meta_batch_size is None else min(args.meta_batch_size, len(train_task_data))
    support_window = args.support_window
    query_horizon = args.query_horizon
    val_interval = args.val_interval
    hidden_dim = 128
    num_layers = 3
    context_encoder_hidden_dim = 80
    context_encoder_num_layers = 2
    meta_lr = 5e-4
    rollout_dt = DATA_DT
    context_injection = "adapter"
    modulation_scale = 0.20
    online_context_ema_decay = 0.72
    online_context_warmup_updates = 1
    grad_clip_norm = 5.0
    loss_cfg = {
        "query_residual_weight": 1.0,
        "query_state_weight": 0.35,
        "context_weight": 0.15,
        "consistency_weight": 0.10,
        "contrastive_weight": 0.06,
        "contrastive_temperature": 0.35,
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
        context_injection=context_injection,
        modulation_scale=modulation_scale,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    train_losses = []
    val_epochs = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = -1
    best_residual_state = None
    best_encoder_state = None

    print(
        f"Fast context-meta config: residual_hidden={hidden_dim}, encoder_hidden={context_encoder_hidden_dim}, "
        f"context_dim={CONTEXT_DIM}, support={support_window}, query={query_horizon}, epochs={epochs}, "
        f"context_injection={context_injection}, ema={online_context_ema_decay:.2f}"
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        task_losses = []
        train_components = {
            "query_residual": [],
            "query_state": [],
            "context": [],
            "consistency": [],
        }
        context_a_batch = []
        context_b_batch = []
        task_ids = np.random.choice(list(train_task_data.keys()), meta_batch_size, replace=False)

        for task_id in task_ids:
            task_entry = train_task_data[task_id]
            task_rollouts = task_entry["rollouts"]
            task_context = task_entry["context"].to(device)
            if not has_valid_rollout(task_rollouts, support_window, query_horizon):
                continue

            sampled = sample_support_pair_query(task_rollouts, support_window, query_horizon)
            if sampled is None:
                continue

            x_support, y_support, x_support_alt, y_support_alt, query_batch = sampled
            x_support = x_support.to(device)
            y_support = y_support.to(device)
            x_support_alt = x_support_alt.to(device)
            y_support_alt = y_support_alt.to(device)
            query_batch = {name: value.to(device) for name, value in query_batch.items()}

            context_a = context_encoder(x_support, y_support)
            context_b = context_encoder(x_support_alt, y_support_alt)
            query_pred = residual_model.forward_with_context(query_batch["features"], context_a)
            query_residual_loss = F.mse_loss(query_pred, query_batch["targets"])

            pred_next_state = rollout_step_torch(query_batch["states"], query_batch["actions"], query_pred, dt=rollout_dt)
            pos_loss, vel_loss, att_loss, rate_loss = one_step_state_loss(pred_next_state, query_batch["next_states"])
            query_state_loss = (
                loss_cfg["position_weight"] * pos_loss
                + loss_cfg["velocity_weight"] * vel_loss
                + loss_cfg["attitude_weight"] * att_loss
                + loss_cfg["rate_weight"] * rate_loss
            )
            context_loss = 0.5 * (
                F.mse_loss(context_a.squeeze(0), task_context) + F.mse_loss(context_b.squeeze(0), task_context)
            )
            consistency_loss = F.mse_loss(context_a, context_b)
            task_loss = (
                loss_cfg["query_residual_weight"] * query_residual_loss
                + loss_cfg["query_state_weight"] * query_state_loss
                + loss_cfg["context_weight"] * context_loss
                + loss_cfg["consistency_weight"] * consistency_loss
            )

            task_losses.append(task_loss)
            train_components["query_residual"].append(query_residual_loss.detach().item())
            train_components["query_state"].append(query_state_loss.detach().item())
            train_components["context"].append(context_loss.detach().item())
            train_components["consistency"].append(consistency_loss.detach().item())
            context_a_batch.append(context_a.squeeze(0))
            context_b_batch.append(context_b.squeeze(0))

        if not task_losses:
            continue

        mean_task_loss = torch.stack(task_losses).mean()
        contrastive_loss = symmetric_contrastive_loss(
            torch.stack(context_a_batch, dim=0),
            torch.stack(context_b_batch, dim=0),
            temperature=loss_cfg["contrastive_temperature"],
        )
        meta_loss = mean_task_loss + loss_cfg["contrastive_weight"] * contrastive_loss
        current_meta_loss = meta_loss.item()
        current_train_components = {name: float(np.mean(values)) for name, values in train_components.items() if values}
        current_train_components["contrastive"] = float(contrastive_loss.detach().item())

        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(residual_model.parameters()) + list(context_encoder.parameters()),
            max_norm=grad_clip_norm,
        )
        optimizer.step()
        scheduler.step()
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
                    f" cons={current_train_components['consistency']:.5f}"
                    f" ctr={current_train_components['contrastive']:.5f}"
                )
            if current_val_loss is not None:
                log_msg += f" | Val Loss: {current_val_loss:.6f}"
            if current_val_components:
                log_msg += (
                    f" | val_q_res={current_val_components['query_residual']:.5f}"
                    f" val_q_state={current_val_components['query_state']:.5f}"
                    f" val_ctx={current_val_components['context']:.5f}"
                    f" val_cons={current_val_components['consistency']:.5f}"
                    f" val_ctr={current_val_components['contrastive']:.5f}"
                )
            if best_residual_state is not None:
                log_msg += f" | Best Val: {best_val_loss:.6f} @ epoch {best_val_epoch}"
            print(log_msg)

    with torch.no_grad():
        residual_model.reset_context()
    effective_residual_state = best_residual_state if best_residual_state is not None else residual_model.state_dict()
    effective_encoder_state = best_encoder_state if best_encoder_state is not None else context_encoder.state_dict()

    save_path = args.save_path
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
        "context_injection": context_injection,
        "modulation_scale": modulation_scale,
        "context_encoder_hidden_dim": context_encoder_hidden_dim,
        "context_encoder_num_layers": context_encoder_num_layers,
        "support_window": support_window,
        "query_horizon": query_horizon,
        "inner_steps": 1,
        "inner_lr": 0.0,
        "online_context_ema_decay": online_context_ema_decay,
        "online_context_warmup_updates": online_context_warmup_updates,
        "rollout_dt": rollout_dt,
        "loss_cfg": loss_cfg,
        "context_center": CONTEXT_CENTER.tolist(),
        "context_scale": CONTEXT_SCALE.tolist(),
        "command_mode": "motor_thrust",
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
    plt.title(f"Fast Context-Meta Loss (Quadrotor3D Motor Thrust, {epochs} epochs)")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plot_path = save_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    main()
