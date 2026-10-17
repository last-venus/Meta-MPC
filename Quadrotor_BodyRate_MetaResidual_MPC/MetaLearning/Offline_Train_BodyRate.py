"""Train an amortized context-meta residual model for body-rate/thrust MPC."""

import argparse
import csv
import json
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
    PHYSICS_CONTEXT_GROUPS,
    PhysicsDisentangledContextResidualMLP,
    GroupedTemporalPhysicsSupportSetEncoderV3,
)


DEFAULT_EPOCHS = 600
DEFAULT_META_BATCH_SIZE_CAP = 24
DEFAULT_SUPPORT_WINDOW = 32
DEFAULT_QUERY_HORIZON = 32
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
    "architecture": "temporal_physics_disentangled_v3",
    "modulation_scale": 0.20,
}
TRAINING_CONFIG = {
    "meta_lr": 1e-4,
    "context_warmup_lr": 2e-4,
    "grad_clip_norm": 5.0,
    "context_warmup_epochs": 120,
    # The support set must change the task model before the query loss is
    # evaluated.  Keeping this latent-only makes the inner loop cheap enough
    # to use identically during closed-loop MPC.
    "context_refinement_steps": 2,
    "context_refinement_lr": 0.15,
    "fast_adapter_steps": 1,
    "fast_adapter_lr": 0.10,
    "online_context_ema_decay": 0.65,
    "online_context_warmup_updates": 1,
    "online_context_refinement_steps": 2,
    "online_context_refinement_lr": 0.15,
    "online_fast_adapter_steps": 1,
    "online_fast_adapter_lr": 0.10,
}
LOSS_CONFIG = {
    "query_residual_weight": 1.0,
    # Train the residual model on the real task coordinates as well as the
    # inferred coordinates.  Without this anchor, it can exploit arbitrary
    # encoder directions and a true (oracle) context becomes unusable.
    "oracle_query_residual_weight": 1.0,
    "context_counterfactual_weight": 2.0,
    "query_state_weight": 0.10,
    "context_translation_weight": 0.15,
    "context_rotation_weight": 0.80,
    "context_aerodynamics_weight": 0.40,
    "cross_trajectory_weight": 10.00,
    "contrastive_weight": 0.001,
    "contrastive_temperature": 0.35,
    "position_weight": 1.4,
    "velocity_weight": 0.25,
    "attitude_weight": 0.25,
    "rate_weight": 0.08,
    # Physical scales make the multi-step state objective dimensionless and
    # comparable. A one-step 20 ms position error is O(dt^2), so it otherwise
    # makes no meaningful contribution to the optimized loss.
    "position_scale": 0.05,
    "velocity_scale": 0.25,
    "attitude_scale": 0.12,
    "rate_scale": 0.60,
    "huber_beta": 1.0,
}
# Jointly optimizing task identification and residual fitting allowed the
# residual objective to pull the rotation/aerodynamic context heads away from
# their supervised task labels. First identify the task from support data,
# then freeze that encoder while fitting the residual dynamics. Both stages
# remain within the same amortized context-meta-MPC formulation.
CONTEXT_WARMUP_LOSS_CONFIG = {
    **LOSS_CONFIG,
    "query_residual_weight": 0.0,
    "oracle_query_residual_weight": 0.0,
    "context_counterfactual_weight": 0.0,
    "query_state_weight": 0.0,
    "context_translation_weight": 1.0,
    "context_rotation_weight": 1.0,
    "context_aerodynamics_weight": 1.0,
    "cross_trajectory_weight": 5.0,
}
PHYSICS_GROUNDED_META_LOSS_CONFIG = {
    **LOSS_CONFIG,
    # Keep the encoder physically identifiable while its initialization is
    # improved by query-after-support-update gradients.
    "context_translation_weight": 0.04,
    "context_rotation_weight": 0.10,
    "context_aerodynamics_weight": 0.05,
    "cross_trajectory_weight": 0.50,
    "contrastive_weight": 0.0005,
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


def multistep_state_loss(residual_model, contexts, states, actions, next_states, dt, loss_cfg, adapters=None):
    """Vectorized open-loop loss for a batch of task-specific query windows."""
    if states.dim() == 2:
        states, actions, next_states, contexts = states.unsqueeze(0), actions.unsqueeze(0), next_states.unsqueeze(0), contexts.unsqueeze(0)
    predicted_state = states[:, 0, :]
    losses = {"position": [], "velocity": [], "attitude": [], "rate": []}
    for step in range(actions.size(1)):
        action = actions[:, step, :]
        model_input = torch.cat([predicted_state, action], dim=-1)
        residual = residual_model.forward_with_context(model_input, contexts) if adapters is None else residual_model.forward_with_context_and_adapter(model_input, contexts, adapters)
        predicted_state = rollout_step_torch(predicted_state, action, residual, dt)
        target = next_states[:, step, :]
        losses["position"].append(((predicted_state[:, [0, 2, 4]] - target[:, [0, 2, 4]]) / loss_cfg["position_scale"]).pow(2).mean())
        losses["velocity"].append(((predicted_state[:, [1, 3, 5]] - target[:, [1, 3, 5]]) / loss_cfg["velocity_scale"]).pow(2).mean())
        losses["attitude"].append((wrap_angle_torch(predicted_state[:, 6:9] - target[:, 6:9]) / loss_cfg["attitude_scale"]).pow(2).mean())
        losses["rate"].append(((predicted_state[:, 9:12] - target[:, 9:12]) / loss_cfg["rate_scale"]).pow(2).mean())
    return tuple(torch.stack(losses[name]).mean() for name in ("position", "velocity", "attitude", "rate"))


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


def load_dataset_frame(csv_path: Path) -> pd.DataFrame:
    """Read either the legacy monolithic CSV or a completed atomic shard set."""
    if not csv_path.is_dir():
        return pd.read_csv(csv_path)
    manifest_path = csv_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sharded dataset directory has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest.get("summary", {})
    expected_tasks = int(summary.get("expected_task_count", 0))
    complete_tasks = int(summary.get("complete_task_count", -1))
    if expected_tasks < 1 or complete_tasks != expected_tasks:
        raise RuntimeError(
            "Refusing to train on an incomplete shard set. Resume collection first; "
            f"complete tasks={complete_tasks}, expected tasks={expected_tasks}."
        )
    shard_paths = sorted(csv_path.glob("shard-*/data.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No committed shard CSV files found under {csv_path}")
    print(f"Loading {len(shard_paths)} committed dataset shard(s) from {csv_path}.")
    return pd.concat((pd.read_csv(path) for path in shard_paths), ignore_index=True)


def load_task_data(csv_path: Path):
    df = load_dataset_frame(csv_path)
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
                    "is_calibration": bool(episode_df["traj_type"].iloc[0] == "calibration_probe"),
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
    query_candidates = [r for r in task_rollouts if r["features"].size(0) >= query_horizon and not r.get("is_calibration", False)]
    if not support_candidates or not query_candidates:
        return None
    # Calibration carries strong angular excitation; references 4--8 are the
    # deliberately fast tracking maneuvers that expose drag and wind. Sample
    # the two modes equally and pair them across modes, so every causal context
    # group sees its identifiable signal without leaking task labels.
    calibration_candidates = [r for r in support_candidates if r.get("is_calibration", False)]
    high_motion_candidates = [r for r in support_candidates if not r.get("is_calibration", False) and r["reference_id"] >= 4]
    if calibration_candidates and high_motion_candidates:
        primary_candidates = calibration_candidates if int(rng.integers(0, 2)) == 0 else high_motion_candidates
    else:
        primary_candidates = calibration_candidates or high_motion_candidates or support_candidates
    support_rollout = primary_candidates[int(rng.integers(0, len(primary_candidates)))]
    if support_rollout.get("is_calibration", False):
        alt_candidates = high_motion_candidates
    else:
        alt_candidates = calibration_candidates
    alt_candidates = alt_candidates or [r for r in support_candidates if r["reference_id"] != support_rollout["reference_id"]] or support_candidates
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
        {k: v[q0:q1] for k, v in query_rollout.items() if k not in {"reference_id", "is_calibration"}},
    )


def grouped_context_losses(context_a, context_b, task_context):
    """Supervise each physics context group and align it across trajectories."""
    group_losses = {}
    cross_losses = []
    for name, indices in PHYSICS_CONTEXT_GROUPS.items():
        group_losses[name] = 0.5 * (
            F.mse_loss(context_a.squeeze(0)[list(indices)], task_context[list(indices)])
            + F.mse_loss(context_b.squeeze(0)[list(indices)], task_context[list(indices)])
        )
        cross_losses.append(F.mse_loss(context_a[:, list(indices)], context_b[:, list(indices)]))
    return group_losses, torch.stack(cross_losses).mean()


def refine_context_on_support(residual_model, initial_context, features, targets, steps, step_size):
    """Refine the encoder context using only the current support window."""
    # The identification encoder is frozen during the residual-fitting stage.
    # Its prediction is therefore made a leaf latent variable for the inner
    # support update; when the encoder is trainable, retain its direct path.
    context = initial_context if initial_context.requires_grad else initial_context.detach().requires_grad_(True)
    for _ in range(max(0, int(steps))):
        support_prediction = residual_model.forward_with_context(features, context)
        support_loss = F.smooth_l1_loss(support_prediction, targets, beta=LOSS_CONFIG["huber_beta"])
        gradient, = torch.autograd.grad(support_loss, context, retain_graph=True)
        # First-order MAML: retain the learned encoder initialization path but
        # avoid noisy, expensive second-order terms for this 11-D adaptation.
        context = context - float(step_size) * gradient.detach()
    return context


def refine_fast_adapter_on_support(residual_model, context, features, targets, steps, step_size):
    """One compact MAML inner loop that updates residual output fast weights."""
    adapter = torch.zeros(1, 2 * RESIDUAL_OUTPUT_DIM, dtype=features.dtype, device=features.device, requires_grad=True)
    for _ in range(max(0, int(steps))):
        prediction = residual_model.forward_with_context_and_adapter(features, context, adapter)
        support_loss = F.smooth_l1_loss(prediction, targets, beta=LOSS_CONFIG["huber_beta"])
        gradient, = torch.autograd.grad(support_loss, adapter, create_graph=True)
        adapter = adapter - float(step_size) * gradient
    return adapter


def evaluate_meta_batch(residual_model, context_encoder, task_data, task_ids, support_window, query_horizon, dt, device, loss_cfg, deterministic=False, rng=None, context_refinement_steps=0, context_refinement_lr=0.0, fast_adapter_steps=0, fast_adapter_lr=0.0, oracle_context_only=False):
    components = {"query_residual": [], "oracle_query_residual": [], "context_counterfactual": [], "context_translation": [], "context_rotation": [], "context_aerodynamics": [], "cross_trajectory": []}
    context_a_batch, context_b_batch = [], []
    state_windows, action_windows, next_state_windows, adapter_batch = [], [], [], []
    for task_id in task_ids:
        task_entry = task_data[task_id]
        if not has_valid_rollout(task_entry["rollouts"], support_window, query_horizon):
            continue
        task_rng = np.random.default_rng(DEFAULT_SPLIT_SEED + int(task_id)) if deterministic else rng
        sampled = sample_support_pair_query(task_entry["rollouts"], support_window, query_horizon, task_rng)
        if sampled is None:
            continue
        x_support, y_support, x_support_alt, y_support_alt, query_batch = sampled
        x_support, y_support = x_support.to(device), y_support.to(device)
        x_support_alt, y_support_alt = x_support_alt.to(device), y_support_alt.to(device)
        query_batch = {name: value.to(device) for name, value in query_batch.items()}
        task_context = task_entry["context"].to(device)
        if oracle_context_only:
            context_a = task_context.unsqueeze(0)
            context_b = task_context.unsqueeze(0)
        else:
            context_a = context_encoder(x_support, y_support)
            context_b = context_encoder(x_support_alt, y_support_alt)
        context_a = refine_context_on_support(residual_model, context_a, x_support, y_support, context_refinement_steps, context_refinement_lr)
        context_b = refine_context_on_support(residual_model, context_b, x_support_alt, y_support_alt, context_refinement_steps, context_refinement_lr)
        fast_adapter = refine_fast_adapter_on_support(residual_model, context_a, x_support, y_support, fast_adapter_steps, fast_adapter_lr)
        query_pred = residual_model.forward_with_context_and_adapter(query_batch["features"], context_a, fast_adapter)
        components["query_residual"].append(F.smooth_l1_loss(query_pred, query_batch["targets"], beta=loss_cfg["huber_beta"]))
        group_losses, cross_loss = grouped_context_losses(context_a, context_b, task_context)
        oracle_prediction = residual_model.forward_with_context_and_adapter(query_batch["features"], task_context.unsqueeze(0), fast_adapter)
        components["oracle_query_residual"].append(F.smooth_l1_loss(oracle_prediction, query_batch["targets"], beta=loss_cfg["huber_beta"]))
        components["context_counterfactual"].append(F.smooth_l1_loss(query_pred, oracle_prediction, beta=loss_cfg["huber_beta"]))
        for name, loss in group_losses.items():
            components[f"context_{name}"].append(loss)
        components["cross_trajectory"].append(cross_loss)
        context_a_batch.append(context_a.squeeze(0))
        context_b_batch.append(context_b.squeeze(0))
        state_windows.append(query_batch["states"])
        action_windows.append(query_batch["actions"])
        next_state_windows.append(query_batch["next_states"])
        adapter_batch.append(fast_adapter.squeeze(0))
    if not context_a_batch:
        return None
    pos_loss, vel_loss, att_loss, rate_loss = multistep_state_loss(
        residual_model, torch.stack(context_a_batch), torch.stack(state_windows),
        torch.stack(action_windows), torch.stack(next_state_windows), dt, loss_cfg, torch.stack(adapter_batch),
    )
    query_state_loss = (
        loss_cfg["position_weight"] * pos_loss
        + loss_cfg["velocity_weight"] * vel_loss
        + loss_cfg["attitude_weight"] * att_loss
        + loss_cfg["rate_weight"] * rate_loss
    )
    mean_components = {name: torch.stack(values).mean() for name, values in components.items()}
    contrastive_loss = symmetric_contrastive_loss(torch.stack(context_a_batch), torch.stack(context_b_batch), loss_cfg["contrastive_temperature"])
    loss_tensor = (
        loss_cfg["query_residual_weight"] * mean_components["query_residual"]
        + loss_cfg["oracle_query_residual_weight"] * mean_components["oracle_query_residual"]
        + loss_cfg["context_counterfactual_weight"] * mean_components["context_counterfactual"]
        + loss_cfg["query_state_weight"] * query_state_loss
        + loss_cfg["context_translation_weight"] * mean_components["context_translation"]
        + loss_cfg["context_rotation_weight"] * mean_components["context_rotation"]
        + loss_cfg["context_aerodynamics_weight"] * mean_components["context_aerodynamics"]
        + loss_cfg["cross_trajectory_weight"] * mean_components["cross_trajectory"]
        + loss_cfg["contrastive_weight"] * contrastive_loss
    )
    scalar_components = {name: float(value.detach().item()) for name, value in mean_components.items()}
    scalar_components["query_state"] = float(query_state_loss.detach().item())
    scalar_components["contrastive"] = float(contrastive_loss.detach().item())
    return {"loss_tensor": loss_tensor, "loss_value": float(loss_tensor.detach().item()), "components": scalar_components}


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
        "architecture": MODEL_CONFIG["architecture"],
        "physics_context_groups": {name: list(indices) for name, indices in PHYSICS_CONTEXT_GROUPS.items()},
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
        "context_injection": "physics_disentangled",
        "modulation_scale": MODEL_CONFIG["modulation_scale"],
        "context_encoder_hidden_dim": MODEL_CONFIG["context_encoder_hidden_dim"],
        "context_encoder_num_layers": MODEL_CONFIG["context_encoder_num_layers"],
        "support_window": support_window,
        "query_horizon": query_horizon,
        "online_context_ema_decay": TRAINING_CONFIG["online_context_ema_decay"],
        "online_context_warmup_updates": TRAINING_CONFIG["online_context_warmup_updates"],
        "context_refinement_steps": TRAINING_CONFIG["context_refinement_steps"],
        "context_refinement_lr": TRAINING_CONFIG["context_refinement_lr"],
        "online_context_refinement_steps": TRAINING_CONFIG["online_context_refinement_steps"],
        "online_context_refinement_lr": TRAINING_CONFIG["online_context_refinement_lr"],
        "fast_adapter_steps": TRAINING_CONFIG["fast_adapter_steps"],
        "fast_adapter_lr": TRAINING_CONFIG["fast_adapter_lr"],
        "online_fast_adapter_steps": TRAINING_CONFIG["online_fast_adapter_steps"],
        "online_fast_adapter_lr": TRAINING_CONFIG["online_fast_adapter_lr"],
        "loss_cfg": LOSS_CONFIG,
        "best_val_loss": None if best_residual_state is None else best_val_loss,
        "best_val_epoch": None if best_residual_state is None else best_val_epoch,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    print(f"Saved checkpoint to {save_path}")


LOSS_KEYS = (
    "total", "query_residual", "oracle_query_residual", "context_counterfactual", "query_state", "context_translation",
    "context_rotation", "context_aerodynamics", "cross_trajectory", "contrastive",
)


def history_row(epoch, split, phase, result):
    row = {"epoch": epoch, "split": split, "phase": phase, "total": result["loss_value"]}
    row.update({key: result["components"].get(key, np.nan) for key in LOSS_KEYS if key != "total"})
    return row


def save_training_diagnostics(save_path, train_history, val_history):
    """Persist all losses in CSV and one compact, complete diagnostic figure."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = save_path.with_name(f"{save_path.stem}_loss_history.csv")
    rows = [*train_history, *val_history]
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "split", "phase", *LOSS_KEYS))
        writer.writeheader()
        writer.writerows(rows)

    n_columns = 3
    n_rows = int(np.ceil((len(LOSS_KEYS) + 1) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4 * n_rows))
    for axis, key in zip(axes.flat, LOSS_KEYS):
        for label, history, style in (("train", train_history, "-"), ("validation", val_history, "o-")):
            if history:
                axis.plot([row["epoch"] for row in history], [row[key] for row in history], style, markersize=3 if label == "validation" else 0, label=label)
        axis.set_title(key.replace("_", " "))
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.set_yscale("symlog", linthresh=1e-5)
        axis.grid(True, alpha=0.3)
        axis.legend()
    # Raw losses have different physical scales.  The final panel shows their
    # optimizer-weighted contributions, so no second visualization is needed.
    weights = {
        "query_residual": LOSS_CONFIG["query_residual_weight"],
        "oracle_query_residual": LOSS_CONFIG["oracle_query_residual_weight"],
        "context_counterfactual": LOSS_CONFIG["context_counterfactual_weight"],
        "query_state": LOSS_CONFIG["query_state_weight"],
        "context_translation": LOSS_CONFIG["context_translation_weight"],
        "context_rotation": LOSS_CONFIG["context_rotation_weight"],
        "context_aerodynamics": LOSS_CONFIG["context_aerodynamics_weight"],
        "cross_trajectory": LOSS_CONFIG["cross_trajectory_weight"],
        "contrastive": LOSS_CONFIG["contrastive_weight"],
    }
    axis = axes.flat[len(LOSS_KEYS)]
    for key, weight in weights.items():
        for label, history, style in (("train", train_history, "-"), ("validation", val_history, "o-")):
            if history:
                axis.plot([row["epoch"] for row in history], [weight * row[key] for row in history], style, alpha=0.85, markersize=3 if label == "validation" else 0, label=f"{key} ({label})")
    axis.set_title("Weighted loss contributions")
    axis.set_xlabel("epoch")
    axis.set_ylabel("contribution to total loss")
    axis.set_yscale("symlog", linthresh=1e-5)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    figure_path = save_path.with_name(f"{save_path.stem}_training_diagnostics.png")
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Saved loss history to {history_path}")
    print(f"Saved training diagnostics to {figure_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train body-rate/thrust context-meta residual model.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Warm-start residual and context encoder weights for a short ablation fine-tune.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--support-window", type=int, default=DEFAULT_SUPPORT_WINDOW)
    parser.add_argument("--query-horizon", type=int, default=DEFAULT_QUERY_HORIZON)
    parser.add_argument("--val-interval", type=int, default=DEFAULT_VAL_INTERVAL)
    parser.add_argument("--context-warmup-epochs", type=int, default=TRAINING_CONFIG["context_warmup_epochs"], help="Supervise the temporal task-identifier before freezing it for residual fitting.")
    parser.add_argument("--context-refinement-steps", type=int, default=TRAINING_CONFIG["context_refinement_steps"], help="Support-loss updates of the context before every query loss.")
    parser.add_argument("--context-refinement-lr", type=float, default=TRAINING_CONFIG["context_refinement_lr"], help="Inner-loop step size for the normalized context.")
    parser.add_argument("--fast-adapter-steps", type=int, default=TRAINING_CONFIG["fast_adapter_steps"], help="Fast-weight support updates before each query loss.")
    parser.add_argument("--fast-adapter-lr", type=float, default=TRAINING_CONFIG["fast_adapter_lr"], help="Fast-weight inner-loop step size.")
    parser.add_argument("--oracle-context-only", action="store_true", help="Diagnostic upper bound: train/query with true task context only, without the support encoder.")
    parser.add_argument("--seed", type=int, default=43, help="Seed for task split, batches, and model initialization.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    task_data, context_center, context_scale, input_mean, input_scale, target_mean, target_scale = load_task_data(args.csv_path)
    train_ids, val_ids = split_task_ids(task_data.keys(), seed=args.seed)
    train_data = {tid: task_data[tid] for tid in train_ids}
    val_data = {tid: task_data[tid] for tid in val_ids}
    print(f"Loaded {len(task_data)} tasks. Train={len(train_data)} Val={len(val_data)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_batch_size = min(DEFAULT_META_BATCH_SIZE_CAP, len(train_data)) if args.meta_batch_size is None else min(args.meta_batch_size, len(train_data))
    residual_model = PhysicsDisentangledContextResidualMLP(
        input_dim=MODEL_INPUT_DIM,
        output_dim=RESIDUAL_OUTPUT_DIM,
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        context_dim=len(CONTEXT_COLS),
        modulation_scale=MODEL_CONFIG["modulation_scale"],
    ).to(device)
    residual_model.set_normalization(input_mean, input_scale, target_mean, target_scale)
    context_encoder = GroupedTemporalPhysicsSupportSetEncoderV3(
        feature_dim=MODEL_INPUT_DIM,
        target_dim=RESIDUAL_OUTPUT_DIM,
        context_dim=len(CONTEXT_COLS),
        hidden_dim=MODEL_CONFIG["context_encoder_hidden_dim"],
        num_layers=MODEL_CONFIG["context_encoder_num_layers"],
    ).to(device)
    context_encoder.set_normalization(input_mean, input_scale, target_mean, target_scale)
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {args.init_checkpoint}")
        initial_payload = torch.load(args.init_checkpoint, map_location=device)
        if initial_payload.get("architecture") != MODEL_CONFIG["architecture"]:
            raise ValueError(f"Initial checkpoint architecture={initial_payload.get('architecture')} is incompatible with {MODEL_CONFIG['architecture']}.")
        residual_model.load_state_dict(initial_payload["model_state_dict"])
        context_encoder.load_state_dict(initial_payload["context_encoder_state_dict"])
        print(f"Warm-started model and context encoder from {args.init_checkpoint}.")
    parameters = list(residual_model.parameters()) + list(context_encoder.parameters())
    requested_warmup = max(0, int(args.context_warmup_epochs))
    context_warmup_epochs = min(requested_warmup, max(0, args.epochs - 1))
    two_stage_training = context_warmup_epochs > 0
    residual_epochs = max(1, args.epochs - context_warmup_epochs)
    initial_lr = TRAINING_CONFIG["context_warmup_lr"] if two_stage_training else TRAINING_CONFIG["meta_lr"]
    optimizer = torch.optim.Adam(parameters, lr=initial_lr)
    context_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, context_warmup_epochs)) if two_stage_training else None
    residual_scheduler = None if two_stage_training else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=residual_epochs)
    if two_stage_training:
        print(f"Two-stage training: {context_warmup_epochs} context-supervision epoch(s), then {residual_epochs} frozen-context residual epoch(s).")
    train_losses, train_history, val_history = [], [], []
    best_val_loss, best_val_epoch = float("inf"), -1
    best_residual_state, best_encoder_state = None, None
    best_context_group_loss = {name: float("inf") for name in PHYSICS_CONTEXT_GROUPS}
    best_context_group_epoch = {name: -1 for name in PHYSICS_CONTEXT_GROUPS}
    best_context_branch_state = {name: None for name in PHYSICS_CONTEXT_GROUPS}
    train_id_list = list(train_data.keys())
    train_rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        is_context_warmup = two_stage_training and epoch < context_warmup_epochs
        if two_stage_training and epoch == context_warmup_epochs:
            for name in PHYSICS_CONTEXT_GROUPS:
                branch_state = best_context_branch_state[name]
                if branch_state is not None:
                    getattr(context_encoder, f"{name}_branch").load_state_dict(branch_state)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = TRAINING_CONFIG["meta_lr"]
            residual_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=residual_epochs)
            selected_epochs = ", ".join(f"{name}={best_context_group_epoch[name]}" for name in PHYSICS_CONTEXT_GROUPS)
            print(f"Restored context branch validation optima ({selected_epochs}); now optimizing support-update/query loss with a true-context residual anchor.")
        phase = "context_supervision" if is_context_warmup else "physics_grounded_meta"
        active_loss_cfg = CONTEXT_WARMUP_LOSS_CONFIG if is_context_warmup else (PHYSICS_GROUNDED_META_LOSS_CONFIG if two_stage_training else LOSS_CONFIG)
        optimizer.zero_grad()
        task_ids = train_rng.choice(train_id_list, meta_batch_size, replace=False)
        refinement_steps = 0 if is_context_warmup else args.context_refinement_steps
        train_result = evaluate_meta_batch(residual_model, context_encoder, train_data, task_ids, args.support_window, args.query_horizon, DATA_DT, device, active_loss_cfg, rng=train_rng, context_refinement_steps=refinement_steps, context_refinement_lr=args.context_refinement_lr, fast_adapter_steps=0 if is_context_warmup else args.fast_adapter_steps, fast_adapter_lr=args.fast_adapter_lr, oracle_context_only=args.oracle_context_only)
        if train_result is None:
            continue
        train_result["loss_tensor"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=TRAINING_CONFIG["grad_clip_norm"])
        optimizer.step()
        if is_context_warmup:
            context_scheduler.step()
        else:
            residual_scheduler.step()
        train_losses.append(train_result["loss_value"])
        train_history.append(history_row(epoch + 1, "train", phase, train_result))
        current_val_loss = None
        if ((epoch + 1) % args.val_interval == 0) or epoch == 0 or epoch == args.epochs - 1:
            residual_model.eval()
            context_encoder.eval()
            val_result = evaluate_meta_batch(residual_model, context_encoder, val_data, list(val_data.keys()), args.support_window, args.query_horizon, DATA_DT, device, active_loss_cfg, deterministic=True, context_refinement_steps=refinement_steps, context_refinement_lr=args.context_refinement_lr, fast_adapter_steps=0 if is_context_warmup else args.fast_adapter_steps, fast_adapter_lr=args.fast_adapter_lr, oracle_context_only=args.oracle_context_only) if val_data else None
            residual_model.train()
            context_encoder.train()
            if val_result is not None:
                current_val_loss = val_result["loss_value"]
                val_history.append(history_row(epoch + 1, "validation", phase, val_result))
                if is_context_warmup:
                    for name in PHYSICS_CONTEXT_GROUPS:
                        group_loss = float(val_result["components"][f"context_{name}"])
                        if group_loss < best_context_group_loss[name]:
                            best_context_group_loss[name] = group_loss
                            best_context_group_epoch[name] = epoch + 1
                            best_context_branch_state[name] = deepcopy(getattr(context_encoder, f"{name}_branch").state_dict())
                elif current_val_loss < best_val_loss:
                    best_val_loss, best_val_epoch = current_val_loss, epoch + 1
                    with torch.no_grad():
                        residual_model.reset_context()
                    best_residual_state = deepcopy(residual_model.state_dict())
                    best_encoder_state = deepcopy(context_encoder.state_dict())
        if epoch % DEFAULT_LOG_INTERVAL == 0 or epoch == args.epochs - 1:
            comps = train_result["components"]
            msg = f"[Epoch {epoch + 1:05d} {phase}] train={train_result['loss_value']:.6f} q_res={comps['query_residual']:.5f} q_state={comps['query_state']:.5f}"
            if current_val_loss is not None:
                msg += f" val={current_val_loss:.6f}"
            if best_residual_state is not None:
                msg += f" best={best_val_loss:.6f}@{best_val_epoch}"
            print(msg)
    save_training_diagnostics(args.save_path, train_history, val_history)
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
    # Keep the historical filename as the public command while routing new
    # runs to the first-stage latent-only body-frame implementation.  The
    # legacy physics-supervised trainer above is retained for old checkpoints.
    from Offline_Train_BodyFrameAccel import main as bodyframe_accel_main

    bodyframe_accel_main()
