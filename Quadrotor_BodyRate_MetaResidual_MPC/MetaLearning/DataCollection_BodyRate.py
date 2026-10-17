"""Collect body-rate/thrust residual data for context-meta MPC."""

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from uuid import uuid4

import casadi as cs
import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bodyrate_common import (  # noqa: E402
    CONTEXT_COLS,
    CONTROL_COLS,
    DEFAULT_CTRL_FREQ,
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_N_HORIZON,
    DEFAULT_SIM_TIME,
    DEFAULT_T_HORIZON,
    DynamicsTask,
    MPC,
    BodyRateNominalDynamics,
    ReferenceConfig,
    STATE_COLS,
    TRACKED_DERIVATIVE_INDICES,
    finite_difference_targets,
    input_reference,
    make_env_config,
    bodyrate_control_bounds,
    reference_state,
    set_tracking_references,
    step_bodyrate_env,
    wrap_state_angles,
)
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor  # noqa: E402


SAVE_DIR = DEFAULT_DATASET_DIR
SAVE_PATH = DEFAULT_DATASET_PATH
DEFAULT_SHARD_DIR = SAVE_DIR / "bodyrate_bodyframe_accel_shards"
DT = 1.0 / DEFAULT_CTRL_FREQ
T = DEFAULT_SIM_TIME
STEPS = int(T / DT)
N = DEFAULT_N_HORIZON
T_HORIZON = DEFAULT_T_HORIZON
COLLECTION_GUI = False
DONE_ON_OUT_OF_BOUND = False
TASK_SEED_STRIDE = 1000
_WORKER_NOMINAL_MODEL = None
_WORKER_NOMINAL_SOLVER = None
_WORKER_NOMINAL_FN = None
SHARD_SCHEMA_VERSION = 2
MAX_TRACKING_POSITION_ERROR_M = 1.5
MAX_CALIBRATION_POSITION_ERROR_M = 1.0
MAX_CALIBRATION_SPEED_MPS = 2.5
MAX_CALIBRATION_ATTITUDE_RAD = 0.65
DEFAULT_TASK_TIMEOUT_SEC = 180.0

MASS_RATIOS = np.array([0.85, 1.0, 1.15, 1.30])
IXX_RATIOS = np.array([0.9, 1.0, 1.15])
IYY_RATIOS = np.array([0.9, 1.0, 1.15])
IZZ_RATIOS = np.array([0.9, 1.0, 1.15])
THRUST_SCALES = np.array([0.95, 1.0, 1.05])
RATE_TAU_SCALES = np.array([0.9, 1.0, 1.15])
# Aerodynamic perturbations must be large enough to be identifiable in the
# 3.5--4 m/s trajectories. The former highest level (.005, .0015) makes a
# light airframe with slow body-rate dynamics fail a 4 m/s closed-loop quality
# gate. The retained upper level is still stronger than the legacy protocol,
# but stays inside the feasible high-speed envelope.
DRAG_LINEAR_VALUES = np.array([0.0, 0.0015, 0.0030, 0.0040])
DRAG_QUAD_VALUES = np.array([0.0, 0.0007, 0.0012])
WIND_VALUES = [
    (0.0, 0.0, 0.0),
    (0.0025, 0.0, 0.0),
    (-0.0025, 0.002, 0.0),
    (0.0, -0.0025, 0.0),
    (0.0025, -0.0015, 0.0),
]

# The wind vector is one joint discrete factor: independently permuting its
# x/y components would fabricate nonphysical wind combinations. The other
# eight factors are scalar physical parameters. A capped collection begins
# with deliberately chosen corner/nominal anchors and balances every factor
# marginally across the remaining tasks.
TASK_AXIS_NAMES = (
    "mass_ratio", "ixx_ratio", "iyy_ratio", "izz_ratio", "thrust_scale",
    "rate_tau_scale", "drag_linear", "drag_quad", "wind",
)
TASK_AXIS_VALUES = (
    MASS_RATIOS, IXX_RATIOS, IYY_RATIOS, IZZ_RATIOS, THRUST_SCALES,
    RATE_TAU_SCALES, DRAG_LINEAR_VALUES, DRAG_QUAD_VALUES, WIND_VALUES,
)
TASK_AXIS_SHAPE = tuple(len(values) for values in TASK_AXIS_VALUES)
ANCHOR_LEVELS = (
    (1, 1, 1, 1, 1, 1, 0, 0, 0),  # nominal
    (0, 0, 0, 0, 2, 0, 0, 0, 0),  # light/low-inertia, high thrust
    (3, 2, 2, 2, 0, 2, 3, 2, 4),  # coupled adverse corner
    (3, 1, 1, 1, 0, 2, 0, 0, 0),  # heavy, weak, slow actuator
    (1, 1, 1, 1, 1, 1, 3, 2, 4),  # aerodynamics/wind dominated
    (0, 0, 0, 0, 2, 0, 3, 2, 3),  # agile, high drag
    (2, 2, 0, 0, 0, 2, 2, 1, 1),  # x-inertia/rate-lag contrast
    (2, 0, 2, 0, 0, 2, 2, 1, 3),  # y-inertia/rate-lag contrast
    (2, 0, 0, 2, 0, 2, 2, 1, 2),  # z-inertia/rate-lag contrast
    (3, 2, 2, 2, 2, 0, 1, 0, 1),  # heavy but responsive/high thrust
    (0, 2, 1, 1, 2, 2, 3, 2, 0),  # light, bounded-inertia, slow, high drag
    (3, 0, 0, 0, 1, 1, 1, 1, 4),  # mass-only contrast under wind
)
MIN_BALANCED_TASKS = 4 * len(ANCHOR_LEVELS)

DATASET_COLUMNS = [
    "task_id",
    "episode_id",
    "reference_id",
    "time",
    *CONTEXT_COLS,
    "center_x",
    "center_y",
    "center_z",
    "radius",
    "z_amp",
    "yaw_ref",
    "period",
    "traj_type",
    "y_radius",
    *STATE_COLS,
    *CONTROL_COLS,
    "x_ddot_true",
    "y_ddot_true",
    "z_ddot_true",
    "x_ddot_nom",
    "y_ddot_nom",
    "z_ddot_nom",
    "res_bx_ddot",
    "res_by_ddot",
    "res_bz_ddot",
    "motor_u1",
    "motor_u2",
    "motor_u3",
    "motor_u4",
    "mass",
    "ixx",
    "iyy",
    "izz",
]


def circle_reference_for_speed(radius: float, speed_mps: float, center=(0.0, 0.0, 1.0), yaw_ref: float = 0.0) -> ReferenceConfig:
    return ReferenceConfig(period=2.0 * np.pi * radius / speed_mps, radius=radius, center=center, z_amp=0.0, yaw_ref=yaw_ref, traj_type="circle")


def figure8_reference_for_speed(x_radius: float, y_radius: float, speed_mps: float, center=(0.0, 0.0, 1.0), z_amp: float = 0.0, yaw_ref: float = 0.0) -> ReferenceConfig:
    """Build a figure-eight whose true peak planar speed is ``speed_mps``.

    ``reference_state`` uses ``x_dot=x_radius*omega*cos(phi)`` and
    ``y_dot=y_radius*omega*cos(2*phi)``. Their maxima coincide at phi=0,
    hence the period below gives the requested peak speed exactly.
    """
    period = 2.0 * np.pi * np.hypot(x_radius, y_radius) / speed_mps
    return ReferenceConfig(period=period, radius=x_radius, y_radius=y_radius, center=center, z_amp=z_amp, yaw_ref=yaw_ref, traj_type="figure8")


REFERENCE_TASKS = [
    ReferenceConfig(period=12.0, radius=0.60, center=(0.0, 0.0, 1.0), z_amp=0.15, yaw_ref=0.0),
    ReferenceConfig(period=10.0, radius=0.60, center=(0.0, 0.0, 1.0), z_amp=0.18, yaw_ref=0.0),
    ReferenceConfig(period=12.0, radius=0.60, y_radius=0.45, center=(0.0, 0.0, 1.0), z_amp=0.15, yaw_ref=0.0, traj_type="figure8"),
    ReferenceConfig(period=10.0, radius=0.52, y_radius=0.40, center=(0.2, -0.15, 1.05), z_amp=0.18, yaw_ref=0.1, traj_type="figure8"),
    circle_reference_for_speed(radius=3.0, speed_mps=2.0),
    circle_reference_for_speed(radius=3.0, speed_mps=3.0),
    # High speed is retained deliberately, while larger radii keep the
    # acceleration demand feasible for the heavy/low-thrust corner tasks.
    circle_reference_for_speed(radius=4.5, speed_mps=3.5, center=(0.0, 0.0, 1.15)),
    circle_reference_for_speed(radius=7.0, speed_mps=4.0, center=(0.0, 0.0, 1.15)),
    figure8_reference_for_speed(x_radius=4.8, y_radius=1.6, speed_mps=4.0, center=(0.0, 0.0, 1.15), z_amp=0.08),
]

# Bounded body-rate/thrust probes make inertia, rate-lag, thrust, and drag
# identifiable from the support window.  Tracking-only rollouts under-excite
# these channels and lead to a context loss that cannot be optimized.
CALIBRATION_PROFILES = (
    {"phase": 0.0, "rate_scale": 1.0, "thrust_phase": 0.0},
    {"phase": 1.17, "rate_scale": 0.85, "thrust_phase": 0.73},
    {"phase": 2.31, "rate_scale": 0.70, "thrust_phase": 1.46},
)


def all_task_specs():
    """Construct the full discrete bank before closed-loop admissibility filtering."""
    return [
        DynamicsTask(
            mass_ratio=float(mass_ratio), ixx_ratio=float(ixx_ratio),
            iyy_ratio=float(iyy_ratio), izz_ratio=float(izz_ratio),
            thrust_scale=float(thrust_scale), rate_tau_scale=float(rate_tau_scale),
            drag_linear=float(drag_linear), drag_quad=float(drag_quad),
            wind=tuple(float(value) for value in wind),
        )
        for mass_ratio, ixx_ratio, iyy_ratio, izz_ratio, thrust_scale, rate_tau_scale, drag_linear, drag_quad, wind in itertools.product(
            MASS_RATIOS, IXX_RATIOS, IYY_RATIOS, IZZ_RATIOS,
            THRUST_SCALES, RATE_TAU_SCALES, DRAG_LINEAR_VALUES,
            DRAG_QUAD_VALUES, WIND_VALUES,
        )
    ]


def _is_closed_loop_admissible(task: DynamicsTask) -> bool:
    """Reject only the geometry/bandwidth corner that fails the data quality gate.

    The benchmark represents one fixed-size airframe.  A 15% lighter vehicle
    with a mean inertia above 1.05 and the slowest rate actuator is outside
    this geometry-limited closed-loop envelope; it produces solver failures
    even on 3--3.5 m/s references.  Other task factors remain independent.
    """
    mean_inertia_ratio = float(np.mean([task.ixx_ratio, task.iyy_ratio, task.izz_ratio]))
    return not (
        task.mass_ratio <= 0.85 + 1e-12
        and task.rate_tau_scale >= 1.15 - 1e-12
        and mean_inertia_ratio > 1.05 + 1e-12
    )


def _admissible_task_indices(candidates: list[DynamicsTask]) -> np.ndarray:
    return np.asarray(
        [index for index, task in enumerate(candidates) if _is_closed_loop_admissible(task)],
        dtype=int,
    )


def _balanced_task_indices(max_tasks: int, rng: np.random.Generator) -> np.ndarray:
    """Return anchor-first, unique tasks with balanced discrete marginals."""
    anchor_indices = np.asarray(
        [np.ravel_multi_index(levels, TASK_AXIS_SHAPE) for levels in ANCHOR_LEVELS],
        dtype=int,
    )
    if len(np.unique(anchor_indices)) != len(anchor_indices):
        raise RuntimeError("Anchor task definitions must be unique.")
    remaining = max_tasks - len(anchor_indices)
    if remaining < 0:
        raise ValueError("Balanced sampling requires room for all anchor tasks.")

    anchor_by_axis = np.asarray(ANCHOR_LEVELS, dtype=int)
    for _ in range(1024):
        level_columns = []
        feasible = True
        for axis_index, axis_size in enumerate(TASK_AXIS_SHAPE):
            target_counts = np.full(axis_size, max_tasks // axis_size, dtype=int)
            extras = max_tasks % axis_size
            if extras:
                target_counts[rng.permutation(axis_size)[:extras]] += 1
            anchor_counts = np.bincount(anchor_by_axis[:, axis_index], minlength=axis_size)
            remaining_counts = target_counts - anchor_counts
            if np.any(remaining_counts < 0):
                feasible = False
                break
            levels = np.repeat(np.arange(axis_size, dtype=int), remaining_counts)
            if len(levels) != remaining:
                feasible = False
                break
            level_columns.append(rng.permutation(levels))
        if not feasible:
            break
        candidates = np.column_stack(level_columns)
        candidate_indices = np.ravel_multi_index(candidates.T, TASK_AXIS_SHAPE)
        selected = np.concatenate((anchor_indices, candidate_indices))
        if len(np.unique(selected)) == max_tasks:
            return selected
    raise RuntimeError("Could not construct a unique balanced task design; change the seed or task count.")


def _constrained_balanced_task_indices(max_tasks: int, rng: np.random.Generator, candidates: list[DynamicsTask]) -> np.ndarray:
    """Replace inadmissible corners while minimizing discrete marginal drift."""
    base_indices = _balanced_task_indices(max_tasks, rng)
    base_levels = np.asarray(np.unravel_index(base_indices, TASK_AXIS_SHAPE)).T
    target_counts = [
        np.bincount(base_levels[:, axis_index], minlength=axis_size)
        for axis_index, axis_size in enumerate(TASK_AXIS_SHAPE)
    ]
    admissible_indices = _admissible_task_indices(candidates)
    admissible_set = {int(index) for index in admissible_indices}
    selected = [int(index) for index in base_indices if int(index) in admissible_set]
    selected_set = set(selected)
    current_counts = [np.zeros(axis_size, dtype=int) for axis_size in TASK_AXIS_SHAPE]
    for candidate_index in selected:
        levels = np.unravel_index(candidate_index, TASK_AXIS_SHAPE)
        for axis_index, level in enumerate(levels):
            current_counts[axis_index][int(level)] += 1

    for _ in range(max_tasks - len(selected)):
        replacement, best_score = None, -1
        for candidate_index in rng.permutation(admissible_indices):
            candidate_index = int(candidate_index)
            if candidate_index in selected_set:
                continue
            levels = np.unravel_index(candidate_index, TASK_AXIS_SHAPE)
            score = sum(
                max(0, int(target_counts[axis_index][int(level)] - current_counts[axis_index][int(level)]))
                for axis_index, level in enumerate(levels)
            )
            if score > best_score:
                replacement, best_score = candidate_index, score
        if replacement is None:
            raise RuntimeError("No admissible replacement is available for constrained task sampling.")
        selected.append(replacement)
        selected_set.add(replacement)
        levels = np.unravel_index(replacement, TASK_AXIS_SHAPE)
        for axis_index, level in enumerate(levels):
            current_counts[axis_index][int(level)] += 1
    return np.asarray(selected, dtype=int)


def _sampling_name(max_tasks: int | None) -> str:
    if max_tasks is None or max_tasks == int(np.prod(TASK_AXIS_SHAPE)):
        return "closed_loop_admissible_bank"
    if max_tasks >= MIN_BALANCED_TASKS:
        return "anchor_stratified_admissible_without_replacement"
    return "uniform_without_replacement_from_admissible_bank"


def iter_task_specs(max_tasks=None, seed=43):
    """Sample unique, closed-loop-admissible dynamics tasks reproducibly."""
    candidates = all_task_specs()
    admissible_indices = _admissible_task_indices(candidates)
    if max_tasks is None:
        selected_indices = admissible_indices
    else:
        if max_tasks < 1:
            return
        if max_tasks > len(admissible_indices):
            raise ValueError(f"Requested {max_tasks} tasks, but only {len(admissible_indices)} admissible combinations exist.")
        rng = np.random.default_rng(seed)
        if max_tasks >= MIN_BALANCED_TASKS:
            selected_indices = _constrained_balanced_task_indices(max_tasks, rng, candidates)
        else:
            selected_indices = rng.choice(admissible_indices, size=max_tasks, replace=False)
    for task_id, candidate_idx in enumerate(selected_indices):
        yield task_id, candidates[int(candidate_idx)]


def task_manifest(task_specs, seed: int) -> dict:
    contexts = [task.context_values() for _, task in task_specs]
    return {
        "sampling": _sampling_name(len(task_specs)),
        "seed": seed,
        "num_tasks": len(task_specs),
        "num_unique_contexts": len({tuple(context[key] for key in CONTEXT_COLS) for context in contexts}),
        "candidate_pool_size": len(_admissible_task_indices(all_task_specs())),
        "context_coverage": {key: sorted({float(context[key]) for context in contexts}) for key in CONTEXT_COLS},
        "reference_count_per_task": len(REFERENCE_TASKS) + len(CALIBRATION_PROFILES),
        "calibration_rollouts_per_task": len(CALIBRATION_PROFILES),
        "calibration_protocol": "closed_loop_hover_mpc_plus_bounded_multisine",
        "control_frequency_hz": DEFAULT_CTRL_FREQ,
        "episode_duration_sec": T,
    }


def worker_nominal_components(env):
    """Build the invariant nominal MPC once per collection worker."""
    global _WORKER_NOMINAL_MODEL, _WORKER_NOMINAL_SOLVER, _WORKER_NOMINAL_FN
    if _WORKER_NOMINAL_SOLVER is None:
        _WORKER_NOMINAL_MODEL = BodyRateNominalDynamics(env).model()
        _WORKER_NOMINAL_FN = cs.Function("f_bodyrate_nom", [_WORKER_NOMINAL_MODEL.x, _WORKER_NOMINAL_MODEL.u], [_WORKER_NOMINAL_MODEL.f_nominal])
        _WORKER_NOMINAL_SOLVER = MPC(model=_WORKER_NOMINAL_MODEL, n_horizon=N, t_horizon=T_HORIZON).solver
    return _WORKER_NOMINAL_SOLVER, _WORKER_NOMINAL_FN


def _checked_step(env, current_state: np.ndarray, control_input: np.ndarray, task: DynamicsTask, ref_cfg: ReferenceConfig, next_time_sec: float, calibration: bool):
    next_obs, _, done, info = step_bodyrate_env(env, current_state, control_input, task)
    next_state = wrap_state_angles(np.array(next_obs[:12], dtype=float))
    motor_forces = np.asarray(info["executed_motor_forces"], dtype=float)
    arrays = (current_state, control_input, next_state, motor_forces)
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("non-finite state, command, or motor force")
    reference = reference_state(next_time_sec, ref_cfg)
    position_error = float(np.linalg.norm(next_state[[0, 2, 4]] - reference[[0, 2, 4]]))
    if calibration:
        speed = float(np.linalg.norm(next_state[[1, 3, 5]]))
        attitude = float(np.max(np.abs(next_state[6:8])))
        if position_error > MAX_CALIBRATION_POSITION_ERROR_M:
            raise RuntimeError(f"calibration position error {position_error:.3f} m exceeds {MAX_CALIBRATION_POSITION_ERROR_M:.3f} m")
        if speed > MAX_CALIBRATION_SPEED_MPS:
            raise RuntimeError(f"calibration speed {speed:.3f} m/s exceeds {MAX_CALIBRATION_SPEED_MPS:.3f} m/s")
        if attitude > MAX_CALIBRATION_ATTITUDE_RAD:
            raise RuntimeError(f"calibration attitude {attitude:.3f} rad exceeds {MAX_CALIBRATION_ATTITUDE_RAD:.3f} rad")
    elif position_error > MAX_TRACKING_POSITION_ERROR_M:
        raise RuntimeError(f"tracking position error {position_error:.3f} m exceeds {MAX_TRACKING_POSITION_ERROR_M:.3f} m")
    return next_state, info


def collect_episode(task_id, episode_id, reference_id, task: DynamicsTask, ref_cfg: ReferenceConfig):
    env = Quadrotor(
        **make_env_config(
            seed=task_id * TASK_SEED_STRIDE + episode_id,
            gui=COLLECTION_GUI,
            done_on_out_of_bound=DONE_ON_OUT_OF_BOUND,
            episode_len_sec=T,
            task=task,
            init_state=reference_state(0.0, ref_cfg),
        )
    )
    rows = []
    try:
        solver, nominal_fn = worker_nominal_components(env)
        solver.reset()
        u_ref = input_reference(env)
        obs, _ = env.reset()
        current_state = wrap_state_angles(np.array(obs[:12], dtype=float))
        y_radius = ref_cfg.radius if ref_cfg.y_radius is None else ref_cfg.y_radius
        context = task.context_values()

        for step in range(STEPS):
            time_sec = step * DT
            set_tracking_references(solver, time_sec, N, T_HORIZON, ref_cfg, u_ref, [])
            solver.set(0, "lbx", current_state)
            solver.set(0, "ubx", current_state)
            status = solver.solve()
            if status != 0:
                raise RuntimeError(f"MPC solve failed at task={task_id}, episode={episode_id}, step={step}, status={status}")

            control_input = np.array(solver.get(0, "u"), dtype=float)
            next_state, info = _checked_step(env, current_state, control_input, task, ref_cfg, time_sec + DT, calibration=False)
            true_targets = (next_state[TRACKED_DERIVATIVE_INDICES] - current_state[TRACKED_DERIVATIVE_INDICES]) / DT
            nominal_targets = nominal_fn(current_state, control_input).full().flatten()[TRACKED_DERIVATIVE_INDICES]
            # Store the residual in body coordinates.  This is the exact
            # three-vector used by the learned MPC dynamics and avoids tying
            # a task representation to a particular world-frame heading.
            residual_targets = finite_difference_targets(current_state, next_state, control_input, nominal_fn, DT)

            rows.append([
                task_id,
                episode_id,
                reference_id,
                time_sec,
                *[context[name] for name in CONTEXT_COLS],
                ref_cfg.center[0],
                ref_cfg.center[1],
                ref_cfg.center[2],
                ref_cfg.radius,
                ref_cfg.z_amp,
                ref_cfg.yaw_ref,
                ref_cfg.period,
                ref_cfg.traj_type,
                y_radius,
                *current_state,
                *control_input,
                *true_targets,
                *nominal_targets,
                *residual_targets,
                *info["executed_motor_forces"],
                env.MASS,
                env.J[0, 0],
                env.J[1, 1],
                env.J[2, 2],
            ])
            current_state = next_state
    finally:
        env.close()
    return rows


def calibration_command(time_sec: float, env, profile: dict) -> np.ndarray:
    """Persistent but bounded excitation for context identifiability."""
    phase = float(profile["phase"])
    rate_scale = float(profile["rate_scale"])
    hover = input_reference(env)[3]
    p_cmd = rate_scale * (0.72 * np.sin(2.0 * np.pi * 0.65 * time_sec + phase) + 0.28 * np.sin(2.0 * np.pi * 1.35 * time_sec + 0.4 * phase))
    q_cmd = rate_scale * (0.66 * np.sin(2.0 * np.pi * 0.83 * time_sec + 0.7 + phase) + 0.22 * np.sin(2.0 * np.pi * 1.57 * time_sec))
    r_cmd = rate_scale * (0.55 * np.sin(2.0 * np.pi * 1.10 * time_sec + 0.3 + phase))
    thrust = hover * (1.0 + 0.10 * np.sin(2.0 * np.pi * 0.42 * time_sec + float(profile["thrust_phase"])) + 0.05 * np.sin(2.0 * np.pi * 0.97 * time_sec))
    return np.array([p_cmd, q_cmd, r_cmd, thrust], dtype=float)


def calibration_excitation(time_sec: float, env, profile: dict) -> np.ndarray:
    """Small persistent probe added to a closed-loop hover MPC command."""
    phase = float(profile["phase"])
    rate_scale = float(profile["rate_scale"])
    hover = input_reference(env)[3]
    p_delta = rate_scale * (0.42 * np.sin(2.0 * np.pi * 0.65 * time_sec + phase) + 0.14 * np.sin(2.0 * np.pi * 1.35 * time_sec + 0.4 * phase))
    q_delta = rate_scale * (0.38 * np.sin(2.0 * np.pi * 0.83 * time_sec + 0.7 + phase) + 0.13 * np.sin(2.0 * np.pi * 1.57 * time_sec))
    r_delta = rate_scale * (0.30 * np.sin(2.0 * np.pi * 1.10 * time_sec + 0.3 + phase))
    thrust_delta = hover * (0.050 * np.sin(2.0 * np.pi * 0.42 * time_sec + float(profile["thrust_phase"])) + 0.020 * np.sin(2.0 * np.pi * 0.97 * time_sec))
    return np.array([p_delta, q_delta, r_delta, thrust_delta], dtype=float)


def collect_calibration_episode(task_id, episode_id, reference_id, task: DynamicsTask, profile: dict):
    """Collect one excitation rollout without leaking the hidden task context."""
    probe_ref = ReferenceConfig(period=T, radius=0.0, center=(0.0, 0.0, 1.0), z_amp=0.0, yaw_ref=0.0)
    env = Quadrotor(
        **make_env_config(
            seed=task_id * TASK_SEED_STRIDE + episode_id,
            gui=COLLECTION_GUI,
            done_on_out_of_bound=DONE_ON_OUT_OF_BOUND,
            episode_len_sec=T,
            task=task,
            init_state=reference_state(0.0, probe_ref),
        )
    )
    rows = []
    try:
        solver, nominal_fn = worker_nominal_components(env)
        solver.reset()
        u_ref = input_reference(env)
        control_lower, control_upper = bodyrate_control_bounds(env)
        obs, _ = env.reset()
        current_state = wrap_state_angles(np.array(obs[:12], dtype=float))
        context = task.context_values()
        for step in range(STEPS):
            time_sec = step * DT
            set_tracking_references(solver, time_sec, N, T_HORIZON, probe_ref, u_ref, [])
            solver.set(0, "lbx", current_state)
            solver.set(0, "ubx", current_state)
            status = solver.solve()
            if status != 0:
                raise RuntimeError(f"calibration MPC solve failed at task={task_id}, episode={episode_id}, step={step}, status={status}")
            nominal_control = np.array(solver.get(0, "u"), dtype=float)
            control_input = np.clip(nominal_control + calibration_excitation(time_sec, env, profile), control_lower, control_upper)
            next_state, info = _checked_step(env, current_state, control_input, task, probe_ref, time_sec + DT, calibration=True)
            true_targets = (next_state[TRACKED_DERIVATIVE_INDICES] - current_state[TRACKED_DERIVATIVE_INDICES]) / DT
            nominal_targets = nominal_fn(current_state, control_input).full().flatten()[TRACKED_DERIVATIVE_INDICES]
            residual_targets = finite_difference_targets(current_state, next_state, control_input, nominal_fn, DT)
            rows.append([
                task_id, episode_id, reference_id, time_sec,
                *[context[name] for name in CONTEXT_COLS],
                probe_ref.center[0], probe_ref.center[1], probe_ref.center[2],
                probe_ref.radius, probe_ref.z_amp, probe_ref.yaw_ref, probe_ref.period,
                "calibration_probe", probe_ref.radius,
                *current_state, *control_input, *true_targets, *nominal_targets,
                *residual_targets, *info["executed_motor_forces"],
                env.MASS, env.J[0, 0], env.J[1, 1], env.J[2, 2],
            ])
            current_state = next_state
    finally:
        env.close()
    return rows


def episode_specs():
    """Return the fixed tracking/probe protocol used by every dynamics task."""
    specs = [
        {"episode_id": reference_id, "reference_id": reference_id, "kind": "tracking"}
        for reference_id in range(len(REFERENCE_TASKS))
    ]
    specs.extend(
        {"episode_id": profile_id, "reference_id": profile_id, "kind": "calibration"}
        for profile_id in range(len(REFERENCE_TASKS), len(REFERENCE_TASKS) + len(CALIBRATION_PROFILES))
    )
    return specs


def _episode_key(task_id: int, episode_id: int) -> str:
    return f"{int(task_id)}:{int(episode_id)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _error_text(exc: Exception) -> tuple[str, str]:
    return type(exc).__name__, str(exc)[:800]


def _failed_episode_result(task_id: int, task: DynamicsTask, episode_id: int, kind: str, exc: Exception) -> dict:
    error_type, error_message = _error_text(exc)
    return {
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "reference_id": int(episode_id),
        "kind": kind,
        "status": "failed",
        "row_count": 0,
        "rows": [],
        "error_type": error_type,
        "error_message": error_message,
        "context": task.context_values(),
    }


def _collect_episode_result(task_id: int, task: DynamicsTask, episode_id: int) -> dict:
    """Collect one complete episode or return a serializable failure record.

    Partial trajectories are intentionally discarded: the trainer assumes a
    rollout is temporally contiguous, so a broken suffix must not silently
    become a shorter support/query trajectory.
    """
    if episode_id < len(REFERENCE_TASKS):
        kind = "tracking"
        collect_fn = lambda: collect_episode(task_id, episode_id, episode_id, task, REFERENCE_TASKS[episode_id])
    else:
        kind = "calibration"
        profile_index = episode_id - len(REFERENCE_TASKS)
        collect_fn = lambda: collect_calibration_episode(task_id, episode_id, episode_id, task, CALIBRATION_PROFILES[profile_index])
    try:
        rows = collect_fn()
        if len(rows) != STEPS:
            raise RuntimeError(f"episode produced {len(rows)} rows; expected {STEPS}")
    except Exception as exc:
        return _failed_episode_result(task_id, task, episode_id, kind, exc)
    return {
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "reference_id": int(episode_id),
        "kind": kind,
        "status": "complete",
        "row_count": len(rows),
        "rows": rows,
        "error_type": None,
        "error_message": None,
        "context": task.context_values(),
    }


def collect_task_rollouts(task_spec):
    """Collect only requested unfinished episodes for one task.

    This outer result boundary ensures an episode failure never terminates the
    worker pool or discards other finished episodes in the same task.
    """
    task_id, task, requested_episode_ids = task_spec
    episode_results = []
    for episode_id in sorted(int(item) for item in requested_episode_ids):
        try:
            episode_results.append(_collect_episode_result(task_id, task, episode_id))
        except Exception as exc:
            kind = "tracking" if episode_id < len(REFERENCE_TASKS) else "calibration"
            episode_results.append(_failed_episode_result(task_id, task, episode_id, kind, exc))
    return {"task_id": int(task_id), "context": task.context_values(), "episodes": episode_results}


def _failed_task_result(task_spec, exc: Exception) -> dict:
    task_id, task, requested_episode_ids = task_spec
    episodes = []
    for episode_id in requested_episode_ids:
        kind = "tracking" if episode_id < len(REFERENCE_TASKS) else "calibration"
        episodes.append(_failed_episode_result(task_id, task, episode_id, kind, exc))
    error_type, error_message = _error_text(exc)
    return {
        "task_id": int(task_id),
        "context": task.context_values(),
        "episodes": episodes,
        "task_error_type": error_type,
        "task_error_message": error_message,
    }


def _reference_description(reference_id: int, cfg: ReferenceConfig) -> dict:
    y_radius = cfg.radius if cfg.y_radius is None else cfg.y_radius
    omega = 2.0 * np.pi / cfg.period
    peak_speed = cfg.radius * omega if cfg.traj_type == "circle" else np.hypot(cfg.radius, y_radius) * omega
    return {
        "reference_id": int(reference_id),
        "traj_type": cfg.traj_type,
        "center": [float(value) for value in cfg.center],
        "radius": float(cfg.radius),
        "y_radius": float(y_radius),
        "z_amp": float(cfg.z_amp),
        "yaw_ref": float(cfg.yaw_ref),
        "period_sec": float(cfg.period),
        "peak_planar_speed_mps": float(peak_speed),
    }


def _new_manifest(task_specs, seed: int, shard_size_tasks: int) -> dict:
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "dataset_columns": DATASET_COLUMNS,
        "collection": {
            "task_manifest": task_manifest(task_specs, seed),
            "task_contexts": [
                {"task_id": int(task_id), "context": task.context_values()}
                for task_id, task in task_specs
            ],
            "tracking_references": [
                _reference_description(reference_id, cfg)
                for reference_id, cfg in enumerate(REFERENCE_TASKS)
            ],
            "calibration_profiles": list(CALIBRATION_PROFILES),
            "episode_ids": [spec["episode_id"] for spec in episode_specs()],
            "steps_per_episode": STEPS,
            "dt_sec": DT,
            "shard_size_tasks": int(shard_size_tasks),
        },
        "shards": [],
        "episodes": {},
        "summary": {},
    }


def _manifest_static_view(manifest: dict) -> dict:
    return {
        "schema_version": manifest["schema_version"],
        "dataset_columns": manifest["dataset_columns"],
        "collection": manifest["collection"],
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _load_or_initialize_manifest(output_dir: Path, expected_manifest: dict, resume: bool) -> dict:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{output_dir} already contains a collection. Use --resume to continue it or choose a new --output-dir.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_static_view(manifest) != _manifest_static_view(expected_manifest):
            raise ValueError("Existing shard manifest does not match the requested task bank or trajectory protocol; refusing to mix datasets.")
        return manifest
    if resume:
        raise FileNotFoundError(f"Cannot resume because {manifest_path} does not exist.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is non-empty and has no manifest; choose a fresh directory instead of risking an overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return expected_manifest


def _serialize_task_result(task_result: dict) -> dict:
    serialized = {key: value for key, value in task_result.items() if key != "episodes"}
    serialized["episodes"] = [
        {key: value for key, value in episode.items() if key != "rows"}
        for episode in task_result["episodes"]
    ]
    return serialized


def _validate_committed_shard(status: dict, data_path: Path, seen_episode_keys: set[str]) -> None:
    """Verify the atomic CSV matches its status before accepting it on resume."""
    frame = pd.read_csv(data_path, usecols=["task_id", "episode_id", "reference_id", "time"])
    if len(frame) != int(status["rows"]):
        raise RuntimeError(f"Shard {status['name']} row count mismatch: CSV={len(frame)}, status={status['rows']}")
    expected = {}
    for task_result in status["task_results"]:
        for episode in task_result["episodes"]:
            if episode.get("status") != "complete":
                continue
            if int(episode.get("row_count", 0)) != STEPS:
                raise RuntimeError(f"Shard {status['name']} marks an incomplete episode complete")
            episode_key = _episode_key(episode["task_id"], episode["episode_id"])
            if episode_key in expected or episode_key in seen_episode_keys:
                raise RuntimeError(f"Duplicate completed episode {episode_key} in shard set")
            expected[(int(episode["task_id"]), int(episode["episode_id"]), int(episode["reference_id"]))] = episode_key
    groups = {
        (int(task_id), int(episode_id), int(reference_id)): group.sort_values("time")
        for (task_id, episode_id, reference_id), group in frame.groupby(["task_id", "episode_id", "reference_id"], sort=False)
    }
    if set(groups) != set(expected):
        raise RuntimeError(f"Shard {status['name']} episode identifiers do not match its status record")
    expected_times = np.arange(STEPS, dtype=float) * DT
    for group_key, episode_frame in groups.items():
        if len(episode_frame) != STEPS or not np.allclose(episode_frame["time"].to_numpy(dtype=float), expected_times, atol=1e-9, rtol=0.0):
            raise RuntimeError(f"Shard {status['name']} contains a truncated or malformed episode {group_key}")
    seen_episode_keys.update(expected.values())


def _read_committed_shards(output_dir: Path) -> list[dict]:
    statuses = []
    seen_episode_keys = set()
    for status_path in sorted(output_dir.glob("shard-*/status.json")):
        data_path = status_path.parent / "data.csv"
        if not data_path.exists():
            raise RuntimeError(f"Committed shard {status_path.parent} has no data.csv; refusing an unsafe resume.")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported shard schema in {status_path}")
        _validate_committed_shard(status, data_path, seen_episode_keys)
        statuses.append(status)
    return statuses


def _update_manifest_summary(manifest: dict) -> None:
    task_ids = [int(item["task_id"]) for item in manifest["collection"]["task_contexts"]]
    episode_ids = [int(item) for item in manifest["collection"]["episode_ids"]]
    complete_keys = {
        key for key, record in manifest["episodes"].items()
        if record.get("status") == "complete" and int(record.get("row_count", 0)) == STEPS
    }
    complete_task_ids = [
        task_id for task_id in task_ids
        if all(_episode_key(task_id, episode_id) in complete_keys for episode_id in episode_ids)
    ]
    manifest["summary"] = {
        "rows_committed": int(sum(int(shard["rows"]) for shard in manifest["shards"])),
        "expected_task_count": len(task_ids),
        "complete_task_count": len(complete_task_ids),
        "incomplete_task_ids": [task_id for task_id in task_ids if task_id not in complete_task_ids],
        "expected_episode_count": len(task_ids) * len(episode_ids),
        "complete_episode_count": len(complete_keys),
        "failed_episode_count": sum(record.get("status") == "failed" for record in manifest["episodes"].values()),
        "updated_at": _utc_now(),
    }


def _recover_manifest_progress(manifest: dict, output_dir: Path) -> None:
    """Rebuild progress from committed shard directories after an interruption.

    A shard directory is atomically renamed only after both CSV and status are
    written.  Scanning it closes the small crash window between that rename and
    the following global-manifest update.
    """
    prior_records = manifest.get("episodes", {})
    manifest["shards"] = []
    manifest["episodes"] = {}
    for status in _read_committed_shards(output_dir):
        task_results = status["task_results"]
        manifest["shards"].append({
            "name": status["name"],
            "task_ids": [int(item["task_id"]) for item in task_results],
            "rows": int(status["rows"]),
            "created_at": status["created_at"],
        })
        for task_result in task_results:
            for episode in task_result["episodes"]:
                key = _episode_key(episode["task_id"], episode["episode_id"])
                record = dict(episode)
                if "attempts" not in record:
                    record["attempts"] = int(prior_records.get(key, {}).get("attempts", 0)) + 1
                manifest["episodes"][key] = record
    _update_manifest_summary(manifest)


def _next_shard_index(manifest: dict) -> int:
    indices = [int(shard["name"].split("-")[-1]) for shard in manifest["shards"]]
    return max(indices, default=-1) + 1


def _commit_shard(output_dir: Path, shard_index: int, task_results: list[dict]) -> dict:
    shard_name = f"shard-{shard_index:05d}"
    final_dir = output_dir / shard_name
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard {final_dir}")
    rows = [
        row
        for task_result in task_results
        for episode in task_result["episodes"]
        if episode["status"] == "complete"
        for row in episode["rows"]
    ]
    status = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "name": shard_name,
        "created_at": _utc_now(),
        "rows": len(rows),
        "task_results": [_serialize_task_result(result) for result in task_results],
    }
    temporary_dir = output_dir / f".{shard_name}.{os.getpid()}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows, columns=DATASET_COLUMNS).to_csv(temporary_dir / "data.csv", index=False)
    (temporary_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_dir.replace(final_dir)
    return status


def _integrate_committed_shard(manifest: dict, status: dict) -> None:
    task_results = status["task_results"]
    manifest["shards"].append({
        "name": status["name"],
        "task_ids": [int(item["task_id"]) for item in task_results],
        "rows": int(status["rows"]),
        "created_at": status["created_at"],
    })
    for task_result in task_results:
        for episode in task_result["episodes"]:
            manifest["episodes"][_episode_key(episode["task_id"], episode["episode_id"])] = episode
    _update_manifest_summary(manifest)


def _completed_episode_keys(manifest: dict) -> set[str]:
    return {
        key for key, record in manifest["episodes"].items()
        if record.get("status") == "complete" and int(record.get("row_count", 0)) == STEPS
    }


def _iter_round_results(round_specs, workers: int, task_timeout_sec: float):
    if workers == 1:
        for task_spec in round_specs:
            try:
                yield collect_task_rollouts(task_spec)
            except Exception as exc:
                yield _failed_task_result(task_spec, exc)
        return
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    pending_specs = iter(round_specs)
    futures = {}

    def submit_next() -> bool:
        try:
            task_spec = next(pending_specs)
        except StopIteration:
            return False
        futures[executor.submit(collect_task_rollouts, task_spec)] = task_spec
        return True

    for _ in range(min(workers, len(round_specs))):
        submit_next()
    try:
        while futures:
            done, _ = wait(futures, timeout=task_timeout_sec, return_when=FIRST_COMPLETED)
            if not done:
                submitted_task_ids = [int(task_spec[0]) for task_spec in futures.values()]
                raise TimeoutError(f"No task completed within {task_timeout_sec:.0f} s; active task ids={submitted_task_ids}")
            for future in done:
                task_spec = futures.pop(future)
                try:
                    yield future.result()
                except Exception as exc:
                    yield _failed_task_result(task_spec, exc)
                submit_next()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=True)


def _stamp_attempts(task_result: dict, attempts: dict[str, int]) -> None:
    for episode in task_result["episodes"]:
        key = _episode_key(episode["task_id"], episode["episode_id"])
        attempts[key] = attempts.get(key, 0) + 1
        episode["attempts"] = attempts[key]


def parse_args():
    default_workers = max(1, min(12, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Collect body-rate/thrust context-meta residual data in atomic task shards.")
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--max-tasks", type=int, default=512, help="Number of uniformly sampled unique dynamics tasks. Use 0 for all.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SHARD_DIR, help="Directory containing manifest.json and shard-*/data.csv.")
    parser.add_argument("--shard-size-tasks", type=int, default=8, help="Number of returned task results to atomically commit per shard.")
    parser.add_argument("--resume", action="store_true", help="Resume only episodes not recorded complete in the matching manifest.")
    parser.add_argument("--max-episode-attempts", type=int, default=3, help="New collection attempts per unfinished episode before reporting an incomplete dataset.")
    parser.add_argument("--task-timeout-sec", type=float, default=DEFAULT_TASK_TIMEOUT_SEC, help="Fail safely after this long without any active worker completing a task.")
    parser.add_argument("--save-path", type=Path, default=None, help="Deprecated alias; a .csv value is mapped to a sibling *_shards directory and is never overwritten.")
    parser.add_argument("--seed", type=int, default=43, help="Seed for unique task-bank sampling.")
    parser.add_argument("--dry-run", action="store_true", help="Print sampled-task coverage without simulating or writing a dataset.")
    return parser.parse_args()


def _resolve_output_dir(args) -> Path:
    if args.save_path is None:
        return args.output_dir
    inferred_dir = args.save_path if not args.save_path.suffix else args.save_path.parent / f"{args.save_path.stem}_shards"
    if args.output_dir != DEFAULT_SHARD_DIR:
        raise ValueError("Use either --output-dir or deprecated --save-path, not both.")
    print(f"[deprecated] --save-path maps to shard directory {inferred_dir}; no monolithic CSV will be written.")
    return inferred_dir


def main(workers: int, max_tasks: int | None, output_dir: Path, seed: int = 43, dry_run: bool = False, shard_size_tasks: int = 8, resume: bool = False, max_episode_attempts: int = 3, task_timeout_sec: float = DEFAULT_TASK_TIMEOUT_SEC):
    if workers < 1:
        raise ValueError("--workers must be positive")
    if shard_size_tasks < 1:
        raise ValueError("--shard-size-tasks must be positive")
    if max_episode_attempts < 1:
        raise ValueError("--max-episode-attempts must be positive")
    if task_timeout_sec <= 0:
        raise ValueError("--task-timeout-sec must be positive")
    if max_tasks == 0:
        max_tasks = None
    task_specs = list(iter_task_specs(max_tasks=max_tasks, seed=seed))
    expected_manifest = _new_manifest(task_specs, seed, shard_size_tasks)
    print(json.dumps(expected_manifest["collection"]["task_manifest"], indent=2, sort_keys=True))
    if dry_run:
        return
    output_dir = Path(output_dir)
    manifest = _load_or_initialize_manifest(output_dir, expected_manifest, resume)
    _recover_manifest_progress(manifest, output_dir)
    _atomic_write_json(output_dir / "manifest.json", manifest)

    episode_ids = tuple(int(spec["episode_id"]) for spec in episode_specs())
    complete_keys = _completed_episode_keys(manifest)
    pending_by_task = {
        int(task_id): {episode_id for episode_id in episode_ids if _episode_key(task_id, episode_id) not in complete_keys}
        for task_id, _ in task_specs
    }
    attempts = {key: int(record.get("attempts", 0)) for key, record in manifest["episodes"].items()}
    shard_buffer = []
    next_shard_index = _next_shard_index(manifest)

    def flush_shard_buffer() -> None:
        nonlocal shard_buffer, next_shard_index
        if not shard_buffer:
            return
        status = _commit_shard(output_dir, next_shard_index, shard_buffer)
        shard_buffer = []
        next_shard_index += 1
        _integrate_committed_shard(manifest, status)
        _atomic_write_json(output_dir / "manifest.json", manifest)
        print(f"[commit] {status['name']}: tasks={len(status['task_results'])} rows={status['rows']} complete_tasks={manifest['summary']['complete_task_count']}/{manifest['summary']['expected_task_count']}", flush=True)

    try:
        for round_index in range(1, max_episode_attempts + 1):
            round_specs = [
                (task_id, task, tuple(sorted(pending_by_task[int(task_id)])))
                for task_id, task in task_specs
                if pending_by_task[int(task_id)]
            ]
            if not round_specs:
                break
            print(f"Round {round_index}/{max_episode_attempts}: collecting {len(round_specs)} task(s) with {workers} worker(s).", flush=True)
            for result_index, task_result in enumerate(_iter_round_results(round_specs, workers, task_timeout_sec), start=1):
                _stamp_attempts(task_result, attempts)
                task_id = int(task_result["task_id"])
                completed_ids = {
                    int(episode["episode_id"]) for episode in task_result["episodes"]
                    if episode["status"] == "complete" and int(episode["row_count"]) == STEPS
                }
                pending_by_task[task_id].difference_update(completed_ids)
                failed_ids = [
                    int(episode["episode_id"]) for episode in task_result["episodes"]
                    if episode["status"] != "complete"
                ]
                rows = sum(int(episode["row_count"]) for episode in task_result["episodes"] if episode["status"] == "complete")
                print(f"[round {round_index} {result_index:04d}/{len(round_specs):04d}] task={task_id} rows={rows} complete={sorted(completed_ids)} failed={failed_ids}", flush=True)
                shard_buffer.append(task_result)
                if len(shard_buffer) >= shard_size_tasks:
                    flush_shard_buffer()
            flush_shard_buffer()
    except KeyboardInterrupt:
        print("Interrupted: all previously committed shards are safe; rerun with --resume to collect only unfinished episodes.", flush=True)
        raise
    finally:
        flush_shard_buffer()

    incomplete = {task_id: sorted(episode_ids_left) for task_id, episode_ids_left in pending_by_task.items() if episode_ids_left}
    if incomplete:
        print(f"Collection incomplete for {len(incomplete)} task(s). Their completed episodes are committed; rerun with --resume. Examples: {list(incomplete.items())[:5]}", flush=True)
        raise RuntimeError("Collection is incomplete; refusing to present partial shards as a training dataset.")
    print(f"Collection complete: {manifest['summary']['complete_task_count']} tasks, {manifest['summary']['rows_committed']} rows, {len(manifest['shards'])} atomic shard(s) in {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.workers,
        args.max_tasks,
        _resolve_output_dir(args),
        args.seed,
        args.dry_run,
        args.shard_size_tasks,
        args.resume,
        args.max_episode_attempts,
        args.task_timeout_sec,
    )
