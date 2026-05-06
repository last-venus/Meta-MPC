"""Collect body-rate/thrust residual data for context-meta MPC."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

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
    reference_state,
    set_tracking_references,
    step_bodyrate_env,
    wrap_state_angles,
)
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor  # noqa: E402


SAVE_DIR = DEFAULT_DATASET_DIR
SAVE_PATH = DEFAULT_DATASET_PATH
DT = 1.0 / DEFAULT_CTRL_FREQ
T = DEFAULT_SIM_TIME
STEPS = int(T / DT)
N = DEFAULT_N_HORIZON
T_HORIZON = DEFAULT_T_HORIZON
COLLECTION_GUI = False
DONE_ON_OUT_OF_BOUND = False
TASK_SEED_STRIDE = 1000

MASS_RATIOS = np.array([0.85, 1.0, 1.15, 1.30])
IXX_RATIOS = np.array([0.9, 1.0, 1.15])
IYY_RATIOS = np.array([0.9, 1.0, 1.15])
IZZ_RATIOS = np.array([0.9, 1.0, 1.15])
THRUST_SCALES = np.array([0.95, 1.0, 1.05])
RATE_TAU_SCALES = np.array([0.9, 1.0, 1.15])
# Keep compound terms mild in training so the model learns their residual
# shape without making the offline MPC data collection brittle.
DRAG_LINEAR_VALUES = np.array([0.0, 0.001, 0.0025, 0.004])
DRAG_QUAD_VALUES = np.array([0.0, 0.0005, 0.001])
WIND_VALUES = [
    (0.0, 0.0, 0.0),
    (0.002, 0.0, 0.0),
    (-0.002, 0.0015, 0.0),
    (0.0, -0.002, 0.0),
    (0.0025, -0.0015, 0.0),
]

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
    "p_dot_true",
    "q_dot_true",
    "r_dot_true",
    "x_ddot_nom",
    "y_ddot_nom",
    "z_ddot_nom",
    "p_dot_nom",
    "q_dot_nom",
    "r_dot_nom",
    "res_x_ddot",
    "res_y_ddot",
    "res_z_ddot",
    "res_p_dot",
    "res_q_dot",
    "res_r_dot",
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


REFERENCE_TASKS = [
    ReferenceConfig(period=12.0, radius=0.60, center=(0.0, 0.0, 1.0), z_amp=0.15, yaw_ref=0.0),
    ReferenceConfig(period=10.0, radius=0.60, center=(0.0, 0.0, 1.0), z_amp=0.18, yaw_ref=0.0),
    ReferenceConfig(period=12.0, radius=0.60, y_radius=0.45, center=(0.0, 0.0, 1.0), z_amp=0.15, yaw_ref=0.0, traj_type="figure8"),
    ReferenceConfig(period=10.0, radius=0.52, y_radius=0.40, center=(0.2, -0.15, 1.05), z_amp=0.18, yaw_ref=0.1, traj_type="figure8"),
    circle_reference_for_speed(radius=3.0, speed_mps=2.0),
    circle_reference_for_speed(radius=3.0, speed_mps=3.0),
]


def iter_task_specs(max_tasks=None):
    rng = np.random.default_rng(43)
    task_id = 0
    base_specs = []
    for mass_ratio in MASS_RATIOS:
        for ixx_ratio in IXX_RATIOS:
            for iyy_ratio in IYY_RATIOS:
                for izz_ratio in IZZ_RATIOS:
                    base_specs.append((mass_ratio, ixx_ratio, iyy_ratio, izz_ratio))
    rng.shuffle(base_specs)
    for mass_ratio, ixx_ratio, iyy_ratio, izz_ratio in base_specs:
        for thrust_scale in THRUST_SCALES:
            for rate_tau_scale in RATE_TAU_SCALES:
                drag_linear = float(rng.choice(DRAG_LINEAR_VALUES))
                drag_quad = float(rng.choice(DRAG_QUAD_VALUES))
                wind = WIND_VALUES[int(rng.integers(0, len(WIND_VALUES)))]
                yield task_id, DynamicsTask(
                    mass_ratio=float(mass_ratio),
                    ixx_ratio=float(ixx_ratio),
                    iyy_ratio=float(iyy_ratio),
                    izz_ratio=float(izz_ratio),
                    thrust_scale=float(thrust_scale),
                    rate_tau_scale=float(rate_tau_scale),
                    drag_linear=drag_linear,
                    drag_quad=drag_quad,
                    wind=wind,
                )
                task_id += 1
                if max_tasks is not None and task_id >= max_tasks:
                    return


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
        model = BodyRateNominalDynamics(env).model()
        nominal_fn = cs.Function("f_bodyrate_nom", [model.x, model.u], [model.f_nominal])
        solver = MPC(model=model, n_horizon=N, t_horizon=T_HORIZON).solver
        u_ref = input_reference(env)
        obs, _ = env.reset()
        current_state = wrap_state_angles(np.array(obs[:12], dtype=float))
        y_radius = ref_cfg.radius if ref_cfg.y_radius is None else ref_cfg.y_radius
        context = task.context_values()

        for step in range(STEPS):
            set_tracking_references(solver, step * DT, N, T_HORIZON, ref_cfg, u_ref, [])
            solver.set(0, "lbx", current_state)
            solver.set(0, "ubx", current_state)
            status = solver.solve()
            if status != 0:
                raise RuntimeError(f"MPC solve failed at task={task_id}, episode={episode_id}, step={step}, status={status}")

            control_input = np.array(solver.get(0, "u"), dtype=float)
            next_obs, _, _, info = step_bodyrate_env(env, current_state, control_input, task)
            next_state = wrap_state_angles(np.array(next_obs[:12], dtype=float))
            true_targets = (next_state[TRACKED_DERIVATIVE_INDICES] - current_state[TRACKED_DERIVATIVE_INDICES]) / DT
            nominal_targets = nominal_fn(current_state, control_input).full().flatten()[TRACKED_DERIVATIVE_INDICES]
            residual_targets = true_targets - nominal_targets

            rows.append([
                task_id,
                episode_id,
                reference_id,
                step * DT,
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


def collect_task_rollouts(task_spec):
    task_id, task = task_spec
    rows = []
    for reference_id, ref_cfg in enumerate(REFERENCE_TASKS):
        try:
            rows.extend(collect_episode(task_id, reference_id, reference_id, task, ref_cfg))
        except RuntimeError as exc:
            print(f"[warn] skip task={task_id} ref={reference_id}: {exc}", flush=True)
    return task_id, task, rows


def parse_args():
    default_workers = max(1, min(12, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Collect body-rate/thrust context-meta residual data.")
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--max-tasks", type=int, default=144, help="Cap tasks for a practical first dataset. Use 0 for all.")
    parser.add_argument("--save-path", type=Path, default=SAVE_PATH)
    return parser.parse_args()


def main(workers: int, max_tasks: int | None, save_path: Path):
    if max_tasks == 0:
        max_tasks = None
    task_specs = list(iter_task_specs(max_tasks=max_tasks))
    print(f"Collecting {len(task_specs)} body-rate dynamics tasks with {workers} worker(s).")
    all_rows = []
    if workers == 1:
        iterator = (collect_task_rollouts(spec) for spec in task_specs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        iterator = executor.map(collect_task_rollouts, task_specs)
    try:
        for idx, (task_id, task, rows) in enumerate(iterator, start=1):
            print(f"[{idx:04d}/{len(task_specs):04d}] task={task_id} ctx={task.context_values()} rows={len(rows)}")
            all_rows.extend(rows)
    finally:
        if executor is not None:
            executor.shutdown()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows, columns=DATASET_COLUMNS).to_csv(save_path, index=False)
    print(f"Saved dataset to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.workers, args.max_tasks, args.save_path)
