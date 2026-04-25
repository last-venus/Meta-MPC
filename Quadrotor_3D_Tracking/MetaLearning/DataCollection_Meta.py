"""为 3D 四旋翼 latent-context meta-learning 收集离线残差数据。"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import casadi as cs
import numpy as np
import pandas as pd

TRACKING_DIR = Path(__file__).resolve().parents[1]
if str(TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKING_DIR))

from quadrotor3D_common import (  # noqa: E402
    BASE_PARAMS,
    DEFAULT_CTRL_FREQ,
    DEFAULT_META_DATASET_DIR,
    DEFAULT_META_DATASET_PATH,
    DEFAULT_N_HORIZON,
    DEFAULT_SIM_TIME,
    DEFAULT_T_HORIZON,
    MPC,
    NOMINAL_RATIOS,
    Quadrotor3DNominalDynamics,
    ReferenceConfig,
    STATE_COLS,
    TRACKED_DERIVATIVE_INDICES,
    control_labels,
    input_reference,
    make_env_config,
    reference_state,
)
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor  # noqa: E402


np.random.seed(43)

# User-editable collection config.
SAVE_DIR = DEFAULT_META_DATASET_DIR
SAVE_PATH = DEFAULT_META_DATASET_PATH
MASS_RATIOS = np.array([0.75, 1.0, 1.25, 1.5, 1.75])
IXX_RATIOS = np.array([0.8, 1.0, 1.25])
IYY_RATIOS = np.array([0.8, 1.0, 1.25])
IZZ_RATIOS = np.array([0.8, 1.0, 1.25])
DT = 1.0 / DEFAULT_CTRL_FREQ
T = DEFAULT_SIM_TIME
STEPS = int(T / DT)
N = DEFAULT_N_HORIZON
T_HORIZON = DEFAULT_T_HORIZON
EPISODES_PER_REFERENCE = 1
COLLECTION_GUI = False
DONE_ON_OUT_OF_BOUND = False
TASK_SEED_STRIDE = 100

# Derived constants. Normally you only need to edit the block above.
BASE_MASS = BASE_PARAMS["M"] * NOMINAL_RATIOS["M"]
BASE_IXX = BASE_PARAMS["Ixx"] * NOMINAL_RATIOS["Ixx"]
BASE_IYY = BASE_PARAMS["Iyy"] * NOMINAL_RATIOS["Iyy"]
BASE_IZZ = BASE_PARAMS["Izz"] * NOMINAL_RATIOS["Izz"]
ACTION_COLS = control_labels()
DERIVATIVE_STATE_INDICES = TRACKED_DERIVATIVE_INDICES
DATASET_COLUMNS = [
    "task_id", "episode_id", "reference_id", "time", "mass_ratio", "ixx_ratio", "iyy_ratio", "izz_ratio",
    "center_x", "center_y", "center_z", "radius", "z_amp", "yaw_ref", "period", "traj_type", "y_radius",
    *STATE_COLS,
    *ACTION_COLS,
    "x_ddot_true", "y_ddot_true", "z_ddot_true", "p_dot_true", "q_dot_true", "r_dot_true",
    "x_ddot_nom", "y_ddot_nom", "z_ddot_nom", "p_dot_nom", "q_dot_nom", "r_dot_nom",
    "res_x_ddot", "res_y_ddot", "res_z_ddot", "res_p_dot", "res_q_dot", "res_r_dot",
    "mass", "ixx", "iyy", "izz",
]


def circle_reference_for_speed(radius: float, speed_mps: float, center=(0.0, 0.0, 1.0), yaw_ref: float = 0.0) -> ReferenceConfig:
    return ReferenceConfig(
        period=2.0 * np.pi * radius / speed_mps,
        radius=radius,
        center=center,
        z_amp=0.0,
        yaw_ref=yaw_ref,
        traj_type="circle",
    )


REFERENCE_TASKS = [
    ReferenceConfig(period=12.0, radius=0.60, center=(0.0, 0.0, 1.0), z_amp=0.15, yaw_ref=0.0),
    ReferenceConfig(period=10.0, radius=0.60, center=(0.0, 0.0, 1.0), z_amp=0.18, yaw_ref=0.0),
    ReferenceConfig(period=12.0, radius=0.60, y_radius=0.45, center=(0.0, 0.0, 1.0), z_amp=0.15, yaw_ref=0.0, traj_type="figure8"),
    ReferenceConfig(period=10.0, radius=0.52, y_radius=0.40, center=(0.2, -0.15, 1.05), z_amp=0.18, yaw_ref=0.1, traj_type="figure8"),
    circle_reference_for_speed(radius=3.0, speed_mps=2.0),
    circle_reference_for_speed(radius=3.0, speed_mps=3.0),
    circle_reference_for_speed(radius=3.0, speed_mps=4.0),
    circle_reference_for_speed(radius=3.0, speed_mps=5.0),
]


def zero_init_randomization() -> dict:
    return {
        "init_x": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_x_dot": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_y": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_y_dot": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_z": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_z_dot": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_phi": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_theta": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_psi": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_p": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_q": {"distrib": "uniform", "low": 0.0, "high": 0.0},
        "init_r": {"distrib": "uniform", "low": 0.0, "high": 0.0},
    }


def iter_task_specs():
    task_id = 0
    for mass_ratio in MASS_RATIOS:
        for ixx_ratio in IXX_RATIOS:
            for iyy_ratio in IYY_RATIOS:
                for izz_ratio in IZZ_RATIOS:
                    yield (
                        task_id,
                        mass_ratio,
                        ixx_ratio,
                        iyy_ratio,
                        izz_ratio,
                        [BASE_MASS * mass_ratio, BASE_IXX * ixx_ratio, BASE_IYY * iyy_ratio, BASE_IZZ * izz_ratio],
                    )
                    task_id += 1


def set_solver_reference_trajectory(solver, current_time, ref_cfg, u_ref):
    for k in range(N):
        x_ref_k = reference_state(current_time + k * T_HORIZON / N, ref_cfg)
        solver.set(k, "yref", np.concatenate([x_ref_k, u_ref]))
    solver.set(N, "yref", reference_state(current_time + T_HORIZON, ref_cfg))


def collect_episode(task_id, episode_id, reference_id, inertial_prop, ref_cfg):
    """为单个任务采集一整条 rollout 的监督数据。"""
    env = Quadrotor(
        **make_env_config(
            seed=task_id * TASK_SEED_STRIDE + episode_id,
            gui=COLLECTION_GUI,
            done_on_out_of_bound=DONE_ON_OUT_OF_BOUND,
            episode_len_sec=T,
            inertial_prop=inertial_prop,
            init_state=reference_state(0.0, ref_cfg),
            init_state_randomization_info=zero_init_randomization(),
        )
    )
    rows = []
    try:
        model = Quadrotor3DNominalDynamics(env).model()
        nominal_fn = cs.Function("f_nom", [model.x, model.u], [model.f_nominal])
        solver = MPC(model=model, n_horizon=N, t_horizon=T_HORIZON).solver
        u_ref = input_reference()

        obs, _ = env.reset()
        current_state = np.array(obs[:12], dtype=float)
        y_radius = ref_cfg.radius if ref_cfg.y_radius is None else ref_cfg.y_radius

        for step in range(STEPS):
            set_solver_reference_trajectory(solver, step * DT, ref_cfg, u_ref)
            solver.set(0, "lbx", current_state)
            solver.set(0, "ubx", current_state)
            status = solver.solve()
            if status != 0:
                raise RuntimeError(f"MPC solve failed at task={task_id}, episode={episode_id}, step={step}, status={status}")

            control_input = np.array(solver.get(0, "u"), dtype=float)
            next_obs, _, _, _ = env.step(control_input)
            next_state = np.array(next_obs[:12], dtype=float)
            true_targets = (next_state[DERIVATIVE_STATE_INDICES] - current_state[DERIVATIVE_STATE_INDICES]) / DT
            nominal_targets = nominal_fn(current_state, control_input).full().flatten()[DERIVATIVE_STATE_INDICES]
            residual_targets = true_targets - nominal_targets

            rows.append([
                task_id, episode_id, reference_id, step * DT,
                inertial_prop[0] / BASE_MASS, inertial_prop[1] / BASE_IXX, inertial_prop[2] / BASE_IYY, inertial_prop[3] / BASE_IZZ,
                ref_cfg.center[0], ref_cfg.center[1], ref_cfg.center[2], ref_cfg.radius, ref_cfg.z_amp, ref_cfg.yaw_ref, ref_cfg.period,
                ref_cfg.traj_type, y_radius,
                *current_state, *control_input, *true_targets, *nominal_targets, *residual_targets,
                env.MASS, env.J[0, 0], env.J[1, 1], env.J[2, 2],
            ])
            current_state = next_state
    finally:
        env.close()

    return rows


def collect_task_rollouts(task_spec):
    task_id, mass_ratio, ixx_ratio, iyy_ratio, izz_ratio, inertial_prop = task_spec
    rows, episode_id = [], 0
    for reference_id, ref_cfg in enumerate(REFERENCE_TASKS):
        for _ in range(EPISODES_PER_REFERENCE):
            try:
                rows.extend(collect_episode(task_id, episode_id, reference_id, inertial_prop, ref_cfg))
            except RuntimeError as exc:
                print(
                    f"[warn] skip rollout task={task_id} episode={episode_id} ref={reference_id} "
                    f"traj={ref_cfg.traj_type} radius={ref_cfg.radius:.2f} period={ref_cfg.period:.3f}: {exc}",
                    flush=True,
                )
            episode_id += 1
    return task_id, mass_ratio, ixx_ratio, iyy_ratio, izz_ratio, rows


def parse_args():
    default_workers = max(1, min(16, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Collect 3D quadrotor meta-learning data.")
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Number of worker processes for task-level parallel collection (default: {default_workers}).")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap on the number of tasks to collect, useful for quick debugging.")
    return parser.parse_args()


def main(workers: int, max_tasks: int | None = None):
    if workers < 1:
        raise ValueError("--workers must be at least 1.")

    task_specs = list(iter_task_specs())
    if max_tasks is not None:
        task_specs = task_specs[:max_tasks]

    print(f"Collecting {len(task_specs)} tasks with {workers} worker(s).")
    all_rows = []
    if workers == 1:
        task_iter = (collect_task_rollouts(task_spec) for task_spec in task_specs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        task_iter = executor.map(collect_task_rollouts, task_specs)

    try:
        for idx, (_, mass_ratio, ixx_ratio, iyy_ratio, izz_ratio, rows) in enumerate(task_iter, start=1):
            print(
                f"[{idx:04d}/{len(task_specs):04d}] M={mass_ratio:.2f}, Ixx={ixx_ratio:.2f}, "
                f"Iyy={iyy_ratio:.2f}, Izz={izz_ratio:.2f}, rollouts={len(REFERENCE_TASKS) * EPISODES_PER_REFERENCE}"
            )
            all_rows.extend(rows)
    finally:
        if executor is not None:
            executor.shutdown()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows, columns=DATASET_COLUMNS).to_csv(SAVE_PATH, index=False)
    print(f"Saved meta-dataset to {SAVE_PATH}")


if __name__ == "__main__":
    args = parse_args()
    main(workers=args.workers, max_tasks=args.max_tasks)
