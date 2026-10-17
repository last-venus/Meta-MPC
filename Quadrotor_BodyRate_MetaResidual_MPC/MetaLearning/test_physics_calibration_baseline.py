"""Small-batch diagnostic for calibration-to-query physical residual fitting.

This is intentionally independent of the learned Meta-MPC checkpoint.  It
fits a low-dimensional, interpretable residual model on calibration probes and
evaluates it only on disjoint high-speed tracking samples of held-out tasks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = MODULE_DIR / "meta_dataset_bodyrate" / "bodyrate_meta_highspeed_shards"
TARGET_COLS = ["res_x_ddot", "res_y_ddot", "res_z_ddot", "res_p_dot", "res_q_dot", "res_r_dot"]
RATE_COLS = [("p", "p_cmd"), ("q", "q_cmd"), ("r", "r_cmd")]
NOMINAL_MASS = 0.027 * 0.66


def load_frame(dataset_dir: Path) -> pd.DataFrame:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["summary"]
    if summary["complete_task_count"] != summary["expected_task_count"]:
        raise RuntimeError("Dataset is incomplete; resume collection before this diagnostic.")
    return pd.concat((pd.read_csv(path) for path in sorted(dataset_dir.glob("shard-*/data.csv"))), ignore_index=True)


def thrust_axis_features(frame: pd.DataFrame) -> np.ndarray:
    phi, theta, psi = (frame[name].to_numpy() for name in ("phi", "theta", "psi"))
    thrust = frame["thrust_cmd"].to_numpy()
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    # Nominal thrust acceleration before gravity.  Its shared coefficient is
    # effective thrust-to-mass gain; other columns are wind/drag terms.
    thrust_acc = np.column_stack([
        (s_phi * s_psi + c_phi * s_theta * c_psi) * thrust / NOMINAL_MASS,
        (-s_phi * c_psi + c_phi * s_theta * s_psi) * thrust / NOMINAL_MASS,
        c_phi * c_theta * thrust / NOMINAL_MASS,
    ])
    velocity = frame[["x_dot", "y_dot", "z_dot"]].to_numpy()
    return np.stack(
        [np.column_stack([thrust_acc[:, axis], np.ones(len(frame)), velocity[:, axis], velocity[:, axis] * np.abs(velocity[:, axis])]) for axis in range(3)],
        axis=1,
    )


def fit_task(calibration: pd.DataFrame, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    trans_features = thrust_axis_features(calibration)
    trans_coefficients = []
    for axis, target in enumerate(TARGET_COLS[:3]):
        x = trans_features[:, axis, :]
        y = calibration[target].to_numpy()
        trans_coefficients.append(np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y))
    rate_coefficients = []
    for target, (rate, command) in zip(TARGET_COLS[3:], RATE_COLS):
        x = (calibration[command].to_numpy() - calibration[rate].to_numpy())[:, None]
        y = calibration[target].to_numpy()
        rate_coefficients.append(float(np.linalg.solve(x.T @ x + ridge * np.eye(1), x.T @ y)[0]))
    return np.asarray(trans_coefficients), np.asarray(rate_coefficients)


def predict(frame: pd.DataFrame, trans_coefficients: np.ndarray, rate_coefficients: np.ndarray) -> np.ndarray:
    features = thrust_axis_features(frame)
    translation = np.column_stack([features[:, axis, :] @ trans_coefficients[axis] for axis in range(3)])
    rotation = np.column_stack([
        rate_coefficients[axis] * (frame[command].to_numpy() - frame[rate].to_numpy())
        for axis, (rate, command) in enumerate(RATE_COLS)
    ])
    return np.column_stack([translation, rotation])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tasks", type=int, default=12, help="Small, deterministic task batch.")
    parser.add_argument("--support-samples", type=int, default=192)
    parser.add_argument("--query-samples", type=int, default=384)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    frame = load_frame(args.dataset)
    task_ids = sorted(frame["task_id"].unique())[: args.tasks]
    nominal_errors, calibrated_errors = [], []
    for task_id in task_ids:
        task = frame[frame["task_id"] == task_id]
        calibration = task[task["traj_type"] == "calibration_probe"].sort_values("time").head(args.support_samples)
        # References 7/8 are 4 m/s circle/F8 and are absent from the support.
        query = task[task["reference_id"] >= 7].sort_values("time").head(args.query_samples)
        if len(calibration) < args.support_samples or len(query) < args.query_samples:
            raise RuntimeError(f"Task {task_id} lacks the requested support/query samples.")
        trans, rate = fit_task(calibration, args.ridge)
        target = query[TARGET_COLS].to_numpy()
        prediction = predict(query, trans, rate)
        nominal_errors.append(np.mean(target ** 2, axis=0))
        calibrated_errors.append(np.mean((target - prediction) ** 2, axis=0))

    nominal_rmse = np.sqrt(np.mean(nominal_errors, axis=0))
    calibrated_rmse = np.sqrt(np.mean(calibrated_errors, axis=0))
    names = ("x_ddot", "y_ddot", "z_ddot", "p_dot", "q_dot", "r_dot")
    print(f"tasks={len(task_ids)} support={args.support_samples} query={args.query_samples}")
    for name, before, after in zip(names, nominal_rmse, calibrated_rmse):
        print(f"{name}: nominal={before:.5f} calibrated={after:.5f} improvement={100.0 * (1.0 - after / max(before, 1e-12)):.1f}%")
    print(f"mean_relative_improvement={100.0 * (1.0 - np.mean(calibrated_rmse / np.maximum(nominal_rmse, 1e-12))):.1f}%")


if __name__ == "__main__":
    main()
