import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def position_rmse(df):
    err = df[["x", "y", "z"]].to_numpy() - df[["x_ref", "y_ref", "z_ref"]].to_numpy()
    return float(np.sqrt(np.mean(err**2)))


def max_position_error(df):
    err = df[["x", "y", "z"]].to_numpy() - df[["x_ref", "y_ref", "z_ref"]].to_numpy()
    return float(np.max(np.linalg.norm(err, axis=1)))


def control_energy(df):
    cols = ["p_cmd", "q_cmd", "r_cmd", "thrust_cmd"]
    if not all(col in df.columns for col in cols):
        return np.nan
    u = df[cols].to_numpy()
    return float(np.mean(np.sum(u**2, axis=1)))


def position_rmse_window(df, start_sec: float, end_sec: float | None):
    mask = df["time"] >= start_sec
    if end_sec is not None:
        mask &= df["time"] < end_sec
    window = df[mask]
    if window.empty:
        return np.nan
    return position_rmse(window)


def case_from_stem(stem: str):
    for suffix in ("_nominal", "_lightmlp", "_meta"):
        marker = f"{suffix}_seed"
        if marker in stem:
            return stem.split(marker)[0]
    return "unknown"


def summarize(result_dir: Path, warmup_sec: float):
    rows = []
    for csv_path in sorted(result_dir.glob("*.csv")):
        if csv_path.name in {"per_run_summary.csv", "method_summary.csv"}:
            continue
        df = pd.read_csv(csv_path)
        required = {"time", "x", "y", "z", "x_ref", "y_ref", "z_ref"}
        if df.empty or not required.issubset(df.columns):
            continue
        warm_df = df[df["time"] >= warmup_sec]
        name = csv_path.stem
        method = str(df["method"].iloc[0]) if "method" in df.columns else "unknown"
        rows.append(
            {
                "file": name,
                "case": case_from_stem(name),
                "method": method,
                "seed": int(df["seed"].iloc[0]) if "seed" in df.columns else -1,
                "full_rmse": position_rmse(df),
                "rmse_0_0p5": position_rmse_window(df, 0.0, 0.5),
                "rmse_0p5_1": position_rmse_window(df, 0.5, 1.0),
                "rmse_1_2": position_rmse_window(df, 1.0, 2.0),
                "post_warmup_rmse": position_rmse(warm_df) if len(warm_df) else np.nan,
                "max_position_error": max_position_error(df),
                "control_energy": control_energy(df),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, summary
    grouped = summary.groupby("method", as_index=False).agg(
        full_rmse_mean=("full_rmse", "mean"),
        full_rmse_std=("full_rmse", "std"),
        rmse_0_0p5_mean=("rmse_0_0p5", "mean"),
        rmse_0p5_1_mean=("rmse_0p5_1", "mean"),
        rmse_1_2_mean=("rmse_1_2", "mean"),
        post_warmup_rmse_mean=("post_warmup_rmse", "mean"),
        post_warmup_rmse_std=("post_warmup_rmse", "std"),
        max_error_mean=("max_position_error", "mean"),
        control_energy_mean=("control_energy", "mean"),
        n=("file", "count"),
    )
    return summary, grouped


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize body-rate/thrust experiment CSVs.")
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "suite")
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    per_run, grouped = summarize(args.result_dir, args.warmup_sec)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.result_dir / "per_run_summary.csv", index=False)
    grouped.to_csv(args.result_dir / "method_summary.csv", index=False)
    print(grouped.to_string(index=False) if not grouped.empty else "No CSV files found.")
