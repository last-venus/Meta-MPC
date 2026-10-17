import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    for csv_path in sorted((result_dir / "raw_runs").glob("*.csv")):
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


def plot_benchmark(per_run: pd.DataFrame, result_dir: Path) -> None:
    """Create publication-readable error and adaptation summaries from run CSVs."""
    if per_run.empty:
        return
    cases = list(per_run["case"].drop_duplicates())
    methods = list(per_run["method"].drop_duplicates())
    colors = {"nominal_bodyrate": "#6c757d", "lightmlp_bodyrate": "#e69f00", "metacontext_bodyrate": "#0072b2"}
    labels = {"nominal_bodyrate": "Nominal MPC", "lightmlp_bodyrate": "Online LightMLP", "metacontext_bodyrate": "Context-meta MPC"}
    offsets = np.linspace(-0.25, 0.25, len(methods))
    x = np.arange(len(cases))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for axis, metric, title in zip(axes, ("full_rmse", "post_warmup_rmse"), ("Full trajectory RMSE", "Post-warmup RMSE")):
        for offset, method in zip(offsets, methods):
            values = per_run[per_run["method"] == method].groupby("case")[metric].mean().reindex(cases)
            errors = per_run[per_run["method"] == method].groupby("case")[metric].std().reindex(cases).fillna(0.0)
            axis.bar(x + offset, values, width=0.22, yerr=errors, capsize=3, color=colors.get(method, None), label=labels.get(method, method))
        axis.set_title(title)
        axis.set_ylabel("position RMSE [m]")
        axis.set_xticks(x, cases, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(result_dir / "benchmark_rmse_comparison.png", dpi=300)
    plt.close(fig)

    phase_columns = ("rmse_0_0p5", "rmse_0p5_1", "rmse_1_2", "post_warmup_rmse")
    phase_labels = ("0–0.5 s", "0.5–1 s", "1–2 s", "2 s–end")
    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, case in zip(axes, cases):
        case_data = per_run[per_run["case"] == case]
        for method in methods:
            means = case_data[case_data["method"] == method][list(phase_columns)].mean()
            axis.plot(phase_labels, means.to_numpy(), marker="o", linewidth=2, color=colors.get(method, None), label=labels.get(method, method))
        axis.set_title(case)
        axis.set_ylabel("position RMSE [m]")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(result_dir / "benchmark_adaptation_comparison.png", dpi=300)
    plt.close(fig)


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
    plot_benchmark(per_run, args.result_dir)
    print(grouped.to_string(index=False) if not grouped.empty else "No CSV files found.")
