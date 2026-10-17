"""Run the main body-rate/thrust benchmark suite."""

import argparse
from pathlib import Path

from bodyrate_common import DynamicsTask, ReferenceConfig, run_tracking


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "suite"
METHODS = ("nominal", "lightmlp", "meta")
SEEDS = range(31, 32)

TEST_CASES = {
    "in_domain_highspeed_circle": {
        "task": DynamicsTask(
            mass_ratio=1.15, ixx_ratio=1.15, iyy_ratio=0.90, izz_ratio=1.00,
            thrust_scale=0.95, rate_tau_scale=1.15, drag_linear=0.0040,
            drag_quad=0.0012, wind=(0.0025, -0.0015, 0.0),
        ),
        # 7 m radius at 4 m/s: same high-speed operating regime as training.
        "reference": ReferenceConfig(period=10.995574, radius=7.0, center=(0.0, 0.0, 1.15), z_amp=0.05),
    },
    "in_domain_highspeed_figure8": {
        "task": DynamicsTask(
            mass_ratio=0.85, ixx_ratio=1.00, iyy_ratio=1.15, izz_ratio=0.90,
            thrust_scale=1.05, rate_tau_scale=1.00, drag_linear=0.0030,
            drag_quad=0.0007, wind=(-0.0025, 0.0020, 0.0),
        ),
        # Peak planar speed is 4 m/s, with a geometry represented during training.
        "reference": ReferenceConfig(period=7.947671, radius=4.8, y_radius=1.6, center=(0.0, 0.0, 1.15), z_amp=0.08, traj_type="figure8"),
    },
    "mild_extrapolation_fast_figure8": {
        "task": DynamicsTask(
            mass_ratio=1.35, ixx_ratio=1.20, iyy_ratio=0.90, izz_ratio=1.10,
            thrust_scale=0.93, rate_tau_scale=1.20, drag_linear=0.0045,
            drag_quad=0.0014, wind=(0.0030, -0.0020, 0.0),
        ),
        # A small simultaneous extrapolation beyond the training envelope; peak speed 4.2 m/s.
        "reference": ReferenceConfig(period=8.514779, radius=5.4, y_radius=1.8, center=(0.10, -0.10, 1.15), z_amp=0.08, traj_type="figure8"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run selected BodyRate MPC benchmark cases.")
    parser.add_argument("--cases", nargs="+", default=("in_domain_highspeed_circle", "in_domain_highspeed_figure8", "mild_extrapolation_fast_figure8"), choices=tuple(TEST_CASES), help="Benchmark cases to run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=(31, 32, 33), help="Independent simulation seeds.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main(case_names, seeds, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_results_dir = results_dir / "raw_runs"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    for case_name in case_names:
        spec = TEST_CASES[case_name]
        for method in METHODS:
            for seed in seeds:
                print(f"\n====== {case_name} | {method} | seed={seed} ======")
                run_tracking(
                    method=method,
                    seed=seed,
                    reference_cfg=spec["reference"],
                    task=spec["task"],
                    results_dir=raw_results_dir,
                    results_basename=f"{case_name}_{method}",
                    save_plot=False,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args.cases, args.seeds, args.results_dir)
