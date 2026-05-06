"""Run the main body-rate/thrust benchmark suite."""

from pathlib import Path

from bodyrate_common import DynamicsTask, ReferenceConfig, run_tracking


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "suite"
METHODS = ("nominal", "lightmlp", "meta")
SEEDS = range(31, 32)

TEST_CASES = {
    "interp_circle": {
        "task": DynamicsTask(mass_ratio=1.15, ixx_ratio=1.0, iyy_ratio=1.0, izz_ratio=1.0, thrust_scale=1.0, rate_tau_scale=1.0),
        "reference": ReferenceConfig(period=12.0, radius=0.8, center=(0.0, 0.0, 1.0), z_amp=0.15),
    },
    "extrap_heavy_fast": {
        "task": DynamicsTask(mass_ratio=1.55, ixx_ratio=1.35, iyy_ratio=1.25, izz_ratio=1.30, thrust_scale=0.90, rate_tau_scale=1.35),
        "reference": ReferenceConfig(period=7.5, radius=1.10, center=(0.0, 0.0, 1.0), z_amp=0.10),
    },
    "unseen_figure8": {
        "task": DynamicsTask(mass_ratio=1.25, ixx_ratio=0.9, iyy_ratio=1.15, izz_ratio=1.1, thrust_scale=1.05, rate_tau_scale=0.9),
        "reference": ReferenceConfig(period=7.5, radius=1.15, y_radius=0.9, center=(0.15, -0.1, 1.05), z_amp=0.12, traj_type="figure8"),
    },
    "compound_disturbance": {
        "task": DynamicsTask(
            mass_ratio=1.35,
            ixx_ratio=1.15,
            iyy_ratio=0.9,
            izz_ratio=1.15,
            thrust_scale=0.92,
            rate_tau_scale=1.30,
            drag_linear=0.004,
            drag_quad=0.0015,
            wind=(0.006, -0.004, 0.0),
        ),
        "reference": ReferenceConfig(period=7.5, radius=1.10, y_radius=0.85, center=(0.0, 0.0, 1.0), z_amp=0.12, traj_type="figure8"),
    },
    "hard_fast_compound": {
        "task": DynamicsTask(
            mass_ratio=1.60,
            ixx_ratio=1.35,
            iyy_ratio=0.85,
            izz_ratio=1.30,
            thrust_scale=0.88,
            rate_tau_scale=1.45,
            drag_linear=0.006,
            drag_quad=0.002,
            wind=(0.008, -0.006, 0.0),
        ),
        "reference": ReferenceConfig(period=6.5, radius=1.25, y_radius=0.95, center=(0.0, 0.0, 1.05), z_amp=0.15, traj_type="figure8"),
    },
}


if __name__ == "__main__":
    for case_name, spec in TEST_CASES.items():
        for method in METHODS:
            for seed in SEEDS:
                print(f"\n====== {case_name} | {method} | seed={seed} ======")
                run_tracking(
                    method=method,
                    seed=seed,
                    reference_cfg=spec["reference"],
                    task=spec["task"],
                    results_dir=RESULTS_DIR,
                    results_basename=f"{case_name}_{method}",
                )
