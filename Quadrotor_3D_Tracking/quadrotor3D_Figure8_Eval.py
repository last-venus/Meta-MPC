from quadrotor3D_common import (
    DEFAULT_EXPORT_ANIMATION,
    DEFAULT_GUI,
    DEFAULT_SAVE_RESULTS,
    DEFAULT_SCRIPT_SEED,
    DEFAULT_SHOW_PLOT_WINDOW,
    ReferenceConfig,
    run_tracking,
)


METHODS = ("nominal", "lightmlp", "meta")
SEED = DEFAULT_SCRIPT_SEED
GUI = DEFAULT_GUI
SAVE_RESULTS = DEFAULT_SAVE_RESULTS
SHOW_PLOT_WINDOW = DEFAULT_SHOW_PLOT_WINDOW
EXPORT_ANIMATION = DEFAULT_EXPORT_ANIMATION
RESULTS_BASENAME_PREFIX = "figure8"

FIGURE8_REFERENCE = ReferenceConfig(
    period=12.0,
    radius=1.6,
    y_radius=1.45,
    center=(0.0, 0.0, 1.0),
    z_amp=0.0,
    yaw_ref=0.0,
    traj_type="figure8",
)


if __name__ == "__main__":
    for method in METHODS:
        run_tracking(
            method=method,
            seed=SEED,
            gui=GUI,
            save_flag=SAVE_RESULTS,
            show_plot_window=SHOW_PLOT_WINDOW,
            export_animation_flag=EXPORT_ANIMATION,
            reference_cfg=FIGURE8_REFERENCE,
            results_basename=f"{RESULTS_BASENAME_PREFIX}_{method}",
        )
