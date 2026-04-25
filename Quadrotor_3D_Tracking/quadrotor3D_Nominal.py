from quadrotor3D_common import (
    DEFAULT_EXPORT_ANIMATION,
    DEFAULT_GUI,
    DEFAULT_SAVE_RESULTS,
    DEFAULT_SCRIPT_SEED,
    DEFAULT_SHOW_PLOT_WINDOW,
    run_tracking,
)


METHOD = "nominal"
SEED = DEFAULT_SCRIPT_SEED
GUI = DEFAULT_GUI
SAVE_RESULTS = DEFAULT_SAVE_RESULTS
SHOW_PLOT_WINDOW = DEFAULT_SHOW_PLOT_WINDOW
EXPORT_ANIMATION = DEFAULT_EXPORT_ANIMATION


if __name__ == "__main__":
    run_tracking(
        method=METHOD,
        seed=SEED,
        gui=GUI,
        save_flag=SAVE_RESULTS,
        show_plot_window=SHOW_PLOT_WINDOW,
        export_animation_flag=EXPORT_ANIMATION,
    )
