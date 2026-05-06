from bodyrate_common import (
    DEFAULT_EXPORT_ANIMATION,
    DEFAULT_GUI,
    DEFAULT_SAVE_RESULTS,
    DEFAULT_SCRIPT_SEED,
    DEFAULT_SHOW_PLOT_WINDOW,
    run_tracking,
)


if __name__ == "__main__":
    run_tracking(
        method="meta",
        seed=DEFAULT_SCRIPT_SEED,
        gui=DEFAULT_GUI,
        save_flag=DEFAULT_SAVE_RESULTS,
        show_plot_window=DEFAULT_SHOW_PLOT_WINDOW,
        export_animation_flag=DEFAULT_EXPORT_ANIMATION,
    )

