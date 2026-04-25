from quadrotor3D_common import (
    DEFAULT_EXPORT_ANIMATION,
    DEFAULT_GUI,
    DEFAULT_SAVE_RESULTS,
    DEFAULT_SHOW_PLOT_WINDOW,
    run_tracking,
)


METHOD = "nominal"
FIRST_SEED = 31
LAST_SEED_EXCLUSIVE = 51
SEEDS = range(FIRST_SEED, LAST_SEED_EXCLUSIVE)
GUI = DEFAULT_GUI
SAVE_RESULTS = DEFAULT_SAVE_RESULTS
SHOW_PLOT_WINDOW = DEFAULT_SHOW_PLOT_WINDOW
EXPORT_ANIMATION = DEFAULT_EXPORT_ANIMATION


if __name__ == "__main__":
    for seed in SEEDS:
        print(f"\n====== Running {METHOD} 3D tracking with seed {seed} ======")
        run_tracking(
            method=METHOD,
            seed=seed,
            gui=GUI,
            save_flag=SAVE_RESULTS,
            show_plot_window=SHOW_PLOT_WINDOW,
            export_animation_flag=EXPORT_ANIMATION,
        )
