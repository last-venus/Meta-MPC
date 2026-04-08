from quadrotor3D_common import run_tracking


if __name__ == "__main__":
    for seed in range(31, 51):
        print(f"\n====== Running meta 3D tracking with seed {seed} ======")
        run_tracking(method="meta", seed=seed, gui=False, save_flag=True, show_plot_window=False, export_animation_flag=False)
