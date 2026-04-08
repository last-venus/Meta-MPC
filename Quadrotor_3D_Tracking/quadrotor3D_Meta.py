from quadrotor3D_common import run_tracking


if __name__ == "__main__":
    run_tracking(method="meta", seed=1, gui=False, save_flag=True, show_plot_window=False, export_animation_flag=False)
