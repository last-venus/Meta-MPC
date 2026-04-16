from quadrotor3D_common import ReferenceConfig, run_tracking


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
    for method in ("nominal", "lightmlp", "meta"):
        run_tracking(
            method=method,
            seed=1,
            gui=False,
            save_flag=True,
            show_plot_window=False,
            export_animation_flag=False,
            reference_cfg=FIGURE8_REFERENCE,
            results_basename=f"figure8_{method}",
        )
