import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import pybullet as p
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TRACKING_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = TRACKING_DIR.parents[1]
SAFE_CONTROL_GYM_ROOT = WORKSPACE_ROOT / "safe-control-gym"
if str(SAFE_CONTROL_GYM_ROOT) not in sys.path and SAFE_CONTROL_GYM_ROOT.exists():
    sys.path.insert(0, str(SAFE_CONTROL_GYM_ROOT))

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor  # noqa: E402
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType  # noqa: E402


REQUIRED_STATE_COLS = [
    "x",
    "x_dot",
    "y",
    "y_dot",
    "z",
    "z_dot",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
]
REFERENCE_COLS = ["x_ref", "y_ref", "z_ref"]
DEFAULT_FPS = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a 3D quadrotor tracking CSV and export an HD video.")
    parser.add_argument(
        "--csv",
        type=str,
        default="nominal_seed1.csv",
        help="CSV filename in this results folder, or an absolute path.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Replay/export FPS.")
    parser.add_argument("--width", type=int, default=1280, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument("--yaw", type=float, default=35.0, help="Camera yaw in degrees.")
    parser.add_argument("--pitch", type=float, default=-25.0, help="Camera pitch in degrees.")
    parser.add_argument("--distance", type=float, default=None, help="Override camera distance directly.")
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=1.0,
        help="Scale the auto-computed camera distance. Values below 1 zoom in.",
    )
    parser.add_argument("--follow", action="store_true", help="Follow the drone instead of using a fixed camera target.")
    return parser.parse_args()


def resolve_csv_path(csv_arg: str) -> Path:
    csv_path = Path(csv_arg)
    if not csv_path.is_absolute():
        csv_path = SCRIPT_DIR / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path


def get_color_from_name(name: str):
    if name.startswith("nominal"):
        return [0.12, 0.47, 0.71]
    if name.startswith("lightmlp"):
        return [1.00, 0.50, 0.05]
    if name.startswith("metacontext"):
        return [0.84, 0.15, 0.16]
    if name.startswith("metamlp") or name.startswith("meta"):
        return [0.17, 0.63, 0.17]
    return [0.5, 0.5, 0.5]


def validate_columns(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_STATE_COLS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV is missing required columns: {missing_cols}")


def compute_camera_setup(df: pd.DataFrame):
    real_xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    all_xyz = real_xyz
    if all(col in df.columns for col in REFERENCE_COLS):
        ref_xyz = df[REFERENCE_COLS].to_numpy(dtype=float)
        all_xyz = np.vstack([all_xyz, ref_xyz])
    xyz_min = np.min(all_xyz, axis=0)
    xyz_max = np.max(all_xyz, axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    span = np.max(xyz_max - xyz_min)
    distance = max(1.2, 2.2 * span + 0.8)
    return center.tolist(), float(distance)


def create_goal_marker(client: int):
    goal_marker_vis_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1, 0, 0, 1],
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_marker_vis_id,
        basePosition=[0, 0, 0],
        physicsClientId=client,
    )


def rotation_between_vectors(vec_from, vec_to):
    vec_from = np.asarray(vec_from, dtype=float)
    vec_to = np.asarray(vec_to, dtype=float)
    vec_from = vec_from / np.linalg.norm(vec_from)
    vec_to = vec_to / np.linalg.norm(vec_to)
    cross = np.cross(vec_from, vec_to)
    dot = np.dot(vec_from, vec_to)

    if dot < -0.999999:
        axis = np.cross(vec_from, np.array([1.0, 0.0, 0.0], dtype=float))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(vec_from, np.array([0.0, 1.0, 0.0], dtype=float))
        axis = axis / np.linalg.norm(axis)
        return [axis[0], axis[1], axis[2], 0.0]

    quat = np.array([cross[0], cross[1], cross[2], 1.0 + dot], dtype=float)
    quat = quat / np.linalg.norm(quat)
    return quat.tolist()


def create_segment_body(client: int, start_xyz, end_xyz, color, radius=0.01):
    start_xyz = np.asarray(start_xyz, dtype=float)
    end_xyz = np.asarray(end_xyz, dtype=float)
    segment = end_xyz - start_xyz
    length = np.linalg.norm(segment)
    if length < 1e-6:
        return None

    midpoint = 0.5 * (start_xyz + end_xyz)
    orientation = rotation_between_vectors([0.0, 0.0, 1.0], segment)
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_CAPSULE,
        radius=radius,
        length=max(1e-6, length - 2.0 * radius),
        rgbaColor=[color[0], color[1], color[2], 1.0],
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=midpoint.tolist(),
        baseOrientation=orientation,
        physicsClientId=client,
    )


def draw_world_frame(client: int):
    axis_len = 0.8
    create_segment_body(client, [0, 0, 0], [axis_len, 0, 0], [1, 0, 0], radius=0.006)
    create_segment_body(client, [0, 0, 0], [0, axis_len, 0], [0, 1, 0], radius=0.006)
    create_segment_body(client, [0, 0, 0], [0, 0, axis_len], [0, 0, 1], radius=0.006)


def draw_segment(client: int, start_xyz, end_xyz, color, width=3):
    radius = 0.006 if width <= 2 else 0.01
    create_segment_body(client, start_xyz, end_xyz, color, radius=radius)


def build_env(df: pd.DataFrame, fps: int):
    inertial_prop = None
    if all(col in df.columns for col in ["mass", "ixx", "iyy", "izz"]):
        inertial_prop = [
            float(df["mass"].iloc[0]),
            float(df["ixx"].iloc[0]),
            float(df["iyy"].iloc[0]),
            float(df["izz"].iloc[0]),
        ]

    if all(col in df.columns for col in REFERENCE_COLS):
        stabilization_goal = [
            float(df["x_ref"].iloc[0]),
            float(df["y_ref"].iloc[0]),
            float(df["z_ref"].iloc[0]),
        ]
    else:
        stabilization_goal = [
            float(df["x"].iloc[0]),
            float(df["y"].iloc[0]),
            float(df["z"].iloc[0]),
        ]

    env_config = {
        "gui": True,
        "ctrl_freq": fps,
        "pyb_freq": fps,
        "quad_type": QuadType.THREE_D,
        "done_on_out_of_bound": False,
        "episode_len_sec": max(1.0, len(df) / fps),
        "task_info": {
            "stabilization_goal": stabilization_goal,
            "stabilization_goal_tolerance": 0.05,
        },
    }
    if inertial_prop is not None:
        env_config["inertial_prop"] = inertial_prop
    return Quadrotor(**env_config)


def main():
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    csv_name = csv_path.name
    video_dir = SCRIPT_DIR / "replay_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{csv_path.stem}_replay_HD.mp4"

    df = pd.read_csv(csv_path)
    validate_columns(df)

    traj_color = get_color_from_name(csv_name)
    has_reference = all(col in df.columns for col in REFERENCE_COLS)
    fixed_camera_target, camera_distance = compute_camera_setup(df)
    if args.distance is not None:
        camera_distance = float(args.distance)
    else:
        camera_distance = max(0.2, camera_distance * float(args.distance_scale))

    env = build_env(df, args.fps)
    env.reset(seed=42)
    client = env.PYB_CLIENT
    drone_id = env.DRONE_ID

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client)
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=args.yaw,
        cameraPitch=args.pitch,
        cameraTargetPosition=fixed_camera_target,
        physicsClientId=client,
    )

    draw_world_frame(client)
    goal_marker_id = create_goal_marker(client)

    frames = []
    prev_ref_pos = None
    prev_real_pos = None

    print(f"Recording replay from {csv_path}")
    print(f"Saving video to {video_path}")

    for i in tqdm(range(len(df)), desc="Replaying"):
        row = df.iloc[i]
        real_pos = [float(row["x"]), float(row["y"]), float(row["z"])]
        real_vel = [float(row["x_dot"]), float(row["y_dot"]), float(row["z_dot"])]
        real_rpy = [float(row["phi"]), float(row["theta"]), float(row["psi"])]
        real_rates = [float(row["p"]), float(row["q"]), float(row["r"])]

        if has_reference:
            ref_pos = [float(row["x_ref"]), float(row["y_ref"]), float(row["z_ref"])]
            if prev_ref_pos is not None:
                draw_segment(client, prev_ref_pos, ref_pos, [1, 0, 0], width=2)
            prev_ref_pos = ref_pos
            p.resetBasePositionAndOrientation(goal_marker_id, ref_pos, [0, 0, 0, 1], client)

        if prev_real_pos is not None:
            draw_segment(client, prev_real_pos, real_pos, traj_color, width=3)
        prev_real_pos = real_pos

        quat = p.getQuaternionFromEuler(real_rpy)
        p.resetBasePositionAndOrientation(drone_id, real_pos, quat, client)
        p.resetBaseVelocity(drone_id, real_vel, real_rates, client)

        p.stepSimulation(client)

        camera_target = real_pos if args.follow else fixed_camera_target
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=camera_distance,
            yaw=args.yaw,
            pitch=args.pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=args.width / args.height,
            nearVal=0.1,
            farVal=100.0,
        )
        _, _, px, _, _ = p.getCameraImage(
            width=args.width,
            height=args.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client,
        )
        frame = np.reshape(px, (args.height, args.width, 4))[:, :, :3]
        frames.append(frame)

    print(f"Writing video to {video_path}")
    with imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame in tqdm(frames, desc="Writing"):
            writer.append_data(frame)

    print(f"Done. Video saved to: {video_path}")
    env.close()


if __name__ == "__main__":
    main()
