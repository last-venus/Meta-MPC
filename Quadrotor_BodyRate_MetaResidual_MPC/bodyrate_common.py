import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
TRACKING_DIR = REPO_ROOT / "Quadrotor_3D_Tracking"
SAFE_CONTROL_GYM_ROOTS = (
    REPO_ROOT / "third_party" / "safe-control-gym",
    WORKSPACE_ROOT / "safe-control-gym",
)
for path in SAFE_CONTROL_GYM_ROOTS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
        break
for path in (TRACKING_DIR,):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import l4casadi as l4c  # noqa: E402
from quadrotor3D_common import (  # noqa: E402
    BASE_PARAMS,
    COST,
    MLP,
    NOMINAL_RATIOS,
    ContextResidualMLP,
    ReferenceConfig,
    SupportSetEncoder,
    default_init_randomization,
    export_3d_animation,
    plot_tracking_results,
    position_rmse,
    reference_state,
    seed_everything,
    wrap_state_angles,
)
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor  # noqa: E402
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType  # noqa: E402


COMMAND_MODE_BODY_RATE_THRUST = "body_rate_thrust"
STATE_COLS = ["x", "x_dot", "y", "y_dot", "z", "z_dot", "phi", "theta", "psi", "p", "q", "r"]
CONTROL_COLS = ["p_cmd", "q_cmd", "r_cmd", "thrust_cmd"]
MOTOR_CONTROL_COLS = ["motor_u1", "motor_u2", "motor_u3", "motor_u4"]
# First-stage adaptation corrects translational dynamics only; angular-rate
# tracking stays in the fixed nominal inner-loop model.
TRACKED_DERIVATIVE_INDICES = np.array([1, 3, 5], dtype=int)
MODEL_INPUT_DIM = len(STATE_COLS) + len(CONTROL_COLS)
RESIDUAL_OUTPUT_DIM = len(TRACKED_DERIVATIVE_INDICES)

DEFAULT_CTRL_FREQ = 50
DEFAULT_PYB_FREQ = 50
DEFAULT_GUI = False
DEFAULT_SAVE_RESULTS = True
DEFAULT_SHOW_PLOT_WINDOW = False
DEFAULT_EXPORT_ANIMATION = False
DEFAULT_ANIMATION_FORMAT = "gif"
DEFAULT_ANIMATION_FPS = 20
DEFAULT_T_HORIZON = 1.0
DEFAULT_N_HORIZON = 20
DEFAULT_SIM_TIME = 12.0
DEFAULT_RMSE_WARMUP_SEC = 2.0
DEFAULT_RUN_SEED = 42
DEFAULT_SCRIPT_SEED = 1

DEFAULT_RESIDUAL_HIDDEN_DIM = 128
DEFAULT_RESIDUAL_NUM_LAYERS = 3
DEFAULT_LIGHTMLP_ADAPTATION_LR = 1e-3
DEFAULT_LIGHTMLP_ADAPTATION_BATCH_SIZE = 96
DEFAULT_LIGHTMLP_ADAPTATION_STEPS = 10
DEFAULT_LIGHTMLP_ADAPTATION_INTERVAL_SEC = 1.0
DEFAULT_META_CONTEXT_INNER_LR = 5e-2
DEFAULT_ONLINE_CONTEXT_EMA_DECAY = 0.0
DEFAULT_ONLINE_CONTEXT_WARMUP_UPDATES = 1

DEFAULT_RATE_TIME_CONSTANTS = np.array([0.08, 0.08, 0.12], dtype=float)
BODY_RATE_LIMITS = np.array([3.0, 3.0, 2.0], dtype=float)
MIN_THRUST_HOVER_RATIO = 0.30
MAX_THRUST_HOVER_RATIO = 3.00
MPC_STATE_WEIGHT_DIAG = np.array([8.0, 0.4, 8.0, 0.4, 12.0, 0.6, 1.5, 1.5, 1.0, 0.12, 0.12, 0.16], dtype=float)
MPC_CONTROL_WEIGHT_DIAG = np.array([0.03, 0.03, 0.04, 0.015], dtype=float)

DEFAULT_DATASET_DIR = MODULE_DIR / "meta_dataset_bodyrate"
DEFAULT_DATASET_PATH = DEFAULT_DATASET_DIR / "bodyrate_bodyframe_accel_shards"
DEFAULT_CHECKPOINT_PATH = MODULE_DIR / "MetaLearning" / "bodyrate_bodyframe_accel_meta.pth"
DEFAULT_RESULTS_DIR = MODULE_DIR / "results"
DEFAULT_ACADOS_EXPORT_DIR = MODULE_DIR / "c_generated_code"
DEFAULT_L4C_BUILD_DIR = MODULE_DIR / "_l4c_bodyframe_generated"
DEFAULT_L4C_MODEL_NAME = "bodyrate_bodyframe_accel_residual_quadrotor3D"
USE_NOMINAL_RATE_CONTROLLER_INERTIA = True
META_RESIDUAL_MPC_SCALE = 0.20
# Absolute 99.9th-percentile residual envelopes from the final 512-task dataset.
# The smooth saturation is a Meta-MPC trust region for predicted horizon states.
META_RESIDUAL_MPC_LIMITS = (6.0, 6.0, 8.0)
LIGHTMLP_RESIDUAL_MPC_SCALE = 0.75

DEFAULT_EVAL_TASK = None
METHOD_LABELS = {"nominal": "nominal_bodyrate", "lightmlp": "lightmlp_bodyrate", "meta": "metacontext_bodyrate"}


@dataclass
class DynamicsTask:
    mass_ratio: float = 1.0
    ixx_ratio: float = 1.0
    iyy_ratio: float = 1.0
    izz_ratio: float = 1.0
    thrust_scale: float = 1.0
    rate_tau_scale: float = 1.0
    drag_linear: float = 0.0
    drag_quad: float = 0.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def inertial_prop(self) -> list[float]:
        base = nominal_params()
        return [
            base["M"] * self.mass_ratio,
            base["Ixx"] * self.ixx_ratio,
            base["Iyy"] * self.iyy_ratio,
            base["Izz"] * self.izz_ratio,
        ]

    @property
    def actual_rate_time_constants(self) -> np.ndarray:
        return DEFAULT_RATE_TIME_CONSTANTS * float(self.rate_tau_scale)

    def context_values(self) -> dict[str, float]:
        return {
            "mass_ratio": self.mass_ratio,
            "ixx_ratio": self.ixx_ratio,
            "iyy_ratio": self.iyy_ratio,
            "izz_ratio": self.izz_ratio,
            "thrust_scale": self.thrust_scale,
            "rate_tau_scale": self.rate_tau_scale,
            "drag_linear": self.drag_linear,
            "drag_quad": self.drag_quad,
            "wind_x": self.wind[0],
            "wind_y": self.wind[1],
            "wind_z": self.wind[2],
        }


CONTEXT_COLS = list(DynamicsTask().context_values().keys())


@contextmanager
def working_directory(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def nominal_params() -> dict[str, float]:
    return {name: BASE_PARAMS[name] * NOMINAL_RATIOS[name] for name in BASE_PARAMS}


def make_env_config(seed: int, gui: bool, done_on_out_of_bound: bool, episode_len_sec: float, task: DynamicsTask | None = None, init_state=None) -> dict:
    task = task or DynamicsTask()
    env_config = {
        "gui": gui,
        "ctrl_freq": DEFAULT_CTRL_FREQ,
        "pyb_freq": DEFAULT_PYB_FREQ,
        "quad_type": QuadType.THREE_D,
        "seed": seed,
        "done_on_out_of_bound": done_on_out_of_bound,
        "episode_len_sec": episode_len_sec,
        "task_info": {"stabilization_goal": [0.0, 0.0, 1.0], "stabilization_goal_tolerance": 0.05},
        "init_state_randomization_info": default_init_randomization(),
        "randomized_init": init_state is None,
        "inertial_prop": task.inertial_prop,
        "adversary_disturbance": "dynamics",
        "adversary_disturbance_scale": 1.0,
    }
    if init_state is not None:
        env_config["init_state"] = init_state
    return env_config


def motor_thrust_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    low = env.KF * (env.PWM2RPM_SCALE * env.MIN_PWM + env.PWM2RPM_CONST) ** 2
    high = env.KF * (env.PWM2RPM_SCALE * env.MAX_PWM + env.PWM2RPM_CONST) ** 2
    return low * np.ones(4), high * np.ones(4)


def bodyrate_control_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    _, motor_high = motor_thrust_bounds(env)
    hover = nominal_params()["M"] * float(env.GRAVITY_ACC)
    lower = np.array(
        [-BODY_RATE_LIMITS[0], -BODY_RATE_LIMITS[1], -BODY_RATE_LIMITS[2], MIN_THRUST_HOVER_RATIO * hover],
        dtype=float,
    )
    upper = np.array(
        [BODY_RATE_LIMITS[0], BODY_RATE_LIMITS[1], BODY_RATE_LIMITS[2], min(float(np.sum(motor_high)), MAX_THRUST_HOVER_RATIO * hover)],
        dtype=float,
    )
    return lower, upper


def input_reference(env=None) -> np.ndarray:
    params = nominal_params()
    g = 9.8 if env is None else float(env.GRAVITY_ACC)
    return np.array([0.0, 0.0, 0.0, params["M"] * g], dtype=float)


def allocation_matrix(env) -> np.ndarray:
    l = float(env.L) / np.sqrt(2.0)
    gamma = float(env.KM / env.KF)
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [l, l, -l, -l],
            [-l, l, l, -l],
            [-gamma, gamma, -gamma, gamma],
        ],
        dtype=float,
    )


def nominal_inertia_matrix() -> np.ndarray:
    params = nominal_params()
    return np.diag([params["Ixx"], params["Iyy"], params["Izz"]])


def body_rate_thrust_to_motor_forces(state: np.ndarray, cmd: np.ndarray, env, rate_time_constants=None, controller_inertia=None) -> np.ndarray:
    cmd = np.asarray(cmd, dtype=float).copy()
    cmd[:3] = np.clip(cmd[:3], -BODY_RATE_LIMITS, BODY_RATE_LIMITS)
    u_min, u_max = bodyrate_control_bounds(env)
    cmd[3] = np.clip(cmd[3], u_min[3], u_max[3])
    tau_constants = np.asarray(rate_time_constants if rate_time_constants is not None else DEFAULT_RATE_TIME_CONSTANTS, dtype=float)
    inertia = np.asarray(controller_inertia if controller_inertia is not None else env.J, dtype=float)
    omega = np.asarray(state[9:12], dtype=float)
    omega_cmd = cmd[:3]
    omega_dot_des = (omega_cmd - omega) / tau_constants
    torque = inertia @ omega_dot_des + np.cross(omega, inertia @ omega)
    wrench = np.array([cmd[3], torque[0], torque[1], torque[2]], dtype=float)
    motor_forces = np.linalg.solve(allocation_matrix(env), wrench)
    low, high = motor_thrust_bounds(env)
    return np.clip(motor_forces, low, high)


def disturbance_force_from_task(state: np.ndarray, task: DynamicsTask) -> np.ndarray:
    vel = np.asarray(state[[1, 3, 5]], dtype=float)
    wind = np.asarray(task.wind, dtype=float)
    drag = task.drag_linear * vel + task.drag_quad * vel * np.abs(vel)
    return wind - drag


def step_bodyrate_env(env, state: np.ndarray, cmd: np.ndarray, task: DynamicsTask):
    controller_inertia = nominal_inertia_matrix() if USE_NOMINAL_RATE_CONTROLLER_INERTIA else env.J
    motor_forces = body_rate_thrust_to_motor_forces(state, cmd, env, task.actual_rate_time_constants, controller_inertia)
    executed_forces = np.clip(motor_forces * float(task.thrust_scale), *motor_thrust_bounds(env))
    disturbance_force = disturbance_force_from_task(state, task)
    env.adv_action = disturbance_force
    obs, reward, done, info = env.step(executed_forces)
    info = dict(info)
    info["motor_forces"] = motor_forces
    info["executed_motor_forces"] = executed_forces
    info["disturbance_force"] = disturbance_force
    return obs, reward, done, info



def nominal_dynamics_terms(env, model_name: str):
    params = nominal_params()
    m = params["M"]
    g = env.GRAVITY_ACC
    tau_p, tau_q, tau_r = DEFAULT_RATE_TIME_CONSTANTS

    x = cs.MX.sym("x")
    x_dot = cs.MX.sym("x_dot")
    y = cs.MX.sym("y")
    y_dot = cs.MX.sym("y_dot")
    z = cs.MX.sym("z")
    z_dot = cs.MX.sym("z_dot")
    phi = cs.MX.sym("phi")
    theta = cs.MX.sym("theta")
    psi = cs.MX.sym("psi")
    p_body = cs.MX.sym("p")
    q_body = cs.MX.sym("q")
    r_body = cs.MX.sym("r")
    p_cmd = cs.MX.sym("p_cmd")
    q_cmd = cs.MX.sym("q_cmd")
    r_cmd = cs.MX.sym("r_cmd")
    thrust_cmd = cs.MX.sym("thrust_cmd")

    x_state = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)
    u = cs.vertcat(p_cmd, q_cmd, r_cmd, thrust_cmd)
    xdot = cs.MX.sym("xdot", 12)

    c_phi, s_phi = cs.cos(phi), cs.sin(phi)
    c_theta, s_theta = cs.cos(theta), cs.sin(theta)
    c_psi, s_psi = cs.cos(psi), cs.sin(psi)
    rot = cs.vertcat(
        cs.horzcat(c_theta * c_psi, -c_phi * s_psi + s_phi * s_theta * c_psi, s_phi * s_psi + c_phi * s_theta * c_psi),
        cs.horzcat(c_theta * s_psi, c_phi * c_psi + s_phi * s_theta * s_psi, -s_phi * c_psi + c_phi * s_theta * s_psi),
        cs.horzcat(-s_theta, s_phi * c_theta, c_phi * c_theta),
    )
    pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
    pos_ddot = rot @ cs.vertcat(0, 0, thrust_cmd) / m - cs.vertcat(0, 0, g)
    euler_rates = cs.vertcat(
        p_body + q_body * s_phi * cs.tan(theta) + r_body * c_phi * cs.tan(theta),
        q_body * c_phi - r_body * s_phi,
        q_body * s_phi / c_theta + r_body * c_phi / c_theta,
    )
    body_rates_dot = cs.vertcat((p_cmd - p_body) / tau_p, (q_cmd - q_body) / tau_q, (r_cmd - r_body) / tau_r)
    f_nominal = cs.vertcat(
        pos_dot[0],
        pos_ddot[0],
        pos_dot[1],
        pos_ddot[1],
        pos_dot[2],
        pos_ddot[2],
        euler_rates,
        body_rates_dot,
    )
    u_min, u_max = bodyrate_control_bounds(env)
    model = cs.types.SimpleNamespace()
    model.x = x_state
    model.xdot = xdot
    model.u = u
    model.u_min = u_min
    model.u_max = u_max
    model.p = cs.vertcat([])
    model.z = cs.vertcat([])
    model.f_nominal = f_nominal
    model.x_start = np.zeros(12)
    model.name = model_name
    return model


class BodyRateNominalDynamics:
    def __init__(self, env):
        self.env = env

    def model(self):
        model = nominal_dynamics_terms(self.env, "quadrotor3D_bodyrate_nominal")
        model.f_expl = model.f_nominal
        return model


class BodyRateLearnedDynamics:
    def __init__(self, env, residual_model, residual_scale: float = 1.0, residual_limits=None):
        self.env = env
        self.residual_model = residual_model
        self.residual_scale = float(residual_scale)
        self.residual_limits = None if residual_limits is None else np.asarray(residual_limits, dtype=float)

    def model(self):
        model = nominal_dynamics_terms(self.env, "quadrotor3D_bodyrate_learned")
        residual = self.residual_model(cs.vertcat(model.x, model.u).T).T
        if self.residual_limits is not None:
            limits = cs.DM(self.residual_limits)
            residual = limits * cs.tanh(residual / limits)
        # The residual is a body-frame acceleration.  It is rotated at every
        # predicted MPC node, so the learned dynamics act through the full
        # horizon rather than as a one-step controller correction.
        phi, theta, psi = model.x[6], model.x[7], model.x[8]
        c_phi, s_phi = cs.cos(phi), cs.sin(phi)
        c_theta, s_theta = cs.cos(theta), cs.sin(theta)
        c_psi, s_psi = cs.cos(psi), cs.sin(psi)
        body_to_world = cs.vertcat(
            cs.horzcat(c_theta * c_psi, -c_phi * s_psi + s_phi * s_theta * c_psi, s_phi * s_psi + c_phi * s_theta * c_psi),
            cs.horzcat(c_theta * s_psi, c_phi * c_psi + s_phi * s_theta * s_psi, -s_phi * c_psi + c_phi * s_theta * s_psi),
            cs.horzcat(-s_theta, s_phi * c_theta, c_phi * c_theta),
        )
        world_accel_residual = body_to_world @ residual
        residual_term = cs.vertcat(0, world_accel_residual[0], 0, world_accel_residual[1], 0, world_accel_residual[2], 0, 0, 0, 0, 0, 0)
        model.f_expl = model.f_nominal + self.residual_scale * residual_term
        return model


class NormalizedContextResidualMLP(ContextResidualMLP):
    def __init__(self, *args, **kwargs):
        input_dim = kwargs.get("input_dim", args[0] if args else MODEL_INPUT_DIM)
        output_dim = kwargs.get("output_dim", args[1] if len(args) > 1 else RESIDUAL_OUTPUT_DIM)
        super().__init__(*args, **kwargs)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(output_dim))
        self.register_buffer("target_std", torch.ones(output_dim))

    @torch.no_grad()
    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        self.input_mean.copy_(torch.as_tensor(input_mean, dtype=self.input_mean.dtype, device=self.input_mean.device))
        self.input_std.copy_(torch.clamp(torch.as_tensor(input_std, dtype=self.input_std.dtype, device=self.input_std.device), min=1e-6))
        self.target_mean.copy_(torch.as_tensor(target_mean, dtype=self.target_mean.dtype, device=self.target_mean.device))
        self.target_std.copy_(torch.clamp(torch.as_tensor(target_std, dtype=self.target_std.dtype, device=self.target_std.device), min=1e-6))

    def forward_with_context(self, x, context):
        x_norm = (x - self.input_mean.to(dtype=x.dtype, device=x.device)) / self.input_std.to(dtype=x.dtype, device=x.device)
        y_norm = super().forward_with_context(x_norm, context)
        return y_norm * self.target_std.to(dtype=y_norm.dtype, device=y_norm.device) + self.target_mean.to(dtype=y_norm.dtype, device=y_norm.device)


class NormalizedSupportSetEncoder(SupportSetEncoder):
    def __init__(self, *args, **kwargs):
        feature_dim = kwargs.get("feature_dim", args[0] if args else MODEL_INPUT_DIM)
        target_dim = kwargs.get("target_dim", args[1] if len(args) > 1 else RESIDUAL_OUTPUT_DIM)
        super().__init__(*args, **kwargs)
        self.register_buffer("input_mean", torch.zeros(feature_dim))
        self.register_buffer("input_std", torch.ones(feature_dim))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))

    @torch.no_grad()
    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        self.input_mean.copy_(torch.as_tensor(input_mean, dtype=self.input_mean.dtype, device=self.input_mean.device))
        self.input_std.copy_(torch.clamp(torch.as_tensor(input_std, dtype=self.input_std.dtype, device=self.input_std.device), min=1e-6))
        self.target_mean.copy_(torch.as_tensor(target_mean, dtype=self.target_mean.dtype, device=self.target_mean.device))
        self.target_std.copy_(torch.clamp(torch.as_tensor(target_std, dtype=self.target_std.dtype, device=self.target_std.device), min=1e-6))

    def forward(self, features, targets):
        features = (features - self.input_mean.to(dtype=features.dtype, device=features.device)) / self.input_std.to(dtype=features.dtype, device=features.device)
        targets = (targets - self.target_mean.to(dtype=targets.dtype, device=targets.device)) / self.target_std.to(dtype=targets.dtype, device=targets.device)
        return super().forward(features, targets)


# The context is supervised with known task parameters during offline training,
# but the network must infer it from a short online support trajectory.  These
# groups encode the causal channels that affect the residual dynamics.
PHYSICS_CONTEXT_GROUPS = {
    "translation": (0, 4),             # mass ratio, thrust scale
    "rotation": (1, 2, 3, 5),          # inertia ratios, rate time constant
    "aerodynamics": (6, 7, 8, 9, 10), # drag and wind
}


class TemporalPhysicsSupportSetEncoder(nn.Module):
    """Infer a task context from an ordered support trajectory.

    Unlike the legacy mean/std set encoder, this module preserves temporal
    residual signatures with a GRU and recency attention.  It still consumes
    exactly the existing ``(state, action, residual)`` support data.
    """

    def __init__(self, feature_dim=MODEL_INPUT_DIM, target_dim=RESIDUAL_OUTPUT_DIM, context_dim=len(CONTEXT_COLS), hidden_dim=96, num_layers=2):
        super().__init__()
        if context_dim != len(CONTEXT_COLS):
            raise ValueError(f"Expected {len(CONTEXT_COLS)} physics context dimensions, got {context_dim}.")
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        layers = [nn.Linear(feature_dim + target_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.sample_net = nn.Sequential(*layers)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.register_buffer("input_mean", torch.zeros(feature_dim))
        self.register_buffer("input_std", torch.ones(feature_dim))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))

    @torch.no_grad()
    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        self.input_mean.copy_(torch.as_tensor(input_mean, dtype=self.input_mean.dtype, device=self.input_mean.device))
        self.input_std.copy_(torch.clamp(torch.as_tensor(input_std, dtype=self.input_std.dtype, device=self.input_std.device), min=1e-6))
        self.target_mean.copy_(torch.as_tensor(target_mean, dtype=self.target_mean.dtype, device=self.target_mean.device))
        self.target_std.copy_(torch.clamp(torch.as_tensor(target_std, dtype=self.target_std.dtype, device=self.target_std.device), min=1e-6))

    def forward(self, features, targets):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        features = (features - self.input_mean.to(dtype=features.dtype, device=features.device)) / self.input_std.to(dtype=features.dtype, device=features.device)
        targets = (targets - self.target_mean.to(dtype=targets.dtype, device=targets.device)) / self.target_std.to(dtype=targets.dtype, device=targets.device)
        encoded = self.sample_net(torch.cat([features, targets], dim=-1))
        temporal, _ = self.temporal_gru(encoded.unsqueeze(0))
        temporal = temporal.squeeze(0)
        attention = torch.softmax(self.attention(temporal).squeeze(-1), dim=0).unsqueeze(-1)
        pooled = torch.cat(
            [
                temporal.mean(dim=0, keepdim=True),
                temporal.std(dim=0, keepdim=True, unbiased=False),
                (attention * temporal).sum(dim=0, keepdim=True),
                temporal[-1:].contiguous(),
            ],
            dim=-1,
        )
        return self.head(pooled)


class GroupedTemporalPhysicsSupportSetEncoder(TemporalPhysicsSupportSetEncoder):
    """Temporal encoder with independent physical-context output heads.

    Shared temporal features capture common motion information, while separate
    heads prevent translation, rotation, and aerodynamic supervision from
    competing in the final context regression layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.head

        def make_head(output_dim):
            return nn.Sequential(
                nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, output_dim),
            )

        self.translation_head = make_head(len(PHYSICS_CONTEXT_GROUPS["translation"]))
        self.rotation_head = make_head(len(PHYSICS_CONTEXT_GROUPS["rotation"]))
        self.aerodynamics_head = make_head(len(PHYSICS_CONTEXT_GROUPS["aerodynamics"]))

    def forward(self, features, targets):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        features = (features - self.input_mean.to(dtype=features.dtype, device=features.device)) / self.input_std.to(dtype=features.dtype, device=features.device)
        targets = (targets - self.target_mean.to(dtype=targets.dtype, device=targets.device)) / self.target_std.to(dtype=targets.dtype, device=targets.device)
        encoded = self.sample_net(torch.cat([features, targets], dim=-1))
        temporal, _ = self.temporal_gru(encoded.unsqueeze(0))
        temporal = temporal.squeeze(0)
        attention = torch.softmax(self.attention(temporal).squeeze(-1), dim=0).unsqueeze(-1)
        pooled = torch.cat(
            [
                temporal.mean(dim=0, keepdim=True),
                temporal.std(dim=0, keepdim=True, unbiased=False),
                (attention * temporal).sum(dim=0, keepdim=True),
                temporal[-1:].contiguous(),
            ],
            dim=-1,
        )
        translation = self.translation_head(pooled)
        rotation = self.rotation_head(pooled)
        aerodynamics = self.aerodynamics_head(pooled)
        return torch.cat(
            [translation[:, :1], rotation[:, :3], translation[:, 1:], rotation[:, 3:], aerodynamics],
            dim=-1,
        )


class _PhysicsTemporalContextBranch(nn.Module):
    """One temporal identifier for a single causal physics group."""

    def __init__(self, feature_dim: int, target_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        layers = [nn.Linear(feature_dim + target_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.sample_net = nn.Sequential(*layers)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, normalized_features, normalized_targets):
        encoded = self.sample_net(torch.cat([normalized_features, normalized_targets], dim=-1))
        temporal, _ = self.temporal_gru(encoded.unsqueeze(0))
        temporal = temporal.squeeze(0)
        attention = torch.softmax(self.attention(temporal).squeeze(-1), dim=0).unsqueeze(-1)
        pooled = torch.cat(
            [
                temporal.mean(dim=0, keepdim=True),
                temporal.std(dim=0, keepdim=True, unbiased=False),
                (attention * temporal).sum(dim=0, keepdim=True),
                temporal[-1:].contiguous(),
            ],
            dim=-1,
        )
        return self.head(pooled)


class GroupedTemporalPhysicsSupportSetEncoderV3(nn.Module):
    """Causally separated temporal task identifier for context-meta MPC.

    Version 2 had independent output heads but one shared sequence encoder.
    The strongly observable translational residual could therefore optimize the
    shared features at the expense of rotational or aerodynamic identification.
    Version 3 keeps the same 11-D context and support/query meta-learning
    protocol, while assigning each causal physics group its own temporal path.
    """

    def __init__(self, feature_dim=MODEL_INPUT_DIM, target_dim=RESIDUAL_OUTPUT_DIM, context_dim=len(CONTEXT_COLS), hidden_dim=96, num_layers=2):
        super().__init__()
        if context_dim != len(CONTEXT_COLS):
            raise ValueError(f"Expected {len(CONTEXT_COLS)} physics context dimensions, got {context_dim}.")
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.translation_branch = _PhysicsTemporalContextBranch(feature_dim, target_dim, hidden_dim, num_layers, len(PHYSICS_CONTEXT_GROUPS["translation"]))
        self.rotation_branch = _PhysicsTemporalContextBranch(feature_dim, target_dim, hidden_dim, num_layers, len(PHYSICS_CONTEXT_GROUPS["rotation"]))
        self.aerodynamics_branch = _PhysicsTemporalContextBranch(feature_dim, target_dim, hidden_dim, num_layers, len(PHYSICS_CONTEXT_GROUPS["aerodynamics"]))
        self.register_buffer("input_mean", torch.zeros(feature_dim))
        self.register_buffer("input_std", torch.ones(feature_dim))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))

    @torch.no_grad()
    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        self.input_mean.copy_(torch.as_tensor(input_mean, dtype=self.input_mean.dtype, device=self.input_mean.device))
        self.input_std.copy_(torch.clamp(torch.as_tensor(input_std, dtype=self.input_std.dtype, device=self.input_std.device), min=1e-6))
        self.target_mean.copy_(torch.as_tensor(target_mean, dtype=self.target_mean.dtype, device=self.target_mean.device))
        self.target_std.copy_(torch.clamp(torch.as_tensor(target_std, dtype=self.target_std.dtype, device=self.target_std.device), min=1e-6))

    def forward(self, features, targets):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        features = (features - self.input_mean.to(dtype=features.dtype, device=features.device)) / self.input_std.to(dtype=features.dtype, device=features.device)
        targets = (targets - self.target_mean.to(dtype=targets.dtype, device=targets.device)) / self.target_std.to(dtype=targets.dtype, device=targets.device)
        translation = self.translation_branch(features, targets)
        rotation = self.rotation_branch(features, targets)
        aerodynamics = self.aerodynamics_branch(features, targets)
        return torch.cat(
            [translation[:, :1], rotation[:, :3], translation[:, 1:], rotation[:, 3:], aerodynamics],
            dim=-1,
        )


class PhysicsDisentangledContextResidualMLP(nn.Module):
    """Physics-informed context-conditioned residual model.

    The translational residual is conditioned only on mass/thrust and
    aerodynamic contexts, while the angular-rate residual is conditioned only
    on inertia/rate contexts.  This hard architectural separation prevents a
    nuisance context from explaining an incompatible physical residual.
    """

    def __init__(self, input_dim=MODEL_INPUT_DIM, output_dim=RESIDUAL_OUTPUT_DIM, hidden_dim=128, num_layers=3, context_dim=len(CONTEXT_COLS), modulation_scale=0.20):
        super().__init__()
        if output_dim != RESIDUAL_OUTPUT_DIM:
            raise ValueError(f"Expected residual output dimension {RESIDUAL_OUTPUT_DIM}, got {output_dim}.")
        if context_dim != len(CONTEXT_COLS):
            raise ValueError(f"Expected context dimension {len(CONTEXT_COLS)}, got {context_dim}.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.modulation_scale = float(modulation_scale)
        self.context_init = nn.Parameter(torch.zeros(context_dim))
        self.context = nn.Parameter(torch.zeros(context_dim), requires_grad=False)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, num_layers - 1))])
        self.trans_scale = nn.Linear(7, hidden_dim)
        self.trans_shift = nn.Linear(7, hidden_dim)
        self.rot_scale = nn.Linear(4, hidden_dim)
        self.rot_shift = nn.Linear(4, hidden_dim)
        self.trans_head = nn.Linear(hidden_dim, 3)
        self.rot_head = nn.Linear(hidden_dim, 3)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(output_dim))
        self.register_buffer("target_std", torch.ones(output_dim))
        # Deployment-time fast weights. They are deliberately not persistent:
        # every episode starts from the meta-trained zero adapter and support
        # data, never from a previous task's fitted correction.
        self.register_buffer("fast_adapter", torch.zeros(2 * output_dim), persistent=False)

    @torch.no_grad()
    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        self.input_mean.copy_(torch.as_tensor(input_mean, dtype=self.input_mean.dtype, device=self.input_mean.device))
        self.input_std.copy_(torch.clamp(torch.as_tensor(input_std, dtype=self.input_std.dtype, device=self.input_std.device), min=1e-6))
        self.target_mean.copy_(torch.as_tensor(target_mean, dtype=self.target_mean.dtype, device=self.target_mean.device))
        self.target_std.copy_(torch.clamp(torch.as_tensor(target_std, dtype=self.target_std.dtype, device=self.target_std.device), min=1e-6))

    @staticmethod
    def _expand_context(x, context):
        if context.dim() == 1:
            return context.unsqueeze(0).expand(x.shape[0], -1)
        if context.dim() == 2 and context.shape[0] == 1 and x.shape[0] != 1:
            return context.expand(x.shape[0], -1)
        return context

    def _modulate(self, hidden, context, scale_layer, shift_layer):
        gamma = 1.0 + self.modulation_scale * torch.tanh(scale_layer(context))
        beta = self.modulation_scale * shift_layer(context)
        return torch.relu(gamma * hidden + beta)

    def forward_with_context_and_adapter(self, x, context, adapter):
        squeeze_output = x.dim() == 1
        if squeeze_output:
            x = x.unsqueeze(0)
        context = self._expand_context(x, context.to(dtype=x.dtype, device=x.device))
        x_norm = (x - self.input_mean.to(dtype=x.dtype, device=x.device)) / self.input_std.to(dtype=x.dtype, device=x.device)
        hidden = torch.relu(self.input_layer(x_norm))
        for layer in self.hidden_layers:
            hidden = torch.relu(layer(hidden))
        translation_context = torch.cat([context[:, PHYSICS_CONTEXT_GROUPS["translation"]], context[:, PHYSICS_CONTEXT_GROUPS["aerodynamics"]]], dim=-1)
        rotation_context = context[:, PHYSICS_CONTEXT_GROUPS["rotation"]]
        translation = self.trans_head(self._modulate(hidden, translation_context, self.trans_scale, self.trans_shift))
        rotation = self.rot_head(self._modulate(hidden, rotation_context, self.rot_scale, self.rot_shift))
        y_norm = torch.cat([translation, rotation], dim=-1)
        adapter = self._expand_context(y_norm, adapter.to(dtype=y_norm.dtype, device=y_norm.device))
        gain, bias = adapter[:, :self.output_dim], adapter[:, self.output_dim:]
        # The compact adapter can correct task-wide residual scale and bias,
        # but tanh bounds preserve the learned model's MPC trust region.
        y_norm = y_norm * (1.0 + 0.25 * torch.tanh(gain)) + 0.25 * torch.tanh(bias)
        output = y_norm * self.target_std.to(dtype=y_norm.dtype, device=y_norm.device) + self.target_mean.to(dtype=y_norm.dtype, device=y_norm.device)
        return output.squeeze(0) if squeeze_output else output

    def forward_with_context(self, x, context):
        return self.forward_with_context_and_adapter(x, context, self.fast_adapter)

    def forward(self, x):
        return self.forward_with_context(x, self.context)

    @torch.no_grad()
    def set_context(self, context):
        self.context.copy_(context.detach().to(dtype=self.context.dtype, device=self.context.device))

    @torch.no_grad()
    def reset_context(self):
        self.context.copy_(self.context_init.detach())

    @torch.no_grad()
    def set_fast_adapter(self, adapter):
        self.fast_adapter.copy_(adapter.detach().to(dtype=self.fast_adapter.dtype, device=self.fast_adapter.device))

    @torch.no_grad()
    def reset_fast_adapter(self):
        self.fast_adapter.zero_()


class BodyFrameAccelerationResidualMLP(nn.Module):
    """Shared residual prior with a small, online-adapted task latent ``z``."""

    online_latent_only = True

    def __init__(self, input_dim=MODEL_INPUT_DIM, output_dim=RESIDUAL_OUTPUT_DIM,
                 hidden_dim=128, num_layers=3, context_dim=6,
                 modulation_scale=0.20):
        super().__init__()
        if output_dim != 3:
            raise ValueError(f"Body-frame residual must have three outputs, got {output_dim}.")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.context_dim, self.modulation_scale = context_dim, float(modulation_scale)
        self.context_init = nn.Parameter(torch.zeros(context_dim))
        self.register_buffer("context", torch.zeros(context_dim), persistent=False)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, num_layers - 1))])
        self.context_scale = nn.Linear(context_dim, hidden_dim)
        self.context_shift = nn.Linear(context_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(output_dim))
        self.register_buffer("target_std", torch.ones(output_dim))

    @torch.no_grad()
    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        self.input_mean.copy_(torch.as_tensor(input_mean, dtype=self.input_mean.dtype))
        self.input_std.copy_(torch.clamp(torch.as_tensor(input_std, dtype=self.input_std.dtype), min=1e-6))
        self.target_mean.copy_(torch.as_tensor(target_mean, dtype=self.target_mean.dtype))
        self.target_std.copy_(torch.clamp(torch.as_tensor(target_std, dtype=self.target_std.dtype), min=1e-6))

    @staticmethod
    def _expand_context(x, context):
        if context.dim() == 1:
            return context.unsqueeze(0).expand(x.shape[0], -1)
        if context.shape[0] == 1 and x.shape[0] != 1:
            return context.expand(x.shape[0], -1)
        return context

    def forward_with_context(self, x, context):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        z = self._expand_context(x, context.to(dtype=x.dtype, device=x.device))
        x = (x - self.input_mean.to(dtype=x.dtype, device=x.device)) / self.input_std.to(dtype=x.dtype, device=x.device)
        hidden = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            hidden = torch.relu(layer(hidden))
        gamma = 1.0 + self.modulation_scale * torch.tanh(self.context_scale(z))
        beta = self.modulation_scale * self.context_shift(z)
        output = self.output_layer(torch.relu(gamma * hidden + beta))
        output = output * self.target_std.to(dtype=output.dtype, device=output.device) + self.target_mean.to(dtype=output.dtype, device=output.device)
        return output.squeeze(0) if squeeze else output

    def forward(self, x):
        return self.forward_with_context(x, self.context)

    @torch.no_grad()
    def set_context(self, context):
        self.context.copy_(context.detach().to(dtype=self.context.dtype, device=self.context.device))

    @torch.no_grad()
    def reset_context(self):
        self.context.copy_(self.context_init.detach())


class MPC:
    def __init__(self, model, n_horizon, t_horizon, external_shared_lib_dir=None, external_shared_lib_name=None):
        self.model = model
        self.n_horizon = n_horizon
        self.t_horizon = t_horizon
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def _codegen_paths(self) -> tuple[Path, Path]:
        process_dir = DEFAULT_ACADOS_EXPORT_DIR / f"{self.model.name}_pid{os.getpid()}"
        return process_dir, process_dir / f"{self.model.name}_ocp.json"

    def ocp(self):
        nx, nu = 12, 4
        ny, ny_e = nx + nu, nx
        ocp = AcadosOcp()
        ocp.model = self.acados_model(self.model)
        code_export_dir, json_path = self._codegen_paths()
        try:
            ocp.code_export_directory = code_export_dir.as_posix()
        except AttributeError:
            pass
        try:
            ocp.code_gen_opts.code_export_directory = code_export_dir.as_posix()
            ocp.code_gen_opts.json_file = json_path.as_posix()
        except AttributeError:
            try:
                ocp.json_file = json_path.as_posix()
            except AttributeError:
                pass
        ocp.dims.N = self.n_horizon
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = self.t_horizon
        ocp.cost.cost_type = COST
        ocp.cost.cost_type_e = COST
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vx_e = np.eye(nx)
        np.fill_diagonal(ocp.cost.Vx[:nx, :], 1.0)
        np.fill_diagonal(ocp.cost.Vu[nx:, :], 1.0)
        q = np.diag(MPC_STATE_WEIGHT_DIAG)
        r = np.diag(MPC_CONTROL_WEIGHT_DIAG)
        ocp.cost.W = scipy.linalg.block_diag(q, r)
        ocp.cost.W_e = q
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        ocp.constraints.x0 = self.model.x_start
        ocp.constraints.lbu = self.model.u_min
        ocp.constraints.ubu = self.model.u_max
        ocp.constraints.idxbu = np.arange(nu)
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        if self.external_shared_lib_dir and self.external_shared_lib_name:
            ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        return ocp

    @staticmethod
    def acados_model(model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.name = model.name
        return model_ac


def finite_difference_targets(prev_state, next_state, action, nominal_func, dt: float):
    true_world_accel = (next_state[TRACKED_DERIVATIVE_INDICES] - prev_state[TRACKED_DERIVATIVE_INDICES]) / dt
    nominal_world_accel = nominal_func(prev_state, action).full().flatten()[TRACKED_DERIVATIVE_INDICES]
    return body_to_world_rotation(prev_state).T @ (true_world_accel - nominal_world_accel)


def body_to_world_rotation(state: np.ndarray) -> np.ndarray:
    """ZYX body-to-world rotation matching the MPC Euler-angle convention."""
    phi, theta, psi = np.asarray(state, dtype=float)[6:9]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    return np.array([
        [c_theta * c_psi, -c_phi * s_psi + s_phi * s_theta * c_psi, s_phi * s_psi + c_phi * s_theta * c_psi],
        [c_theta * s_psi, c_phi * c_psi + s_phi * s_theta * s_psi, -s_phi * c_psi + c_phi * s_theta * s_psi],
        [-s_theta, s_phi * c_theta, c_phi * c_theta],
    ], dtype=float)


def set_tracking_references(solver, current_time: float, n_horizon: int, t_horizon: float, reference_cfg: ReferenceConfig, input_ref: np.ndarray, ref_history: list):
    for k in range(n_horizon):
        x_ref_k = reference_state(current_time + k * t_horizon / n_horizon, reference_cfg)
        if k == 0:
            ref_history.append(x_ref_k.copy())
        solver.set(k, "yref", np.concatenate([x_ref_k, input_ref]))
    solver.set(n_horizon, "yref", reference_state(current_time + t_horizon, reference_cfg))


def resolve_online_adaptation_config(method: str, dt: float, checkpoint, batch_size, adaptation_steps, adaptation_interval_sec):
    if method == "meta" and checkpoint is not None:
        support_window = int(checkpoint.get("support_window", 32))
        # Re-adapt as soon as a full recent support window is available.  A
        # fixed one-second floor otherwise makes a 20--50 sample window stale
        # at the 50 Hz controller rate.
        default_interval_sec = support_window * dt
        return (
            support_window if batch_size is None else batch_size,
            int(checkpoint.get("online_context_refinement_steps", 2)) if adaptation_steps is None else adaptation_steps,
            default_interval_sec if adaptation_interval_sec is None else adaptation_interval_sec,
        )
    return (
        DEFAULT_LIGHTMLP_ADAPTATION_BATCH_SIZE if batch_size is None else batch_size,
        DEFAULT_LIGHTMLP_ADAPTATION_STEPS if adaptation_steps is None else adaptation_steps,
        DEFAULT_LIGHTMLP_ADAPTATION_INTERVAL_SEC if adaptation_interval_sec is None else adaptation_interval_sec,
    )


def build_meta_components_from_checkpoint(checkpoint: dict):
    architecture = checkpoint.get("architecture", "legacy_context_residual")
    if architecture == "bodyframe_accel_latent_v1":
        model = BodyFrameAccelerationResidualMLP(
            input_dim=checkpoint["input_dim"], output_dim=checkpoint["output_dim"],
            hidden_dim=checkpoint["hidden_dim"], num_layers=checkpoint["num_layers"],
            context_dim=checkpoint["context_dim"], modulation_scale=float(checkpoint.get("modulation_scale", 0.2)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.reset_context()
        return model, None
    if architecture in {"temporal_physics_disentangled", "temporal_physics_disentangled_v2", "temporal_physics_disentangled_v3"}:
        model = PhysicsDisentangledContextResidualMLP(
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            context_dim=checkpoint["context_dim"],
            modulation_scale=float(checkpoint.get("modulation_scale", 0.2)),
        )
        encoder_cls = (
            GroupedTemporalPhysicsSupportSetEncoderV3 if architecture == "temporal_physics_disentangled_v3" else GroupedTemporalPhysicsSupportSetEncoder if architecture == "temporal_physics_disentangled_v2" else TemporalPhysicsSupportSetEncoder
        )
        encoder = encoder_cls(
            feature_dim=checkpoint["input_dim"],
            target_dim=checkpoint["output_dim"],
            context_dim=checkpoint["context_dim"],
            hidden_dim=checkpoint["context_encoder_hidden_dim"],
            num_layers=checkpoint["context_encoder_num_layers"],
        )
    else:
        model_cls = NormalizedContextResidualMLP if checkpoint.get("uses_io_normalization", False) else ContextResidualMLP
        model = model_cls(
            input_dim=checkpoint["input_dim"], output_dim=checkpoint["output_dim"],
            hidden_dim=checkpoint["hidden_dim"], num_layers=checkpoint["num_layers"],
            context_dim=checkpoint["context_dim"], context_injection=checkpoint.get("context_injection", "adapter"),
            modulation_scale=float(checkpoint.get("modulation_scale", 0.2)),
        )
        encoder_cls = NormalizedSupportSetEncoder if checkpoint.get("uses_io_normalization", False) else SupportSetEncoder
        encoder = encoder_cls(
            feature_dim=checkpoint["input_dim"], target_dim=checkpoint["output_dim"],
            context_dim=checkpoint["context_dim"], hidden_dim=checkpoint["context_encoder_hidden_dim"],
            num_layers=checkpoint["context_encoder_num_layers"],
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.reset_context()
    encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
    encoder.eval()
    return model, encoder


def initialize_tracking_solver(method: str, env, checkpoint_path, hidden_dim, num_layers, n_horizon, t_horizon):
    checkpoint = context_encoder = None
    if method == "nominal":
        model = BodyRateNominalDynamics(env).model()
        with working_directory(MODULE_DIR):
            solver = MPC(model, n_horizon, t_horizon).solver
        return model, solver, METHOD_LABELS[method], None, None, None, None, checkpoint, context_encoder, False

    if method == "meta":
        ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Meta checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if checkpoint.get("command_mode") != COMMAND_MODE_BODY_RATE_THRUST:
            raise ValueError(f"Checkpoint command_mode={checkpoint.get('command_mode')} is not {COMMAND_MODE_BODY_RATE_THRUST}.")
        residual_mlp, context_encoder = build_meta_components_from_checkpoint(checkpoint)
        method_label = METHOD_LABELS[method]
        amortized_context_model = True
        residual_mpc_scale = META_RESIDUAL_MPC_SCALE
        residual_optimizer = None
    else:
        residual_mlp = MLP(input_dim=MODEL_INPUT_DIM, output_dim=RESIDUAL_OUTPUT_DIM, hidden_dim=hidden_dim, num_layers=num_layers)
        method_label = METHOD_LABELS[method]
        amortized_context_model = False
        residual_mpc_scale = LIGHTMLP_RESIDUAL_MPC_SCALE
        residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=DEFAULT_LIGHTMLP_ADAPTATION_LR)

    for param in residual_mlp.parameters():
        param.requires_grad = False
    l4c_residual = l4c.L4CasADi(residual_mlp, name=DEFAULT_L4C_MODEL_NAME, build_dir=DEFAULT_L4C_BUILD_DIR.as_posix(), mutable=True)
    residual_criterion = nn.SmoothL1Loss(beta=1.0)
    residual_limits = META_RESIDUAL_MPC_LIMITS if method == "meta" else None
    model = BodyRateLearnedDynamics(env, l4c_residual, residual_scale=residual_mpc_scale, residual_limits=residual_limits).model()
    with working_directory(MODULE_DIR):
        solver = MPC(
            model,
            n_horizon,
            t_horizon,
            external_shared_lib_dir=l4c_residual.shared_lib_dir,
            external_shared_lib_name=l4c_residual.name,
        ).solver
    return model, solver, method_label, residual_mlp, l4c_residual, residual_optimizer, residual_criterion, checkpoint, context_encoder, amortized_context_model


def refine_context_from_support(residual_mlp, context, features, targets, steps, step_size, criterion):
    """Perform the deployed latent-context inner loop on a support buffer."""
    refined = context.detach()
    for _ in range(max(0, int(steps))):
        refined = refined.detach().requires_grad_(True)
        prediction = residual_mlp.forward_with_context(features, refined)
        support_loss = criterion(prediction, targets)
        gradient, = torch.autograd.grad(support_loss, refined)
        refined = (refined - float(step_size) * gradient).detach()
    return refined


def _body_to_world_torch(state):
    phi, theta, psi = state[..., 6], state[..., 7], state[..., 8]
    cp, sp, ct, st, cy, sy = torch.cos(phi), torch.sin(phi), torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
    return torch.stack((
        torch.stack((ct * cy, -cp * sy + sp * st * cy, sp * sy + cp * st * cy), dim=-1),
        torch.stack((ct * sy, cp * cy + sp * st * sy, -sp * cy + cp * st * sy), dim=-1),
        torch.stack((-st, sp * ct, cp * ct), dim=-1),
    ), dim=-2)


def refine_latent_from_velocity_support(residual_mlp, context, features, next_states, steps, step_size):
    """Deployment inner loop: update only z from next-velocity prediction."""
    refined = context.detach()
    for _ in range(max(0, int(steps))):
        refined = refined.detach().requires_grad_(True)
        state, action = features[:, :12], features[:, 12:]
        phi, theta, psi = state[:, 6], state[:, 7], state[:, 8]
        cp, sp, ct, st, cy, sy = torch.cos(phi), torch.sin(phi), torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
        thrust = action[:, 3]
        m, g = 0.027 * 0.66, 9.8
        nominal_accel = torch.stack(((sp * sy + cp * st * cy) * thrust / m, (-sp * cy + cp * st * sy) * thrust / m, cp * ct * thrust / m - g), dim=-1)
        body_residual = residual_mlp.forward_with_context(features, refined)
        world_residual = torch.bmm(_body_to_world_torch(state), body_residual.unsqueeze(-1)).squeeze(-1)
        predicted_velocity = state[:, [1, 3, 5]] + (1.0 / DEFAULT_CTRL_FREQ) * (nominal_accel + world_residual)
        support_loss = ((predicted_velocity - next_states[:, [1, 3, 5]]) / 0.25).pow(2).mean()
        gradient, = torch.autograd.grad(support_loss, refined)
        refined = refined - float(step_size) * gradient
    return refined.detach()


def refine_fast_adapter_from_support(residual_mlp, context, features, targets, steps, step_size, criterion):
    adapter = torch.zeros(1, 2 * RESIDUAL_OUTPUT_DIM, dtype=features.dtype, device=features.device, requires_grad=True)
    for _ in range(max(0, int(steps))):
        prediction = residual_mlp.forward_with_context_and_adapter(features, context, adapter)
        support_loss = criterion(prediction, targets)
        gradient, = torch.autograd.grad(support_loss, adapter)
        adapter = (adapter - float(step_size) * gradient).detach().requires_grad_(True)
    return adapter.detach()


def maybe_adapt_residual_model(
    step_idx, adapt_every_steps, batch_size, feature_buffer, target_buffer,
    amortized_context_model, residual_mlp, context_encoder, residual_criterion,
    residual_optimizer, adaptation_steps, l4c_residual, context_state=None,
    context_ema_decay=0.0, context_warmup_updates=0, context_updates=0,
    context_refinement_lr=0.15, fast_adapter_steps=0, fast_adapter_lr=0.10,
):
    """Update the online residual model and return persistent context state."""
    if step_idx <= 0 or step_idx % adapt_every_steps != 0 or len(feature_buffer) < batch_size:
        return context_state, context_updates
    x_batch = torch.tensor(np.array(feature_buffer[-batch_size:]), dtype=torch.float32)
    y_batch = torch.tensor(np.array(target_buffer[-batch_size:]), dtype=torch.float32)
    if amortized_context_model:
        if getattr(residual_mlp, "online_latent_only", False):
            if context_state is None:
                context_state = residual_mlp.context_init.detach()
            context_state = refine_latent_from_velocity_support(
                residual_mlp, context_state, x_batch, y_batch, adaptation_steps,
                context_refinement_lr,
            )
            context_updates += 1
            residual_mlp.set_context(context_state)
            l4c_residual.update(residual_mlp)
            return context_state, context_updates
        with torch.no_grad():
            candidate = context_encoder(x_batch, y_batch).squeeze(0)
        if context_state is None or context_updates < context_warmup_updates:
            context_state = candidate
        else:
            context_state = context_ema_decay * context_state + (1.0 - context_ema_decay) * candidate
        context_updates += 1
        context_state = refine_context_from_support(
            residual_mlp, context_state, x_batch, y_batch, adaptation_steps,
            context_refinement_lr, residual_criterion,
        )
        residual_mlp.set_context(context_state)
        residual_mlp.set_fast_adapter(refine_fast_adapter_from_support(
            residual_mlp, context_state, x_batch, y_batch, fast_adapter_steps,
            fast_adapter_lr, residual_criterion,
        ))
    else:
        for param in residual_mlp.parameters():
            param.requires_grad = True
        for _ in range(adaptation_steps):
            residual_optimizer.zero_grad()
            loss = residual_criterion(residual_mlp(x_batch), y_batch)
            loss.backward()
            residual_optimizer.step()
        for param in residual_mlp.parameters():
            param.requires_grad = False
    l4c_residual.update(residual_mlp)
    return context_state, context_updates


def build_result_dataframe(t_grid_inputs, x_history, u_history, ref_history, method_name, seed, env, task, executed_motor_history=None):
    min_length = min(len(t_grid_inputs), len(x_history) - 1, len(ref_history), len(u_history))
    data = {
        "time": t_grid_inputs[:min_length],
        "seed": np.full(min_length, seed),
        "method": np.full(min_length, method_name),
        "mass": np.full(min_length, env.MASS),
        "ixx": np.full(min_length, env.J[0, 0]),
        "iyy": np.full(min_length, env.J[1, 1]),
        "izz": np.full(min_length, env.J[2, 2]),
        "nominal_rate_controller_inertia": np.full(min_length, USE_NOMINAL_RATE_CONTROLLER_INERTIA),
    }
    for key, value in task.context_values().items():
        data[key] = np.full(min_length, value)
    for idx, col in enumerate(STATE_COLS):
        data[col] = x_history[:min_length, idx]
    for idx, col in enumerate(CONTROL_COLS):
        data[col] = u_history[:min_length, idx]
    if executed_motor_history is not None:
        for idx, col in enumerate(MOTOR_CONTROL_COLS):
            data[col] = executed_motor_history[:min_length, idx]
    for idx, col in enumerate([f"{name}_ref" for name in STATE_COLS]):
        data[col] = ref_history[:min_length, idx]
    return pd.DataFrame(data)


def position_rmse_interval(x_history, ref_history, dt: float, start_sec: float, end_sec: float | None) -> float | None:
    start_idx = int(np.floor(start_sec / dt))
    end_idx = len(ref_history) if end_sec is None else int(np.ceil(end_sec / dt))
    end_idx = min(end_idx, len(ref_history), len(x_history) - 1)
    if start_idx >= end_idx:
        return None
    position_error = x_history[start_idx:end_idx, [0, 2, 4]] - ref_history[start_idx:end_idx, [0, 2, 4]]
    return float(np.sqrt(np.mean(position_error ** 2)))


def run_tracking(
    method: str,
    seed: int = DEFAULT_RUN_SEED,
    gui: bool = DEFAULT_GUI,
    save_flag: bool = DEFAULT_SAVE_RESULTS,
    save_plot: bool = True,
    show_plot_window: bool = DEFAULT_SHOW_PLOT_WINDOW,
    export_animation_flag: bool = DEFAULT_EXPORT_ANIMATION,
    animation_format: str = DEFAULT_ANIMATION_FORMAT,
    checkpoint_path: Path | None = None,
    hidden_dim: int = DEFAULT_RESIDUAL_HIDDEN_DIM,
    num_layers: int = DEFAULT_RESIDUAL_NUM_LAYERS,
    batch_size: int | None = None,
    adaptation_steps: int | None = None,
    adaptation_interval_sec: float | None = None,
    t_horizon: float = DEFAULT_T_HORIZON,
    n_horizon: int = DEFAULT_N_HORIZON,
    sim_time: float = DEFAULT_SIM_TIME,
    reference_cfg: ReferenceConfig | None = None,
    task: DynamicsTask | None = None,
    results_basename: str | None = None,
    results_dir: Path | None = None,
    rmse_warmup_sec: float | None = DEFAULT_RMSE_WARMUP_SEC,
    init_state=None,
):
    seed_everything(seed)
    method = method.lower()
    task = task or DynamicsTask(mass_ratio=1.0, ixx_ratio=1.0, iyy_ratio=1.0, izz_ratio=1.0)
    reference_cfg = reference_cfg or ReferenceConfig(period=sim_time)
    if init_state is None:
        init_state = reference_state(0.0, reference_cfg)
    env = Quadrotor(**make_env_config(seed, gui, False, sim_time, task=task, init_state=init_state))
    obs, _ = env.reset()
    xt = wrap_state_angles(np.array(obs[:12], dtype=float))
    dt = 1.0 / env.CTRL_FREQ
    steps = int(sim_time / dt)
    model, solver, method_label, residual_mlp, l4c_residual, residual_optimizer, residual_criterion, checkpoint, context_encoder, amortized_context_model = initialize_tracking_solver(
        method, env, checkpoint_path, hidden_dim, num_layers, n_horizon, t_horizon
    )
    batch_size, adaptation_steps, adaptation_interval_sec = resolve_online_adaptation_config(method, dt, checkpoint, batch_size, adaptation_steps, adaptation_interval_sec)
    nominal_func = cs.Function("f_bodyrate_nominal", [model.x, model.u], [model.f_nominal])
    x_history = [xt.copy()]
    u_history = []
    executed_motor_history = []
    ref_history = []
    opt_times = []
    feature_buffer = []
    target_buffer = []
    adapt_every_steps = max(1, int(adaptation_interval_sec / dt))
    context_state = None
    context_ema_decay = float(checkpoint.get("online_context_ema_decay", 0.0)) if amortized_context_model else 0.0
    context_warmup_updates = int(checkpoint.get("online_context_warmup_updates", 0)) if amortized_context_model else 0
    context_refinement_lr = float(checkpoint.get("online_context_refinement_lr", 0.15)) if amortized_context_model else 0.0
    fast_adapter_steps = int(checkpoint.get("online_fast_adapter_steps", 0)) if amortized_context_model else 0
    fast_adapter_lr = float(checkpoint.get("online_fast_adapter_lr", 0.10)) if amortized_context_model else 0.0
    context_updates = 0
    u_ref = input_reference(env)

    for step_idx in range(steps):
        current_time = step_idx * dt
        set_tracking_references(solver, current_time, n_horizon, t_horizon, reference_cfg, u_ref, ref_history)
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        prev_xt = xt.copy()
        start = time.time()
        status = solver.solve()
        opt_times.append(time.time() - start)
        if status != 0:
            raise RuntimeError(f"acados returned status {status} at step {step_idx}")
        ut = np.array(solver.get(0, "u"), dtype=float)
        u_history.append(ut.copy())
        next_obs, _, done, info = step_bodyrate_env(env, prev_xt, ut, task)
        executed_motor_history.append(info["executed_motor_forces"])
        xt = wrap_state_angles(np.array(next_obs[:12], dtype=float))
        x_history.append(xt.copy())

        if method != "nominal":
            feature_buffer.append(np.concatenate([prev_xt, ut]))
            # The latent-only model adapts from velocity transitions, avoiding
            # a noisy online acceleration finite difference.  Keep the legacy
            # residual target path only for the LightMLP baseline.
            target_buffer.append(xt.copy() if amortized_context_model and getattr(residual_mlp, "online_latent_only", False) else finite_difference_targets(prev_xt, xt, ut, nominal_func, dt))
            context_state, context_updates = maybe_adapt_residual_model(
                step_idx, adapt_every_steps, batch_size, feature_buffer, target_buffer,
                amortized_context_model, residual_mlp, context_encoder, residual_criterion,
                residual_optimizer, adaptation_steps, l4c_residual, context_state,
                context_ema_decay, context_warmup_updates, context_updates,
                context_refinement_lr, fast_adapter_steps, fast_adapter_lr,
            )
        if done:
            print(f"Episode ended early at step {step_idx}.")
            break

    x_history = np.array(x_history)
    u_history = np.array(u_history)
    executed_motor_history = np.array(executed_motor_history)
    ref_history = np.array(ref_history[: len(u_history)])
    t_grid_states = np.linspace(0.0, dt * (len(x_history) - 1), len(x_history))
    t_grid_inputs = np.linspace(0.0, dt * (len(u_history) - 1), len(u_history))
    full_pos_rmse = position_rmse(x_history, ref_history, 0)
    warmup_steps = None
    post_warmup_pos_rmse = None
    if rmse_warmup_sec is not None:
        warmup_steps = int(np.ceil(rmse_warmup_sec / dt))
        if warmup_steps < len(ref_history):
            post_warmup_pos_rmse = position_rmse(x_history, ref_history, warmup_steps)
    early_rmse_0_0p5 = position_rmse_interval(x_history, ref_history, dt, 0.0, 0.5)
    early_rmse_0p5_1 = position_rmse_interval(x_history, ref_history, dt, 0.5, 1.0)
    early_rmse_1_2 = position_rmse_interval(x_history, ref_history, dt, 1.0, 2.0)
    late_rmse_2_end = position_rmse_interval(x_history, ref_history, dt, 2.0, None)
    mean_solve_time_ms = 1000.0 * float(np.mean(opt_times))
    print(
        f"{method_label}: RMSE={full_pos_rmse:.4f} m, post-warmup={post_warmup_pos_rmse}, "
        f"early0-0.5={early_rmse_0_0p5}, solve={mean_solve_time_ms:.1f} ms"
    )

    fig = plot_tracking_results(x_history, ref_history, t_grid_states, t_grid_inputs, method_label)
    results = {
        "position_rmse": full_pos_rmse,
        "full_position_rmse": full_pos_rmse,
        "post_warmup_position_rmse": post_warmup_pos_rmse,
        "rmse_0_0p5_sec": early_rmse_0_0p5,
        "rmse_0p5_1_sec": early_rmse_0p5_1,
        "rmse_1_2_sec": early_rmse_1_2,
        "rmse_2_end_sec": late_rmse_2_end,
        "rmse_warmup_sec": rmse_warmup_sec,
        "rmse_warmup_steps": warmup_steps,
        "mean_solve_time_ms": mean_solve_time_ms,
        "csv_path": None,
        "plot_path": None,
        "animation_path": None,
    }
    if save_flag:
        output_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = results_basename or method_label
        csv_path = output_dir / f"{base_name}_seed{seed}.csv"
        build_result_dataframe(t_grid_inputs, x_history, u_history, ref_history, method_label, seed, env, task, executed_motor_history).to_csv(csv_path, index=False)
        results["csv_path"] = csv_path
        if save_plot:
            plot_path = output_dir / f"{base_name}_seed{seed}.png"
            fig.savefig(plot_path, dpi=300)
            results["plot_path"] = plot_path
        if export_animation_flag:
            animation_path = output_dir / f"{base_name}_seed{seed}.{animation_format}"
            export_3d_animation(x_history[:-1], ref_history, animation_path, dt=dt, fps=DEFAULT_ANIMATION_FPS)
            results["animation_path"] = animation_path
    if show_plot_window:
        plt.show()
    else:
        plt.close(fig)
    env.close()
    return results
