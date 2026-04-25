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


TRACKING_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TRACKING_DIR.parents[2]
SAFE_CONTROL_GYM_ROOT = WORKSPACE_ROOT / "safe-control-gym"
if str(SAFE_CONTROL_GYM_ROOT) not in sys.path and SAFE_CONTROL_GYM_ROOT.exists():
    sys.path.insert(0, str(SAFE_CONTROL_GYM_ROOT))

import l4casadi as l4c  # noqa: E402
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor  # noqa: E402
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType  # noqa: E402


COST = "LINEAR_LS"
BASE_PARAMS = {
    "M": 0.027,
    "Ixx": 1.4e-5,
    "Iyy": 1.4e-5,
    "Izz": 2.17e-5,
}
NOMINAL_RATIOS = {
    "M": 0.66,
    "Ixx": 0.8,
    "Iyy": 0.8,
    "Izz": 0.8,
}

# Frequently edited experiment defaults live here so new runs do not require
# digging through the control loop, data collection, or plotting code.
META_DATASET_DIR_NAME = "meta_dataset_quadrotor3D"
META_DATASET_FILENAME = "0425_meta_dataset.csv"
META_CHECKPOINT_FILENAME = "0425_init_3_128.pth"
META_FINAL_CHECKPOINT_FILENAME = "0425_final_3_128.pth"
RESULTS_DIR_NAME = "results-0425"
ACADOS_EXPORT_DIR_NAME = "c_generated_code"
L4C_BUILD_DIR_NAME = "_l4c_generated"

DEFAULT_META_DATASET_DIR = TRACKING_DIR / META_DATASET_DIR_NAME
DEFAULT_META_DATASET_PATH = DEFAULT_META_DATASET_DIR / META_DATASET_FILENAME
DEFAULT_META_CHECKPOINT_PATH = TRACKING_DIR / "MetaLearning" / META_CHECKPOINT_FILENAME
DEFAULT_META_FINAL_CHECKPOINT_PATH = TRACKING_DIR / "MetaLearning" / META_FINAL_CHECKPOINT_FILENAME
DEFAULT_RESULTS_DIR = TRACKING_DIR / RESULTS_DIR_NAME
DEFAULT_ACADOS_EXPORT_DIR = TRACKING_DIR / ACADOS_EXPORT_DIR_NAME
DEFAULT_L4C_BUILD_DIR = TRACKING_DIR / L4C_BUILD_DIR_NAME

DEFAULT_RUN_SEED = 42
DEFAULT_SCRIPT_SEED = 1
DEFAULT_GUI = False
DEFAULT_SAVE_RESULTS = True
DEFAULT_SHOW_PLOT_WINDOW = False
DEFAULT_EXPORT_ANIMATION = False
DEFAULT_ANIMATION_FORMAT = "gif"
DEFAULT_ANIMATION_FPS = 20
DEFAULT_ANIMATION_AXIS_PAD = 0.15
DEFAULT_ANIMATION_VIEW_ELEV = 24
DEFAULT_ANIMATION_VIEW_AZIM = 35
DEFAULT_T_HORIZON = 1.0
DEFAULT_N_HORIZON = 20
DEFAULT_SIM_TIME = 12.0
DEFAULT_RMSE_WARMUP_SEC = 2.0
DEFAULT_RESIDUAL_HIDDEN_DIM = 128
DEFAULT_RESIDUAL_NUM_LAYERS = 3
DEFAULT_LIGHTMLP_ADAPTATION_LR = 1e-3
DEFAULT_LIGHTMLP_ADAPTATION_BATCH_SIZE = 96
DEFAULT_LIGHTMLP_ADAPTATION_STEPS = 10
DEFAULT_LIGHTMLP_ADAPTATION_INTERVAL_SEC = 1.0
DEFAULT_META_CONTEXT_INNER_LR = 5e-2
DEFAULT_ONLINE_CONTEXT_EMA_DECAY = 0.0
DEFAULT_ONLINE_CONTEXT_WARMUP_UPDATES = 1

DEFAULT_CTRL_FREQ = 50
DEFAULT_PYB_FREQ = 50
DEFAULT_STABILIZATION_GOAL = (0.0, 0.0, 1.0)
DEFAULT_STABILIZATION_GOAL_TOLERANCE = 0.05
DEFAULT_EVAL_INERTIAL_RATIOS = {
    "M": 1.0,
    "Ixx": 0.8,
    "Iyy": 0.8,
    "Izz": 0.8,
}
DEFAULT_EVAL_INERTIAL_PROP = [
    BASE_PARAMS["M"] * DEFAULT_EVAL_INERTIAL_RATIOS["M"],
    BASE_PARAMS["Ixx"] * DEFAULT_EVAL_INERTIAL_RATIOS["Ixx"],
    BASE_PARAMS["Iyy"] * DEFAULT_EVAL_INERTIAL_RATIOS["Iyy"],
    BASE_PARAMS["Izz"] * DEFAULT_EVAL_INERTIAL_RATIOS["Izz"],
]

MPC_STATE_WEIGHT_DIAG = np.array([8.0, 0.4, 8.0, 0.4, 12.0, 0.6, 1.5, 1.5, 1.0, 0.15, 0.15, 0.2], dtype=float)
MPC_CONTROL_WEIGHT = 0.05
DEFAULT_L4C_MODEL_NAME = "residual_quadrotor3D"
COMMAND_MODE_MOTOR_THRUST = "motor_thrust"
STATE_COLS = ["x", "x_dot", "y", "y_dot", "z", "z_dot", "phi", "theta", "psi", "p", "q", "r"]
CONTROL_COLS = ["u1", "u2", "u3", "u4"]
MOTOR_CONTROL_COLS = ["motor_u1", "motor_u2", "motor_u3", "motor_u4"]
TRACKED_DERIVATIVE_INDICES = np.array([1, 3, 5, 9, 10, 11], dtype=int)
METHOD_LABELS = {"nominal": "nominal", "meta": "metamlp", "lightmlp": "lightmlp"}
MODEL_INPUT_DIM = len(STATE_COLS) + len(CONTROL_COLS)
RESIDUAL_OUTPUT_DIM = len(TRACKED_DERIVATIVE_INDICES)


@dataclass
class ReferenceConfig:
    period: float = 12.0
    radius: float = 0.6
    center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    z_amp: float = 0.15
    yaw_ref: float = 0.0
    traj_type: str = "circle"
    y_radius: float | None = None


class MLP(nn.Module):
    def __init__(self, input_dim=16, output_dim=6, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ContextResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim=16,
        output_dim=6,
        hidden_dim=128,
        num_layers=3,
        context_dim=6,
        context_injection="concat",
        modulation_scale=0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.context_injection = context_injection
        self.modulation_scale = modulation_scale
        self.context_init = nn.Parameter(torch.zeros(context_dim))
        self.context = nn.Parameter(torch.zeros(context_dim), requires_grad=False)
        self.net = None
        self.input_layer = None
        self.hidden_layers = None
        self.pre_head = None
        self.context_scale_layer = None
        self.context_shift_layer = None
        self.output_layer = None

        if context_injection == "concat":
            layers = [nn.Linear(input_dim + context_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)
        elif context_injection == "adapter":
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, num_layers - 1))])
            self.pre_head = nn.Linear(hidden_dim, hidden_dim)
            self.context_scale_layer = nn.Linear(context_dim, hidden_dim)
            self.context_shift_layer = nn.Linear(context_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported context injection mode: {context_injection}")

    def _expand_context(self, x, context):
        if context.dim() == 1:
            return context.unsqueeze(0).expand(x.shape[0], -1)
        if context.dim() == 2 and context.shape[0] == 1 and x.shape[0] != 1:
            return context.expand(x.shape[0], -1)
        return context

    def forward_with_context(self, x, context):
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        context = self._expand_context(x, context.to(dtype=x.dtype, device=x.device))
        if self.context_injection == "concat":
            output = self.net(torch.cat([x, context], dim=-1))
        else:
            hidden = torch.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                hidden = torch.relu(layer(hidden))
            hidden = torch.relu(self.pre_head(hidden))
            gamma = 1.0 + self.modulation_scale * torch.tanh(self.context_scale_layer(context))
            beta = self.modulation_scale * self.context_shift_layer(context)
            hidden = torch.relu(gamma * hidden + beta)
            output = self.output_layer(hidden)
        return output.squeeze(0) if squeeze_output else output

    def forward(self, x):
        return self.forward_with_context(x, self.context)

    @torch.no_grad()
    def set_context(self, context):
        self.context.copy_(context.detach().to(dtype=self.context.dtype, device=self.context.device))

    @torch.no_grad()
    def reset_context(self):
        self.context.copy_(self.context_init.detach())


class SupportSetEncoder(nn.Module):
    def __init__(self, feature_dim=16, target_dim=6, context_dim=4, hidden_dim=96, num_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = [nn.Linear(feature_dim + target_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.sample_net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, features, targets):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        encoded = self.sample_net(torch.cat([features, targets], dim=-1))
        pooled = torch.cat(
            [
                encoded.mean(dim=0, keepdim=True),
                encoded.std(dim=0, keepdim=True, unbiased=False),
            ],
            dim=-1,
        )
        return self.head(pooled)


@contextmanager
def working_directory(path: Path):
    """Temporarily switch cwd so generated artifacts stay inside the tracking folder."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_meta_checkpoint_path(checkpoint_path: Path | None = None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    if DEFAULT_META_CHECKPOINT_PATH.exists():
        return DEFAULT_META_CHECKPOINT_PATH
    return DEFAULT_META_FINAL_CHECKPOINT_PATH


def resolve_online_adaptation_config(
    method: str,
    dt: float,
    checkpoint: dict | None,
    batch_size: int | None,
    adaptation_steps: int | None,
    adaptation_interval_sec: float | None,
) -> tuple[int, int, float]:
    if method == "meta" and checkpoint is not None:
        model_type = checkpoint.get("model_type", "maml")
        support_window = int(checkpoint.get("support_window", 32))
        inner_steps = 1 if model_type == "amortized_context_meta" else int(checkpoint.get("inner_steps", 1))
        default_interval_sec = max(1.0, support_window * dt) if model_type == "amortized_context_meta" else max(dt, support_window * dt)
        return (
            support_window if batch_size is None else batch_size,
            inner_steps if adaptation_steps is None else adaptation_steps,
            default_interval_sec if adaptation_interval_sec is None else adaptation_interval_sec,
        )

    return (
        DEFAULT_LIGHTMLP_ADAPTATION_BATCH_SIZE if batch_size is None else batch_size,
        DEFAULT_LIGHTMLP_ADAPTATION_STEPS if adaptation_steps is None else adaptation_steps,
        DEFAULT_LIGHTMLP_ADAPTATION_INTERVAL_SEC if adaptation_interval_sec is None else adaptation_interval_sec,
    )


def build_meta_components_from_checkpoint(checkpoint: dict):
    model_type = checkpoint.get("model_type", "maml")
    context_injection = checkpoint.get("context_injection", "concat")
    modulation_scale = float(checkpoint.get("modulation_scale", 0.25))
    if model_type in {"context_meta", "amortized_context_meta"}:
        model = ContextResidualMLP(
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            context_dim=checkpoint["context_dim"],
            context_injection=context_injection,
            modulation_scale=modulation_scale,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.reset_context()
        if model_type == "context_meta":
            return model, None, model_type
        context_encoder = SupportSetEncoder(
            feature_dim=checkpoint["input_dim"],
            target_dim=checkpoint["output_dim"],
            context_dim=checkpoint["context_dim"],
            hidden_dim=checkpoint["context_encoder_hidden_dim"],
            num_layers=checkpoint["context_encoder_num_layers"],
        )
        context_encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
        context_encoder.eval()
        return model, context_encoder, model_type

    model = MLP(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, None, model_type


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_state_angles(state: np.ndarray) -> np.ndarray:
    wrapped = np.array(state, dtype=float).copy()
    wrapped[6] = wrap_angle(wrapped[6])
    wrapped[7] = wrap_angle(wrapped[7])
    wrapped[8] = wrap_angle(wrapped[8])
    return wrapped


def reference_state(t: float, cfg: ReferenceConfig) -> np.ndarray:
    omega = 2.0 * np.pi / cfg.period
    phase = omega * t
    if cfg.traj_type == "circle":
        x_ref = cfg.center[0] + cfg.radius * np.cos(phase)
        y_ref = cfg.center[1] + cfg.radius * np.sin(phase)
        x_dot_ref = -cfg.radius * omega * np.sin(phase)
        y_dot_ref = cfg.radius * omega * np.cos(phase)
    elif cfg.traj_type == "figure8":
        y_radius = cfg.radius if cfg.y_radius is None else cfg.y_radius
        x_ref = cfg.center[0] + cfg.radius * np.sin(phase)
        y_ref = cfg.center[1] + 0.5 * y_radius * np.sin(2.0 * phase)
        x_dot_ref = cfg.radius * omega * np.cos(phase)
        y_dot_ref = y_radius * omega * np.cos(2.0 * phase)
    else:
        raise ValueError(f"Unsupported trajectory type: {cfg.traj_type}")
    z_ref = cfg.center[2] + cfg.z_amp * np.sin(0.5 * omega * t)
    z_dot_ref = cfg.z_amp * 0.5 * omega * np.cos(0.5 * omega * t)
    return np.array([
        x_ref,
        x_dot_ref,
        y_ref,
        y_dot_ref,
        z_ref,
        z_dot_ref,
        0.0,
        0.0,
        cfg.yaw_ref,
        0.0,
        0.0,
        0.0,
    ])


def default_init_randomization() -> dict:
    return {
        "init_x": {"distrib": "uniform", "low": -0.2, "high": 0.2},
        "init_x_dot": {"distrib": "uniform", "low": -0.05, "high": 0.05},
        "init_y": {"distrib": "uniform", "low": -0.2, "high": 0.2},
        "init_y_dot": {"distrib": "uniform", "low": -0.05, "high": 0.05},
        "init_z": {"distrib": "uniform", "low": 0.8, "high": 1.2},
        "init_z_dot": {"distrib": "uniform", "low": -0.05, "high": 0.05},
        "init_phi": {"distrib": "uniform", "low": -0.08, "high": 0.08},
        "init_theta": {"distrib": "uniform", "low": -0.08, "high": 0.08},
        "init_psi": {"distrib": "uniform", "low": -0.15, "high": 0.15},
        "init_p": {"distrib": "uniform", "low": -0.05, "high": 0.05},
        "init_q": {"distrib": "uniform", "low": -0.05, "high": 0.05},
        "init_r": {"distrib": "uniform", "low": -0.05, "high": 0.05},
    }


def make_env_config(
    seed: int,
    gui: bool,
    done_on_out_of_bound: bool,
    episode_len_sec: float,
    inertial_prop=None,
    init_state_randomization_info=None,
    init_state=None,
) -> dict:
    env_config = {
        "gui": gui,
        "ctrl_freq": DEFAULT_CTRL_FREQ,
        "pyb_freq": DEFAULT_PYB_FREQ,
        "quad_type": QuadType.THREE_D,
        "seed": seed,
        "done_on_out_of_bound": done_on_out_of_bound,
        "episode_len_sec": episode_len_sec,
        "task_info": {
            "stabilization_goal": list(DEFAULT_STABILIZATION_GOAL),
            "stabilization_goal_tolerance": DEFAULT_STABILIZATION_GOAL_TOLERANCE,
        },
        "init_state_randomization_info": init_state_randomization_info or default_init_randomization(),
    }
    if inertial_prop is not None:
        env_config["inertial_prop"] = inertial_prop
    if init_state is not None:
        env_config["init_state"] = init_state
    return env_config


def nominal_params() -> dict:
    return {name: BASE_PARAMS[name] * NOMINAL_RATIOS[name] for name in BASE_PARAMS}


def control_labels() -> list[str]:
    return CONTROL_COLS.copy()


def motor_thrust_bounds(gym_env) -> tuple[np.ndarray, np.ndarray]:
    a_low = gym_env.KF * (gym_env.PWM2RPM_SCALE * gym_env.MIN_PWM + gym_env.PWM2RPM_CONST) ** 2
    a_high = gym_env.KF * (gym_env.PWM2RPM_SCALE * gym_env.MAX_PWM + gym_env.PWM2RPM_CONST) ** 2
    return a_low * np.ones(4), a_high * np.ones(4)


def control_bounds(gym_env) -> tuple[np.ndarray, np.ndarray]:
    return motor_thrust_bounds(gym_env)


def input_reference() -> np.ndarray:
    return np.zeros(4, dtype=float)


def nominal_dynamics_terms(gym_env, model_name: str):
    params = nominal_params()
    m = params["M"]
    ixx = params["Ixx"]
    iyy = params["Iyy"]
    izz = params["Izz"]
    g = gym_env.GRAVITY_ACC
    length = gym_env.L
    gamma = gym_env.KM / gym_env.KF

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

    f1 = cs.MX.sym("f1")
    f2 = cs.MX.sym("f2")
    f3 = cs.MX.sym("f3")
    f4 = cs.MX.sym("f4")

    x_state = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)
    u = cs.vertcat(f1, f2, f3, f4)
    xdot = cs.MX.sym("xdot", 12)
    j = cs.diag(cs.vertcat(ixx, iyy, izz))
    j_inv = cs.diag(cs.vertcat(1.0 / ixx, 1.0 / iyy, 1.0 / izz))

    c_phi, s_phi = cs.cos(phi), cs.sin(phi)
    c_theta, s_theta = cs.cos(theta), cs.sin(theta)
    c_psi, s_psi = cs.cos(psi), cs.sin(psi)
    rot = cs.vertcat(
        cs.horzcat(c_theta * c_psi, -c_phi * s_psi + s_phi * s_theta * c_psi, s_phi * s_psi + c_phi * s_theta * c_psi),
        cs.horzcat(c_theta * s_psi, c_phi * c_psi + s_phi * s_theta * s_psi, -s_phi * c_psi + c_phi * s_theta * s_psi),
        cs.horzcat(-s_theta, s_phi * c_theta, c_phi * c_theta),
    )

    total_thrust = f1 + f2 + f3 + f4
    pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
    pos_ddot = rot @ cs.vertcat(0, 0, total_thrust) / m - cs.vertcat(0, 0, g)
    body_rates = cs.vertcat(p_body, q_body, r_body)
    body_torque = cs.vertcat(
        length / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
        length / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
        gamma * (-f1 + f2 - f3 + f4),
    )
    body_rates_dot = j_inv @ (body_torque - cs.cross(body_rates, j @ body_rates))
    euler_rates = cs.vertcat(
        p_body + q_body * s_phi * cs.tan(theta) + r_body * c_phi * cs.tan(theta),
        q_body * c_phi - r_body * s_phi,
        q_body * s_phi / c_theta + r_body * c_phi / c_theta,
    )
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
    u_min, u_max = control_bounds(gym_env)

    model = cs.types.SimpleNamespace()
    model.x = x_state
    model.xdot = xdot
    model.u = u
    model.u_min = u_min
    model.u_max = u_max
    model.z = cs.vertcat([])
    model.p = cs.vertcat([])
    model.f_nominal = f_nominal
    model.x_start = np.zeros(12)
    model.constraints = cs.vertcat([])
    model.name = model_name
    return model


class Quadrotor3DNominalDynamics:
    def __init__(self, gym_env):
        self.gym_env = gym_env

    def model(self):
        model = nominal_dynamics_terms(self.gym_env, "quadrotor3D_nominal")
        model.f_expl = model.f_nominal
        return model


class Quadrotor3DLearnedDynamics:
    def __init__(self, gym_env, residual_model):
        self.gym_env = gym_env
        self.residual_model = residual_model

    def model(self):
        model = nominal_dynamics_terms(self.gym_env, "quadrotor3D_learned")
        residual = self.residual_model(cs.vertcat(model.x, model.u).T).T
        residual_term = cs.vertcat(
            0,
            residual[0],
            0,
            residual[1],
            0,
            residual[2],
            0,
            0,
            0,
            residual[3],
            residual[4],
            residual[5],
        )
        model.f_expl = model.f_nominal + residual_term
        return model


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
        # acados code generation is not process-safe when multiple workers share
        # the same JSON/export paths, so isolate artifacts per PID.
        process_dir = DEFAULT_ACADOS_EXPORT_DIR / f"{self.model.name}_pid{os.getpid()}"
        json_path = process_dir / f"{self.model.name}_ocp.json"
        return process_dir, json_path

    def ocp(self):
        model_ac = self.acados_model(self.model)
        nx = 12
        nu = 4
        ny = nx + nu
        ny_e = nx

        ocp = AcadosOcp()
        ocp.model = model_ac
        code_export_dir, json_path = self._codegen_paths()
        try:
            ocp.code_export_directory = code_export_dir.as_posix()
        except AttributeError:
            # Older acados_template builds may not expose this attribute.
            pass
        try:
            ocp.code_gen_opts.code_export_directory = code_export_dir.as_posix()
            ocp.code_gen_opts.json_file = json_path.as_posix()
        except AttributeError:
            # Keep compatibility with older acados_template versions that only
            # expose the deprecated properties.
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
        ocp.cost.Vz = np.array([[]])

        q = np.diag(MPC_STATE_WEIGHT_DIAG)
        r = MPC_CONTROL_WEIGHT * np.eye(nu)
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


def build_result_dataframe(
    t_grid_inputs,
    x_history,
    u_history,
    ref_history,
    method_name,
    seed,
    env,
    action_labels: list[str],
    executed_motor_history=None,
) -> pd.DataFrame:
    min_length = min(len(t_grid_inputs), len(x_history) - 1, len(ref_history), len(u_history))
    ref_cols = [f"{name}_ref" for name in STATE_COLS]

    data = {
        "time": t_grid_inputs[:min_length],
        "seed": np.full(min_length, seed),
        "method": np.full(min_length, method_name),
        "mass": np.full(min_length, env.MASS),
        "ixx": np.full(min_length, env.J[0, 0]),
        "iyy": np.full(min_length, env.J[1, 1]),
        "izz": np.full(min_length, env.J[2, 2]),
    }
    for idx, col in enumerate(STATE_COLS):
        data[col] = x_history[:min_length, idx]
    for idx, col in enumerate(action_labels):
        data[col] = u_history[:min_length, idx]
    if executed_motor_history is not None:
        for idx, col in enumerate(MOTOR_CONTROL_COLS):
            data[col] = executed_motor_history[:min_length, idx]
    for idx, col in enumerate(ref_cols):
        data[col] = ref_history[:min_length, idx]
    return pd.DataFrame(data)


def plot_tracking_results(x_history, ref_history, t_grid_states, t_grid_inputs, method_name):
    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(x_history[:, 0], x_history[:, 2], x_history[:, 4], label="Trajectory", linewidth=2)
    ax3d.plot(ref_history[:, 0], ref_history[:, 2], ref_history[:, 4], "--", label="Reference", linewidth=2)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title(f"{method_name} 3D Tracking")
    ax3d.legend()

    ax_pos = fig.add_subplot(2, 2, 2)
    ax_pos.plot(t_grid_states, x_history[:, 0], label="x", linewidth=2)
    ax_pos.plot(t_grid_states, x_history[:, 2], label="y", linewidth=2)
    ax_pos.plot(t_grid_states, x_history[:, 4], label="z", linewidth=2)
    ax_pos.plot(t_grid_inputs, ref_history[:, 0], "--", label="x_ref", alpha=0.8)
    ax_pos.plot(t_grid_inputs, ref_history[:, 2], "--", label="y_ref", alpha=0.8)
    ax_pos.plot(t_grid_inputs, ref_history[:, 4], "--", label="z_ref", alpha=0.8)
    ax_pos.set_xlabel("Time [s]")
    ax_pos.set_ylabel("Position [m]")
    ax_pos.set_title("Position Tracking")
    ax_pos.grid(True)
    ax_pos.legend(ncol=2)

    ax_att = fig.add_subplot(2, 2, 3)
    ax_att.plot(t_grid_states, x_history[:, 6], label="phi", linewidth=2)
    ax_att.plot(t_grid_states, x_history[:, 7], label="theta", linewidth=2)
    ax_att.plot(t_grid_states, x_history[:, 8], label="psi", linewidth=2)
    ax_att.plot(t_grid_inputs, ref_history[:, 8], "--", label="psi_ref", alpha=0.8)
    ax_att.set_xlabel("Time [s]")
    ax_att.set_ylabel("Angle [rad]")
    ax_att.set_title("Attitude")
    ax_att.grid(True)
    ax_att.legend()

    ax_rates = fig.add_subplot(2, 2, 4)
    ax_rates.plot(t_grid_states, x_history[:, 9], label="p", linewidth=2)
    ax_rates.plot(t_grid_states, x_history[:, 10], label="q", linewidth=2)
    ax_rates.plot(t_grid_states, x_history[:, 11], label="r", linewidth=2)
    ax_rates.set_xlabel("Time [s]")
    ax_rates.set_ylabel("Rate [rad/s]")
    ax_rates.set_title("Body Rates")
    ax_rates.grid(True)
    ax_rates.legend()

    fig.tight_layout()
    return fig


def export_3d_animation(x_history, ref_history, output_path: Path, dt: float, fps: int = DEFAULT_ANIMATION_FPS):
    from matplotlib import animation

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_x = np.concatenate([x_history[:, 0], ref_history[:, 0]])
    all_y = np.concatenate([x_history[:, 2], ref_history[:, 2]])
    all_z = np.concatenate([x_history[:, 4], ref_history[:, 4]])
    pad = DEFAULT_ANIMATION_AXIS_PAD
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_zlim(max(0.0, all_z.min() - pad), all_z.max() + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Quadrotor 3D Tracking Animation")
    ax.view_init(elev=DEFAULT_ANIMATION_VIEW_ELEV, azim=DEFAULT_ANIMATION_VIEW_AZIM)

    ref_line, = ax.plot(ref_history[:, 0], ref_history[:, 2], ref_history[:, 4], "--", color="C0", linewidth=2, label="Reference")
    traj_line, = ax.plot([], [], [], color="C1", linewidth=2, label="Trajectory")
    drone_point, = ax.plot([], [], [], "o", color="C3", markersize=6, label="Quadrotor")
    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        drone_point.set_data([], [])
        drone_point.set_3d_properties([])
        time_text.set_text("")
        return traj_line, drone_point, ref_line, time_text

    def update(frame):
        traj_line.set_data(x_history[: frame + 1, 0], x_history[: frame + 1, 2])
        traj_line.set_3d_properties(x_history[: frame + 1, 4])
        drone_point.set_data([x_history[frame, 0]], [x_history[frame, 2]])
        drone_point.set_3d_properties([x_history[frame, 4]])
        time_text.set_text(f"t = {frame * dt:.2f} s")
        return traj_line, drone_point, ref_line, time_text

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(x_history), interval=1000 / fps, blit=False)
    output_path = Path(output_path)
    writer = animation.PillowWriter(fps=fps) if output_path.suffix.lower() == ".gif" else animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)


def finite_difference_targets(prev_state, next_state, action, nominal_func, dt: float):
    true_dyn = (next_state[TRACKED_DERIVATIVE_INDICES] - prev_state[TRACKED_DERIVATIVE_INDICES]) / dt
    nominal = nominal_func(prev_state, action).full().flatten()[TRACKED_DERIVATIVE_INDICES]
    return true_dyn - nominal


def set_tracking_references(solver, current_time: float, n_horizon: int, t_horizon: float, reference_cfg: ReferenceConfig, input_ref: np.ndarray, ref_history: list) -> None:
    for k in range(n_horizon):
        x_ref_k = reference_state(current_time + k * t_horizon / n_horizon, reference_cfg)
        if k == 0:
            ref_history.append(x_ref_k.copy())
        solver.set(k, "yref", np.concatenate([x_ref_k, input_ref]))
    solver.set(n_horizon, "yref", reference_state(current_time + t_horizon, reference_cfg))


def initialize_tracking_solver(
    method: str,
    env,
    checkpoint_path,
    hidden_dim: int,
    num_layers: int,
    meta_online_context_ema_decay: float | None,
    meta_online_context_warmup_updates: int | None,
    n_horizon: int,
    t_horizon: float,
):
    residual_mlp = l4c_residual = residual_optimizer = residual_criterion = checkpoint = context_encoder = None
    context_inner_lr = None
    context_model = amortized_context_model = False
    online_context_ema_decay = DEFAULT_ONLINE_CONTEXT_EMA_DECAY
    online_context_warmup_updates = DEFAULT_ONLINE_CONTEXT_WARMUP_UPDATES
    method_label = METHOD_LABELS[method]

    if method == "nominal":
        model = Quadrotor3DNominalDynamics(env).model()
        with working_directory(TRACKING_DIR):
            solver = MPC(model=model, n_horizon=n_horizon, t_horizon=t_horizon).solver
        return {
            "model": model,
            "solver": solver,
            "method_label": method_label,
            "residual_mlp": residual_mlp,
            "l4c_residual": l4c_residual,
            "residual_optimizer": residual_optimizer,
            "residual_criterion": residual_criterion,
            "checkpoint": checkpoint,
            "context_encoder": context_encoder,
            "context_inner_lr": context_inner_lr,
            "context_model": context_model,
            "amortized_context_model": amortized_context_model,
            "online_context_ema_decay": online_context_ema_decay,
            "online_context_warmup_updates": online_context_warmup_updates,
        }

    if method == "meta":
        ckpt_path = resolve_meta_checkpoint_path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Meta checkpoint not found: {ckpt_path}")
        print(f"Loading meta checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if checkpoint.get("command_mode", COMMAND_MODE_MOTOR_THRUST) != COMMAND_MODE_MOTOR_THRUST:
            raise ValueError(f"Checkpoint command_mode={checkpoint.get('command_mode')} is incompatible with direct motor thrust control.")
        residual_mlp, context_encoder, model_type = build_meta_components_from_checkpoint(checkpoint)
        context_model = isinstance(residual_mlp, ContextResidualMLP)
        amortized_context_model = model_type == "amortized_context_meta"
        if context_model:
            method_label = "metacontext"
            context_inner_lr = float(checkpoint.get("inner_lr", DEFAULT_META_CONTEXT_INNER_LR))
            online_context_ema_decay = float(checkpoint.get("online_context_ema_decay", DEFAULT_ONLINE_CONTEXT_EMA_DECAY)) if meta_online_context_ema_decay is None else float(meta_online_context_ema_decay)
            online_context_warmup_updates = int(checkpoint.get("online_context_warmup_updates", DEFAULT_ONLINE_CONTEXT_WARMUP_UPDATES)) if meta_online_context_warmup_updates is None else int(meta_online_context_warmup_updates)
    else:
        residual_mlp = MLP(input_dim=MODEL_INPUT_DIM, output_dim=RESIDUAL_OUTPUT_DIM, hidden_dim=hidden_dim, num_layers=num_layers)

    for param in residual_mlp.parameters():
        param.requires_grad = False
    l4c_residual = l4c.L4CasADi(residual_mlp, name=DEFAULT_L4C_MODEL_NAME, build_dir=DEFAULT_L4C_BUILD_DIR.as_posix(), mutable=True)
    if not context_model or not amortized_context_model:
        residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=DEFAULT_LIGHTMLP_ADAPTATION_LR)
    residual_criterion = nn.MSELoss()
    model = Quadrotor3DLearnedDynamics(env, l4c_residual).model()
    with working_directory(TRACKING_DIR):
        solver = MPC(
            model=model,
            n_horizon=n_horizon,
            t_horizon=t_horizon,
            external_shared_lib_dir=l4c_residual.shared_lib_dir,
            external_shared_lib_name=l4c_residual.name,
        ).solver
    return {
        "model": model,
        "solver": solver,
        "method_label": method_label,
        "residual_mlp": residual_mlp,
        "l4c_residual": l4c_residual,
        "residual_optimizer": residual_optimizer,
        "residual_criterion": residual_criterion,
        "checkpoint": checkpoint,
        "context_encoder": context_encoder,
        "context_inner_lr": context_inner_lr,
        "context_model": context_model,
        "amortized_context_model": amortized_context_model,
        "online_context_ema_decay": online_context_ema_decay,
        "online_context_warmup_updates": online_context_warmup_updates,
    }


def maybe_adapt_residual_model(
    step_idx: int,
    adapt_every_steps: int,
    batch_size: int,
    feature_buffer: list,
    target_buffer: list,
    amortized_context_model: bool,
    context_model: bool,
    residual_mlp,
    context_encoder,
    residual_criterion,
    residual_optimizer,
    adaptation_steps: int,
    context_inner_lr: float | None,
    online_context_ema_decay: float,
    online_context_warmup_updates: int,
    context_update_count: int,
    l4c_residual,
):
    if step_idx <= 0 or step_idx % adapt_every_steps != 0 or len(feature_buffer) < batch_size:
        return context_update_count

    x_batch = torch.tensor(np.array(feature_buffer[-batch_size:]), dtype=torch.float32)
    y_batch = torch.tensor(np.array(target_buffer[-batch_size:]), dtype=torch.float32)
    if amortized_context_model:
        with torch.no_grad():
            context = context_encoder(x_batch, y_batch).squeeze(0)
            if context_update_count >= online_context_warmup_updates:
                context = online_context_ema_decay * residual_mlp.context.detach() + (1.0 - online_context_ema_decay) * context
        residual_mlp.set_context(context)
        context_update_count += 1
    elif context_model:
        context = residual_mlp.context.detach().clone().requires_grad_(True)
        for _ in range(adaptation_steps):
            loss = residual_criterion(residual_mlp.forward_with_context(x_batch, context), y_batch)
            grad, = torch.autograd.grad(loss, context)
            context = context - context_inner_lr * grad
        residual_mlp.set_context(context)
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
    return context_update_count


def save_tracking_artifacts(
    save_flag: bool,
    results_dir,
    results_basename,
    method_label: str,
    seed: int,
    t_grid_inputs,
    x_history,
    u_history,
    ref_history,
    env,
    action_labels,
    fig,
    export_animation_flag: bool,
    animation_format: str,
    dt: float,
    results: dict,
):
    if not save_flag:
        return
    output_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = results_basename or method_label
    csv_path = output_dir / f"{base_name}_seed{seed}.csv"
    plot_path = output_dir / f"{base_name}_seed{seed}.png"
    build_result_dataframe(t_grid_inputs, x_history, u_history, ref_history, method_label, seed, env, action_labels=action_labels).to_csv(csv_path, index=False)
    fig.savefig(plot_path, dpi=300)
    print(f"Saved trajectory to {csv_path}")
    print(f"Saved plot to {plot_path}")
    results["csv_path"] = csv_path
    results["plot_path"] = plot_path
    if export_animation_flag:
        animation_path = output_dir / f"{base_name}_seed{seed}.{animation_format}"
        export_3d_animation(x_history[:-1], ref_history, animation_path, dt=dt, fps=DEFAULT_ANIMATION_FPS)
        print(f"Saved animation to {animation_path}")
        results["animation_path"] = animation_path


def position_rmse(x_history, ref_history, start_idx: int = 0) -> float:
    if start_idx >= len(ref_history):
        raise ValueError("start_idx is beyond the available trajectory horizon.")
    position_error = x_history[start_idx:-1, [0, 2, 4]] - ref_history[start_idx:, [0, 2, 4]]
    return float(np.sqrt(np.mean(position_error ** 2)))


def run_tracking(
    method: str,
    seed: int = DEFAULT_RUN_SEED,
    gui: bool = DEFAULT_GUI,
    save_flag: bool = DEFAULT_SAVE_RESULTS,
    show_plot_window: bool = DEFAULT_SHOW_PLOT_WINDOW,
    export_animation_flag: bool = DEFAULT_EXPORT_ANIMATION,
    animation_format: str = DEFAULT_ANIMATION_FORMAT,
    checkpoint_path: Path | None = None,
    hidden_dim: int = DEFAULT_RESIDUAL_HIDDEN_DIM,
    num_layers: int = DEFAULT_RESIDUAL_NUM_LAYERS,
    batch_size: int | None = None,
    adaptation_steps: int | None = None,
    adaptation_interval_sec: float | None = None,
    meta_online_context_ema_decay: float | None = None,
    meta_online_context_warmup_updates: int | None = None,
    t_horizon: float = DEFAULT_T_HORIZON,
    n_horizon: int = DEFAULT_N_HORIZON,
    sim_time: float = DEFAULT_SIM_TIME,
    reference_cfg: ReferenceConfig | None = None,
    inertial_prop=None,
    results_basename: str | None = None,
    results_dir: Path | None = None,
    rmse_warmup_sec: float | None = DEFAULT_RMSE_WARMUP_SEC,
    init_state=None,
):
    seed_everything(seed)
    method = method.lower()
    reference_cfg = reference_cfg or ReferenceConfig(period=sim_time)
    if inertial_prop is None:
        inertial_prop = list(DEFAULT_EVAL_INERTIAL_PROP)
    env = Quadrotor(**make_env_config(seed, gui, False, sim_time, inertial_prop=inertial_prop, init_state=init_state))
    obs, _ = env.reset()
    xt = wrap_state_angles(np.array(obs[:12], dtype=float))
    dt = 1.0 / env.CTRL_FREQ
    steps = int(sim_time / dt)
    action_labels = control_labels()
    input_ref = input_reference()
    controller = initialize_tracking_solver(
        method,
        env,
        checkpoint_path,
        hidden_dim,
        num_layers,
        meta_online_context_ema_decay,
        meta_online_context_warmup_updates,
        n_horizon,
        t_horizon,
    )
    model = controller["model"]
    solver = controller["solver"]
    method_label = controller["method_label"]
    residual_mlp = controller["residual_mlp"]
    l4c_residual = controller["l4c_residual"]
    residual_optimizer = controller["residual_optimizer"]
    residual_criterion = controller["residual_criterion"]
    checkpoint = controller["checkpoint"]
    context_encoder = controller["context_encoder"]
    context_inner_lr = controller["context_inner_lr"]
    context_model = controller["context_model"]
    amortized_context_model = controller["amortized_context_model"]
    online_context_ema_decay = controller["online_context_ema_decay"]
    online_context_warmup_updates = controller["online_context_warmup_updates"]
    context_update_count = 0

    batch_size, adaptation_steps, adaptation_interval_sec = resolve_online_adaptation_config(
        method=method,
        dt=dt,
        checkpoint=checkpoint,
        batch_size=batch_size,
        adaptation_steps=adaptation_steps,
        adaptation_interval_sec=adaptation_interval_sec,
    )
    if method != "nominal":
        if amortized_context_model:
            print(
                "Online encoded-context config: "
                f"batch_size={batch_size}, "
                f"interval={adaptation_interval_sec:.2f}s, "
                f"context_dim={residual_mlp.context_dim}, "
                f"ema={online_context_ema_decay:.2f}, "
                f"warmup_updates={online_context_warmup_updates}"
            )
        elif context_model:
            print(
                "Online latent adaptation config: "
                f"batch_size={batch_size}, "
                f"steps={adaptation_steps}, "
                f"interval={adaptation_interval_sec:.2f}s, "
                f"context_dim={residual_mlp.context_dim}, "
                f"inner_lr={context_inner_lr:.4f}"
            )
        else:
            print(
                "Online adaptation config: "
                f"batch_size={batch_size}, "
                f"steps={adaptation_steps}, "
                f"interval={adaptation_interval_sec:.2f}s"
            )

    nominal_func = cs.Function("f_nominal", [model.x, model.u], [model.f_nominal])
    x_history = [xt.copy()]
    u_history = []
    ref_history = []
    opt_times = []
    feature_buffer = []
    target_buffer = []
    adapt_every_steps = max(1, int(adaptation_interval_sec / dt))

    for step_idx in range(steps):
        current_time = step_idx * dt
        set_tracking_references(solver, current_time, n_horizon, t_horizon, reference_cfg, input_ref, ref_history)
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)

        prev_xt = xt.copy()
        start = time.time()
        status = solver.solve()
        opt_times.append(time.time() - start)
        if status != 0:
            raise RuntimeError(f"acados returned non-zero status {status} at step {step_idx}")

        ut = np.array(solver.get(0, "u"), dtype=float)
        u_history.append(ut.copy())
        next_obs, _, done, _ = env.step(ut)
        xt = wrap_state_angles(np.array(next_obs[:12], dtype=float))
        x_history.append(xt.copy())

        if method != "nominal":
            feature_buffer.append(np.concatenate([prev_xt, ut]))
            target_buffer.append(finite_difference_targets(prev_xt, xt, ut, nominal_func, dt))
            context_update_count = maybe_adapt_residual_model(
                step_idx,
                adapt_every_steps,
                batch_size,
                feature_buffer,
                target_buffer,
                amortized_context_model,
                context_model,
                residual_mlp,
                context_encoder,
                residual_criterion,
                residual_optimizer,
                adaptation_steps,
                context_inner_lr,
                online_context_ema_decay,
                online_context_warmup_updates,
                context_update_count,
                l4c_residual,
            )

        if done:
            print(f"Episode ended early at step {step_idx}.")
            break

    x_history = np.array(x_history)
    u_history = np.array(u_history)
    ref_history = np.array(ref_history[: len(u_history)])
    t_grid_states = np.linspace(0.0, dt * (len(x_history) - 1), len(x_history))
    t_grid_inputs = np.linspace(0.0, dt * (len(u_history) - 1), len(u_history))
    full_pos_rmse = position_rmse(x_history, ref_history, start_idx=0)
    post_warmup_pos_rmse = None
    warmup_steps = None
    if rmse_warmup_sec is not None:
        warmup_steps = int(np.ceil(rmse_warmup_sec / dt))
        if warmup_steps < len(ref_history):
            post_warmup_pos_rmse = position_rmse(x_history, ref_history, start_idx=warmup_steps)
    mean_solve_time_ms = 1000 * np.mean(opt_times)
    print(f"Mean MPC solve time: {mean_solve_time_ms:.1f} ms -- {1 / np.mean(opt_times):.1f} Hz")
    print(f"Full Position RMSE: {full_pos_rmse:.4f} m")
    if post_warmup_pos_rmse is not None:
        print(f"Post-warmup Position RMSE ({rmse_warmup_sec:.1f}s+): {post_warmup_pos_rmse:.4f} m")

    fig = plot_tracking_results(x_history, ref_history, t_grid_states, t_grid_inputs, method_label)
    results = {
        "position_rmse": full_pos_rmse,
        "full_position_rmse": full_pos_rmse,
        "post_warmup_position_rmse": post_warmup_pos_rmse,
        "rmse_warmup_sec": rmse_warmup_sec,
        "rmse_warmup_steps": warmup_steps,
        "mean_solve_time_ms": float(mean_solve_time_ms),
        "csv_path": None,
        "plot_path": None,
        "animation_path": None,
    }

    save_tracking_artifacts(
        save_flag,
        results_dir,
        results_basename,
        method_label,
        seed,
        t_grid_inputs,
        x_history,
        u_history,
        ref_history,
        env,
        action_labels,
        fig,
        export_animation_flag,
        animation_format,
        dt,
        results,
    )

    if show_plot_window:
        plt.show()
    else:
        plt.close(fig)

    env.close()
    return results
