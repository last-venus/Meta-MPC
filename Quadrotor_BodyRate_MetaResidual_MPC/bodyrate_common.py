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
SAFE_CONTROL_GYM_ROOT = WORKSPACE_ROOT / "safe-control-gym"
for path in (SAFE_CONTROL_GYM_ROOT, TRACKING_DIR):
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
TRACKED_DERIVATIVE_INDICES = np.array([1, 3, 5, 9, 10, 11], dtype=int)
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
DEFAULT_DATASET_PATH = DEFAULT_DATASET_DIR / "bodyrate_meta_dataset.csv"
DEFAULT_CHECKPOINT_PATH = MODULE_DIR / "MetaLearning" / "bodyrate_context_meta.pth"
DEFAULT_RESULTS_DIR = MODULE_DIR / "results"
DEFAULT_ACADOS_EXPORT_DIR = MODULE_DIR / "c_generated_code"
DEFAULT_L4C_BUILD_DIR = MODULE_DIR / "_l4c_generated"
DEFAULT_L4C_MODEL_NAME = "bodyrate_residual_quadrotor3D"
USE_NOMINAL_RATE_CONTROLLER_INERTIA = True
META_RESIDUAL_MPC_SCALE = 0.50
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
    def __init__(self, env, residual_model, residual_scale: float = 1.0):
        self.env = env
        self.residual_model = residual_model
        self.residual_scale = float(residual_scale)

    def model(self):
        model = nominal_dynamics_terms(self.env, "quadrotor3D_bodyrate_learned")
        residual = self.residual_model(cs.vertcat(model.x, model.u).T).T
        residual_term = cs.vertcat(0, residual[0], 0, residual[1], 0, residual[2], 0, 0, 0, residual[3], residual[4], residual[5])
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
    true_dyn = (next_state[TRACKED_DERIVATIVE_INDICES] - prev_state[TRACKED_DERIVATIVE_INDICES]) / dt
    nominal = nominal_func(prev_state, action).full().flatten()[TRACKED_DERIVATIVE_INDICES]
    return true_dyn - nominal


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
        default_interval_sec = max(1.0, support_window * dt)
        return (
            support_window if batch_size is None else batch_size,
            1 if adaptation_steps is None else adaptation_steps,
            default_interval_sec if adaptation_interval_sec is None else adaptation_interval_sec,
        )
    return (
        DEFAULT_LIGHTMLP_ADAPTATION_BATCH_SIZE if batch_size is None else batch_size,
        DEFAULT_LIGHTMLP_ADAPTATION_STEPS if adaptation_steps is None else adaptation_steps,
        DEFAULT_LIGHTMLP_ADAPTATION_INTERVAL_SEC if adaptation_interval_sec is None else adaptation_interval_sec,
    )


def build_meta_components_from_checkpoint(checkpoint: dict):
    model_cls = NormalizedContextResidualMLP if checkpoint.get("uses_io_normalization", False) else ContextResidualMLP
    model = model_cls(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        context_dim=checkpoint["context_dim"],
        context_injection=checkpoint.get("context_injection", "adapter"),
        modulation_scale=float(checkpoint.get("modulation_scale", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.reset_context()
    encoder_cls = NormalizedSupportSetEncoder if checkpoint.get("uses_io_normalization", False) else SupportSetEncoder
    encoder = encoder_cls(
        feature_dim=checkpoint["input_dim"],
        target_dim=checkpoint["output_dim"],
        context_dim=checkpoint["context_dim"],
        hidden_dim=checkpoint["context_encoder_hidden_dim"],
        num_layers=checkpoint["context_encoder_num_layers"],
    )
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
    model = BodyRateLearnedDynamics(env, l4c_residual, residual_scale=residual_mpc_scale).model()
    with working_directory(MODULE_DIR):
        solver = MPC(
            model,
            n_horizon,
            t_horizon,
            external_shared_lib_dir=l4c_residual.shared_lib_dir,
            external_shared_lib_name=l4c_residual.name,
        ).solver
    return model, solver, method_label, residual_mlp, l4c_residual, residual_optimizer, residual_criterion, checkpoint, context_encoder, amortized_context_model


def maybe_adapt_residual_model(step_idx, adapt_every_steps, batch_size, feature_buffer, target_buffer, amortized_context_model, residual_mlp, context_encoder, residual_criterion, residual_optimizer, adaptation_steps, l4c_residual):
    if step_idx <= 0 or step_idx % adapt_every_steps != 0 or len(feature_buffer) < batch_size:
        return
    x_batch = torch.tensor(np.array(feature_buffer[-batch_size:]), dtype=torch.float32)
    y_batch = torch.tensor(np.array(target_buffer[-batch_size:]), dtype=torch.float32)
    if amortized_context_model:
        with torch.no_grad():
            context = context_encoder(x_batch, y_batch).squeeze(0)
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
            target_buffer.append(finite_difference_targets(prev_xt, xt, ut, nominal_func, dt))
            maybe_adapt_residual_model(
                step_idx, adapt_every_steps, batch_size, feature_buffer, target_buffer,
                amortized_context_model, residual_mlp, context_encoder, residual_criterion,
                residual_optimizer, adaptation_steps, l4c_residual,
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
        plot_path = output_dir / f"{base_name}_seed{seed}.png"
        build_result_dataframe(t_grid_inputs, x_history, u_history, ref_history, method_label, seed, env, task, executed_motor_history).to_csv(csv_path, index=False)
        fig.savefig(plot_path, dpi=300)
        results["csv_path"] = csv_path
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
