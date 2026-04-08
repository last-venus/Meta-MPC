import sys
import time
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
DEFAULT_META_DATASET_PATH = TRACKING_DIR / "meta_dataset_quadrotor3D" / "quadrotor3d_meta_residual_mpc.csv"
DEFAULT_META_CHECKPOINT_PATH = TRACKING_DIR / "MetaLearning" / "maml_quadrotor3d_meta_init_3_128.pth"
DEFAULT_RESULTS_DIR = TRACKING_DIR / "results"


@dataclass
class ReferenceConfig:
    period: float = 12.0
    radius: float = 0.6
    center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    z_amp: float = 0.15
    yaw_ref: float = 0.0


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


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    x_ref = cfg.center[0] + cfg.radius * np.cos(omega * t)
    y_ref = cfg.center[1] + cfg.radius * np.sin(omega * t)
    z_ref = cfg.center[2] + cfg.z_amp * np.sin(0.5 * omega * t)
    x_dot_ref = -cfg.radius * omega * np.sin(omega * t)
    y_dot_ref = cfg.radius * omega * np.cos(omega * t)
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
) -> dict:
    env_config = {
        "gui": gui,
        "ctrl_freq": 50,
        "pyb_freq": 50,
        "quad_type": QuadType.THREE_D,
        "seed": seed,
        "done_on_out_of_bound": done_on_out_of_bound,
        "episode_len_sec": episode_len_sec,
        "task_info": {
            "stabilization_goal": [0.0, 0.0, 1.0],
            "stabilization_goal_tolerance": 0.05,
        },
        "init_state_randomization_info": init_state_randomization_info or default_init_randomization(),
    }
    if inertial_prop is not None:
        env_config["inertial_prop"] = inertial_prop
    return env_config


def nominal_params() -> dict:
    return {name: BASE_PARAMS[name] * NOMINAL_RATIOS[name] for name in BASE_PARAMS}


def control_bounds(gym_env) -> tuple[np.ndarray, np.ndarray]:
    a_low = gym_env.KF * (gym_env.PWM2RPM_SCALE * gym_env.MIN_PWM + gym_env.PWM2RPM_CONST) ** 2
    a_high = gym_env.KF * (gym_env.PWM2RPM_SCALE * gym_env.MAX_PWM + gym_env.PWM2RPM_CONST) ** 2
    return a_low * np.ones(4), a_high * np.ones(4)


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
    body_torque = cs.vertcat(
        length / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
        length / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
        gamma * (-f1 + f2 - f3 + f4),
    )
    body_rates = cs.vertcat(p_body, q_body, r_body)
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

    def ocp(self):
        model_ac = self.acados_model(self.model)
        nx = 12
        nu = 4
        ny = nx + nu
        ny_e = nx

        ocp = AcadosOcp()
        ocp.model = model_ac
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

        q = np.diag([8.0, 0.4, 8.0, 0.4, 12.0, 0.6, 1.5, 1.5, 1.0, 0.15, 0.15, 0.2])
        r = 0.05 * np.eye(nu)
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


def build_result_dataframe(t_grid_inputs, x_history, u_history, ref_history, method_name, seed, env) -> pd.DataFrame:
    min_length = min(len(t_grid_inputs), len(x_history) - 1, len(ref_history), len(u_history))
    state_cols = ["x", "x_dot", "y", "y_dot", "z", "z_dot", "phi", "theta", "psi", "p", "q", "r"]
    ref_cols = [f"{name}_ref" for name in state_cols]
    action_cols = ["u1", "u2", "u3", "u4"]

    data = {
        "time": t_grid_inputs[:min_length],
        "seed": np.full(min_length, seed),
        "method": np.full(min_length, method_name),
        "mass": np.full(min_length, env.MASS),
        "ixx": np.full(min_length, env.J[0, 0]),
        "iyy": np.full(min_length, env.J[1, 1]),
        "izz": np.full(min_length, env.J[2, 2]),
    }
    for idx, col in enumerate(state_cols):
        data[col] = x_history[:min_length, idx]
    for idx, col in enumerate(action_cols):
        data[col] = u_history[:min_length, idx]
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


def export_3d_animation(x_history, ref_history, output_path: Path, dt: float, fps: int = 20):
    from matplotlib import animation

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_x = np.concatenate([x_history[:, 0], ref_history[:, 0]])
    all_y = np.concatenate([x_history[:, 2], ref_history[:, 2]])
    all_z = np.concatenate([x_history[:, 4], ref_history[:, 4]])
    pad = 0.15
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_zlim(max(0.0, all_z.min() - pad), all_z.max() + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Quadrotor 3D Tracking Animation")
    ax.view_init(elev=24, azim=35)

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
    true_dyn = np.array([
        (next_state[1] - prev_state[1]) / dt,
        (next_state[3] - prev_state[3]) / dt,
        (next_state[5] - prev_state[5]) / dt,
        (next_state[9] - prev_state[9]) / dt,
        (next_state[10] - prev_state[10]) / dt,
        (next_state[11] - prev_state[11]) / dt,
    ])
    nominal = nominal_func(prev_state, action).full().flatten()[[1, 3, 5, 9, 10, 11]]
    return true_dyn - nominal


def run_tracking(
    method: str,
    seed: int = 42,
    gui: bool = False,
    save_flag: bool = True,
    show_plot_window: bool = False,
    export_animation_flag: bool = False,
    animation_format: str = "gif",
    checkpoint_path: Path | None = None,
    hidden_dim: int = 128,
    num_layers: int = 3,
    batch_size: int = 64,
    adaptation_steps: int = 20,
    adaptation_interval_sec: float = 0.5,
    t_horizon: float = 1.0,
    n_horizon: int = 20,
    sim_time: float = 12.0,
    reference_cfg: ReferenceConfig | None = None,
    inertial_prop=None,
    results_basename: str | None = None,
):
    seed_everything(seed)
    method = method.lower()
    method_label = {"nominal": "nominal", "meta": "metamlp", "lightmlp": "lightmlp"}[method]
    reference_cfg = reference_cfg or ReferenceConfig(period=sim_time)

    env = Quadrotor(**make_env_config(seed, gui, False, sim_time, inertial_prop=inertial_prop))
    obs, _ = env.reset()
    xt = wrap_state_angles(np.array(obs[:12], dtype=float))
    dt = 1.0 / env.CTRL_FREQ
    steps = int(sim_time / dt)

    residual_mlp = None
    l4c_residual = None
    residual_optimizer = None
    residual_criterion = None

    if method == "nominal":
        model = Quadrotor3DNominalDynamics(env).model()
        solver = MPC(model=model, n_horizon=n_horizon, t_horizon=t_horizon).solver
    else:
        if method == "meta":
            ckpt_path = Path(checkpoint_path or DEFAULT_META_CHECKPOINT_PATH)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Meta checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            residual_mlp = MLP(
                input_dim=checkpoint["input_dim"],
                output_dim=checkpoint["output_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                num_layers=checkpoint["num_layers"],
            )
            residual_mlp.load_state_dict(checkpoint["model_state_dict"])
        else:
            residual_mlp = MLP(input_dim=16, output_dim=6, hidden_dim=hidden_dim, num_layers=num_layers)

        for param in residual_mlp.parameters():
            param.requires_grad = False
        l4c_residual = l4c.L4CasADi(residual_mlp, name="residual_quadrotor3D", mutable=True)
        residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=1e-3)
        residual_criterion = nn.MSELoss()
        model = Quadrotor3DLearnedDynamics(env, l4c_residual).model()
        solver = MPC(
            model=model,
            n_horizon=n_horizon,
            t_horizon=t_horizon,
            external_shared_lib_dir=l4c_residual.shared_lib_dir,
            external_shared_lib_name=l4c_residual.name,
        ).solver

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
        for k in range(n_horizon):
            x_ref_k = reference_state(current_time + k * t_horizon / n_horizon, reference_cfg)
            if k == 0:
                ref_history.append(x_ref_k.copy())
            solver.set(k, "yref", np.concatenate([x_ref_k, np.zeros(4)]))

        solver.set(n_horizon, "yref", reference_state(current_time + t_horizon, reference_cfg))
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
            if step_idx > 0 and step_idx % adapt_every_steps == 0 and len(feature_buffer) >= batch_size:
                x_batch = torch.tensor(np.array(feature_buffer[-batch_size:]), dtype=torch.float32)
                y_batch = torch.tensor(np.array(target_buffer[-batch_size:]), dtype=torch.float32)
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

        if done:
            print(f"Episode ended early at step {step_idx}.")
            break

    x_history = np.array(x_history)
    u_history = np.array(u_history)
    ref_history = np.array(ref_history[: len(u_history)])
    t_grid_states = np.linspace(0.0, dt * (len(x_history) - 1), len(x_history))
    t_grid_inputs = np.linspace(0.0, dt * (len(u_history) - 1), len(u_history))
    pos_rmse = np.sqrt(np.mean((x_history[:-1, [0, 2, 4]] - ref_history[:, [0, 2, 4]]) ** 2))
    print(f"Mean MPC solve time: {1000 * np.mean(opt_times):.1f} ms -- {1 / np.mean(opt_times):.1f} Hz")
    print(f"Position RMSE: {pos_rmse:.4f} m")

    fig = plot_tracking_results(x_history, ref_history, t_grid_states, t_grid_inputs, method_label)
    results = {"position_rmse": float(pos_rmse), "csv_path": None, "plot_path": None, "animation_path": None}

    if save_flag:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        base_name = results_basename or method_label
        csv_path = DEFAULT_RESULTS_DIR / f"{base_name}_seed{seed}.csv"
        plot_path = DEFAULT_RESULTS_DIR / f"{base_name}_seed{seed}.png"
        build_result_dataframe(t_grid_inputs, x_history, u_history, ref_history, method_label, seed, env).to_csv(csv_path, index=False)
        fig.savefig(plot_path, dpi=300)
        print(f"Saved trajectory to {csv_path}")
        print(f"Saved plot to {plot_path}")
        results["csv_path"] = csv_path
        results["plot_path"] = plot_path
        if export_animation_flag:
            animation_path = DEFAULT_RESULTS_DIR / f"{base_name}_seed{seed}.{animation_format}"
            export_3d_animation(x_history[:-1], ref_history, animation_path, dt=dt, fps=20)
            print(f"Saved animation to {animation_path}")
            results["animation_path"] = animation_path

    if show_plot_window:
        plt.show()
    else:
        plt.close(fig)

    env.close()
    return results
