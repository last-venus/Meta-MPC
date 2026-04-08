"""为 3D 四旋翼残差动力学收集离线元学习数据。

这个脚本的目标是为后续的 MAML 训练生成数据集，整体流程如下：

1. 先定义很多个“任务”。
   每个任务由一组不同的惯性参数和一条不同的参考轨迹组成。
2. 对每个任务，用 nominal MPC 在仿真器里运行。
3. 在每个控制时刻，记录：
   - 当前状态 x
   - 当前控制输入 u
   - 仿真器中真实观测到的加速度/角加速度
   - 名义模型对这些量的预测
   - 两者之差 residual = true - nominal
4. 最后把所有样本存成一个 CSV，供后续元学习脚本按 task_id 分组使用。

后面要训练的残差模型形式为：
    输入: [状态(12维), 控制(4维)] -> 共 16 维
    输出: [x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot] 的残差 -> 共 6 维
"""

import sys
from pathlib import Path

import casadi as cs
import numpy as np
import pandas as pd


TRACKING_DIR = Path(__file__).resolve().parents[1]
if str(TRACKING_DIR) not in sys.path:
    # 让当前脚本可以直接导入 Quadrotor_3D_Tracking/ 下的公共工具模块。
    sys.path.insert(0, str(TRACKING_DIR))

from quadrotor3D_common import ReferenceConfig, Quadrotor3DNominalDynamics, MPC, make_env_config, reference_state  # noqa: E402
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor  # noqa: E402


np.random.seed(43)
OUTPUT_DIR = TRACKING_DIR / "meta_dataset_quadrotor3D"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = OUTPUT_DIR / "quadrotor3d_meta_residual_mpc.csv"

# 下面这些比例网格定义了元训练阶段会看到的动力学分布。
# 每一种比例组合都对应一类不同的系统参数。
MASS_RATIOS = np.round(np.linspace(0.75, 1.75, 5), 3)
IXX_RATIOS = np.round(np.linspace(0.8, 1.2, 3), 3)
IYY_RATIOS = np.round(np.linspace(0.8, 1.2, 3), 3)
IZZ_RATIOS = np.round(np.linspace(0.8, 1.2, 3), 3)

# 每个 ReferenceConfig 描述一类参考轨迹任务。
# 把“动力学变化”和“轨迹变化”同时放进任务分布里，可以让元学习不只适配一种情形。
REFERENCE_TASKS = [
    ReferenceConfig(period=12.0, radius=0.55, center=(0.0, 0.0, 1.0), z_amp=0.12, yaw_ref=0.0),
    ReferenceConfig(period=12.0, radius=0.45, center=(0.25, -0.25, 1.1), z_amp=0.18, yaw_ref=0.2),
    ReferenceConfig(period=12.0, radius=0.65, center=(-0.3, 0.2, 0.95), z_amp=0.10, yaw_ref=-0.2),
]

# BASE_* 表示控制器内部 nominal model 默认使用的参数。
# 仿真环境会使用它们的缩放版本，从而制造 model mismatch。
BASE_MASS = 0.027 * 0.66
BASE_IXX = 1.4e-5 * 0.8
BASE_IYY = 1.4e-5 * 0.8
BASE_IZZ = 2.17e-5 * 0.8
DT = 0.02
T = 12.0
STEPS = int(T / DT)
N = 20
T_HORIZON = 1.0
EPISODES_PER_TASK = 1


def collect_episode(task_id, episode_id, inertial_prop, ref_cfg):
    """为单个任务采集一整条 rollout 的监督数据。

    Args:
        task_id: 当前元任务的编号。
        episode_id: 当前任务下的第几个 episode。
        inertial_prop: 仿真环境使用的真实惯性参数 [mass, Ixx, Iyy, Izz]。
        ref_cfg: 当前任务对应的参考轨迹配置。

    Returns:
        一个 Python 列表，列表中的每个元素对应 CSV 里的一行样本。
    """
    # 用当前任务对应的“真实惯性参数”创建仿真环境。
    env = Quadrotor(**make_env_config(seed=task_id * 100 + episode_id, gui=False, done_on_out_of_bound=False, episode_len_sec=T, inertial_prop=inertial_prop))

    # 构造 MPC 内部使用的 nominal dynamics。
    # 这里故意不和真实环境完全一致，这样系统里才会存在可学习的 residual。
    model = Quadrotor3DNominalDynamics(env).model()

    # 把名义动力学封装成可调用的 CasADi 函数，后面打标签时要用它来计算 nominal 预测。
    nominal_func = cs.Function("f_nom", [model.x, model.u], [model.f_nominal])
    solver = MPC(model=model, n_horizon=N, t_horizon=T_HORIZON).solver

    obs, _ = env.reset()
    # 这里只取前 12 维，也就是 3D 四旋翼状态：
    # [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    x = np.array(obs[:12], dtype=float)
    rows = []

    for step in range(STEPS):
        current_time = step * DT

        # 给整个 MPC horizon 填入未来参考轨迹。
        for k in range(N):
            x_ref_k = reference_state(current_time + k * T_HORIZON / N, ref_cfg)
            # 对 LINEAR_LS 代价来说，acados 的 yref 形式是 [state_ref, control_ref]。
            # 这里控制参考设成 0，所以在状态参考后拼接 4 个 0。
            solver.set(k, "yref", np.concatenate([x_ref_k, np.zeros(4)]))

        # 终端时刻的状态参考。
        solver.set(N, "yref", reference_state(current_time + T_HORIZON, ref_cfg))

        # 标准 MPC 做法：把第一个 shooting node 固定为当前状态。
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        solver.solve()

        # 取优化序列里的第一个控制量，作用到真实仿真环境。
        u = np.array(solver.get(0, "u"), dtype=float)
        next_obs, _, _, _ = env.step(u)
        x_next = np.array(next_obs[:12], dtype=float)

        # 用相邻时刻状态差分来近似“真实”的二阶量。
        # 这里只学习平动加速度和机体系角速度导数的残差，
        # 不去学习像 x_dot 这种本来就是运动学恒等式的项。
        true_targets = np.array([
            (x_next[1] - x[1]) / DT,
            (x_next[3] - x[3]) / DT,
            (x_next[5] - x[5]) / DT,
            (x_next[9] - x[9]) / DT,
            (x_next[10] - x[10]) / DT,
            (x_next[11] - x[11]) / DT,
        ])

        # 计算 nominal model 在同一个 (x, u) 下的预测值。
        nominal_targets = nominal_func(x, u).full().flatten()[[1, 3, 5, 9, 10, 11]]

        # 真值减去名义预测，就是后面监督学习要拟合的 residual label。
        residual_targets = true_targets - nominal_targets

        # 保存一条监督学习样本，同时把任务元信息也存下来，
        # 这样后面可以按 task_id 重新组织成元学习任务。
        rows.append([
            task_id, episode_id, step * DT,
            inertial_prop[0] / BASE_MASS, inertial_prop[1] / BASE_IXX, inertial_prop[2] / BASE_IYY, inertial_prop[3] / BASE_IZZ,
            ref_cfg.center[0], ref_cfg.center[1], ref_cfg.center[2], ref_cfg.radius, ref_cfg.z_amp, ref_cfg.yaw_ref,
            *x, *u, *true_targets, *nominal_targets, *residual_targets,
            env.MASS, env.J[0, 0], env.J[1, 1], env.J[2, 2],
        ])

        # 状态向前推进，进入下一步采样。
        x = x_next

    env.close()
    return rows


def main():
    # all_rows 用来累计所有任务、所有 episode 产生的样本。
    all_rows = []
    task_id = 0

    # 下面这组嵌套循环就是在枚举任务分布。
    # 每一个唯一组合
    # (mass_ratio, ixx_ratio, iyy_ratio, izz_ratio, reference)
    # 都对应一个独立的 meta-task。
    for mass_ratio in MASS_RATIOS:
        for ixx_ratio in IXX_RATIOS:
            for iyy_ratio in IYY_RATIOS:
                for izz_ratio in IZZ_RATIOS:
                    inertial_prop = [BASE_MASS * mass_ratio, BASE_IXX * ixx_ratio, BASE_IYY * iyy_ratio, BASE_IZZ * izz_ratio]
                    for ref_cfg in REFERENCE_TASKS:
                        print(
                            f"[Task {task_id}] M={mass_ratio:.2f}, Ixx={ixx_ratio:.2f}, "
                            f"Iyy={iyy_ratio:.2f}, Izz={izz_ratio:.2f}, center={ref_cfg.center}"
                        )
                        for episode_id in range(EPISODES_PER_TASK):
                            all_rows.extend(collect_episode(task_id, episode_id, inertial_prop, ref_cfg))
                        task_id += 1

    # 这些列名定义了后续 Offline_Train_Meta.py 读取时依赖的数据格式。
    # 大致可以分成几部分：
    # - 任务元信息
    # - 当前状态 x
    # - 当前控制 u
    # - 仿真器真实导数标签
    # - 名义模型预测
    # - residual 标签 = true - nominal
    # - 环境真实惯性参数
    columns = [
        "task_id", "episode_id", "time", "mass_ratio", "ixx_ratio", "iyy_ratio", "izz_ratio",
        "center_x", "center_y", "center_z", "radius", "z_amp", "yaw_ref",
        "x", "x_dot", "y", "y_dot", "z", "z_dot", "phi", "theta", "psi", "p", "q", "r",
        "u1", "u2", "u3", "u4",
        "x_ddot_true", "y_ddot_true", "z_ddot_true", "p_dot_true", "q_dot_true", "r_dot_true",
        "x_ddot_nom", "y_ddot_nom", "z_ddot_nom", "p_dot_nom", "q_dot_nom", "r_dot_nom",
        "res_x_ddot", "res_y_ddot", "res_z_ddot", "res_p_dot", "res_q_dot", "res_r_dot",
        "mass", "ixx", "iyy", "izz",
    ]

    # 最后把所有任务的数据统一写成一个大 CSV。
    # 后面的元学习脚本会再按 task_id 把它们重新分组，并采样 support/query 集。
    pd.DataFrame(all_rows, columns=columns).to_csv(SAVE_PATH, index=False)
    print(f"Saved meta-dataset to {SAVE_PATH}")


if __name__ == "__main__":
    main()
