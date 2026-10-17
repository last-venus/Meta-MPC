# MetaResidual-MPC：无人机残差模型预测控制

本仓库基于 **Fast Online Adaptive Neural MPC via Meta-Learning**，保留三维四旋翼无人机的控制、数据采集、离线训练、在线适配和结果分析代码。

论文：Yu Mei, Xinyu Zhou, Shuyang Yu, Vaibhav Srivastava, Xiaobo Tan，
[Fast Online Adaptive Neural MPC via Meta-Learning](https://arxiv.org/abs/2504.16369)。

## 目录

| 目录 | 用途 |
| --- | --- |
| `Quadrotor_3D_Tracking/` | 以四个电机推力为输入的三维轨迹跟踪，以及 BodyRate 分支共用的模型和工具 |
| `Quadrotor_BodyRate_MetaResidual_MPC/` | 以机体系角速度和总推力为输入的残差 MPC、数据采集和元训练 |
| `third_party/l4casadi/` | PyTorch 残差网络与 CasADi 的接口，Git 子模块 |
| `third_party/safe-control-gym/` | 四旋翼仿真环境，Git 子模块 |
| `environment.yml` | Conda 环境配置 |

BodyRate 分支会导入 `Quadrotor_3D_Tracking/quadrotor3D_common.py`，需要同时保留这两个目录。

详细说明：

- [BodyRate 数据采集、训练与实验](Quadrotor_BodyRate_MetaResidual_MPC/README.md)
- [三维跟踪中文详解](Quadrotor_3D_Tracking/3D_TRACKING_中文详解.md)

## 安装依赖

在仓库根目录执行：

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate l4control
```

已有 `l4control` 环境时直接激活即可。安装 `l4casadi` 前需要先安装 PyTorch；原项目使用的组合为：

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -e third_party/l4casadi --no-build-isolation
pip install -e third_party/safe-control-gym
```

`safe-control-gym` 的依赖安装可能改变 PyTorch 版本；安装后检查环境，必要时重新安装上述版本。

### acados 使用外部安装

`third_party/` **没有 acados**。它是包含本机编译产物的外部求解器依赖，当前机器使用：

```text
/home/jingxiuya/czc_code/acados
```

保留这个目录。新机器请先构建 acados，再在已激活的 Conda 环境中安装 Python 接口：

```bash
export ACADOS_SOURCE_DIR=/home/jingxiuya/czc_code/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
```

其他机器应将 `ACADOS_SOURCE_DIR` 改为实际安装路径。

外部的 `/home/jingxiuya/czc_code/l4casadi` 源码副本已清理，后续安装、修改源码统一使用 `third_party/l4casadi/`。清理时确认当前 `l4control` 使用独立安装于 `site-packages` 的 `l4casadi`，无需因删除旧源码副本而重新安装。

外部的 `/home/jingxiuya/czc_code/safe-control-gym` 重复副本也已清理，`l4control` 环境的可编辑安装路径已迁移到 `third_party/safe-control-gym/`，并通过无界面三维四旋翼初始化和一步仿真验证。该子模块保留完整上游内容，以便更新和使用仿真环境。

## 运行无人机实验

以下命令均从仓库根目录开始，在 `l4control` 环境中执行。学习控制器需要对应目录中已有训练好的模型；没有模型时先运行该模块的采集、训练脚本。

### 三维电机推力控制

```bash
cd Quadrotor_3D_Tracking
python quadrotor3D_Nominal.py
python quadrotor3D_Lightmlp.py
python quadrotor3D_Meta.py
```

`quadrotor3D_Figure8_Eval.py` 提供八字轨迹对比，`*_seeds.py` 提供多随机种子实验。参数和数据路径集中在 `quadrotor3D_common.py`。

### BodyRate 控制与元学习

```bash
cd Quadrotor_BodyRate_MetaResidual_MPC
python quadrotor3D_BodyRate_Nominal.py
```

完整的数据采集、训练、对比实验和结果汇总见 [BodyRate README](Quadrotor_BodyRate_MetaResidual_MPC/README.md)。

## 清理范围与历史恢复

此次整理删除了 `Cartpole/`、`VanderPol/`、`Quadrotor_2D_Tracking/`、`Quadrotor_2D_Stabilization/` 及对应的展示图片，图片删空后移除了 `assets/`。现保留三维轨迹跟踪和 BodyRate 两个无人机模块，以及这些模块的现有数据、模型和结果。三维跟踪和 BodyRate 不依赖已删除的二维模块。没有改写 Git 历史，也没有删除保留模块的未提交修改。

`.gitignore` 允许跟踪无人机 Python 源码和 Markdown 说明，忽略 Python 缓存以及新增的部分实验产物。已有的已跟踪文件仍由 Git 管理。

删除内容可以从清理前的提交恢复。例如，在本次清理尚未提交时：

```bash
git restore --source=HEAD -- Cartpole VanderPol Quadrotor_2D_Tracking Quadrotor_2D_Stabilization assets
```

清理提交后，将 `HEAD` 换成清理前的提交号。忽略或未提交的数据、模型和结果需要另外备份，不能依靠 Git 历史恢复。

## 引用

```bibtex
@article{mei2025fast,
  title={Fast Online Adaptive Neural MPC via Meta-Learning},
  author={Mei, Yu and Zhou, Xinyu and Yu, Shuyang and Srivastava, Vaibhav and Tan, Xiaobo},
  journal={IFAC-PapersOnLine},
  volume={59},
  number={30},
  pages={377--382},
  year={2025},
  publisher={Elsevier}
}
```
