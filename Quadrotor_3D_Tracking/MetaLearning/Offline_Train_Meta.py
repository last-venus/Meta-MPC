"""对 3D 四旋翼残差数据做离线元学习训练。

这个脚本实现的是一个简化版的 MAML 训练流程，目标不是直接学一个
"全局最优模型"，而是学到一个"好的初始化"：

1. 先从 CSV 中读取所有任务的数据。
2. 按 task_id 分组，每个 task 对应一种动力学/轨迹组合。
3. 每个 epoch 随机抽若干个 task。
4. 对每个 task：
   - 先用 support set 做一次内循环更新
   - 再用 query set 评估更新后的参数
5. 把多个 task 的 query loss 求平均，作为 meta loss。
6. 用这个 meta loss 更新模型初始参数。

最终保存下来的 .pth，就是后面在线 residual adaptation 的初始化权重。
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import functional_call


TRACKING_DIR = Path(__file__).resolve().parents[1]
if str(TRACKING_DIR) not in sys.path:
    # 让当前脚本可以直接导入 Quadrotor_3D_Tracking/ 下的公共模块。
    sys.path.insert(0, str(TRACKING_DIR))

from quadrotor3D_common import DEFAULT_META_CHECKPOINT_PATH, DEFAULT_META_DATASET_PATH, MLP  # noqa: E402


torch.manual_seed(43)


def load_task_data(csv_path: Path):
    """从 CSV 读取数据，并按 task_id 重组成元学习任务。"""
    df = pd.read_csv(csv_path)

    # tasks[task_id] 里先暂存该任务下的所有样本。
    tasks = defaultdict(list)
    for _, row in df.iterrows():
        tasks[row["task_id"]].append((
            row["x"], row["x_dot"], row["y"], row["y_dot"], row["z"], row["z_dot"],
            row["phi"], row["theta"], row["psi"], row["p"], row["q"], row["r"],
            row["u1"], row["u2"], row["u3"], row["u4"],
            row["res_x_ddot"], row["res_y_ddot"], row["res_z_ddot"], row["res_p_dot"], row["res_q_dot"], row["res_r_dot"],
        ))

    task_data = {}
    for task_id, samples in tasks.items():
        # 每一行样本前 16 维是输入：[state(12), control(4)]
        # 后 6 维是监督标签：[res_x_ddot, res_y_ddot, res_z_ddot, res_p_dot, res_q_dot, res_r_dot]
        data = torch.tensor(samples, dtype=torch.float32)
        task_data[task_id] = (data[:, :16], data[:, 16:])
    return task_data


def main():
    # 默认读取 DataCollection_Meta.py 生成的数据集。
    csv_path = DEFAULT_META_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Meta-dataset not found: {csv_path}")

    task_data = load_task_data(csv_path)
    print(f"Loaded data for {len(task_data)} different 3D tasks.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # MAML 超参数
    # -------------------------
    # meta_lr: 外循环学习率，用来更新“初始化参数”
    # inner_lr: 内循环学习率，用来做每个 task 的快速适配
    # meta_batch_size: 每个 epoch 采样多少个 task
    # inner_steps: 每个 task 的内循环更新次数
    # epochs: 总共训练多少个 epoch
    # k: support / query 各取多少条样本
    meta_lr = 1e-4
    inner_lr = 1e-3
    meta_batch_size = min(12, len(task_data))
    inner_steps = 1
    epochs = 30000
    k = 32

    # 模型结构：16 维输入，6 维输出。
    # hidden_dim=128, num_layers=3 表示一个相对较小但足够表达残差的 MLP。
    input_dim = 16
    output_dim = 6
    hidden_dim = 128
    num_layers = 3

    model = MLP(input_dim, output_dim, hidden_dim, num_layers).to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    train_losses = []

    for epoch in range(epochs):
        # 每个 epoch 开始时，先清空外循环优化器里的梯度。
        meta_optimizer.zero_grad()
        meta_loss = 0.0
        n_tasks_used = 0

        # 每个 epoch 随机采样若干个 task，构成一个 meta-batch。
        task_ids = np.random.choice(list(task_data.keys()), meta_batch_size, replace=False)

        for task_id in task_ids:
            x, y = task_data[task_id]

            # 一个 task 至少要有 support + query 两部分数据，所以要求 >= 2*k。
            if len(x) < 2 * k:
                continue

            # 每次都重新打乱 task 内样本，避免 support/query 固定不变。
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            # 前 k 条作为 support set，用于 task 内快速适配。
            # 后 k 条作为 query set，用于评价适配后的效果。
            x_support, y_support = x[:k].to(device), y[:k].to(device)
            x_query, y_query = x[k:2 * k].to(device), y[k:2 * k].to(device)

            # adapted_params 表示“当前 task 经过内循环更新后的参数副本”。
            # 注意这里不是直接改 model 本体，而是复制出一份 task-specific 参数。
            adapted_params = {name: param.clone() for name, param in model.named_parameters()}
            for _ in range(inner_steps):
                # support 集上的前向与损失：模拟“拿到少量该任务数据后的快速适配”。
                support_pred = functional_call(model, adapted_params, (x_support,))
                support_loss = F.mse_loss(support_pred, y_support)

                # create_graph=True 是 MAML 的关键：
                # 这样 query loss 才能反向传播穿过 inner update，更新到初始化参数。
                grads = torch.autograd.grad(support_loss, adapted_params.values(), create_graph=True)
                adapted_params = {
                    name: param - inner_lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

            # 用“适配后的参数”在 query 集上做评价。
            # 多个 task 的 query loss 平均后，就是这一轮的 meta loss。
            query_pred = functional_call(model, adapted_params, (x_query,))
            meta_loss += F.mse_loss(query_pred, y_query)
            n_tasks_used += 1

        if n_tasks_used == 0:
            continue

        # 对整个 meta-batch 的 task loss 求平均。
        meta_loss = meta_loss / n_tasks_used

        # 外循环更新：真正更新的是共享初始化参数 model.parameters()。
        meta_loss.backward()
        meta_optimizer.step()
        train_losses.append(meta_loss.item())

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch + 1:05d}] Meta Loss: {meta_loss.item():.6f}")

    # 训练结束后，把元初始化参数和模型结构信息一起保存。
    save_path = DEFAULT_META_CHECKPOINT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        },
        save_path,
    )
    print(f"Saved model to {save_path}")

    # 同时把训练损失曲线画出来，便于你观察是否收敛。
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss (MSE)")
    plt.title(f"MAML Meta-Training Loss (Quadrotor3D Residuals, {epochs} epochs)")
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plot_path = save_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved loss plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
