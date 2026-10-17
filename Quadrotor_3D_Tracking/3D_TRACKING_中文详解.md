# Quadrotor_3D_Tracking 中文详解

这份文档只说明 `MetaResidual-MPC/Quadrotor_3D_Tracking` 这一部分，目标是让刚接触控制、动力学、元学习的人，也能逐步明白这个仓库到底在做什么，以及为什么这样设计。

如果你只想先抓住一句话，可以先记住下面这句：

> 这个 3D tracking 模块做的事情，是让四旋翼在存在模型失配的情况下，依然能稳定跟踪 3D 轨迹；它的核心思路是“名义物理模型 + 可学习残差模型 + MPC”，而元学习负责让这个残差模型在遇到新的动力学参数时，能用很少的在线数据快速适应。


## 1. 这个模块到底在做什么

`Quadrotor_3D_Tracking` 研究的是一个很具体的问题：

- 控制对象是 3D 四旋翼无人机。
- 任务目标是跟踪一条给定的空间轨迹，例如圆轨迹或 8 字轨迹。
- 控制器主体是 MPC，也就是模型预测控制。
- 问题在于：MPC 需要一个动力学模型，而真实无人机和控制器内部使用的模型并不完全一样。
- 一旦存在模型失配，MPC 的预测就会偏，跟踪误差会变大。

所以这个模块的真正目标不是“重新发明一个控制器”，而是：

- 保留 MPC 作为主控制框架；
- 保留一个物理上可解释的名义模型；
- 再学习一个“残差动力学模型”，专门补偿名义模型和真实系统之间的差；
- 用元学习让这个残差模型在新任务上能更快完成在线适配。

换句话说，这个仓库不是让神经网络直接输出控制量，而是让神经网络去学习“名义物理模型还差了多少”，然后把这个补偿项塞回 MPC 里。


## 2. 为什么要这样做

### 2.1 只用名义模型有什么问题

在控制里，我们通常先写一个理想的动力学模型：

`x_dot = f_nominal(x, u)`

这里：

- `x` 是状态；
- `u` 是控制输入；
- `f_nominal` 是控制器内部相信的动力学。

但是真实系统其实更接近：

`x_dot = f_true(x, u)`

如果 `f_true != f_nominal`，就出现模型失配。四旋翼上常见的失配来源包括：

- 质量和建模值不一致；
- 转动惯量不一致；
- 实际执行器和理想推进模型有差异；
- 某些未建模非线性或耦合项没有进名义模型。

这时，如果 MPC 还按照错误模型做预测，就会出现：

- 预测状态和真实状态偏离；
- 轨迹跟踪误差变大；
- 在更激进轨迹或更重载荷下性能明显下降。


### 2.2 为什么不是直接学完整动力学

因为物理模型并不是一无是处。

名义模型已经知道很多正确结构：

- 平移和姿态的基本运动学关系；
- 推力如何影响加速度；
- 力矩如何影响角速度；
- 输入约束和系统结构。

如果把这些全丢掉，直接学 `f_true(x, u)`，会有几个问题：

- 数据需求更大；
- 泛化通常更差；
- 可解释性变弱；
- 和现有 MPC 的结合更麻烦。

所以更稳妥的办法是学习残差：

`residual(x, u) = f_true(x, u) - f_nominal(x, u)`

然后控制器内部用：

`x_dot ~= f_nominal(x, u) + residual_hat(x, u)`

这就是“残差学习”的核心思想。


### 2.3 为什么又要元学习

如果只有一个固定的残差网络，它通常会有一个问题：

- 训练时见过的动力学条件和测试时的新条件不完全相同；
- 一旦真实飞机更重、更轻、惯量不同，残差分布就会变。

于是，普通监督学习只能回答：

“在训练分布平均意义上，我怎么拟合残差？”

而元学习想回答的是：

“当我只拿到少量新系统的数据时，我怎么尽快判断现在这个系统是哪一类，并快速调整残差模型？”

这就是这个仓库中元学习的作用：不是替代 MPC，而是让残差模型具备“快速适应新动力学任务”的能力。


## 3. 用一个直观比喻理解整套方法

可以把它想成下面这个类比：

- 名义模型像一本教科书，讲的是“标准四旋翼应该怎么动”。
- 真实系统像不同学生写出来的实际答案，每个人都有一点自己的偏差。
- 残差学习是在学“这名学生相对教科书偏了多少”。
- 元学习是在看过很多学生后，学会“只看这个学生前几道题，就大概知道他的习惯和偏差风格”。

在这个仓库里：

- “学生”就是不同动力学参数下的四旋翼；
- “前几道题”就是在线收集到的一小段最近数据；
- “判断偏差风格”就是估计 context；
- “根据 context 去修正残差模型”就是元学习部署阶段真正做的事情。


## 4. 这个模块里的系统组成

从实现上看，整个 `3D_Tracking` 可以拆成 4 层：

### 4.1 仿真环境层

来自 `safe-control-gym` 的 3D 四旋翼环境：

- 文件依赖入口在 `quadrotor3D_common.py`
- 实际环境类是 `Quadrotor`
- 频率是 `50 Hz`
- 每个控制步长 `dt = 0.02 s`

这个环境扮演“真实系统”的角色。


### 4.2 名义动力学层

`quadrotor3D_common.py` 里定义了一个四旋翼名义模型，控制器把它当成物理先验。

它包含：

- 位置和速度；
- 欧拉角和机体系角速度；
- 推力到平动加速度的映射；
- 力矩到角速度变化的映射。

这个模型不是完全错误的，它只是和真实环境故意不完全一致。


### 4.3 残差模型层

残差模型的输入是：

- 当前状态 `x`
- 当前控制 `u`

拼起来一共 16 维：

- 状态 12 维
- 控制 4 维

残差模型的输出是 6 维：

- `res_x_ddot`
- `res_y_ddot`
- `res_z_ddot`
- `res_p_dot`
- `res_q_dot`
- `res_r_dot`

这里很关键的一点是：它只学习“加速度类”的残差，不学习位置导数这种本来就由运动学直接给出的量。

这样做有两个好处：

- 任务更聚焦；
- 学习对象更接近“真正不确定的动力学部分”。


### 4.4 MPC 层

MPC 每个时刻都会做以下事：

1. 拿当前状态作为起点。
2. 看未来一段时间的参考轨迹。
3. 用模型预测未来状态。
4. 优化一串控制输入。
5. 只执行第一个控制量。
6. 下一个时刻重新求解。

这正是标准 receding horizon 控制思路。

所以这个仓库的核心不是“神经网络控制”，而是：

- 主体仍然是 MPC；
- 神经网络只是给 MPC 提供更准确的预测模型。


## 5. 状态、输入、参考轨迹分别是什么

### 5.1 状态定义

这个模块使用 12 维状态：

`[x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]`

可以按两组去理解：

- 平动部分：位置和线速度
- 转动部分：姿态角和角速度

其中：

- `phi, theta, psi` 分别可以理解为滚转、俯仰、偏航；
- `p, q, r` 是机体系角速度。


### 5.2 控制输入定义

控制输入是 4 个旋翼推力：

`[u1, u2, u3, u4]`

MPC 优化的就是每个时刻这 4 个推力该取什么值。


### 5.3 参考轨迹

这个模块里主要用了两种轨迹：

- 圆轨迹 `circle`
- 8 字轨迹 `figure8`

参考轨迹不仅给出位置，也给出速度参考，姿态参考多数设为 0 或固定偏航参考。

这么做的意义是：

- 圆轨迹比较平滑，是标准跟踪任务；
- 8 字轨迹会带来更复杂的曲率变化，更能考验控制器和模型。


## 6. 这个仓库里“任务”到底是什么意思

这是理解元学习最重要的一点。

在很多元学习工作里，一个任务可能是“一条轨迹”或者“一种目标函数”。但在这个仓库中，任务定义得更物理化：

> 一个 task 对应一组真实动力学参数，也就是一组真实质量和转动惯量组合。

具体来说，数据采集脚本里枚举了：

- 质量比例 `MASS_RATIOS = [0.75, 1.0, 1.25, 1.5, 1.75]`
- 惯量比例 `IXX_RATIOS = [0.8, 1.0, 1.25]`
- `IYY_RATIOS = [0.8, 1.0, 1.25]`
- `IZZ_RATIOS = [0.8, 1.0, 1.25]`

总任务数是：

`5 x 3 x 3 x 3 = 135`

也就是说，元学习不是在记某一条轨迹，而是在学习：

- 当真实四旋翼更重一些会怎样；
- 当某个轴惯量更大时会怎样；
- 当这些参数组合变化时，残差动力学会怎么变。

这也是为什么这个实现更接近“面向动力学身份识别的元学习”。


## 7. 元学习在这里到底学什么

这部分建议你讲给别人时重点展开。

### 7.1 一般性的元学习概念

元学习常被概括成一句话：

> 学会如何学习。

更具体一点：

- 普通训练是在很多样本上学一个固定函数；
- 元学习是在很多任务上训练，让模型具备快速适应新任务的能力。

所以“元”字不是神秘，而是多了一层：

- 第一层是任务内学习；
- 第二层是跨任务学习如何更快适应。


### 7.2 这个仓库不是在做哪种元学习

虽然 checkpoint 文件名里有 `maml`，但从当前代码实现看，真正部署的主版本更接近：

- context-based meta-learning
- amortized context meta-learning

也就是：

- 不是在在线阶段大幅更新整张网络权重；
- 而是通过一个 context encoder，用最近少量数据估计“当前任务上下文”；
- 再把这个 context 注入残差网络，让同一张共享网络在不同任务下表现不同。

这点非常重要，因为它解释了为什么这个方法在线适配更快。


### 7.3 这个仓库里的元学习公式可以怎么理解

可以把它写成两步：

第一步，用最近一小段数据估计当前任务上下文：

`c = g(D_support)`

其中：

- `D_support` 是最近收集到的一小段 `(state, action, residual_label)` 数据；
- `g(.)` 是 `SupportSetEncoder`；
- `c` 是低维 context。

第二步，用 context 调制残差模型：

`residual_hat = f_theta(x, u, c)`

其中：

- `f_theta` 是共享残差网络；
- `theta` 是跨任务共享的参数；
- `c` 让同一个网络面对不同动力学任务时能切换出不同表现。

最后，控制器用：

`x_dot ~= f_nominal(x, u) + lift(residual_hat)`

这里的 `lift(.)` 是把 6 维残差放回 12 维状态导数里：

- 在平动和角加速度通道上加残差；
- 运动学恒等式通道保持不变。


### 7.4 support / query 在这个仓库里分别是什么

这是元学习里最容易说空的地方，这里一定要落到实现。

在训练时，每个 task 有多条 rollout。训练脚本会从同一个 task 中抽：

- 一段 support window
- 另一段 support window
- 一段 query horizon

含义分别是：

- support：给模型少量样本，让它“识别当前任务是谁”
- query：检查模型在识别完任务后，能不能在同一任务的另一段数据上也预测得好

这里特意从不同 rollout，甚至不同参考轨迹里抽 support / query，是为了避免模型只记住某条轨迹，而是真的识别动力学身份。


## 8. 数据是怎么收集出来的

数据收集脚本是：

- `MetaLearning/DataCollection_Meta.py`

它做的事情不是“随机录飞行数据”，而是很系统地构造监督数据。


### 8.1 基本思路

对每一个动力学任务：

1. 用这一组真实质量和惯量创建仿真环境。
2. 让一个只知道名义模型的 MPC 去跟踪参考轨迹。
3. 在闭环过程中记录状态、控制、名义预测和真实响应。
4. 用真实响应减去名义响应，得到 residual label。

这一步非常关键，因为它回答了“监督学习的标签从哪来”。


### 8.2 为什么用名义 MPC 去采数

因为我们真正要学的是：

> 当名义控制器面对真实系统时，名义模型和真实系统的差值是什么。

如果不用闭环控制采数，而是完全随机打控制量，虽然也能收集数据，但数据可能和真实部署时看到的状态分布不一致。

这个仓库选择：

- 用 MPC 在闭环里采数；
- 让数据分布更接近未来真实部署时会遇到的分布。


### 8.3 采集了多少任务、多少轨迹

每个 task 对应一组惯性参数。

每个 task 下又跑了 4 类参考轨迹配置：

- 默认圆轨迹
- 更快且 z 激励更强的圆轨迹
- 居中的 8 字轨迹
- 平移后的 8 字轨迹

每条 rollout：

- 时长 `12 s`
- 步长 `0.02 s`
- 共 `600` 步

所以当前数据集实际规模是：

- 任务数：`135`
- episode 数：`540`
- 总样本行数：`324000`
- CSV 列数：`55`

这份数据已经保存在：

- `meta_dataset_quadrotor3D/quadrotor3d_meta_residual_mpc.csv`


### 8.4 监督标签是怎么构造的

每一步都会记录当前：

- 当前状态 `x_t`
- 当前控制 `u_t`
- 下一时刻状态 `x_{t+1}`

然后用有限差分近似真实导数：

- `x_ddot_true`
- `y_ddot_true`
- `z_ddot_true`
- `p_dot_true`
- `q_dot_true`
- `r_dot_true`

再用名义模型在同一个 `(x_t, u_t)` 下算出：

- `x_ddot_nom`
- `y_ddot_nom`
- `z_ddot_nom`
- `p_dot_nom`
- `q_dot_nom`
- `r_dot_nom`

最后做差：

`residual = true - nominal`

得到训练标签：

- `res_x_ddot`
- `res_y_ddot`
- `res_z_ddot`
- `res_p_dot`
- `res_q_dot`
- `res_r_dot`

所以这份数据不是原始传感器日志，而是已经组织好的“残差监督学习数据”。


### 8.5 为什么数据里要同时存 task_id 和物理参数

因为元学习需要两层信息：

- `task_id` 用来告诉训练脚本哪些样本属于同一个任务；
- `mass_ratio, ixx_ratio, iyy_ratio, izz_ratio` 则给了任务的物理语义。

在这个实现里，context 不只是一个完全无语义的隐变量，它还被监督去接近归一化后的物理参数向量。这会让学到的 context 更可解释。


## 9. 离线训练是怎么做的

离线训练脚本是：

- `MetaLearning/Offline_Train_Meta.py`

它做的不是普通的“一次性回归残差”，而是“带 context 的元训练”。


### 9.1 训练数据如何组织

脚本先读 CSV，然后按 `task_id` 分组。

对每个 task：

- 取该 task 对应的 context 标签，也就是归一化后的惯性参数向量；
- 再按 `episode_id` 切成多条 rollout；
- 每条 rollout 里保存：
  - `features = [state, action]`
  - `targets = residual`
  - `states`
  - `actions`
  - `next_states`

这里保存 `states/actions/next_states` 的原因是：训练时不仅要拟合残差，还要检查一步滚动后的状态误差。


### 9.2 训练时的两个网络分别做什么

当前主版本训练两个网络：

#### A. `ContextResidualMLP`

它是主残差网络，负责做：

`(x, u, context) -> residual`

也就是说，真正输出残差的是它。


#### B. `SupportSetEncoder`

它负责做：

`support features + support targets -> context`

直观理解就是：

- 看一小段最近数据；
- 推断当前系统的动力学身份；
- 输出一个低维 context 给残差网络使用。


### 9.3 context 是怎么注入残差网络的

这个仓库里保留了两种注入方式：

- `concat`
- `adapter`

当前训练脚本默认使用的是 `adapter`。

它的思想是：

- 主干先处理 `(x, u)`；
- context 再通过缩放和偏移去调制隐藏特征；
- 相当于让共享主干根据任务上下文改变表达方式。

这比简单拼接更像“条件化残差模型”。


### 9.4 训练时 support 和 query 的窗口大小

默认配置里：

- `support_window = 32`
- `query_horizon = 16`
- 数据步长 `dt = 0.02 s`

所以大致可以理解为：

- support 大约是最近 `0.64 s` 的数据
- query 大约是后面 `0.32 s` 的数据

这说明模型的设计目标不是依赖很长历史，而是用很短的一段数据快速判断当前任务。


### 9.5 损失函数不是只有一个 MSE

这是这个训练脚本最值得讲的地方之一。

它最终优化的是一个组合损失，主要包含 5 类：

#### 1. Query residual loss

让模型在 query 段上直接拟合残差。

这是最基本的监督学习目标。


#### 2. Query state loss

把预测残差加回名义动力学，滚动一步得到预测下一状态，再和真实下一状态比较。

这一步很重要，因为它不只关心“残差数值像不像”，还关心“这个残差放回动力学后对状态预测是否真的有帮助”。


#### 3. Context supervision loss

要求 encoder 输出的 context 接近真实的归一化惯性参数。

这会把 meta-learning 的 latent 和真实物理属性对齐。


#### 4. Consistency loss

同一个 task 的两段 support 数据编码出来的 context 应该相近。

这鼓励 context 真正表示任务身份，而不是偶然的轨迹片段。


#### 5. Contrastive loss

脚本把同一个 task 的两段 support context 当作正样本对，并和其他 task 区分开。

这让不同任务的 context 更可分，有助于快速识别动力学身份。


### 9.6 为什么这套损失很有意义

如果只有 residual MSE，模型可能学到：

- 平均意义下还行；
- 但 context 不稳定；
- 或者对真实状态滚动帮助不大。

而组合损失把几个需求同时塞进去了：

- 残差要拟合准；
- 放回动力学后状态要预测准；
- context 要和物理任务相关；
- 同任务 context 要一致；
- 不同任务 context 要可分。

这就是它比“普通回归一个残差网络”更像元学习系统的原因。


### 9.7 训练输出了什么

训练完成后会保存：

- 最优 checkpoint：`maml_quadrotor3d_meta_init_3_128.pth`
- 最终 checkpoint：`maml_quadrotor3d_meta_init_3_128_final.pth`
- 训练曲线图：同名 `.png`

虽然文件名前缀是 `maml`，但从 payload 里的 `model_type` 来看，当前实际部署的是：

- `amortized_context_meta`

也就是说，核心是“编码 support set 得到 context”，而不是在线做多步权重微调。


## 10. 在线部署时到底发生了什么

这一部分是讲理论和讲工程连接的关键。

主入口在：

- `quadrotor3D_common.py` 里的 `run_tracking(...)`


### 10.1 这里的“部署”是什么意思

在这个仓库语境下，“部署”主要指：

- 把残差模型接到 MPC 内部；
- 在闭环仿真时一边控制、一边在线适配；
- 不是上 ROS 或真实飞控硬件的最终上机部署。

所以它更准确地说是：

> learned model integrated into MPC for online closed-loop control


### 10.2 一共有三种运行模式

#### 1. `nominal`

- 不用神经网络；
- 只用名义动力学；
- 是最基础 baseline。


#### 2. `lightmlp`

- 使用一个普通 MLP 残差网络；
- 没有离线元训练；
- 在线阶段直接用最近数据更新网络参数。

可以把它理解成：

- “纯在线学习”的残差基线。


#### 3. `meta`

- 加载离线训练好的 meta checkpoint；
- 使用 `SupportSetEncoder` 从最近数据估计 context；
- 把 context 注入 `ContextResidualMLP`；
- 在线主要更新的是 context，而不是整张网络。

这是当前仓库最核心的方法。


### 10.3 online loop 每一步在做什么

部署时每个控制步大致做下面这些事：

1. 根据当前时间生成未来参考轨迹。
2. 把参考轨迹喂给 MPC 的整个 horizon。
3. 用当前状态固定 MPC 初值。
4. 求解最优控制序列。
5. 取第一个控制输入作用到真实环境。
6. 观察下一个状态。
7. 如果是 learned 方法，就把这一步变成一个新的监督样本。

这个“新的监督样本”正是：

- 输入：`[prev_state, action]`
- 标签：由有限差分得到的 residual target

所以在线阶段不是凭空适配，而是不断把新闭环数据转成监督数据。


### 10.4 meta 方法在线怎么更新

这一步是整个仓库最核心的亮点。

部署时会维护一个最近数据缓冲区：

- `feature_buffer`
- `target_buffer`

每隔一段时间，取最近 `batch_size` 个样本：

- 对于当前 checkpoint，默认 support window 是 32；
- 部署时适配间隔默认至少 1 秒；
- 也就是大约每隔 50 个控制步更新一次 context。

然后：

1. 把最近样本送入 `SupportSetEncoder`
2. 得到新的 context
3. 用 EMA 做一点平滑
4. 把这个 context 写回 `ContextResidualMLP`
5. 更新 L4CasADi 包装的残差模型
6. 下一个 MPC 求解开始用新的 learned dynamics

这套逻辑意味着：

- 网络主干参数在离线已经学好；
- 在线只需要识别“当前系统是哪一类”；
- 所以适配更快、计算更稳。


### 10.5 为什么它能和 MPC 结合起来

这是通过 `l4casadi` 和 `acados` 完成的。

流程可以理解为：

1. PyTorch 残差网络先由 `l4casadi` 包装成 CasADi 可调用模块。
2. 再作为 `acados` 的外部 shared library 接入 OCP。
3. MPC 使用“名义动力学 + learned residual”作为预测模型。
4. 当 online adaptation 更新后，再调用 `update(...)` 把新模型同步进去。

因此，MPC 在优化时看到的不是纯手工模型，而是：

- 物理名义模型
- 加上可微、可导出的神经网络残差项

这正是“learning for control”里非常典型的工程集成方式。


## 11. 这个方法为什么比普通在线学习更有优势

你可以从 3 个角度讲。

### 11.1 数据效率

普通在线学习通常要从随机初始化开始调整参数，前期样本不够时很慢。

元学习方法因为已经看过很多动力学任务，所以在线只需要少量数据就能判断：

- 当前系统更像“更重”
- 还是“惯量更大”
- 或者“某个轴更难转”


### 11.2 稳定性

在线直接更新整张网络容易抖动，因为参数太多。

而这个仓库当前主方法主要更新 context：

- 更新变量更少；
- 结构更稳定；
- 更适合和 MPC 这种需要实时求解的系统配合。


### 11.3 泛化性

如果训练时看过很多不同动力学参数组合，那么部署时遇到一个新组合，模型更容易通过少量样本推断出它属于哪个区域。

这就是“跨任务迁移”的本质。


## 12. 这个仓库里有哪些关键文件

如果你要给别人讲结构，可以按下面这张表讲。

### 12.1 核心实现文件

- `quadrotor3D_common.py`
  - 公共核心文件
  - 定义参考轨迹、名义动力学、learned dynamics、MPC、在线运行逻辑

- `MetaLearning/DataCollection_Meta.py`
  - 采集元学习数据集
  - 把真实响应和名义响应的差做成 residual label

- `MetaLearning/Offline_Train_Meta.py`
  - 训练 amortized context meta residual model


### 12.2 运行入口脚本

- `quadrotor3D_Nominal.py`
  - 只跑名义模型基线

- `quadrotor3D_Lightmlp.py`
  - 跑普通在线 MLP 残差基线

- `quadrotor3D_Meta.py`
  - 跑元学习方法

- `quadrotor3D_Figure8_Eval.py`
  - 在 8 字轨迹上分别评估 nominal / lightmlp / meta

- `quadrotor3D_PressureSuite.py`
  - 更难场景的批量压力测试


### 12.3 生成产物目录

- `meta_dataset_quadrotor3D/`
  - 训练数据 CSV

- `MetaLearning/*.pth`
  - 训练好的 checkpoint

- `_l4c_generated/`
  - L4CasADi 生成的共享库和导数文件

- `c_generated_code/`
  - acados 代码生成产物

- `results/`、`results-0410/`
  - 已保存的轨迹 CSV 和图像


## 13. 如果你要复现实验，流程应该怎么走

### 13.1 第一步：收集数据

运行：

```bash
python MetaLearning/DataCollection_Meta.py --workers 8
```

输出：

- `meta_dataset_quadrotor3D/quadrotor3d_meta_residual_mpc.csv`


### 13.2 第二步：离线训练元学习模型

运行：

```bash
python MetaLearning/Offline_Train_Meta.py
```

输出：

- `MetaLearning/maml_quadrotor3d_meta_init_3_128.pth`
- `MetaLearning/maml_quadrotor3d_meta_init_3_128_final.pth`
- `MetaLearning/maml_quadrotor3d_meta_init_3_128.png`


### 13.3 第三步：部署到闭环跟踪

分别运行：

```bash
python quadrotor3D_Nominal.py
python quadrotor3D_Lightmlp.py
python quadrotor3D_Meta.py
```

输出：

- `results/*.csv`
- `results/*.png`


### 13.4 第四步：换更难任务做评测

8 字轨迹：

```bash
python quadrotor3D_Figure8_Eval.py
```

压力测试：

```bash
python quadrotor3D_PressureSuite.py --seeds 31-50
```


## 14. 你可以怎么讲“这套理论”

如果你需要给别人做汇报，下面这套话术比较稳。

### 14.1 一分钟版

这个模块解决的是四旋翼轨迹跟踪中的模型失配问题。它没有抛弃 MPC，而是在名义物理模型外面加了一个可学习的残差模型。为了让残差模型在新动力学条件下也能快速适应，它不是做普通监督学习，而是做面向“不同质量和惯量任务”的元学习。训练阶段它在很多动力学任务上学会如何根据少量 support 数据识别当前系统身份；部署阶段它只要看最近一小段闭环数据，就能编码出 context，快速修正残差模型，再把这个模型实时嵌入 MPC 里，提高轨迹跟踪精度。


### 14.2 三分钟版

这个仓库的关键思想是“physics prior + residual learning + meta adaptation”。名义模型负责保留四旋翼的物理结构，神经网络只学习模型失配造成的那部分残差。为了让残差模型在不同真实惯性参数下都能快速适配，作者把每组质量和转动惯量看成一个元任务，并在每个任务下采集多条不同参考轨迹。离线训练时，模型需要从少量 support 数据推断当前任务的 context，再在 query 段上既拟合残差，也让一步状态预测更准确。在线部署时，控制器持续从最近闭环数据估计新的 context，把它注入残差网络，再通过 L4CasADi 接入 acados，使 MPC 的预测模型随着当前任务动态更新。


### 14.3 如果别人问“它到底学到了什么”

你可以这样回答：

- 它不是直接学控制律；
- 它也不是完全替代物理模型；
- 它学到的是“不同动力学任务下，名义模型还差多少”；
- 更进一步，它学到的是“如何从少量新数据里快速识别这种差异属于哪一种任务风格”。


### 14.4 如果别人问“为什么这算元学习”

你可以这样回答：

- 因为训练对象不是单一数据分布，而是很多不同动力学任务；
- 训练目标不是只在已见样本上拟合，而是让模型在见到新任务时能用少量 support 数据迅速适配；
- support/query/task 这三层结构正是元学习范式的核心。


### 14.5 如果别人问“为什么不用普通神经网络就好”

可以这样答：

- 普通网络可以学平均残差，但不一定能快速适应新动力学；
- 在线直接更新整张网络参数往往更慢、更不稳定；
- 元学习方法把跨任务经验提前学到离线阶段，部署时只需少量数据就能识别新任务并快速调整。


## 15. 这个实现的几个重要特点和局限

这部分讲出来会显得你对仓库理解更扎实。

### 15.1 特点

- 没有放弃物理模型，而是把学习放在残差层面。
- 任务定义和物理参数直接对应，解释性较强。
- support/query 采样会跨不同轨迹，避免把轨迹本身当成任务。
- 损失函数同时约束残差预测、状态预测和 context 结构。
- 在线部署主要更新 context，而不是整张网络，更适合实时控制。


### 15.2 局限

- 当前任务变化主要集中在质量和转动惯量，没有覆盖更多现实误差源。
- 标签来自有限差分，在真实传感器噪声下可能更难。
- 这里的“部署”主要还是仿真内闭环部署，不等于真实飞控上机。
- 当前方法只学习 6 个残差分量，默认其余部分由名义模型已足够描述。


## 16. 最后给你的一个总总结

如果你要把整个 `Quadrotor_3D_Tracking` 用一句更专业的话概括，可以说：

> 这是一个面向 3D 四旋翼轨迹跟踪的 residual meta-learning MPC 框架。它通过在多组真实惯性参数任务上离线学习带 context 的残差动力学模型，使控制器在在线闭环时能利用少量最近数据快速识别当前动力学身份，并把适配后的残差模型实时嵌入 MPC，从而在模型失配条件下提升轨迹跟踪性能。

如果你要把它说得更通俗一点，可以说：

> 它先学会很多种“四旋翼偏差风格”，然后在新飞机上只看一小段最近飞行数据，就能猜出这台飞机哪里和教科书不一样，再把这个猜测实时反馈给 MPC，让控制器跟得更准。


## 17. 增量补充：把整套方法先压缩成一张“总公式图”

前面 1 到 16 节已经把思路讲清楚了。下面这一部分不替换前文，而是把前面的每个概念重新按“公式 -> 数据 -> 代码”再走一遍，尽量让第一次接触控制和元学习的人也能看懂。

如果把整个系统压缩成 3 行，最核心就是：

```text
真实系统:         state_dot = f_true(state, u)
控制器名义模型:   state_dot = f_nominal(state, u)
控制器最终使用:   state_dot ~= f_nominal(state, u) + lift(residual_hat(state, u, c))
```

其中：

- `state` 是四旋翼当前 12 维状态。
- `u` 是 4 个旋翼推力。
- `f_true` 是仿真环境里的真实动力学。
- `f_nominal` 是 MPC 内部相信的名义动力学。
- `residual_hat` 是神经网络学到的残差。
- `c` 是根据最近一小段数据编码出来的 context。
- `lift(.)` 的意思是：把网络输出的 6 维残差，重新塞回 12 维状态导数里对应的位置。

也就是说，这个项目不是让网络“直接控制飞机”，而是让网络回答一个更具体的问题：

> 在当前这台真实四旋翼上，名义模型到底还差了多少？


### 17.1 先把符号和代码变量对齐

这一步对小白非常重要，否则很容易被公式和代码里的命名绕晕。

| 数学概念 | 代码里常见变量 | 含义 |
| --- | --- | --- |
| 当前状态 | `x`、`xt`、`prev_xt` | 12 维四旋翼状态 |
| 当前控制 | `u`、`ut` | 4 个旋翼推力 |
| 下一时刻状态 | `x_next`、`next_states` | 执行控制后的下一拍状态 |
| 名义动力学 | `f_nominal`、`nominal_func` | 纯物理模型给出的状态导数 |
| 残差标签 | `residual_targets`、`targets` | 真实导数减名义导数 |
| support 特征 | `x_support` | 一小段 `[state, action]` |
| support 标签 | `y_support` | 一小段残差标签 |
| context | `context`、`context_a`、`context_b` | 当前任务的低维表示 |

还有一个特别容易混淆的点：

- 状态向量里的 `x_dot, y_dot, z_dot` 是“速度状态分量”。
- 整个状态的导数，我下面统一写成 `state_dot`，避免和 `x_dot` 这个变量名混在一起。


### 17.2 12 维状态和 4 维控制到底是什么

本项目状态顺序是：

```text
state =
[x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]^T
```

对应含义是：

- `x, y, z`：世界坐标系位置。
- `x_dot, y_dot, z_dot`：世界坐标系线速度。
- `phi, theta, psi`：欧拉角，通常可理解成滚转、俯仰、偏航。
- `p, q, r`：机体系角速度。

控制输入是：

```text
u = [f1, f2, f3, f4]^T
```

也就是 4 个电机产生的推力，而不是直接给姿态角、角速度或者总推力总力矩。


### 17.3 名义模型的连续时间动力学，逐项是什么意思

这一部分直接对应：

- `quadrotor3D_common.py` 里的 `nominal_dynamics_terms(...)`
- `Offline_Train_Meta.py` 里的 `nominal_dynamics_torch(...)`

项目中的名义模型可以拆成 4 块来看。

#### A. 位置运动学

```text
d/dt x = x_dot
d/dt y = y_dot
d/dt z = z_dot
```

这部分完全是运动学恒等式，几乎没有学习的必要，所以残差模型不会去拟合这些通道。

#### B. 平动动力学

先定义总推力：

```text
T = f1 + f2 + f3 + f4
```

那么项目里写出来的 3 个平动加速度是：

```text
x_ddot = (sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)) * T / m
y_ddot = (-sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)) * T / m
z_ddot = cos(phi) * cos(theta) * T / m - g
```

直观解释：

- 4 个旋翼总推力本来是在机体 `z` 轴方向。
- 但无人机会倾斜，所以这个推力会被旋转到世界坐标系里。
- 旋转后的推力在 `x/y/z` 三个方向上都有投影。
- 再除以质量 `m`，得到加速度。
- 最后 `z` 方向还要减去重力 `g`。

所以你可以把平动理解成一句话：

> 无人机靠“先倾斜，再把总推力投影到世界坐标系中”来产生水平加速度。

#### C. 力矩映射

这个项目用的是四旋翼最经典的“推力 -> 机体系力矩”映射：

```text
tau_x = L / sqrt(2) * (f1 + f2 - f3 - f4)
tau_y = L / sqrt(2) * (-f1 + f2 + f3 - f4)
tau_z = gamma * (-f1 + f2 - f3 + f4)
```

其中：

- `L` 是机臂长度。
- `gamma = KM / KF`，表示偏航扭矩和升力之间的比例。

它表达的物理意义是：

- 左右、前后不同电机推力不平衡，会产生滚转/俯仰力矩。
- 旋翼旋转方向不同，会在偏航方向产生反扭矩。

#### D. 角速度动力学

机体系角速度满足刚体转动方程：

```text
omega = [p, q, r]^T
J * omega_dot = tau - omega x (J * omega)
omega_dot = J^{-1} * (tau - omega x (J * omega))
```

这就是代码里：

- `body_rates = [p_body, q_body, r_body]`
- `body_rates_dot = j_inv @ (body_torque - cross(body_rates, j @ body_rates))`

的数学形式。

这里最关键的是 `omega x (J omega)` 这一项，它表示刚体自转带来的陀螺耦合效应。对初学者来说，不需要一开始就把它背下来，只要知道：

> 角速度变化不只由力矩决定，还会受到当前转动状态和惯量分布的耦合影响。


### 17.4 欧拉角速度和机体系角速度不是一回事

这是很多小白第一次看四旋翼动力学时最容易搞混的地方。

- `p, q, r` 是机体系角速度。
- `phi_dot, theta_dot, psi_dot` 是欧拉角变化率。

它们之间不是简单相等，而是：

```text
phi_dot   = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
theta_dot = q * cos(phi) - r * sin(phi)
psi_dot   = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
```

代码里这部分叫：

- `euler_rates`

为什么会这样？

- 因为 `p, q, r` 是机体坐标系里绕机体轴转的角速度。
- `phi, theta, psi` 是一组参数化姿态的方法。
- “姿态参数的变化率”和“真实角速度向量”之间本来就有一个非线性变换关系。

所以这个项目保留了完整的欧拉角运动学，而没有偷懒写成 `phi_dot = p` 这种简化。


### 17.5 learned residual 到底加回了哪几个通道

网络输出是 6 维：

```text
residual_hat =
[res_x_ddot, res_y_ddot, res_z_ddot, res_p_dot, res_q_dot, res_r_dot]^T
```

它不会直接输出 12 维 `state_dot`，而是先经过一个“抬升”操作：

```text
lift(residual_hat) =
[0,
 res_x_ddot,
 0,
 res_y_ddot,
 0,
 res_z_ddot,
 0,
 0,
 0,
 res_p_dot,
 res_q_dot,
 res_r_dot]^T
```

然后控制器真正使用：

```text
state_dot = f_nominal(state, u) + lift(residual_hat)
```

这样设计非常有针对性，因为：

- 位置导数 `d/dt x = x_dot` 本来就是确定的，不需要学。
- 欧拉角导数和当前姿态/角速度的关系，名义模型已经能写得比较准。
- 真正最容易错的，通常是平动加速度和角加速度这 6 个通道。

从代码对应关系看：

- 训练里是 `residual_to_state_derivative(...)`
- 部署里是 `Quadrotor3DLearnedDynamics.model()` 里手工拼出的 `residual_term`


### 17.6 离散时间一步预测在代码里是什么样

控制和训练最终都是离散时间运行的，步长是：

```text
dt = 0.02 s
```

训练脚本里的一步滚动近似是：

```text
state_{t+1} ~= wrap(state_t + dt * state_dot_t)
```

其中：

```text
state_dot_t = f_nominal(state_t, u_t) + lift(residual_hat_t)
```

`wrap(.)` 的意思是只把欧拉角 `phi, theta, psi` 包回 `[-pi, pi)` 区间，避免角度无限增长导致误差计算奇怪。

这个逻辑在训练里对应：

- `rollout_step_torch(...)`
- `wrap_state_angles_torch(...)`


### 17.7 MPC 优化的目标函数在代码里可以近似写成什么

虽然代码里没有把公式写成论文形式，但 `MPC.ocp()` 里本质上就是标准 tracking MPC：

```text
min  sum_{k=0}^{N-1} (x_k - x_ref_k)^T Q (x_k - x_ref_k)
   + sum_{k=0}^{N-1} u_k^T R u_k
   + (x_N - x_ref_N)^T Q (x_N - x_ref_N)
```

其中：

- 预测步数 `N = 20`
- 预测总时域 `T_horizon = 1.0 s`
- 所以每个 shooting interval 大约是 `1.0 / 20 = 0.05 s`

这和环境控制周期 `0.02 s` 不是一回事：

- 环境每 `0.02 s` 真正执行一次控制。
- MPC 内部预测网格是 `0.05 s` 一格。

这是工程里很常见的设置，不一定要求预测网格和控制周期完全一致。

状态权重矩阵 `Q` 的对角元素在代码里是：

```text
[8.0, 0.4, 8.0, 0.4, 12.0, 0.6, 1.5, 1.5, 1.0, 0.15, 0.15, 0.2]
```

输入权重矩阵 `R` 是：

```text
0.05 * I
```

这说明：

- 位置误差比速度误差更重要。
- `z` 方向位置权重比 `x/y` 更大。
- 姿态和角速度也有代价，但比位置轻一些。

而 MPC 的参考输入在代码里直接设成 0：

```text
yref_k = [x_ref_k, 0, 0, 0, 0]
```

所以它是在做：

> 一边尽量贴近参考轨迹，一边尽量别把电机推力用得太激进。


## 18. 增量补充：参考轨迹到底是怎么生成的

这部分直接对应 `reference_state(t, cfg)`。

### 18.1 圆轨迹公式

当 `traj_type == "circle"` 时：

```text
omega = 2 * pi / period

x_ref     = cx + r * cos(omega * t)
y_ref     = cy + r * sin(omega * t)
x_dot_ref = -r * omega * sin(omega * t)
y_dot_ref =  r * omega * cos(omega * t)
z_ref     = cz + z_amp * sin(0.5 * omega * t)
z_dot_ref = z_amp * 0.5 * omega * cos(0.5 * omega * t)
psi_ref   = yaw_ref
```

这里可以看到两个小细节：

- 平面内是标准圆周运动。
- `z` 方向变化频率只有平面角频率的一半，所以它不是“绕圆时同时上下抖得很快”，而是更缓慢的高度起伏。


### 18.2 8 字轨迹公式

当 `traj_type == "figure8"` 时：

```text
x_ref     = cx + radius * sin(omega * t)
y_ref     = cy + 0.5 * y_radius * sin(2 * omega * t)
x_dot_ref = radius * omega * cos(omega * t)
y_dot_ref = y_radius * omega * cos(2 * omega * t)
z_ref     = cz + z_amp * sin(0.5 * omega * t)
z_dot_ref = z_amp * 0.5 * omega * cos(0.5 * omega * t)
psi_ref   = yaw_ref
```

为什么它会形成“8”字？

- 因为 `x` 用的是 `sin(omega t)`。
- `y` 用的是 `sin(2 omega t)`。
- 两个方向频率不同，就形成了 Lissajous 曲线，也就是常见的 8 字轨迹。


### 18.3 为什么姿态参考大部分是 0

代码返回的参考状态中：

```text
phi_ref = 0
theta_ref = 0
p_ref = q_ref = r_ref = 0
```

这不是说无人机飞行时姿态真的始终为 0，而是说：

- 轨迹跟踪目标主要放在位置和速度上；
- 控制器会自己找到一个实现该轨迹所需的姿态；
- 姿态通道更多起“别过度偏离”的作用。

这是很常见的 tracking 设计方法。


## 19. 增量补充：数据集里的一行样本到底是怎么做出来的

这一部分对应 `MetaLearning/DataCollection_Meta.py`，建议你把下面这段流程当作“数据标签生产线”来理解。

### 19.1 一条样本的完整生成流程

对某个固定任务 `task_id`，脚本每一步会做：

1. 先用该任务对应的真实参数创建仿真环境。
2. 但 MPC 内部仍然只用名义模型。
3. 在当前时刻 `t`，MPC 求出控制 `u_t`。
4. 把 `u_t` 作用到真实环境，得到下一时刻状态 `state_{t+1}`。
5. 用有限差分估计真实二阶量。
6. 用名义模型在同一个 `(state_t, u_t)` 上算名义二阶量。
7. 做差，得到 residual label。

所以一条训练样本不是“凭空标注”的，而是：

```text
label = 真实系统响应 - 名义模型响应
```


### 19.2 真实标签为什么可以用有限差分来近似

代码里真实标签是：

```text
x_ddot_true ~= (x_dot_{t+1} - x_dot_t) / dt
y_ddot_true ~= (y_dot_{t+1} - y_dot_t) / dt
z_ddot_true ~= (z_dot_{t+1} - z_dot_t) / dt
p_dot_true  ~= (p_{t+1} - p_t) / dt
q_dot_true  ~= (q_{t+1} - q_t) / dt
r_dot_true  ~= (r_{t+1} - r_t) / dt
```

这就是 `collect_episode(...)` 和 `finite_difference_targets(...)` 里的核心做法。

对于仿真数据来说，这种做法通常够用，因为：

- 仿真状态相对干净；
- 采样频率固定；
- 我们只需要一个稳定的一步监督标签。

如果是真机，这一步通常会更麻烦，因为噪声会把有限差分放大。


### 19.3 名义标签怎么来

同一时刻，代码还会计算：

```text
nominal_targets = f_nominal(state_t, u_t)[1, 3, 5, 9, 10, 11]
```

也就是只取状态导数中对应：

- `x_ddot`
- `y_ddot`
- `z_ddot`
- `p_dot`
- `q_dot`
- `r_dot`

这 6 个分量。

然后残差标签就是：

```text
residual_targets = true_targets - nominal_targets
```

这一步一定要理解透，因为它就是整个项目的监督学习目标。


### 19.4 为什么每个 task 不是只跑一条轨迹

当前数据采集里，每个动力学 task 都会配多种参考轨迹：

- 默认圆轨迹
- 更快圆轨迹
- 居中的 8 字轨迹
- 平移后的 8 字轨迹

这么做的目的是：

> 让模型学“动力学身份”，而不是把“某条固定轨迹长什么样”误当成任务。

如果一个任务只包含一条轨迹，那 encoder 可能学偏，变成“看到这个轨迹段我就知道 task 是谁”，这就不是我们想要的东西。


### 19.5 这份 CSV 里到底有哪些列

数据集最终写出的列可以分成 8 组：

1. 任务元信息
   `task_id, episode_id, reference_id, time`
2. 任务物理标签
   `mass_ratio, ixx_ratio, iyy_ratio, izz_ratio`
3. 轨迹配置
   `center_x, center_y, center_z, radius, z_amp, yaw_ref, period, traj_type, y_radius`
4. 当前状态
   `x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r`
5. 当前控制
   `u1, u2, u3, u4`
6. 真实二阶标签
   `x_ddot_true ... r_dot_true`
7. 名义模型预测
   `x_ddot_nom ... r_dot_nom`
8. 残差标签和真实惯性参数
   `res_x_ddot ... res_r_dot, mass, ixx, iyy, izz`

你可以把其中最关键的监督三元组记成：

```text
输入:   [state_t, u_t]
标签:   residual_t
任务ID: task_id
```


### 19.6 为什么数据集总规模正好是 324000 行

代码中：

- 任务数 `135`
- 每个任务 `4` 条参考轨迹
- 每条 rollout `12 s`
- 步长 `0.02 s`
- 所以每条 rollout `600` 步

总样本数就是：

```text
135 * 4 * 600 = 324000
```

这个数字不是随便来的，而是任务网格和每条轨迹时长共同决定的。


## 20. 增量补充：离线元训练到底在优化什么

这一部分对应 `MetaLearning/Offline_Train_Meta.py`。

### 20.1 训练前，CSV 会先被整理成什么结构

`load_task_data(...)` 做的事情是：

1. 先按 `task_id` 分组。
2. 对每个 task，再按 `episode_id` 分成多条 rollout。
3. 每条 rollout 里保存：
   `features, targets, states, actions, next_states`

更具体地说：

```text
features   = [state_t, u_t]
targets    = residual_t
states     = state_t
actions    = u_t
next_states = state_{t+1}
```

这里代码用的是：

```text
features[:-1], residuals[:-1], states[:-1], actions[:-1], states[1:]
```

原因很简单：

- 第 `t` 个输入要配第 `t+1` 个真实下一状态。
- 所以最后一个状态没有再往后的 `next_state`，需要对齐后截掉。


### 20.2 context 为什么正好是 4 维

这里的 context 不是任意维度的神秘 latent，而是被刻意设计成 4 维，和物理任务参数一一对应：

```text
context = [mass_ratio, ixx_ratio, iyy_ratio, izz_ratio]
```

当然，训练时不是直接把原始比例拿来回归，而是先做归一化：

```text
context_norm = (context - CONTEXT_CENTER) / CONTEXT_SCALE
```

代码里写的是：

```text
CONTEXT_CENTER = [1.25, 1.025, 1.025, 1.025]
CONTEXT_SCALE  = [0.50, 0.225, 0.225, 0.225]
```

这样做的作用是：

- 让不同维度尺度接近；
- 让 encoder 输出更容易训练；
- 让后续 `context loss` 的数值更稳定。


### 20.3 SupportSetEncoder 到底在做什么

`SupportSetEncoder` 可以拆成 3 步：

第一步，对 support 里的每一个样本分别编码：

```text
z_i = h_encoder([feature_i, target_i])
```

其中：

- `feature_i = [state_i, action_i]`
- `target_i = residual_i`

第二步，对整个 support window 做集合池化：

```text
z_mean = mean_i(z_i)
z_std  = std_i(z_i)
pooled = [z_mean, z_std]
```

第三步，用 head 产生 context：

```text
c = h_head(pooled)
```

代码里这部分非常值得注意，因为它不是 RNN，也不是 Transformer，而是：

- 先逐样本编码；
- 再做 `mean + std` 池化；
- 最后输出一个固定长度 context。

这种设计很适合当前任务，因为 support set 的重点是“概括这一小段数据表现出的动力学风格”，而不是逐时间步做复杂序列建模。


### 20.4 ContextResidualMLP 是怎么把 context 用进去的

当前主版本使用的是：

```text
context_injection = "adapter"
```

也就是说，不是最简单的把 `context` 和输入直接拼起来，而是先用主干网络提取特征，再让 context 去调制这个特征。

其核心形式可以写成：

```text
h = backbone([state, action])
gamma = 1 + alpha * tanh(W_gamma c + b_gamma)
beta  = alpha * (W_beta c + b_beta)
h_tilde = ReLU(gamma ⊙ h + beta)
residual_hat = W_out h_tilde + b_out
```

其中：

- `⊙` 表示逐元素乘法。
- `alpha` 就是代码里的 `modulation_scale`，当前训练脚本默认 `0.20`。

直观理解：

- 共享主干负责提取“所有任务都通用”的残差信息。
- context 负责告诉网络：“当前系统更像哪一种动力学条件”。
- `gamma` 像可学习缩放。
- `beta` 像可学习平移。

所以它比简单拼接更像“根据任务上下文动态调整中间特征”。


### 20.5 support / query 是怎么抽的，为什么要跨轨迹

`sample_support_pair_query(...)` 会从同一个 task 里抽 3 段：

- `support`
- `alternate support`
- `query`

并尽量让它们来自不同 `reference_id`。

这样做的意义是：

- `support` 和 `alternate support` 用来检查“同一任务不同轨迹段编码出来的 context 应该一致”。
- `query` 用来检查“识别出这个 task 之后，模型在同任务另一段数据上还能不能预测得好”。

这一步非常重要，因为它强迫模型关注“动力学差异”，而不是“轨迹局部形状”。


### 20.6 组合损失的数学形式

当前训练脚本的总损失可以概括成：

```text
L_total
= 1.00 * L_query_residual
+ 0.35 * L_query_state
+ 0.15 * L_context
+ 0.10 * L_consistency
+ 0.06 * L_contrastive
```

下面把每一项都解释清楚。

#### A. Query residual loss

```text
L_query_residual = MSE(residual_hat_query, residual_query)
```

这是最直接的监督目标：让预测残差接近真实残差标签。


#### B. Query state loss

先根据预测残差做一步滚动：

```text
state_hat_{t+1} = wrap(state_t + dt * (f_nominal(state_t, u_t) + lift(residual_hat_t)))
```

然后分 4 组算误差：

```text
L_query_state
= 1.6  * L_pos
+ 0.20 * L_vel
+ 0.35 * L_att
+ 0.08 * L_rate
```

其中：

- `L_pos` 比较 `[x, y, z]`
- `L_vel` 比较 `[x_dot, y_dot, z_dot]`
- `L_att` 比较 `[phi, theta, psi]`
- `L_rate` 比较 `[p, q, r]`

这里姿态误差还专门做了角度 wrap，避免 `pi` 和 `-pi` 附近的人为大误差。

这一项的深层意义是：

> 不是只要残差数值好看，而是要把残差放回动力学后，真的能让一步状态预测更准。


#### C. Context supervision loss

```text
L_context
= 0.5 * (MSE(context_a, true_context) + MSE(context_b, true_context))
```

这一步把 context 明确拉向真实物理参数的归一化值，因此这个项目里的 context 不是完全不可解释的黑箱变量。


#### D. Consistency loss

```text
L_consistency = MSE(context_a, context_b)
```

同一个 task 的两段 support 数据，无论来自哪条轨迹，编码出来的 context 都应该接近。


#### E. Contrastive loss

代码里用的是对称式对比损失：

```text
L_contrastive = 0.5 * (CE(A B^T / tau, diag) + CE((A B^T)^T / tau, diag))
```

其中：

- `A` 是一个 batch 中所有 `context_a` 的归一化向量。
- `B` 是所有 `context_b` 的归一化向量。
- 对角线位置表示同一 task 的正样本对。
- 温度参数 `tau = 0.35`。

这一步会让：

- 同一 task 的 context 更靠近；
- 不同 task 的 context 更分开。


### 20.7 为什么这套训练比普通残差回归更“像元学习”

如果只是普通回归，目标通常只有：

```text
[state, action] -> residual
```

而这里训练的是：

```text
support set -> context
context + query feature -> query residual
```

而且还要求：

- context 能反映 task 身份；
- context 在同 task 不同轨迹上稳定；
- context 在不同 task 之间可区分；
- query 段的一步状态预测也要更准。

这就是典型的“跨任务学习如何快速适应”的元学习思路。


### 20.8 checkpoint 里到底保存了什么

训练结束后，payload 里会保存这些关键字段：

- `model_type = "amortized_context_meta"`
- `model_state_dict`
- `context_encoder_state_dict`
- `support_window`
- `query_horizon`
- `context_injection`
- `modulation_scale`
- `online_context_ema_decay`
- `online_context_warmup_updates`
- `loss_cfg`
- `context_center`
- `context_scale`

这说明 checkpoint 保存的不只是网络权重，还保存了：

- 这张网该怎么用；
- support 多长；
- context 如何注入；
- 在线阶段如何平滑更新 context；
- context 是如何归一化的。

这也是为什么部署脚本可以直接从 checkpoint 里恢复完整的 meta 运行配置。


## 21. 增量补充：在线闭环部署时每一步到底发生了什么

这一部分对应 `quadrotor3D_common.py` 里的 `run_tracking(...)`。

### 21.1 三种模式在初始化阶段分别做什么

#### `nominal`

- 创建名义动力学模型。
- 建立 acados 求解器。
- 不加载任何神经网络。

#### `lightmlp`

- 创建普通 `MLP` 残差网络。
- 用 `L4CasADi` 包装它。
- 建立“名义模型 + 神经网络残差”的 learned dynamics。
- 在线用最近数据直接更新整张网络。

#### `meta`

- 从 checkpoint 加载 `ContextResidualMLP`
- 同时加载 `SupportSetEncoder`
- 初始 context 先 reset
- 再通过 `L4CasADi` 包装残差网络
- 最后建立 learned dynamics 和 acados solver

所以 `meta` 和 `lightmlp` 的最大区别不是“有没有神经网络”，而是：

- `lightmlp` 在线更新的是整张网络参数；
- `meta` 在线主要更新的是 context。


### 21.2 为什么 PyTorch 网络能进 MPC

这一步的桥梁是 `L4CasADi`。

可以把流程理解成：

1. PyTorch 残差网络先交给 `L4CasADi`。
2. `L4CasADi` 把它包装成 CasADi/外部函数可调用形式。
3. `Quadrotor3DLearnedDynamics` 在构建动力学时，把残差项接到 `f_nominal` 后面。
4. `MPC` 再把这个 learned dynamics 交给 acados。

也就是：

```text
PyTorch residual model
    -> L4CasADi wrapper
    -> CasADi external function
    -> acados OCP model
    -> MPC online optimization
```

这就是“learning model 被嵌进传统优化控制器”的关键工程链路。


### 21.3 在线控制主循环，按时间顺序看一遍

每个控制周期，`run_tracking(...)` 大致做下面几件事。

#### 第 1 步：生成未来参考

对 `k = 0 ... N-1`：

```text
x_ref_k = reference_state(current_time + k * t_horizon / n_horizon, reference_cfg)
```

然后把 `yref_k` 设成：

```text
[x_ref_k, 0, 0, 0, 0]
```

终端点再单独设置一次 `yref_N`。


#### 第 2 步：把当前状态固定成优化初值

代码里是：

```text
solver.set(0, "lbx", xt)
solver.set(0, "ubx", xt)
```

意思就是：

> 预测地平线第一个节点必须等于当前真实状态。

这在 MPC 里是标准做法。


#### 第 3 步：求解 OCP，拿第一个控制量

```text
status = solver.solve()
ut = solver.get(0, "u")
```

MPC 会算出一整串未来控制，但 receding horizon 只执行第一个。


#### 第 4 步：真实环境执行这个控制

```text
next_obs = env.step(ut)
xt <- next_state
```

这一步会把系统真正推进到下一拍。


#### 第 5 步：如果是 learned 方法，就把这一拍转成监督样本

代码会缓存：

```text
feature_buffer.append([prev_state, action])
target_buffer.append(finite_difference_targets(...))
```

也就是：

- 输入缓冲区存 `[state_t, u_t]`
- 标签缓冲区存 `residual_t`

所以在线适配并不是“猜”，而是不断把新闭环数据重新加工成监督信号。


### 21.4 meta 方法在线是如何更新 context 的

当前主版本 `model_type = "amortized_context_meta"`，所以在线更新最核心的 4 步是：

1. 从最近 `batch_size` 条数据中取一个 support batch。
2. 用 `context_encoder(x_batch, y_batch)` 编出新 context。
3. 如果已经过了 warmup，就对新旧 context 做 EMA 平滑。
4. 把新 context 写回 `residual_mlp`，再调用 `l4c_residual.update(...)` 同步给 MPC。

公式上可以写成：

```text
c_new = encoder(X_recent, Y_recent)
```

如果已经超过 warmup 次数，则：

```text
c_used = alpha * c_old + (1 - alpha) * c_new
```

其中：

- `alpha = online_context_ema_decay`
- 当前训练脚本保存到 checkpoint 的默认值是 `0.72`

最后：

```text
residual_mlp.set_context(c_used)
l4c_residual.update(residual_mlp)
```

这一步非常关键。因为如果只更新 PyTorch 里的 context，不把它同步到 `L4CasADi`，MPC 里面看到的还是旧模型。


### 21.5 为什么要加 EMA 平滑

因为在线每次只用最近一小段数据编码 context，如果完全不平滑，context 可能会随着短时噪声和局部轨迹片段抖动。

加了 EMA 之后，相当于做：

```text
新 context = 一部分旧估计 + 一部分新证据
```

这样：

- 更新更稳；
- 更不容易把短时偶然波动误当成“动力学真的变了”；
- 更适合接在 MPC 这种实时闭环系统后面。


### 21.6 `lightmlp` 为什么更慢、更容易不稳

`lightmlp` 的在线更新方式是：

1. 先把网络参数 `requires_grad = True`
2. 用最近 batch 做若干步梯度下降
3. 再把参数冻结
4. 更新 `L4CasADi` 包装模型

也就是说，它在线更新的是整张网络参数，而不是一个小小的 context。

这会带来两个典型问题：

- 更新变量很多，容易抖。
- 每次更新计算量更大。

所以这个项目里 `meta` 的优势并不神秘，本质上就是：

> 把“该怎么适应不同任务”这件事尽量挪到离线阶段学好，在线只做快速识别和小幅调整。


### 21.7 输入约束不是拍脑袋写的常数

`control_bounds(gym_env)` 里，最小/最大推力来自仿真器的电机模型：

```text
thrust = KF * (PWM2RPM_SCALE * PWM + PWM2RPM_CONST)^2
```

再分别取 `MIN_PWM` 和 `MAX_PWM` 对应的推力上下界。

这说明 MPC 的控制约束和仿真器本身是一致的，而不是随便写一个抽象范围。


### 21.8 acados 求解器配置说明了什么

代码里主要配置是：

- `qp_solver = FULL_CONDENSING_HPIPM`
- `hessian_approx = GAUSS_NEWTON`
- `integrator_type = ERK`
- `nlp_solver_type = SQP_RTI`

对初学者来说，不需要立刻把这些缩写背下来，只要知道：

- 这是一个偏实时控制的 OCP 求解配置；
- 目标是在每个控制周期里尽快给出一个可用解；
- `SQP_RTI` 很常见于实时 MPC 场景。


## 22. 增量补充：把几个最容易卡住的新手问题一次讲清楚

### 22.1 为什么任务定义成“惯性参数组合”，而不是“一条轨迹”

因为这个项目想学的是动力学身份，而不是运动图案。

如果把 task 定义成不同轨迹，那模型学到的很可能是：

- 圆轨迹长什么样
- 8 字轨迹长什么样

而不是：

- 这台飞机更重还是更轻
- 哪个轴转起来更费劲

所以这里把：

```text
mass_ratio, ixx_ratio, iyy_ratio, izz_ratio
```

作为 task 的本体，是非常关键的设计选择。


### 22.2 为什么只学 6 个残差，不学满 12 个

因为不是所有通道都同样“不确定”。

像下面这些关系：

```text
d/dt x = x_dot
d/dt y = y_dot
d/dt z = z_dot
```

本来就没什么好学的。

欧拉角运动学虽然非线性，但结构也相对明确。真正最容易因为质量、惯量失配而发生明显误差的，是：

- 平动加速度
- 角加速度

所以项目把学习能力集中在最值得学的地方，这比“所有通道都让网络重学一遍”更高效。


### 22.3 为什么 support window 是 32，query horizon 是 16

因为：

- `dt = 0.02 s`
- `32` 步大约是 `0.64 s`
- `16` 步大约是 `0.32 s`

这说明设计目标不是拿很长历史做慢悠悠的系统辨识，而是：

> 只靠最近不到 1 秒的短窗口，就快速判断当前系统属于哪一种动力学条件。


### 22.4 为什么 query state loss 只做一步滚动，不做长时 rollout

一步滚动的好处是：

- 监督信号清晰；
- 误差不会被长时滚动迅速放大；
- 训练更稳；
- 还能直接约束“残差是否真的改善局部动力学预测”。

如果一开始就做很长 rollout loss，训练难度会明显提高。


### 22.5 为什么说这个项目更像“残差元学习 MPC”，而不是“神经网络控制器”

因为最终执行控制的核心仍然是 MPC：

- 参考轨迹由 MPC 跟踪；
- 代价函数和约束由 MPC 处理；
- 优化器求出每一步控制输入；
- 神经网络只是改进了动力学预测模型。

所以这类方法更准确的名字是：

```text
learning-enhanced MPC
或者
residual-learning MPC
```

而不是端到端 policy learning。


## 23. 增量补充：如果你要继续读源码，推荐按这个顺序

如果你是第一次真正去看代码，建议按下面顺序，不容易迷路。

### 23.1 第一步先看参考轨迹

先看：

- `reference_state(...)`
- `ReferenceConfig`

目的：

- 搞清楚控制器到底想跟踪什么。


### 23.2 第二步看名义动力学

再看：

- `nominal_dynamics_terms(...)`
- `Quadrotor3DNominalDynamics`

目的：

- 搞清楚 MPC 心里以为飞机是怎么动的。


### 23.3 第三步看数据标签是怎么来的

接着看：

- `DataCollection_Meta.py` 里的 `collect_episode(...)`
- `finite_difference_targets(...)`

目的：

- 搞清楚监督学习标签为什么是 `true - nominal`。


### 23.4 第四步看元训练

重点看：

- `load_task_data(...)`
- `sample_support_pair_query(...)`
- `SupportSetEncoder`
- `ContextResidualMLP`
- `evaluate_meta_loss(...)`
- `main()`

目的：

- 搞清楚 support/query/context/组合损失到底是如何落地的。


### 23.5 第五步看在线闭环

最后看：

- `run_tracking(...)`

目的：

- 搞清楚模型是怎样真正被接进 MPC，并在闭环中在线更新的。


## 24. 再给小白一个“最后总结版”

如果你读到这里还是觉得信息很多，可以只记住下面这 8 句话：

1. 四旋翼真实动力学和控制器内部模型不完全一样，这叫模型失配。
2. MPC 需要模型，所以模型不准时，跟踪就会变差。
3. 这个项目不直接用神经网络替代 MPC，而是只学习“名义模型差了多少”。
4. 网络学习的不是 12 维完整状态导数，而是 6 个最关键的残差通道。
5. 每个元任务对应一组真实质量和惯量参数，而不是一条轨迹。
6. support set 的作用是让模型快速识别“当前这台飞机属于哪一种动力学身份”。
7. 离线训练时既约束残差拟合，也约束一步状态预测和 context 结构。
8. 在线部署时，MPC 仍然负责求控制，元学习模型只负责快速更新预测模型。

如果你把这 8 句话真正想明白了，这个 `Quadrotor_3D_Tracking` 模块的主线就已经吃透了。
