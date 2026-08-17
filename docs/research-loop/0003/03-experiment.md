# R0003 冻结实验

## `R0001-P16` 冻结合同

- 实现提交：`03de3fcfaa151efc6dd92ae2081ca3ad08732798`
- run ID：`r0001-p16-probe-power-s20260916`
- 历史产物路径：
  `runs/research-loop/0001/r0001-p16-probe-power-s20260916`

路径中的 `0001` 是迁移前的历史 artifact 路径，不表示文档轮次。

### 固定设计

- P09 96 Episode 作为设计矩阵；
- 每 Episode 前 112 transition；
- 三臂：
  - `fragmented_7x16`
  - `continuous_same_7_starts`
  - `continuous_all_starts`
- rho：0.50、0.96；
- 三任务；
- horizon：1/4/8/16；
- ridge：`1e-3`；
- bootstrap：每 trial 200 samples，同步 Episode 合同；
- 每种条件 500 trial。

### 合成条件

- null：state signal + AR(1) noise；
- planted：state signal + 0.5×action signal + noise；
- permutation：使用置换 action signal；
- state signal RMS 1.0；
- noise RMS 0.5，AR rho 0.8；
- 系数、噪声、bootstrap seed 按冻结公式生成；
- 不读取真实 target、reward 或安全标签。

### 接受标准

每个可用设计臂：

- null FPR `<=0.05`；
- permutation FPR `<=0.05`；
- planted power `>=0.80`。

连续全起点自身必须合格，才能路由到 `R0001-P14`。

## 资源与命令

```bash
.venv/bin/python -m hwr.apps.evaluate_action_probe_power
```

- CPU；
- 无模型训练；
- Host 一次性 runner，回调前移除自身服务；
- 保留全部 trial 和 manifest hash。

## `R0001-P17` 冻结入口

- 任务：三个正式 MuJoCo 家务任务。
- 分叉状态：每个 seed 的 Episode 初始 reset snapshot。
- 运动方向：14 维 Rademacher，seed
  `20261017 + sorted_task_index * 104729 + episode_index * 1000003`。
- 归一化动作幅值：每维 `0.5 / sqrt(14)`，再使用正式 action scaling；两个 gripper
  固定为 reset 时当前位置。
- 分支：
  - plus：`+d`
  - minus：`-d`
  - sham-a / sham-b：同一 `+d`
- 每个分支最多 17 control step，评分 horizon 为 1/4/8/16。
- horizon 0：plus/minus actual normalized plant action 差首次 L2 norm `>1e-8` 的 step。
- 所有分支经过原 actuator scale、action latency、预测安全和硬安全层。
- 保存完整 proposal、actual plant action、physical snapshot、visible proprioception、
  safety event、碰撞和终止证据。

### 预检

1. 同动作 sham 的 proposal、actual action、state、reward、event 和最终 snapshot
   逐元素一致。
2. actual normalized action-difference RMS `>=0.10`。
3. actual action-difference 与冻结方向的每步 cosine `>=0.95`。
4. plus/minus first-stage RMS 相对差 `<=5%`。
5. 任一分支严重碰撞、提前终止或安全改写率超过 5%：`inconclusive`。

### 统计

- 主 outcome：固定 16 维可控物理状态
  （双臂关节速度、双 gripper 位置、base twist）的 plus-minus 差。
- 物理状态来自每个 control step 后 `backend.observe()` 的当前无延迟 proprioception；
  不使用返回给策略的延迟 observation。
- 每个 Episode、每个 horizon 的 first-stage：
  从 actual plant action 首次产生差异的 step 开始，对该 horizon 内 14 维 normalized
  actual action difference 求均值。
- 每个 Episode、每个 horizon 的 outcome：
  plus/minus 当前无延迟可控状态之差。
- 主 estimand：同状态配对后的归一化 action–state cross-moment：
  `||X^T Y / N||_F^2 / (mean(||X||^2) * mean(||Y||^2))`。
- `X` 为 Episode×14 的 actual first-stage，`Y` 为 Episode×16 的 physical outcome；
  两者都不拟合 state nuisance。
- 正式数据在 Episode 层对 `Y` 做 999 次随机 sign-flip，重算完整统计量。
- family：三任务×四 horizon，共 12 个确认性检验。
- 每分区 p-value：
  `(1 + permutation_stat >= observed_stat 的次数) / 1000`。
- 多重检验：Holm，family-wise alpha 0.05。
- sham 和 blind injection 使用独立预检 seed，不进入正式确认 family。

### Seed 与样本

- 预检 seed：64 个，
  `20261101 + episode_index * 104729`，`episode_index = 0..63`。
- 正式确认 seed：64 个，
  `620261101 + episode_index * 104729`，`episode_index = 0..63`。
- 两组与 R0001、R0002、P09、训练和正式评测 seed 互斥。
- 三个任务使用同一 seed bank，但报告和确认检验按任务独立。
- 分支执行顺序由
  `20261017 + task_index * 1000003 + episode_index * 1009`
  固定置换，避免顺序偏差。

### Sham 与功效预检

- 64 个预检 seed 全部运行 sham-a/sham-b。
- sham divergence 定义为 proposal、actual action、当前无延迟状态、reward、event 或最终
  runtime snapshot 任一不相等。
- 观测 divergence 为 0/64 时，Clopper-Pearson 单侧 95% 上界必须 `<=0.05`。
- blind injection 使用实际 preflight first-stage和相同 cross-moment/Holm 统计，运行
  1,000 个 null 与 1,000 个 planted trial。
- 每个 task×horizon 的响应矩阵由 seed
  `20261017 + task_index * 104729 + horizon * 1009`
  生成，列归一化。
- null outcome：RMS 0.5 的独立高斯噪声。
- planted outcome：
  `0.5 * normalized_first_stage @ response_matrix + null_noise`。
- 每分区用 1,000 个经验 null 统计校准 p-value；null trial 使用 leave-one-out 校准。
- 全 12 分区经 Holm 通过才算 trial passed。
- null family-wise FPR 的 Clopper-Pearson 单侧 95% 上界 `<=0.05`。
- planted family-wise power 的 Clopper-Pearson 单侧 95% 下界 `>=0.80`。
- 不得根据预检结果修改动作幅值、注入强度、seed、fold、ridge、permutation 数或检验 family。

### 正式接受

1. snapshot/sham、first-stage 和 blind injection 预检全部通过；
2. 正式 64 seed 全部产物、失败和安全改写均报告；
3. 12 个分区 observed statistic 均为正；
4. 12 个分区经 Holm 后全部显著；
5. 每任务 actual action-difference RMS `>=0.10`；
6. 每个评分 step 的方向 cosine `>=0.95`；
7. 安全改写率 `<=0.05`，严重碰撞和提前终止均为 0；
8. 无非有限值、弱 first-stage 或缺失 horizon。

## `R0001-P11` head-only 冻结入口

### 问题

> 在不读取真实 action latency、actuator scale 或未来 action 的条件下，过去 proposal 与
> applied-action feedback 是否足以因果识别 plant FIFO，并把非干预 actual action 的
> normalized RMSE 降到 0.05 以下？

### 唯一变量

- 对照：current proposal residual。
- 候选：因果 plant estimator：
  - 保存最近 4 个 proposal；
  - 用过去 16 个非干预 feedback 对每个 lag 0/1/2/3 拟合一个共享 scalar gain；
  - 选择过去 feedback 误差最小的 lag；
  - 当前预测为该 lag 的 proposal × gain；
  - gripper 使用同一 lag proposal，不拟合连续 gain。
- 不训练视觉、RSSM、Actor、安全分类或世界模型主干。

### 数据

- 训练/开发：P09 96 Episode，rho 0.50/0.96，latency 0/1。
- 正式确认：新 MuJoCo evaluation-profile 短物理数据：
  - 三个正式任务；
  - rho 0.50/0.96；
  - 只强制 action latency 1/2/3，其余 evaluation-profile 随机化保持原 seed 采样；
  - 每个任务×rho×latency 使用同一组 8 seed：
    `720261101, 720365830, 720470559, 720575288, 720680017, 720784746,
    720889475, 720994204`；
  - 每 Episode 64 个 proposal/feedback transition，共 144 Episode、9,216 transition；
  - task-blind correlated random source，gripper flip probability 0.05；
  - run ID：`r0003-p11-causal-plant-s20261101`。
- 真实 latency/scale 只用于结果分层，不输入 estimator。
- 确认采集必须记录 action-latency-only override provenance；不得改写 actuator scale、
  observation latency、场景随机化或动作序列。

### 固定估计器

- action 维归一化使用正式上下界之差。
- 对当前 step `t`，lag 候选都只使用过去 feedback step `s<t`，并要求 `s>=3`，
  从而四个候选共享完全相同的可用反馈索引。
- 从最近 16 个无安全干预的共享反馈 step，对每个 lag 0/1/2/3：
  - 连续 14 维用无截距最小二乘拟合一个共享 scalar gain；
  - 用该 gain 与对应 lag proposal 预测连续维；
  - gripper 直接使用同一 lag proposal；
  - 最终预测裁剪到正式 action bounds；
  - 以 16 维 normalized feedback MSE 选最小候选；并列时选较小 lag。
- 在没有 16 个共同反馈前，候选回退为裁剪后的 current proposal。
- 因此主要稳定阶段从每 Episode 第 20 个动作开始；此前 19 步全部作为冷启动报告。
- 固定负控：按 Episode seed
  `11000003 + seed` 对 proposal 时间轴作无固定点确定性 derangement，再运行完全相同的
  estimator；不得改动 applied feedback、干预标签或真实目标。

### 指标

- 每任务、每 rho、每 latency：
  - normalized RMSE `<=0.05`；
  - latency0 相对 current baseline 绝对退化 `<=0.005`；
  - out-of-bounds rate 0。
- 冷启动前 19 步单独报告，不进入主要接受，但不得出现非有限值或越界。
- 每个分区必须保留全部 8 Episode；若安全干预导致任一步没有 16 条共同无干预反馈，
  该步不进入稳定阶段并必须计数。
- latency0 守护只在 P09 开发集判定；正式确认集不含 latency0。
- 历史 proposal derangement 的每个正式确认分区 stable normalized RMSE 必须比候选
  至少高 0.05。
- 严重碰撞、提前终止、artifact 缺失、非有限值或 provenance 不完整均使实验
  `inconclusive`，不能删除 Episode 后判定。

### 命令与资源

```bash
.venv/bin/python -m hwr.apps.evaluate_causal_plant_estimator
```

- CPU/MuJoCo；不训练模型，不使用加速器。
- 正式确认只允许从干净、已提交源码运行。
- 保存 144 个 Episode artifact、P09 输入 manifest hash、完整 report 和 artifact hash。

### 路由

- head-only 通过：实施正式 plant/safety 分解。
- lag2/3 不通过或 lag0 回归：P11 `rejected`，转 P05。
- 不得根据确认结果修改 history、warmup、gain、lag 候选或门槛。

## P17 路由

- 预检失败：P17 `inconclusive`，不运行正式确认。
- 正式确认全部通过：P17 `accepted as training-data causality evidence`，解冻 P11。
- 任一任务/horizon 在功效合格前提下失败：P17 `rejected`，重审随机激励。
