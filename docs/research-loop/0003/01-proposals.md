# R0003 改进提案

## 提案总表

| 稳定 ID | 名称 | 类型 | 状态 |
|---|---|---|---|
| `R0001-P14` | 等 transition 预算的连续 Probe 证据合同 | 测量修复 | 阻断 |
| `R0001-P15` | 结果盲的 Replay 起点选择 | 数据保留修复 | 阻断 |
| `R0001-P16` | Action Probe 设计功效门 | 测量修复 | 拒绝 |
| `R0001-P17` | 同状态配对实际动作干预 | 物理因果诊断 | 接受 |
| `R0001-P11` | 因果 latent proposal-history gate | 训练候选 | 正式确认待运行 |
| `R0001-P05` | 跨 source batch 三臂归因 | 训练候选 | 条件 |
| `R0001-P10` | 安全正例窗口分层采样 | 训练候选 | 延后 |

## `R0001-P14`：等预算连续 Probe

- 每 Episode 固定相同 112 transition。
- 三臂：
  - 7×16 硬切；
  - 连续 112 但固定相同 7 个起点；
  - 连续 112 全部合法起点。
- 只改变测量视图，不增加 Episode、原始 transition 或动作。
- bootstrap 单位保持 source Episode。
- 只有新 seed 上连续全起点全任务/horizon 过线，且绝对 state-action MSE 非劣，才能接受。

## `R0001-P15`：结果盲起点

- physical-salience selector 使用未来运动、动作创新、安全干预和交互结果选窗。
- 仅当 P01 后 h1 仍失败时，对比 salience 与预注册均匀起点。
- 接触、安全、碰撞和交互覆盖不得下降。

## `R0001-P16`：设计功效门

- 使用 P09 固定 state/action 设计矩阵生成 null、planted 和 permutation target。
- 三个设计臂、两个 rho、三个任务、四个 horizon。
- 每种效应 500 trial；每 trial 200 次同步 Episode bootstrap。
- null FPR `<=5%`、permutation FPR `<=5%`、planted power `>=80%`。
- 功效不足只能标记 `inconclusive`，不能把失败改为通过。

## `R0001-P11`：因果 latent proposal-history gate

- P09 分层诊断显示 action latency=1 时 previous proposal 可将 normalized RMSE 从
  0.197/0.529 降到约 0.0205/0.0157。
- 未知 lag 混合下直接拼接历史会伤害 lag=0，因此候选使用不读取真实 latency 的因果 gate。
- 等待可信数据因果门后再实施。

## `R0001-P05`：跨 source batch 三臂归因

- A：重复同一窗口；
- B：同 source 不同窗口；
- C：跨 source 不同窗口。
- 只有 C 稳定优于 B，才支持原跨 source 假设。
- 仅在数据因果门可信且世界模型动作利用仍失败时触发。

## `R0001-P10`：安全正例分层

- 仅在 P04 后执行 RMSE 可表示、但安全 recall/PR-AUC 仍失败时触发。
- P04 若已解决动作执行失败，则 `rejected without run`。

## `R0001-P17`：同状态配对实际动作干预

- 类型：训练前物理因果诊断，不是能力改进。
- 证据：
  - P16 证明现有 state-nuisance ridge 在高相关长 horizon 下功效不足；
  - 三个正式任务从同 seed、同初始 `PhysicalStateSnapshot` 重置后，同动作分支在
    proposal、实际 action、state、reward、event 和最终 snapshot 上逐元素一致；
  - 任务盲 `+d/-d` Rademacher 运动动作经原 plant 和安全层后，实际归一化动作差 RMS
    约 0.267、方向余弦 1.0，三个任务均无安全干预或严重碰撞，并在 1/4/8/16 步产生
    非零物理状态差。
- 假设：从相同初始物理状态出发，随机分配的 `+d/-d` 动作符号通过实际 plant action
  差稳定改变后续可控状态，可直接证明动作的增量物理因果效应。
- 最小验证：
  - 只从 Episode 初始 reset 状态分叉，避免未保存的中途 FIFO 或安全历史；
  - plus、minus、sham 使用相同 seed、snapshot、随机化和预算；
  - horizon 从实际 plant action 差首次非零开始；
  - snapshot 和物理 state 只作离线 outcome；
  - Episode/seed 是随机化单位。
- 主要门：
  - snapshot/sham 逐元素重放一致；
  - actual first-stage 非弱且方向对称；
  - sham family-wise FPR 单侧 95% 上界 `<=5%`；
  - blind-injection family-wise power 单侧 95% 下界 `>=80%`；
  - 三任务×四 horizon 的确认性 family 经 Holm 后全部通过；
  - 不删除安全改写或零响应样本。
- 通过后只解锁可信训练前数据因果证据，再单独路由 P11/P05；不得宣称任务能力。

## `R0001-P18`：动作序列编码可解码性

- 从同一 snapshot 执行边际匹配、时序编码不同的动作序列并解码标签。
- 两名筛选 Agent 认为该设计易利用时序生成器、安全裁剪和动作路径指纹，不能直接证明
  任务价值或因果控制。
- 状态：`rejected`，仅可作不影响决策的探索性负控。

## P17 后候选路由

### `R0001-P11`：因果 plant FIFO 与安全 rewrite 分解

- P17 已证明实际 plant action 对三任务、四个 horizon 都有稳定物理因果效应。
- P09 显示 action latency=1 时，当前 proposal 无法表示当前 actual action；使用前一
  proposal 可把 normalized RMSE 从 0.197/0.529 降到约 0.0205/0.0157。
- 直接拼接 proposal history 虽改善 lag1，却显著伤害 lag0，说明需要显式识别 plant
  latency，而不是一个共享线性 residual。
- 假设：用过去 proposal 与过去 applied feedback 因果估计固定 Episode 的 actuator gain
  和 lag，再从 proposal FIFO 产生 plant action baseline；学习头只负责安全 rewrite，可将
  确定性 plant 变换与稀有安全事件解耦。
- 最小验证先只评估非干预动作，不修改正式世界模型：
  - P09 训练 latency 0/1；
  - 独立短物理集确认 latency 1/2/3；
  - 不把真实 latency 或 actuator scale作为输入；
  - 报告冷启动与稳定阶段。
- 通过后再单独实现正式模型；安全 recall/PR-AUC 仍失败时才触发 P10。

### `R0001-P05`：跨 source batch 三臂归因

- 次选。A 为重复同一窗口，B 为同 source 不同窗口，C 为跨 source 不同窗口。
- 只有 C 稳定优于 B 才支持跨 Episode 假设。
- P11 被否定或世界模型 action-shuffle 仍失败时启动。

### `R0001-P10`：安全正例分层

- 只在 P11 已使非干预/干预动作可表示、但自然 holdout 的安全 recall/PR-AUC 仍失败时启动。
- 不与 P11 或 P05 首次捆绑。
