# R0003 改进提案

## 提案总表

| 稳定 ID | 名称 | 类型 | 状态 |
|---|---|---|---|
| `R0001-P14` | 等 transition 预算的连续 Probe 证据合同 | 测量修复 | 条件 |
| `R0001-P15` | 结果盲的 Replay 起点选择 | 数据保留修复 | 延后 |
| `R0001-P16` | Action Probe 设计功效门 | 测量修复 | 入选 |
| `R0001-P11` | 因果 latent proposal-history gate | 训练候选 | 延后 |
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
