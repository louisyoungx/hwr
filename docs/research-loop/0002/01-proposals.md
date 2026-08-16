# R0002 改进提案

## 提案总表

| 稳定 ID | 名称 | 类型 | 状态 |
|---|---|---|---|
| `R0001-P09` | 观测转移动作的物理时间对齐 | 测量诊断 | 入选 |
| `R0001-P10` | 安全干预正例窗口分层采样 | 训练候选 | 条件保留 |
| `R0001-P11` | 动作执行头的因果 proposal 历史 | 训练候选 | 延后 |
| `R0001-P12` | 未见语言的评测侧冻结物化 | 平台修复 | 入选 |
| `R0001-P13` | 闭环安全干预负担门 | 评测修复 | 入选 |

## `R0001-P09`：观测转移动作的物理时间对齐

### 证据与假设

- 正式训练 observation latency 为 0/1 步，评测扩大到 1～3 步。
- 后端返回当前物理步实际 plant action，但可见 observation 可能来自更早物理状态。
- Replay 没有保存 observation lag，旧 probe 默认 `action[t]` 连接相邻可见 observation。
- 对 P01 v4 事后重建 lag 的离线诊断让三任务全部 horizon 过线，但 lag=1、horizon=16
  因旧 16-transition shard 缺前置动作而丢样本，不能接受。

假设：显式保存 observation lag 与前置实际动作，并保持 Episode 数和样本数不变，可检验
旧 data probe 失败是否由时序错位造成。

### 最小验证

- 三个正式任务；
- 每任务 8 training + 8 holdout Episode；
- 每条 128 transition；
- observation lag 预注册交替 0/1；
- 主诊断 `rho=0.96`，确认组 `rho=0.50`；
- old-index 与 lag-aligned 双报告；
- 使用 R0001-P04 的同步 Episode bootstrap。

### 主要与守护指标

- 全任务、全 horizon ratio `>=1.05`、同步 `p05>=1.01`；
- lag=0 新旧结果 `1e-10` 内一致；
- lag=1 不丢 Episode；
- lag=1 每个 horizon state-action MSE 至少下降 10%；
- 不改模型、任务、安全层、动作幅值或门槛。

## `R0001-P10`：安全干预正例窗口分层采样

- 证据：P01 Replay 只有约 37/2688 个安全干预 transition、7/168 个正例窗口；
  三任务独立平衡留出 recall 均为 0。
- 假设：普通 batch 的安全正例窗口比例固定为 0.25，可避免全负局部最优。
- 只改 batch sampler，不改 loss、模型、目标或安全层。
- 仅在测量合同稳定后，使用冻结 Replay、相同 update 和三个优化 seed 验证。
- 不得与 P03 或跨 source batch 首次捆绑。

## `R0001-P11`：动作执行头的因果 proposal 历史

- 证据：plant action latency 为训练 0/1、评测 1～3 步；当前执行头只接收当前 proposal。
- 假设：只含过去和当前 proposal 的有界 FIFO 可表示未知 plant latency。
- 不允许未来 proposal、真实 latency 标签或仿真器状态。
- 在未见 latency 2/3 上确认；latency=0 不得回归。
- 与 P02 拆分。

## `R0001-P12`：未见语言的评测侧冻结物化

- 三任务各 3 条、共 9 条 evaluation instruction 在首个物理 Episode 前物化。
- 使用训练时同一锁定 Qwen3 encoder。
- 只写入单次评测目录的 evaluation-only cache 和 manifest。
- 训练 Replay、训练 feature index、checkpoint、deployment 保持只读。
- 已见 embedding 与确定性动作逐元素一致。
- 该项不代表语言泛化提升。

## `R0001-P13`：闭环安全干预负担门

- Episode 干预率：
  `safety_interventions / max(steps, 1)`。
- 每任务 normal mode 同时满足：
  - empirical p95 `<=0.01`；
  - 2,000 次 Episode bootstrap 的 95% upper `<=0.02`；
  - maximum Episode rate `<=0.05`。
- 与成功率、Wilson 下界、零严重碰撞、稳定、并发接触和左右臂消融联合。
- 不修改安全层、策略或训练。
