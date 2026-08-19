# R0003 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P16` | `rejected` |
| `R0001-P17` | `accepted as training-data causality evidence` |
| `R0001-P14` | `blocked` |
| `R0001-P15` | `blocked` |
| `R0001-P11` | `rejected` |
| `R0001-P05` | `rejected` |
| `R0001-P06` | `rejected without training` |
| `R0001-P19` | `diagnostic selected` |
| `R0001-P10` | `deferred` |

## 关键发现

1. 7×16 硬分片严重丢失长 horizon 证据。
2. 即使恢复全部连续起点，高相关动作下的线性 Probe 仍只有 5.8% 整体功效。
3. rho=0.50 近满功效，rho=0.96 长时病态是核心失败指纹。
4. 当前 Action Probe 不能作为可靠训练准入证据。
5. 同状态配对物理干预已证明实际 plant action 在三任务、1/4/8/16 步均有稳定因果效应。
6. P11 对非 stale-frame 子集能精确识别 lag1/2/3，但 observation latency=3 会使
   100ms action validity 窗口过期，正式 144 Episode 合同因此拒绝。
7. 跨 source batch 在三个 seed 上没有稳定优于同 source 不同窗口，action-shuffle ratio
   仍约为 1.0，且真实误差与 action execution 守护回归。
8. 真实动作 posterior overshooting 相对 zero/shifted 只有 0.88%/0.06% 优势，不足以支持
   新训练目标。

## 新基线

- 能力基线不变：没有 causality-qualified deployment。
- 测量结论：data probe 失败为 `inconclusive`，不是可信能力否定。
- P17 取代低功效 data probe，成为训练前数据动作物理因果证据。
- 不允许绕过门槛直接训练，也不允许通过降低门槛修复功效。

## 下一阶段问题

需要构建不依赖 state nuisance 拟合的训练前物理因果证据，例如同状态配对实际动作干预。
该问题继续在 `docs/research-loop/0003/` 内推进；`0001` 和 `0002` 保持冻结只读。

P17 已接受，P11/P05/P06 已拒绝。下一步只做 P19 free-nats 梯度死区诊断；先证明当前
1.0 截断确实消除 prior/action 梯度，才允许冻结 0.1 单变量训练候选。P10 与 stale-frame
修复保持独立。

## 清理

- 未删除 P09/P16 正式报告与 trial。
- 未删除 R0001 v4 负基线。
- 本轮未执行清理 Agent。
