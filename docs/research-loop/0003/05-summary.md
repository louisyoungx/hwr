# R0003 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P16` | `rejected` |
| `R0001-P14` | `blocked` |
| `R0001-P15` | `blocked` |
| `R0001-P11` | `deferred` |
| `R0001-P05` | `deferred` |
| `R0001-P10` | `deferred` |

## 关键发现

1. 7×16 硬分片严重丢失长 horizon 证据。
2. 即使恢复全部连续起点，高相关动作下的线性 Probe 仍只有 5.8% 整体功效。
3. rho=0.50 近满功效，rho=0.96 长时病态是核心失败指纹。
4. 当前 Action Probe 不能作为可靠训练准入证据。

## 新基线

- 能力基线不变：没有 causality-qualified deployment。
- 测量结论：data probe 失败为 `inconclusive`，不是可信能力否定。
- 不允许绕过门槛直接训练，也不允许通过降低门槛修复功效。

## 下一阶段问题

需要构建不依赖 state nuisance 拟合的训练前物理因果证据，例如同状态配对实际动作干预。
该问题继续在 `docs/research-loop/0003/` 内推进；`0001` 和 `0002` 保持冻结只读。

当前入选候选为 `R0001-P17`；先执行配对物理干预的 snapshot/sham/injection/first-stage
预检，通过后再做正式确认。

## 清理

- 未删除 P09/P16 正式报告与 trial。
- 未删除 R0001 v4 负基线。
- 本轮未执行清理 Agent。
