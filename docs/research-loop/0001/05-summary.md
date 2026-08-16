# R0001 轮次总结

## 轮次边界

- 起始提交：`fc10cca515a541b0fcb94c4284b13f800246f896`
- 结束提交：`b665c9d96049d80e1951c6a8e941af4695d23d2a`
- 主要问题：当前谱系能否在 24 Episode 校准预算内形成可信动作可辨识性并解锁探索 Actor。
- 本轮只包含：
  - `R0001-P01` 当前谱系无行为改动校准基线；
  - `R0001-P04` 跨 horizon 同步 Episode bootstrap 评测修复。
- 观测延迟、未见语言、安全负担和后续功效实验属于后续轮次，已迁出本目录。

## 结论

| ID | 结论 | 说明 |
|---|---|---|
| `R0001-P01` | `rejected` | v4 在 24 Episode、1,600 update 判定点形成完整负基线，但未通过数据动作可辨识、世界模型动作利用和动作执行验证。 |
| `R0001-P04` | `accepted as evaluation repair` | 同一任务的全部 horizon 在每个 bootstrap replicate 中共享 Episode multiplicity；点估计、MSE、Episode 列表和门槛不变。 |

v1～v3 因平台缺陷或资源路径问题标记为 `inconclusive`。只有 v4 是本轮可信负基线。

## 基线

- 当前能力基线：没有 causality-qualified deployment，没有未见分布闭环能力通过结果。
- 当前证据基线：`r0001-p01-baseline-v4-s20260812` 是可复核负基线。
- 后续测量合同必须使用 `hwr.foundation-data-action-probe/v4`。
- 不得把 P04 的统计变化解释为能力提升。

## 未解决问题

1. P01 的数据 Probe 失败是否来自动作激励、观测/动作时序、Replay 分片或统计功效。
2. 世界模型 action-shuffle ratio 约为 1 的根因。
3. 动作执行验证三任务 recall 为 0、RMSE 超限的根因。
4. 正式闭环评测尚未具备完整的未见语言和安全干预负担合同。

以上问题从 `docs/research-loop/0002/` 开始处理。

## 清理

- 未删除 P01 v4 Replay、checkpoint、留出、manifest、报告或日志。
- 未删除 v1～v3 的平台失败证据。
- 本轮未执行资源清理，磁盘空间变化不作为结论。
