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
| `R0001-P19` | `diagnostic rejected` |
| `R0001-P20` | `diagnostic rejected` |
| `R0001-P21` | `diagnostic rejected` |
| `R0001-P22` | `rejected without run, dependency failed` |
| `R0001-P23` | `diagnostic rejected` |
| `R0001-P24` | `diagnostic rejected, both heads not_localized` |
| `R0001-P25a` | `rejected without run` |
| `R0001-P25b` | `rejected without run` |
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
9. raw dynamics KL 中位数 8.05，current free-nats=1.0 仍有强 prior/action 梯度，
   free-nats 死区假设被否定。
10. canonical action 的 variation contribution 相对 raw 稳定放大 2.19～3.04 倍，但
    raw/stochastic ratio 只有 17/24 Episode 低于 0.20，未达到 20/24 一致性门槛；
    action-scale 假设被否定，不进入 normalization smoke。
11. action effect 在全部 72 个 shift 都能进入 RSSM transition，但只有 23/72 出现
    `<0.50` 的相邻层 retention；P21 仅 8/24 Episode 通过，且 tidy living room 为
    0/12，deterministic shortcut 联合假设被否定。
12. prior argmax flip 虽仅 0%～2.73%，hard one-hot effect 相对同一 probability scale
    通常被放大而非抹除；P23 仅 5/24 Episode 通过，argmax 瓶颈被否定。与此同时 hard
    decoder feature effect 24/24 过线，允许重筛 P24。
13. P24-R2 中 visual/proprio 的 decoder input/output effect 都 24/24 存活，但系统性低
    retention 均未定位；两头只各有 1/24 Episode 定位到 feature→linear，decoder low-gain
    假设被否定。两头均按冻结决策表进入 `not_localized`，允许分头重筛 P25。
14. P25a 在运行前被更严格的 per-shift output guard 拒绝：shift=1 仅 visual 23/24、
    proprio 21/24，未达到每 shift 24/24。P25a 未运行，P25b 依赖自动拒绝。

## 新基线

- 能力基线不变：没有 causality-qualified deployment。
- 测量结论：data probe 失败为 `inconclusive`，不是可信能力否定。
- P17 取代低功效 data probe，成为训练前数据动作物理因果证据。
- 不允许绕过门槛直接训练，也不允许通过降低门槛修复功效。

## 下一阶段问题

P17 已证明训练数据中的实际 plant action 具有物理因果性，但 P05/P06/P19/P20/P21/P23/P24
依次排除了 batch source、posterior overshooting、free-nats、纯 action scale、普遍
GRU/prior shortcut、argmax 离散化和 decoder 系统低 gain 作为充分解释。

下一轮主要问题：

1. 为什么少数 shift=1/任务 Episode 的 decoded output effect 低于门槛；
2. 现有 absolute reconstruction 与 KL 目标是否没有稳定奖励 action-discriminative
   方向；
3. stale-frame 修复和 P10 安全正例分层仍保持独立，不与能力候选捆绑。

新一轮在 `docs/research-loop/0004/` 建立；`0001`、`0002`、`0003` 全部冻结只读。

## 清理

- 未删除 P09/P16 正式报告与 trial。
- 未删除 R0001 v4 负基线。
- 本轮未执行清理 Agent。
- 清理前可用空间：`79,829,604 KiB`。
- 清理后可用空间：`79,829,604 KiB`。
- 删除内容：无。P20/P21/P23/P24 的 invalid、failed、complete 产物、manifest、日志和
  dispatch 证据全部保留；P25a/P25b 没有 run 产物。
