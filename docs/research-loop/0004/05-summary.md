# R0004 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P26` | `deferred` |
| `R0001-P27` | `rejected without implementation` |
| `R0001-P28` | `rejected before implementation after independent re-review` |
| `R0001-P29` | `accepted as runtime contract diagnostic` |
| `R0001-P30` | `rejected without run, duplicate of R0001-P09` |

## 关键发现

1. 当前 20Hz action envelope 从 visible observation timestamp 起只允许 100ms。
2. observation latency 0/1/2 的最大 age 为 0/50/100ms，均合法；latency 3 为 150ms，
   从 64-step Episode 的 step 3 起产生 61 次 `outside_validity_window`。
3. 合成时间线与 P11 的 144 Episode 完全一致：
   - latency 1：18 Episode、0 intervention；
   - latency 2：90 Episode、0 intervention；
   - latency 3：36 Episode、每个 61 次、总计 2,196 次。
4. 这不是随机 runtime bug，而是 evaluation latency=3 与当前
   observation-timestamp-based 100ms action contract 的结构性不相容。
5. P29 不改变 P11 的 `rejected` 结论，不允许延长窗口、移除 latency 或用 latest bundle
   绕过安全层。
6. P30 与 R0001-P09 的显式 observation-lag / actual-action 对齐实验重复；P09 已正式否定
   单一对齐跨 horizon 稳定改善，因此 P30 不实施、不重标 Replay。
7. P28 的 fresh-head 草案在两次独立复审中暴露：
   - successor posterior oracle 对 prior 不公平；
   - `sample=False` 与拟议训练路径不匹配；
   - action 时间重配负例仍可能带位置/幅值/时距指纹并复刻正式 audit；
   - source split、metric 和统计合同仍有自由度；
   - head-only 可解码性不能直接授权 world-model objective。
8. 为避免在多轮设计反馈后继续扫描负例、采样、预算和门槛，P28 在实现前拒绝。
9. 本轮没有启动正式训练，没有新任务成功，也没有能力改善。

## 新基线

- 能力基线不变：仍没有 causality-qualified deployment。
- 数据因果证据仍为 P17。
- Action Probe 低功效结论不变。
- 新增运行时合同证据：
  latency=3 是当前安全动作时效合同下的不可执行域，而不是可通过放宽 safety 制造提升的
  评测噪声。
- P29 不合入能力模型，只保留诊断实现与证据。

## 下一轮问题

下一轮不得直接复活当前 P28 负例草案。优先考虑以下互斥方向之一：

1. 是否能设计不依赖任何 action-shuffle/derangement 形状、带信息匹配 control 的训练目标
   诊断；
2. 若 latency=3 必须是正式可执行域，如何设计单变量的 latency-aware action scheduling，
   同时保持真正 stale 输入仍被独立安全层拒绝；
3. 若 latency=3 被定义为当前系统安全不可执行域，如何在不削弱评测难度的前提下冻结新的
   闭环域支持声明。

## 提交与产物

- R0004 冻结文档：`ce63f95235cae7df4265db3101edc6e9c821c3db`
- P29 实现：`b34716d32cccfe5e619b18aaecaaeda8954a765a`
- P29 结果文档：`057b34a0b5585f20e0fb9b77b4d087718a96459a`
- P29 run：
  `runs/research-loop/0004/r0004-p29-stale-validity-s20262901`
- report SHA-256：
  `7729eed53e034bef5a9d5c50bd1d87d6025370b290226f3055c480f3014274fb`
- manifest SHA-256：
  `41495b1a1efa9f24b65e8c5359bc98cfa28ecee4dbdbc20a5c20d37aa7900cef`
- 全量 pytest、Python size、architecture、training semantics 与 physics integrity 均通过。

## 清理

- 清理前可用空间：`131,178,496 KiB`。
- 清理后可用空间：`131,178,496 KiB`。
- 删除内容：无。
- 未启动清理 Agent。
- P11、P29、R0001 v4、P09/P16/P17/P20/P21/P23/P24 的数据、checkpoint、report、
  manifest、日志和失败证据全部保留。
