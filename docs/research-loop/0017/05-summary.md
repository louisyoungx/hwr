# R0017 轮次总结

## 结论

| ID | 判定 |
|---|---|
| `R0001-P87-E1` | `inconclusive (invalid_design)` |
| `R0001-P85-E1` | `deferred` |
| `R0001-P88` | `deferred` |
| `R0001-P68-E4` | `hard defer` |
| `R0001-P76-E5` | `deferred` |
| `R0001-P76-E6` | `hard defer` |
| `R0001-P86-E1` | `rejected in current form` |

本轮没有训练、policy inference、MuJoCo physical acquisition、B0/B1/B2 action、
contact phase、capability evaluation 或新家务任务成功，能力基线不变。

P87 没有进入 formal evaluator：

- 冻结 C01 要求只把 selector target `18→8` 后整个合同由不可达变为可达；
- 但 living 与 dining 的 choice-opportunity 上限均为 `1`，task floor 仍为 `5`；
- 因而单变量 mutation 后合同仍不可达；
- 候选实现为让 control 通过又删除 task floors、缩小 claim scope，违反冻结合同；
- Solver B 对不可达合同只检查空 assignment，却标记为 exhaustive；
- contradiction verifier 会接受 `required<=available` 的伪 contradiction；
- 独立红队结论为 2 blocker、3 major、1 minor，不允许运行 formal evaluator。

因此本轮结论是**实验设计无效、证据不足**，不是 P87 假设的科学拒绝。formal output
不存在，候选实现已从主基线回退。

## 当前基线

- 能力基线不变：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`
  - 24 Episode；
  - 1,600 update；
  - 0 success；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor 未解锁。
- measurement/design baseline 保持：
  - `R0001-P79-E1`：isolated v2 deterministic mask-ownership correction；
  - `R0001-P83-E1`：consumer-local v2 selection-lineage evidence。
- P87 没有加入 measurement baseline。
- selector、默认 v2 migration、Replay、Actor、世界模型训练和 capability evaluation
  仍未授权。
- 不宣称机器人已学会任何家务任务。

## 关键发现

1. 冻结 bank 中只有 8 个 multi-candidate Episode：
   - living `1`；
   - dining `1`；
   - kitchen `6`。
   旧 P68 `selector_negative >=18/22` 及 task floor `5/5/5` 数学不可达。
2. P76 只用 total `19/22` 与 task floor `4/6/6` 时，可隐藏某 latency pair
   `3/6` 的 coverage collapse；task 分账不能替代 latency-pair 分账。
3. R0017 创新阶段已看到三个 `obs=1/action=1` prefix sentinel：
   - living `45b8ab11...`；
   - dining `7e039594...`；
   - kitchen `c8a3a55e...`。
   三者的 safe-entry 与 wall 已暴露，未来 confirmatory prefix contract必须登记并
   排除相同 outcome 字段，或使用新鲜 holdout；不能继续把完整 22 Episode称为未见。
4. 排除三个已见 sentinel 后，living `obs=1/action=1` 的唯一 nonempty Episode被耗尽；
   当前固定 bank 无法同时支持“排除已见样本”与“12/12 cell均有未见 safe entry”。
5. P85-E1 不能把 wall 改善单因果归于 comparator 替换，因为相对 legacy P68还同时
   改变 v1→v2、24→22、双 replay→单 replay；未来只能称绝对 execution-feasibility
   certificate。
6. P88 的 candidate bytes、support count和三遍历一致不足以唯一证明 coordinate
   ledger；需要人工可知的 self-mask/component/merge 精确 oracle。
7. 当前 `PhysicalStateSnapshot` 不是 full-runtime continuation：
   - 缺 queues、history、servo targets、runtime/task/contact/safety counters、
     cached camera与完整 RNG；
   - reset会清空计数与队列并重置时间。
8. P76 prefix与 P86 restore必须拆分。same-process authoritative entry不需要 restore；
   future clone/search 才需要事件完整 restore gate。
9. no-B2不能由 producer自报：
   - 需要独立 action service能力边界；
   - 只允许 post-selection step `0..399`；
   - 401 mutation必须在 B2 action bytes产生前停止。
10. 一个 contract oracle若只用第二份解析式规则、同源 contradiction verifier或
    预期 verdict controls，仍可能自证。独立性必须体现在不同求解机制与真实 witness
    验证上，而不是模块命名。

## 实现与回退

### 冻结与文档

- R0017 初始化：`8e1002c`
- 提案冻结：`6f34c51`
- 独立筛选：`a349808`
- 实验冻结：`9c07cf6`

### 候选实现

- 实验分支：`exp/R0001-P87-contract-oracle`
- 最终候选提交：
  `485367fe1a4901f69407329af1e25bd7cdf5498b`
- 只包含四个冻结允许文件。

### 主分支

- 候选实现曾以 `9f1cf0d` 合入；
- 红队禁止正式运行后，由 `9eed083` 完整回退；
- 当前主基线不包含 P87 实现。

## 验证

- 候选 focused：`23 passed`。
- Python size：目标四文件和 repository-wide 462 files通过。
- architecture、compileall、`git diff --check`：通过。
- 单一实现提交、四文件 scope、历史 tree：通过。
- 候选完整 pytest：
  - `1,172 passed`；
  - `1 failed`；
  - `11 skipped`；
  - failure ID 与 R0016相同：
    `tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete`；
  - 新增 failure：0；
  - wall `224.15s`；
  - maximum resident set size `1,992,130,560 bytes`。
- 独立红队：2 blocker、3 major、1 minor；formal evaluator不放行。
- formal evaluator：未启动。
- formal artifact：不存在。

## 下一轮问题

下一轮必须重新创新和筛选，不自动继承 P87 或 P76。优先问题：

1. 是否重提 `R0001-P87-E2`：
   - C01 只验证 `required_gt_eligible` contradiction在 `18→8` 后消失，不要求整个
     含 task floors 的合同变为可达；或者单独建立 overall-only fixture；
   - C02 保持同一 claim scope，仅增加 latency floor；
   - Solver B 真正穷尽，或明确更名为独立约束 solver并重新冻结；
   - contradiction verifier检查不等式方向；
   - report要求精确 control inventory；
   - finalized resource进入 hash-bound receipt；
   - 明确祖先 symlink边界；
   - 拆分压线模块。
2. 若优先重提 P76：
   - 必须使用新鲜 holdout，或明确只做 descriptive coverage；
   - 不能在当前 24-Episode bank 上同时排除三个已见 sentinel并要求12/12 cell；
   - 固定22/22 live acquisition与24/24 offline lineage账本；
   - 建立外部 no-B2 action service和文件系统级 blind root；
   - 不捆绑 geometry或restore。
3. 若优先 association：
   - P85改名为 absolute execution-feasibility certificate；
   - P88先建立手工精确 coordinate/component oracle；
   - P68只声明overall/task selected relevance，除非另有latency floors；
   - 不发布全局 selector verdict。
4. P76-E6仍严格依赖有效的 P76-E5 accepted/descriptive eligibility。
5. P86须缩成有明确case、horizon、tolerance与resource上限的最小 restore gate。
6. P77、selector、默认 v2 migration、Replay、Actor、训练继续 no-go。

## 清理

本轮正式 artifact不存在，无 artifact需要清理。

- 清理前可用空间：`77,499,108 KiB`。
- 已移除：
  - `/private/tmp/hwr-r0017-baseline-wt`，仅用于无效 baseline 尝试；
  - `/Users/louis/Developer/AIWorkspace/50-housework-robot-r0017-p87`，候选实施
    worktree。
- 清理后可用空间：`77,935,700 KiB`。
- 实际释放：`436,592 KiB`，约 `426.4MiB`。
- 两个临时路径均已从 `git worktree list` 移除。
- 分支 `exp/R0001-P87-contract-oracle` 仍指向
  `485367fe1a4901f69407329af1e25bd7cdf5498b`。

清理不得删除：

- P50/P79/P83 输入与 artifact；
- 当前能力基线；
- `exp/R0001-P87-contract-oracle` 分支；
- 候选提交 `485367f`；
- R0017 文档、registry、测试日志证据。
