# R0013 轮次总结

## 结论

| ID | 判定 |
|---|---|
| `R0001-P66-E1` | `accepted as predictive-safety witness contract` |
| `R0001-P72-E1` | `accepted as residual P61 contract gap evidence` |
| `R0001-P68-E1` | `inconclusive_budget_exceeded` |
| `R0001-P67` | `deferred` |
| `R0001-P69` | `deferred` |
| `R0001-P70` | `deferred` |
| `R0001-P71` | `deferred` |

本轮没有训练、policy inference、B2/B3–B7 action、contact phase、capability Episode
或新家务任务成功，能力基线不变。

P66 接受为 production-isomorphic measurement contract。固定 P60 anchor 的实际
delayed/scaled plant command 在 two-control-step clone predictor 的第二个 boundary，
由 `robot_base/body_box_collision` 与
`tea_table/tea_table_top_collision` 的单 contact point `356.9928N` 触发 `220N`
拒绝门；安全层改写为 hold，authoritative physics 未推进，实际 severe collision 为 0。

P72 接受为 residual contract-gap evidence。P61 的 initial annotation 四类反事实均
fail closed，足以解锁 P68；但五类 exact source reference drift 不会使 final audit
fail closed，且 role field 可被 current source audit 错误等同于 planner call state，
因此 P61 的 exact-reference 与 external-planner 表述必须收缩。

P68 实现通过单 Episode observer-off/on identity、24/24 historical candidate
reconstruction 与 sidecar isolation 门，但固定 24-Episode正式 cohort 未在 30min
预算内完成。runner fail closed 且无正式/partial artifact，所以没有 association 比例、
task 分账或 selector relevance 结论。

## 当前基线

- 能力基线不变：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`
  - 24 Episode；
  - 1,600 update；
  - 0 success；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor 未解锁。
- P66 新进入 measurement contract baseline，不进入能力基线。
- P72 新进入静态合同证据，收缩 P61 的 exact-reference/planner claim。
- P68 没有结果，不进入 association、selector 或能力基线。
- P57 command-support deficit、P50-E4 evaluator mapping、P52 FK agreement继续有效。
- 不宣称机器人已学会任何家务任务。

## 关键发现

1. P60 safety hard-stop 的具体预测 witness 不是先前粗 AABB 猜测的 sofa：
   - robot base 对 living-room tea table 顶面；
   - 单 contact point `356.9928N`；
   - 第二 predictive control boundary；
   - actual physics 未提交该动作。
2. 被 predictor 检查的不是同一步 policy proposal，而是 action-latency FIFO 中
   step `1278` 的 delayed/scaled plant command：
   `(0.1056309462, -0.1605351262)`。
3. P66 instrumentation 不能留在共享 backend；迁移到专用 diagnostic subclass后，
   R0010 P51 frozen backend blob恢复，witness内容不变。
4. P61 的五个 exact schema/signature/phase match flags没有进入 final fail-closed gate。
5. P61 的 `planner_call_state_available` 可被 role field直接置真，即使 contract planner
   fields为空、transition ID unavailable、validated external planner为 false。
6. P68 的 candidate support可以对 24/24 P50 capsule按 canonical bytes重建；过程中发现
   generator 的 `patch_valid` view会原地收缩，具有扫描顺序依赖。本轮不修该行为。
7. P68 30min预算不足以完成固定 24-Episode baseline+treatment replay；不得用部分运行
   或单 Episode smoke宣称 association结果。

## 提交与产物

### 关键提交

- R0013 context/proposals：`e21c01d`
- R0013 review/experiment freeze：`b1db136`
- P66 initial implementation：`df760da`
- P72 initial implementation：`fa88d36`
- P68 implementation：`8b6ab26`
- P68 canonical support fix：`f9ee12b`
- P66 diagnostic isolation fix：`6c4dc23`
- P66 pre-fix artifact archive：`bb35cd2`
- P66 final artifact：`609b9ad`
- P72 independent planner evidence fix：`406af7d`
- P72 pre-fix artifact archive：`0edd113`
- P72 final artifact：`b49d8b2`

### 正式 artifact

- P66：
  `runs/research-loop/0013/r0013-p66-predictive-witness-s20266601`
- P72：
  `runs/research-loop/0013/r0013-p72-p61-mutation-s20267201`
- superseded：
  - `r0013-p66-predictive-witness-s20266601-superseded-199c5a8`
  - `r0013-p72-p61-mutation-s20267201-superseded-199c5a8`
- P68：无正式或 partial artifact。

## 验证

- P66 focused：27 passed。
- P61/P72 focused：23 passed。
- P68 focused：43 passed；final core 11 passed。
- Python size、architecture、physics integrity、compileall、`git diff --check` 通过。
- 全量 pytest：`1190 passed, 11 skipped, 1 failed`；唯一失败为既有 R0012
  frozen-context provenance failure；另有 18 个既有 deprecation warnings。
- 该唯一失败已在 R0013 起始提交 detached worktree复现，本轮不改历史文档。
- 独立代码审计修复后复审：0 blocker、0 major；P68两个 minor已记录并通过
  `inconclusive_budget_exceeded` 边界处理。

## 下一轮问题

下一轮必须重新创新和筛选，不自动继承本轮候选。优先问题：

1. P68 若重提，先做不改变 estimand 的执行设计审计：
   - source-state segmentation只在 capture identity渲染；
   - 给 runner 增加 Episode 内精确 deadline；
   - 在看到任何 association classification前冻结新预算；
   - 不复用本轮未发布的内存中间值；
   - 保持同一 24-Episode cohort、0.80 ratio、18/24与6/24门。
2. P66 已把单 anchor拒绝定位为 robot-base/tea-table predicted contact。下一轮可在重新
   筛选后考虑 P69 same-state action-component ablation；单 anchor仍不能支持 controller
   修改。
3. 若 P68 再次不可执行或给出低 relevance，优先转向 P71 independent bilateral endpoint
   feasibility witness，而不是继续扩展 B1 route。
4. P61 auditor需要独立修复：
   - exact reference flags进入 final fail-closed gate；
   - planner evidence从 role availability中拆开；
   - 评测修复不得与后续 planner能力改进放在同一因果对比。
5. P50 generator 的 order-dependent `patch_valid` mutation应作为独立 evaluator/generator
   缺陷审计；不得在 association cohort中同时修复。
6. 不恢复 selector、Replay、Actor或世界模型训练。

## 清理

- R0013 正式与 superseded artifact约 `157MiB`。
- R0013 起始数据卷可用空间约 `106GiB`。
- 收尾前数据卷可用空间：`88,960,604 KiB`，约 `84.8GiB`。
- 可用空间变化主要来自项目外并发活动，不能归因于本轮约 157MiB artifact。
- 删除内容：无。
- 保留理由：
  - P66 superseded artifact证明隔离修复前后 witness bytes一致；
  - P72 superseded artifact记录 planner-evidence自证修复时序；
  - P68 无 artifact可清理；
  - 所有失败、manifest与唯一证据均保留。
