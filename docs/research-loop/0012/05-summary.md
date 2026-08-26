# R0012 轮次总结

## 结论

| ID | 判定 |
|---|---|
| `R0001-P61` | `accepted as interaction-contract gap evidence` |
| `R0001-P50-E4` | `accepted as exact-geom evaluator mapping contract` |
| `R0001-P60` | `invalid` |
| `R0001-P62` | `deferred` |
| `R0001-P63` | `rejected` |
| `R0001-P64` | `deferred` |
| `R0001-P65` | `deferred` |

本轮没有训练、policy inference、B2 action、contact phase、capability Episode 或新家务
任务成功。

P61 接受为有限静态信息合同证据：当前 same-function direct-call 边界没有把 entity role、
interaction type、destination target 或 articulation threshold 提供给 generic
candidate-centered primitive。该结论不覆盖动态调用、跨函数 dataflow 或潜在外部 planner。

P50-E4 接受为 evaluator mapping 合同：三个冻结场景可通过 exact geom claim 和 8 个一跳
same-body visual alias 构造确定 role table；同一 body 可合法承载 articulation 与
target-container，不再使用 body-exclusive propagation。

P60 在首个 cell 的第二个 latency-matched prefix 触发
`action_rejected / predicted_severe_collision`。冻结合同要求首个 safety intervention
立即停止，因此没有形成 36-Episode cohort，也没有 strict/nominal phase-entry geometry
结论。没有发生实际 severe collision，不能把该结果称为碰撞或能力失败。

## 当前基线

- 能力基线不变：最新完整三维世界模型负基线仍为
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`：
  - 24 Episode；
  - 1,600 update；
  - 0 success；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor 未解锁。
- P61 新进入静态设计证据基线，不进入能力基线。
- P50-E4 新进入 evaluator mapping 合同基线，不进入 entity-coverage 或能力基线。
- P60 没有产生测量证据，不进入任何基线。
- P57 的 fixed P51 cohort command-support deficit 测量仍有效。
- P51-E1 仍 rejected；P41-E2 仍 inconclusive。
- 不宣称机器人已学会任何家务任务。

## 关键发现

1. P50-E3 的根因是抽象层错误，不是 kitchen 场景不可分：
   - exact `drawer_handle_visual` 与 drawer six surface geoms 可区分；
   - visual/collision alias 需要显式冻结；
   - body 只能做 alias 同刚体验证，不能传播 role。
2. P50-E4 三场景、8 alias、40 negative guard、271-source AST isolation 全部通过；
   table 重建 bit-identical。
3. P61 的 7 个 transition 无法由当前一个 selected candidate 与 generic B0–B7 primitive
   唯一表达 full-task entity、interaction 与 destination。
4. 初始 P61 实现的 verdict 有自证缺陷；修复后由可翻转的 AST/schema/signature evidence
   推导，旧 artifact 完整保留。
5. P60 尚未触及 geometry estimand：一个 prefix candidate empty，另一个在 B1 被预测
   severe collision 安全检查拒绝。
6. safety intervention 的作用是阻止危险动作进入真实 physics；actual severe collision
   count 为 0。
7. 不能通过忽略安全事件、改 220N 阈值、继续后续 seed 或换 salt 来制造完整 cohort。

## 提交与产物

### 关键提交

- R0012 上下文与提案：`b397c10a63d623096281e6c88ed0a3ac63755cfd`
- 筛选与冻结实验：`a95dbbeacc80a974d7a234f2dc79442249eaf07b`
- P50-E4 实现：`6c3abee111d84a65add72c5f4cdb93c39fdae667`
- P50-E4 architecture 修复：`be965f868c3c21e234130d6b5e9f725bf73efc78`
- P60 实现：`d13393632f9c4723852cc73e8c7338f190ecf2f0`
- P61 实现：`e51c1ea02aef5e97d7b3f6f97254872f95212035`
- P50-E4 结果后合同修复：`c9d1f01afc3214fd239e505ec6865e6e33c5f5c3`
- P61 结果后自证修复：`6bf0400f51a25bfb6f45e951299c410efd5c2c7a`

### 正式 artifact

- P61：
  `runs/research-loop/0012/r0012-p61-interaction-contract-s20266101`
- P50-E4：
  `runs/research-loop/0012/r0012-p50-e4-mapping-s20265004`
- P60：
  `runs/research-loop/0012/r0012-p60-phase-entry-s20266001`
- 结果后修复前的 P61/P50-E4 artifact：
  - `...p61...-superseded-be965f8`
  - `...p50...-superseded-be965f8`

## 验证

- 全量 pytest：`1057 passed, 11 skipped`。
- 18 条 warning 均为既有 `torch.jit.script` deprecation。
- Python size、architecture、physics integrity、compileall、`git diff --check` 全部通过。
- P60 独立审计：实现/计数/seed/latency/trace/provenance major `0`，hard-stop 有效。
- 修复后 P61/P50-E4 独立审计：blocker `0`、major `0`、minor `1`。
- 剩余 artifact 外部锚定 minor 通过将正式与 superseded artifact 强制提交到 Git 关闭。

## 下一轮问题

下一轮必须重新创新和筛选，不能自动继承本轮选择。优先问题：

1. P60 的 B1 预测安全拒绝需要一个独立、结果前冻结的诊断：
   - 保存预测分支最大 forbidden force、contact pair 与 `physics_advanced`；
   - 区分 base-path obstacle、candidate-induced approach 和 safety predictor margin；
   - 不降低 220N 阈值、不绕过 safety、不把预测拒绝称实际碰撞。
2. 若能证明该安全事件来自通用 B1 base routing，再提出一个单变量、fixed-budget、
   safety-preserving base-path controller；不得与 entity mapping 或 arm phase 同时修改。
3. P61 已证明 full-task information contract 不足。下一轮若设计 planner/transition
   interface，必须：
   - 只使用部署可得信息；
   - 把 task decomposition 与 low-level primitive 改动分开；
   - 在独立未见任务/布局上做闭环物理对照。
4. P50-E4 只建立 mapping 基础。P64 仍需：
   - 有效的 action-chain positive control；
   - 冻结 final-set association、mixed/unknown 上限与 direction gate；
   - 使用固定 cohort 只作诊断，不在同 cohort 调参后宣称改善。
5. P62 只有获得独立 dual-arm joint-limit/collision/safety feasibility witness 后才可重提。
6. 不恢复当前 Replay 上的世界模型训练，不启动 selector、Actor 或能力评测。

## 清理

- 本轮正式与 superseded artifact 总计约 `6.2MiB`。
- 起始数据卷可用空间：`128,171,676 KiB`。
- 收尾测得数据卷可用空间：`111,192,876 KiB`。
- 可用空间变化主要来自项目外并发活动，不能归因于本轮约 6.2MiB artifact。
- 删除内容：无。
- P60 invalid artifact 是安全 hard-stop 的唯一正式证据；P61/P50-E4 superseded artifact
  记录结果后合同修复时序；全部保留。
