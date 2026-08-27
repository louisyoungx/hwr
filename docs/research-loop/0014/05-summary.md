# R0014 轮次总结

## 结论

| ID | 判定 |
|---|---|
| `R0001-P79-E1` | `accepted as deterministic isolated v2 candidate-generator correction` |
| `R0001-P73` | `rejected as standalone remedy` |
| `R0001-P74` | `deferred` |
| `R0001-P75` | `rejected` |
| `R0001-P76` | `deferred` |
| `R0001-P77` | `rejected in current form` |
| `R0001-P78` | `deferred` |

本轮没有训练、policy inference、physical acquisition、B0–B7 action、contact phase、
capability evaluation 或新家务任务成功，能力基线不变。

P79 接受为**隔离 v2 candidate generator 的确定性正确性修复**：

- legacy v1 `target_selection.py` 保持冻结 blob，不破坏 P51/P60 provenance 与旧
  P50/P68 重建；
- v2 generator 独立于 P79 evaluator 模块；
- local `patch_valid` 不再原地修改 parent validity mask；
- 384 个 frame、三种遍历、2,806,272 个 probe 和两次完整 bank build 全部满足冻结
  identity 与确定性门；
- 新 v2 bank 已固化到 Git。

P79 **不表示默认 runtime/production pipeline 已切换到 v2**。任何新训练、association、
reachability、routing 或 capability evaluation 必须显式选择 v2 bank，并重新冻结相应
producer/consumer 合同。

## 当前基线

- 能力基线不变：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`
  - 24 Episode；
  - 1,600 update；
  - 0 success；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor 未解锁。
- measurement/design baseline 新增：
  - `R0001-P79-E1`：isolated v2 mask-ownership correction；
  - v2 bank：
    `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`。
- legacy v1 证据继续作为历史系统证据：
  - P50/P51/P57/P60/P66 artifact 原字节保留；
  - 不再把 legacy candidate-conditioned 结论外推到 v2 pipeline。
- 不宣称机器人已学会任何家务任务。

## 关键发现

1. P50 generator 的 `patch_valid` 确实是 parent `valid` 的 slice view，原地 `&=`
   会造成扫描顺序副作用。
2. 修复后的隔离 v2 generator：
   - parent mutation count 为 0；
   - row-major、reverse-row-major、column-major raw multiset 相同；
   - final candidate bytes 相同；
   - 两次完整 bank build bit-identical。
3. 旧 v1 与 v2 对同一 24 Episode 的 candidate-set hash 为 24/24 不同，selected
   identity 为 22/24 不同；这说明旧-bank P68/P76 不能授权 v2 pipeline 决策。
4. v2 总 candidate 数为 36，legacy v1 为 39；3 个旧 empty Episode 在 v2 变为
   nonempty，没有 v1 nonempty 变为 v2 empty。
5. 初版把 v2 直接放入共享 `target_selection.py` 会触发旧 P51/P60 frozen-source
   guard；最终实现改为隔离 v2，恢复共享 blob。
6. 正式验收将完整 pytest 原始输出保存为 artifact，只允许在真正 `61d85cd` detached
   cwd 可复现的唯一既有 failure。
7. Episode 级进程并行把双 bank 重建从超过 10min 降到总流程约 5min；bank process-tree
   保守 RSS 上界约 1.45GiB。

## 提交与产物

### 关键提交

- R0014 初始化：`9e51f7e`
- 提案冻结：`5966e3f`
- 独立筛选：`41eae65`
- 实验冻结：`61d85cd`
- P79 初始实现：`3003095`
- Episode 并行：`62cec9c`
- v1 reader/receipt 过渡修复：`6bad253`
- receipt test fixture：`d4d49dc`
- isolated v2 最终修复：`9eef995`
- 最终 artifact：`93ea4e7`

### 正式 artifact

`runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`

关键 hash：

- bank：
  `888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e`
- regression：
  `1f2f94c39c6f9b8799ec1f3c1def9ce049571fe1edf7bc4dd5a4567e0dbd3582`
- pytest output：
  `9ea09269023dfab495a4dffe7287ae70115fb83c4e6a7872136b334abe9a03cf`
- report：
  `13913e80070ff415c895f78a78a5210e27611f682a64cd9904756600aa62db6e`
- manifest：
  `162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9`

## 验证

- 完整 pytest：`1115 passed, 11 skipped, 1 failed`。
- 唯一失败为既有 R0012 frozen-context provenance failure，已在真正 detached
  `61d85cd` cwd 复现。
- P79 focused、P68 legacy reconstruction、P60 selected-record、P51/P60 frozen-source
  guard：通过。
- Python size、architecture、compileall：通过。
- 独立最终审计：0 blocker、0 未解决 major；同意 accepted，但要求保持 isolated-v2
  结论边界。

## 下一轮问题

下一轮必须重新创新与筛选，不自动继承本轮候选。优先问题：

1. 明确 v2 bank 的 consumer/version contract：
   - 选择版本化 runner 或显式 generator dependency；
   - 禁止静默把 legacy v1 与 v2 candidate 混用；
   - 默认 production 是否迁移必须作为独立行为改动和回归实验。
2. 在 v2 bank 上重提 P68：
   - 先冻结 v2 support reconstruction 与 association semantics；
   - P74 head-RGBD execution optimization 可作为独立前置；
   - 不沿用旧 v1 P68 的未发布中间结果。
3. 在 v2 candidate/base/target lineage 上重提 revised P76：
   - 严格区分 fixed-base outer-envelope 与 free-base dynamics；
   - 若外包络排除充分，再决定是否需要 P77 path-existence search。
4. P78 若重提，应拆为：
   - exact-reference fail-closed；
   - external-planner evidence 与 role availability 解耦；
   两个独立修复。
5. 在 v2 association 或 feasibility 前置证据通过前，不启动 selector、Replay、Actor、
   世界模型训练或 capability evaluation。

## 清理

- R0014 起始可用空间约 `69GiB`；最终可用空间约 `108.7GiB`。
- 空间增加来自重启及项目外并发状态变化，不能归因于本轮清理。
- 最终 P79 artifact约 `2.7MiB`，全部保留并已提交。
- 删除内容：
  - 两个未通过最终独立审计的 preliminary P79 output，均未提交；
  - `/private/tmp/hwr-r0014-p79` 和 `/private/tmp/hwr-r0014-baseline` 可重建 worktree
    将在 push 后移除。
- 未删除任何当前基线、唯一原始数据、正式结果、checkpoint、manifest 或日志。
