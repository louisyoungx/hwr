# R0008 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P40-E2` | `accepted as entity-contact measurement contract evidence` |
| `R0001-P41-E2` | `inconclusive` |
| `R0001-P47` | `deferred` |
| `R0001-P48` | `approved, not selected` |
| `R0001-P44-E1` | `approved, not selected` |
| `R0001-P32-E2` | `changes_required` |
| `R0001-P49` | `changes_required` |
| `R0001-P46-E1` | `approved, not selected` |

本轮没有启动正式训练，没有执行 P41 的正式 selector 对照，没有产生新家务任务成功，
也没有世界模型、Actor、闭环成功率、泛化或硬件安全能力改善。P40-E2 的接受只代表
evaluator-private 实体接触测量合同成立；P41-E2 因 smoke 中一个冻结 cell 无候选而证据
不足。

## 关键发现

1. `R0001-P40-E2` 建立了 `robot part × task entity` 接触图、task-relevant
   world–world edge、双臂接触模式和接触同期实体运动账本。
2. 三个正式任务的 48 个 robot geom 均唯一映射到 `base/left_arm/right_arm`，环境 geom
   均可解析为冻结实体角色。
3. reset 后首个 control period 被明确标记为 settling；该 period 保留原始 contact 和
   motion，但不计入 contact-associated motion。
4. P41 首次长轨迹 smoke 暴露 P40-E2 与 P40-E1 数学等价但求和层级不同，最大浮点差为
   `2.837623469531536e-10`，超过冻结 `1e-12` 门。
5. `8b79597` 将 P40-E2 episode category 改为与 P40-E1 一致的 control-period 累加顺序；
   1,655-period 回归和最终三任务正式 P40 run 的最大守恒差均为 `0.0`。
6. P40 measurement disabled/enabled 的 legacy trace 在三任务上保持 bit-identical；
   测量没有进入 policy、reward、termination、success 或安全决策。
7. P41 结果前 synthetic power 在候选 `36/54/72/90/108` 中选择 54 supported pair：
   54-pair planted power 95% lower 为 `0.8618210816892118`，最坏 null FPR 95% upper
   为 `0.021870261025271856`。
8. 最终 smoke 完成全部 6 个 planned pair、12 个 branch，无 infrastructure unresolved，
   并完整记录 30 个自然 latency profile 不匹配的 rejected seed。
9. 六个 cell 的候选数依次为 `4/0/1/3/5/3`。客厅 `(observation=2, action=2)` 的
   frozen candidate set 为空。
10. 所有 pair 的 candidate bytes/hash 一致，同索引 twin-run 均 bit-identical；全部
    branch 的 action bounds、severe collision、stale action、invalid force 与 P40 守恒
    守护通过。
11. smoke 中所有 branch 的主事件都为 false，但 smoke 未执行 selector 优劣比较，不能
    用该结果接受或拒绝 target-index 假设。
12. 按结果前合同，空候选 cell 直接阻止正式 54 supported + 18 challenge pair 评测；
    没有放宽阈值、挑换 seed、删除任务或启动正式 run。

## 当前基线

- 无 causality-qualified deployment；
- 最新完整 3D 世界模型负基线仍为
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`：
  - 24 Episode；
  - 1,600 update；
  - 0 成功；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor 未解锁；
  - 三任务没有 bilateral contact 或 controlled motion。
- P17 plant causality、P29 runtime、P36-E1 support-domain、P39 seed isolation、
  P36-E2 balanced benchmark 与 P40-E1 safety ledger 合同继续保持。
- 新增 P40-E2 entity-contact measurement contract evidence。
- P41-E2 为 `inconclusive`，其 target-index selector 不进入能力基线。
- 不宣称机器人已经学会任何家务任务。

## 提交与产物

- R0008 上下文、提案、筛选与 P40 冻结：`b56ee96`
- P40-E2 初始实现：`b578da6`
- P40-E2 settling exclusion：`cefb240`
- P40-E2 首次结果记录：`f5c65ca`
- P41-E2 冻结：`602adfe`
- P41-E2 smoke cell 澄清：`027eb0b`
- P41-E2 safety 模块冻结：`565a881`
- P41-E2 初始实现：`8c47ec8`
- P40-E2 长轨迹数值顺序修复：`8b79597`
- P41-E2 绑定修复后 P40 证据：`bd8dd2a`
- 最终 P40-E2 run：
  `runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002`
  - report：
    `987a2217cf9f5c6eb08b018b3bf13164917c75bccbfa77140a8786c80987f841`
  - manifest：
    `fdb847a41f55a7a3bb362d650baa2d131e2a5178ac73166336d57368ba60546b`
- 最终 P41-E2 smoke：
  `runs/research-loop/0008/r0008-p41-target-selection-e2-smoke-s20264101`
  - plan：
    `2d0d9d11e5f6d8b82106e92501015dab05fb13e1bb4a81950e808755d45a9fb4`
  - report：
    `5f5e8d50eb2c66043cc92721091fd895cca2dd925531890a87e65dedfd8885a7`
  - power：
    `ed6f117a67bd1e290cdbf1302e1d666e3a20b6447e38e3a31a4707806a1f90f9`
  - terminals：
    `c061c906c82fa4a6e8a2b11137d8e03e1873c2960759dd2c399ae76a95282b3a`
  - manifest：
    `649a5fdd84e133270d05fd3f70977b6b212a0d1eda185bda4faed07136dc48fb`

## 下一轮问题

下一轮必须重新执行创新与独立筛选，不直接继承本轮批准状态。优先问题：

1. 把 P41 的失败视为 observation/candidate coverage 问题，而不是 selector 效果：
   - 分析客厅 `(2,2)` 为什么在自然 latency 下无可用候选；
   - 使用新的结果前冻结实验区分感知覆盖、候选聚类与可达性；
   - 不在 R0008 后验修改阈值、相机、mask、seed 或任务。
2. 若重提 P41，必须先以独立 smoke 证明每个 supported cell 都有非空、可重建的共享
   candidate set，再冻结新的正式 salt 和样本量。
3. 不要在当前 24-source Replay 上继续 P31、P43 或新的 action objective 训练：
   P32 功效不足且 controller history 已解释主信号。
4. 若重提 Replay 诊断，应先增加 task-balanced 独立 source、降低设计方差，并保留
   controller-history 与 configuration-target 守护。
5. P48 deployment 防火墙与 P46 原子恢复可作为可靠性侧车重新筛选，但不能替代当前
   交互支持瓶颈。
6. P42、P44 继续等待 qualified deployment；不得通过 scripted policy、评测泄露或放宽
   safety/action-causality 门制造结果。

## 清理

- 启动记录可用空间：`91,561,144 KiB`，约 87.3 GiB。
- 收尾记录前可用空间：`91,235,648 KiB`，约 87.0 GiB。
- R0008 当前 run 目录合计：`9,164 KiB`。
- 删除内容：无。
- 未启动清理 Agent：
  - 本轮新增与 superseded 产物总量很小；
  - P40 两次语义/数值修复前证据、沙箱 CGL failure、P41 interrupted/provenance
    fail-closed/smoke 均用于追溯；
  - 没有主 Agent 确认可重建且无引用的大体量资源。
- 共享数据卷空间变化不归因于本轮。
