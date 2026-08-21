# R0009 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P50` | `approved, not selected` |
| `R0001-P51` | `accepted as Cartesian primitive correctness evidence` |
| `R0001-P52` | `accepted as FK agreement contract evidence` |
| `R0001-P53a` | `deferred` |
| `R0001-P53b` | `deferred` |
| `R0001-P54` | `rejected` |
| `R0001-P49-E1` | `deferred` |
| `R0001-P47-E1` | `deferred` |
| `R0001-P55` | `changes_required, not selected` |

本轮没有启动正式训练，没有执行 closed-loop capability Episode，没有运行 P41 正式
selector 对照，也没有产生新家务任务成功。P51 接受为坐标合同正确性证据；P52 接受为
policy FK 与 MuJoCo grasp-center site 的一致性证据。两者均不是学习、任务成功、泛化或
硬件安全能力证据。

## 关键发现

1. P41 fixed primitive 原实现存在确定性的坐标语义错误：
   - target 与 policy FK tool position 位于 acquisition frame；
   - 两者误差原样写入 base-frame tool twist；
   - backend 又按 current-base rotation 转到 world frame；
   - relative yaw 非零时，实际方向偏离 acquisition-frame 目标。
2. P51 只增加
   `Rz(acquisition_yaw - current_base_yaw)` 线速度旋转，不改变 target、phase、速度
   上限、gripper、candidate、selector、backend 或安全。
3. 144-cell 解析矩阵全部通过；最大水平角误差
   `2.7105054312137616e-16 rad`，远低于 `1e-12 rad` 门。
4. 48 个 relative-yaw `±pi/2` legacy 反例全部被拒绝；测试能区分新旧坐标语义。
5. 实际 `primitive_action` 集成守护覆盖 20 个 phase/yaw case、28 次双臂 transform
   调用和 2 个 hold/fail-closed case，14 项检查全部通过。
6. P52 在 latency-free 同一物理 state 上比较 policy FK 与 MuJoCo site：
   - 每 task 153 state；
   - 三任务、两臂共 918 terminal；
   - aggregate p95 `4.652682298944613e-16m`；
   - aggregate max `7.021666937153402e-16m`。
7. P52 frame-invariance 最大误差 `4.440892098500626e-16m`；三任务机器人 joint/site
   mapping 一致。
8. 当前 policy FK 与真实控制 site 在冻结机器人模型上数值一致。因此厘米级 FK mismatch
   不是 P41 零 arm contact 的支持解释，不应再以 FK 偏差为理由同时改运动学与 primitive。
9. 独立代码审查发现首版 runner 有两条高风险假阳性路径：
   - P51 只检查 helper，未检查 `primitive_action` 集成；
   - P52 evaluator-private 隔离由常量自证。
10. 两条路径及两项 provenance 缺口已分别由 `b810f73`、`0ad97bb` 修复。最终复核无
    remaining finding。
11. P51/P52 首版 accepted artifact 均保留为 `superseded-8fbfbfa`，不作为最终证据。
12. P52 agreement 允许未来进行 P51 fixed-candidate 物理 smoke，但本轮冻结文档未唯一化
    seed、候选、样本量和判定门；本轮不后验补设计、不运行该 smoke。

## 当前基线

- 无 causality-qualified deployment；
- 最新完整三维世界模型负基线仍为
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`：
  - 24 Episode；
  - 1,600 update；
  - 0 成功；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor 未解锁；
  - 三任务无 bilateral contact 或 controlled motion。
- P17、P29、P36-E1、P39-E1、P36-E2、P40-E1 与 P40-E2 合同证据继续保持。
- P51 坐标修复进入代码基线，但尚无物理交互或闭环能力收益证据。
- P52 证明当前 FK agreement，不构成行为变化。
- P41-E2 仍为 `inconclusive`；不改变其历史判定，不运行原正式 salt。
- 不宣称机器人已经学会任何家务任务。

## 提交与产物

- R0009 上下文、提案、筛选与冻结：
  `4385ceee2fffcbd23788b498d258747dc273465c`
- P51 初始实现：
  `74ec4332830d93ecfb60e560450999a8ae917cf9`
- P52 初始实现：
  `8fbfbfafc37fe8181d2f953a9a72aaf1435fa7ba`
- P51 primitive integration gate 加固：
  `b810f73df6be05d25041ee64b3e898df08598c35`
- P52 isolation/provenance 加固：
  `0ad97bb7ee0d37a8f308c1ea9ffc705550891acb`
- 最终 P51 run：
  `runs/research-loop/0009/r0009-p51-cartesian-frame-s20265101`
  - report：
    `763270993a9199c7997c305f1f040794a631dc13fee961e022204b25c0b6c016`
  - manifest：
    `5f47bdddf829ba0b08d94fa499e495ffc542919f07ae48ed225dda9417301465`
- 最终 P52 run：
  `runs/research-loop/0009/r0009-p52-tool-kinematics-s20265201`
  - report：
    `af8b245ac4654dcebf0e0af57ceac9b7f5fc288b3cac42deb3c894186fafe3fb`
  - manifest：
    `8309a6366b8eeb277db237bfcfca06b6fe3601a3aa1a2e01eed9b67d2a65b0a2`
- 首版 superseded 产物：
  - `runs/research-loop/0009/r0009-p51-cartesian-frame-s20265101-superseded-8fbfbfa`
  - `runs/research-loop/0009/r0009-p52-tool-kinematics-s20265201-superseded-8fbfbfa`

## 验证

- 最终 focused suite：48 passed。
- P41/P51/P52 相关宿主回归：22 passed。
- 全量宿主 pytest：
  - 822 collected；
  - 11 skipped；
  - 其余通过；
  - 18 条既有 `torch.jit.script` deprecation warning。
- Python size：404 files 通过。
- architecture、physics integrity、compileall、`git diff --check` 通过。
- P51/P52 artifact hash/bytes、source commit、冻结祖先、历史 tree 和 claim flags 通过。
- P52 action isolation audit：2 个 source、0 violation。
- P52 task × arm deterministic replay：6/6 bit-identical。
- P52 每个正式模型的 7 个递归 XML dependency 与 8 个关键源码 identity 已绑定。
- 沙箱 CGL 回归失败在宿主环境复跑通过，不是实现回归。

## 下一轮问题

下一轮必须重新执行创新与独立筛选，不能自动继承本轮选择。优先问题：

1. 为 P51 修复冻结真正的 fixed-candidate paired physical smoke：
   - 预提交 environment seed commitment；
   - 冻结 candidate bytes/hash 与 candidate/entity 分账；
   - 冻结 baseline/candidate paired block 数和功效；
   - 冻结 tool-target distance 时间窗、方向导数与停止门；
   - 冻结 contact、安全、stale action、动作非塌缩和 early-termination 处理；
   - 不与候选生成、FK、phase、幅值或 gripper 变化捆绑。
2. 并行评审 P50 候选漏斗测量，但必须：
   - 离线处理不可变 acquisition bytes；
   - 覆盖完整 `task × observation latency {1,2} × action latency {1,2}`；
   - 将 first-rejection 明确为描述性流水线统计；
   - 用 evaluator-private entity association 区分“空集合”和“错误家具候选”；
   - R0008 seed 只用于 regression，新 seed 用于判定。
3. 只有 fixed-candidate primitive 确实收敛且候选 coverage/identity 成立，才重新筛选
   P47-E1 或 P41 selector 对照。
4. P53a/P53b 必须等待 P50，并且只能选择一个单变量 generator 修订；不得后验并行择优。
5. P49-E1 暂限 design/power；不采集更多 history-predictable Replay。
6. P55 需要逐实体 arm-pad contact→motion→placement 时序自动机后才能重提。
7. P54 在本轮因输入白名单冲突被拒；未来若重提必须在新轮次显式重新评审语言输入边界。

## 清理

- 启动时数据卷可用空间约 87GiB。
- 收尾时数据卷可用 `94,712,556 KiB`，约 90.3GiB。
- R0009 当前 run 目录合计 `2,920 KiB`。
- 删除内容：无。
- 未启动清理 Agent：
  - 本轮正式与 superseded 产物总量很小；
  - superseded 结果记录独立审查前后的证据门变化，仍有追溯价值；
  - 没有主 Agent 确认可重建且无引用的大体量资源。
- 共享数据卷空间变化不归因于本轮。
