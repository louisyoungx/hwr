# R0010 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P51-E1` | `rejected` |
| `R0001-P50-E1` | `accepted as immutable acquisition evidence contract` |
| `R0001-P50-E2` | `accepted as candidate-funnel measurement evidence` |
| `R0001-P50-E3` | `deferred` |
| `R0001-P56` | `deferred` |

本轮没有训练、policy inference、closed-loop task capability Episode 或新家务任务成功。
P50-E1/E2 被接受为测量合同；P51-E1 的 frame-fixed 公式在冻结 cohort 上有小幅一致正向，
但未达到结果前冻结的最小有意义效应，因此拒绝为物理收敛改善证据。

## 关键发现

1. P50-E1 建立了首个完整
   `3 task × observation latency {1,2} × action latency {1,2}` acquisition evidence
   capsule：
   - 24 planned Episode；
   - 24 independent validation replay；
   - 47,760 control step；
   - 384 个原始 policy-input blob；
   - offline candidate replay、same-seed physics replay、capture-disabled identity、runtime
     latency 与安全守护全部通过。
2. 24 个 Episode 共得到 39 个 final candidate，5 个 Episode 为空。
3. P50-E2 精确统计：
   - 876,960 个 anchor；
   - 225 个 raw candidate；
   - 149 个 connected component；
   - 110 个 component 被 `view_count<2` 拒绝；
   - 39 个 pre-top64/final candidate；
   - top-64 截断 0。
4. prominence 的条件拒绝率为 `99.1446%`，center depth spread 为 `91.4021%`；二者在
   12/12 cell 的两个 replicate 中均达到冻结的重复描述性 loss 标签。
5. 5 个空 Episode 都不是 raw depth/anchor 阶段直接为空；它们已有 raw candidate 与
   connected component，最终全部在 component→view-count-qualified 阶段归零。
6. 最弱 cell 是 living `(observation latency=2, action latency=1)`：
   - 2/2 Episode 空；
   - raw 14；
   - component 13；
   - final 0。
7. 本 cohort 的 candidate keyframe 为 360/360 unique observation identity；加 A4
   final 后为 384/384。unique-observation shadow 与 ordinal merge 的 final candidate
   总数都是 39，24 个 Episode 差均为 0。
8. 因此“重复 observation identity 导致当前空集合”在本 cohort 被否定。view-count
   是部分空集合的直接描述性 terminal stage，但尚无证据证明放宽它会提高 task-entity
   candidate 或物理交互。
9. P51 final bank 从 405 个 raw seed 中取得：
   - natural latency mismatch 361；
   - latency matched 44；
   - candidate empty 5；
   - yaw 暴露不足 3；
   - eligible 36。
10. P51 36-pair B2-only 对照：
    - point estimate `+0.023449928237828013`；
    - one-sided bootstrap 95% lower `+0.0200166364588308`；
    - 12 个 cell mean 全为正；
    - 30/36 pair delta 为正；
    - 33/36 endpoint 有利于 fixed。
11. 该正向幅度只达到冻结 MDE `0.10` 的约 23.45%；最大单 pair delta
    `0.07591725404801863`，0/36 pair 达到冻结 win，所有 task/latency 二项门失败。
12. P51 的 hard safety 与 identity 门全部通过；拒绝来自效果量不足，不是 safety、
    infrastructure、candidate-bank 或统计实现失败。
13. P51 结果不能解释为“坐标修复无效”。它说明解析正确的 frame transform 在当前
    B2 controller/eligible candidate domain 中只带来小幅收敛变化，不足以成为当前零交互
    的主要解释。

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
- P17、P29、P36-E1、P39-E1、P36-E2、P40-E1、P40-E2、P51 解析坐标合同与 P52
  FK agreement 合同继续保持。
- P50-E1/E2 新进入**测量证据基线**，不进入能力基线。
- P51-E1 被拒绝，不把其小幅正向 trend 计作能力改善。
- P41-E2 仍为 `inconclusive`，不运行原 selector salt，不改变历史判定。
- 不宣称机器人已经学会任何家务任务。

## 提交与产物

### 关键提交

- R0010 上下文、提案、筛选与初始冻结：
  `5fad6cec27e8f797c31a202497745a5616ab220b`
- replay 预算结果前澄清：
  `2d1752f2c0c8b9e39d7f3ebaa8e9ff0ec1d13f38`
- P50-E1/E2 初始实现：
  `3dbe18503479b86c9b8d55bb7e5176a4d24d2ff7`
  `356d777374754ea5b3a8d480c51b0c94fa0c39af`
- P50 最终审查修复：
  `0cb5b4a6a2566ad9d10c3fb9da798fb5d820fcad`
- P51-E1 初始实现：
  `294e82debf59cc46c68578f6e383cadb798dbaa0`
- P51 runtime/provenance 加固：
  `d67791a53491ce37cddaef4bd7d6b71ad3e66ac2`
- candidate-empty validator 修复：
  `38daff6631c32af2f34ae0cd77578004c0a0768e`
- JSON role-order roundtrip 修复：
  `656c759235f1a736b11e8cc2a75c69e2c0b8a3f6`
- final bank artifact：
  `63b6c2dc0221f37b71b4356d0b154ec3e8e97c0a`

### Final artifacts

- P50-E1：
  `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
  - report：
    `b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0`
  - manifest：
    `cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86`
- P50-E2：
  `runs/research-loop/0010/r0010-p50-e2-funnel-s20265001`
  - report：
    `4c7f36d20356d2f0f9c83d024412da5ec3a95dea8714e9a04d91d0cd686d0e39`
  - manifest：
    `a2f2498ac1d23f22dae337b5ad0836ae72230861d4f33284cdff19c1b46e268e`
- P51 final bank：
  `runs/research-loop/0010/r0010-p51-e1-bank-s20265101`
  - bank：
    `09d2fe4e05f2bd8d23ebfe6886fe260d1b34b41771da42992f0f432a8a04f3d3`
  - seed audit：
    `1b8ca0d93dd4324d0f6e14a1932bbce69b24964bc0aa08a53a9b4bdf2eb69407`
  - manifest：
    `7e0d5f9c7757b59ceb8d4dfe3ddcba38cc1d1037c43c358e7168d700310d5e45`
- P51 final evaluate：
  `runs/research-loop/0010/r0010-p51-e1-convergence-s20265101`
  - terminals：
    `1c54f93a95bfbf4e08076b3c633b22dce295990a6808a48f0f10de18a2b3c2c7`
  - report：
    `3fcac95c2362923d9eb94ef4d7121d5bcb31ea859a308ed352321dfa93771cc9`
  - manifest：
    `821f3cf6fea922a86b4096ee5d0ba9c64b9d8f444eacc98dcfc1f164da1328d2`

### Superseded evidence

- 首次 bank validator failure：
  `r0010-p51-e1-bank-s20265101-superseded-d67791a`
- 首版完整 bank：
  `r0010-p51-e1-bank-s20265101-superseded-38daff6`
- 首版完整 formal：
  `r0010-p51-e1-convergence-s20265101-superseded-3ab93c8`

上述失败和 superseded 产物均保留。第一次 failed bank 后修复 validator，第一次 formal
结果后修复 JSON role-order roundtrip；两次重跑使用同 salt、同 seed audit、同 cohort，
最终数值与首版数值完全相同。不能声称本轮从未发生结果后评测修复。

## 验证

- P50 focused：85 passed。
- P51 最终 focused：69 passed；role-order 修复后回归 68 passed，candidate-empty 修复后
  69 passed。
- 主 Agent 联合新功能与相关回归：195 passed。
- 宿主全量 pytest：976 collected、11 skipped、其余全部通过。
- 18 条 warning 为既有 `torch.jit.script` deprecation。
- Python size：418 files 通过。
- architecture、physics integrity、compileall、`git diff --check` 通过。
- P50 与 P51 最终独立 artifact 审计均通过；P51 final 落盘
  `analysis == report`。
- 历史 `docs/research-loop/0001/`～`0009/` tree 零差异。

## 下一轮问题

下一轮必须重新创新与筛选，不能自动继承本轮选择。优先问题：

1. 选择并冻结 `R0001-P50-E3` 的 observation-time evaluator-private entity sidecar：
   - 原始 observation 生成时、进入 latency queue 前同步封存 segmentation/geom identity；
   - 与 RGB-D、动态标定和 `(timestamp, sequence)` 绑定；
   - 结果前冻结 visible pixel、mixed/unknown、raw patch 与 component association 规则；
   - 私有 truth 单向隔离，绝不能进入 candidate、selector 或动作。
2. 用 P50-E3 区分：
   - task entity 未进入视野；
   - task entity 可见但没有 raw candidate；
   - task-entity raw candidate 被 prominence/depth/view gate 删除；
   - final candidate 只对应 distractor furniture。
3. 不因 P50-E2 的描述性 rejection rate 直接放宽 prominence、depth spread 或 view-count。
   只有 entity-conditioned evidence 才能支持选择一个单变量 generator 修订。
4. P51-E1 已排除“frame transform 是零交互主要瓶颈”的假设；不继续调整坐标公式、FK、
   velocity cap、phase 或 gripper 来追逐这组结果。
5. 在 candidate entity coverage 成立前，不恢复 P41 selector 正式对照、P47、Replay
   采集、Actor 或世界模型训练。
6. 若下一轮进入 B3～B6/contact-yield 对照，先重新筛选 P56，排除 base-only contact
   和 phase 混淆。

## 清理

- R0010 run 目录合计约 365MiB。
- 收尾数据卷可用空间：`88,544,612 KiB`，约 84.4GiB。
- 删除内容：无。
- 未启动清理 Agent：
  - P50 原始 capsule 是唯一不可重建而无需重跑 48 个 acquisition 的证据；
  - final 与 superseded P51 artifact 记录 validator/serialization failure、相同 cohort 和
    数值复现，仍有追溯价值；
  - 总体体积相对可用空间较小。
