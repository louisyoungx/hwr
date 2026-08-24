# R0011 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P50-E3` | `inconclusive_design_infeasible` |
| `R0001-P57` | `accepted as bilateral pre-contact reachability measurement evidence` |
| `R0001-P56` | `deferred` |
| `R0001-P58` | `deferred` |
| `R0001-P59` | `rejected` |

本轮没有训练、policy inference、post-selection capability Episode 或新家务任务成功。

P50-E3 在任何 Episode 前按冻结合同 fail-closed：kitchen 的 `kitchen_drawer` 同一 body
同时承载 articulation handle 与 target-container geoms，纯 body-exclusive role mapping
不可构造。因此本轮没有 entity visibility/candidate coverage 结果，不能据此修改 generator。

P57 接受为测量证据：固定 P51 cohort 的 36/36 pair 从未在同一步达到左右臂均
`<=0.10m` 的 pre-contact readiness，且 36/36 pair 的两臂 actual-applied command
budget 均小于各自 B2 起始 target distance。

## 关键发现

1. P50-E3 的 living 与 dining mapping preflight 通过，kitchen 在 Episode 0 前失败：
   - `body_id=30`；
   - `body_name=kitchen_drawer`；
   - 角色冲突为 `articulation:drawer` 与 `target_container`；
   - Episode `0`、physical acquisition `0`。
2. 冻结设计中的“一个 body 只能有一个 task role”不适合 articulated container：
   handle 与 drawer interior/wall 可以合法属于同一刚体，但在评测中需要不同局部角色。
3. 不能后验给 articulation 或 target-container 设置优先级，也不能删除 kitchen cell；
   下一轮必须重新设计并冻结 geom-level/local-region role contract。
4. P57 完整分析 36 pair、72 arm、12 cell：
   - `ever_bilateral_ready=0/36`；
   - `endpoint_bilateral_ready=0/36`；
   - 两臂都改善 `10/36`；
   - 两臂 initial command margin 都为负 `36/36`。
5. 三任务 readiness 均为 `0/12`；observation latency 1/2 与 action latency 1/2 的
   边际分账也全部为 0。
6. B2 起始 arm-to-preposition distance：
   - 最小 `1.0782476291624683m`；
   - 均值 `2.279013066240849m`；
   - 最大 `3.400694272089616m`。
7. 100 step actual-applied arm command budget：
   - 最小 `0.3486734392248524m`；
   - 均值 `0.39061773300557145m`；
   - 最大 `0.43271220081555894m`。
8. initial command margin 在 72/72 arm 都为负：
   - 最小 `-3.010679833816681m`；
   - 均值 `-1.8883953332352779m`；
   - 最大 `-0.7254330230916336m`。
9. preposition→contact distance 为 `0.17699–0.17951m`，而 B3+B4 nominal command
   budget 固定 `0.095m`；72/72 arm 的 nominal transition margin 都为负
   `0.08199–0.08451m`。
10. P51 的双臂平均距离下降掩盖明显不对称：
    - 仅左臂改善 `15/36`；
    - 仅右臂改善 `11/36`；
    - 两臂同时改善 `10/36`。
11. P57 独立审计从 raw distance/applied action 重算通过：
    - blocker/major/minor 均为 0；
    - 最大数值误差 `8.881784197001252e-16`；
    - 3,600 个 applied action vector 无 bounds violation；
    - B3/B4 target 重建误差不超过 `4.72e-16`。
12. P57 支持的是固定 P51 cohort 的 command-support deficit，不证明：
    - candidate 属于 task entity；
    - 严格动力学/碰撞可达；
    - contact、grasp 或 task success；
    - 增加 phase 时长或速度一定有效。
13. P59 被拒绝：simulator-private segmentation 不能进入正式 candidate generator。
14. P56/P58 延期：当前没有有效 entity-hit cohort，也没有冻结的 phase-resolved contact
    合同，不能运行 B3 action-vs-hold。

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
- P17、P29、P36-E1、P39-E1、P36-E2、P40-E1、P40-E2、P50-E1、P50-E2、
  P51 解析坐标合同与 P52 FK agreement 合同继续保持。
- P57 新进入**测量证据基线**，不进入能力基线。
- P50-E3 没有产生 measurement evidence，不进入测量或能力基线。
- P51-E1 仍为 `rejected`；P41-E2 仍为 `inconclusive`。
- 不宣称机器人已学会任何家务任务。

## 提交与产物

### 关键提交

- R0011 上下文、提案、筛选与冻结实验：
  `88992a773ee2b0f214dba7975cdddf25f282d679`
- P57 离线可达性实现：
  `07c18fc179bb2ca713f09519673c6488bd6646b5`
- P50-E3 最小 mapping preflight：
  `e16e3ae2ad33fc95c12bcb4a5e5d48527b62fb0b`
- P57 ignored terminal evidence provenance 修复：
  `bbf6d666f8071fa3a5d26be2d774ceeae2ebc7a1`
- P51 frozen-document guard 作用域修复：
  `13766f44f23eb463d852a0a94ffb94aefeea38f5`
- R0011 正式诊断产物：
  `c9e6819e3c862757a8c4790a44b5a384f1431579`

### P50-E3 failure artifact

目录：

`runs/research-loop/0011/r0011-p50-e3-entity-coverage-s20265003`

- `failure.json`：
  `4931e98cfaf7bde8c2fadcaf81f65dc35add7843a7d4b34801309aa2b288701f`
- `manifest.json`：
  `fd87ad98ec2755db27f4ecb2b3bc21889a8c2c208da7bffbb039b4608011740d`

### P57 final artifact

目录：

`runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701`

- `pairs.json`：
  `d2cae19d21492bea39972bbe3e50e06aa7f19eedf617fbd8dd6c708d8543ddb3`
- `report.json`：
  `15fcfb3a0e9b706f9fbfdd9e2dd69a50d096db3435b6cb73100a69fb9697a39d`
- `manifest.json`：
  `21ca8837232ac715e209884f355d37a64298d2f4115dc45c81c12f3a7e812e40`

P57 资源：

- wall time：`8.866050041979179s`；
- peak RSS：`293,273,600 bytes`；
- tracemalloc peak：`72,394,026 bytes`；
- artifact：`2,014,821 bytes`。

## 验证

- P50-E3 preflight + P50 acquisition 回归：37 passed。
- P57 focused：17 passed。
- P51 tree-lock 修复 + 本轮联合 focused：101 passed。
- 全量 pytest：`1008 collected`，宿主图形环境 `997 passed, 11 skipped`。
- 18 条 warning 均为既有 `torch.jit.script` deprecation。
- Python size：424 files 通过。
- architecture：通过。
- physics integrity：通过。
- compileall：通过。
- `git diff --check`：通过。
- 沙箱全量 pytest 的失败均为 macOS `invalid CoreGraphics connection`；使用宿主
  `MUJOCO_GL=glfw` 后全量通过。
- development-ready 隔离 worktree 因多个历史 ignored run artifact 不随 Git checkout
  出现而失败；这是实现前既有门禁问题，不是本轮逻辑回归。
- P57 正式 artifact 独立审计通过。
- 历史 `docs/research-loop/0001/`～`0010/` tree 零差异。

## 下一轮问题

下一轮必须重新创新与筛选，不能自动继承本轮选择。优先问题：

1. 为 P50-E3 设计可表达 articulated body 局部角色的 evaluator-only mapping：
   - 以 segmentation 的 exact geom 为第一层；
   - 为 RGB-visible visual geom 建立结果前冻结的 body-local alias；
   - articulation handle、target-container interior/wall 保持不同角色；
   - body 只用于 visual/collision alias，不再强制 body-exclusive role；
   - site/background/无映射仍 fail closed 为 unknown。
2. 新 mapping 先做三场景 exhaustive preflight 和 synthetic ambiguity tests；只有全部通过，
   才重提 24-Episode sidecar，不先跑部分 task。
3. P57 已证明当前 P51 cohort 的 B2 支持不足，但 candidate entity identity 未知。下一轮
   不得直接增加 phase 时长或速度；必须先有 entity-hit cohort，或设计与 entity truth
   隔离的 candidate-to-base/tool geometry 诊断。
4. 若 entity-hit cohort 成立，重新筛选单一 phase-entry/readiness controller 变量，而不是
   同时修改 target、phase、velocity、IK、gripper。
5. P56 必须在任何 contact-yield/B3 因果实验前接受；P58 仍不得自动启动。
6. 不恢复当前 Replay 上的世界模型训练，不启动 selector、Actor 或能力评测。

## 清理

- R0011 正式 run 目录总计 `2,028,542 bytes`，约 1.9MiB。
- 起始数据卷可用空间：`88,558,612 KiB`。
- 收尾测得数据卷可用空间：`70,202,532 KiB`。
- pytest 临时目录约 37MiB，不能解释约 17.5GiB 的系统卷可用空间变化；该变化可能来自
  项目外并发活动，不归因于本轮。
- 删除内容：无。
- P50 failure artifact 是冻结设计不可行的唯一正式证据；P57 artifact 是 accepted
  measurement evidence，均不得删除。
- 清理 Agent 因调度拥塞未在两个完整等待窗口内返回；主 Agent 没有据此执行任何删除。
