# R0012 研究上下文

## 轮次身份

- 轮次：`R0012`
- 状态：完成
- 起始分支：`feat/research-loop`
- 起始提交：`9679931ea80b0723458e3c8da41b88a091187c48`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-24`
- 前一轮：`R0011`
- 本轮能力基线：不变

本轮只写入 `docs/research-loop/0012/`。历史轮次冻结只读：

| 目录 | 起始提交下的 Git tree |
|---|---|
| `docs/research-loop/0001/` | `416912b7dc1c19611bcfc4375028180014a1989b` |
| `docs/research-loop/0002/` | `6fb603dbd52451fe1749157daf05aa482ca7222f` |
| `docs/research-loop/0003/` | `f56011eda321ea803bc24051db001e632c1549fb` |
| `docs/research-loop/0004/` | `611c420e539a53a8c7578cd66aa8bdfe46fe82b7` |
| `docs/research-loop/0005/` | `0352d379d5754adb03e9158c0fa72393ab322d58` |
| `docs/research-loop/0006/` | `ee3a6f5b25887f67f812750d2a75424df12823d4` |
| `docs/research-loop/0007/` | `0a696caa153abc9c13403fbc9bd3c081ce71c327` |
| `docs/research-loop/0008/` | `65e626cddbcb0ec9c2e17cca5184b7d40950e1c6` |
| `docs/research-loop/0009/` | `316db8b9ad9739ef491778f641603dbca25e75c9` |
| `docs/research-loop/0010/` | `8a193a24788027d715750c3cd89c2509e71fdbda` |
| `docs/research-loop/0011/` | `85bb445726ecb8e35ff4d8e90606874e2ee36fe4` |

## 可信基线

- 当前没有通过动作因果与未见分布闭环门禁的 deployment，不宣称机器人已学会家务。
- 最新完整三维世界模型负基线仍为
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`：
  - 24 个独立 Episode；
  - 1,600 次参数更新；
  - 0 次任务成功；
  - action causality aggregate ratio `1.0012791539504238`，低于冻结门槛 `1.05`；
  - Actor 未解锁；
  - 三个正式任务均无 bilateral contact 或 controlled motion。
- 已接受的关键测量合同继续有效：
  - `R0001-P17`：实际 plant action 在正式任务与多个 horizon 上有物理因果效应；
  - `R0001-P29`：50ms 控制周期、100ms visible-source-age validity，observation latency
    1/2 为支持域；
  - `R0001-P36-E1/E2`、`R0001-P39-E1`：支持域、平衡 factorial 与 policy-seed 隔离；
  - `R0001-P40-E1/E2`：report-only contact safety 与实体接触图测量；
  - `R0001-P50-E1/E2`：不可变 acquisition evidence 与 candidate funnel 守恒测量；
  - `R0001-P51`：acquisition-frame 线速度到 current-base frame 的转换合同；
  - `R0001-P52`：policy FK 与 MuJoCo grasp-center site 数值一致；
  - `R0001-P57`：固定 P51 cohort 的 bilateral pre-contact reachability 与
    command-support deficit 测量证据。
- `R0001-P51-E1` 仍为 `rejected`；`R0001-P41-E2` 仍为 `inconclusive`。
- `R0001-P50-E3` 为 `inconclusive_design_infeasible`，没有形成 entity coverage
  measurement evidence。

## 上一轮结论

`R0011` 没有训练、policy inference、post-selection capability Episode 或新家务任务成功。

1. `R0001-P50-E3` 在 Episode 0 前 fail closed：
   - living 与 dining 的纯 body-role mapping preflight 通过；
   - kitchen 的 `kitchen_drawer` 同一 body 同时承载
     `articulation:drawer` 与 `target_container`；
   - 冻结合同要求 body-exclusive role，因而设计不可构造；
   - 没有 entity visibility、raw/component/final coverage 结果。
2. `R0001-P57` 接受为测量证据：
   - 36/36 pair 从未在同一步达到双臂 `<=0.10m` readiness；
   - 36/36 pair 的两臂 initial command margin 都为负；
   - 起始 tool-to-preposition distance 为 `1.078–3.401m`；
   - 每臂 100-step actual-applied command budget 仅为 `0.349–0.433m`；
   - 该结果只证明固定 P51 cohort 的 command-support deficit，不证明接触或严格可达。
3. simulator-private segmentation 不得进入正式 candidate generator。
4. 在 entity-hit cohort 与 phase-resolved contact 合同成立前，不启动 B3 action-vs-hold。

## 本轮主要瓶颈

`R0012-C01`：

> 当前仍无法把 P50 的 policy-visible candidate 与 exact task entity 可靠关联。上一轮证明
> 纯 body-exclusive role mapping 不能表达 articulated container：handle 与
> target-container interior/wall 可以合法共享一个 MuJoCo body，但属于不同局部任务角色。
> 若没有结果前冻结、可穷举审计且 evaluator-only 的 geom/local-region alias 合同，就无法
> 判断 task entity 是未进入视野、未形成 raw candidate、在漏斗中被删除，还是 final set
> 只覆盖 distractor；也不能据此修改 generator 或动作支持链。

本轮优先回答：

1. exact task geom 能否覆盖 collision 与 visible geom；若不能，body-local alias 是否能
   在不合并 articulation 与 target-container 的前提下完整、确定且保守地表达角色；
2. 三个正式场景能否在 Episode 前完成 exhaustive geom mapping，并对未命名 geom、
   site、background、body 内多角色边界、重复 alias 与未知像素 fail closed；
3. evaluator-private truth 能否与正式 RGB-D、动态标定和 observation identity 同步，
   且对 candidate、selector、动作、安全和终止保持可执行的单向隔离；
4. 若映射合同成立，是否值得在同一轮恢复固定 P50 cohort 的 24-Episode sidecar；若不
   成立，是否存在不依赖 entity truth 的更低成本诊断，避免继续扩大不可执行设计。

## 当前入口、证据与资源

- P50 mapping preflight：
  - `src/hwr/adapters/mujoco/entity_candidate_mapping.py`
  - `src/hwr/apps/evaluate_entity_candidate_coverage.py`
  - `tests/test_entity_candidate_coverage.py`
- P50 acquisition/funnel：
  - `src/hwr/adapters/mujoco/candidate_acquisition.py`
  - `src/hwr/eval/candidate_funnel.py`
  - `src/hwr/eval/target_selection.py`
- MuJoCo observation、rendering 与 latency queue：
  - `src/hwr/adapters/mujoco/formal_household_backend.py`
  - `src/hwr/adapters/mujoco/rendering.py`
- 场景与绑定：
  - `assets/mujoco/`
  - `configs/adapters/mujoco/formal_3d_v1.json`
  - `configs/tasks/formal_3d_v1.json`
- 固定 P50 输入：
  `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
- P50 funnel：
  `runs/research-loop/0010/r0010-p50-e2-funnel-s20265001`
- P57 结果：
  `runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701`
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- Python：`3.11.0`
- PyTorch：`2.13.0`
- MuJoCo：`3.10.0`
- 当前进程环境中 MPS：built，但 `is_available=False`；本轮不得据此假定可用加速器。
- 起始数据卷可用空间：`128,171,676 KiB`，约 122.2GiB。

## 启动期只读探针

主 Agent 在不写 artifact、不修改模型、不执行动作的条件下，用 MuJoCo `3.10.0` 在宿主
图形环境对三个正式场景的初始 `head_depth` 相机做一次 segmentation API 探针。该探针
只用于判断新 mapping 合同能否被实现，不作为正式实验结果：

- segmentation 为 `int32[192,256,2]`，内容是 `(object_id, object_type)`；
- kitchen 同一初始帧可分别看到：
  - `drawer_handle_visual`：18 pixel；
  - `drawer_front`：308 pixel；
  - `drawer_back`：15 pixel；
  - `drawer_bottom`：11 pixel；
  - `drawer_left/right`：3/12 pixel；
  - `cleaner_yellow_visual`、`cleaner_pink_visual`：117/163 pixel；
- kitchen handle visual geom 与六个 target-container geom 因而可按 exact geom 分开；
  body-exclusive 冲突不是场景不可分，而是上一轮映射抽象过早；
- manipulated object 的 segmentation 主要命中 `*_visual`，但 binding 只明确给出
  `*_collision`，所以必须结果前建立并审计 visual/collision alias；
- dining 的两个 target site 以 `object_type=SITE` 出现；site 必须固定为 unknown/excluded，
  不能因名称含 target 而计入实体分子；
- living 在该单个初始帧没有 task-relevant geom，不得把单帧探针外推为 acquisition
  visibility 结果。

## 本轮组织

- 创新 Agent A：geom/local-region evaluator mapping，独立只读。
- 创新 Agent B：与 entity truth 隔离的动作支持几何诊断，独立只读。
- 创新 Agent C：反方审计、评测泄露与更低成本替代路线，独立只读。
- 两名筛选 Agent 将在提案集冻结后独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责；主 Agent维护冻结合同、集成、门禁与最终归因。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提保留原 ID 并增加实验后缀。
3. 一个候选只验证一个主假设；测量合同、行为修订、数据采集和能力训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0011/`。
5. 不后验修改 P50、P51、P52 或 P57 的历史冻结门、artifact 与判定。
6. 不延长 100ms safety validity，不读取 future/latest observation，不降低碰撞、安全、
   deployment 或 action-causality 门槛。
7. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、geom/body/site
   ID、仿真物体位姿、reward、stage、success、contact truth 或专家动作。
8. evaluator-private entity truth 只能由 measurement sidecar 消费；不得进入正式
   observation、latency queue、candidate、selector、动作、安全控制或 termination。
9. 样本单位是独立 Episode/seed 或结果前定义的 paired block；pixel、point、anchor、
   component、candidate、frame、control step、geom 或 arm 不得冒充独立样本。
10. 不按结果挑 task、seed、latency、camera、entity、threshold、统计方法、MDE 或接受门。
11. report-only 结果不得称为学习、泛化、安全能力、闭环成功或 deployment。
12. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
13. 未获得可信 entity-hit 与 readiness 前，不恢复 selector、Replay、Actor 或世界模型训练。
14. 正式长时训练必须使用 `traex-host-exec` 与 tmux，并设置含“阅读 AGENTS.md”的完成唤起
    和定时看门狗；小型验证不得休眠。
