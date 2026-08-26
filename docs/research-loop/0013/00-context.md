# R0013 研究上下文

## 轮次身份

- 轮次：`R0013`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`e49e3d7112d0d4773475f53deaa7e97a5c20f6ad`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-26`
- 前一轮：`R0012`
- 本轮能力基线：不变

本轮只写入 `docs/research-loop/0013/`。历史轮次冻结只读：

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
| `docs/research-loop/0012/` | `db73bb9a6c6155d0366d7d92718aec614e044a5f` |

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
- 已接受的测量与设计证据继续有效：
  - `R0001-P17`：实际 plant action 在正式任务与多个 horizon 上有物理因果效应；
  - `R0001-P29`：50ms 控制周期、100ms visible-source-age validity，observation latency
    1/2 为支持域；
  - `R0001-P36-E1/E2`、`R0001-P39-E1`：支持域、平衡 factorial 与 policy-seed 隔离；
  - `R0001-P40-E1/E2`：report-only contact safety 与实体接触图测量；
  - `R0001-P50-E1/E2`：不可变 acquisition evidence 与 candidate funnel 守恒测量；
  - `R0001-P50-E4`：exact-geom + 一跳 same-body visual alias 的 evaluator mapping 合同；
  - `R0001-P51`：acquisition-frame 线速度到 current-base frame 的转换合同；
  - `R0001-P52`：policy FK 与 MuJoCo grasp-center site 数值一致；
  - `R0001-P57`：固定 P51 cohort 的 bilateral pre-contact command-support deficit；
  - `R0001-P61`：有限静态 same-function direct-call 边界的 interaction-contract gap。
- `R0001-P51-E1` 仍为 `rejected`；`R0001-P41-E2` 仍为 `inconclusive`。
- `R0001-P60` 为 `invalid`，没有产生 phase-entry geometry 结论。

## 上一轮结论

`R0012` 没有训练、policy inference、B2 action、contact phase、capability Episode 或新家务
任务成功。

1. `R0001-P61` 接受为有限静态信息合同证据：
   - 当前 generic candidate-centered primitive 缺少 full-task transition 所需的
     entity role、interaction selector、destination target 与 articulation threshold；
   - 该结果不覆盖动态调用、跨函数 dataflow 或潜在外部 planner。
2. `R0001-P50-E4` 接受为 evaluator mapping 合同：
   - 3/3 场景、8/8 visual alias 与 40 个负守护通过；
   - 只证明 mapping 可构造，不是 entity visibility、candidate coverage 或能力证据。
3. `R0001-P60` 在首个 cell 的第二个 latency-matched physical prefix 于 B1 触发：
   - `action_rejected / predicted_severe_collision`；
   - proposed base command 为 `(linear=0.12, angular=-0.1822977766101701)`；
   - safety 把动作改写为 hold；
   - actual severe collision count 为 `0`；
   - 冻结合同要求首个 safety intervention 立即停止，故 cohort 清空且判定 `invalid`。
4. 不允许忽略 safety rewrite、降低 `220N` 阈值、继续同一 run、换 salt 或把预测拒绝称为
   实际碰撞。

## 本轮主要瓶颈

`R0013-C01`：

> `R0001-P60` 尚未触及 phase-entry geometry estimand，唯一有 candidate 的新鲜 prefix
> 在 B1 被独立安全层以 `predicted_severe_collision` 拒绝。当前 artifact 没有保存预测
> 分支的最大 forbidden force、对应 contact pair、预测推进状态或拒绝相对路径阶段，
> 因而无法区分：通用 base routing 与静态障碍冲突、candidate-conditioned approach
> 诱发、预测器保守裕量、还是测量合同本身缺少必要观测。没有这一可审计归因，不应修改
> B1 controller，也不能绕过安全层恢复 P60。

本轮优先回答：

1. safety predictor 是否能在不改变 action decision、阈值、物理状态和 observation 的前提
   下，确定地导出预测最大 forbidden force、contact geom/body pair、预测时间点、
   `physics_advanced` 与 proposed/applied action；
2. 单个 hard-stop 是否只足以建立可执行诊断合同，而不足以支持总体来源结论；若需要新鲜
   cohort，独立 Episode/seed 的样本单位、停止门和预算应如何冻结；
3. 能否用结果前定义的 action-blind 或 path-prefix counterfactual 区分 base-path obstacle、
   candidate-conditioned heading 与 predictor margin，而不让 evaluator-private truth
   进入 action path；
4. 若不能形成非自证、非泄露、safety-preserving 的归因合同，是否应停止 B1 路线，优先
   建立 P61 planner/interface 或 P62 feasibility 的更低成本前置证据。

## 当前入口、证据与资源

- P60 正式 artifact：
  `runs/research-loop/0012/r0012-p60-phase-entry-s20266001`
- P60 evaluator：
  - `src/hwr/apps/evaluate_phase_entry_geometry.py`
  - `src/hwr/eval/phase_entry_geometry.py`
  - `tests/test_phase_entry_geometry.py`
  - `tests/test_phase_entry_geometry_app.py`
- 独立安全层：
  - `src/hwr/safety/dual_arm.py`
  - `src/hwr/safety/supervisor.py`
  - `src/hwr/adapters/mujoco/formal_household_backend.py`
  - 相关 safety/backend tests
- P61：
  - `configs/eval/interaction_contract_v1.json`
  - `src/hwr/eval/interaction_contract.py`
  - `src/hwr/apps/audit_interaction_contract.py`
- P50-E4：
  - `configs/eval/entity_candidate_aliases_v1.json`
  - `src/hwr/adapters/mujoco/entity_candidate_mapping.py`
  - `src/hwr/apps/audit_entity_candidate_mapping.py`
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 启动环境：
  - Apple `Mac17,8`，48GiB RAM，18 logical CPU；
  - Python `3.11.0`；
  - PyTorch `2.13.0`；
  - MuJoCo `3.10.0`；
  - MPS built 且启动时 `is_available=True`；
  - 数据卷可用空间约 `106GiB`；
  - 启动时存在两个 Android emulator、Defender 与其他高 CPU 进程。任何正式训练前必须
    重新检查并获得可比、近似独占的资源条件，不能把当前资源状态视为训练许可。

## 本轮组织

- 创新 Agent A：安全预测器证据与 evaluator-only 诊断，独立只读。
- 创新 Agent B：B0/B1 base routing 与 candidate-conditioned path geometry，独立只读。
- 创新 Agent C：反方审计、P61/P62/P64 替代路线与 stopping gate，独立只读。
- 两名筛选 Agent 将在提案集冻结后独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责；主 Agent 维护冻结合同、集成、门禁与最终归因。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提保留原 ID 并增加实验后缀。
3. 一个候选只验证一个主假设；instrumentation、测量合同、行为修订和训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0012/`。
5. 不后验修改 P50、P51、P52、P57、P60 或 P61 的历史合同、artifact 与判定。
6. 不降低 `220N` 安全阈值，不绕过或忽略 safety rewrite，不读取 future/latest
   observation，不延长 100ms safety validity。
7. 预测安全拒绝不得称为实际碰撞；预测分支不得推进正式 physics。
8. evaluator-private geom/body/contact/force truth 只用于 measurement sidecar；不得进入
   observation、latency queue、candidate、selector、动作、安全 decision、reward、
   termination 或训练数据。
9. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、geom/body/site
   ID、仿真物体位姿、reward、stage、success、contact truth 或专家动作。
10. 样本单位是独立 Episode/seed 或结果前定义的 paired block；contact、geom、预测步、
    frame、candidate、control step 或 arm 不得冒充独立样本。
11. 不按结果挑 task、seed、latency、camera、candidate、contact pair、threshold、统计
    方法、MDE 或接受门。
12. report-only 结果不得称为学习、泛化、安全能力、闭环成功或 deployment。
13. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
14. 在 safety attribution 与独立 feasibility witness 前，不启动 selector、Replay、Actor、
    世界模型训练或 capability evaluation。
15. 正式长时训练必须使用 `traex-host-exec` 与 tmux，并设置含“阅读 AGENTS.md”的完成唤起
    和定时看门狗；小型验证不得休眠。
