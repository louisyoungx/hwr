# R0016 研究上下文

## 轮次身份

- 轮次：`R0016`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`7c0a226bf9fbac621ce056cff891d9ea8608a5a4`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 前一轮：`R0015`
- 本轮能力基线：不变

本轮只写入 `docs/research-loop/0016/`。历史轮次文档冻结只读：

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
| `docs/research-loop/0013/` | `2d885c0ad96af70b1f8808f0d8a6b700444b4a51` |
| `docs/research-loop/0014/` | `0f25b24cf5a854af7f9712ed52610cb417395ad7` |
| `docs/research-loop/0015/` | `e2495c4d70014a231ccbaf3ba5900f0ed57acc88` |

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
- 当前接受的关键测量与设计证据：
  - `R0001-P17`：实际 plant action 在正式任务和多个 horizon 上具有物理因果效应；
  - `R0001-P29`：50ms 控制周期下，observation latency 1/2 位于 100ms
    visible-source-age 支持域；
  - `R0001-P40-E2`：实体接触图测量合同；
  - `R0001-P50-E1/E2/E4`：candidate funnel、不可变 acquisition evidence、
    exact-geom 加一跳 same-body visual alias；
  - `R0001-P51`：acquisition-frame 到 current-base frame 的速度转换合同；
  - `R0001-P52`：policy FK 与 MuJoCo grasp-center site 数值一致；
  - `R0001-P57`：legacy v1 cohort 的 bilateral pre-contact command-support deficit；
  - `R0001-P61/P72`：有限 direct-call 边界的信息缺口及其残余审计缺口；
  - `R0001-P66-E1`：legacy v1 lineage 上的 production-isomorphic predictive-safety
    witness；
  - `R0001-P79-E1`：隔离 v2 candidate generator 的 deterministic mask-ownership
    correction。
- 当前隔离 v2 bank：
  `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`。
- legacy v1 与隔离 v2 对同一 24 Episode 的 candidate-set hash 为 24/24 不同，
  selected canonical identity 为 22/24 不同；旧 candidate-conditioned 结论不得
  无条件外推到 v2。

## 上一轮结论

`R0015` 没有训练、参数更新、checkpoint、policy inference、MuJoCo physical
acquisition、B0–B7 action、contact phase、capability evaluation 或新家务任务成功。

1. `R0001-P80-E1` rejected：
   - version-sealed 双根 reader、Git receipt 与 25 类 mutation 具有局部证据价值；
   - 但 default-generator audit 可被 dead v1 reference 加动态 v2 schema 绕过；
   - architecture fingerprint 与实现同源，不能证明未来任意 consumer 完备；
   - P79 bank 缺少独立 v2 score bytes/selection receipt；
   - full-validation process-tree RSS 约 `2.443GiB`，超过冻结 `2GiB` 门。
2. P80 正式 runner 未启动，没有正式或 partial artifact；rejected 实现已由
   `788a61c` 回退，当前基线不包含其代码，原实现提交 `3fff838` 保留可追溯。
3. `R0001-P81` 作为独立提案因与 P79 replay 重复而 rejected。
4. `R0001-P82`、`R0001-P68-E2`、`R0001-P74-E1`、`R0001-P76-E2` deferred；
   不自动进入本轮。
5. `R0001-P77-E2` 因缺少具体 witness/rejection 门且 search objective 有
   evaluator-private truth 泄露风险，以当前形式 rejected。

## 本轮主要瓶颈

`R0016-C01`：

> 隔离 v2 generator 已通过确定性正确性门，但当前 v2 bank 的 selected metadata
> 缺少可独立复核的 score/selection lineage，下游也没有结果前冻结的 initial
> association 或 authoritative B2-entry feasibility 证据。R0015 证明试图建立
> “所有未来 consumer 都不会绕过”的通用 Python 完备性合同不可行。下一步必须在
> producer-side selection receipt、具体 consumer 的 source identity、association
> estimand 与物理 feasibility 之间重新生成最小可证伪方案，并只选择一个主假设。

本轮优先回答：

1. 能否在不覆盖 P79 历史 bank、不改 selector 语义的前提下，生成可由独立 oracle
   重建的 v2 score/selection receipt；该证据是否是 association/feasibility 的真实
   阻塞项；
2. 能否把 v2 initial association 拆为 24 Episode route availability 与 22 个
   nonempty Episode 的 selected-support relevance，并建立 unique-support-coordinate
   oracle，避免 pixel/support multiplicity 伪重复；
3. 能否建立 v2 capture 到 authoritative B2-entry prefix 的显式 bridge，先报告
   safe-prefix coverage，再在 eligible denominator 上判断 fixed-base outer-envelope
   必要条件；
4. 哪项最低成本证据最能缩小下一步决策空间，同时不捆绑 producer 修复、consumer
   迁移、evaluator 优化、物理行为修改与能力训练。

## 当前入口、证据与资源

- v2 generator 与 bank：
  - `src/hwr/eval/candidate_mask_ownership.py`
  - `src/hwr/apps/evaluate_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership_app.py`
  - `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`
- legacy P68：
  - `src/hwr/apps/evaluate_initial_candidate_association.py`
  - `src/hwr/eval/initial_candidate_association.py`
  - `tests/test_initial_candidate_association.py`
  - `tests/test_initial_candidate_association_app.py`
- legacy physical evidence：
  - P51：`runs/research-loop/0010/`
  - P57：`runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701`
  - P66：`runs/research-loop/0013/r0013-p66-predictive-witness-s20266601`
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 启动资源：
  - Apple `Mac17,8`，18 logical CPU，48GiB RAM；
  - Python `3.11.0`、PyTorch `2.13.0`、MuJoCo `3.10.0`、NumPy `2.4.6`；
  - MPS 可用；
  - 数据卷可用空间约 `103GiB`；
  - tmux 可用；
  - 未发现本项目活跃训练或评测；
  - 存在项目外 Android emulator、Gradle 与 tmux 任务，当前不满足正式训练近似独占
    条件。

## 本轮组织

- 创新 Agent A：producer-side v2 score/selection receipt 与最小独立 oracle，独立只读。
- 创新 Agent B：v2 initial association、unique support coordinate 与执行预算，
  独立只读。
- 创新 Agent C：v2 authoritative-prefix bridge、bilateral feasibility 与反方
  stopping gate，独立只读。
- 两名筛选 Agent 在提案冻结后独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责；主 Agent 维护冻结合同、集成、门禁与最终归因。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提保留原 ID 并增加实验后缀。
3. 一个候选只验证一个主假设；producer receipt、consumer source lock、association、
   feasibility、性能优化、默认迁移、行为修订和训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0015/`。
5. 不后验修改历史合同、artifact、阈值、seed、cohort 或判定。
6. 不重提“证明所有未来 Python consumer 不会绕过 resolver”的通用 P80 假设。
7. P79 v2 bank 只来自冻结旧 P50 policy-visible captures，不冒充修复后新鲜未见分布
   cohort；任何新 receipt 必须版本化并写入新路径，不覆盖 P79 bank。
8. 具体 consumer 若需要锁定，只锁定该实验结果前冻结的 source blob、imports、入口与
   artifact identity，不声称 whole-program completeness。
9. v1 与 v2 schema 必须显式区分；未知或错误版本 fail closed，禁止静默转换。
10. evaluator-private geom/body/contact/segmentation/force truth 不得进入 observation、
    latency queue、candidate、selector、动作、安全 decision、reward、termination 或
    训练数据。
11. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、
    geom/body/site ID、仿真物体位姿、reward、stage、success、contact truth 或专家动作。
12. 样本单位是独立 Episode/seed 或结果前定义的 paired block；frame、candidate、
    support pixel、contact、control step 或 arm 不得冒充独立样本。
13. association 必须分开 route availability、nonempty denominator、relevant candidate
    availability 与 selected relevance；support multiplicity 不得当作独立票数。
14. 静态外包络只能作为必要条件；不得把 fixed-base exclusion 外推为 free-base 动态
    不可达；coverage 不足必须判 inconclusive。
15. 不按结果挑 task、seed、latency、camera、candidate、threshold、统计方法、MDE 或
    接受门。
16. report-only 结果不得称为学习、泛化、安全能力、闭环成功或 deployment。
17. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
18. 在 v2 association 或 independent feasibility 前置证据通过前，不启动 selector、
    Replay、Actor、世界模型训练或 capability evaluation。
19. 正式长时训练必须使用 `traex-host-exec` 与 tmux，并设置含“阅读 AGENTS.md”的完成
    唤起和定时看门狗；小型验证不得休眠。
