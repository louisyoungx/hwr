# R0017 研究上下文

## 轮次身份

- 轮次：`R0017`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`26fb24ed2575bef1f268dc731323010607092fc8`
- 起始远端：`origin/feat/research-loop` 位于 `065e0b1496d5780a1dfcc8862c204e0dbcb792c7`，本地领先 1 个 `AGENTS.md` 执行连续性提交
- 起始工作区：干净
- 前一轮：`R0016`
- 本轮能力基线：不变

本轮只写入 `docs/research-loop/0017/`。历史轮次文档冻结只读：

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
| `docs/research-loop/0016/` | `077a1c032bd9e96d71f499634db46dcc76792e6c` |

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
- 当前接受的关键测量与设计证据包括：
  - `R0001-P17`：实际 plant action 在正式任务和多个 horizon 上具有物理因果效应；
  - `R0001-P29`：50ms 控制周期下，observation latency 1/2 位于 100ms visible-source-age 支持域；
  - `R0001-P40-E2`：实体接触图测量合同；
  - `R0001-P50-E1/E2/E4`：candidate funnel、不可变 acquisition evidence、exact-geom 加一跳 same-body visual alias；
  - `R0001-P51`：acquisition-frame 到 current-base frame 的速度转换合同；
  - `R0001-P52`：policy FK 与 MuJoCo grasp-center site 数值一致；
  - `R0001-P57`：legacy v1 cohort 的 bilateral pre-contact command-support deficit；
  - `R0001-P61/P72`：有限 direct-call 边界的信息缺口及其残余审计缺口；
  - `R0001-P66-E1`：legacy v1 lineage 上的 production-isomorphic predictive-safety witness；
  - `R0001-P79-E1`：隔离 v2 candidate generator 的 deterministic mask-ownership correction；
  - `R0001-P83-E1`：具体 P68/P76 consumer 的 source-disjoint v2 selection-lineage 重建证据。
- 当前隔离 v2 bank：
  `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`。
- 当前 selection-lineage artifact：
  `runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`。
- legacy v1 与隔离 v2 对同一 24 Episode 的 candidate-set hash 为 24/24 不同，
  selected canonical identity 为 22/24 不同；旧 candidate-conditioned 结论不得
  无条件外推到 v2。

## 上一轮结论

`R0016` 没有训练、参数更新、checkpoint、policy inference、MuJoCo physical
acquisition、B0–B7 action、contact phase、capability evaluation 或新家务任务成功。

1. `R0001-P83-E1` accepted as consumer-local v2 selection-lineage evidence：
   - 24/24 candidate bytes/hash/count 精确一致；
   - 24/24 full-precision score hash 与 selected index 精确一致；
   - 22/22 nonempty selected identity 与 2/2 empty selection 精确一致；
   - 384/384 capture、768/768 blind blob、796/796 manifest-bound input 完整；
   - 两次 blind rebuild bit-identical，29/29 mutation/control 通过；
   - 正式 process-tree RSS 上界约 `1.272GiB`，wall 上界约 `24.93s`。
2. P83 只证明冻结 P68/P76 consumer 可从仍存在的 P50 bytes 重建 selection lineage：
   - 不使 P79 artifact 自包含；
   - 不证明 candidate 与任务实体相关、可达、安全或可控制；
   - 不授权默认 v2 migration、selector、Replay、Actor 或训练。
3. `R0001-P85` 因遗漏 action latency 且 capture count 不能界定 eager segmentation
   wall-time，以当前形式 rejected。
4. `R0001-P86` 因三步 hold 不能覆盖非零动作、predictive rejection、contact/task
   counter 与 multi-clone 隔离，以当前形式 rejected。
5. `R0001-P68-E3` 与 `R0001-P76-E3` 均 deferred，必须在新一轮重新创新与独立筛选；
   不得自动继承。

## 本轮主要瓶颈

`R0017-C01`：

> v2 generator 与具体 consumer selection lineage 已有可复核证据，但 22 个 nonempty
> Episode 的 selected candidate 是否与初始任务实体相关，以及这些 Episode 能否安全
> 到达 authoritative B2-entry，仍完全未知。前者受 unique-coordinate association 与
> 12 个 task/observation/action latency cell 执行预算约束；后者受 continuation bridge、
> safe-prefix coverage 与 restore 完整性约束。两条路线不能在同一因果实验中捆绑，
> 本轮必须重新提出、筛选并只冻结一个主假设。

本轮优先回答：

1. P68 路线能否先建立覆盖 12 个 latency cell、结果前固定且可验证的执行预算上界，
   并把 support 投票单位改为唯一
   `(Episode, capture ordinal, row, column)`，同时分开 route availability、relevant
   candidate availability 与 selected relevance；
2. P76 路线能否在不生成或执行 B2 action 的前提下，把 P50 acquisition、v2 selected
   candidate 与 authoritative B2-entry prefix 串为完整 bridge，并对 safe-entry、
   safety-stopped、terminal 与 invalid 做穷尽分账；
3. 哪条路线的最低成本前置证据能最大幅度缩小下一步决策空间，且不同时改变
   candidate、selector、动作、安全、runtime restore、评测口径与训练；
4. 若两条路线都缺少可执行的单变量设计，应先建立必要的 measurement contract，
   不以“继续推进”为理由启动正式 cohort 或训练。

## 当前入口、证据与资源

- v2 generator 与 bank：
  - `src/hwr/eval/candidate_mask_ownership.py`
  - `src/hwr/apps/evaluate_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership_app.py`
  - `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`
  - `bank.json` SHA-256：
    `888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e`
  - `manifest.json` SHA-256：
    `162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9`
- P83 selection lineage：
  - `scripts/evaluate_v2_selection_lineage_oracle.py`
  - `src/hwr/apps/evaluate_v2_selection_lineage.py`
  - `runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`
  - `report.json` SHA-256：
    `0f09944aadb052d8b92f0b0c54cd40fb31f4835fdab1c44d2f24ef48fffd2513`
  - `manifest.json` SHA-256：
    `f149a586e30cb8d29156c685e084507b6a9adf490786993ffbaba589a04bd565`
- legacy P68：
  - `src/hwr/apps/evaluate_initial_candidate_association.py`
  - `src/hwr/eval/initial_candidate_association.py`
  - `tests/test_initial_candidate_association.py`
  - `tests/test_initial_candidate_association_app.py`
  - 当前实现明确要求 legacy v1、保留 multiplicity support，并做双 replay；不能直接
    作为 v2 正式 evaluator。
- legacy P60/P76 前置：
  - `src/hwr/apps/evaluate_phase_entry_geometry.py`
  - `src/hwr/eval/phase_entry_geometry.py`
  - `src/hwr/adapters/mujoco/phase_entry_geometry.py`
  - 当前 P60 cohort 与 P79 Episode 无交集，且没有可恢复的 authoritative
    B2-entry continuation；只能作为机制证据。
- P50 输入：
  - `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
  - 当前约 `292MiB`，受 `.gitignore` 排除但由 P79 manifest 绑定。
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 启动资源：
  - macOS `26.5.1`，18 logical/18 physical CPU，48GiB RAM；
  - Python `3.11.0`、PyTorch `2.13.0`、MuJoCo `3.10.0`、NumPy `2.4.6`；
  - MPS 可用；
  - 数据卷可用空间约 `77GiB`；
  - tmux 可用；
  - 未发现本项目活跃训练或评测；
  - 存在三个项目外 tmux 会话，当前不满足正式训练近似独占条件。

## 本轮组织

- 创新 Agent A：v2 unique-coordinate association、12-cell 执行预算与最小分类合同，
  独立只读。
- 创新 Agent B：authoritative-prefix bridge、safe-prefix coverage 与 restore/continuation
  边界，独立只读。
- 创新 Agent C：从反方审查两条路线，寻找更低成本的决策证据、泄露风险与停止门，
  独立只读。
- 两名筛选 Agent 在提案冻结后独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责；主 Agent维护冻结合同、集成、门禁和最终归因。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史 P68、P76、P77、P85、P86 重提时保留原 ID 并增加实验后缀。
3. 一个候选只验证一个主假设；预算优化、association、prefix、restore、geometry、
   selector、默认迁移、行为修改与训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0016/`。
5. 不后验修改历史合同、artifact、阈值、seed、cohort、统计方法或判定。
6. P79/P83 artifact 只读；新结果写入 R0017 新路径，不覆盖旧 artifact。
7. P50 bytes 缺失或 hash 漂移必须 fail closed；Git commitment 不冒充原始 bytes。
8. v1 与 v2 schema 显式区分；未知或错误版本 fail closed，禁止静默转换。
9. evaluator-private geom/body/contact/segmentation/force truth 不得进入 observation、
   latency queue、candidate、selector、动作、安全 decision、reward、termination 或训练数据。
10. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、
    geom/body/site ID、仿真物体位姿、reward、stage、success、contact truth 或专家动作。
11. 样本单位是独立 Episode/seed 或结果前定义的 paired block；frame、candidate、
    support pixel、contact、control step、arm、branch 或 search node 不得冒充独立样本。
12. association 必须分开 route availability、nonempty denominator、relevant candidate
    availability 与 selected relevance；support multiplicity 不得当作独立票数。
13. association label 只能由进程隔离的揭盲阶段读取，不得影响 replay、candidate 或
    selection；执行预算必须覆盖 12 个 task/observation/action latency cell 或提供
    可验证的组成上界。
14. prefix 路线不得生成或执行 B2 action；safety-stopped、terminal、input-invalid 与
    safe-entry 必须穷尽分账，coverage 不足判 inconclusive。
15. 静态外包络只作为瞬时必要条件；不得把 fixed-base exclusion 外推为 free-base
    动态不可达。
16. 不按结果挑 task、seed、latency、camera、candidate、threshold、统计方法、MDE 或
    接受门。
17. report-only 结果不得称为学习、泛化、安全能力、闭环成功或 deployment。
18. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
19. 在 v2 association 或 independent feasibility 前置证据通过前，不启动 selector、
    Replay、Actor、世界模型训练或 capability evaluation。
20. 正式长时训练必须使用 `traex-host-exec` 与 tmux，并设置含“阅读 AGENTS.md”的完成
    唤起和定时看门狗；小型验证不得休眠。本轮在训练候选通过筛选前不启动训练。
