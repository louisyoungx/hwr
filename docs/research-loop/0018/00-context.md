# R0018 研究上下文

## 轮次身份

- 轮次：`R0018`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`93b28efe4f743400593c2c357ff41d84104603c7`
- 起始远端：`origin/feat/research-loop` 与本地起始提交一致
- 起始工作区：干净
- 前一轮：`R0017`
- 本轮能力基线：不变
- 当前数据卷可用空间：约 `74GiB`
- 当前主机：Apple M5 Pro，18 logical CPU，48GiB unified memory，20-core GPU

本轮只写入 `docs/research-loop/0018/`。历史轮次
`docs/research-loop/0001/`～`docs/research-loop/0017/` 冻结只读。

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
| `docs/research-loop/0017/` | `7f7c96c01a55e20baa84d658268704e10d26d95e` |

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
  - `R0001-P29`：50ms 控制周期下 observation latency 1/2 位于 100ms
    visible-source-age 支持域；
  - `R0001-P40-E2`：实体接触图测量合同；
  - `R0001-P50-E1/E2/E4`：candidate funnel、不可变 acquisition evidence、
    exact-geom 加一跳 same-body visual alias；
  - `R0001-P51`：acquisition-frame 到 current-base frame 的速度转换合同；
  - `R0001-P52`：policy FK 与 MuJoCo grasp-center site 数值一致；
  - `R0001-P57`：legacy v1 cohort 的 bilateral pre-contact command-support deficit；
  - `R0001-P61/P72`：有限 direct-call 边界的信息缺口及其审计证据；
  - `R0001-P66-E1`：legacy v1 lineage 上的 production-isomorphic
    predictive-safety witness；
  - `R0001-P79-E1`：隔离 v2 candidate generator 的 deterministic
    mask-ownership correction；
  - `R0001-P83-E1`：具体 P68/P76 consumer 的 source-disjoint v2
    selection-lineage 重建证据。
- 当前隔离 v2 bank：
  `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`。
- 当前 selection-lineage artifact：
  `runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`。
- legacy v1 与隔离 v2 对同一 24 Episode 的 candidate-set hash 为 24/24 不同，
  selected canonical identity 为 22/24 不同；旧 candidate-conditioned 结论不得
  无条件外推到 v2。

## 上一轮结论

`R0017` 没有训练、参数更新、checkpoint、policy inference、MuJoCo physical
acquisition、B0–B7 action、contact phase、capability evaluation 或新家务任务成功。

1. `R0001-P87-E1` 判为 `inconclusive (invalid_design)`：
   - 冻结 C01 要求只把 selector target `18→8` 后整个合同由不可达变为可达；
   - living 与 dining 的 choice-opportunity 上限仍均为 `1`，task floor `5` 未变，
     所以单变量 control 数学上不能翻转；
   - C02 为制造 latency-floor control 又改变了 claim scope，不是单变量对照；
   - Solver B 只检查空 assignment 却标记 exhaustive；
   - contradiction verifier 可接受 `required<=available` 的伪矛盾；
   - 独立红队发现 2 blocker、3 major、1 minor并禁止正式 evaluator。
2. P87 formal evaluator 未启动，formal artifact 不存在；候选实现已从主基线回退，
   仅保留分支 `exp/R0001-P87-contract-oracle` 与提交 `485367f` 供审计。
3. `R0001-P85-E1`、`R0001-P88`、`R0001-P68-E4`、`R0001-P76-E5/E6`
   均未实施；不能自动继承为本轮候选。
4. 当前固定 bank 只有 8 个 multi-candidate Episode，living/dining/kitchen 分别
   `1/1/6`；旧 selector-negative 总门 `18/22` 与 task floor `5/5/5` 不可达。
5. 三个 `obs=1/action=1` prefix sentinel 的 safe-entry outcome 已在创新阶段暴露。
   排除它们后 living 的该 cell 没有剩余 nonempty Episode，因此不能继续把固定
   24-Episode bank 的完整 12-cell prefix cohort称为未见确认性评测。

## 本轮主要瓶颈

`R0018-C01`：

> v2 candidate generation 与 selection lineage 已有证据，但下一项高价值前置实验尚未
> 形成可执行合同。P87 重提必须证明 controls 真正单变量、solver 独立且 verifier 不会
> 自证；P88 必须先有人工可知的 coordinate/component 真值；P76 必须解决已见 sentinel、
> holdout、外部 no-B2 边界与 cohort 分账。若在冻结前不能消除这些设计缺口，任何正式
> 物理 cohort 或训练都无法产生可归因的新能力证据。

本轮优先回答：

1. 是否存在比重做完整 P87 更小、可由手工 fixture 和穷尽枚举共同验证的合同检查器，
   且其 controls 在冻结前已证明单变量和数学可达；
2. 是否应优先建立 `R0001-P88` 所需的人工精确 self-mask/component/merge oracle，
   使后续 v2 association 有独立 coordinate ledger；
3. 是否能为 P76 构造真正未见或明确 descriptive 的 cohort，并在不生成 B2 action、
   不捆绑 geometry/restore 的情况下形成 authoritative-prefix receipt；
4. 哪个候选能以最低成本最大幅度缩小决策空间，同时严格避免修改 candidate、
   selector、动作、安全、训练和评测口径中的多个变量。

## 当前入口与资源

- v2 generator 与 bank：
  - `src/hwr/eval/candidate_mask_ownership.py`
  - `src/hwr/apps/evaluate_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership_app.py`
  - `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`
- P83 selection lineage：
  - `scripts/evaluate_v2_selection_lineage_oracle.py`
  - `src/hwr/apps/evaluate_v2_selection_lineage.py`
  - `runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`
- legacy association：
  - `src/hwr/apps/evaluate_initial_candidate_association.py`
  - `src/hwr/eval/initial_candidate_association.py`
  - `tests/test_initial_candidate_association.py`
  - `tests/test_initial_candidate_association_app.py`
- legacy prefix/geometry：
  - `src/hwr/apps/evaluate_phase_entry_geometry.py`
  - `src/hwr/eval/phase_entry_geometry.py`
  - `src/hwr/adapters/mujoco/phase_entry_geometry.py`
- 项目门：
  - `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python scripts/check_architecture.py`
  - `.venv/bin/python scripts/check_python_size.py`
- 当前未启动本轮训练、正式物理 cohort、后台任务或休眠。
- 当前主机存在显著项目外 CPU 负载；在候选通过筛选且资源独占条件满足前，不启动
  正式训练。

## 本轮组织

- 创新 Agent A：独立审查 P87-E2 的最小可验证重构与单变量 controls。
- 创新 Agent B：独立审查 P88 人工精确 oracle、coordinate/component lineage 与
  association 前置。
- 创新 Agent C：独立审查 P76 holdout/descriptive prefix、外部 no-B2 边界和更低成本
  替代证据。
- 两名筛选 Agent 只在提案冻结后独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责；主 Agent维护冻结合同、Git 集成、门禁与最终归因。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 历史观点保留稳定 ID；重提必须增加新实验后缀，不覆盖历史结论。
3. 一个候选只验证一个主假设，不捆绑合同工具、coordinate lineage、association、
   prefix、geometry、restore、selector、训练或 capability evaluation。
4. 不修改 `docs/research-loop/0001/`～`0017/`。
5. 不后验修改历史合同、artifact、阈值、seed、cohort、统计方法或判定。
6. P50/P79/P83 artifact 只读；新结果写入 R0018 新路径，不覆盖旧 artifact。
7. 已见 sentinel 必须显式登记；不得把暴露 outcome 的 Episode称为未见确认性样本。
8. evaluator-private truth 不得进入 observation、latency queue、candidate、selector、
   action、安全 decision、reward、termination 或训练数据。
9. 样本单位是独立 Episode/seed 或结果前定义的 paired block；frame、candidate、
   support pixel、contact、control step、arm、branch 或 solver state不得冒充独立样本。
10. contract controls 在冻结前必须证明单变量、判定可达且有独立 verifier；不能为通过
    control 临时删除 claim、floor 或约束。
11. no-B2 必须由外部能力边界与调用级 tripwire 证明，不能仅依赖 producer 自报。
12. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
13. 在 v2 association 或 independent feasibility 前置证据通过前，不启动 selector、
    Replay、Actor、世界模型训练或 capability evaluation。
14. 正式长时训练必须使用 `traex-host-exec` 与 tmux，并设置含“阅读 AGENTS.md”的完成
    唤起和定时看门狗；小型验证不得休眠。本轮在训练候选通过筛选前不启动训练。
