# R0015 研究上下文

## 轮次身份

- 轮次：`R0015`
- 状态：完成
- 起始分支：`feat/research-loop`
- 起始提交：`2ade5203e5ec229522fd4d5876b74f77c50e6b2e`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 前一轮：`R0014`
- 本轮能力基线：不变

本轮只写入 `docs/research-loop/0015/`。历史轮次文档冻结只读：

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
- 已接受的关键测量与设计证据：
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
  - `R0001-P79-E1`：隔离 v2 generator 的 deterministic mask-ownership correction。

## 上一轮结论

`R0014` 没有训练、参数更新、policy inference、MuJoCo physical acquisition、
B0–B7 action、contact phase、capability evaluation 或新家务任务成功。

1. `R0001-P79-E1` 接受为隔离 v2 candidate-generator 正确性修复：
   - local `patch_valid` 不再原地修改 parent validity mask；
   - 384 个 capture frame 在 row-major、reverse-row-major、column-major 下产生相同
     raw multiset 和 final candidate bytes；
   - 两次完整 v2 bank build bit-identical。
2. v2 bank 固化为：
   `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`。
3. legacy v1 与隔离 v2 对同一 24 Episode 的 candidate-set hash 为 24/24 不同，
   selected canonical identity 为 22/24 不同，总 candidate 数为 `39 → 36`。
4. 默认 runtime/production pipeline 未迁移到 v2；旧 P68/P76、P51/P57/P60/P66 的
   candidate-conditioned 结论不得无条件外推到 v2。
5. `R0001-P74`、revised `R0001-P76` 与拆分后的 `R0001-P78` 均为 deferred，
   下一轮必须重新创新和筛选，不自动继承。

## 本轮主要瓶颈

`R0015-C01`：

> P79 已消除 candidate generator 的扫描副作用并生成版本化 v2 bank，但 v2 尚无经过
> 验证的 consumer/version 合同，也没有结果前冻结的 interaction relevance 或
> bilateral feasibility 证据。当前不能判断 v2 selected candidate 是否值得进入
> selector/routing，更不能据此恢复 Replay、Actor、世界模型训练或 capability
> evaluation。应先在 consumer 隔离、association 与物理必要条件之间独立生成和筛选
> 最小可证伪方案，只选一个主假设进入本轮正式验证。

本轮优先回答：

1. v2 bank 应由版本化 runner、显式 generator dependency 还是默认 production migration
   消费，怎样防止 v1/v2 静默混用；
2. 能否在不沿用旧 P68 未发布结果的前提下，以 v2 support 和冻结 association semantics
   判断 initial selected candidate 与 initial microinteraction 的相关性；
3. 能否在 v2 candidate/base/target lineage 上建立 fixed-base 或 free-base bilateral
   feasibility 的独立必要条件，而不把静态外包络误称为动态不可达；
4. 哪个前置证据最能缩小下一步决策空间，且不会把 evaluator 修复、行为迁移和能力改进
   捆绑在一个因果比较中。

## 当前入口、证据与资源

- v2 generator：
  - `src/hwr/eval/candidate_mask_ownership.py`
  - `src/hwr/apps/evaluate_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership.py`
  - `tests/test_candidate_mask_ownership_app.py`
- v2 bank：
  `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`
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
  - Python `3.11.0`；
  - 数据卷可用空间约 `109GiB`；
  - tmux 可用；
  - 没有发现本项目活跃训练或评测；
  - 存在项目外高负载 Gradle/Android emulator 进程，当前不满足正式训练近似独占条件。

## 本轮组织

- 创新 Agent A：v2 consumer/version contract 与迁移隔离，独立只读。
- 创新 Agent B：v2 initial association 的 estimand、执行预算与泄露边界，独立只读。
- 创新 Agent C：v2 bilateral feasibility、反方 stopping gate 与替代路线，独立只读。
- 两名筛选 Agent 在提案冻结后独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责；主 Agent 维护冻结合同、集成、门禁与最终归因。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提保留原 ID 并增加实验后缀。
3. 一个候选只验证一个主假设；consumer 修复、默认迁移、测量优化、association、
   feasibility、行为修订和训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0014/`。
5. 不后验修改历史合同、artifact、阈值、seed、cohort 或判定。
6. v1 与 v2 schema 必须显式区分；未知或错误版本 fail closed，禁止静默转换。
7. P79 v2 bank 只来自冻结旧 P50 policy-visible captures，不冒充修复后新鲜未见分布
   cohort。
8. evaluator-private geom/body/contact/segmentation/force truth 不得进入 observation、
   latency queue、candidate、selector、动作、安全 decision、reward、termination 或训练数据。
9. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、geom/body/site
   ID、仿真物体位姿、reward、stage、success、contact truth 或专家动作。
10. 样本单位是独立 Episode/seed 或结果前定义的 paired block；frame、candidate、
    contact、pixel、control step 或 arm 不得冒充独立样本。
11. 不按结果挑 task、seed、latency、camera、candidate、threshold、统计方法、MDE 或
    接受门。
12. 静态外包络只能作为必要条件；不得把 fixed-base exclusion 外推为 free-base 动态
    不可达。
13. report-only 结果不得称为学习、泛化、安全能力、闭环成功或 deployment。
14. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
15. 在 v2 association 或 independent feasibility 前置证据通过前，不启动 selector、
    Replay、Actor、世界模型训练或 capability evaluation。
16. 正式长时训练必须使用 `traex-host-exec` 与 tmux，并设置含“阅读 AGENTS.md”的完成
    唤起和定时看门狗；小型验证不得休眠。
