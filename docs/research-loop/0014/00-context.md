# R0014 研究上下文

## 轮次身份

- 轮次：`R0014`
- 状态：完成
- 起始分支：`feat/research-loop`
- 起始提交：`9a5b5b57b089353741b217a72161eaa02532dd93`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 前一轮：`R0013`
- 本轮能力基线：不变

本轮只写入 `docs/research-loop/0014/`。历史轮次文档冻结只读：

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
  - `R0001-P57`：固定 cohort 的 bilateral pre-contact command-support deficit；
  - `R0001-P61/P72`：有限 direct-call 边界的信息缺口及其残余审计缺口；
  - `R0001-P66-E1`：production-isomorphic predictive-safety witness。
- `R0001-P51-E1` 为 `rejected`；`R0001-P41-E2` 为 `inconclusive`；
  `R0001-P60` 为 `invalid`；`R0001-P68-E1` 为
  `inconclusive_budget_exceeded`。

## 上一轮结论

`R0013` 没有训练、参数更新、policy inference、B2/B3–B7 action、contact phase、
capability Episode 或新家务任务成功。

1. `R0001-P66-E1` 接受为预测安全 witness 合同：
   - 固定 P60 anchor 的 delayed/scaled plant action 为
     `(0.1056309462, -0.1605351262)`；
   - clone predictor 在第二个 control boundary 预测到
     `robot_base/body_box_collision` 对
     `tea_table/tea_table_top_collision` 的单 contact point force 为
     `356.9927529N`；
   - 超过冻结 `220N` 门后动作被改写为 hold；
   - authoritative physics 未推进，实际 severe collision 为 0。
2. `R0001-P72-E1` 接受为 residual P61 contract-gap evidence：
   - initial annotation 四类反事实均 fail closed，P68 前置门通过；
   - 五类 exact source reference drift 未进入最终 fail-closed gate；
   - role field 可使当前 `planner_call_state_available=true`，但没有独立 planner
     state、transition ID 或 validated external planner evidence。
3. `R0001-P68-E1` 的实现通过 observer isolation 和 24/24 历史 candidate 重建，
   但固定 24-Episode baseline+treatment cohort 未在 30 分钟内完成；
   无正式或 partial artifact，不能给出 association、selector relevance 或能力结论。
4. P68 运行暴露性能根因候选：segmentation 仍在 source observation 生成时逐控制步捕获，
   而 estimand 只需要结果前定义的 capture identity。
5. P50 generator 的 `patch_valid` 是 view 上的原地收缩，具有扫描顺序依赖；这是独立
   evaluator/generator 缺陷候选，不得在 P68 因果运行中顺便修改。

## 本轮主要瓶颈

`R0014-C01`：

> 当前 capability 路线被一个未完成的前置测量阻塞：P68 尚未判断初始 selected
> candidate 是否与任务的 initial microinteraction 相关。其测量合同与 24-Episode
> cohort 已冻结且实现隔离门通过，但 runner 由于逐控制步 segmentation capture 超出
> 30 分钟预算。若直接增加预算，可能掩盖不必要计算；若同时修改 candidate generator
> 或 association estimand，则破坏可归因性。应先独立验证一个只减少冗余渲染、不改变
> capture bytes、candidate bytes、Episode cohort、分类逻辑和冻结门槛的执行方案，
> 再决定是否正式重跑 P68；若该路线仍不可执行，则转向独立 bilateral feasibility
> witness，而不是恢复训练。

本轮优先回答：

1. 能否把 source-state segmentation capture 从每个 control step 缩减到结果前定义的
   capture identity，并用相同状态快照重建 bit-identical segmentation、candidate 与
   association 输入；
2. 能否在不查看任何 association classification 的前提下，通过 timing-only preflight
   证明固定 24-Episode cohort 可在新冻结预算内完成；
3. P68 若完成，结果是否足以授权后续 selector/routing 诊断，还是应因低 relevance
   转向 `R0001-P71` independent bilateral endpoint feasibility；
4. P61 exact-reference/planner audit 修复和 P50 `patch_valid` 顺序依赖是否应作为独立
   evaluator 修复候选，且与 P68 物理测量严格分开。

## 当前入口、证据与资源

- P68 实现：
  - `src/hwr/apps/evaluate_initial_candidate_association.py`
  - `src/hwr/eval/initial_candidate_association.py`
  - `tests/test_initial_candidate_association.py`
  - `tests/test_initial_candidate_association_app.py`
- P68 冻结输入：
  - `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
  - `runs/research-loop/0012/r0012-p50-e4-mapping-s20265004`
  - `runs/research-loop/0012/r0012-p61-interaction-contract-s20266101`
  - `runs/research-loop/0013/r0013-p72-p61-mutation-s20267201`
- P66：
  `runs/research-loop/0013/r0013-p66-predictive-witness-s20266601`
- P57：
  `runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701`
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 启动环境：
  - Apple M5 Pro，20 GPU cores，48GiB RAM，18 logical CPU；
  - Python `3.11.0`、PyTorch `2.13.0`、MuJoCo `3.10.0`、NumPy `2.4.6`；
  - MPS built 且可用；
  - 数据卷可用空间约 `69GiB`；
  - 没有活跃的本项目训练或评测进程；
  - 存在项目外 tmux/Android 开发进程，任何正式训练前仍需重新核验近似独占资源。

## 本轮组织

- 创新 Agent A：P68 执行设计、快照/渲染 identity 与 bit-identity 守护，独立只读。
- 创新 Agent B：P71 bilateral feasibility 与 P66/P57 物理约束替代路线，独立只读。
- 创新 Agent C：P61/P72、P50 generator 缺陷和总体反方 stopping gate，独立只读。
- 两名筛选 Agent 在提案冻结后独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责；主 Agent维护冻结合同、集成、门禁与最终归因。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提保留原 ID 并增加实验后缀。
3. 一个候选只验证一个主假设；性能修复、测量合同、generator 修复、行为修订和训练
   分开归因。
4. 不修改 `docs/research-loop/0001/`～`0013/`。
5. 不后验修改 P50、P51、P52、P57、P60、P61、P66、P68 或 P72 的历史合同、
   artifact 与判定。
6. P68 若重提，保持同一 24-Episode cohort、`0.80` ratio、`18/24` 与 `6/24` 门；
   不复用上一轮未发布内存值，不按结果替换 seed。
7. 性能优化必须证明 source snapshot、segmentation、candidate、association 输入、
   authoritative trace 与 observer isolation 保持冻结 identity。
8. 不降低 `220N` 安全阈值，不绕过 safety rewrite，不延长 100ms validity。
9. evaluator-private geom/body/contact/segmentation/force truth 不得进入 observation、
   latency queue、candidate、selector、动作、安全 decision、reward、termination 或训练数据。
10. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、geom/body/site
    ID、仿真物体位姿、reward、stage、success、contact truth 或专家动作。
11. 样本单位是独立 Episode/seed 或结果前定义的 paired block；frame、candidate、
    contact、pixel、control step 或 arm 不得冒充独立样本。
12. 不按结果挑 task、seed、latency、camera、candidate、threshold、统计方法、MDE 或接受门。
13. report-only 结果不得称为学习、泛化、安全能力、闭环成功或 deployment。
14. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
15. 在 association 或 independent feasibility 前置证据通过前，不启动 selector、
    Replay、Actor、世界模型训练或 capability evaluation。
16. 正式长时训练必须使用 `traex-host-exec` 与 tmux，并设置含“阅读 AGENTS.md”的完成唤起
    和定时看门狗；小型验证不得休眠。
