# R0011 研究上下文

## 轮次身份

- 轮次：`R0011`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`8d14fadb2b9103386788dc3d5426d3624fd624d7`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-23`
- 前一轮：`R0010`
- 本轮能力基线：不变

本轮只写入 `docs/research-loop/0011/`。历史轮次冻结只读：

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
- 已接受的证据合同继续有效：
  - `R0001-P17`：实际 plant action 在三个正式任务及 `1/4/8/16` 步 horizon 上有物理因果效应；
  - `R0001-P29`：50ms 控制周期、100ms visible-source-age validity，observation latency
    1/2 为支持域，latency 3 为挑战域；
  - `R0001-P36-E1`：支持域与挑战域分账；
  - `R0001-P39-E1`：标准 policy seed 隔离；
  - `R0001-P36-E2`：平衡 factorial benchmark 合同；
  - `R0001-P40-E1`：report-only allowed-contact 安全测量合同；
  - `R0001-P40-E2`：evaluator-private 实体—机器人部位接触图与接触同期运动测量合同；
  - `R0001-P51`：P41 primitive 的 acquisition-frame 线速度正确转换到当前 base frame；
  - `R0001-P52`：冻结机器人模型上 policy FK 与 MuJoCo grasp-center site 数值一致；
  - `R0001-P50-E1`：完整 12-cell acquisition policy-input bytes 可不可变封存和确定性重放；
  - `R0001-P50-E2`：冻结 candidate generator 的 anchor/component/ranking 漏斗可守恒测量。
- `R0001-P32-E1` 仍为 `inconclusive`，不授权新动作目标、Actor 解锁或训练。
- `R0001-P41-E2` 仍为 `inconclusive`，不授权 selector 正式对照。
- `R0001-P51-E1` 已被 `rejected`：frame-fixed 相对 legacy 的冻结 B2 normalized-AUC
  改善为 `+0.023449928237828013`，远低于结果前 MDE `0.10`。

## 上一轮结论

`R0010` 接受：

1. `R0001-P50-E1` 为 immutable acquisition evidence contract；
2. `R0001-P50-E2` 为 candidate-funnel measurement evidence。

`R0010` 拒绝：

1. `R0001-P51-E1` 作为有意义的 paired physical Cartesian convergence 改善证据。

关键观测：

- 24 个 acquisition Episode 共得到 39 个 final candidate，5 个 Episode 为空；
- 876,960 个 anchor 产生 225 个 raw candidate、149 个 connected component 和 39 个
  final candidate；
- 110 个 component 被 `view_count<2` 拒绝；
- 5 个空 Episode 均已有 raw candidate 和 component，最终在 view-count 阶段归零；
- prominence 与 center depth spread 是普遍的大比例描述性损失阶段；
- 当前证据没有回答候选是否对应 task-relevant entity，因此不能据 rejection rate
  后验放宽任何 generator gate。

本轮不得继续追逐已被 P51-E1 排除的坐标、FK、速度、phase 或 gripper 组合，也不得在
candidate entity coverage 成立前恢复 selector、Replay、Actor 或世界模型训练。

## 本轮主要瓶颈

`R0011-C01`：

> P50-E1/E2 已证明候选输入可重放，并定位了候选漏斗的描述性损失，但现有 artifact
> 没有与 policy-visible observation 同时刻、进入 observation-latency queue 前绑定的
> evaluator-private 实体真值。因此无法区分“task entity 未进入视野”“task entity
> 可见但未形成 raw candidate”“task-entity component 被某一 gate 删除”和“final
> candidate 只覆盖 distractor”四种正交失败，也无法为任何单变量 generator 修订提供
> 因果依据。

本轮优先回答：

1. 能否建立 observation-time、queue-before、只读单向隔离的 entity sidecar，并与
   RGB-D、动态标定及 `(timestamp_ns, sequence_id)` 一一绑定；
2. 能否在不修改正式 candidate 输出和动作轨迹的前提下，把 task-entity pixel/point
   支持守恒归因到 candidate funnel 的各阶段；
3. mixed、unknown、遮挡、机器人 self-mask、透明或同像素多 geom 等情况能否结果前
   定义并 fail closed；
4. entity-conditioned evidence 是否足以选择下一轮的一个 generator 主变量，还是只能
  判定当前设计或可见性不足。

## 当前入口、证据与资源

- 正式任务：
  - `tidy_living_room_3d/v1`
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
- P50 acquisition：
  - `src/hwr/adapters/mujoco/candidate_acquisition.py`
  - `src/hwr/apps/evaluate_candidate_acquisition.py`
- P50 funnel：
  - `src/hwr/eval/candidate_funnel.py`
  - `src/hwr/eval/target_selection.py`
- MuJoCo observation 与 latency queue：
  - `src/hwr/adapters/mujoco/formal_household_backend.py`
- P40/P56 可复用实体与机器人部位解析：
  - `src/hwr/adapters/mujoco/entity_contact_graph.py`
  - `src/hwr/adapters/mujoco/contact_ledger.py`
- P50-E1 正式产物：
  `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
- P50-E2 正式产物：
  `runs/research-loop/0010/r0010-p50-e2-funnel-s20265001`
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 训练入口：
  `scripts/start_foundation_training_tmux.sh <RUN_ID> [--resume] [--seed <SEED>]`
- 起始数据卷可用空间：`88,558,612 KiB`，约 84.5GiB。

## 本轮组织

- 创新 Agent A：实体可见性与候选覆盖方向，只读。
- 创新 Agent B：因果控制与物理交互方向，只读。
- 创新 Agent C：评测有效性、安全与统计设计方向，只读。
- 两名筛选 Agent 将对同一冻结提案集独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责，文件所有权、实验预算和停止条件在结果前冻结。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提保留原 ID 并增加实验后缀。
3. 一个候选只改变一个主变量；测量诊断、行为修订、数据采集与能力训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0010/`。
5. 不后验修改 R0010 P50-E1/E2 或 P51-E1 的冻结门与历史判定。
6. 不延长 100ms safety validity，不移除 latency 3，不读取 future/latest observation，
   不降低碰撞、安全、deployment 或 action-causality 门槛。
7. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、geom/body/site
   ID、仿真物体位姿、reward、stage、success、contact truth、专家 waypoint/action。
8. evaluator-private entity truth 只能由 measurement sidecar 消费；不得进入
   observation、latency queue、candidate、selector、动作、安全控制或 termination。
9. 样本单位是独立 Episode/seed 或结果前定义的 paired block；不得把 pixel、point、
   anchor、component、candidate、frame、control step 或 branch 冒充独立样本。
10. 不按结果挑 task、seed、latency、frame、camera、entity、threshold、统计方法、MDE
    或接受门。
11. report-only entity coverage、funnel attribution 和 safety 结果不得称为学习、泛化、
    安全能力或闭环任务成功。
12. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
13. 能力结论只能来自冻结未见分布上的闭环物理结果。
14. 正式长时训练必须经 `traex-host-exec` 与 tmux 启动，并配置带“阅读 AGENTS.md”的完成
    唤起和定时看门狗；小型验证不得休眠。
