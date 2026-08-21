# R0009 研究上下文

## 轮次身份

- 轮次：`R0009`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`0fea5f3fce43a9d00ab902138ff1aea63015f1d0`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-21`
- 前一轮：`R0008`
- 本轮能力基线：不变

本轮文档只写入 `docs/research-loop/0009/`。历史轮次冻结只读：

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
  - `R0001-P40-E2`：evaluator-private 实体—机器人部位接触图与接触同期运动测量合同。
- `R0001-P32-E1` 仍为 `inconclusive`。现有 24 个独立 Replay source 的 ordinary
  action information 被 controller history 解释，exact-pipeline 10% planted power 仅
  `1/200`，不授权新动作目标、Actor 解锁或训练。
- `R0001-P41-E2` 仍为 `inconclusive`。其 smoke 完成 6 个 planned pair、12 个 branch，
  全部安全与一致性守护通过，但没有执行 selector 对照：
  - 候选数按客厅 `(1,1)/(2,2)`、餐桌 `(1,1)/(2,2)`、厨房 `(1,1)/(2,2)` 依次为
    `4/0/1/3/5/3`；
  - 客厅 `(observation latency=2, action latency=2)` 的冻结 candidate set 为空；
  - 该 cell acquisition 未失败，取得 18 个 keyframe，完整执行 1,655 control step；
  - 因结果前 nonempty 守护失败，未启动 54 supported + 18 challenge pair 正式对照；
  - 不允许后验放宽 P41 的 threshold、mask、camera、seed、task、primitive 或样本量。

## 上一轮结论

`R0008` 接受 `R0001-P40-E2` 为实体接触测量合同证据，保留
`R0001-P41-E2` 为证据不足。没有正式训练、没有新家务任务成功、没有能力基线变化。

R0008 明确要求下一轮：

1. 将 P41 停止原因作为 observation/candidate coverage 问题，而不是 selector 效果；
2. 用新的结果前冻结实验区分感知覆盖、候选局部筛选、跨视点聚类与可达性；
3. 若重提 target-index 对照，先独立证明每个支持域 cell 的共享候选集非空且可重建；
4. 不在当前 24-source Replay 上继续 P31、P43 或新的 action objective 训练；
5. 可靠性与 deployment 防火墙只能作为侧车，不能替代零交互瓶颈。

## 本轮主要瓶颈

`R0009-C01`：

> 当前最接近物理交互的无训练诊断尚未进入 selector 因果比较，因为一个完整、安全、具有
> 18 个 policy-visible RGB-D keyframe 的支持域 Episode 在冻结候选生成器中得到空集合。
> 现有证据没有定位候选在哪个结果盲阶段消失，也没有证明可行的通用单变量修订会提升支持域
> 非空覆盖。若不先完成这一归因，直接改阈值、改相机、改 seed 或扩大正式样本都会形成
> 后验调参；直接恢复世界模型训练则仍缺可识别的物理交互数据。

本轮优先回答：

1. 能否以 report-only、policy-visible 的候选漏斗诊断，定位 raw depth coverage、anchor
   validity、prominence/planarity/width/range/self-mask、跨视点合并与 view-count 门中的
   首个主要损失阶段；
2. 能否在不读取 task/object ID、仿真真值、contact、reward 或 success 的前提下，结果前
   冻结一个且仅一个候选覆盖变量，并在独立 seed 上提高三任务支持域非空率；
3. 候选非空是否仍不足以支撑 P41 primitive；是否必须先设置可执行性、动作非塌缩、安全与
   实体接触测量守护；
4. 若覆盖修订仍无充分功效，是否转向 task-balanced 独立 source 采集合同，而不是继续
   修改世界模型。

## 当前入口、证据与资源

- 正式任务：
  - `tidy_living_room_3d/v1`
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
- P41 纯候选合同：`src/hwr/eval/target_selection.py`
- P41 MuJoCo bridge：
  `src/hwr/adapters/mujoco/target_selection_diagnostic.py`
- P41 入口：`src/hwr/apps/evaluate_target_selection.py`
- P40-E2 测量：
  - `src/hwr/adapters/mujoco/entity_contact_graph.py`
  - `src/hwr/apps/evaluate_entity_contact_graph.py`
- R0008 P41 smoke：
  `runs/research-loop/0008/r0008-p41-target-selection-e2-smoke-s20264101`
- 当前普通 Replay：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/replay/autonomous`
  - 24 个独立 source；
  - 168 个 shard；
  - 2,688 个保留 transition；
  - task/source 为餐桌 6、厨房 6、客厅 12。
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 训练入口：
  `scripts/start_foundation_training_tmux.sh <RUN_ID> [--resume] [--seed <SEED>]`
- 本机：
  - Apple M5 Pro，18 核，48GB 内存；
  - 数据卷可用空间约 87GiB；
  - Python 3.11.0、PyTorch 2.13.0、MuJoCo 3.10.0；
  - 当前沙箱 `torch.backends.mps.is_available() == False`；
  - 正式 MPS 任务必须在 host-exec 环境重新核验。

## 本轮组织

- 创新 Agent A：因果学习与世界模型方向，只读。
- 创新 Agent B：数据效率、策略与表征泛化方向，只读。
- 创新 Agent C：评测有效性、安全与闭环物理方向，只读。
- 两名筛选 Agent 将对同一冻结提案集独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责，文件所有权与停止条件在 `03-experiment.md` 结果前冻结。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提必须保留原 ID 并增加修订号。
3. 一个候选只改变一个主变量；测量诊断、行为修订、数据采集与能力训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0008/`。
5. 不后验修改 R0008 P41-E2，也不以新结果覆盖其 `inconclusive` 结论。
6. 不延长 100ms safety validity，不移除 latency 3，不读取 future/latest observation，
   不降低碰撞、安全、deployment 或 action-causality 门槛。
7. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、geom/body/site
   ID、仿真物体位姿、reward、stage、success、contact truth、专家 waypoint/action。
8. 任何候选覆盖实验必须以 Episode/seed 为独立单位；不得把 frame、anchor、raw candidate、
   transition、branch 或优化 seed 冒充独立样本。
9. 不按结果挑 task、seed、latency、frame、camera、mask、阈值、统计方法、MDE 或接受门。
10. 无训练候选覆盖、交互 smoke、离线诊断和 report-only safety 结果不得称为学习、泛化、
    安全能力或闭环任务成功。
11. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
12. 能力结论只能来自冻结未见分布上的闭环物理结果。
13. 正式长时训练必须经 `traex-host-exec` 与 tmux 启动，并配置带“阅读 AGENTS.md”的完成
    唤起和定时看门狗；小型验证不得休眠。
