# R0010 研究上下文

## 轮次身份

- 轮次：`R0010`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`921f07fc7e2d6cf6c5eabc21063d163c4dbfc288`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-21`
- 前一轮：`R0009`
- 本轮能力基线：不变

本轮文档只写入 `docs/research-loop/0010/`。历史轮次冻结只读：

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
  - `R0001-P51`：P41 primitive 的 acquisition-frame 线速度会正确转换到当前 base frame；
  - `R0001-P52`：冻结机器人模型上 policy FK 与 MuJoCo grasp-center site 数值一致。
- `R0001-P32-E1` 仍为 `inconclusive`。现有 24 个独立 Replay source 的 ordinary
  action information 被 controller history 解释，exact-pipeline 10% planted power 仅
  `1/200`，不授权新动作目标、Actor 解锁或训练。
- `R0001-P41-E2` 仍为 `inconclusive`。R0008 smoke 完成 6 个 planned pair、12 个
  branch，安全与一致性守护通过，但一个支持域 cell 的 candidate set 为空，因此没有执行
  selector 对照。

## 上一轮结论

`R0009` 接受：

1. `R0001-P51` 为 Cartesian primitive correctness evidence；
2. `R0001-P52` 为 FK agreement contract evidence。

没有正式训练、policy inference、closed-loop capability Episode、新任务成功或能力基线
变化。P51 修复已进入代码基线，但只通过解析坐标与集成合同；P52 排除了厘米级 FK mismatch
作为零 arm contact 的当前解释。

R0009 明确要求下一轮：

1. 若重提 P51 物理 smoke，必须在结果前冻结 environment seed、fixed candidate
   bytes/hash、pair 数、功效、tool-target 时间窗、接触与安全守护；
2. 并行评审 P50 候选漏斗，但只能离线处理不可变 acquisition bytes，覆盖完整
   `task × observation latency {1,2} × action latency {1,2}`；
3. 只有 fixed-candidate primitive 收敛且 candidate coverage/identity 成立，才重新筛选
   P47-E1 或 P41 selector 对照；
4. 不在当前 24-source Replay 上恢复世界模型训练。

## 本轮主要瓶颈

`R0010-C01`：

> P51 已修复并解析证明了坐标语义，P52 也排除了 policy FK 与 plant tool site 的数值
> 错配，但尚无结果前冻结的闭环物理证据证明 fixed primitive 会使工具朝固定候选收敛。
> 同时，P41 的候选生成仍可能为空或指向错误家具。因此当前不能区分“primitive 即使面对
> 正确目标也无法物理收敛”与“primitive 可收敛但候选覆盖/身份错误”这两个正交瓶颈，
> 更不能把任一无训练 smoke 外推为家务能力。

本轮优先回答：

1. 能否用与候选生成解耦、结果前提交的 fixed-candidate paired physical experiment，
   因果验证 P51 坐标修复是否改善工具到目标的闭环收敛；
2. 能否用不改变正式候选输出的 report-only 漏斗，定位 P41 空集合和错误实体候选的首要
   损失阶段；
3. 两项诊断能否使用独立主变量和独立结论并行，而不把正确候选、坐标修复、phase、幅值、
   gripper 或安全变化捆绑；
4. 若 primitive 或候选覆盖任一失败，应停在哪个证据门，而不是继续启动 selector 对照、
   Replay 采集或世界模型训练。

## 当前入口、证据与资源

- 正式任务：
  - `tidy_living_room_3d/v1`
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
- P41/P51 纯合同：`src/hwr/eval/target_selection.py`
- P41 MuJoCo bridge：
  `src/hwr/adapters/mujoco/target_selection_diagnostic.py`
- P41 入口：`src/hwr/apps/evaluate_target_selection.py`
- P40-E2 测量：
  - `src/hwr/adapters/mujoco/entity_contact_graph.py`
  - `src/hwr/apps/evaluate_entity_contact_graph.py`
- P51 正式解析产物：
  `runs/research-loop/0009/r0009-p51-cartesian-frame-s20265101`
- P52 正式运动学产物：
  `runs/research-loop/0009/r0009-p52-tool-kinematics-s20265201`
- P41 smoke：
  `runs/research-loop/0008/r0008-p41-target-selection-e2-smoke-s20264101`
  - 6 个 planned pair；
  - candidate 数依次为 `4/0/1/3/5/3`；
  - `selector_comparison_executed=false`；
  - 全部 hard safety guard 通过。
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 训练入口：
  `scripts/start_foundation_training_tmux.sh <RUN_ID> [--resume] [--seed <SEED>]`
- 本机：
  - Apple M5 Pro，18 核 CPU、20 核 GPU、48GB 内存；
  - 数据卷可用空间约 91GiB；
  - 沙箱内 MPS 可用性以本轮实测为准，正式 MPS 任务必须在 host-exec 环境重新核验。

## 本轮组织

- 创新 Agent A：因果控制与物理交互方向，只读。
- 创新 Agent B：感知覆盖、数据效率与泛化方向，只读。
- 创新 Agent C：评测有效性、安全与闭环物理方向，只读。
- 两名筛选 Agent 将对同一冻结提案集独立评分，完成前互不查看结果。
- 入选候选由唯一实施 Agent 负责，文件所有权与停止条件在 `03-experiment.md` 结果前冻结。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 新观点使用稳定 ID；历史观点重提必须保留原 ID 并增加修订号。
3. 一个候选只改变一个主变量；测量诊断、行为修订、数据采集与能力训练分开归因。
4. 不修改 `docs/research-loop/0001/`～`0009/`。
5. 不后验修改 R0008 P41-E2 或 R0009 P51/P52 的冻结门与历史判定。
6. 不延长 100ms safety validity，不移除 latency 3，不读取 future/latest observation，
   不降低碰撞、安全、deployment 或 action-causality 门槛。
7. policy/candidate 禁止使用 task/object/target ID、语言语义、颜色标签、geom/body/site
   ID、仿真物体位姿、reward、stage、success、contact truth、专家 waypoint/action。
8. evaluator-private 真值只允许用于 report-only 标签和接受判定，不能进入动作、候选或
   selector。
9. 样本单位是独立 Episode/seed 或结果前定义的 paired block；不得把 frame、anchor、
   raw candidate、transition、control step 或 branch 冒充独立样本。
10. 不按结果挑 task、seed、latency、candidate、frame、camera、mask、threshold、统计
    方法、MDE 或接受门。
11. 无训练候选覆盖、fixed-candidate smoke、离线诊断和 report-only safety 结果不得称为
    学习、泛化、安全能力或闭环任务成功。
12. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
13. 能力结论只能来自冻结未见分布上的闭环物理结果。
14. 正式长时训练必须经 `traex-host-exec` 与 tmux 启动，并配置带“阅读 AGENTS.md”的完成
    唤起和定时看门狗；小型验证不得休眠。
