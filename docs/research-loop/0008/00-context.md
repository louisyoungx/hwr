# R0008 研究上下文

## 轮次身份

- 轮次：`R0008`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`4c4efda16759577cb05098a7628f29d3bfbef890`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-21`
- 前一轮：`R0007`
- 本轮能力基线：不变

本轮文档只写入 `docs/research-loop/0008/`。只有经本轮独立筛选和结果前冻结后，才允许
修改明确归属的新实现、测试和实验配置。下列历史目录冻结只读：

| 目录 | 起始提交下的 Git tree |
|---|---|
| `docs/research-loop/0001/` | `416912b7dc1c19611bcfc4375028180014a1989b` |
| `docs/research-loop/0002/` | `6fb603dbd52451fe1749157daf05aa482ca7222f` |
| `docs/research-loop/0003/` | `f56011eda321ea803bc24051db001e632c1549fb` |
| `docs/research-loop/0004/` | `611c420e539a53a8c7578cd66aa8bdfe46fe82b7` |
| `docs/research-loop/0005/` | `0352d379d5754adb03e9158c0fa72393ab322d58` |
| `docs/research-loop/0006/` | `ee3a6f5b25887f67f812750d2a75424df12823d4` |
| `docs/research-loop/0007/` | `0a696caa153abc9c13403fbc9bd3c081ce71c327` |

## 可信基线

- 当前没有通过因果与未见分布闭环门禁的 deployment，不宣称机器人已学会家务。
- 最新完整 3D 世界模型负基线仍为
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`：
  - 24 Episode；
  - 1,600 update；
  - 0 成功；
  - action causality aggregate ratio `1.0012791539504238`，低于冻结门槛 `1.05`；
  - Actor 从未解锁，task/exploration Actor update 均为 0；
  - 三任务 interaction coverage 均没有 bilateral contact 或 controlled motion。
- `R0001-P17` 已证明实际 plant action 在三个正式任务和 `1/4/8/16` 步 horizon 上具有
  稳定物理因果效应；环境并非完全不响应动作。
- `R0001-P29` 已接受为 50ms 控制周期、100ms visible-source-age validity 和
  observation latency 1/2 支持域、latency 3 挑战域的 runtime 合同证据。
- `R0001-P36-E1`、`R0001-P39-E1` 与 `R0001-P36-E2` 分别建立支持域分账、标准 policy
  seed 隔离和平衡 factorial benchmark 合同，但没有产生能力数值。
- R0007 的 `R0001-P40-E1` 已接受为 report-only safety measurement contract evidence：
  allowed contact 已按 floor/support、manipulated object、target container 和 articulation
  分账，且测量开关不改变 runtime 行为。
- R0007 的 `R0001-P32-E1` 为 `inconclusive`：
  - state-only 条件下 rate target aggregate ratio 为 `1.1114809121271965`；
  - 加入 controller history 后 ratio 反转为 `0.9069818836808229`；
  - exact-pipeline 10% planted power 仅 `1/200`；
  - 现有 24 个独立 source 不足以支持接受或拒绝；
  - 不授权 P31、P43、P33、Actor 解锁或任何训练。

## 上一轮结论与未决依赖

- 不得继续在当前 24-source Replay 上通过改门槛、删守护或挑 target 重跑 P32。
- 若重提普通 Replay 条件信息诊断，必须先增加 task-balanced 独立 source 或降低设计方差，
  保留 controller-history 与 configuration-target 守护，并重新冻结 exact-pipeline MDE。
- `R0001-P41-E1` 上轮为 `changes_required`：
  - 只能保留 target-index 选择作为唯一主变量；
  - candidate/control 必须从相同 policy-visible RGB-D candidate set 选择；
  - primitive、动作幅值、双臂耦合和提前终止处理必须固定；
  - 必须先冻结 paired power 与 MDE；
  - P40 测量前置现已具备，但不能把 220N legacy forbidden threshold 当成 allowed-contact
    硬件阈值。
- `R0001-P43` 仍缺每任务足够的独立 safety-positive source，不得用 transition 重采样或
  多优化 seed 冒充独立数据。
- `R0001-P42` 与 `R0001-P44` 依赖未来 qualified deployment。
- `R0001-P46` 可作为训练可靠性侧车重新筛选，但不能挤占当前交互支持瓶颈。

## 本轮主要瓶颈

`R0008-C01`：

> 当前正式三维数据在三个任务上都没有 bilateral contact 或 controlled motion，普通
> Replay 的 apparent action information 又被 controller history 解释，且只有 24 个独立
> source，导致继续改世界模型目标既缺乏可识别监督，也无法以足够功效归因。R0008 必须优先
> 找到一个单主变量、policy-visible、无任务脚本捷径的低成本物理交互干预，或建立增加独立
> source 所需的结果前合同；在此之前不启动昂贵世界模型训练。

本轮优先回答：

1. 能否把 P41 修订为只改变 policy-visible RGB-D candidate 中的 target index，而完全固定
   后续控制 primitive、幅值、双臂耦合、时长和提前终止；
2. 能否在 paired 未见物理随机化下，以 object-category contact、受控位移、安全和
   report-only contact ledger 建立不依赖任务成功的最小可证伪门；
3. 若 P41 仍不可归因，是否应先建立 task-balanced 独立普通 source 的采集与功效合同，而非
   在现有 24-source Replay 上继续离线挖掘；
4. 可靠性侧车是否能在不改变学习行为的前提下保护未来正式训练，但不得取代主要能力瓶颈。

## 当前入口、证据与资源

- 主要正式任务：
  - `tidy_living_room_3d/v1`
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
- 3D backend 与 binding：
  - `src/hwr/adapters/mujoco/formal_household_backend.py`
  - `configs/adapters/mujoco/formal_3d_v1.json`
- P40 contact ledger：
  - `src/hwr/adapters/mujoco/contact_ledger.py`
  - `src/hwr/apps/evaluate_contact_ledger.py`
  - `runs/research-loop/0007/r0007-p40-contact-ledger-e1-s20264001`
- 当前普通 Replay：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/replay/autonomous`
  - 24 source Episode；
  - 168 shard；
  - 2,688 transition；
  - task/source 为餐桌 6、厨房 6、客厅 12。
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 启动时数据卷可用空间约 87 GiB。
- 当前沙箱：
  - Python 3.11.0；
  - PyTorch 2.13.0；
  - `torch.backends.mps.is_built() == True`；
  - `torch.backends.mps.is_available() == False`。
- 正式 MPS 运行必须在 host-exec 环境重新核验加速器与进程独占状态。

## 本轮组织

- 创新 Agent A：学习信号、动作因果路由和双臂交互控制。
- 创新 Agent B：评测有效性、三维泛化和物理安全。
- 创新 Agent C：数据效率、计算效率、训练可靠性与采集支持。
- 两名筛选 Agent 将在同一冻结提案集上独立评分，完成前互不查看结果。
- 每个入选候选由唯一实施 Agent 负责，文件所有权在 `03-experiment.md` 中冻结。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 所有新观点使用稳定 ID，并在提案、实现、提交、运行和结果间保持引用。
3. 一个候选只改变一个主变量；评测/runtime 修复、数据采集、无训练诊断与能力训练分离。
4. 不修改 `docs/research-loop/0001/`～`0007/`。
5. 不延长 100ms safety validity，不移除 latency 3，不读取 future/latest observation，
   不降低碰撞、安全、deployment 或 action-causality 门槛。
6. 不允许 privileged object pose、task stage、scripted waypoint、专家动作或 task-specific
   target ID 进入 candidate；candidate 与 control 只能使用完全相同的 policy-visible 输入。
7. 普通 Replay 的统计独立单位是 source Episode；不得用 shard、transition 或优化 seed
   冒充独立 source。
8. 不按结果挑 seed、任务、对象、target、相机、mask、统计方法、MDE 或门槛。
9. 任何无训练交互 smoke、离线诊断或 report-only safety 结果都不得称为学习、泛化、
   安全能力或闭环任务成功。
10. 所有行为变化必须有测试；正式运行只从干净、已提交且通过相关门禁的提交启动。
11. 能力结论只能来自冻结未见分布上的闭环物理结果。
12. 正式长时训练必须经 `traex-host-exec` 与 tmux 启动，并设置带“阅读 AGENTS.md”的
    完成唤起和定时看门狗；小型验证不得休眠。
