# R0005 独立评审

## 评审过程

- 两名筛选 Agent 在同一冻结 `00-context.md` 与 `01-proposals.md` 上独立只读评审。
- 两者均未修改文件、未启动训练，完成前未看到对方评分。
- 两者可只读核查历史与当前代码，但不得改变提案或门槛。

## 评审合同

两名筛选 Agent 必须在互不查看对方结果的情况下，对每项候选分别按 1–5 分评分：

1. 目标价值；
2. 证据强度；
3. 可检验性；
4. 因果可归因性；
5. 通用性；
6. 实施成本；
7. 回归风险。

其中实施成本 5 表示低成本，回归风险 5 表示低风险。

## 强制拒绝条件

- 没有明确指标或无法证伪；
- 与历史已拒绝或已完成实验实质重复而没有新证据；
- 依赖评测泄露、结果后选或降低安全约束；
- 首次实验同时改变多个不可独立归因的主变量；
- 只优化代理量，没有预注册的未见分布闭环物理验证路径；
- runtime/评测修复与能力改进捆绑；
- 对 latency=3 的处理等价于删除该域、使用 future/latest frame 或延长 validity；
- 使用 action shuffle/derangement 形状训练或选择候选。

## 主 Agent 决策

分数顺序统一为：

`目标价值 / 证据强度 / 可检验性 / 因果可归因性 / 通用性 / 实施成本 / 回归风险`

其中实施成本 5 表示低成本，回归风险 5 表示低风险。

## S1 独立评分

| ID | 分数 | verdict |
|---|---|---|
| `R0001-P31` | `5/4/4/3/4/3/4` | `changes_required` |
| `R0001-P32` | `4/4/5/4/4/5/4` | `changes_required` |
| `R0001-P33` | `5/2/4/3/4/1/2` | `defer` |
| `R0001-P34` | `5/2/3/2/3/2/1` | `reject` |
| `R0001-P35` | `4/2/4/3/3/1/2` | `defer` |
| `R0001-P36` | `5/5/5/5/4/5/3` | `approve` |
| `R0001-P37` | `4/4/4/4/3/2/2` | `defer` |
| `R0001-P38` | `4/5/5/4/5/5/3` | `changes_required` |

### S1 关键反驳

- P31：fresh probe 若在专门配对干预数据上训练，而 production prior 从普通 Replay
  训练，则差异混入数据分布，不能直接归因为 objective。
- P32：actual action residual 可能编码遗漏的策略内部状态、collector RNG、时间位置或
  safety rewrite；必须 nested source cross-fit，并把结论限制为条件预测信息。
- P33：可能只学会 P17 局部随机方向族；P31/P32 未形成缺口前没有训练依据。
- P34：新的 100ms command lease 仍基于 150ms old observation。可信 timestamp 和单调
  sequence 只能证明“这确实是一帧旧图”，不能证明动态环境中执行它是安全的。P34 实质把
  source-age 边界从 100ms 放宽到 150ms，触发安全强制拒绝。
- P35：observation latency 与 plant action latency 会叠加；固定四槽按 observation offset
  选 slot3 未必对应实际 plant apply tick。
- P36：最大风险是只展示 supported conditional rate，隐藏 complete challenge failure；
  因此完整挑战账本必须是首要总账。
- P37：`min(left,right)` 可被单臂接近利用，且 privileged geometry shaping 可能产生
  reward hacking。
- P38：P17 只能证明 plant controllability，不能替代普通 Replay conditional
  identifiability；本轮只允许 shadow/report-only。

S1 建议依赖顺序：

`P36 -> 新评测基线 -> P32 -> 修订 P31 -> P38 shadow -> 重新筛选 P33`

## S2 独立评分

| ID | 分数 | verdict |
|---|---|---|
| `R0001-P31` | `5/4/4/3/4/3/4` | `changes_required` |
| `R0001-P32` | `4/3/4/3/3/5/4` | `changes_required` |
| `R0001-P33` | `5/2/4/3/4/1/2` | `defer` |
| `R0001-P34` | `5/3/3/2/3/2/1` | `reject` |
| `R0001-P35` | `4/2/3/3/3/1/2` | `defer` |
| `R0001-P36` | `5/5/5/5/4/5/4` | `approve` |
| `R0001-P37` | `2/1/3/4/2/2/1` | `reject` |
| `R0001-P38` | `4/5/5/4/4/5/3` | `changes_required` |

### S2 关键反驳

- P31：同 snapshot 分支必须同 fold；若配对动作 support 相对普通 Replay 明显 OOD，
  只能形成 paired-data 可学习性结论。
- P32：nuisance action predictor 欠拟合或过强都会改变 residual 结论；需要严格
  source cross-fit、欠拟合敏感性和可重建 manifest。
- P33：即使 paired alignment 改善，也可能牺牲视觉与长期任务表征；baseline 必须获得
  相同 paired data，正式比较只能改变 loss。
- P34：一个持续更新 sequence、但固定落后 150ms 的故障流与正式 latency=3 在当前
  provenance 上不可区分；把 lease 绑定 scheduler 时钟不会恢复丢失的环境状态。
- P35：slot3 是旧观测下的未来行为克隆标签，不自动等于当前真实状态应执行动作；且
  horizon=4 checkpoint 与 slot selection 必须拆开。
- P36：域划分必须由 runtime 可核验 source age 与冻结合同得到；支持域和完整域要同屏、
  同版本发布，不能跨评测合同比较旧数值。
- P37：代码核查发现当前 `BimanualTaskTracker` 已有 left/right reach、worst-side reach、
  bilateral reach occupancy、near-handle closure、joint grasp readiness 和 bilateral
  contact shaping。P37 的核心“前接触奖励平坦”前提不成立，且 `min(left,right)` 比现有
  双臂约束更弱，触发重复与捷径强制拒绝。
- P38：应先修复报告语义，不改变 unlock；未来实际硬门变更依赖 P32 或等价证据，并需
  新候选。

S2 建议依赖顺序：

`P36 -> 可信评测基线 -> P32 -> 修订 P31 -> 重新筛选 P33`

P38 可并行做 report-only shadow；P34/P37 拒绝，P35 随 P34 延后。

## 主 Agent 代码复核

### P34 安全边界

- `dual_arm_action_frame()` 与 evaluator `_action_frame()` 均把 100ms action validity
  绑定 visible observation timestamp。
- backend latency queue 返回保留原 timestamp 的旧 observation；latency=3 稳态 age
  为 150ms。
- safety 在当前 runtime timestamp 上判断有效期。
- P34 即使保留 100ms command lease，也会允许原合同拒绝的 100～150ms source age
  驱动运动；没有独立实时感知或 reachable-set safety 证明。
- 因此 P34 不是单纯时钟修复，而是 safety threat model 变更，本轮拒绝。

### P37 历史重复

当前 `src/hwr/tasks/bimanual.py` 已包含：

- left/right reach distance；
- worst-side reach；
- bilateral reach occupancy；
- near-handle closure；
- joint grasp readiness；
- 单侧与双侧 contact；
- severe-collision penalty 与双臂并发成功条件。

现有测试也验证平衡双臂接近优于单臂接近。P37 未给出这些信号失效的新证据，并提议更弱
的 `min(left,right)`，正式拒绝。

## 主 Agent 非机械决策

### `R0001-P36`：唯一立即入选

- 两名筛选者一致批准。
- 它直接解决 `R0005-C01` 的结果归因，不改变 runtime、policy、模型、安全或任务。
- 先执行最低成本的 E1：冻结双账本合同、实现纯聚合器，并重放 P11/P29 历史证据。
- E1 不更改现有正式 benchmark seed schedule 或 acceptance；若通过，再独立冻结未来
  factorial benchmark integration，避免一次实现引入未冻结预算和统计自由度。

### `R0001-P32`：修订后第二候选

- 先冻结 executed action、nested source-Episode cross-fit、nuisance model、residual
  standardization、有效秩、统计单位和 null/planted power。
- 结论只允许为“给定冻结 state representation 的普通 Replay 条件动作信息”。
- P36-E1 未完成前不执行。

### `R0001-P31`：排在 P32 后

- 必须消除 fresh probe 与 production prior 的训练数据分布混淆。
- P32 阴性不机械否决 P31，但会降低采集新 paired bank 的优先级。
- P31 通过最多授权重新冻结 P33。

### `R0001-P38`：仅保留 shadow/report 草案

- 当前不实施硬门变化。
- 不把 P17 当作普通 Replay identifiability，不删除旧原始数值。
- 未来行为变化必须使用新稳定 ID 独立筛选。

### 拒绝或延后

| ID | 决策 | 原因 |
|---|---|---|
| `R0001-P33` | `defer` | 依赖 P31，正式训练成本高 |
| `R0001-P34` | `reject` | 实质放宽 source-age safety，且多变量 |
| `R0001-P35` | `defer` | 缺安全 runtime 前置，horizon 与总 latency 未冻结 |
| `R0001-P37` | `reject` | 与现有更强双臂 shaping 重复，并引入单臂捷径 |
| `R0001-P38` | `changes_required` | 只允许 report-only，不改变 unlock |

本轮不得把 P36 与任何训练候选首次同时上线，不得运行 P34+P35，也不得捆绑 P33+P37。
