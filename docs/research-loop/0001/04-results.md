# R0001 实验结果

## `R0001-P01` 第一次启动：`inconclusive`

### 运行身份

- run ID：`r0001-p01-baseline-s20260812`
- 源码提交：`f45e9f7b986a3469f317794f2713a8a2192ca921`
- 开始时间：2026-08-15 07:55:52 +08:00
- 停止时间：2026-08-15 11:01:46 +08:00
- 退出状态：143，主 Agent 通过看门狗判断无价值后主动发送 `SIGTERM`
- 运行目录：`runs/foundation-world-model/r0001-p01-baseline-s20260812`
- 监督日志：`logs/research-loop/0001/p01-supervisor.raw.log`
- 看门狗快照：`logs/research-loop/0001/p01-watchdog-latest.log`

该运行没有完成第一个训练 Episode、梯度更新或 checkpoint，不是能力基线，不得恢复或进入
后续评测。

### 已完成证据

- 当前提交开发总门禁全部通过；
- 三个正式任务的系统辨识留出均完成 8 Episode；
- 合计 24 Episode、3,072 transition；
- 每个 Episode 保留 128 transition；
- 运行目录停止时约 655 MiB；
- 未发生 OOM、非有限值、磁盘不足或进程异常；
- 训练进程、周期看门狗和防休眠均由独立 `tmux` 会话持有。

### 看门狗判定

Tick 1 发现 `metrics/latest.json` 未变化，但系统辨识 manifest 从 0 增长到 15 shard，因此
继续运行并升级看门狗，同时追踪 holdout/replay manifest。

Tick 2 后确认：

- 系统辨识已经完成 24/24 Episode；
- 之后约 80 分钟没有产生动作执行留出的第一个正例；
- 进程仍持续消耗约一个 CPU 核和 25%～35% GPU，调用栈主要在 MuJoCo 四相机渲染；
- 当前 collector 只有完整 Episode 命中目标类别后才落盘，单次 attempt 最多 6,000～8,000
  步，每个正例槽位最多 64 次 attempt；
- 继续等待会重复消耗长 Episode，但不能增加可用训练证据。

因此主 Agent按预注册规则停止同一 run，没有启动重复 run。

## 平台缺陷归因

### `R0001-F01`：正式任务预测安全未启用

`MujocoFormalHouseholdDualArmBackend` 继承的默认实现返回
`_predictive_safety_enabled() == False`。随机探索动作已经位于静态动作边界内，帧有效期也正常，
所以动作执行留出要求的硬安全干预正例极难或不可达。

这不是模型能力失败，而是启动留出与正式运行时安全合同不一致。

### `R0001-F02`：正例即使命中也可能被尾部截断丢失

动作执行正例按“Episode 是否出现过安全干预”筛选，但落盘只保留最后 16 transition。若安全
干预发生在长 Episode 早期，保存的终端窗口可能全是负标签，和后续逐 transition 验证合同
矛盾。

### `R0001-F03`：无需搜索的留出仍运行完整任务 horizon

系统辨识仅需 128 transition，动作执行和碰撞负例仅需 16 transition，但旧 collector
仍运行完整 6,000～8,000 步任务 Episode。第一次启动仅系统辨识就耗时约 105 分钟。

### `R0001-F04`：非安全 plant rewrite 被错误当作 identity

正式环境在硬安全层前还施加 actuator scale 和 0/1 步动作延迟。动作执行头的目标是实际执行
动作，但旧验证在非安全 transition 上比较“预测动作与 Actor 原提议”，会把正确预测 plant
rewrite 的模型判错。

### `R0001-F05`：正式严重碰撞没有统一终止合同

碰撞留出筛选依赖 `result_reason == "severe_collision"`，但正式后端只累计
`severe_collision_count`，没有立即发布相同终止结果。即使发生真实严重碰撞，正例也可能无法
进入后续终端窗口验证。

## 平台修复

修复不降低任何正例数量、误差阈值、碰撞阈值或安全约束：

1. 正式三任务启用与既有 bimanual 后端一致的两步预测安全层；
2. 继续使用 220 N 严重碰撞阈值；
3. 若预测层漏掉并实际发生严重碰撞，正式 Episode 立即以 `severe_collision` 失败终止；
4. 专用正例留出在达到最小 16 transition 且命中证据后立即停止，确保终端窗口保留事件；
5. 系统辨识精确采集 128 transition，负例精确采集 16 transition；
6. 碰撞分类同时核验真实物理 interaction audit；
7. 非安全样本的执行动作误差改为比较预测动作与实际 plant action，同时单独报告观测到的
   plant rewrite 幅度。

### 低成本物理验证

使用动作执行留出的真实 seed、冻结随机源、完全相同 MuJoCo 物理和安全阈值，仅在诊断中
关闭相机渲染以降低验证成本：

| 任务 | 首个预测安全正例 |
|---|---:|
| `clear_dining_table_3d/v1` | attempt 2，第 4,356 步 |
| `store_kitchen_items_3d/v1` | attempt 1，第 3,208 步 |
| `tidy_living_room_3d/v1` | attempt 3，第 3,044 步 |

三任务均证明正例在预注册 64 attempt 预算内可达。该诊断不写正式数据，不构成能力结果。
餐桌和厨房的更早 attempt 分别产生真实严重碰撞，并按修复后的安全合同立即以
`severe_collision` 失败终止；它们不被混作安全干预正例。

### 已通过验证

- 正式后端预测安全行为测试；
- 实际严重碰撞终止测试；
- 专用正例命中后最小窗口保留测试；
- 系统辨识与负例有界采集测试；
- 物理审计碰撞分类测试；
- 非安全 plant rewrite 执行动作误差测试；
- foundation online、loading、collision、action execution 相关回归；
- 全量 `pytest`；
- `scripts/verify_training_semantics.py`；
- Python 文件/函数尺寸检查；
- 架构边界检查。

## 重新冻结

- 第一次 run 保留为 `inconclusive` 启动失败证据，不恢复、不删除。
- 修复提交通过完整项目门禁后，使用新 run ID：
  `r0001-p01-baseline-v2-s20260812`。
- v2 仍是无行为能力改动的当前谱系基线；平台修复和能力候选保持隔离。
- v2 使用相同训练 seed、任务、模型、更新预算、准入门槛和最终评测。
- v2 继续使用完成即时唤起、训练启动后 15 分钟首次检查、之后每 60 分钟唤起的
  `tmux` 看门狗。

## 当前状态

`R0001-P01` 仍为 `inconclusive`，等待 v2 总门禁和正式运行结果。

## `R0001-P01` 第二次启动：`inconclusive`

### 运行身份

- run ID：`r0001-p01-baseline-v2-s20260812`
- 源码提交：`be7ad047c2f4a1577636ade134ad5ae55e17fa9d`
- 开始时间：2026-08-15 12:17:36 +08:00
- 停止时间：2026-08-15 13:19:10 +08:00
- 退出状态：143，主 Agent 主动停止
- 运行目录：`runs/foundation-world-model/r0001-p01-baseline-v2-s20260812`

v2 开发总门禁通过。修复后的有界系统辨识在约 3 分钟内完成三个任务共 24 Episode、
3,072 transition，相比 v1 约 105 分钟的启动阶段显著消除了无效物理步。

但随后动作执行留出仍未产生第一个正例。Tick 1 后继续观察约 35 分钟，进程持续执行
MuJoCo 物理和渲染、无异常，但 holdout manifest 仍停留在系统辨识的 24 shard。

### `R0001-F06`：系统辨识激励误用于所有留出阶段

`collect_causality_holdout` 无条件按 Episode 循环
`SYSTEM_IDENTIFICATION_CORRELATIONS=(0.0, 0.5, 0.9, 0.96)`。动作执行正例槽 0 因此使用
`ρ=0.0`，而前一轮可达性验证和正式训练随机源使用 `ρ=0.96`。

只读、无渲染反例验证使用完全相同的正式环境、动作执行 slot 0 seed 与 `ρ=0.0`：

- 餐桌任务前 16 个预注册 attempt；
- 每个 attempt 运行完整 6,000 步；
- 全部以 `formal_household_timeout` 结束；
- 没有一次 `predicted_severe_collision`。

这与 v2 正式运行约 60 分钟无第一个动作执行正例一致。继续到每槽 64 attempt 只会增加
无效计算，主 Agent 因此停止 v2。v2 同样没有训练 Episode、update 或 checkpoint，不恢复。

## 第二次平台修复

- 四档相关系数只用于 `system_identification`；
- `action_execution_validation` 与 `collision_validation` 使用冻结正式随机探索参数
  `motion_correlation=0.96`；
- holdout provenance 升级为 `foundation-causality-holdout/v8`；
- metadata 改为通用 `holdout_excitation` 并显式记录 phase；
- 增加 phase 隔离回归，禁止系统辨识激励泄漏到其他留出。

前一轮真实 `ρ=0.96` 诊断已经证明三任务在 attempt 2/1/3 内分别产生预测安全正例，显著低于
64 attempt 上限。

## v3 重新冻结

- v1、v2 均保留为 `inconclusive` 平台失败证据；
- v3 run ID：`r0001-p01-baseline-v3-s20260812`；
- v3 使用相同训练 seed、模型、任务、预算、准入和最终评测；
- v3 继续由 detached `tmux` 监督、定时看门狗和防休眠会话持有；
- v3 通过总门禁前不启动正式训练。
