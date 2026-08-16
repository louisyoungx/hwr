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

## `R0001-P01` 第三次启动：`inconclusive`

### 运行身份

- run ID：`r0001-p01-baseline-v3-s20260812`
- 源码提交：`a13206a53d7e2c0cdfdd01a6fd5798e2bd273f6f`
- 开始时间：2026-08-15 14:08:45 +08:00
- 停止时间：2026-08-15 15:28:34 +08:00
- 退出状态：143，主 Agent 在资源效率审计后主动停止
- 无训练 Episode、update 或 checkpoint，不恢复

v3 证明 `R0001-F06` 修复有效：

- 系统辨识完成 24 Episode；
- 动作执行留出完成餐桌任务 8 正 + 8 负 Episode；
- 厨房任务完成 2 个正例；
- 正例最大 attempt 为 4；
- 所有正例末尾窗口均保留 `safety_intervention_evidence`。

### 资源效率审计

多点采样而非单点观察显示：

- 18 核 CPU 中训练进程约使用一个核心，约占总 CPU 容量 5%；
- GPU 利用率稳定约 27%～29%；
- RSS 约 6 GiB，系统内存余量约 85%；
- 当前阶段是单环境串行 holdout 采集，不是梯度训练；
- 每一步都渲染头部 RGB、深度、左右腕部四路相机；
- 动作执行正例平均约 3.8 分钟/个，部分正例需 5～10 分钟；
- 剩余启动留出预计仍需约 2 小时以上。

继续运行虽能产生有效 shard，但明显没有充分利用本机资源。主 Agent 因此停止 v3，单独优化
可重建的启动留出采集；该决定不改变能力算法、seed、动作、任务、阈值或模型。

## 吞吐优化筛选

### 三任务线程并行：`rejected`

曾实现每任务一个线程、共享 store 加锁原子写入。正式 256×192 四相机 smoke 显示三个
MuJoCo renderer 在 macOS 上争用全局 `CGL` context mutex：

- 16 步三任务 smoke 运行 272.57 秒仍未完成；
- 三个 worker 栈均阻塞在 `CGLLockContext`；
- GPU 利用率约 1%；
- 仅发生 2 次 voluntary context switch。

该方案比串行更差，已经完全撤回，不进入提交。

### 两遍确定性采集：`accepted` 作为平台优化

动作执行/碰撞正例改为：

1. 搜索 pass 使用完全相同的 MuJoCo 物理、seed、随机动作和安全层，但复用首帧
   payload，不渲染不会保存的中间相机；
2. 命中后从同一 seed 重置，按相同动作过程确定性重放；
3. 在最终保留窗口前 4 步恢复正式相机渲染，以覆盖最多 1 步观测延迟；
4. 只保留最终 16 transition 的环形缓冲；
5. 搜索和重放的命中步数、类别必须一致，否则实验立即失败。

餐桌任务真实正式配置 smoke：

- seed：`920262896`；
- 搜索与重放均在第 4,356 步命中；
- 末尾 16 transition 中最后一个标签为安全干预正例；
- 正式四相机输出保留；
- 墙钟从 v3 同类正例约 9 分钟降到 68.15 秒；
- collector 主体耗时 53.47 秒；
- 峰值 RSS 约 0.8 GiB。

该优化约带来 8 倍墙钟加速，不改变最终保存数据的 seed、动作轨迹、物理、安全标签、任务或
门槛。

## v4 重新冻结

- v1～v3 均保留为 `inconclusive` 平台证据，不恢复、不删除；
- v4 run ID：`r0001-p01-baseline-v4-s20260812`；
- holdout collector provenance 升级到 v10；
- autonomous collector provenance 升级到 v4；
- v4 使用相同训练 seed、模型、任务、预算、准入和最终评测；
- v4 由 detached `tmux` 监督、定时看门狗和防休眠会话持有。

## `R0001-P01` 第四次启动：有效负基线，`rejected`

### 运行身份与退出归因

- run ID：`r0001-p01-baseline-v4-s20260812`
- 源码提交：`d6d9a43b187591c43262efe1938130b358d63799`
- 正式训练开始时间：2026-08-15 16:25:00 +08:00
- 停止时间：2026-08-15 20:04:53 +08:00
- 最终规模：24 Episode、1,600 update、约 4.7 GiB
- 最终 checkpoint：`checkpoints/update-000001600`
- 最新指针：`latest.json` 已发布并指向 update 1,600
- 运行目录：`runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`
- 监督日志：`logs/research-loop/0001/p01-v4-supervisor.raw.log`
- 最后看门狗快照：`logs/research-loop/0001/p01-v4-watchdog-latest.log`

开发总门禁以状态 0 通过，留出、Replay、指标、因果报告、恢复快照和最终 checkpoint 均完整，
源码提交与运行 manifest 一致。训练进程随后以状态 1 退出，错误为：

```text
foundation calibration stopped early after missing evidence:
one_step_physical_action_utilization,
data_action_probe_bootstrap_lower_bound,
data_action_probe_all_tasks,
action_execution_model_validation
```

该退出发生在冻结的 24 Episode/1,600 update 判定点，符合 `03-experiment.md` 的“未通过”
合同。因此它不是开发门禁失败、OOM、进程异常或 checkpoint 失败，而是具有完整负证据的
校准停止。监督器、周期看门狗和防休眠进程均已退出；不得恢复或重复启动此 run。

### 数据动作可辨识性

整体 action probe 点估计为 `1.0647199774736131`，高于 `1.05`，但整体保守
bootstrap `p05` 只有 `0.9177844030581959`，且冻结合同要求每个任务、每个 horizon 均通过，
不能用整体均值掩盖局部失败。

| 任务 | 聚合 ratio | 聚合 `p05` | 失败位置 |
|---|---:|---:|---|
| `clear_dining_table_3d/v1` | 1.079737 | 1.007304 | 聚合 `p05 < 1.01` |
| `store_kitchen_items_3d/v1` | 1.002280 | 0.962667 | 1-step ratio 与 `p05` |
| `tidy_living_room_3d/v1` | 1.023931 | 0.917784 | 16-step ratio 与 `p05` |

餐桌、厨房、客厅分别有 6、6、12 个训练 source Episode。最弱任务并不是唯一的样本最少
任务：厨房在 1-step 失败，而样本更多的客厅在 16-step 失败，因此当前证据不支持将
`R0001-P03` 的任务配额作为首选解释。三任务的 4-step 和 8-step 均明显过线，失败更符合
固定高相关随机动作在短时与长时尺度上提供的激励不均衡。

### 世界模型动作利用

最终物理动作 shuffle 因果结果：

- aggregate ratio：`1.0012791539504238`，低于 `1.05`；
- worse-horizon fraction：`0.625`，达到 `0.60`；
- 本体 ratio：`1.0012907760687166`，虽然 horizon fraction 为 `0.625`，ratio 仍失败；
- 视觉潜变量 ratio：`0.9861666265301663`，horizon fraction 为 `0.0`；
- required components、aggregate 和 partitions 均未通过。

这说明打乱动作并未稳定恶化预测，世界模型基本没有利用物理动作因果。该结论来自冻结物理
留出，不以 loss 下降或训练回报替代。

### 动作执行模型

三个任务均有 8 个正例和 8 个负例，验证数据数量完整，但全部失败：

| 任务 | recall | PR-AUC | 干预动作 RMSE | 非干预动作 RMSE |
|---|---:|---:|---:|---:|
| `clear_dining_table_3d/v1` | 0.000 | 0.1938 | 0.3266 | 0.1193 |
| `store_kitchen_items_3d/v1` | 0.000 | 0.1165 | 0.3573 | 0.1113 |
| `tidy_living_room_3d/v1` | 0.000 | 0.1599 | 0.3016 | 0.1126 |

冻结门槛分别为 recall `>= 0.8`、PR-AUC `>= 0.5`、干预 RMSE `<= 0.15`、非干预
RMSE `<= 0.05`。所有任务的 recall 均为 0，且误差显著超限；该失败不能归因于正负例缺失。

### 覆盖、准入与产物

- readiness 窗口活跃动作维度比例为 `1.0`，有效秩为 `15.3401`；
- 全 run 动作覆盖有效秩为 `15.6340`；
- Replay Episode 数为 24，最低 Replay 数量检查通过；
- 动作覆盖不是当前主要瓶颈；
- exploration readiness 连续通过次数为 0；
- 探索 Actor 未解锁、未 warm-up、update count 为 0；
- 没有发布 causality-qualified deployment；
- 由于校准未通过，没有进入 120 Episode，也没有运行未见分布闭环能力评测。

因此本结果只能建立当前谱系的有效负基线，不能宣称任何家务能力、泛化或安全性改善。

### 资源与停止决策

v4 的两遍确定性留出采集完成全部 72 个留出 shard，正式训练阶段 MPS 采样曾达到约
65%～100%，没有复现 v3 的长期单核、低 GPU 无效渲染问题。最终周期记录的主要墙钟为：

- collection：495.73 秒；
- update：758.54 秒；
- evaluation：196.67 秒；
- materialization：44.92 秒；
- checkpoint：1.91 秒。

在预注册停止点之后继续训练既违反冻结预算，又没有探索 Actor 新数据来源，不能修复已经
观察到的跨任务/horizon 数据失败。按合同停止比盲目继续使用资源更有价值。

### 结论与下一候选路由

`R0001-P01` 已完成“建立可信当前谱系基线”的目的，但其 24 Episode 校准通过假设被否定，
本轮结论标记为有效负基线、`rejected`。v1～v3 仍为 `inconclusive` 平台证据，只有 v4
可作为后续候选的冻结比较基线。

按预注册条件路由：

1. 厨房 1-step 与客厅 16-step 的 probe 点估计失败，先进入 `R0001-P02`；
2. `P02` 只先做不训练模型的短物理配对验证，比较固定 `ρ=0.96` 与冻结多时间尺度
   `ρ=0.0/0.5/0.9/0.96`，保持任务、seed、物理步数、动作幅值、留出和安全层不变；
3. 只有短验证使最弱任务、最弱 horizon 过既有门槛，才允许另开 run 做 24 Episode
   正式校准对照；
4. 整体 probe 已高于 `1.05`，但世界模型 ratio 仍约为 1，因此将 `R0001-P05` 保留为
   后续分支；只有数据 probe 全部通过后才能验证跨 source Episode mini-batch；
5. `R0001-P04` 可对同一冻结数据独立执行 bootstrap 统计修复，但不得与 `P02` 的行为改动
   捆绑，也不得把统计差异计为能力提升；
6. 当前不选择 `P03/P06/P07/P08`，不在 v4 run 内切换任何条件。

## `R0001-P04`：同步 Episode bootstrap，`accepted as evaluation repair`

### 实现与验证

- 实现提交：`b665c9d96049d80e1951c6a8e941af4695d23d2a`
- action probe schema：`hwr.foundation-data-action-probe/v4`
- 同一任务的 `1/4/8/16` horizon 在每个 bootstrap replicate 中使用同一组 Episode
  multiplicity，再在 replicate 内取最弱 horizon。
- 相关测试：

```text
.venv/bin/python -m pytest \
  tests/test_foundation_actor_readiness.py \
  tests/test_foundation_online.py -q
16 passed
```

- Python 文件/函数尺寸检查通过。
- 新测试覆盖跨 horizon 相关、零效应、动作打乱、单 horizon 失败、固定 seed 重现性和
  多 seed 稳定性。

### v4 冻结数据旧/新双报告

产物：

- `diagnostics/action-probe-p04-v3.json`
- `diagnostics/action-probe-p04-v4.json`
- `diagnostics/action-probe-p04-comparison.json`

在同一 v4 Replay 和系统辨识留出上：

- 所有逐任务逐 horizon 点估计零差异；
- state-only / state-action MSE 零差异；
- 训练/留出 Episode 数、transition 数和 Episode ID 列表零差异；
- 只改变跨 horizon bootstrap replicate 的耦合方式。

| 分区 | v3 `p05` | v4 同步 `p05` |
|---|---:|---:|
| aggregate | 0.917784 | 0.911595 |
| 餐桌 | 1.007304 | 0.996411 |
| 厨房 | 0.962667 | 0.955427 |
| 客厅 | 0.917784 | 0.911595 |

同步合同在该数据上更保守，餐桌也从仅 bootstrap 角度由接近过线变为未过线；厨房和客厅的
点估计失败完全不变。该变化不能算能力回归或改善，只表示旧 v3 的独立 horizon 重采样不符合
原定义。

因此 `R0001-P04` 标记为 `accepted as evaluation repair`。后续新基线必须使用 v4 测量合同，
但 `P01` 的能力结论仍是 `rejected`。

## observation lag 离线强诊断

### 发现

二次创新审查发现正式后端存在两个独立延迟：

- plant action latency：训练 `0/1` 步；
- observation latency：训练 `0/1` 步。

后端返回的 `info["applied_action"]` 是当前物理步实际 plant action，但返回 observation 可能
来自更早的物理状态。Replay 未保存 observation lag，action probe 和 RSSM 均把当前物理步
动作与相邻可见 observation 直接配对。

主 Agent 对 v4 冻结 Replay 和系统辨识留出执行只读离线诊断：

1. 用不可变任务配置、task ID 和 seed 确定性重建每条 Episode 的 observation lag；
2. 不训练模型；
3. 只将用于 probe 的实际 plant action 窗口按 lag 回移；
4. 使用 `P04` v4 同步 bootstrap。

48 条训练与系统辨识源 Episode 中，18 条 lag=0，30 条 lag=1。

### 结果

| 任务 | 原最弱位置 | 原 ratio | 对齐后 ratio | 对齐后任务 `p05` |
|---|---|---:|---:|---:|
| 餐桌 | 1-step | 1.079737 | 1.251437 | 1.196090 |
| 厨房 | 1-step | 1.002280 | 1.331355 | 1.238924 |
| 客厅 | 16-step | 1.023931 | 10.805416 | 1.286383 |

对齐后所有任务的 `1/4/8/16` 点估计与同步 bootstrap 均过原门槛。尤其厨房 1-step
state-action MSE 从 `0.010234` 降至 `0.007236`，客厅 16-step 虽 ratio 大幅上升，
state-action MSE 从 `0.088836` 上升到 `0.315394`，同时 state-only MSE 上升到
`3.407966`，说明长 horizon 结果对样本组成高度敏感。

### 不能直接接受的原因

旧训练 Replay 每个 shard 只有 16 transition。lag=1、horizon=16 需要该 shard 前一物理步
动作，但旧 shard 没有保存。离线脚本只能删除这些无上下文 shard，导致：

- 餐桌、厨房 16-step 训练 transition 从 42 降到 14；
- 客厅 16-step 训练 transition 从 84 降到 42；
- 可能选择性保留 lag=0 或较易样本。

因此该结果是足以改变实验路由的强诊断，但不是可接受的测量修复证据。产物保存在
`diagnostics/r0001-observation-lag-offline.json`。

## 更新后的路由

本节取代上一节“直接进入 `P02`”的路由：

1. 冻结并实施 `R0001-P09`，用完整 128-transition 轨迹、显式 lag 和不丢 Episode 的配对
   数据确认 observation-action 时间合同；
2. `P09` 若在正式 `rho=0.96` 和确认 `rho=0.5` 上全部过线，则接受测量修复，并将
   `P02` 标为 `rejected without run`；
3. `P09` 无丢样本后仍存在真实失败，才运行增强 `P02`；
4. 世界模型、动作执行头和安全正例采样仍是独立瓶颈，分别按 `P05/P10/P11` 路由，不与
   `P09` 捆绑；
5. 当前仍没有家务闭环能力提升证据。

## `R0001-P09` 正式诊断：`rejected`

### 运行身份与完整性

- run ID：`r0001-p09-observation-lag-s20260901`
- 源码提交：`3d199c5bfe9ff17634ac423113e85beedc5b2be5`
- 类型：无模型训练的物理时间对齐诊断
- 运行目录：
  `runs/research-loop/0001/r0001-p09-observation-lag-s20260901`
- 报告 schema：`hwr.observation-action-alignment/v1`
- 报告 SHA-256：
  `21d385fe4c39072652e1571a0e1254ab029b08548ad204651a50b9981c53a8cb`
- Episode：96/96
- 每条 Episode：1 个前置动作、128 个评分 transition、130 个物理状态
- artifact：97 个，逐文件 SHA-256 和字节数全部复核通过
- 分布：
  - 三任务各 32 Episode；
  - `rho=0.96` 与 `rho=0.50` 各 48；
  - lag=0 与 lag=1 各 48；
  - training 与 holdout 各 48；
  - 每个任务、每个 rho 均为 8 training + 8 holdout，lag 0/1 各半。
- 所有 provenance 完整，lag=1、horizon=16 没有丢 Episode 或 transition。

正式 run 的第一次执行完整发布上述产物并以状态 0 回调。macOS `launchctl submit` 随后把
短时 runner 误判为 keepalive job，重复启动 9 次；后续启动均因不可覆盖的 run 目录已存在而
立即失败，只覆盖了有界 supervisor 日志，没有修改任何正式产物。主 Agent 已移除该
supervisor，并确认没有残留进程。该运行器问题不改变首次正式结果，但后续 Host runner 必须
在一次回调后显式注销服务。

### 聚合 Probe 结果

完整 8+8 Episode 的主诊断和确认组中，对齐后的聚合数据均通过点估计与同步 bootstrap：

| rho | 任务 | 最弱 horizon ratio | 同步 `p05` |
|---:|---|---:|---:|
| 0.96 | 餐桌 | 1.151361 | 1.022206 |
| 0.96 | 厨房 | 1.337591 | 1.200115 |
| 0.96 | 客厅 | 1.183184 | 1.050999 |
| 0.50 | 餐桌 | 1.707457 | 1.533979 |
| 0.50 | 厨房 | 1.510770 | 1.267329 |
| 0.50 | 客厅 | 1.702504 | 1.480154 |

lag=0 分区的旧索引与对齐索引在 `1e-10` 内完全一致。固定 `rho=0.96` 的旧索引聚合结果
本身也全部过线，说明 `P01` 的 Replay 失败不能仅由随机过程相关系数或 observation lag
解释；完整连续 128-transition 证据、seed、Episode 长度和 Replay 物理显著性窗口选择之间
至少还有一个未隔离变量。

### lag=1 守护指标

冻结合同要求 lag=1 分区在每个 horizon 的 state-action MSE 相对旧索引至少下降 10%。
该守护在两个 rho 均失败。

`rho=0.96`：

| 任务 | 1-step | 4-step | 8-step | 16-step |
|---|---:|---:|---:|---:|
| 餐桌 | -1.93% | +1.79% | +0.44% | +9.47% |
| 厨房 | +11.58% | +11.41% | +4.92% | +6.08% |
| 客厅 | -2.37% | +1.67% | +0.51% | +9.85% |

`rho=0.50`：

| 任务 | 1-step | 4-step | 8-step | 16-step |
|---|---:|---:|---:|---:|
| 餐桌 | +44.52% | +6.46% | -30.13% | -19.82% |
| 厨房 | +46.66% | +5.40% | -34.03% | -27.29% |
| 客厅 | +44.55% | +6.38% | -30.49% | -19.51% |

正值表示对齐后 MSE 下降，负值表示恶化。低相关动作在 1-step 明显受益，但长 horizon
动作窗口只有边界一项不同，对齐后反而恶化；高相关动作的相邻动作相似，对齐收益很小。
因此“单一 observation lag 修复会跨全部 horizon 稳定改善”被正式否定。

### 结论与路由

`R0001-P09` 按预注册合同标记 `rejected`，不能将其实现作为新的能力基线。它仍提供三项
有效诊断：

1. observation lag 必须显式记录，不能依靠 seed 事后重建；
2. 旧 16-transition shard 会使 lag=1、16-step 诊断缺少前置动作；
3. P01 的失败与 P09 完整短轨迹的通过存在明显证据池差异。

`R0001-P02` 不立即进入正式 24 Episode：P09 中固定 `rho=0.96` 的完整短轨迹已经过线，
而 `rho=0.50` 虽有更大聚合 margin，却没有预注册的“基线失败、候选过线”因果对比。
不得在看到结果后把 margin 提升改成新的接受标准。

下一步重新筛选以下互斥解释：

- 正式 Replay 的 16-transition 物理显著性窗口是否造成 horizon 证据不足或选择偏差；
- `P05` 的单 source Episode batch 是否仍是世界模型动作利用失败的主因；
- `P10` 的安全正例稀疏和 `P11` 的 plant proposal history 分别如何解释动作执行失败。

## `R0001-P12`：未见语言评测物化，`accepted as platform repair`

- 实现提交：`34a7bb0`
- 在任何环境或物理 Episode 创建前，用训练时同一锁定 Qwen3 encoder 物化三任务各
  3 条、共 9 条 evaluation instruction。
- 产物严格写入单次评测目录的 `evaluation-language/cache/` 与
  `evaluation-language/manifest.json`。
- 最终评测 artifact manifest 直接哈希语言 manifest 和 9 个 embedding 文件。
- 训练 run 的 Replay、language index、feature cache、checkpoint 和 deployment 保持只读。
- fixture 验证训练已见 embedding 与确定性动作逐元素不变。
- focused tests 18 项、架构和 Python 尺寸门通过。

该项只修复正式评测在未见语言上的启动阻断，不是语言泛化提升。真实锁定 Qwen3 权重的
9/9 物化仍需在首次闭环评测前执行一次 preflight。

## `R0001-P13`：闭环安全干预负担门，`accepted as evaluation repair`

- 实现提交：`aeaedd7`
- `hwr.bimanual-acceptance/v2` 升级到 v3。
- 正常模式每 Episode 计算
  `safety_interventions / max(steps, 1)`。
- 每任务固定 2,000 次 Episode-cluster bootstrap 和预注册 seed。
- 与原接受合同联合增加：
  - empirical `p95 <= 0.01`；
  - bootstrap 95% upper bound `<= 0.02`；
  - maximum Episode rate `<= 0.05`。
- 高撞盾、零干预、单极端 Episode、bootstrap 重现和不运动负对照均有测试。
- focused 与组合测试通过，架构和 Python 尺寸门通过。

该项不修改安全层或策略，只防止把依赖硬安全层持续兜底的策略误报为安全能力提升。

## `R0001-P16` 正式功效评测：`rejected`

### 运行身份与完整性

- 源码提交：`03de3fcfaa151efc6dd92ae2081ca3ad08732798`
- 运行目录：
  `runs/research-loop/0001/r0001-p16-probe-power-s20260916`
- 报告 schema：`hwr.action-probe-power/v1`
- 报告 SHA-256：
  `b81615e80fe10bfb7314d6519d182e3da88449c8948c3adaca192e870e662821`
- P09 输入 manifest SHA-256：
  `0b23fcba819a8aa62200d0d286895d75c3766e373138d4af932481998eed5200`
- P09 输入 report SHA-256：
  `21d385fe4c39072652e1571a0e1254ab029b08548ad204651a50b9981c53a8cb`
- 每种效应条件 500 trial，每 trial 200 次同步 Episode bootstrap；
- null、planted 和 action-permutation 三种条件全部运行；
- 500 个 trial 的全部 pass/fail 均保存在报告中；
- 墙钟 65.02 秒，峰值常驻内存约 258 MiB；
- runner 在回调前成功移除自身 launchd 服务，没有重复启动或残留进程。

报告和 manifest 哈希一致，source commit 与运行时 HEAD 一致，产物完整。该 run 未训练任何
模型，也没有使用真实 target、任务奖励或安全标签生成合成 target。

### 三个设计臂

| 设计臂 | null FPR | permutation FPR | planted power | 判定 |
|---|---:|---:|---:|---|
| `fragmented_7x16` | 0.0% | 0.0% | 0.0% | 功效不足 |
| `continuous_same_7_starts` | 0.0% | 0.0% | 0.0% | 功效不足 |
| `continuous_all_starts` | 0.0% | 0.0% | 5.8% | 功效不足 |

三个设计均没有发现零效应或 action permutation 假通过，但没有任何设计达到冻结的
`>=80%` planted-effect 检出率，因此 `qualified_arms=[]`，`p14_route=blocked`。

### 失败指纹

`continuous_all_starts` 在 `rho=0.50` 上几乎具有满功效：

- 三任务 1/4/8-step 检出率均为 100%；
- 三任务 16-step 检出率为 99.8%、99.8%、100%；
- 逐任务全 horizon 通过率为 99.8%、99.8%、100%。

同一设计在 `rho=0.96` 上显著退化：

| 任务 | 1-step | 4-step | 8-step | 16-step | 全 horizon |
|---|---:|---:|---:|---:|---:|
| 餐桌 | 99.0% | 96.6% | 86.4% | 41.2% | 40.2% |
| 厨房 | 98.8% | 95.6% | 81.4% | 43.0% | 42.4% |
| 客厅 | 99.6% | 97.0% | 87.6% | 39.8% | 39.4% |

正式 trial 要求两个 rho、三个任务、四个 horizon 全部通过；高相关三任务的长 horizon
共同限制整体功效到 5.8%。

长 horizon 设计矩阵也显示强病态：

- `continuous_all_starts`：每任务 776 行、54 列、数值秩 53；
- `rho=0.50` condition number 约 `4.67e6`；
- `rho=0.96` condition number 约 `5.20e6～5.35e6`；
- 两个 7-start 设计只有 56 行、54 列、数值秩 53，功效为 0。

这说明硬分片确实丢弃长时证据，但仅恢复全部重叠起点仍不能在固定高相关随机过程中可靠
检出冻结的动作效应。当前瓶颈不是单一实现 bug，而是训练数据激励与条件辨识设计共同造成。

### 结论与路由

`R0001-P16` 按预注册合同标记 `rejected`，不是基础设施失败。由于所有设计的 power 都低于
80%，当前 Action Probe 的失败结论只能标记为测量 `inconclusive`；不得降低 `1.05/1.01`
门槛，也不得继续 `P14/P15` 物理确认。

该结果重新提供了支持多时间尺度激励的因果证据：

- `rho=0.50` 在完全相同的合成效应、噪声、Episode 数和模型容量下具有近满功效；
- `rho=0.96` 的长 horizon 功效不足；
- 旧 `P02` 不能直接复活，因为其原接受标准没有预注册 P16 功效结果；
- 下一阶段必须重新生成并独立筛选“预算不变的多时间尺度训练激励”或新的条件随机化测量，
  不能事后把单分区通过改成接受标准。

当前依然没有启动正式训练：在低功效准入门下训练会把有效候选随机误杀，既不能形成可信
能力结论，也不符合训练前先做最低成本判别验证的约束。P16 完成不是研究循环终点，下一轮
提案和筛选已继续启动。
