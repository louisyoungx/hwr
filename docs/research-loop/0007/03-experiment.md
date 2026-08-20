# R0007 冻结实验

## 冻结范围

本轮只实施两个互不依赖的候选：

1. `R0001-P32-E1`：双重正交普通 Replay 条件信息门；
2. `R0001-P40-E1`：分类型 allowed-contact 力—冲量安全总账。

本轮不启动正式训练，不运行 policy 闭环能力 Episode，不修改 deployment gate，不生成
能力成功率。P32 与 P40 可由独立实施 Agent 并行开发，但正式结果均由主 Agent 在各自
干净、已提交、通过门禁的源码提交上运行。

## 共同源码与数据身份

- 冻结文档父提交：`a722f3522cdb8f12c1a78c56ce8c1d7c873e9190`
- 分支：`feat/research-loop`
- Python：3.11.0
- PyTorch：2.13.0
- 当前沙箱 MPS：built，not available
- 正式运行 device：CPU
- 历史 `docs/research-loop/0001/`～`0006/` 必须保持与起始 tree 完全一致。

## 文件所有权

| 候选 | 唯一负责人 | 分支/工作区 | 可修改文件 |
|---|---|---|---|
| `R0001-P32-E1` | 实施 Agent F | 独立 fork | `src/hwr/eval/replay_conditional_information.py`、`src/hwr/apps/evaluate_replay_conditional_information.py`、`tests/test_replay_conditional_information.py`、`tests/test_replay_conditional_information_app.py` |
| `R0001-P40-E1` | 实施 Agent G | 独立 fork | `configs/adapters/mujoco/formal_3d_v1.json`、`src/hwr/adapters/mujoco/bindings.py`、`src/hwr/adapters/mujoco/contact_ledger.py`、`src/hwr/adapters/mujoco/formal_household_backend.py`、`src/hwr/adapters/mujoco/__init__.py`、`src/hwr/apps/evaluate_contact_ledger.py`、`tests/test_contact_ledger.py`、`tests/test_contact_ledger_app.py`、`tests/test_formal_household_dual_arm_backend.py` |
| 集成、正式运行、结果与总结 | 主 Agent | `feat/research-loop` | `docs/research-loop/0007/`、`runs/research-loop/0007/` |

实施 Agent 不得修改对方文件，不得修改 `docs/research-loop/0007/`，不得扩展范围；发现
需要额外文件时先返回主 Agent决定。每项提交必须原子化并引用提案 ID。

## `R0001-P32-E1`

### 冻结问题与结论边界

问题：

> 当前 salience-retained 普通 Replay 中，在给定 pre-action visible state 后，
> executed-action residual 是否仍含稳定的 successor visible-proprio residual 条件预测
> 信息？

唯一比较变量：

- control：`m_y(S)`；
- candidate：`m_y(S) + B(a - m_a(S))`。

共享同一个 state nuisance baseline；不得改为两个独立容量的 state/state-action model。

即使通过，结论也只能写为：

`accepted as retained-Replay conditional information evidence`

禁止写为 plant causality、production RSSM utilization、模型能力、训练改善、闭环成功、
安全改善或完整自主采集分布结论。

### 冻结输入

- root：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/replay/autonomous`
- manifest：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/replay/autonomous/manifest.json`
- manifest SHA-256：
  `c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985`
- schema：`hwr.autonomous-trajectory-dataset/v2`
- 24 source Episode、168 shard、2,688 transition；
- 每 source 7 个不重叠 16-transition shard；
- task/source：餐桌 6、厨房 6、客厅 12。

loader 必须逐 shard 验证 manifest bytes/hash、NPZ shape、有限值、source ID、task、绝对
transition 范围和无重叠；任何漂移在创建输出目录前 fail-closed。

### 冻结字段与 target

37-D visible proprio 顺序：

| 索引 | 字段 |
|---|---|
| `0:6` | left joint position |
| `6:12` | left joint velocity |
| `12:18` | right joint position |
| `18:24` | right joint velocity |
| `24` | left gripper position |
| `25` | right gripper position |
| `26:29` | base pose `(x,y,yaw)` |
| `29:31` | base twist |
| `31:37` | IMU |

主输入和动作：

- `S_t = proprioception[t, 0:37]`；
- `a_t = executed_action[t, 0:16]`。

主 `rate` target 为现有 probe 的 16-D controllable-state 一步 innovation：

- source indices：
  `[6,7,8,9,10,11,18,19,20,21,22,23,24,25,29,30]`；
- `y_t = controllable(proprioception[t+1]) - controllable(proprioception[t])`；
- 不除以 50ms，不转换为连续时间导数；
- 不读取 IMU、reward、success、task ID、seed、source position、真实 latency/scale 或
  successor 之外的未来帧。

`configuration` 守护 target 为 17-D 一步 delta：

- source indices：
  `[0,1,2,3,4,5,12,13,14,15,16,17,24,25,26,27,28]`；
- 其余维度直接 `t+1 - t`；
- yaw 对应 source index 28，使用
  `atan2(sin(yaw[t+1]-yaw[t]), cos(yaw[t+1]-yaw[t]))`。

### 冻结 source、fold 与 history

- 最外层独立单位：source Episode。
- outer 3-fold，按 task 分层；每折固定留出餐桌 2、厨房 2、客厅 4 个 source。
- task 内 source 按
  `SHA256("R0001-P32-E1|outer-v1|" + task_id + "|" + source_id)` 排序后 round-robin
  分配到 fold 0/1/2。
- 每个 outer-train 内做 inner 2-fold；按
  `SHA256("R0001-P32-E1|inner-v1|" + outer_fold + "|" + task_id + "|" + source_id)`
  排序后 round-robin 分配。
- 实现必须在正式 run 前输出完整 fold manifest；任何 source 跨 outer 或 inner
  train/validation 即 fail-closed。
- 主 analysis 的 nuisance input 仅为 `S_t`。
- controller-context guard 在 `S_t` 后追加：
  - current `actor_proposal[t]` 16-D；
  - 同一连续 shard 内过去 4 步的 `actor_proposal` 与 `executed_action`，共 128-D；
  - 4-D availability mask；
  - 缺失 history 以零填充并由 mask 标识，不删除 transition；
  - 不跨不连续 shard 拼 history。

### 冻结拟合

- 所有 ridge penalty 固定为 `1e-3`，bias 不惩罚。
- `m_a`、`m_y`：
  - inner-train feature mean/std 只由 inner-train 计算；
  - std `<1e-8` 固定为 1；
  - 在 inner-train 拟合，在 inner-validation 产生 OOF residual；
  - action 与 target nuisance 分开拟合；
  - inner-validation 不参与对应 nuisance 的标准化或权重。
- residual map `B`：
  - 只使用 outer-train 的 inner-OOF `u=a-m_a(S)` 与 `v=y-m_y(S)`；
  - `u` mean/std 只由 outer-train OOF residual 计算，std `<1e-8` 为 1；
  - target error scale 使用 outer-train 原始 `y` 的逐维 std，`<1e-8` 为 1；
  - `B` 直接预测 raw `v`，ridge `1e-3`，含不惩罚 bias。
- outer-test：
  - `m_a`、`m_y` 用全部 outer-train 重新拟合；
  - feature mean/std、action-residual scale 和 target error scale 全部来自 outer-train；
  - outer-test 不参与任何拟合、scale、rank、门槛或 planted calibration。
- 不扫描 ridge、target、fold、history length、rank threshold 或任何接受门。

### 冻结统计

- 每 transition error：
  对 16-D 或 17-D 误差按 outer-train target scale 标准化后平方并跨维平均。
- 先在每 source 内平均，再在 task 内 source 等权平均，再对三个 task 等权平均。
- ratio：
  `MSE_control / max(MSE_candidate, 1e-12)`。
- source log-ratio：
  `log(max(MSE_control_source,1e-12)/max(MSE_candidate_source,1e-12))`。
- bootstrap：
  - 每个 outer fold 内按 task 对 outer-test source 有放回重采样；
  - 三个 task 等权；
  - 5,000 replicate；
  - seed `20263201`；
  - lower bound 为 one-sided 95% empirical quantile `p05`；
  - 相同 replicate multiplicity 同步用于 control/candidate、target family 和 guard。
- task point estimate：对应 task 全部 outer-test source 合并后的 source 等权 ratio。
- effective rank：全部 outer-train inner-OOF action residual 经各 outer-train scale 后的
  covariance entropy rank；报告三个 outer fold 的最小值。

### 冻结功效

- formal power trial：200；
- 每 trial bootstrap：1,000；
- power base seed：`20263211`；
- 使用实际 source/task/shard/history mask 结构和完整 nested pipeline。
- 两个 null：
  - `zero_action_residual`：生成 target 时 action-residual 系数固定为 0；
  - `random_target`：在每个 task/source 内用预定 seed 生成与训练折 target scale 相同的
    独立 Gaussian residual；不得 shuffle action。
- planted：
  - 方向矩阵由每个 trial 的训练折 seed 生成并正交化；
  - scale 只由相应训练折 target residual RMS 标定；
  - 冻结为 oracle control/candidate MSE ratio `1.10`；
  - 不读取 outer-test target 或 residual 标定 effect。
- 每 trial 按正式主门判 pass，但不要求 controller/configuration guard。
- null FPR 以 exact Clopper-Pearson 95% upper bound 评估，两个 null 均须 `<=0.05`。
- planted power 以 exact Clopper-Pearson 95% lower bound 评估，须 `>=0.80`。

smoke test 可显式降低 trial/bootstrap 次数，但正式模式必须拒绝参数覆盖。

### 冻结守护与判定

主门全部满足：

1. `rate` aggregate ratio `>=1.05`；
2. `rate` aggregate mean log-ratio bootstrap `p05 > 0`；
3. 三任务 `rate` ratio 分别 `>1`；
4. aggregate candidate absolute MSE `<` control；
5. 三个 outer fold 的 action-residual effective rank 最小值 `>=6`；
6. 两个 null FPR upper 均 `<=0.05`；
7. planted power lower `>=0.80`。

机制守护全部满足：

1. controller-context `rate` aggregate ratio `>=1.02` 且 log-ratio `p05 >0`；
2. `no_rewrite` 的 `rate` aggregate ratio `>1` 且 log-ratio `p05 >0`；
3. `shard_interior` 的 `rate` aggregate ratio `>1` 且 log-ratio `p05 >0`；
4. `configuration` aggregate ratio `>1` 且 log-ratio `p05 >0`；
5. configuration 三任务 ratio 分别 `>1`；
6. safety-rewrite、shard-prefix、每任务、每 source、fold 和 target-family 结果全部发布，
   即使样本不足也不得隐藏。

判定：

- `accepted as retained-Replay conditional information evidence`：主门与机制守护全部通过；
- `rejected`：
  - rank/power 合格但主门失败；
  - 信号只存在于 safety rewrite 或 shard prefix；
  - controller context 消除信号；
  - configuration guard 失败；
  - candidate 绝对误差恶化；
- `inconclusive`：
  - rank 或 exact-pipeline power 不足；
  - 任一 required stratum 无 source-level 测量功效；
  - artifact、lineage、shape、fold 或 hash 无法重建；
- `invalid`：任一 source 泄露、outer-test 调参、结果后改 target/fold/scale/门槛。

### 运行与产物

正式 run：

`runs/research-loop/0007/r0007-p32-replay-conditional-e1-s20263201`

命令：

```text
.venv/bin/python -m hwr.apps.evaluate_replay_conditional_information \
  --input-run runs/foundation-world-model/r0001-p01-baseline-v4-s20260812 \
  --output runs/research-loop/0007/r0007-p32-replay-conditional-e1-s20263201
```

必须原子写入：

- `report.json`；
- `folds.json`；
- `manifest.json`；
- 失败时 `failure.json`，且不得留下伪完整 report。

manifest 必须记录 source commit、稳定命令、输入 manifest hash/bytes、所有直接产物
hash/bytes、正式常量和 capability-claim 禁止位。

## `R0001-P40-E1`

### 冻结问题与结论边界

问题：

> 现有 forbidden-contact safety 指标跳过 allowed-contact geom；能否在完全不改变行为和
> 安全决策的前提下，建立 robot–environment 分类型 normal-force/impulse 总账？

唯一主变量是 report-only measurement。禁止改变：

- `allowed_robot_contact_geoms` 的并集；
- forbidden force 220N 阈值；
- predictive safety；
- action、reward、termination、success；
- physics timestep、solver 或 control frequency。

通过后的唯一结论：

`accepted as safety measurement contract evidence`

不能声称仿真或硬件安全改善；220N 仅作为旧 forbidden-contact 内部阈值参照。

### 冻结语义角色

在现有 binding 中新增显式 `allowed_robot_contact_roles`，其所有 geom 的无重复并集必须
严格等于现有 `allowed_robot_contact_geoms`：

- `floor_support`
- `manipulated_object`
- `target_container`
- `articulation`

冻结映射：

| task | floor/support | manipulated object | target container | articulation |
|---|---|---|---|---|
| living | `floor` | `toy_duck_collision`, `toy_football_collision` | `basket_front_collision`, `basket_back_collision`, `basket_left_collision`, `basket_right_collision`, `basket_bottom_collision` | 空 |
| dining | `floor` | `dining_cup_collision`, `dining_plate_collision` | `cup_holder`, `plate_holder` | 空 |
| kitchen | `floor` | `cleaner_yellow_collision`, `cleaner_pink_collision` | `drawer_bottom`, `drawer_front`, `drawer_back`, `drawer_left`, `drawer_right`, `drawer_divider` | `drawer_handle` |

语义范围：

- 只统计恰有一端属于 robot root 的 robot–environment contact；
- robot self-contact 与 world–world contact 不进入四类或 forbidden force/impulse，但分别
  计 ignored contact-point 数；
- robot–other 且 other 不在 allow-list 时归 `forbidden`；
- allowed geom 缺角色、重复角色、角色名未知、角色并集与 allow-list 不一致均在 backend
  初始化时 fail-closed；
- 不使用 geom 名称模式推断角色。

### 冻结 substep 聚合

每个 physics substep：

1. 枚举全部 contact point；
2. 对 robot–environment contact 调用 `mj_contactForce`；
3. normal force 为 `abs(force[0])`，必须有限且非负；
4. 以排序后的无序 `(robot_geom_id, other_geom_id)` 为 pair key；
5. 同一 pair 多个 contact point 的 normal force 先求和；
6. pair 依据 other geom 的冻结角色进入一个类别；
7. 每类别 substep total force 为该类别全部 pair force 求和。

每个 control period：

- `pair_peak_force`：所有 substep/pair force 的最大值；
- `category_peak_force`：所有 substep category-total force 的最大值；
- `category_impulse`：`Σ_substep category_total_force × model.opt.timestep`；
- `contact_duration_seconds`：该类别 category-total force `>0` 的 substep 数乘 timestep；
- Episode cumulative impulse 为 control-period impulse 之和；
- Episode peak 为 control-period peak 的最大值；
- 同时记录 nonfinite count、contact-point count、unique-pair observation count、
  ignored self/world count。

缺失/nonfinite force 必须单独计数并使正式合同失败，不得替换为 0。

### 冻结最小验证

1. 配置合同：
   - 三任务四角色键完整；
   - role geom 无重复；
   - role union 等于 legacy allow-list；
   - 所有 geom 可解析；
   - 未知/冲突/missing 立即拒绝。
2. 单元 fixture：
   - pair 顺序反转不改变 key；
   - 同一 pair 多 contact point 先求和；
   - 不同 pair 不误合并；
   - self/world contact 只进 ignored count；
   - nonfinite force fail-closed；
   - peak、impulse、duration 公式逐字段核对。
3. timestep stability：
   - 固定最小 MuJoCo 接触 fixture、相同物理时长和初态；
   - timestep `dt` 与 `dt/2`；
   - 每类别 cumulative impulse 相对差 `<=10%`；
   - 不要求 solver 轨迹逐步 bit-identical。
4. 正式三任务 deterministic trace：
   - evaluation profile，每任务固定 seed `20264001 + task_index*104729`；
   - camera rendering disabled；
   - 32 个 control step 或提前终止；
   - 使用固定 hold action，不作能力或安全成绩；
   - 同一 trace 在 measurement disabled/enabled 两臂运行；
   - applied action vector、proprioception、terminated、truncated、success、reason、
     severe collision count、maximum forbidden force、maximum forbidden pair、
     safety intervention 逐步/最终 bit-identical；
   - measurement-enabled 报告五类别总账。

### 冻结判定

`accepted as safety measurement contract evidence` 必须全部满足：

1. 配置角色完备、互斥且与 legacy allow-list 并集一致；
2. contact point → unordered pair → category 聚合顺序正确；
3. substep impulse、control-period peak、Episode cumulative 公式正确；
4. nonfinite/missing fail-closed；
5. timestep-halving fixture 各非零类别 impulse 相对差 `<=10%`；
6. measurement enabled/disabled deterministic trace 的旧行为与旧安全指标
   bit-identical；
7. 三任务 trace、全部类别、零值、ignored count 和 forbidden 类均完整发布；
8. 新测量未进入 safety、reward、termination 或 success；
9. 报告固定：
   - `capability_claim_allowed=false`
   - `hardware_safety_claim_allowed=false`
   - `measurement_only=true`
   - `legacy_safety_decision_unchanged=true`

以下任一发生则 `rejected`：

- 分类不完备或不互斥；
- pair/contact point 重复或漏记；
- timestep 稳定性失败；
- enabled/disabled 旧行为或旧安全指标不同；
- 将 220N 解释为硬件安全阈值；
- 新测量参与运行时决策。

若 MuJoCo API、fixture 或 action trace 无法重建，则 `inconclusive`。

### 运行与产物

正式 run：

`runs/research-loop/0007/r0007-p40-contact-ledger-e1-s20264001`

命令：

```text
.venv/bin/python -m hwr.apps.evaluate_contact_ledger \
  --output runs/research-loop/0007/r0007-p40-contact-ledger-e1-s20264001
```

必须原子写入：

- `report.json`；
- `manifest.json`；
- 失败时 `failure.json`。

manifest 必须记录 source commit、稳定命令、binding hash/bytes、solver/timestep、fixture
身份、所有直接产物 hash/bytes 和禁止声明位。

## 实施后共同门禁

先运行各自 focused tests，再运行：

```text
.venv/bin/python scripts/check_python_size.py
.venv/bin/python scripts/check_architecture.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m compileall -q src tests scripts
git diff --check
```

两项集成后运行全量：

```text
.venv/bin/pytest -q
```

正式 run 前必须：

1. 文档冻结提交是实现提交祖先；
2. 实现已原子提交；
3. 工作区干净；
4. relevant focused tests 与共同门禁通过；
5. 输入 hash、binding hash 与冻结值一致；
6. 历史 `docs/research-loop/0001/`～`0006/` 零差异。

## 本轮停止条件

- P32 rank/power 不足：记录 `inconclusive`，不改门槛、不启动 P43 或训练。
- P32 功效充分但主门/守护失败：记录 `rejected`，下一轮优先改变数据采集支持。
- P32 通过：只授权未来修订 P31，不启动 P43、P33 或正式训练。
- P40 任一测量合同失败：记录 `rejected` 或 `inconclusive`，不启动 P41-E1。
- 本轮无论结果如何都不运行 capability benchmark，不启动正式训练。
