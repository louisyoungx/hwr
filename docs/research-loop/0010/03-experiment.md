# R0010 冻结实验

## 冻结状态

- 冻结候选：
  - `R0001-P51-E1`
  - `R0001-P50-E1`
  - `R0001-P50-E2`
- 延期：
  - `R0001-P50-E3`
  - `R0001-P56`
- 本文件必须在实现、salt reveal、candidate bank 构建和正式运行前提交。
- 实现完成后先通过 focused tests、项目门禁和独立代码审查，再提交原子实现。
- 所有正式运行只能从干净、已提交且以本冻结提交为祖先的 source commit 启动。
- 本轮无正式训练，不需要 MPS、tmux 或 host-exec 休眠。
- 正式 MuJoCo acquisition/physical runs 串行执行，不并发污染时序与资源测量。

## 固定任务与支持域

任务顺序固定为：

1. `tidy_living_room_3d/v1`
2. `clear_dining_table_3d/v1`
3. `store_kitchen_items_3d/v1`

支持域 cell 固定为每任务的：

```text
(observation latency, action latency)
(1,1), (1,2), (2,1), (2,2)
```

- latency 由正式 evaluation randomization 自然采样，不允许 reset override。
- latency 3 不删除，作为 seed rejection 记录，但不进入本轮支持域 cohort。
- task、cell 和 replicate 按上述顺序确定，不按结果重排。

## 通用 seed 与 provenance 合同

两项实验使用不同、未公开的 256-bit lowercase-hex salt。当前提交只冻结字符串本身的
SHA-256 commitment：

| 实验 | salt commitment |
|---|---|
| `R0001-P51-E1` | `a12c867f79013a830a89ea1e76b7c50c6df260bff2bf9ee502f49a56cc501d2b` |
| `R0001-P50-E1/E2` | `ed945b2dcfe90c6aab639164da32cc8a1a905df56534c42a443d1bd4753e16a4` |

salt reveal 保存在 ignored 的 `runs/research-loop/0010/.host/`，不进入当前冻结提交。
runner 使用 `hwr.opaque-episode-seeds/v1`：

```text
planned_episode_id =
  SHA256(schema || plan_id || task_id || cell_id || candidate_ordinal)

environment_seed =
  int63(SHA256(salt || "environment" || planned_episode_id))

policy_rng_seed =
  int63(SHA256(salt || "policy" || planned_episode_id))
```

- role 不进入 seed derivation。
- 必须验证 reveal 与 commitment。
- 每个 experiment 的环境种子和 policy seed 跨全部 checked/planned Episode 唯一且域分离。
- manifest 绑定：
  - source commit；
  - 本冻结文档提交；
  - Python、NumPy、MuJoCo 版本；
  - task/binding、递归 XML、机器人模型与相关源码 bytes/hash；
  - salt commitment/reveal；
  - seed derivation；
  - command；
  - artifact bytes/hash。
- clean-source gate 必须使用
  `git status --porcelain --untracked-files=all`，但允许输出目录本身不存在。
- 历史 `docs/research-loop/0001/`～`0009/` tree 必须与 `00-context.md` 一致。

## `R0001-P50-E1`：immutable acquisition evidence capsule

### 负责人、分支与文件所有权

- 唯一实施 Agent：`P50 worker`
- 分支：`exp/R0001-P50-E1-capsule`
- worktree：由主 Agent 在实现前创建。
- 允许修改：
  - 新 `src/hwr/eval/candidate_funnel.py`
  - 新 `src/hwr/adapters/mujoco/candidate_acquisition.py`
  - 新 `src/hwr/apps/evaluate_candidate_acquisition.py`
  - 新 `tests/test_candidate_funnel.py`
  - 新 `tests/test_candidate_acquisition.py`
  - 新 `tests/test_candidate_acquisition_app.py`
  - 必要的 `src/hwr/eval/__init__.py`、`src/hwr/apps/__init__.py`
- 不允许修改：
  - `src/hwr/eval/target_selection.py`
  - P41/P51/P52 现有 app 与 bridge
  - task、binding、XML、safety、backend
  - `docs/research-loop/0001/`～`0009/`

若无法在不修改 `target_selection.py` 的情况下复用 gate，必须停止并交主 Agent决定，不得
复制一套近似 generator。

### 唯一主变量

增加 acquisition 输入证据的持久化，不改变：

- acquisition action；
- observation source age；
- candidate generator、bytes、顺序、score 与 selected index；
- primitive、safety、termination、task 或 backend。

artifact 中为每个 A1/A3 capture 与 A4 final input 保存：

- 完整 serialized `PolicyVisibleInput` bytes；
- capture ordinal；
- `(observation_timestamp_ns, sequence_id)`；
- 完整 bytes SHA-256 与 byte length；
- candidate-visible observation subpayload bytes/hash：
  - RGB；
  - depth；
  - depth-valid mask；
  - camera intrinsics；
  - `robot_from_head_camera`；
  - base pose、左右关节和 self-mask 所需 proprioception。

phase、history、policy seed 与 safety 等完整 policy fields 保留在完整 payload，但不进入
candidate-visible identity hash。相同 observation identity 下 candidate-visible
subpayload 不同必须 fail-closed；完整 payload 因 phase/history 不同可以不同。

### 计划与资源

- `plan_id = R0001-P50-E1-formal`
- 每个 12-way cell 固定前 2 个 latency-matched seed，共 24 个 planned Episode。
- 每 cell 最大原始 `candidate_ordinal = 95`，即最多检查 96 个 seed。
- seed 只因自然 sampled latency 与 cell 不匹配而在 planned cohort 前拒绝。
- 一旦 latency 匹配并成为 planned Episode：
  - acquisition failure；
  - empty candidate；
  - terminal；
  - missing payload；
  均保留，不 replacement。
- 每个 planned Episode 与 validation replay 最多执行 P41 原 acquisition 的 995 个
  control step，物理 runtime terminal 必须提前停止并原样保留。
- 上界：1,152 个 latency sampler 调用、24 个 planned MuJoCo acquisition Episode 加
  24 个同 seed validation replay，最多 47,760 个 control step；无 post-selection。
- validation replay 使用新的 backend/reset 完整重跑同一 seed，并同时关闭 artifact
  capture side effect；它不是在同一 payload stream 上调用第二个 controller state。
- 原始 capsule 预估上限 24 × 49 × 0.35MiB 约 412MiB；运行前须确认数据卷至少 5GiB
  可用。

### 确定性与负向测试

1. 单 blob deserialize→serialize bit-identical。
2. candidate-visible subpayload 的 canonical identity 忽略 phase/history，但任何可见字段
   变化都会改变 hash。
3. 同 observation identity、不同 candidate-visible payload fail-closed。
4. 从保存 bytes 离线调用正式 `generate_candidate_set`，candidate canonical bytes/hash
   与在线完全一致。
5. 同 seed 重放一次，capture identity 序列、payload bytes/hash、candidate bytes 与
   acquisition proposed/applied action trace bit-identical。
6. capture enabled/disabled 的 proposed/applied action trace、observation identity、
   final candidate bytes bit-identical。
7. missing/duplicate/unplanned Episode、replacement、salt mismatch、dirty source、
   source/config/XML drift 均 fail-closed。

### 正式命令与产物

实现必须提供：

```text
.venv/bin/python -m hwr.apps.evaluate_candidate_acquisition \
  --output runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --salt-file runs/research-loop/0010/.host/p50-e1-salt.txt
```

产物至少包含：

- `plan.json`
- `capsules.json` 或一个 canonical index 加独立 binary blobs；
- `report.json`
- `manifest.json`
- 失败时 `failure.json`

### 判定

`accepted as immutable acquisition evidence contract` 仅当：

- 24/24 planned Episode 均发布，12 cell 完整；
- 所有 round-trip、离线 candidate replay、same-seed replay 与 enabled/disabled identity
  门通过；
- 所有 anchor source bytes/hash 可定位且无 duplicate/missing；
- 所有 planned failure 原样保留，无 replacement；
- action bounds、stale action、severe collision、invalid force 与 acquisition safety 守护
  通过；
- source/provenance/claim flags 全部有效。

任一证据字节缺失、identity 不一致、capture 改变行为、不可重放或私有字段进入 policy
capsule：`invalid`。合同有效但 planned acquisition 本身失败不自动使合同 invalid；必须
在 report 中分账。

## `R0001-P50-E2`：离线有序候选漏斗

### 负责人、分支与文件所有权

- 与 P50-E1 同一实施 Agent、同一分支和同一 worktree。
- 只允许在 P50-E1 所列新文件中实现。
- E2 必须是 E1 之后的独立原子提交、独立 report 与独立判定。

### 唯一主变量

对 E1 immutable bytes 运行 report-only 漏斗，不修改正式 candidate output。实现必须
复用正式 generator 的同一 gate 定义或把其无行为变化的纯 gate 重构为共享函数；禁止复制
近似阈值或近似计算。

实际冻结顺序与守恒层：

#### Anchor ledger

1. center/ring validity；
2. prominence；
3. center depth spread；
4. patch support before self-mask；
5. support after self-mask；
6. height/range；
7. planarity；
8. width；
9. raw candidate accepted。

每 frame 必须满足：

```text
enumerated anchor count
= sum(anchor first-rejection counts)
+ raw candidate count
```

每个 anchor 只能进入第一个 terminal stage。

#### Component ledger

1. connected component built；
2. `view_count < 2` rejection；
3. aggregate normal zero rejection；
4. pre-top64 candidate。

必须满足：

```text
component count
= view rejection
+ aggregate-normal rejection
+ pre-top64 candidate count
```

#### Ranking ledger

```text
pre-top64 candidate count
= retained candidate count
+ truncated candidate count
```

随后 canonical reorder 不改变集合成员。

### Unique-observation shadow

- unique identity 固定为 `(timestamp_ns, sequence_id)`。
- RGB/depth/subpayload hash 只作 identity consistency audit。
- shadow component 的 view count 使用 unique observation identity，而不是
  `frame_ordinal`。
- shadow 只报告，不进入正式 candidate。
- shadow view count 不得大于 ordinal view count。
- shadow retained candidate 不得多于正式 retained candidate；否则实现错误。

### 描述性指标与判定

对全部 24 个 Episode 和 12 cell 报告：

- capture ordinal、unique observation identity、unique candidate-visible payload 数；
- 每层 input、rejection、survival 与守恒；
- 空集合 Episode 的最后非空阶段；
- raw/component/pre-top64/final candidate 数；
- ordinal 与 unique-observation shadow candidate 差；
- 最弱 task/cell；
- wall time 与峰值内存。

“同一 cell 的两个 planned Episode 均在同一 stage 拒绝至少 60% 的该 stage 输入”只标记
`repeatable_descriptive_loss_stage`。分母严格是该 stage 输入，anchor/component/ranking
不得混用。该标签不是 gate 的因果伤害证据。

`accepted as candidate-funnel measurement evidence` 仅当：

- E1 已接受；
- 所有三层 ledger 守恒；
- enabled/disabled 的正式 candidate bytes、顺序、score、selected index bit-identical；
- 对同一 capsule 两次 report bit-identical；
- unique-observation shadow 单调性与 identity consistency 通过；
- 24 个 Episode 全部发布，无删除或 replacement；
- 报告不把描述性 stage 外推为 candidate 改善。

计数不守恒、candidate 漂移、重复运行不一致、输入不全：`invalid`。没有重复主导 stage、
raw stage 已空或 unique identity 全唯一都不是实现失败，应标记有效的否定/描述性结果。

## `R0001-P51-E1`：paired physical Cartesian convergence

### 负责人、分支与文件所有权

- 唯一实施 Agent：`P51 worker`
- 分支：`exp/R0001-P51-E1-convergence`
- worktree：由主 Agent 在实现前创建。
- 允许修改：
  - 新 `src/hwr/eval/cartesian_convergence.py`
  - 新 `src/hwr/adapters/mujoco/cartesian_convergence.py`
  - 新 `src/hwr/apps/evaluate_cartesian_convergence.py`
  - 新 `tests/test_cartesian_convergence.py`
  - 新 `tests/test_cartesian_convergence_app.py`
  - 新 `tests/test_cartesian_convergence_mujoco.py`
  - 必要的 `src/hwr/eval/__init__.py`、`src/hwr/apps/__init__.py`
- 不允许修改：
  - `src/hwr/eval/target_selection.py`
  - P41/P50/P51/P52 现有 app、bridge 或合同
  - task、binding、XML、safety、backend
  - `docs/research-loop/0001/`～`0009/`

若不能通过新 runner 注入精确 legacy/fixed transform 并调用同一 `primitive_action`，必须
停止并交主 Agent决定。

### 唯一主变量与 estimand

唯一 treatment 是 P51 frame transform：

```text
frame_fixed:
v_B = Rz(theta_A - theta_B) * clip_norm(2 * e_A, velocity_max)

frame_legacy:
v_B = clip_norm(2 * e_A, velocity_max)
```

estimand 限定为：

> 自然支持 latency 下、P41 acquisition 产生非空候选、冻结 selector index 有效、B2
> 起点相对 yaw 至少 `pi/6` 且首个 B2 proposed action bytes 在两个 role 间不同的
> Episode 中，frame-fixed 相对 legacy 对 B2 tool-to-preposition-target convergence
> 的效果。

不得外推到全部支持域或无候选 Episode。

### 两阶段 candidate bank

#### 阶段一：当前设计提交

当前提交冻结：

- `plan_id = R0001-P51-E1-formal`
- 12-way cell；
- 每 cell 3 个 eligible pair，共 36；
- 每 cell 最多 64 个 latency-matched acquisition/prefix；
- 每 cell 最大原始 `candidate_ordinal = 767`，即最多检查 768 个 raw seed；
- 全实验最大检查 9,216 个 raw seed、768 个 MuJoCo acquisition/prefix；
- eligibility、endpoint、统计、停止与 claim boundary；
- salt commitment。

#### 阶段二：bank 提交

实施提交通过门禁后才读取 salt。按 cell 和 raw ordinal 顺序：

1. 纯 sampler 做 natural latency rejection；
2. 对 latency-matched seed 运行 treatment-free acquisition 与 B0/B1；
3. 依次检查：
   - acquisition/prefix 无 failure、hard safety violation 或 terminal；
   - candidate set 非空；
   - frozen selector index 有效；
   - B2 起点 `abs(wrap(theta_B-theta_A)) >= pi/6`；
   - 首个 B2 `frame_fixed` 与 `frame_legacy` proposed action bytes 确实不同；
4. 每 cell 接受前 3 个，其余不运行。

bank artifact 必须发布并提交：

- salt reveal 与 commitment verification；
- 全部 raw seed 的 latency audit；
- 全部 latency-matched seed 的 eligibility result/reason；
- environment/policy seed、planned/pair ID；
- candidate canonical bytes/hash、selected index 与 selected record；
- acquisition input hashes；
- B0/B1 proposed/applied trace hash；
- B2 起点完整 continuation identity：
  - MuJoCo model/data state；
  - actuator/servo targets；
  - action latency queue；
  - observation latency queue；
  - policy history/availability；
  - timestamp/sequence 与 runtime/safety counters；
- role order，由独立 committed domain seed 决定。

若任一 cell 在 64 个 latency-matched acquisition 或 768 个 raw seed 内不足 3 个，整体
`inconclusive_design_infeasible`，不生成部分正式结果，不放宽条件、不减少 cell、不补
已看过的 seed。

bank commit 后禁止替换 seed、candidate、index、prefix、role order 或 endpoint。P50-E3
entity truth、contact、reward、success 和任何 B2 结果不得用于 bank。

### 正式 branch

- 每 pair 从同一 committed continuation identity 分叉，或分别确定性重放并验证上述全部
  identity 一致。
- 两个 role 只运行 B2 100 control step，B2 后立即停止。
- `d_0` 在第一个 B2 action 前测量。
- `d_1...d_100` 在每个 B2 control step 完成后测量。
- evaluator-private真实 grasp-center sites 转到 acquisition frame，只用于距离。
- 左右 frozen preposition target 必须从 bank candidate 和未修改 P41 公式重建。
- 普通 terminal 后，用最后一个 finite distance 对称 carry-forward 到 `d_100`。
- action latency 使 applied action 暂时相同时仍保留；这是 efficacy outcome，不是
  eligibility rejection。

每个 pair：

```text
normalized_AUC_role =
  mean(d_1 ... d_100) / max(d_0, 0.05m)

delta_i =
  normalized_AUC_legacy - normalized_AUC_fixed
```

正值有利于 fixed。

### 连续主分析

- 36 个独立 pair；每 cell 3 个。
- 先求每个 cell 的 3 个 `delta_i` 均值，再对 12 cell 等权求总体 point estimate。
- paired stratified bootstrap：
  - 10,000 replicates；
  - seed `20265102`；
  - 每个 replicate 在每个 cell 内有放回抽样 3 个完整 pair；
  - 12 个 cell 等权聚合；
  - 单侧 95% lower 为 bootstrap distribution 的 `0.05` quantile；
  - NumPy linear quantile method；
  - 非有限 replicate 使分析 invalid。
- 连续 MDE/接受门：
  - point estimate `>=0.10`；
  - one-sided 95% lower `>0.0`；
  - 三个 task 各自等权 cell mean `>0.0`；
  - observation latency 1/2 各自等权 mean `>0.0`；
  - action latency 1/2 各自等权 mean `>0.0`。

不得宣称单 cell 显著，不做多重探索性显著性选择。

### 二项与机制守护

`frame_fixed_win=1` 当且仅当：

- `delta_i >=0.10`；
- `d_100_fixed <= d_100_legacy`；
- pair identity 与 hard guard 有效。

稳健性门：

- 总 wins `>=24/36`；
- 每 task `>=6/12`；
- 每个 latency combination `>=4/9`。

完整发布：

- 每 pair/arm/task/cell 的 `d_0`、`d_1...d_100`、AUC、endpoint、minimum；
- 前 10 个实际 applied 非零 arm step 的 signed derivative；
- proposed/applied action hashes、RMS 与 active dimensions；
- first treatment step bytes 差；
- task/observation/action latency 分账。

### Hard guards 与停止

必须全部满足：

- pair 的 seed、candidate、B0/B1 trace 和完整 continuation identity 相同；
- 首个 treatment step 除左右臂线速度 xy 外，其余 action 字段相同；
- treatment bytes 确实不同且 arm action 非塌缩；
- action bounds 有效；
- supported stale action applied count 为 0；
- severe collision、invalid force、nonfinite metric、P40 conservation violation 为 0；
- safety、cap、gripper、phase、target、FK、backend identity 不变；
- 全部 36 pair 发布，无 duplicate、replacement 或 complete-case deletion。

停止规则：

1. bank infeasible：`inconclusive_design_infeasible`，不启动正式 branch；
2. artifact corruption、source/config drift、evaluator leakage、candidate/prefix/continuation
   identity 不一致：立即停止，`invalid`；
3. 观测到有效 hard safety violation：立即停止，候选 `rejected`，保留已完成 terminal；
4. 普通 runtime terminal：作为 outcome 对称 carry-forward，不补 seed；
5. 基础设施中断：只允许恢复同一 committed pair；不可恢复则 unresolved，整体
   `inconclusive`；
6. 禁止 efficacy/futility peek，禁止 24→36 扩样。

### 正式命令与产物

实现必须提供两个阶段：

```text
.venv/bin/python -m hwr.apps.evaluate_cartesian_convergence \
  --mode build-bank \
  --output runs/research-loop/0010/r0010-p51-e1-bank-s20265101 \
  --salt-file runs/research-loop/0010/.host/p51-e1-salt.txt

.venv/bin/python -m hwr.apps.evaluate_cartesian_convergence \
  --mode evaluate \
  --bank runs/research-loop/0010/r0010-p51-e1-bank-s20265101/bank.json \
  --output runs/research-loop/0010/r0010-p51-e1-convergence-s20265101
```

bank 提交前禁止运行 `--mode evaluate`。

bank 产物至少含：

- `seed-audit.json`
- `bank.json`
- `report.json`
- `manifest.json`

正式产物至少含：

- `terminals.json`
- `report.json`
- `manifest.json`
- 失败时 `failure.json`

### 判定

判定顺序：
1. invalid/provenance/identity/leakage；
2. unresolved/infrastructure；
3. hard safety；
4. 连续主门；
5. 二项与分层稳健性门。

全部通过：

`accepted as paired physical Cartesian convergence evidence`

合同有效且连续主门或稳健性门失败：

`rejected`

设计无法构建 36-pair bank或不可恢复基础设施缺失：

`inconclusive`

任何结果都不得声明 candidate 正确、接触、抓取、任务成功、学习、泛化、deployment 或
硬件安全改善。

## 实施、提交与运行顺序

1. 提交本冻结文档。
2. 主 Agent 建立两个 worktree，分配互斥新文件。
3. 两个实施 Agent分别完成 P50-E1/E2 与 P51-E1，运行 focused tests 并各自原子提交。
4. 主 Agent检查提交、合并到 `feat/research-loop`，运行 focused 与全局门禁。
5. 运行 P50-E1 正式 24-Episode cohort；通过后运行 P50-E2 离线分析。
6. 运行 P51 `build-bank`；只有 bank 完整时提交 bank artifact index。
7. 确认 bank commit 为正式 evaluate source 的祖先后，运行 P51 36-pair B2 对照。
8. 汇总 `04-results.md`、`05-summary.md`；本轮不启动训练。

## 全局门禁

```text
.venv/bin/python scripts/check_python_size.py
.venv/bin/python scripts/check_architecture.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m compileall -q src tests scripts
git diff --check
```

正式结果必须记录：

- 所有 focused 和全量测试；
- source commit 与冻结祖先；
- 历史 round tree；
- artifact bytes/hash；
- 全部 planned/rejected/unresolved seed；
- wall time、CPU/device、峰值内存和磁盘；
- `training_executed=false`
- `policy_inference_executed=false`
- `capability_claim_allowed=false`
- `task_success_claim_allowed=false`
- `generalization_claim_allowed=false`
- `hardware_safety_claim_allowed=false`
