# R0008 冻结实验

## 冻结状态

- 本文件先冻结 `R0001-P40-E2`。
- `R0001-P41-E2` 仅为条件选择，尚未冻结、不得实现：
  - 只有 P40-E2 正式结果为 accepted；
  - 且 candidate set、selector、primitive、ITT、MDE、样本量和安全非劣门全部结果前冻结；
  - 才允许在本文件追加独立的 P41-E2 实验段。
- 本轮不启动 P47、P48、P44-E1、P32-E2、P49、P46-E1 或正式训练。

## `R0001-P40-E2`：任务实体—机器人部位接触图与实体级冲量总账

### 研究问题

在不改变 action、physics、safety、reward、termination、success 或 policy observation 的
前提下，能否建立一个 evaluator-private、可守恒、可反例验证的接触图，使未来交互诊断能
区分：

1. base、left arm 与 right arm；
2. 具体 manipulated object、target container、floor/support 与 articulation；
3. 左右臂接触同一对象、左右臂接触不同对象和单臂接触；
4. robot–environment contact 与 task-relevant world–world contact；
5. 同一控制周期内的 contact-associated entity motion 与无接触运动。

本实验只验证测量合同。即使通过，也不把接触同期运动称为 controlled motion、action
causality、任务能力或安全改善。

### 冻结实现范围与负责人

- 唯一负责人：P40-E2 实施 Agent。
- 工作分支：`feat/research-loop` 当前分支上的原子实现提交。
- 允许修改：
  - 新增 `src/hwr/adapters/mujoco/entity_contact_graph.py`；
  - `src/hwr/adapters/mujoco/__init__.py`；
  - 新增 `src/hwr/apps/evaluate_entity_contact_graph.py`；
  - 新增 `tests/test_entity_contact_graph.py`；
  - 新增 `tests/test_entity_contact_graph_app.py`。
- 原则上不修改：
  - `src/hwr/adapters/mujoco/formal_household_backend.py`；
  - `src/hwr/adapters/mujoco/contact_ledger.py`；
  - `src/hwr/adapters/mujoco/bindings.py`；
  - `configs/adapters/mujoco/formal_3d_v1.json`；
  - 任何训练、policy、reward、safety 或 success 文件。
- 若实现发现必须越过上述边界，停止并交主 Agent 重新冻结，不自行扩展。

### 冻结身份与命名

- proposal：`R0001-P40-E2`
- report schema：`hwr.entity-contact-graph-contract-report/v1`
- artifact schema：`hwr.entity-contact-graph-contract-artifacts/v1`
- measurement schema：`hwr.mujoco-entity-contact-graph/v1`
- robot part：
  - `base`
  - `left_arm`
  - `right_arm`
- robot part body root：
  - `base -> robot_base`
  - `left_arm -> left_shoulder_pan_link`
  - `right_arm -> right_shoulder_pan_link`
- 每个 robot geom 沿 body parent 链选择最近的上述 root；全部 robot geom 必须恰好解析到一类。
- environment entity：
  - `floor_support:<geom-name>`
  - `manipulated_object:<object-id>`
  - `target_container:<geom-name>`
  - `articulation:<articulation-id>`
  - `forbidden:<geom-name-or-id>`
- task-relevant world–world edge 只记录：
  - manipulated object–floor/support；
  - manipulated object–target container；
  - manipulated object–articulation；
  - 两个不同 manipulated object。
- 其他 world–world 与 robot self-contact 继续单独计数，但不伪装成任务实体边。

### 冻结聚合语义

- 每个 physics substep：
  1. 读取所有 contact point；
  2. 每个 contact point 调用 `mj_contactForce` 读取 normal force；
  3. 先按无序 geom pair 求和；
  4. robot–environment pair 唯一映射为
     `robot part × environment entity`；
  5. task-relevant world–world pair 唯一映射为无序 entity pair；
  6. 再更新 substep peak、control-period impulse 与 Episode cumulative impulse。
- `normal_force is missing/nonfinite/negative` 一律计数并 fail-closed。
- robot–environment 子账按 P40 role 聚合后必须与并行启用的 P40-E1 ledger 完全一致：
  - pair peak；
  - category peak；
  - cumulative impulse；
  - contact duration；
  - contact point count；
  - unique pair observation count。
- world–world 子账单独报告，不参与上述 P40 守恒式。
- 同一 entity 的左右臂接触：
  - `same_substep_dual_arm_contact` 只在同一 physics substep 内 left arm 和 right arm
    都接触该 entity 时成立；
  - `distinct_entity_dual_arm_contact` 单独报告，永不折算成 same-entity 正例。
- manipulated object grasp：
  - 复用现有 `GraspContactMonitor` 语义；
  - 每只臂必须该臂两个 gripper pad 同时接触同一 object 才为 grasp-qualified；
  - same-object dual-arm grasp 要求同一 physics substep 两只臂均对同一 object
    grasp-qualified。
- entity motion：
  - manipulated object 使用同一 control period 起止 collision geom 世界坐标的欧氏距离；
  - articulation 使用同一 control period 起止 joint position 的绝对差；
  - 只报告 `contact_associated_motion`，不使用 `controlled_motion` 命名；
  - motion 只有在同一 entity 的有效 robot contact 于该 control period 内出现时才关联；
  - 无接触运动、reset settling 和接触结束后下一 period 的惯性运动不得关联；
  - 报告原始位移，不在 P40-E2 中使用 `0.01m` 作为能力或接受阈值。

### 冻结纯逻辑 fixture

固定构造下列反例并要求精确分类：

1. left/right 同时接触同一 object；
2. left/right 同时接触不同 object；
3. 只有一臂接触 object；
4. base/floor-only contact；
5. object 在无 robot contact 时移动；
6. object 在上一 period 接触、本 period 无接触时惯性移动；
7. arm/articulation handle contact 与 articulation position change；
8. object/target-container impact；
9. object/floor-support contact；
10. 同一 geom pair 的多个 contact point；
11. missing、NaN、Inf 与 negative normal force；
12. 映射缺失、重叠、未知 root、未知 entity role。

接受要求：

- task-relevant edge classification precision/recall 均为 1；
- same-entity、distinct-entity、single-arm 互不混淆；
- contact-associated motion 正例、无接触和惯性反例全部正确；
- pair 去重后 force/impulse 数值与手算一致；
- 所有无效证据 fail-closed。

### 冻结正式 trace

- 任务：
  - `tidy_living_room_3d/v1`
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
- seed：
  - base `20264002`
  - task stride `104729`
- 每任务 32 control step。
- action：reset 后固定 hold，保持当时左右 gripper position；无 policy inference。
- camera：reset 取得初帧后关闭 rendering。
- 每任务运行：
  - P40-E1 与 P40-E2 均 disabled 的 control trace；
  - P40-E1 与 P40-E2 均 enabled 的同 seed trace。
- legacy trace 必须逐字段 bit-identical：
  - applied action；
  - proprioception；
  - reward；
  - terminated/truncated；
  - result success/reason；
  - severe collision；
  - maximum forbidden force/pair；
  - safety intervention。
- 正式结果必须发布：
  - 完整 mapping；
  - 每 task robot–environment graph；
  - 每 task task-relevant world–world graph；
  - 每 period contact 与 entity motion；
  - P40 聚合守恒差；
  - ignored/invalid counts；
  - physics timestep、solver、iterations、tolerance 与 substeps。

### 接受标准

全部满足才标记：

`accepted as entity-contact measurement contract evidence`

检查：

1. 三任务 robot geom 全部且唯一解析到 base/left/right。
2. 三任务 object、allowed role 和 articulation entity 全部可解析且无冲突。
3. 全部纯逻辑 fixture 通过。
4. 三任务 enabled/disabled legacy trace bit-identical。
5. P40-E2 robot–environment 子账聚合回 P40-E1 的全部冻结字段，绝对差 `<=1e-12`。
6. world–world 子账不进入 P40 守恒式。
7. formal trace 与 fixture 的 missing/nonfinite/negative count 均为 0。
8. same-entity dual-arm、distinct-entity dual-arm、single-arm 和 no-contact motion 分账完整。
9. report 固定：
   - `measurement_only=true`
   - `policy_inference_executed=false`
   - `closed_loop_capability_episode_executed=false`
   - `capability_claim_allowed=false`
   - `hardware_safety_claim_allowed=false`
   - `action_causality_claim_allowed=false`
   - `legacy_runtime_behavior_unchanged=true`
10. report、manifest、命令、source commit、binding identity 与所有直接产物 hash/bytes
    完整可重建。

若映射、fixture、守恒、fail-closed、bit-identity 或 provenance 任一失败：

- 可明确否定合同则 `rejected`；
- 环境/API 无法提供所需证据则 `inconclusive`；
- 均停止 P41-E2，不修改门槛。

### 稳定命令与产物

正式 run：

`runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002`

命令：

```text
.venv/bin/python -m hwr.apps.evaluate_entity_contact_graph \
  --output runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002
```

必须原子写入：

- `report.json`；
- `manifest.json`；
- 失败时 `failure.json`。

manifest 必须记录：

- proposal 与 schema；
- source commit；
- 冻结文档提交为祖先；
- 稳定命令；
- binding hash/bytes；
- robot body-root identity；
- MuJoCo version 与 physics 配置；
- 所有直接产物 hash/bytes；
- 禁止声明位。

## 实施后门禁

先运行 focused tests，再运行：

```text
.venv/bin/python scripts/check_python_size.py
.venv/bin/python scripts/check_architecture.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m compileall -q src tests scripts
git diff --check
```

正式 run 前必须：

1. 本冻结文档提交是实现提交祖先；
2. 实现已原子提交；
3. 工作区干净；
4. focused tests 与共同门禁通过；
5. binding hash 与冻结提交一致；
6. 历史 `docs/research-loop/0001/`～`0007/` tree 零差异。

正式 run 后、P41-E2 冻结前必须运行全量：

```text
.venv/bin/pytest -q
```

## 本阶段停止条件

- P40-E2 不是能力实验，不启动正式训练，不使用 MPS，不休眠。
- P40-E2 未接受：记录结果并停止 P41-E2。
- P40-E2 接受：只授权追加 P41-E2 的结果前冻结合同，不自动授权实现或正式 run。

## `R0001-P41-E2`：共享 RGB-D 候选集上的 target-index-only 交互诊断

### 前置与研究问题

- P40-E2 已在提交 `f5c65ca220e95dce6fc4b332983e3b6852098b57` 记录为
  `accepted as entity-contact measurement contract evidence`。
- 最终 P40-E2 report：
  `runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002/report.json`
- report SHA-256：
  `987a2217cf9f5c6eb08b018b3bf13164917c75bccbfa77140a8786c80987f841`
- manifest SHA-256：
  `fdb847a41f55a7a3bb362d650baa2d131e2a5178ac73166336d57368ba60546b`
- P40-E2 数值顺序修复提交：
  `8b79597a6af36233b1a2e6437c769db12567f88c`。该修复只令 episode category
  与 P40-E1 一样按 control period 累加，未改变 P41 的候选、score、phase、primitive、
  阈值、salt、样本量或判定标准。旧 `cefb240` 产物和首次 provenance 拒绝均保留为
  superseded 证据；本节在新 smoke Episode 执行前重绑定上述固定哈希。
- P41-E2 研究问题：

> 在相同 policy-visible 几何候选集、相同采集阶段、相同双臂 assignment、相同 Cartesian
> primitive、相同安全层和相同物理预算下，只改变候选目标 index 的选择方法，能否提高同一
> manipulated object 或 articulation 上双臂接触并伴随同期实体运动的 Episode 产率？

唯一实验变量：

- candidate：选择冻结几何 score 最大的候选 index；
- control：通过独立 policy RNG 和 candidate-set hash 选择 hash-uniform index。

除 target index 外，候选生成、候选顺序、acquisition、导航、双臂 assignment、动作、
gripper、phase 时长、撤回、停止、environment seed、policy seed 和评测均相同。

### 冻结实现范围与负责人

- 唯一负责人：P41-E2 实施 Agent。
- 工作分支：`feat/research-loop` 当前分支上的原子实现提交。
- 允许修改：
  - 新增 `src/hwr/eval/target_selection.py`；
  - 新增 `src/hwr/eval/target_selection_safety.py`；
  - 新增 `src/hwr/adapters/mujoco/target_selection_diagnostic.py`；
  - 新增 `src/hwr/apps/evaluate_target_selection.py`；
  - 新增 `tests/test_target_selection.py`；
  - 新增 `tests/test_target_selection_diagnostic.py`；
  - 新增 `tests/test_target_selection_app.py`。
- 原则上不修改现有 backend、contact graph、policy、训练、task、reward、safety、success、
  配置或 package export。
- app 可直接从新模块导入，避免为导出符号扩大所有权。
- 实现审查发现 force/impulse 非劣与 MuJoCo bridge 合并会超过单文件 800 行门禁；因此在
  行为实现提交前冻结新增上述纯统计模块。该拆分不改变输入、统计、门槛或运行范围。
- 若必须越过上述边界，停止并交主 Agent 重新冻结。

### 冻结身份

- proposal：`R0001-P41-E2`
- input schema：`hwr.p41-target-index-input/v1`
- candidate schema：`hwr.p41-target-candidates/v1`
- plan schema：`hwr.p41-target-selection-plan/v1`
- terminal schema：`hwr.p41-target-selection-terminal/v1`
- report schema：`hwr.p41-target-selection-report/v1`
- manifest schema：`hwr.p41-target-selection-artifacts/v1`
- power schema：`hwr.p41-target-selection-power/v1`
- smoke salt：`R0001-P41-E2-smoke-s20264101`
- formal salt：`R0001-P41-E2-formal-s20264102`
- power seed：`20264102`
- force/impulse bootstrap seed：`20264142`

### Policy-visible 输入

每个控制步只允许：

| 字段 | 类型/形状 | 用途 |
|---|---|---|
| `observation_timestamp_ns` | `int64` | action validity 与 frame age |
| `sequence_id` | `int64` | 单调性 |
| `head_rgb_uint8` | `[192,256,3]` | 只保存/hash；v1 算法不读取 RGB 数值 |
| `head_depth_m` | `[192,256] float32` | 候选与障碍 |
| `head_depth_valid` | `[192,256] bool` | 有效 mask |
| `head_camera_intrinsics` | `[4] float64` | `(fx,fy,cx,cy)` |
| `robot_from_head_camera` | `[4,4] float64` | 深度反投影 |
| `proprioception` | `[37] float64` | base、joint、gripper |
| `executed_action_history` | `[4,16] float64` | 确定性控制状态 |
| `history_available` | `[4] bool` | history mask |
| `safety_state` | enum | fail-closed |
| `phase_index/phase_step` | `int32` | 控制器内部状态 |
| `policy_rng_seed` | `uint64` | control uniform index |

- acquisition frame `A` 是 reset 时机器人 base frame。
- 后续帧仅用 visible `base_pose=(x,y,yaw)` 计算 `T_A<-B_t`。
- robot FK 只使用 visible joint position 和冻结机器人运动学模型。
- 数组按 C-contiguous little-endian 序列化；内部 float 为 `float64`，最终输出 canonical
  Python float tuple。

禁止输入：

- `task_id`、instruction、language embedding；
- object/body/geom/site ID 或颜色类别；
- simulator object pose；
- reward、success、task stage、target mapping；
- P40/P40-E2 audit、contact truth、force、entity motion；
- expert waypoint/action；
- future/latest observation；
- 真实 latency label。

selector 与 primitive 只能接收上述序列化输入，不持有 backend、audit 或 evaluator 对象。

### 共享 acquisition

reset 单帧不作为候选输入。冻结的只读 feasibility 检查显示：

- 餐桌对象初帧可见，但距最近夹爪约 3.14–3.67m；
- 厨房对象初帧可见，但距最近夹爪约 2.14–2.33m；
- 客厅鸭子只在少量 reset 初帧可见；
- 客厅足球在所查 reset 初帧不可见。

因此两个 role 在 selector 分流前执行完全相同的 acquisition：

1. A0 stable：
   - 10 step；
   - base、双臂 twist 全 0；
   - 双 gripper target 0。
2. A1 panorama：
   - 最多 380 step；
   - base linear 0、angular `+0.35rad/s`；
   - 双臂 twist 0、gripper 0；
   - visible base yaw 解包累计达到 `2pi` 后 hold；
   - 首次越过 `k*pi/12` 时记录 keyframe，`k=0..23`；
   - 未达到阈值不得补帧。
3. A2 forward：
   - 最多 220 step；
   - base linear `+0.12m/s`、angular 0；
   - 沿 acquisition 初始朝向达到 `1.20m` 后 hold；
   - 当前有效 depth 投影到当前 robot frame；
   - 障碍 corridor：
     `x in [0.20,0.65]m`、`abs(y)<=0.38m`、`z in [-0.18,1.15]m`；
   - 连续 3 帧各至少 25 个有效点进入 corridor 时永久停止前进；
   - 不绕障碍、不转向、不搜索替代路径。
4. A3 panorama：
   - 与 A1 相同，最多 380 step、24 keyframe。
5. A4 seal：
   - hold 5 step；
   - 原子封存 keyframe、输入 hash 和 candidate bytes/hash；
   - 此时才计算两个 selector index。

acquisition 期间任一项发生即 `acquisition_failed`：

- safety intervention；
- nonfinite input；
- camera/calibration 缺失；
- sequence/timestamp 非单调；
- supported 条件 source age `>100ms`；
- P40-E2 主事件发生。

失败后两个 role 剩余 horizon 均 hold，主事件记 0，保留在 ITT；不补 seed、不重试、不加入
任务专属导航。

### 候选生成

只使用 A1/A3 冻结 keyframe。每帧：

1. depth 必须为 float32，范围 `0.10m..5.00m`，camera valid；
2. supported 主分析要求 visible source age `<=100ms`；
3. 排除图像边缘 12px；
4. row-major、stride 4px 枚举 anchor；
5. 中心窗 `5x5`，至少 20/25 valid；
6. 外环为 `21x21` 去除中心 `9x9`，至少 240/360 valid；
7. `z_c = median(center valid depth)`；
8. `z_b = median(annulus valid depth)`；
9. `prominence = z_b - z_c`；
10. 保留：
    - `0.025m <= prominence <=0.45m`；
    - center depth `q90-q10 <=0.04m`。

局部 patch：

- 在 `21x21` 内保留
  `abs(depth-z_c) <= max(0.025m, 0.015*z_c)`；
- 至少 24 点；
- 由 frame intrinsics 反投影，经 `robot_from_head_camera` 和 visible base pose 转到
  acquisition frame；
- 只用 visible joint position 与冻结机器人模型做 self-mask，距任一 robot collision
  surface `<0.06m` 删除；
- center 在 acquisition frame 的 `z in [-0.18,1.30]m`；
- center 相对采样时 base 的水平距离为 `[0.35,4.00]m`。

patch PCA：

- 最小特征值方向为 normal，normal 朝向采样相机；
- `lambda_min / sum(lambda) <=0.12`；
- 两个切向的 `q95-q05` 为 `w1,w2`，`w=max(w1,w2)`；
- `0.035m <=w<=0.40m`。

raw candidate 只含：

- acquisition-frame center；
- normal；
- width；
- prominence；
- support point count；
- frame ordinal、row、column。

跨视点去重：

- center distance `<=0.08m`；
- normal cosine `>=0.80`；
- width difference `<=0.10m`；
- 对满足边构图并取 connected components；
- component center 为逐坐标 median；
- normal 为归一化向量和；
- width 为 median；
- prominence 为 maximum；
- support 为 sum；
- view count 为不同 keyframe 数；
- 只保留 view count `>=2`。

最多保留 64 个，截断排序键：

```text
(-support_count, -view_count, -prominence_um,
 quantized_x_mm, quantized_y_mm, quantized_z_mm,
 first_frame, first_row, first_column)
```

canonical 顺序：

```text
(x_mm, y_mm, z_mm,
 normal_x_1e4, normal_y_1e4, normal_z_1e4,
 width_mm, first_frame, first_row, first_column)
```

- position、prominence、width 量化到 1mm；
- normal 量化到 `1e-4`；
- canonical bytes 包含 schema、acquisition input hashes、candidate count 和全部整数；
- empty set 也有唯一 bytes；
- `candidate_set_sha256 = SHA256(canonical_bytes)`；
- 两 role 的 bytes/hash 不同：pair 为 `unresolved_infrastructure`，整项 `inconclusive`。
- `K=0`：index `-1`，后续全部 hold，主事件 0，保留 ITT；不回退随机动作。

### Selector

candidate score：

```text
s_prom = clip((prominence - 0.025) / 0.125, 0, 1)
s_size = exp(-0.5 * ((width - 0.14) / 0.08)^2)
s_view = min(view_count / 4, 1)
s_iso = clip((nearest_candidate_distance - 0.08) / 0.32, 0, 1)
s_range = clip((3.0 - final_base_horizontal_range) / 2.0, 0, 1)
score = 0.30*s_prom + 0.25*s_size + 0.20*s_view + 0.15*s_iso + 0.10*s_range
```

- `K=1` 时 nearest distance 固定 0.40m；
- candidate 取最高 score，完全相同取最小 canonical index；
- 不扫描权重或阈值。

control 使用 rejection-sampled hash-uniform：

```text
domain = "R0001-P41-E2|uniform-index-v1"
digest_j = SHA256(domain || policy_rng_seed || candidate_set_sha256 || counter=j)
L = floor(2^256 / K) * K
```

从 `j=0` 起取首个 `int(digest_j)<L`，index 为 `int(digest_j) mod K`。

- condition/role 不进入 environment seed、policy seed 或 candidate-set hash；
- candidate selector 不消费 control RNG；
- control 不读取 score；
- `K=1` 或两 selector 恰选同 index 时作为 concordant ITT pair 保留。

### 双臂 assignment 与固定 primitive

对选中 candidate：

- `p`：candidate center；
- `b`：acquisition 结束时 visible base center；
- `f=normalize((p_x-b_x,p_y-b_y,0))`；
- `n=-f`，从 target 指向 robot；
- `l=(-f_y,f_x,0)`；
- `z=(0,0,1)`；
- 若水平距离 `<0.35m`，candidate invalid，按 empty 处理；
- `s_pre=clip(width+0.12,0.18,0.34)`；
- `s_contact=clip(width+0.04,0.10,0.24)`；
- left 固定 `+l`，right 固定 `-l`，不因 task、entity、contact 或 IK error 换边。

目标：

```text
left_pre  = p + 0.18*n + 0.5*s_pre*l + 0.05*z
right_pre = p + 0.18*n - 0.5*s_pre*l + 0.05*z
left_contact  = p + 0.015*n + 0.5*s_contact*l
right_contact = p + 0.015*n - 0.5*s_contact*l
```

20Hz primitive：

| phase | 最大 step | base | 双臂 | `v_max` | gripper |
|---|---:|---|---|---:|---|
| B0 定向 | 100 | `v=0`，`omega=clip(heading_error, +-0.35)` | hold | 0 | 0 |
| B1 接近 | 300 | `v=clip(0.6*(range-0.85),0,0.12)`；`abs(heading)>0.35` 时 `v=0`；`omega=clip(heading,+-0.25)` | hold | 0 | 0 |
| B2 预位 | 100 | hold | pre targets | `0.08m/s` | 0 |
| B3 接触趋近 | 50 | hold | contact targets | `0.03m/s` | 0 |
| B4 闭合 | 20 | hold | contact targets | `0.02m/s` | `0.75*(j+1)/20` |
| B5 同步拉动 | 30 | hold | contact targets `+0.08*n+0.02*z` | `0.04m/s` | 0.75 |
| B6 撤回 | 50 | hold | pre targets `+0.05*n` | `0.06m/s` | 前30步0.75，后20步线性到0 |
| B7 停止 | 10 | hold | zero twist | 0 | 0 |

工具位置仅由 visible joints 和冻结 FK 得到。每臂平移控制：

```text
v = clip_norm(2.0 * (target-current_tool), v_max)
normalized_xyz = v / 0.30
angular_twist = (0,0,0)
```

- action frame 以 visible observation timestamp 创建；
- validity 固定 `+100ms`，不得延长；
- 除 runtime terminal、invalid input 或 safety fail-closed，不按 contact/reward/audit 提前换
  phase；
- 首次 safety intervention 后剩余 planned horizon 全部 hold；
- runtime terminal 后停止 physics；未执行剩余 slot 仍记录 failure；
- 任一臂不可达不改派；
- acquisition 995 step，post-selection 660 step，总 planned horizon 1,655 step/branch。

### Same-index bit identity

1. 纯函数：相同 serialized input、candidate bytes、index、phase state 和 action history，两个
   role 输出 16-D action bytes 相同。
2. 正式前 twin-run：每个 task 至少两个、不用于正式结果的 seed；强制同 index；acquisition、
   proposed/applied action、proprio、event、termination 和 P40-E2 trace bit-identical。
3. 正式自然 same-index pair：整条 trace 必须 bit-identical；仍保留 ITT。

role label 不得传入 selector 后的 primitive。

### 主事件、ITT 与分账

主事件名：

`same_entity_dual_arm_contact_associated_motion`

Episode event 为 1 当且仅当存在 entity `e`：

1. `e` 为 manipulated object 或 articulation；
2. 同一 physics substep 内：
   - manipulated object 要求 left/right arm 均对同一 object 双垫
     grasp-qualified；
   - articulation 要求 left/right arm 均有效接触同一 articulation entity；
3. 形成连续 same-entity dual-arm contact control-period bout；
4. 从 bout 起点到任一 bout period 终点：
   - rigid entity world position displacement `>=0.01m`；或
   - articulation joint absolute change `>=0.01m`。

不计：

- 左右臂分别接触不同 entity；
- 单臂接触；
- floor/support、container-only 或 base contact；
- 无接触运动；
- 接触结束后下一 period 的惯性运动；
- acquisition 中发生的事件。

这是 contact-associated motion，不是 action-caused 或 controlled motion。

统计单位为 planned environment-seed pair。所有 planned pair 必须保留：

- empty candidate；
- acquisition failure；
- same index；
- FK/IK 不可达；
- no contact；
- safety intervention；
- stale action；
- early termination/timeout；
- 一方先结束。

主事件每 role 独立记 0/1。只有 host/process 意外终止、artifact/hash 损坏、无法重建 pair
identity 或不可归因异常为 `unresolved_infrastructure`。

```text
planned = valid_pair + unresolved_infrastructure
```

- 不允许 replacement seed 或 complete-case deletion；
- unresolved 大于 0 时只能 `inconclusive`；
- 轨迹写入独立 run，不进入普通 Replay。

supported primary：

- observation latency `1/2`；
- action latency `1/2/3`；
- 18 cell，每 cell 3 pair；
- 54 pair、108 branch Episode。

challenge descriptive：

- observation latency `3`；
- action latency `1/2/3`；
- 9 cell，每 cell 2 pair；
- 18 pair、36 branch Episode。

不调用 reset-only latency override 执行动作。使用 evaluation profile 的自然采样；plan builder
按 task/cell/replicate 的预提交顺序派生候选 seed，并只根据 evaluator-private sampled
latency pair 作确定性 rejection，直到填满 cell。每个 rejected seed 也写入 plan audit，不能
因其他 outcome 选择。role 不进入 seed。两个 role 使用同一 environment/policy seed。

latency 3 不进入主效应、MDE 或接受判定，只完整发布 challenge 的 empty、stale rejection
和失败；不得把 supported 结果称为完整 profile 支持。

### 功效与主要判定

对每个 supported pair：

| candidate | control | 计数 |
|---|---|---|
| 1 | 0 | `B` |
| 0 | 1 | `C` |
| 1 | 1 | concordant positive |
| 0 | 0 | concordant negative |

```text
D = B + C
p_exact = P[Binomial(D, 0.5) >= B]
delta_ITT = (B-C)/N
```

冻结 alternative：

- `p10=0.40`
- `p01=0.10`
- discordance `0.50`
- true ITT delta `+0.30`
- acceptance effect floor `+0.20`

功效 pipeline：

- candidate pair 数：36、54、72、90、108；
- 每 task/cell 等额；
- 10,000 trial；
- power seed `20264102`；
- 每 trial 同时执行 exact McNemar、`delta_ITT>=0.20`、三 task 方向非负、两个 supported
  observation-latency 层方向非负；
- null 至少覆盖：
  - `p10=p01=0.05`；
  - `p10=p01=0.25`；
- 选择第一个满足：
  - worst null FPR 95% Clopper-Pearson upper `<=0.05`；
  - planted power 95% Clopper-Pearson lower `>=0.80`；
  的 pair 数。

独立预检估计 36/54/72 pair 的 empirical power 约为 `0.758/0.867/0.896`，因此预计选择
54 pair；该估计不是正式结果，最终实现必须从冻结 seed 复算并写入 report。若正式 power
没有选择 54，实验在物理 run 前 `inconclusive_power`，不得自动改用更大样本。

接受主门：

1. 54 个 supported pair 全部 resolved；
2. one-sided exact McNemar `p<=0.05`；
3. discordant conditional exact 95% lower `>0.5`；
4. `delta_ITT>=0.20`；
5. 三 task 均 `B_task>=C_task`；
6. observation latency 1/2 均 `B_latency>=C_latency`。

不声称对 true 20pp effect 有 80% power；20pp 只是观察值接受下限。

### 安全与行为守护

硬门：

- missing/nonfinite/negative force 为 0；
- severe collision 为 0；
- supported 域 stale action 实际应用为 0；
- P40-E2 对 P40-E1 的聚合守恒误差 `<=1e-12`；
- candidate/control action bounds 与 canonical 16-D contract 通过；
- 非目标动作的 active dimension、executed RMS 与 route completion 全部发布。

非目标 allowed contact：

- 类别：floor/support、target container，以及没有进入主事件 entity 的 allowed contact；
- 每个 supported ITT Episode 计算 category peak force 与 cumulative impulse；
- 10,000 次 task/cell 分层 paired bootstrap；
- seed `20264142`；
- control/candidate 都为 0 时 ratio=1；
- control=0 且 candidate>0 时失败；
- point ratio `<=1.10`；
- one-sided 95% ratio upper `<=1.25`。

目标 object/articulation contact intensity：

- 不要求 candidate 总接触量低于 control；
- 对 role 内出现目标接触的 Episode 报告：
  - peak force；
  - `mean_contact_force = cumulative_impulse/contact_duration`；
- 每 role 总计至少 12 个 positive-contact Episode，且每 task 每 role至少 2 个，才计算
  task-stratified 10,000-bootstrap noninferiority；
- 比较 `log1p(force/1N)` 的 role mean：point ratio `<=1.25`，one-sided 95% upper
  `<=1.50`；
- 支持不足则正式结论 `inconclusive`，不得按 0 填充或删除无接触 Episode。

这些只是仿真内部相对接触强度守护，不是硬件安全阈值。全部 raw peak、impulse、duration 和
entity 分区必须发布。

### Smoke、正式命令与产物

smoke：

- 本节的精确 smoke 设计取代前文 Same-index bit identity 第 2 条中
  “每个 task × observation latency × action latency 至少 2 个 seed”的宽泛表述；
- 每 task 固定两个 supported 自然 latency cell：
  - `(observation latency=1, action latency=1)`；
  - `(observation latency=2, action latency=2)`；
- 每 cell 1 pair，共 6 pair、12 branch Episode；
- seed 只按自然 evaluation profile 的 sampled latency 作结果前 rejection，所有 rejected
  seed 均写入 plan audit；
- 强制 same index twin-run；
- 只验证 candidate-set nonempty、contract、bit-identity、运行时和产物；
- 不比较 selector 优劣；
- smoke 后如修改任何阈值、score、phase、primitive、MDE 或样本量，必须先追加新冻结提交并
  更换正式 salt。

```text
.venv/bin/python -m hwr.apps.evaluate_target_selection \
  --output runs/research-loop/0008/r0008-p41-target-selection-e2-smoke-s20264101 \
  --salt R0001-P41-E2-smoke-s20264101 \
  --smoke
```

正式：

```text
.venv/bin/python -m hwr.apps.evaluate_target_selection \
  --output runs/research-loop/0008/r0008-p41-target-selection-e2-s20264102 \
  --salt R0001-P41-E2-formal-s20264102
```

必须原子写入：

- `report.json`；
- `plan.json`；
- `terminals.json`；
- `power.json`；
- `manifest.json`；
- 失败时 `failure.json`。

manifest 必须绑定：

- source commit 与本 P41 冻结提交祖先；
- P40-E2 report/manifest identity；
- task/binding/config identity；
- seed commitment/reveal；
- policy input schema、candidate schema 与 primitive constants；
- planned/rejected seed audit；
- 每个 terminal 与直接 artifact hash/bytes；
- 全部禁止声明位。

### 判定

`accepted as target-selection interaction-yield evidence` 仅当：

- P40-E2 identity 与接受状态通过；
- frozen power 选择 54 supported pair；
- 54 supported 与 18 challenge pair 全部 resolved；
- candidate hash、seed/role 隔离、same-index bit-identity 全部通过；
- 主统计六项门全部通过；
- safety、stale action、P40 守恒、非目标 allowed-contact 与目标接触强度守护全部通过；
- challenge 完整发布。

判定优先级：

1. lineage、泄露、role/seed、后验更改或删除 planned pair：`invalid`；
2. power 未选择 54、unresolved、候选/primitive 不可重建、目标接触强度支持不足：
   `inconclusive`；
3. 功效充分但主门或其他守护失败：`rejected`；
4. 全部通过：`accepted as target-selection interaction-yield evidence`。

任何结果都不授权 P47、世界模型训练、Actor 解锁或能力 benchmark；下一步仍需新一轮独立
提案与筛选。
