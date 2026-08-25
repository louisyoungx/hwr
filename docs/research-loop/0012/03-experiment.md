# R0012 冻结实验

## 冻结状态

- 状态：结果前冻结
- 冻结日期：`2026-08-25`
- 起始提交：`b397c10a63d623096281e6c88ed0a3ac63755cfd`
- 筛选输入 SHA-256：
  `b555c98be651077b759fec459c5c0f826c885bbea484e9ef69d088eeab99b450`
- 入选静态审计：`R0001-P61`
- 入选 evaluator mapping 合同：`R0001-P50-E4`
- 唯一物理诊断：`R0001-P60`
- 本轮不实施：`R0001-P62`、`R0001-P64`、`R0001-P65`
- 本轮拒绝：`R0001-P63`
- 正式训练：不授权
- P50 observation sidecar / Episode：不授权
- selector、Replay、Actor、世界模型：不授权

本文件提交后，不得根据运行结果修改 transition、role、alias、geom inventory、cohort、
seed、公式、frame、容差、阈值、分账、判定门或允许声明边界。评测实现缺陷只能修复合同
执行；若修复发生在读取结果后，必须保留旧 artifact、记录时序并使用同一输入原样重跑。

## 实验隔离

### `R0001-P61`

- 唯一负责人：实施 Agent P61。
- 分支：`feat/r0012-p61`。
- 文件所有权：
  - `configs/eval/interaction_contract_v1.json`（新）
  - `src/hwr/eval/interaction_contract.py`（新）
  - `src/hwr/apps/audit_interaction_contract.py`（新）
  - `tests/test_interaction_contract.py`（新）
- 禁止修改 runtime、task config、binding、candidate、primitive、动作、安全、reward、
  termination 和历史文档。

### `R0001-P50-E4`

- 唯一负责人：实施 Agent P50-E4。
- 分支：`feat/r0012-p50-e4`。
- 文件所有权：
  - `configs/eval/entity_candidate_aliases_v1.json`（新）
  - `src/hwr/adapters/mujoco/entity_candidate_mapping.py`
  - `src/hwr/apps/audit_entity_candidate_mapping.py`（新）
  - `tests/test_entity_candidate_mapping.py`（新）
- 历史 `evaluate_entity_candidate_coverage.py` 与 P50-E3 artifact 不得改写。
- 禁止修改 renderer、backend、task/binding、candidate、selector、动作、安全、reward、
  termination 和历史文档。

### `R0001-P60`

- 唯一负责人：实施 Agent P60。
- 分支：`feat/r0012-p60`。
- 文件所有权：
  - `src/hwr/eval/phase_entry_geometry.py`（新）
  - `src/hwr/adapters/mujoco/phase_entry_geometry.py`（新）
  - `src/hwr/apps/evaluate_phase_entry_geometry.py`（新）
  - `tests/test_phase_entry_geometry.py`（新）
  - `tests/test_phase_entry_geometry_app.py`（新）
- 可导入既有 P50/P51/P52/P57 helper，但不得修改这些 helper 或历史 source。
- 禁止修改 candidate、selector、primitive、phase、速度、gripper、安全、task/binding、
  reward、termination 和历史文档。

三个实施 Agent 写集不重叠。每项必须独立提交并引用提案 ID；主 Agent串行集成、门禁和
正式运行。

## 共同来源与完整性

三项正式 app 均需：

1. 要求工作区干净且 source commit 已提交；
2. 验证本文件首次提交是 source commit 的 ancestor；
3. 验证本文件 content/blob 与冻结提交一致；
4. 验证 `docs/research-loop/0001/`～`0011/` tree hash 与 `00-context.md` 一致；
5. 记录 source commit、命令、Python/MuJoCo/Numpy 版本、平台、输入文件 bytes/SHA-256；
6. output 已存在或 staging 已存在时拒绝覆盖；
7. 使用临时目录和原子 rename；
8. report 与原始证据分开；manifest 绑定全部 artifact；
9. claim flags 明确：
   - `training_executed=false`
   - `policy_inference_executed=false`
   - `capability_claim_allowed=false`
   - `task_success_claim_allowed=false`
   - `generalization_claim_allowed=false`
   - `hardware_safety_claim_allowed=false`
   - `entity_coverage_claim_allowed=false`

冻结历史 tree：见 `00-context.md`，不得修改 `0001`～`0011`。

## `R0001-P61`

### 研究问题

当前 P50 初始 acquisition 与 P51 generic B0–B7 primitive 的研究对象，究竟是完整家务
transition，还是单次 controlled surface-interaction microbenchmark？在不把 private task
stage 泄露给 policy 的前提下，能否为后续 evaluator 冻结唯一的 stage-compatible role？

### 固定 transition

每个 transition 必须有稳定 ID，并保存：

- `precondition`
- `allowed_entity_instance_or_role`
- `forbidden_roles`
- `interaction_type`
- `expected_state_change`
- `evaluator_predicate`
- `policy_visible_fields`
- `planner_call_state`
- `primitive_input_fields`
- `evaluator_private_fields`

冻结任务 transition：

| ID | Task | Precondition | Allowed target | Interaction | Expected state |
|---|---|---|---|---|---|
| `R0001-P61-T01` | living | duck 未稳定在 target | `object:duck` | pick-transport-place | duck 稳定进入 living target volume |
| `R0001-P61-T02` | living | football 未稳定在 target | `object:football` | pick-transport-place | football 稳定进入 living target volume |
| `R0001-P61-T03` | dining | cup 未稳定在 target | `object:cup` | pick-transport-place | cup 稳定进入 cup target volume |
| `R0001-P61-T04` | dining | plate 未稳定在 target | `object:plate` | pick-transport-place | plate 稳定进入 plate target volume |
| `R0001-P61-T05` | kitchen | drawer position `<0.30m` | `articulation:drawer` | articulate-pull | drawer position `>=0.30m` |
| `R0001-P61-T06` | kitchen | drawer `>=0.30m` 且 yellow 未放置 | `object:cleaner_yellow` | pick-transport-place | yellow 稳定进入 yellow target volume |
| `R0001-P61-T07` | kitchen | drawer `>=0.30m` 且 pink 未放置 | `object:cleaner_pink` | pick-transport-place | pink 稳定进入 pink target volume |

living 的 T01/T02 与 dining 的 T03/T04 没有额外顺序；kitchen T06/T07 都依赖 T05，但
彼此不规定顺序。不得从自然语言中的“依次”新增 success predicate 未要求的顺序。

### 四类信息边界

1. `evaluator_private`：
   - task transition ID；
   - object/articulation/target identity；
   - target-volume containment；
   - drawer requirement；
   - success/stability predicate。
2. `planner_call_state`：
   - 当前代码没有一个已验证、会向 P50/P51 primitive 提供 transition ID 的 planner；
   - 不得假定隐含 planner 已消解 role 歧义。
3. `policy_visible`：
   - RGB-D、动态 calibration、proprioception、history、phase、safety state、语言编码；
   - 不含 geom/body/site ID、transition ID、reward、success 或 private stage。
4. `primitive_input`：
   - 一个 selected `Candidate`；
   - acquisition/current base pose 与 policy-visible joints/history；
   - 不含 task target、destination target 或 transition ID。

### P50 初始 microinteraction target

本轮只为“reset 后第一次 P50 acquisition + 一次 generic candidate-centered primitive”
冻结 evaluator annotation：

- living：`object:duck` 或 `object:football`；
- dining：`object:cup` 或 `object:plate`；
- kitchen：仅 `articulation:drawer`；
- target-container、floor/support、other furniture、robot、site、unknown/mixed 均不兼容。

该 annotation 只允许用于未来 evaluator；不得进入 policy/candidate/selector/action。

### 审计判定

报告必须分别判断：

1. full-task contract：
   - 现有一次 selected candidate + generic B0–B7 primitive 是否能唯一表达并实现七个
     transition 的 target、interaction 和 destination。
2. initial microinteraction contract：
   - 是否能在 evaluator-only 边界为三任务唯一冻结上节 role；
   - 是否存在合法外部 planner 已在当前调用边界提供该信息。

`accepted as interaction-contract gap evidence`：

- 七个 transition 全部由版本化输入重建；
- 至少一个 full-task transition 的 destination/interaction 无法由当前 primitive input
  表达；或 kitchen 初始 role 无法由当前 caller/planner state 唯一提供；
- 反例检查证明这不是已存在外部 planner 合法消解。

`rejected`：

- 当前代码已有可执行 planner call state，能为每个 transition 提供无泄露 target；且
  primitive input/semantics 覆盖所需 interaction 与 destination。

`invalid`：

- transition 不能从冻结 task/binding/runtime predicate 重建；
- 依赖人工未版本化解释；
- 把 evaluator-private stage 写入正式行为。

### 命令、产物与成本

```text
.venv/bin/python -m hwr.apps.audit_interaction_contract \
  --contract configs/eval/interaction_contract_v1.json \
  --output runs/research-loop/0012/r0012-p61-interaction-contract-s20266101
```

产物：

- `transitions.json`
- `report.json`
- `manifest.json`

预算：wall time `<60s`，RSS `<1GiB`，artifact `<10MiB`。纯 CPU 静态审计，不休眠。

允许声明仅为 evaluator/primitive information contract 是否有缺口；不得声明物理不可达、
task failure 原因、学习或能力改善。

## `R0001-P50-E4`

### Exact claim inventory

canonical role claim 只来自冻结 binding：

- manipulated object：每个 `binding.objects[*].collision_geom`，保留 exact instance；
- articulation：`binding.articulation.handle_geom`；
- target-container、floor-support：`allowed_robot_contact_roles` 的 exact geom；
- robot：robot root body closure 内每个 exact geom；
- 其余具名非 robot geom：`other_furniture`；
- 无名 geom、site、background、非法 object type：`unknown`。

同一个 exact geom 出现两个不同 task role：`invalid`。同 body 不同 geom 可有不同 role。
不得把一个 geom claim 传播到同 body 其他 geom。

### Frozen alias inventory

每个 alias 保存：

`source_visual_geom → canonical_exact_claimed_geom → role/instance`

| Task | Source visual geom | Canonical exact geom |
|---|---|---|
| living | `toy_duck_visual` | `toy_duck_collision` |
| living | `toy_football_visual` | `toy_football_collision` |
| living | `storage_basket_visual` | `basket_bottom_collision` |
| dining | `dining_cup_visual` | `dining_cup_collision` |
| dining | `dining_plate_visual` | `dining_plate_collision` |
| kitchen | `cleaner_yellow_visual` | `cleaner_yellow_collision` |
| kitchen | `cleaner_pink_visual` | `cleaner_pink_collision` |
| kitchen | `drawer_handle_visual` | `drawer_handle` |

alias 必须一跳、同 body、source 具名且没有 independent exact task claim、target 是唯一
canonical exact claim。禁止 chain、cycle、cross-body、nearest、prefix-name matching、
role priority 或结果后追加。

### Frozen task-visible inventory

这是“可成为 task-role visual surface 的结果前清单”，不是实际 visible 结果：

- living：
  - `toy_duck_visual`
  - `toy_football_visual`
  - `storage_basket_visual`
- dining：
  - `dining_cup_visual`
  - `dining_plate_visual`
  - `cup_holder`
  - `plate_holder`
- kitchen：
  - `cleaner_yellow_visual`
  - `cleaner_pink_visual`
  - `drawer_handle_visual`
  - `drawer_bottom`
  - `drawer_front`
  - `drawer_back`
  - `drawer_left`
  - `drawer_right`
  - `drawer_divider`

site 不在 inventory；container empty volume 没有 visible claim。

### 负守护

- dining `sideboard_top`、`sideboard_body` 必须为 `other_furniture`；
- kitchen `drawer_handle_visual` 必须为 articulation；
- kitchen drawer 六个 surface 必须为 target-container；
- kitchen `drawer_frame_*` 必须为 `other_furniture`；
- wall、rug 和其他 world-body geom 不得因 `floor` claim 变为 floor-support；
- target sites 必须为 `unknown_site`；
- background `(-1,-1)` 必须为 `background`。

### 运行与判定

```text
.venv/bin/python -m hwr.apps.audit_entity_candidate_mapping \
  --aliases configs/eval/entity_candidate_aliases_v1.json \
  --output runs/research-loop/0012/r0012-p50-e4-mapping-s20265004
```

必须输出：

- `tables.json`：三个场景全部 body/geom 的 deterministic table；
- `report.json`；
- `manifest.json`。

`accepted as exact-geom evaluator mapping contract` 仅当：

1. 3/3 scene 全部 preflight；
2. 8/8 alias 全部一跳、同 body、target exact-claimed；
3. exact geom role conflict `0`；
4. task-visible inventory unknown `0`；
5. 全部负守护误标 `0`；
6. table bytes/hash 重复构造 bit-identical；
7. source、XML、binding、alias 与环境 provenance 完整；
8. import/AST isolation 证明 alias 不被 candidate/policy/action path 导入。

任一语义/负守护失败：`rejected_design_not_expressive`。完整性、identity、determinism 或
isolation 失败：`invalid`。

预算：wall time `<2min`，RSS `<1GiB`，artifact `<20MiB`。允许固定 scene/camera/reset 的
只读 segmentation smoke，但只用于“inventory observed label uniquely mapped”守护，不得
报告 visibility rate。无 Episode、无动作、无训练、不休眠。

通过只接受 mapping contract，不产生 entity coverage、物体识别、泛化或能力结论。

## `R0001-P60`

### 研究问题

在不读取 entity truth、B2 outcome、contact、reward 或 success 的新 seed prefix 上，现有
acquisition+B0/B1 结束时，selected candidate 派生的双臂 preposition target 是否已经：

1. 被严格 arm-chain outer envelope 排除；或
2. 虽未被结构排除，但超出现有 100-step B2 的 nominal command support；或
3. 上述两者均未否定，需要更强 IK/collision/action-chain 诊断。

### Cohort 与 seed

- plan ID：`R0001-P60-E1-formal`
- task：三个正式 task
- observation latency：`{1,2}`
- action latency：`{1,2}`
- 12 cell
- 每 cell 3 个完整 prefix Episode
- 共 36 个独立 Episode
- 每 Episode：
  - 995-step frozen acquisition；
  - 100-step B0；
  - 300-step B1；
  - 在 B2 首个 action 前停止；
  - 不执行 B2、contact phase 或 post-prefix action。
- evaluation randomization，natural sampled latency；不得 reset override。
- raw seed ordinal 每 cell 最多 `768`。
- latency-matched physical prefix 每 cell最多 `16`。
- 只有 latency match 才运行物理 prefix。
- eligible：
  - candidate set 非空；
  - selected index 有效；
  - 1,395-step prefix 完整；
  - policy-visible input、action bounds、stale-action、safety、terminal、severe collision、
    invalid force 与 P40 conservation 守护全部通过；
  - target formula crosscheck 通过。
- 不使用 P51 专属：
  - `relative_yaw >= π/6` eligibility；
  - frame_legacy/frame_fixed first-action difference；
  - B2 treatment outcome。
- cell 在 16 个 latency-matched prefix 内不足 3 个 eligible：整项
  `inconclusive_design_infeasible`，不以部分 cell 发布诊断。

salt reveal 保存在 ignored：

`runs/research-loop/0012/.host/p60-salt.txt`

结果前仅冻结 commitment：

`263e9f85e32f4a3f5f1560ba82cd820a558cc0aad9a5710bbdf6a3306e3f9c55`

seed 派生使用现有 `hwr.opaque-episode-seeds/v1`：

```text
planned_episode_id =
  SHA256(schema || plan_id || task_id || cell_id || raw_ordinal)

environment_seed =
  int63(SHA256(salt || "environment" || planned_episode_id))

policy_rng_seed =
  int63(SHA256(salt || "policy" || planned_episode_id))
```

salt 在 implementation commit、测试和门禁通过后才允许 reveal 给正式 app。不得根据 seed、
候选或几何结果换 salt。

### 固定 frame 与目标

- 所有 candidate、base、shoulder、tool 和 target 计算在 acquisition frame。
- `acquisition_base_pose`：reset 后第一个 policy-visible observation 的 base pose。
- `b2_policy_base_pose`：B1 完成后、将进入 B2 的 policy-visible observation base pose。
- selected candidate：冻结 generator 与 selector 的 exact canonical record。
- preposition target：复用并独立 crosscheck
  `hwr.eval.cartesian_convergence.preposition_targets()`。
- current-base shoulder local origin：
  - left：`(0.02, +0.31, 0.82)m`
  - right：`(0.02, -0.31, 0.82)m`
- shoulder 通过
  `_acquisition_from_robot(acquisition_base_pose,b2_policy_base_pose)`
  转入 acquisition frame。
- arm outer length：

```text
L_outer =
  0.13 + 0.31 + 0.27 + 0.09 + 0.08 + sqrt(0.255^2 + 0.045^2)
```

该值来自冻结 `_arm_chain()` 在 shoulder point 之后的所有 translation norm 之和。

- `strict_outer_margin_m = L_outer - ||preposition_target - shoulder||_2`
- 浮点 tolerance：`1e-12m`
- arm `strict_outer_impossible` 当 margin `< -1e-12m`
- margin `>= -1e-12m` 只记为 `not_disproven`，不得称 reachable。

### 三层指标

#### 1. Strict structural certificate

- 每臂 strict outer margin；
- pair `hard_bilateral_impossible`：
  同一 B2 入口至少一臂 `strict_outer_impossible`；
- pair `both_arms_strict_outer_impossible`：
  两臂均 strict negative，只作严重度；
- 不检查或宣称 joint-limit、self/environment collision、IK、path 或 dynamics。

#### 2. Finite-horizon nominal support

- policy-visible tool position：复用 P52-validated FK；
- `tool_to_preposition_d0_m`；
- nominal B2 max command：
  `100 × 0.08m/s ÷ 20Hz = 0.40m`；
- readiness allowance：`0.10m`；
- `nominal_b2_support_margin_m = 0.40 + 0.10 - d0`；
- pair `nominal_bilateral_support_deficit`：
  至少一臂 margin `<-1e-12m`；
- 这是 nominal finite-horizon command proxy，不是 actual path 或 dynamics。

#### 3. 描述性 entry geometry

- candidate-base horizontal range；
- candidate heading error；
- existing B1 linear/angular residual command；
- left/right shoulder、tool、preposition coordinates；
- task、cell、observation latency、action latency 分账。

不得把描述量加入 strict certificate。

### 主要判定

Episode/prefix 是唯一独立样本；arm、step、candidate、cell 和 task 不是额外样本。

`accepted as phase-entry necessary-geometry measurement evidence` 仅当：

1. 36/36 Episode、12/12 cell、每 task 12 Episode 完整；
2. salt commitment、seed derivation、natural latency rejection 与 complete-case 规则通过；
3. candidate/input/prefix/target/source/model identity 完整；
4. 1,395-step safety、bounds、terminal、force、P40 conservation 守护通过；
5. frame、shoulder、outer length、FK 与 target 能独立重算；
6. finite、tolerance、determinism 和全量分账通过。

合同有效后分别判定：

`strict_phase_entry_deficit_supported`：

- hard-impossible `>=30/36`；
- 每 task hard-impossible `>=8/12`。

`strict_phase_entry_deficit_rejected`：

- hard-impossible `<=12/36`；
- 每 task hard-impossible `<=6/12`。

其他：`strict_phase_entry_diagnostic_inconclusive`。

独立判定 nominal support：

- supported：deficit `>=30/36` 且每 task `>=8/12`；
- rejected：deficit `<=12/36` 且每 task `<=6/12`；
- 其他：inconclusive。

不得合并两个判定，也不得要求“两臂均失败”才判 pair hard-impossible。

### 命令、产物、停止条件与预算

正式命令：

```text
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_phase_entry_geometry \
  --salt-file runs/research-loop/0012/.host/p60-salt.txt \
  --output runs/research-loop/0012/r0012-p60-phase-entry-s20266001
```

产物至少包含：

- `plan.json`
- `seed-audit.json`
- `episodes.json`
- `report.json`
- `manifest.json`

同一 output/staging 已存在时拒绝覆盖。raw seed mismatch 只记 audit、不运行物理；任何已执行
prefix 的 raw trace 与 geometry 必须保留。不得在正式结果后补 seed 或删除 invalid case。

硬停止：

- 首个 hard safety failure、source/provenance drift 或 nonfinite geometry；
- 任一 cell 达到 16 个 latency-matched physical prefix 仍不足 3 eligible；
- wall time `90min`；
- peak RSS `8GiB`；
- artifact `2GiB`；
- 数据卷可用空间低于 `20GiB`。

最大物理预算：

- 12 cell × 16 latency-matched prefix = 192 prefix；
- 192 × 1,395 = 267,840 control step；
- 接受样本最多 36 Episode；
- 无 B2、无接触实验、无训练。

这是小型、前缀式正式诊断，不休眠；主 Agent持续轮询。宿主图形环境只能解决 CGL 运行
条件，不改变合同。

### 允许声明与结果路由

最多声明：

- strict arm-chain outer envelope 是否在 B2 入口排除 bilateral readiness；
- nominal 100-step B2 command support 是否不足；
- 这些诊断在固定三任务与 latency cell 中的完整分账。

禁止声明：

- candidate 与 task entity 相关；
- target IK/collision/dynamics reachable；
- 接触、抓取、任务成功、泛化、安全改善；
- 增加 phase 时长、速度或修改 selector 一定有效。

结果路由：

- strict deficit supported：下一轮优先提出一个 target/base/workspace 单主变量，不自动
  启动 P65；P62/P64 继续 defer。
- strict rejected、nominal supported：下一轮可修订 P62 的 independent feasibility
  witness，检查 finite-horizon action chain。
- 两者 rejected：entity attribution 的相对价值上升，但 P64 仍需重新筛选。
- 任一 invalid/infeasible：保留 artifact，先修合同，不用部分结果改行为。

## 实施前门禁

1. 本文件与 `02-review.md` 提交到 `feat/research-loop`。
2. 三名实施 Agent 从同一冻结提交开始。
3. 每项行为变化有测试；本轮三项均不得改变正式行为。
4. 每项实施由另一 Agent 或主 Agent 做独立代码审计。
5. focused tests、Python size、architecture、physics integrity、compileall 与
   `git diff --check` 通过。
6. P61/P50-E4 先运行；P60 仅在两项实现门禁通过后运行，但 P61/P50-E4 的科学判定不改变
   P60 门槛。
7. 正式运行只从干净、已提交 source commit 启动。
