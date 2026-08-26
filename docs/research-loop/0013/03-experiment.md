# R0013 冻结实验

## 冻结状态

- 提案冻结提交：
  `e21c01dea625ca3a4f032a4b9978f9f1849c21ef`
- 入选：
  - `R0001-P66-E1`：production-isomorphic predictive safety witness；
  - `R0001-P72-E1`：P61 exact-reference anti-self-certification audit；
  - `R0001-P68-E1`：initial-microinteraction association stopping gate，仅在 P72
    的 P68 dependency gate 通过后实施和运行。
- deferred：`R0001-P67`、`R0001-P69`、`R0001-P70`、`R0001-P71`。
- 本轮不训练、不执行 B2/B3–B7、不修改 selector、primitive、safety threshold、
  reward、termination 或 capability evaluator。
- 本文件提交后，实施不得修改本文件；runner 必须绑定该 Git blob，而不是锁定整个
  `docs/research-loop/0013/` tree。

## 共同不变量

1. 历史 `docs/research-loop/0001/`～`0012/` 的起始 tree 必须与
   `00-context.md` 一致。
2. 正式运行只允许从干净、已提交且本冻结文档 blob 一致的 source commit 启动。
3. 所有输出使用临时目录后原子 rename；目标已存在时 fail closed。
4. 所有 artifact 都包含 source commit、命令、输入文件 SHA-256、源码/config/XML
   identity、运行环境、预算、claim flags 与各输出 hash。
5. policy/private boundary：
   - evaluator-only geom/body/contact/force/role truth 不进入 observation、latency queue、
     candidate、selector、动作、安全 decision、reward、termination 或训练数据；
   - 正式 policy/candidate 路径保持 bit-identical；
   - 禁止将 diagnostic 字段加入通用 `RuntimeStepOutcome.info` 或
     `EpisodeEvent.details`。
6. `220N` severe-force threshold、2-control-step predictive horizon、100ms
   observation validity、action-latency FIFO 与 actuator scale 均不得改变。
7. 预测拒绝只能称为 simulator clone 中的 safety witness；不得称为实际碰撞或硬件安全。
8. `training_executed=false`、`policy_inference_executed=false`、
   `capability_claim_allowed=false`、`task_success_claim_allowed=false`、
   `generalization_claim_allowed=false`、`hardware_safety_claim_allowed=false`。

## 实施所有权

### `R0001-P66-E1`

唯一负责人：Implementation Agent P66。

允许修改：

- `src/hwr/adapters/mujoco/dual_arm_backend.py`
- `src/hwr/adapters/mujoco/predictive_safety_diagnostic.py`（新）
- `src/hwr/eval/predictive_safety_witness.py`（新）
- `src/hwr/apps/evaluate_predictive_safety_witness.py`（新）
- `tests/test_predictive_safety_diagnostic.py`（新）
- `tests/test_predictive_safety_witness.py`（新）
- `tests/test_predictive_safety_witness_app.py`（新）

允许在 `dual_arm_backend.py` 增加默认 no-op 的薄 observation hook；production
decision 仍必须由现有 `_predictive_safety_violation()` 产生。不得把 witness 数据写入
通用 runtime output。核心文件仍受 `<=800` 行、函数 `<=200` 行门禁。

### `R0001-P72-E1`

唯一负责人：Implementation Agent P72。

允许修改：

- `src/hwr/eval/interaction_contract_mutation.py`（新）
- `src/hwr/apps/audit_interaction_contract_mutations.py`（新）
- `tests/test_interaction_contract_mutation.py`（新）
- `tests/test_interaction_contract_mutation_app.py`（新）

不得修改 P61 producer、P61 artifact、`interaction_contract.py` 或
`audit_interaction_contract.py`。本项先测量残余依赖，不在同一因果步骤修复。

### `R0001-P68-E1`

仅当 P72 report 中 `p68_dependency_gate_passed=true` 时分配唯一负责人。

允许修改：

- `src/hwr/adapters/mujoco/candidate_association.py`（新）
- `src/hwr/eval/initial_candidate_association.py`（新）
- `src/hwr/apps/evaluate_initial_candidate_association.py`（新）
- `tests/test_candidate_association.py`（新）
- `tests/test_initial_candidate_association.py`（新）
- `tests/test_initial_candidate_association_app.py`（新）

不得修改 P50/P50-E4/P61 producer、历史 artifact 或
`target_selection.py`。若无法在独立模块中重建 raw support/component 且保持原
candidate bytes bit-identical，P68 直接 `invalid`，不得为方便而改变 generator。

## `R0001-P66-E1`

### 研究问题

P60 anchor 的 production predictor 是否存在一个完整、确定、与原判据同构的
forbidden-contact witness，同时 authoritative physics 确实未执行被拒动作？

本项不是 occurrence-rate 实验。唯一样本/anchor 是一个历史 Episode，不计算置信区间。

### 冻结输入

P60 输入目录：

`runs/research-loop/0012/r0012-p60-phase-entry-s20266001`

输入 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `episodes.json` | `681b2ac5f49d8af7fa21108e3adb96f1fcc0bbc894d2a3f6b2d544cb28f64c4e` |
| `plan.json` | `df69da8606f78f94fdaaecef0021a64eb33c8cc5e2b62720bdbdd1f5e2255e5a` |
| `seed-audit.json` | `5bc6fb284f14c689da0167a40e76775c83c2116fb9541d96d45371307dde71f3` |
| `report.json` | `1674278396eb51f647c261d6514b2411cb09ba786deda659ed189c6124841737` |
| `manifest.json` | `ee21c04f009d0ab89bc83f3f00516a36a91955f9369ab50d66bea0f04f9c75df` |

唯一 anchor：

- planned Episode：
  `838dd73530fc555aefa4084574fccb4b5a9ca91496f2cca6791f702d95a27bb8`
- task/cell：`tidy_living_room_3d/v1` / `cell-00-o1-a1`
- raw seed ordinal：`35`
- environment seed：`1774785007276425316`
- policy RNG seed：`6474844404874254699`
- candidate set SHA-256：
  `6d51da1277c6e84ebfe3e6b1b977a30e08ebcf2db106e63e064434fa7c855134`
- selected index：`2`
- runtime randomization SHA-256：
  `cd653560fe05b26894828e2052f2aaf63872754135ce224068da471e03fbd9b6`
- policy input sequence SHA-256：
  `e5558b0cf76b58625818da503b3716b5207a3beec8056ad24038c69d34667316`
- raw prefix trace SHA-256：
  `09dc4fa6ccf72a77d08c34aba740e5ff2af03fed2c8623041df4a45c2159489a`
- prefix step count：`1280`
- expected rejection：step `1279`,
  `action_rejected / predicted_severe_collision`。

### 生产同构 witness

production safety decision 保持：

1. 对传入 `_predictive_filter` 的实际 frame 创建 `MjData` clone；
2. 使用原 `_write_controls` 与原 2-control-step horizon；
3. 只在原 control boundary 调用原 `_predictive_safety_violation()`；
4. production predicate 仍为当前所有 forbidden **contact point** 的
   `max(abs(normal_force)) >= 220.0`；
5. 若拒绝，authoritative 分支执行 hold，且不执行该 control step 的 `mj_step`。

observer 只能在 production predicate 得出 boolean 后旁路读取同一 trial `MjData`，
记录：

- boundary ordinal `1..2` 与 cumulative substep；
- 每个有效 forbidden contact point：
  contact index、normal force、geom IDs/names、body IDs/names、robot/environment side；
- 按 `(force desc, canonical geom pair, contact index)` 冻结排序得到的 display maximum；
- production violation boolean 与 `threshold=220.0`；
- policy proposal、delayed/scaled plant action、queue source step、predictor input、
  final applied action；
- trial 与 authoritative `physics_advanced` 的明确区分。

nonfinite force 不在本项改变 production 行为。若出现 nonfinite：witness fail closed，
P66 判 `invalid_nonfinite_production_semantics`，另立 safety 修复。

### 对照与测试

1. observer disabled 与 enabled 各完整重放一次 anchor。
2. 两次必须在以下字段 bit-identical：
   - policy input、candidate set/index、policy proposal、plant/predictor/applied action；
   - event、拒绝 step、raw prefix trace；
   - authoritative qpos、qvel、time、action/observation queue、arm targets、steps、
     sequence、task counters 与 final hard-stop。
3. 在拒绝 step，pre/post authoritative qpos、qvel、time 必须 bit-identical；
   `physics_advanced=false`；actual severe collision 为 0。
4. 独立 fixture 固定：
   - 同 pair 两个 130N point 不触发 220N；
   - 219.999N 不触发，220N 触发；
   - allowed pair、robot-self、world-world 不计入；
   - equal-force tie 只影响 display pair，不影响 decision；
   - nonfinite force 被 witness 标记 invalid。
5. fixture 不能通过直接传 production boolean 给 witness 判定来获得一致率。

### 判定

`accepted as predictive-safety witness contract` 当且仅当：

- anchor 全部 identity 与拒绝位置复现；
- observer off/on authoritative identity 全部一致；
- production event 与 observer 从 raw contact point 独立计算的 threshold crossing 一致；
- rejection witness 字段完整率 100%；
- display maximum force finite 且 `>=220N`；
- pair 是 frozen binding 下的 forbidden robot/environment pair；
- dangerous action 未进入 authoritative physics；
- actual severe collision、invalid force、P40 drift 均为 0；
- 所有 focused 与仓库门禁通过。

若 anchor identity 或 event 不复现：

`inconclusive_anchor_not_reproduced`

若 witness/production 不一致、observer 污染状态、危险动作推进或 nonfinite：

`invalid`

### 正式命令与预算

```text
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_predictive_safety_witness \
  --p60-input runs/research-loop/0012/r0012-p60-phase-entry-s20266001 \
  --output runs/research-loop/0013/r0013-p66-predictive-witness-s20266601
```

- 最多 2 次完整 anchor replay；不得换 seed或补 Episode。
- wall time：20min。
- peak RSS：4GiB。
- artifact：512MiB。
- 最低磁盘余量：20GiB。
- 小型正式诊断，前台运行并持续轮询，不休眠。

## `R0001-P72-E1`

### 研究问题

P61 accepted verdict 是否对冻结 exact source reference 与 planner evidence 的关键反事实
敏感；P68 依赖的 initial microinteraction annotation 是否独立、可执行且 fail closed？

### 冻结输入

P61 输入：

| 文件 | SHA-256 |
|---|---|
| `transitions.json` | `1cc139e7f8b02a6325d16282f9b7882e9736c40d03f56644c1739d79ee7bcc0a` |
| `report.json` | `d9a760eaa30198eda95d20e90a4ebf4c9d9f5bcd2e118b6e139446866545719a` |
| `manifest.json` | `6a019a7591a2614c6082dea102c29f9cb24e101f78da8ce21ce3725f60df221d` |

producer commit：
`6bf0400f51a25bfb6f45e951299c410efd5c2c7a`

### 冻结 mutation 集

每个 mutation 都从独立、未修改的 source/config 副本开始，重新运行
`build_source_audit()` 与 `audit_interaction_contract()`；不得直接改输出 JSON boolean。

1. `candidate_schema_extra_field`
2. `policy_schema_extra_field`
3. `primitive_signature_extra_argument`
4. `selector_signature_extra_argument`
5. `primitive_phase_extra_value`
6. `initial_annotation_remove_allowed_role`
7. `initial_annotation_unknown_role`
8. `initial_annotation_duplicate_task`
9. `initial_annotation_consumer_changed`
10. `role_field_without_independent_planner_state`
11. `fully_expressive_direct_caller_positive_control`
12. `remove_interaction_field_negative_control`
13. `remove_destination_field_negative_control`
14. `remove_articulation_threshold_negative_control`

mutation 1–5 是 exact-reference dependency；6–9 是 P68 dependency；10 检查 planner
evidence 是否与 role field 错误耦合；11–14 检查 verdict 可翻转性与各 gap 的独立贡献。

### 判定与 P68 依赖门

- harness 需证明每个 mutation 只改变预定字段、实际到达 parser/auditor，且重复运行
  canonical bytes bit-identical；否则 P72 `invalid`。
- 若 1–5 任一没有使完整 audit fail closed，记录 residual exact-reference gap。
- 若 10 把 `planner_call_state_available` 置真而没有独立 planner state/call evidence，记录
  residual planner-evidence gap。
- 若 6–9 全部使 audit fail closed，且 baseline initial annotations 与 P61
  `transitions.json` bit-identical，则
  `p68_dependency_gate_passed=true`；否则为 false。
- 11 必须能够让预期的 gap verdict 翻转；12–14 必须分别恢复对应 gap。若不能，记录
  residual verdict dependency gap。

无 residual：

`accepted as P61 anti-self-certification audit`

存在至少一个可复现 residual：

`accepted as residual P61 contract gap evidence`

不得把 residual 数量当统计样本；不得外推 whole-program planner。

### 正式命令与预算

```text
.venv/bin/python -m hwr.apps.audit_interaction_contract_mutations \
  --contract configs/eval/interaction_contract_v1.json \
  --p61-input runs/research-loop/0012/r0012-p61-interaction-contract-s20266101 \
  --output runs/research-loop/0013/r0013-p72-p61-mutation-s20267201
```

- wall time：60s。
- peak RSS：1GiB。
- artifact：10MiB。
- 小型静态审计，前台运行，不休眠。

## `R0001-P68-E1`

### 条件启动

只有 P72 正式 report：

`p68_dependency_gate_passed=true`

时才允许实施。P72 即使发现与 P68 无关的 exact-reference 或 planner residual，也不自动
阻止 P68；必须在结果中明确收缩 P61 对应 claim。

### 研究问题

固定 P50 24-Episode acquisition cohort 中，原 selector 的 selected final candidate 是否
主要由当前 initial microinteraction 允许的 exact task entity 支持？

### 冻结输入

P50 acquisition：

| 文件 | SHA-256 |
|---|---|
| `capsules.json` | `223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf` |
| `plan.json` | `5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab` |
| `report.json` | `b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0` |
| `manifest.json` | `cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86` |

P50-E4 mapping：

| 文件 | SHA-256 |
|---|---|
| `tables.json` | `88540ddb87e6df129eaaa9666b0011598a422629b1085c27b7b844e735918fbc` |
| `report.json` | `96fdc3abde155e09715bb7e1314c9b6cbef26dffdeda193f6dc336e4cd767402` |
| `manifest.json` | `7c950b5e132a0c24395a63ee6dec150f5244466cf5b3f38118a4efa00a2a3579` |

P61 annotations 使用上节冻结的三文件与 hash。

输入 cohort 固定 24 Episode：living/dining/kitchen 各 8；当前 candidate empty
分别为 4/0/1，但该数量只用于输入完整性核对，不用于调阈值。

### 关联算法

1. 对每个 P50 Episode 按原 environment/policy seed、自然 latency、995-step acquisition
   确定性 replay；不得增加或替换 Episode。
2. 在每个原 capture identity 对应的 observation source state，使用独立 renderer 捕获
   `head_depth` camera segmentation `(object_id, object_type)`；不得用当前 authoritative
   state 代替 latency queue 中的 observation source。
3. 同步条件：
   - timestamp/sequence 与原 capture 完全一致；
   - replay policy input bytes、candidate-visible bytes、candidate set bytes/order/hash、
     selected index、proposed/applied action trace 与原 capsule 完全一致；
   - segmentation observer on/off 的上述字段与 physics trace bit-identical。
4. 在独立 evaluator 模块重建原 raw candidate support pixel：
   - 使用与原 generator 相同的 depth valid、center/ring、prominence、depth spread、
     patch、robot-self mask 与 geometry helper；
   - 旁路保留每个 raw candidate 最终用于 `_candidate_from_points` 的 pixel coordinates；
   - 使用原 adjacency/component 规则聚合；
   - 重建的 final candidate canonical bytes 必须与 P50 artifact bit-identical。
5. 对 final candidate component 的全部 raw support pixels，用同 observation identity 的
   segmentation 和 P50-E4 table 分类；每个重复 view/pixel 只作为 candidate 内部支持，
   不是独立样本。
6. candidate 分类：
   - `stage_compatible`：P61 allowed label 支持比例 `>=0.80`；
   - `stage_incompatible`：明确非 allowed、非 unknown/background/site 的已知 label 支持
     比例 `>=0.80`；
   - 其他为 `mixed_or_unknown`。
7. Episode 分类：
   - candidate empty：`no_relevant_final_candidate`，subtype `candidate_set_empty`；
   - selected 为 compatible：`stage_compatible_selected`；
   - selected incompatible，且至少一个其他 final candidate compatible：
     `relevant_exists_but_distractor_selected`；
   - selected incompatible，且所有 final candidate incompatible：
     `no_relevant_final_candidate`；
   - 其他：`mixed_or_unknown`。

### 主要指标与判定

样本单位为 24 个 Episode。必须报告五类计数、每任务 8-Episode 分账、每个 candidate 的
role/label support counts 与 selected status。

高关联门：

- `stage_compatible_selected >=18/24`；
- 每任务 `>=5/8`；
- `mixed_or_unknown <=2/24` 且每任务 `<=1/8`。

全部满足：

`accepted as initial-association stopping-gate evidence`

低关联门：

- `stage_compatible_selected <=6/24`。

满足时：

`accepted as selector-relevance stopping evidence`

其余：

`inconclusive`

无论结果如何，都不能声明识别、可达、可交互、路径安全、任务成功或泛化。

### 失效条件

- P72 dependency gate 未通过；
- 任一 P50 input hash、candidate bytes/order/index 或 replay identity 不一致；
- segmentation 未与 delayed observation identity 对齐；
- candidate support 重建不完整；
- sidecar on/off 改变正式 trace；
- 新增/修改 alias、用 world coordinate/name 猜测 entity、用 task truth 调 selector；
- 结果后修改 0.80、18/24、6/24 或 mixed 上限。

### 正式命令与预算

```text
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_initial_candidate_association \
  --p50-input runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --mapping-input runs/research-loop/0012/r0012-p50-e4-mapping-s20265004 \
  --interaction-input runs/research-loop/0012/r0012-p61-interaction-contract-s20266101 \
  --p72-input runs/research-loop/0013/r0013-p72-p61-mutation-s20267201 \
  --output runs/research-loop/0013/r0013-p68-initial-association-s20266801
```

- 固定 24 Episode，每个只允许 observer disabled/enabled 两次 acquisition replay；
  不得补 seed。
- wall time：30min。
- peak RSS：8GiB。
- artifact：2GiB。
- 最低磁盘余量：20GiB。
- 小型正式诊断，前台运行并持续轮询，不休眠。

## 排程与停止规则

1. 提交 `02-review.md` 与本文件，记录 frozen document blob。
2. P66 与 P72 实施可并行，写集严格分离。
3. 两项分别由非作者或主 Agent 做独立代码审计。
4. 先运行 P72；若 mutation harness invalid，停止 P68。
5. 运行 P66；其结果不改变 P72/P68 的阈值。
6. 仅在 P72 `p68_dependency_gate_passed=true` 时实施并运行 P68。
7. 本轮不因 P66/P68/P72 结果临时补入 P67/P69/P70/P71。
8. 不启动训练；无需 `traex-host-exec` 休眠。

## 门禁

- focused pytest；
- `PYTHONPATH=.:src MUJOCO_GL=glfw .venv/bin/pytest -q`；
- `.venv/bin/python scripts/check_python_size.py`；
- `.venv/bin/python scripts/check_architecture.py`；
- `.venv/bin/python scripts/verify_physics_integrity.py`；
- `.venv/bin/python -m compileall -q src tests scripts`；
- `git diff --check`；
- 历史轮次 tree 与 `00-context.md` 完全一致。
