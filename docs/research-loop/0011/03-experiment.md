# R0011 冻结实验

## 冻结状态

- 状态：结果前冻结
- 冻结日期：`2026-08-23`
- 起始提交：`8d14fadb2b9103386788dc3d5426d3624fd624d7`
- 唯一主实验：`R0001-P50-E3`
- 独立辅助实验：`R0001-P57`
- 本轮不实施：`R0001-P56`、`R0001-P58`、`R0001-P59`
- 正式训练：不授权
- selector/Replay/Actor/世界模型：不授权

本文件提交后，不得根据运行结果修改 cohort、映射、阈值、样本单位、指标、分账、判定门、
invalid 条件或允许声明边界。评测实现缺陷只能修复合同执行；若修复发生在读取结果后，必须
保留旧 artifact、记录时序并用同一输入原样重跑，不得更换 cohort 或门槛。

## 实验隔离

### `R0001-P50-E3`

- 唯一负责人：实施 Agent P50-E3。
- 分支：`feat/r0011-p50-e3`。
- 文件所有权：
  - `src/hwr/adapters/mujoco/rendering.py`
  - `src/hwr/adapters/mujoco/formal_household_backend.py`
  - `src/hwr/adapters/mujoco/candidate_acquisition.py`
  - `src/hwr/eval/entity_candidate_coverage.py`（新）
  - `src/hwr/apps/evaluate_entity_candidate_coverage.py`（新）
  - 必要的窄导出文件
  - `tests/test_entity_candidate_coverage.py`（新）
  - 与上述集成直接相关的既有 P50 测试
- 禁止修改：
  - `src/hwr/eval/target_selection.py` 的正式 candidate 行为；
  - candidate score、selector、primitive、动作、安全、reward、termination；
  - `docs/research-loop/0001/`～`0010/`；
  - P50/P51 历史 artifact。

### `R0001-P57`

- 唯一负责人：实施 Agent P57。
- 分支：`feat/r0011-p57`。
- 文件所有权：
  - `src/hwr/eval/precontact_reachability.py`（新）
  - `src/hwr/apps/evaluate_precontact_reachability.py`（新）
  - 必要的窄导出文件
  - `tests/test_precontact_reachability.py`（新）
- 禁止修改 P51 runner、bank、terminal、candidate、primitive、phase、速度或 gripper。

两个实施 Agent 写集不重叠。每项必须独立提交并引用提案 ID；主 Agent 串行集成、运行项目
门禁与正式实验。

## 固定输入

### P50-E3 输入

复用 R0010 的完整 frozen plan，不生成新 salt、不换 seed、不补 Episode：

| 文件 | SHA-256 |
|---|---|
| `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/plan.json` | `5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab` |
| `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/capsules.json` | `223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf` |
| `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/report.json` | `b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0` |
| `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/manifest.json` | `cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86` |
| `runs/research-loop/0010/r0010-p50-e2-funnel-s20265001/report.json` | `4c7f36d20356d2f0f9c83d024412da5ec3a95dea8714e9a04d91d0cd686d0e39` |

Cohort：

- 3 个正式 task；
- observation latency `{1,2}`；
- action latency `{1,2}`；
- 12 cell；
- 每 cell 2 个结果前已冻结 Episode；
- 共 24 Episode；
- 每个 Episode 995 个 acquisition control step；
- sidecar-enabled 主运行与 sidecar-disabled validation replay 各 24 次；
- 最大 48 次物理 acquisition、47,760 control step；
- 不执行 post-selection primitive。

R0010 已经看到这些 Episode 的 candidate/funnel 结果，但没有看到 entity sidecar 结果。本轮
只允许声明该固定 cohort 的 measurement evidence，不把它称为新的未见分布泛化评测。

### P57 输入

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `runs/research-loop/0010/r0010-p51-e1-bank-s20265101/bank.json` | 7,160,016 | `09d2fe4e05f2bd8d23ebfe6886fe260d1b34b41771da42992f0f432a8a04f3d3` |
| `runs/research-loop/0010/r0010-p51-e1-bank-s20265101/manifest.json` | 53,605 | `7e0d5f9c7757b59ceb8d4dfe3ddcba38cc1d1037c43c358e7168d700310d5e45` |
| `runs/research-loop/0010/r0010-p51-e1-convergence-s20265101/terminals.json` | 26,314,706 | `1c54f93a95bfbf4e08076b3c633b22dce295990a6808a48f0f10de18a2b3c2c7` |
| `runs/research-loop/0010/r0010-p51-e1-convergence-s20265101/report.json` | 4,643 | `3fcac95c2362923d9eb94ef4d7121d5bcb31ea859a308ed352321dfa93771cc9` |
| `runs/research-loop/0010/r0010-p51-e1-convergence-s20265101/manifest.json` | 53,753 | `821f3cf6fea922a86b4096ee5d0ba9c64b9d8f444eacc98dcfc1f164da1328d2` |

- 固定分析 `frame_fixed` role 的 36 pair；
- 12 cell × 3 pair；
- 三任务各 12 pair；
- 不读取 superseded artifact；
- 不执行 MuJoCo，不创建新样本；
- pair 是独立样本；arm、step、action dimension 不是独立样本。

## `R0001-P50-E3`

### 研究问题

在固定 P50 cohort 上：

1. task entity 是否进入 candidate keyframe 的可见视野；
2. 可见 task entity 是否形成 raw candidate；
3. task-entity raw provenance 是否形成 component 与 final candidate；
4. 若未进入 final，在哪个冻结 gate 首次清空；
5. final set 是否只含 distractor；
6. 这些现象是否跨 task 与 latency cell 重复。

### Sidecar 采集时序

原始 observation 的固定顺序：

1. physics state 已完成当前 control step；
2. 使用动态随机化后的 `head_rgb`/`head_depth` 相机生成正式 RGB-D；
3. 生成正式动态 calibration；
4. 仍在同一 `MjData`、同一相机位姿、同一
   `(timestamp_ns, sequence_id)` 下生成 evaluator-private segmentation；
5. 校验 renderer mode 已恢复；
6. 原始 observation 与 private sidecar 分别进入并行 queue；
7. policy 获取 delayed observation 时，evaluator 只能按相同 identity 取对应历史 sidecar。

禁止：

- 用 delayed observation 返回时的当前 `MjData` 重画；
- 用 latest/future segmentation；
- 把 segmentation、entity role 或 sidecar handle 写入 `DualArmObservation`；
- 在 observation identity 缺失时按顺序、时间最近或 frame ordinal 猜测匹配。

### Segmentation 与角色映射

MuJoCo segmentation 像素为 `(object_id, object_type)`。

1. `object_type == mjOBJ_GEOM`：
   - 取 `geom_id`；
   - 解析 `geom_bodyid` 与 `body_rootid`；
   - 使用结果前构造的 body-role table 归类。
2. `object_type == mjOBJ_SITE`：固定为 `unknown_site`，不计入任何 task-entity 分子。
3. `(-1,-1)`：`background`。
4. 其他 object type、越界 ID、无名 geom/body 或角色冲突：`unknown`。

body-role table：

- 每个 manipulated object 使用 binding 的 `object.body`，保留 exact object instance；
- articulation 使用 `binding.articulation.handle_geom` 所属 body；
- target container 使用 `allowed_robot_contact_roles.target_container` 各 geom 所属 body；
- robot 使用 backend 的 robot root body 闭包；
- floor/support 使用 `floor_support` geom 所属 body；
- 其余具名非 robot body 为 `other_furniture`；
- 同一 body 若被两个任务角色占用，合同 `invalid`，不得按优先级覆盖。

报告必须同时保存：

- segmentation 原始 int32 bytes/hash；
- `(timestamp_ns, sequence_id)`；
- camera name、width、height；
- head RGB/depth frame identity；
- dynamic calibration hash；
- geom→body→instance/role table 及 hash；
- segmentation 中 geom/site/background/unknown 像素计数。

### 可见与 anchor association

所有阈值在此冻结：

- 一个 exact task-object instance 在一个 candidate keyframe 中 `visible`：
  - segmentation 属于该 instance；
  - 对应 `head_depth_valid=true`；
  - 像素数 `>=8`。
- Episode `any_task_entity_visible`：任一 required object 在任一 candidate keyframe visible。
- Episode `all_task_entities_visible`：该 task 的两个 required object 都至少在一个 candidate
  keyframe visible；不要求同帧。
- A4 final input 不参与可见分子，只作 identity 完整性守护。

每个正式 enumerated anchor 使用 `5×5` center patch：

- 只计 `head_depth_valid=true` 的像素；
- `known_support` 包含 exact manipulated-object instance、articulation、target-container、
  floor/support、other-furniture 和 robot；
- background、site 与 unknown 不进入 known support，但进入质量报告；
- exact instance/role 至少 4 个像素，且占 known support `>=0.80`，才赋该 label；
- known support <4：`unknown`；
- 首位 label <0.80 或并列：`mixed`。

不得根据 raw/final 结果改变 patch、4-pixel 或 0.80 门。

### Raw、component 与 final association

- raw candidate 继承其 `(frame_ordinal,row,column)` anchor label。
- component 必须保存组成它的全部 raw provenance。
- component exact-instance label：
  - 至少两个不同 observation identity 的 raw member 标为同一 exact instance；
  - 该 instance 占 component 中全部 known、非 mixed raw member `>=0.80`；
  - 否则为 `mixed` 或 `unknown`。
- final candidate 继承其 pre-top64 component label；top64 不得改变 label。
- 一个 Episode 的 exact task-object raw/component/final coverage，只要求至少一个对应项；
  不把数量作为样本。
- `distractor_only_final_set`：final candidate 非空，但没有 exact manipulated object 或
  articulation；target-container、floor、furniture、robot、mixed/unknown 均不算
  manipulated-object hit。
- kitchen articulation 单独报告，不与两个 cleaner 的 any/all manipulated-object 指标
  合并。

### Entity-conditioned 漏斗守恒

对每个 exact task object、每个 Episode：

1. enumerated entity-labeled anchor；
2. 各 anchor first-rejection stage；
3. raw accepted；
4. component terminal：`view_count_lt_2`、`aggregate_normal_zero`、pre-top64；
5. ranking terminal：retained 或 truncated。

要求：

- entity-labeled anchor 的 first terminal 计数守恒；
- raw provenance 对 component 分区守恒；
- component terminal 计数守恒；
- pre-top64 = retained + truncated；
- formal candidate canonical bytes、顺序、selected index 与 P50 historical capsule
  bit-identical。

Episode 的首次清空阶段：

- task object 不 visible：`task_entity_not_visible`；
- visible 但没有 entity-labeled enumerated anchor：`visible_no_labeled_anchor`；
- anchor 存在时，按正式 gate 顺序逐步扣除 first-rejection；首次 survival 变 0 的 anchor
  gate；
- 有 entity raw、无 entity component：使其最后一个相关 component 归零的 component gate；
- 有 entity component、无 retained final：`top64`；
- final retained：`retained`；
- mixed/unknown 导致不能决定：`association_unknown`，不得猜测为某 gate。

### 运行与产物

正式命令固定为：

```text
.venv/bin/python -m hwr.apps.evaluate_entity_candidate_coverage \
  --plan runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/plan.json \
  --historical-capsules runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/capsules.json \
  --output runs/research-loop/0011/r0011-p50-e3-entity-coverage-s20265003
```

产物至少包含：

- `plan.json`：输入 plan 的不可变副本与 hash；
- `sidecars.json`：24 Episode sidecar 索引；
- `terminals.json`：24 个主运行与 validation replay；
- `report.json`；
- `manifest.json`；
- 每 capture 的 segmentation binary blob；
- 映射表与递归 XML/source identity。

同一 output 已存在时必须拒绝覆盖。写入使用临时目录和原子 rename。

### 主要指标与分账

样本单位为 Episode，完整报告 24/24：

- any/all required task object visible Episode count/rate；
- visible 条件下 any/all raw/component/final coverage Episode count/rate；
- distractor-only final-set Episode count/rate；
- selected-index exact association，只作描述性；
- 每种首次清空阶段的 Episode count；
- mixed/unknown Episode count；
- kitchen articulation visible/raw/component/final，独立附表；
- 3 task；
- 12 task×observation-latency×action-latency cell；
- observation latency、action latency 各自边际分账。

不对 24 Episode 做帧级或像素级显著性检验。比例仅为固定 cohort 的完整描述。

### 诊断假设门

合同有效后按以下顺序解释：

1. `acquisition_visibility_bottleneck_supported`：
   `any_task_entity_visible <18/24`。
2. `candidate_entity_bottleneck_supported`：
   - `any_task_entity_visible >=18/24`；
   - 在 visible Episode 中，any exact task entity final coverage `<=50%`；
   - 同一个首次清空 stage 覆盖 visible-but-uncovered Episode 的 `>=50%`；
   - 该 stage 至少出现在 2 个 task 且 observation latency 1/2 都出现。
3. `candidate_entity_coverage_not_primary`：
   - `any_task_entity_visible >=18/24`；
   - visible Episode 中 any exact task entity final coverage `>=80%`。
4. 其他：`diagnostic_inconclusive`。

这些是当前固定 cohort 的瓶颈路由，不是能力 accepted/rejected 门。

### 合同判定

`accepted as entity-conditioned candidate coverage measurement evidence` 仅当：

1. 24/24 主 Episode 和 24/24 validation replay 完整；
2. plan、historical capsule、source、scene、mapping、blob 与 manifest identity 全通过；
3. sidecar identity、RGB-D/calibration 对齐和 renderer-mode restoration 全通过；
4. formal candidate bytes、selected index、policy-input/proposed/applied action、physics trace、
   terminal 与 historical P50 bit-identical；
5. enabled/disabled replay 行为 bit-identical；
6. private truth AST/import/integration isolation 全通过；
7. entity-conditioned anchor/component/ranking 守恒全通过；
8. 24 Episode、全部 task/cell、mixed/unknown 与异常完整报告。

`invalid`：

- 任一 identity、行为、isolation、determinism、mapping、守恒或完整性门失败；
- segmentation 使用当前/latest/future state；
- renderer mode 污染 RGB/depth；
- sidecar 进入正式 observation、candidate、selector、动作或安全；
- 后验改 mapping、阈值、分母或分类。

本测量合同不因某个实体覆盖率高低而 rejected；指标按上节路由假设。

### 停止条件与资源

- 首个 private-truth leakage、historical behavior drift 或 hard safety failure：立即停止正式
  run，保留 failure artifact，不补 seed。
- 单个 Episode 的 segmentation/mapping 缺失：整项 invalid，不跳过 Episode。
- 预算：48 acquisition、47,760 control step、wall time 上限 90 分钟、RSS 上限 8GiB、
  新 artifact 上限 2GiB。
- 这是无训练测量，不休眠；主 Agent 持续轮询到结束。

### 允许声明

最多声明：

- measurement contract 是否有效；
- 固定 P50 cohort 的 task-entity visibility、raw/component/final coverage；
- 固定 cohort 的 candidate/entity 瓶颈路由。

禁止声明物体识别、语义理解、affordance、学习、泛化、交互、抓取、任务成功、安全改善、
deployment 或硬件迁移。

## `R0001-P57`

### 距离与 actual-applied command budget

对每个 pair 的 `frame_fixed`：

- `left/right d0`：`tool_distances[0]`；
- `left/right d100`：`tool_distances[-1]`；
- `left/right minimum`：101 个 carried distance 的最小值；
- `left/right normalized_AUC`：steps 1..100 distance 平均除以
  `max(d0,0.05m)`；
- `both_arms_improved`：left 与 right 都满足 `d100 < d0`；
- `max_arm_endpoint_distance = max(left_d100,right_d100)`。

actual-applied command budget 只使用落盘 applied action：

```text
left_budget_m =
  sum_t ||applied_action[t][2:5]||_2 * 0.30m/s / 20Hz

right_budget_m =
  sum_t ||applied_action[t][8:11]||_2 * 0.30m/s / 20Hz
```

- 不使用 proposed action；
- action latency 留下的 B1/base action自然计为零 arm budget；
- 不把 budget 称为 actual tool path、reachable set 或 collision-free path。

`initial_command_margin = budget_m - d0`。

### 同步 readiness

阈值结果前固定为 `0.10m`：

- 每一步 `bilateral_ready` 仅当同一 step 的 left、right distance 都 `<=0.10m`；
- `ever_bilateral_ready`：101 个同步 step 中至少一个成立；
- `endpoint_bilateral_ready`：step 100 成立；
- 禁止分别选择左右臂各自最佳时刻。

### Contact-transition nominal budget

从 bank 的 exact selected candidate、B2 base pose 与 preposition targets重建 B3/B4 contact
targets并交叉校验：

- B3 nominal maximum：`50 × 0.03 / 20 = 0.075m`；
- B4 nominal maximum：`20 × 0.02 / 20 = 0.020m`；
- total nominal maximum：`0.095m`；
- 每臂 `contact_transition_margin = 0.095m - ||contact-preposition||`。

该指标不执行 B3/B4，不称 actual-applied 或物理结果。

### 运行与产物

正式命令：

```text
.venv/bin/python -m hwr.apps.evaluate_precontact_reachability \
  --bank runs/research-loop/0010/r0010-p51-e1-bank-s20265101/bank.json \
  --terminals runs/research-loop/0010/r0010-p51-e1-convergence-s20265101/terminals.json \
  --output runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701
```

输出：

- `report.json`；
- `pairs.json`：36 个 pair 的逐臂完整重算；
- `manifest.json`。

同一输入分析两次，canonical report 与 pairs 必须 bit-identical。

### 诊断判定

`accepted as bilateral pre-contact reachability measurement evidence` 仅当：

1. 36/36 pair、72 arms、所有 101-step distance 和 100-step applied action 完整；
2. 输入 artifact/source/manifest/hash 完整；
3. 落盘 P51 d0/d100/min/AUC 与 raw distances 重算一致；
4. applied budget 只由 raw applied action 重算；
5. exact target reconstruction 与 bank identity 一致；
6. 12 cell、3 task 与 latency 分账完整；
7. determinism、有限值、bounds 与样本单位门通过。

合同有效后：

- `precontact_support_deficit_supported`：
  - `ever_bilateral_ready <=6/36`；
  - 至少 `30/36` pair 的两个 initial command margin 均 `<0`；
  - 每个 task `ever_bilateral_ready <=4/12`。
- `precontact_support_deficit_rejected`：
  - `ever_bilateral_ready >=24/36`；
  - 每个 task `ever_bilateral_ready >=6/12`。
- 其他：`diagnostic_inconclusive`。

若 input/identity/raw recomputation 不成立，合同 `invalid`，不发布诊断判定。

### 停止条件、成本与声明

- 纯 CPU 离线分析，wall time 上限 5 分钟，RSS 上限 2GiB，artifact 上限 100MiB。
- 不运行 MuJoCo、不休眠、不训练。
- 最多声明固定 P51 cohort 的 bilateral precontact readiness 与 command-support deficit。
- 不得据此修改 phase、速度、gripper、IK/FK、candidate 或 selector；不得声明接触、抓取、
  任务能力、泛化或安全改善。

## 实施前门禁

1. 本文件与 `00-context.md`～`02-review.md` 提交到 `feat/research-loop`。
2. 两名实施 Agent 从该冻结提交开始。
3. 每项行为变化有测试。
4. 相关 focused tests、Python size、architecture、physics integrity、compileall 和
   `git diff --check` 通过。
5. 主 Agent 对 private truth isolation、artifact provenance、raw recomputation 与 output
   overwrite protection 做独立审查。
6. 正式运行只能从干净、已提交的 source commit 启动。

## 结果后的路由

- P50-E3 合同 invalid：停止，不实施 generator 或动作修订；下一轮修复测量合同。
- visibility bottleneck supported：下一轮只允许提出 acquisition/viewpoint 单主变量。
- candidate entity bottleneck supported：下一轮只允许针对重复 entity-conditioned 首次
  清空 stage 提出一个 generator 单主变量。
- candidate coverage not primary：下一轮优先动作支持链，但仍需重新筛选。
- P57 support deficit supported：下一轮可提出一个 phase-entry/readiness controller 主
  变量，但不得直接后验加时长或速度。
- P57 deficit rejected：不能再以简单 command budget不足解释零交互。
- P58 只有 P50-E3 entity-hit 支持、P57 readiness 支持且 P56 接受后才可重新筛选。
- 无论本轮结果如何，都不自动授权 selector、Replay、Actor、世界模型训练或 capability
  claim。
