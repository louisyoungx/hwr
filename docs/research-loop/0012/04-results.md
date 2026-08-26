# R0012 结果

## 结果总表

| ID | 判定 | 结论边界 |
|---|---|---|
| `R0001-P61` | `accepted as interaction-contract gap evidence` | 当前有限静态 direct-call 边界缺少 transition、entity role、interaction selector 与 destination 输入 |
| `R0001-P50-E4` | `accepted as exact-geom evaluator mapping contract` | 三个冻结 MuJoCo 场景的 exact-geom + 一跳 same-body visual alias 可确定构造 |
| `R0001-P60` | `invalid` | 首个正式 cell 的第二个 latency-matched prefix 触发预测安全拒绝，按冻结合同立即停止；无 geometry 诊断结论 |
| `R0001-P62` | `deferred` | 缺少独立 joint/collision/safety feasibility witness |
| `R0001-P63` | `rejected` | 全 unknown 平凡解、同场景拟合与 P50-E4 重复 |
| `R0001-P64` | `deferred` | P62 前置与 final-set direction gate 未满足 |
| `R0001-P65` | `deferred` | P60 未产生可归因的 heading/time-allocation 结果 |

本轮没有训练、参数更新、checkpoint、policy inference、B2 action、post-prefix action、
contact phase、capability Episode 或新家务任务成功。

## `R0001-P61`

### 实施与结果后修复

初始实现提交：

`e51c1ea02aef5e97d7b3f6f97254872f95212035`

首次正式 artifact 在 source commit `be965f8` 上得到 accepted。结果后独立审计发现：

- `caller_role_gap_present` 与 `validated_external_planner_present` 由常量给出；
- transition availability 与 final verdict 没有从实际 schema、signature 和 direct caller
  逐项推导；
- 测试主要复述固定值，形成自证式判定；
- in-process test 使用整个 pytest 进程的历史 peak RSS，受此前测试影响。

这属于合同执行缺陷，不改变冻结 transition、输入、阈值或判定语义。旧 artifact 原样保存在：

`runs/research-loop/0012/r0012-p61-interaction-contract-s20266101-superseded-be965f8`

修复提交：

`6bf0400f51a25bfb6f45e951299c410efd5c2c7a`

修复后：

- verdict 由冻结 contract requirement、`Candidate` 字段、policy schema、selector/primitive
  signature 和 same-function direct-call AST 逐项推导；
- 新增可执行反事实 fixture，可把 planner/role/interaction/destination 能力打开并使 verdict
  从 accepted 翻转为 rejected；
- 分别验证 destination 与 drawer threshold 能独立改变 transition 判定；
- 明确审计只覆盖 `finite_static_same_function_direct_calls`，不声称覆盖 cross-function
  dataflow、dynamic dispatch、reflection、runtime value 或 whole-program planner proof；
- 正式 `<1GiB` RSS 门不变，单元测试改为注入隔离 resource probe。

### 正式命令

```text
.venv/bin/python -m hwr.apps.audit_interaction_contract \
  --contract configs/eval/interaction_contract_v1.json \
  --output runs/research-loop/0012/r0012-p61-interaction-contract-s20266101
```

### 核心结果

- source commit：`6bf0400f51a25bfb6f45e951299c410efd5c2c7a`
- 7/7 transition 从冻结 task/binding/runtime predicate 重建；
- 7/7 transition 在当前直接调用边界均不能由一个 selected candidate 与 generic
  B0–B7 primitive 唯一表达并实现；
- `Candidate` 只有 center、normal、width、prominence、support/view count 和首次观测索引；
- primitive 只有 serialized input、candidate、acquisition base pose、post-selection step；
- direct caller 没有提供 entity/role identity、required interaction type、destination target
  或 articulation threshold；
- 当前调用边界没有验证过的 external planner；
- 三任务 reset 后首次 microinteraction 的 evaluator-only role 可唯一冻结：
  - living：duck 或 football；
  - dining：cup 或 plate；
  - kitchen：drawer articulation。

因此：

`accepted as interaction-contract gap evidence`

允许声明：

> 在 commit `6bf0400` 的有限静态 same-function direct-call 边界中，当前 generic
> candidate-centered primitive 缺少完成七个 full-task transition 所需的 entity role、
> interaction selector 与 destination 信息。

不得声明所有可能的动态 planner/dataflow 已被排除，也不得据此宣称物理不可达或解释全部
任务失败。

### Artifact

| 文件 | SHA-256 |
|---|---|
| `transitions.json` | `1cc139e7f8b02a6325d16282f9b7882e9736c40d03f56644c1739d79ee7bcc0a` |
| `report.json` | `d9a760eaa30198eda95d20e90a4ebf4c9d9f5bcd2e118b6e139446866545719a` |
| `manifest.json` | `6a019a7591a2614c6082dea102c29f9cb24e101f78da8ce21ce3725f60df221d` |

## `R0001-P50-E4`

### 实施与结果后修复

初始实现：

- `6c3abee111d84a65add72c5f4cdb93c39fdae667`
- architecture boundary 修复：
  `be965f868c3c21e234130d6b5e9f725bf73efc78`

首次正式 artifact 得到 accepted。结果后独立审计发现两个 minor：

1. app 内硬编码历史 tree，但没有直接绑定并解析 `00-context.md`；
2. 非法 segmentation object type 只在 unit test fail closed，未进入正式 report guard。

旧 artifact 原样保存在：

`runs/research-loop/0012/r0012-p50-e4-mapping-s20265004-superseded-be965f8`

修复提交：

`c9d1f01afc3214fd239e505ec6865e6e33c5f5c3`

修复后 app：

- 绑定 frozen `00-context.md` 的 commit、content 与 blob；
- 从 context Markdown 解析 0001–0011 tree inventory，再与当前 Git tree 对账；
- 将非法 segmentation object type 的 deterministic fail-closed case 纳入正式三场景 guard；
- 保持 MuJoCo import 在 adapter boundary 内。

### 正式命令

```text
.venv/bin/python -m hwr.apps.audit_entity_candidate_mapping \
  --aliases configs/eval/entity_candidate_aliases_v1.json \
  --output runs/research-loop/0012/r0012-p50-e4-mapping-s20265004
```

### 核心结果

- source commit：`6bf0400f51a25bfb6f45e951299c410efd5c2c7a`
- 3/3 scene preflight 通过；
- 8/8 frozen alias 一跳、同 body、canonical target exact-claimed；
- exact geom role conflict：`0`；
- task-visible inventory unknown：`0`；
- 40 个正式负守护全部通过：
  - sideboard body/top 未误标 target-container；
  - drawer handle visual 与六个 container surface 保持不同角色；
  - drawer frame 未误标 task role；
  - wall/rug 未因 world body 被传播成 floor-support；
  - site 为 `unknown_site`；
  - background 为 `background`；
  - 非法 segmentation object type fail closed 为 unknown；
- 271 个 Python source 的静态 import/AST isolation violation：`0`；
- 两次 table 构造 bytes/hash bit-identical；
- table SHA-256 在修复前后保持：
  `88540ddb87e6df129eaaa9666b0011598a422629b1085c27b7b844e735918fbc`。

因此：

`accepted as exact-geom evaluator mapping contract`

该结果只证明三个冻结场景的 evaluator mapping 可构造，不是 entity visibility、candidate
coverage、识别、泛化或任务能力证据。

### Artifact

| 文件 | SHA-256 |
|---|---|
| `tables.json` | `88540ddb87e6df129eaaa9666b0011598a422629b1085c27b7b844e735918fbc` |
| `report.json` | `96fdc3abde155e09715bb7e1314c9b6cbef26dffdeda193f6dc336e4cd767402` |
| `manifest.json` | `7c950b5e132a0c24395a63ee6dec150f5244466cf5b3f38118a4efa00a2a3579` |

资源：

- wall time：`1.701241916976869s`
- peak RSS：`368,132,096 bytes`
- tracemalloc peak：`5,907,233 bytes`

## `R0001-P60`

### 正式命令

```text
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_phase_entry_geometry \
  --salt-file runs/research-loop/0012/.host/p60-salt.txt \
  --output runs/research-loop/0012/r0012-p60-phase-entry-s20266001
```

### 停止经过

- source commit：`be965f868c3c21e234130d6b5e9f725bf73efc78`
- salt commitment：
  `263e9f85e32f4a3f5f1560ba82cd820a558cc0aad9a5710bbdf6a3306e3f9c55`
- 扫描 raw seed：36 个，均属于首个
  `tidy_living_room_3d/v1 / observation latency 1 / action latency 1` cell；
- natural latency match：2 个；
- physical prefix：2 个；
- eligible prefix：0；
- selected Episode：0；
- B2 action generated/executed：0/0。

两个 physical prefix：

1. raw ordinal 12：
   - 995-step acquisition 完成；
   - candidate count `0`；
   - `candidate_set_empty`；
   - 无 safety intervention、severe collision、invalid force、terminal 或 P40 conservation
     drift。
2. raw ordinal 35：
   - candidate count `3`；
   - 在 prefix step `1279`、B1 base command
     `(linear=0.12, angular=-0.1822977766101701)` 时；
   - safety event：
     `action_rejected / predicted_severe_collision`；
   - safety 将运动 action 改写为 hold；
   - `prefix_safety_intervention_count=1`；
   - actual severe collision count `0`；
   - invalid force count `0`；
   - P40 conservation maximum absolute difference `0.0`；
   - runtime terminal `false`；
   - B2 action 未生成、未执行。

冻结合同规定首个 safety intervention 立即停止正式 run，清空 publishable cohort，不继续
后续 seed 或 cell。因此：

`invalid`

且：

- strict diagnostic：`null`
- nominal diagnostic：`null`
- 不能发布 task、cell、latency 或 geometry 分账；
- 不能声明 strict outer-envelope deficit supported/rejected；
- 不能声明 nominal B2 support deficit supported/rejected；
- 不能称为实际发生严重碰撞，只能称为预测安全层拒绝。

### 独立审计

独立审计结论：

- blocker：P60 cohort 因正确 hard stop 而无效；
- major：`0`；
- seed commitment、planned Episode ID、environment/policy seed、natural latency、trace
  hash、artifact bytes/hash 全部复算一致；
- 36 个 raw ordinal 连续为 `0..35`；
- 两个 latency-match audit 与 physical records 一一对应；
- safety rewrite 是真实记录，不是计数误触发；
- 没有发现 evaluator 实现缺陷，因此不允许修复、忽略 safety 或继续后续 seed；
- 不重跑同一 run。

### Artifact 与资源

| 文件 | SHA-256 |
|---|---|
| `plan.json` | `df69da8606f78f94fdaaecef0021a64eb33c8cc5e2b62720bdbdd1f5e2255e5a` |
| `seed-audit.json` | `5bc6fb284f14c689da0167a40e76775c83c2116fb9541d96d45371307dde71f3` |
| `episodes.json` | `681b2ac5f49d8af7fa21108e3adb96f1fcc0bbc894d2a3f6b2d544cb28f64c4e` |
| `report.json` | `1674278396eb51f647c261d6514b2411cb09ba786deda659ed189c6124841737` |
| `manifest.json` | `ee21c04f009d0ab89bc83f3f00516a36a91955f9369ab50d66bea0f04f9c75df` |

- wall time：`110.89117479196284s`
- peak RSS：`552,566,784 bytes`
- tracemalloc peak：`75,138,708 bytes`
- artifact：约 `6.11MiB`

## 验证

- 实施 Agent focused：
  - P61：19 passed；相关集 116 passed；
  - P50-E4：44 passed；
  - P60：98 passed。
- 主 Agent统一 focused：152 passed。
- 全量 pytest：
  - `1068 collected`
  - `1057 passed`
  - `11 skipped`
  - 18 条 warning 均为既有 `torch.jit.script` deprecation。
- 第一次全量命令未设置根目录 `PYTHONPATH`，在 collection 阶段出现 13 个既有
  `scripts`/`tests` import error；改用
  `PYTHONPATH=.:src MUJOCO_GL=glfw .venv/bin/pytest -q` 后全量通过。
- Python size：434 files 通过。
- architecture：通过。
- physics integrity：通过。
- compileall：通过。
- `git diff --check`：通过。
- P61/P50-E4 修复后独立复审：`0 blocker / 0 major / 1 minor`。
- 唯一剩余 minor：正式 artifact 位于 ignored `runs/`，manifest 自身尚未有目录外不可变
  锚点；本轮将其强制加入 Git 以关闭该风险。
