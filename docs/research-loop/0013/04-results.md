# R0013 结果

## 结果总表

| ID | 判定 | 结论边界 |
|---|---|---|
| `R0001-P66-E1` | `accepted as predictive-safety witness contract` | P60 anchor 的 clone predictor 在第二个 control boundary 由 robot base 与 tea table 的单 contact point `356.9928N` 触发 `220N` 门 |
| `R0001-P72-E1` | `accepted as residual P61 contract gap evidence` | P61 对五类 exact-reference drift 不 fail closed，且 role field 可被误当成 planner call state；initial annotation 四项反事实均 fail closed |
| `R0001-P68-E1` | `inconclusive_budget_exceeded` | 固定 24-Episode association cohort 未在 30min 内完成；无正式或部分 artifact，不发布 association 分账 |
| `R0001-P67` | `deferred` | P66/P68 前置未共同形成后续运行许可 |
| `R0001-P69` | `deferred` | 仅有单 anchor，不能外推 controller 因果 |
| `R0001-P70` | `deferred` | P68 未完成，candidate-conditioned paired routing 不授权 |
| `R0001-P71` | `deferred` | 保留为下一轮高优先级替代路线 |

本轮没有训练、参数更新、checkpoint、policy inference、B2/B3–B7 action、contact phase、
capability Episode 或新家务任务成功。

## `R0001-P66-E1`

### 实施与结果后修复

初始实现提交：

`df760dae76961ba5caeb1bc698d62831bf558bf3`

首次正式 artifact 在 source commit
`199c5a8f171e6bc553652cd4734a85a410ce8cbc` 上得到 accepted。独立审计发现 P66
为获取 boundary witness，在共享
`src/hwr/adapters/mujoco/dual_arm_backend.py` 增加了默认 no-op hook。虽然 observer
disabled/enabled 的行为与状态 bit-identical，但该共享 backend blob 改动触发了 P51
冻结 provenance。

修复提交：

`6c4dc2339c6262fdc41b736a96dab586ae70c6d9`

修复后：

- `dual_arm_backend.py` 恢复到 R0010 冻结 blob
  `39941d7b8721b210ac3c2cd8275069387919f931`；
- production `_predictive_filter()` 完全恢复原实现；
- P66 专用 subclass 包裹 production filter，并覆写 production
  `_predictive_safety_violation()`，在原 predicate 计算完成后只读同一 trial `MjData`；
- witness 不进入 `RuntimeStepOutcome.info`、event、observation、candidate、action 或
  training data。

旧 artifact 原字节保存在：

`runs/research-loop/0013/r0013-p66-predictive-witness-s20266601-superseded-199c5a8`

最终正式 source commit：

`bb35cd2a6a6f150188b3541350da1b233dbca347`

### 正式命令

```text
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_predictive_safety_witness \
  --p60-input runs/research-loop/0012/r0012-p60-phase-entry-s20266001 \
  --output runs/research-loop/0013/r0013-p66-predictive-witness-s20266601
```

### 核心结果

- P60 anchor identity 全部复现：
  - planned Episode：
    `838dd73530fc555aefa4084574fccb4b5a9ca91496f2cca6791f702d95a27bb8`
  - candidate set、selected index、runtime randomization、policy input sequence、
    raw prefix trace、prefix step count 全部匹配；
  - rejection step：`1279`。
- action lineage：
  - policy proposal base：`(0.12, -0.1822977766101701)`；
  - action latency FIFO source step：`1278`；
  - delayed/scaled plant action base：
    `(0.10563094615897749, -0.1605351261728603)`；
  - safety applied base：`(0.0, 0.0)`。
- predictor：
  - boundary ordinal：`2`；
  - cumulative physics substep：`50`；
  - maximum forbidden **contact-point** normal force：
    `356.9927528897417N`；
  - robot geom/body：`body_box_collision` / `robot_base`；
  - environment geom/body：`tea_table_top_collision` / `tea_table`；
  - 同一 pair 的另一个 contact point 为 `121.60432009046283N`；
  - production 判据与从 raw contact points 独立重算的 witness 一致。
- observer disabled/enabled：
  - prefix、event、action、qpos、qvel、time、queue、targets、runtime counters
    bit-identical；
  - authoritative `physics_advanced=false`；
  - actual severe collision count `0`；
  - invalid force `0`；
  - P40 conservation difference `0.0`。

因此：

`accepted as predictive-safety witness contract`

允许声明：

> 固定 P60 anchor 在 production two-control-step predictor 的第二个检查边界，预测到
> robot base 与 living-room tea table 顶面的单 contact point normal force 为
> `356.9928N`，超过冻结 `220N` 门，因此危险 plant action 被改写为 hold，且未提交到
> authoritative physics。

不得声明实际发生碰撞、该事件代表总体发生率、所有 B1 path 都有问题，或调整 controller
一定改善。

### Artifact 与资源

| 文件 | SHA-256 |
|---|---|
| `disabled.json` | `42bc8430ca4a731ceb9dfef46395ea58638edfc6507eed833e2134003b626014` |
| `witness.json` | `a1a38c1d1cbdad3c2a435f3c6fb3713d41facd7264f41635030bd7947519478e` |
| `report.json` | `d542182ad3a4552f22853cd2e085b6083a3b6c378384fc29425cb1b26a5f5718` |
| `manifest.json` | `a6a17bfad73a79c8014724ddecaed5ba0b3469ddc17ce3357530c050e9713ad7` |

- wall time：`87.08585729100741s`
- peak RSS：`648,429,568 bytes`
- artifact：约 `78MiB`

## `R0001-P72-E1`

### 实施与结果后修复

初始实现提交：

`fa88d3600eb24fa2a4b749dfee1ce64368a75cc0`

首次正式 artifact 在 source commit
`199c5a8f171e6bc553652cd4734a85a410ce8cbc` 上运行。独立审计发现
`independent_planner_state_or_call_evidence` 被常量设为 false，导致 planner residual
虽方向正确但有自证成分。

修复提交：

`406af7d105a58b0a4375a1403ab74cc9ebda123d`

修复后 independent planner evidence 同时要求：

- direct caller 的 planner call state 可见；
- contract planner fields 非空；
- `transition_id_available=true`；
- `validated_external_planner_present=true`。

role-only mutation 的正式结果为：

- `planner_call_state_available=true`；
- `independent_planner_state_or_call_evidence=false`。

旧 artifact 原字节保存在：

`runs/research-loop/0013/r0013-p72-p61-mutation-s20267201-superseded-199c5a8`

最终正式 source commit：

`0edd113d82027587055f0980f3873ae762c0fff1`

### 正式命令

```text
.venv/bin/python -m hwr.apps.audit_interaction_contract_mutations \
  --contract configs/eval/interaction_contract_v1.json \
  --p61-input runs/research-loop/0012/r0012-p61-interaction-contract-s20266101 \
  --output runs/research-loop/0013/r0013-p72-p61-mutation-s20267201
```

### 核心结果

- 14/14 frozen mutation 执行；
- 14/14 从 clean copy 开始；
- 14/14 到达 parser/auditor；
- 14/14 单变量成立；
- 14/14 mutated source 可编译；
- 14/14 重复运行 canonical bytes bit-identical；
- baseline 与 P61 transitions bit-identical；
- positive control 可把 full contract verdict 翻转；
- interaction、destination、articulation-threshold 三个 negative control 分别恢复预期
  gap；
- P68 dependency mutation 6–9 全部 fail closed；
- `p68_dependency_gate_passed=true`。

残余 gap：

1. candidate schema extra field；
2. policy schema extra field；
3. primitive signature extra argument；
4. selector signature extra argument；
5. primitive phase extra value；
6. role field 可使 current `planner_call_state_available` 为 true，但没有独立 planner
   fields、transition ID 或 validated planner evidence。

因此：

`accepted as residual P61 contract gap evidence`

这会收缩 P61 结论：

- P61 的 transition reconstruction、当前 direct-call information gap 和 initial
  annotation 仍有证据；
- P61 的 exact-reference drift 不是当前 final verdict 的 hard dependency；
- `planner_call_state_available` 不能单独当作 validated external planner evidence；
- 不得把 P61 外推为 whole-program planner absence proof。

### Artifact 与资源

| 文件 | SHA-256 |
|---|---|
| `mutations.json` | `e3a2732365953f49981730491cab07268ec16dbc2c2023d36cb0270a89d70d70` |
| `report.json` | `5964666a44f79f7b2d5938eda863b489d96d4546bb40d2e2fcd894751b109943` |
| `manifest.json` | `d3bb084ceead011862cbf3f8cf27a407dfe328063ce1cce6dd54af6a16892a30` |

- wall time：`32.99177745805355s`
- peak RSS：`180,830,208 bytes`
- artifact：约 `520KiB`

## `R0001-P68-E1`

### 实施

- 初始实现：`8b6ab26273f3050d77da809cb029b5b5b663e3de`
- canonical support 修复：
  `f9ee12bc04f3c5443f7da0bc373e194c35bc1283`

实现门禁：

- segmentation 在 source observation 生成时捕获，并按 observation latency queue 的
  timestamp/sequence identity 取回；
- 单 Episode observer-off/on replay 的 27 个冻结 identity 字段全部一致；
- 24/24 历史 P50 capsule 可从原 policy input 离线重建相同 candidate-set SHA-256 与
  candidate count；
- raw support mask 保留原 generator 的扫描顺序语义；该 generator 使用 view 上的
  in-place `patch_valid &= ...`，因此具有 order-dependent mask 收缩，本轮只忠实复现，
  不修改 generator。

### 正式命令

```text
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_initial_candidate_association \
  --p50-input runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --mapping-input runs/research-loop/0012/r0012-p50-e4-mapping-s20265004 \
  --interaction-input runs/research-loop/0012/r0012-p61-interaction-contract-s20266101 \
  --p72-input runs/research-loop/0013/r0013-p72-p61-mutation-s20267201 \
  --output runs/research-loop/0013/r0013-p68-initial-association-s20266801
```

### 正式运行时序

1. source commit `8b6ab26` 的第一次正式尝试在约 `9.6min` 时，dining Episode
   `7e039594...` 的 independent support tuple 与 official candidate 的浮点末位不完全
   相等，runner fail closed。未发布 output 或 `.tmp`。
2. `f9ee12b` 将 support component 按 frozen canonical record 绑定回 official candidate，
   并新增 24/24 capsule 回归；所有 candidate-set hash/count 均一致。
3. source commit `f9ee12b` 的第二次正式尝试执行固定 24-Episode cohort，不替换 seed，
   不读取中间 association 结果，不发布 partial artifact；在冻结 `30min` wall-time
   hard gate 触发后退出。
4. output
   `runs/research-loop/0013/r0013-p68-initial-association-s20266801`
   与对应 `.tmp` 均不存在。

因此：

`inconclusive_budget_exceeded`

没有可发布的 24-Episode classification、task 分账或高/低 association gate。不能从
实现期单 Episode smoke、未发布内存状态或执行顺序推断 selector relevance。

## 验证

### Focused

- P66 + runtime：27 passed。
- P61/P72：23 passed。
- P68 + P50/selector/isolation：43 passed。
- P68 final core：11 passed。

### 仓库门禁

- Python size：450 files 通过。
- architecture：通过。
- physics integrity：通过。
- compileall：通过。
- `git diff --check`：通过。
- P51 frozen backend provenance 在 P66 隔离修复后恢复通过。

### 全量 pytest

- 收集：`1202` tests；
- `1190 passed`；
- `11 skipped`；
- 18 条 warning 均为既有 `torch.jit.script` deprecation；
- `1 failed`，唯一失败：
  `tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete`。

该失败在 R0013 起始提交
`e49e3d7112d0d4773475f53deaa7e97a5c20f6ad` 的独立 detached worktree 已复现。
原因是 R0012 P50-E4 runner冻结了实现前 `00-context.md` blob，而 R0012 收尾修改了同一
context 文件；不是 R0013 代码或能力回归。本轮不修改历史 R0012 文档或合同。

## 独立审计

独立审计初始结论：

- blocker：0；
- major：1，P72 planner residual 使用常量 false；
- minor：2，P68 latency test 判别力与 wall-time检查粒度。

处理：

- major 已由 `406af7d` 修复，并从新 source commit重跑 P72；
- 修复后快速复审：blocker `0`、major `0`；
- P68 latency alignment 由真实 latency=1 Episode 的 baseline/treatment 27-field
  identity、source identity 与 candidate bytes一致性验证；
- wall-time minor 未改变科学结果：第二次正式 run由 frozen 30min gate停止，且无 output
  或 `.tmp`；P68 判 inconclusive，不使用部分结果。
