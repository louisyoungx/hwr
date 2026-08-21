# R0008 结果

## `R0001-P40-E2`：`accepted as entity-contact measurement contract evidence`

### 完整性

- 冻结文档提交：`b56ee96953652e2e80d644b8167181e8449c0a8b`
- 初始实现提交：`b578da6a46a7d02fd2e5fa437dcff02521f4f06e`
- reset-settling 语义修复提交：`cefb240b1190fea8021fc3580ba171ed12ad58b0`
- 冻结提交是两个实现提交的祖先。
- 最终正式 run：
  `runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002`
- 稳定命令：

```text
.venv/bin/python -m hwr.apps.evaluate_entity_contact_graph \
  --output runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002
```

- mode：formal measurement；
- device：CPU；
- 每任务 fixed hold 32 control step，共 96 control step；
- 每任务 800 physics substep，共 2,400 physics substep；
- policy inference：未执行；
- closed-loop capability Episode：未执行；
- 参数更新/checkpoint：无；
- 最终 run 磁盘占用：2,932 KiB。

最终产物：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 2,994,715 | `ac311f4b7ed05010d5e5b940ab40675df584fa65018f9e9d9a2daa7ea8d956d1` |
| `manifest.json` | 3,541 | `0cb716b9616b1aa9a4199dc64e7734ab1c52283b0898a58a0742fc298c988109` |

report 与 manifest 的 `source_commit` 均为
`cefb240b1190fea8021fc3580ba171ed12ad58b0`。manifest 记录冻结文档祖先、稳定命令、binding
identity、robot body roots、MuJoCo/physics 配置、直接产物 hash/bytes 和全部禁止声明位。

### 主 Agent 发现与修复

初始提交 `b578da6` 的第一次正式 run 通过了当时实现的自动检查，但主 Agent 独立审查发现：

- 冻结合同要求 reset settling 不得成为 contact-associated motion；
- 初始实现若首个 control period 同时发生 robot contact 与实体位移，仍会把位移关联到
  contact；
- 这不会改变 runtime 行为，但会污染未来 P41 的测量语义。

因此没有接受该 run 作为最终证据。原产物未删除，保留为：

`runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002-superseded-b578da6`

其身份：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 2,978,008 | `14943993c4f55004d240c5193a9e0af3f1075120093f3b9f4ee96f02fd6460bc` |
| `manifest.json` | 3,337 | `c48a93f0fbf8190ef190f08f261ed23db06d87296a628c67a313ddf4d46c8c75` |

修复提交 `cefb240` 显式冻结：

```text
reason = reset_settling
rule = period_index < excluded_initial_periods
excluded_initial_periods = 1
```

首个 period 仍保留原始 contact 和 motion，但
`contact_associated_motion = 0`；从第二个 period 起，只有同一实体在同一 period 出现
robot contact 时才关联该 period 的实体位移。fixture 同时验证首 period 排除和后续 period
正常关联。

### 映射与分类合同

三任务均通过：

- 48 个 robot geom 全部且唯一映射为：
  - `base`：12；
  - `left_arm`：18；
  - `right_arm`：18；
- body root 名称固定为：
  - `robot_base`；
  - `left_shoulder_pan_link`；
  - `right_shoulder_pan_link`；
- manipulated object、floor/support、target container、articulation 与 forbidden
  environment geom 全部可解析；
- robot–environment 与 task-relevant world–world 分账；
- same-entity dual-arm、distinct-entity dual-arm、single-arm 和 same-object dual-arm
  grasp 分账；
- contact-associated motion 与 no-contact motion 分账；
- evaluator-private entity graph 没有进入 policy、训练数据、reward、安全或 success。

纯逻辑 fixture：

- classification precision：`1.0`；
- classification recall：`1.0`；
- 同一 geom pair 多 contact point 先求和；
- same-entity、distinct-entity 与 single-arm 反例全部通过；
- object–container、object–support、object–object 与 articulation edge 全部通过；
- reset settling、无接触运动和接触结束后惯性运动反例全部通过；
- missing、NaN、Inf、negative force 全部 fail-closed；
- missing、overlap、unknown root、unknown entity role 映射全部 fail-closed；
- valid fixture 的全部 invalid count 为 0。

### P40-E1 守恒与行为不变

P40-E2 与 P40-E1 在相同 substep 并行读取 contact。robot–environment 子账聚合回 P40-E1
后的最大绝对差：

| 任务 | 最大绝对差 |
|---|---:|
| 客厅 | `3.410605131648481e-13` |
| 餐桌 | `2.2737367544323206e-13` |
| 厨房 | `3.410605131648481e-13` |

均通过冻结 `<=1e-12` 门。world–world edge 单独报告，不进入上述守恒式。

三任务 measurement disabled/enabled 的 legacy trace 均 bit-identical，覆盖：

- applied action；
- proprioception；
- reward；
- terminated/truncated；
- success/reason；
- severe collision；
- maximum forbidden force/pair；
- safety intervention。

所有 formal trace 的 missing、nonfinite、negative 与 unknown mapping count 均为 0。

### 固定 hold trace 的描述性结果

- 三任务均为 32 control step、800 physics substep。
- 三任务都只观察到 `base × floor_support:floor` robot–environment edge。
- 客厅额外观察到两条 task-relevant world–world edge：
  - `floor_support:floor × manipulated_object:duck`；
  - `floor_support:floor × manipulated_object:football`。
- 餐桌和厨房固定 hold trace 未观察到 task-relevant world–world edge。
- 三任务均未观察到 same-entity dual-arm、distinct-entity dual-arm、single-arm 或
  same-object dual-arm grasp。
- 排除首个 settling period 后，三任务均无非零 contact-associated motion。

这些结果符合 fixed hold 的预期，只证明分类与观测合同可运行；不证明未来 P41 一定能产生
物体交互。

### 判定

全部冻结检查通过，正式标记：

`accepted as entity-contact measurement contract evidence`

报告固定：

- `measurement_only=true`
- `policy_inference_executed=false`
- `closed_loop_capability_episode_executed=false`
- `capability_claim_allowed=false`
- `hardware_safety_claim_allowed=false`
- `action_causality_claim_allowed=false`
- `legacy_runtime_behavior_unchanged=true`

该结论只接受实体接触图、接触同期运动和守恒测量合同。它：

- 不改变当前能力基线；
- 不建立 allowed-contact 或硬件安全阈值；
- 不把 contact-associated motion 称为 controlled motion 或 action causality；
- 不授权正式训练；
- 只允许下一步结果前冻结 P41-E2。

## 验证

- 实施 Agent：
  - P40-E2 focused tests：28 passed；
  - Python size、architecture、physics integrity、compileall、diff check 通过。
- 主 Agent：
  - P40-E2/P40-E1/formal backend focused suite 在宿主环境通过；
  - 修复后全量 `.venv/bin/python -m pytest -q` 通过；
  - 11 项既有 skip；
  - 18 条 warning 均为 `torch.jit.script` deprecation；
  - Python size：391 files 通过；
  - architecture、physics integrity、compileall、`git diff --check` 通过；
  - report/manifest source commit、冻结祖先、binding identity、claim flags、artifact
    hash/bytes、settling exclusion、三任务 period 数与 P40 守恒逐字段核验通过；
  - 历史 `docs/research-loop/0001/`～`0007/` tree 零差异。

## 当前能力结果

- 正式训练：未启动；
- policy inference：未执行；
- 闭环能力 Episode：未运行；
- 新家务任务成功：0；
- qualified deployment：无；
- 世界模型、Actor、任务成功率、泛化或硬件安全改善：无声明。

