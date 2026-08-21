# R0008 结果

## `R0001-P40-E2`：`accepted as entity-contact measurement contract evidence`

### 完整性

- 冻结文档提交：`b56ee96953652e2e80d644b8167181e8449c0a8b`
- 初始实现提交：`b578da6a46a7d02fd2e5fa437dcff02521f4f06e`
- reset-settling 语义修复提交：`cefb240b1190fea8021fc3580ba171ed12ad58b0`
- 长轨迹守恒数值顺序修复提交：
  `8b79597a6af36233b1a2e6437c769db12567f88c`
- 冻结提交是上述实现与修复提交的祖先。
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
| `report.json` | 2,994,604 | `987a2217cf9f5c6eb08b018b3bf13164917c75bccbfa77140a8786c80987f841` |
| `manifest.json` | 3,541 | `fdb847a41f55a7a3bb362d650baa2d131e2a5178ac73166336d57368ba60546b` |

report 与 manifest 的 `source_commit` 均为
`8b79597a6af36233b1a2e6437c769db12567f88c`。manifest 记录冻结文档祖先、稳定命令、binding
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

P41 第一次完整 smoke 又发现第二个问题：

- P40-E2 的 episode category 在每个 physics substep 上累加；
- P40-E1 先生成 control-period category subtotal，再在 episode 层累加；
- 两者数学等价，但 1,655 control period 的长轨迹产生最大
  `2.837623469531536e-10` 浮点差，超过冻结 `1e-12` 守恒门。

修复提交 `8b79597` 只把 P40-E2 episode category 改为按 control period 累加，不改变
substep、period、edge 或 interaction 语义。新增 1,655 period × 25 substep 回归夹具后，
与 P40-E1 的六个 category 字段最大差为 `0.0`。

旧 `cefb240` 正式产物保留在：

`runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002-superseded-cefb240`

沙箱 CGL 失败证据保留在：

`runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002-superseded-sandbox-cgl-8b79597`

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
| 客厅 | `0.0` |
| 餐桌 | `0.0` |
| 厨房 | `0.0` |

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

## `R0001-P41-E2`：`inconclusive`

### 完整性

- 冻结提交：
  - `602adfe91d61c41d349e62925f2c3cd8379d567f`
  - `027eb0ba384ac7ac3100dca9046c64b0a109eb9c`
  - `565a881a6e09d3136bbc0f311d386b418b7b55fe`
- 初始实现提交：`8c47ec8ca9f084f0fcbf1bcdc30aa5a16d5589f5`
- P40-E2 provenance 重绑定提交：
  `bd8dd2aa1e43ae6d2e420952042dde8ff416330e`
- 最终 smoke：
  `runs/research-loop/0008/r0008-p41-target-selection-e2-smoke-s20264101`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_target_selection \
  --output runs/research-loop/0008/r0008-p41-target-selection-e2-smoke-s20264101 \
  --salt R0001-P41-E2-smoke-s20264101 \
  --smoke
```

最终产物：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `plan.json` | 16,232 | `2d0d9d11e5f6d8b82106e92501015dab05fb13e1bb4a81950e808755d45a9fb4` |
| `report.json` | 8,095 | `5f5e8d50eb2c66043cc92721091fd895cca2dd925531890a87e65dedfd8885a7` |
| `power.json` | 5,995 | `ed6f117a67bd1e290cdbf1302e1d666e3a20b6447e38e3a31a4707806a1f90f9` |
| `terminals.json` | 146,004 | `c061c906c82fa4a6e8a2b11137d8e03e1873c2960759dd2c399ae76a95282b3a` |
| `manifest.json` | 3,346 | `649a5fdd84e133270d05fd3f70977b6b212a0d1eda185bda4faed07136dc48fb` |

report 与 manifest 的 `source_commit` 均为
`bd8dd2aa1e43ae6d2e420952042dde8ff416330e`。6 个 planned pair 均有 terminal，
无 infrastructure unresolved；按自然 evaluation profile 拒绝的 30 个 seed 全部保留在
plan audit。

### 结果前功效

冻结 synthetic power 的候选 pair 数为 `36/54/72/90/108`，选择 54：

- 36 pair planted power 95% lower：`0.7464469149015809`，未通过；
- 54 pair planted power 95% lower：`0.8618210816892118`，通过；
- 54 pair 最坏 null FPR 95% upper：`0.021870261025271856`，通过。

该结果只决定正式样本量；smoke 不执行 selector 优劣比较。

### Smoke cell

| 任务 | latency `(observation, action)` | candidate 数 | resolved | 同索引 bit-identical |
|---|---:|---:|---:|---:|
| 客厅 | `(1, 1)` | 4 | true | true |
| 客厅 | `(2, 2)` | 0 | true | true |
| 餐桌 | `(1, 1)` | 1 | true | true |
| 餐桌 | `(2, 2)` | 3 | true | true |
| 厨房 | `(1, 1)` | 5 | true | true |
| 厨房 | `(2, 2)` | 3 | true | true |

12 个 branch 均执行 1,655 control step。所有 pair 的 candidate bytes/hash 一致；同索引
twin-run 均 bit-identical。所有 branch：

- action bounds 通过；
- severe collision 为 0；
- supported stale action applied 为 0；
- invalid force 为 0；
- P40-E1/P40-E2 最大守恒差为 `0.0`。

所有 branch 的主事件均为 false。该 smoke 本来只验证可运行性，不比较 selector，因此这些
零事件既不能接受也不能拒绝 target-index 假设。

### 停止与判定

客厅 `(2,2)` 在冻结 candidate 生成、阈值、相机和自然 latency seed 下得到空候选集。
因此 `candidate_set_nonempty=false`，smoke 总检查失败。按结果前合同：

- 不修改候选阈值、score、phase、primitive、MDE、salt 或样本量；
- 不挑换 seed、任务、相机或 mask；
- 不启动 54 supported + 18 challenge pair 的正式实验；
- 结论为 `inconclusive`，而不是 `rejected`，因为 selector 的因果比较没有执行。

第一次完整 smoke 绑定 `8c47ec8`，除同一空候选 cell 外还暴露了上述 P40 长轨迹数值顺序
问题，保留在：

`runs/research-loop/0008/r0008-p41-target-selection-e2-smoke-s20264101-superseded-8c47ec8`

该 run 的 P40 最大差为 `2.837623469531536e-10`。数值修复后，P40 守护恢复为
`true`，候选数仍精确为 `4/0/1/3/5/3`，证明最终停止原因不是 P40 测量误差。
中断 run 与 provenance fail-closed 产物也分别保留，没有删除或冒充 terminal。

## 验证

- 实施 Agent：
  - P40-E2 focused tests：28 passed；
  - Python size、architecture、physics integrity、compileall、diff check 通过。
- 主 Agent：
  - P40-E2 1,655-period 数值顺序回归通过，最大守恒差 `0.0`；
  - P40/P41 focused suite 在宿主环境通过；
  - 沙箱中的 3 个真实后端测试因
    `mujoco.cgl.cgl.CGLError: invalid CoreGraphics connection` 失败，宿主复跑通过；
  - Python size：398 files 通过；
  - architecture、physics integrity、compileall、`git diff --check` 通过；
  - P40 与 P41 report/manifest 的 source commit、冻结祖先、binding identity、claim
    flags、artifact hash/bytes、settling exclusion、terminal 完整性与 P40 守恒逐字段核验
    通过；
  - 历史 `docs/research-loop/0001/`～`0007/` tree 零差异。

## 当前能力结果

- 正式训练：未启动；
- policy inference：未执行；
- P41 smoke 物理 branch：12 个，但只做无训练、同索引合同验证；
- P41 正式 selector 对照：未运行；
- 新家务任务成功：0；
- qualified deployment：无；
- 世界模型、Actor、任务成功率、泛化或硬件安全改善：无声明。

