# R0011 结果

## 结果总表

| ID | 判定 | 结论边界 |
|---|---|---|
| `R0001-P50-E3` | `inconclusive_design_infeasible` | 冻结 body-role mapping 在 kitchen 场景不可构造；0 Episode、无 entity-coverage 结果 |
| `R0001-P57` | `accepted as bilateral pre-contact reachability measurement evidence` | 固定 P51 cohort 的双臂 pre-contact readiness 与 command-support deficit 测量 |
| `R0001-P56` | `deferred` | 本轮未执行 contact attribution |
| `R0001-P58` | `deferred` | 前置证据未满足，不执行 B3 action-vs-hold |
| `R0001-P59` | `rejected` | simulator-private segmentation 不得进入正式 candidate generator |

本轮没有正式训练、参数更新、checkpoint、policy inference、post-selection capability
Episode 或新家务任务成功。

## `R0001-P50-E3`

### 实施与独立审计

初版 full sidecar 实现曾覆盖 observation-time queue、segmentation、entity funnel 与正式
runner。主 Agent focused 门禁通过后，独立审计发现：

1. clean committed source/provenance 未强制；
2. private truth isolation 仍含自证式 flag；
3. hard safety failure 未进入接受门；
4. component provenance 使用第二套手写 merge 自证；
5. renderer mode、完整分账、mapping collision 和核心 runner 负例仍有缺口。

该版本没有提交、没有运行正式 Episode。由于后续真实场景 preflight 已证明冻结设计必然在
Episode 0 前失败，主 Agent 没有继续完善不可执行的 full pipeline，而是撤回全部未提交
runtime/candidate 改动，仅保留最小、可审计的 mapping preflight：

- `src/hwr/adapters/mujoco/entity_candidate_mapping.py`
- `src/hwr/apps/evaluate_entity_candidate_coverage.py`
- `tests/test_entity_candidate_coverage.py`

提交：

`e16e3ae2ad33fc95c12bcb4a5e5d48527b62fb0b`

### 正式命令

```text
.venv/bin/python -m hwr.apps.evaluate_entity_candidate_coverage \
  --plan runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/plan.json \
  --historical-capsules runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001/capsules.json \
  --output runs/research-loop/0011/r0011-p50-e3-entity-coverage-s20265003
```

进程按设计以 exit code `2` 返回，并原子写出 failure artifact。

### Preflight 结果

- source commit：`e16e3ae2ad33fc95c12bcb4a5e5d48527b62fb0b`
- frozen document commit：`88992a773ee2b0f214dba7975cdddf25f282d679`
- frozen document blob：`2ea08a0ab8fea5b0444d4ba7b162e4129e5765b8`
- clean committed source：通过
- `target_selection.py`、task config、binding config、递归 XML：相对冻结提交无漂移
- 历史 `docs/research-loop/0001/`～`0010/` tree：全部匹配
- 五个冻结输入：bytes/SHA-256/lineage 全部匹配
- living-room mapping：
  - `mapping_preflight_passed`
  - 30 bodies、70 geoms
  - mapping SHA-256：
    `38a355fc7b561f35751efdf0703479de4c6430afe66942540113db12543f2cb7`
- dining-room mapping：
  - `mapping_preflight_passed`
  - 32 bodies、72 geoms
  - mapping SHA-256：
    `75324ff46e8dcf3599492b55bbd8ab2ebcbe8998b79a3aadb0cf4101f96b27cd`
- kitchen mapping：失败
  - task：`store_kitchen_items_3d/v1`
  - body ID：`30`
  - body：`kitchen_drawer`
  - 首个角色：`articulation:drawer`
  - 第二角色：`target_container`
- Episode：`0`
- physical acquisition：`0`

### 判定

冻结合同规定同一 body 被两个任务角色占用时必须 `invalid`，不能按优先级覆盖。Kitchen
场景中 articulation handle 与 drawer target-container geoms 的确都属于
`kitchen_drawer` body，因此：

`inconclusive_design_infeasible`

这不是 entity coverage 的负结果，也不是 simulator/renderer 故障。它证明冻结的纯
body-role 映射无法表达 kitchen 中“同一 articulated body 的 handle 与 container 区域”
这两个局部角色。没有运行任何 Episode，也没有测得 visible/raw/component/final recall。

不能把 living/dining 的 preflight 通过外推为完整 12-cell 合同有效；不得删除 kitchen、
合并角色或后验给 articulation/target-container 设置优先级。

### Artifact

目录：

`runs/research-loop/0011/r0011-p50-e3-entity-coverage-s20265003`

| 文件 | SHA-256 |
|---|---|
| `failure.json` | `4931e98cfaf7bde8c2fadcaf81f65dc35add7843a7d4b34801309aa2b288701f` |
| `manifest.json` | `fd87ad98ec2755db27f4ecb2b3bc21889a8c2c208da7bffbb039b4608011740d` |

## `R0001-P57`

### 实施

提交：

- 初始实现：`07c18fc179bb2ca713f09519673c6488bd6646b5`
- ignored terminal evidence provenance 修复：
  `bbf6d666f8071fa3a5d26be2d774ceeae2ebc7a1`

实现只读取冻结 P51 bank 与正式 convergence artifact，不运行 MuJoCo、不修改行为。

### 正式命令

```text
.venv/bin/python -m hwr.apps.evaluate_precontact_reachability \
  --bank runs/research-loop/0010/r0010-p51-e1-bank-s20265101/bank.json \
  --terminals runs/research-loop/0010/r0010-p51-e1-convergence-s20265101/terminals.json \
  --output runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701
```

### Provenance

- source commit：`bbf6d666f8071fa3a5d26be2d774ceeae2ebc7a1`
- frozen document commit：`88992a773ee2b0f214dba7975cdddf25f282d679`
- tracked committed bank：
  - bank SHA-256：
    `09d2fe4e05f2bd8d23ebfe6886fe260d1b34b41771da42992f0f432a8a04f3d3`
  - bank manifest SHA-256：
    `7e0d5f9c7757b59ceb8d4dfe3ddcba38cc1d1037c43c358e7168d700310d5e45`
  - commit：`63b6c2dc0221f37b71b4356d0b154ec3e8e97c0a`
- manifest-bound ignored formal evidence：
  - terminals：
    `1c54f93a95bfbf4e08076b3c633b22dce295990a6808a48f0f10de18a2b3c2c7`
  - report：
    `3fcac95c2362923d9eb94ef4d7121d5bcb31ea859a308ed352321dfa93771cc9`
  - manifest：
    `821f3cf6fea922a86b4096ee5d0ba9c64b9d8f444eacc98dcfc1f164da1328d2`
  - producer source commit：
    `63b6c2dc0221f37b71b4356d0b154ec3e8e97c0a`

### 核心结果

- 36/36 pair、72/72 arm；
- 12/12 cell，每 cell 3 pair；
- 三任务各 12 pair；
- 101 个同步 distance sample/arm；
- 100 个 actual applied action/pair；
- `ever_bilateral_ready`：`0/36`；
- `endpoint_bilateral_ready`：`0/36`；
- 两臂末端距离都改善：`10/36`；
- 两臂 initial command margin 都为负：`36/36`。

任务分账：

| Task | Pair | Ever Ready | 两臂都改善 | 双负 Margin |
|---|---:|---:|---:|---:|
| living | 12 | 0 | 6 | 12 |
| dining | 12 | 0 | 0 | 12 |
| kitchen | 12 | 0 | 4 | 12 |

Latency 分账：

| 维度 | Pair | Ever Ready | 两臂都改善 | 双负 Margin |
|---|---:|---:|---:|---:|
| observation latency 1 | 18 | 0 | 3 | 18 |
| observation latency 2 | 18 | 0 | 7 | 18 |
| action latency 1 | 18 | 0 | 5 | 18 |
| action latency 2 | 18 | 0 | 5 | 18 |

Latency-combination 的两臂改善：

- `o1-a1`：`1/9`
- `o1-a2`：`2/9`
- `o2-a1`：`4/9`
- `o2-a2`：`3/9`

每个 latency combination 的 readiness 都为 `0/9`；每个 cell 的 readiness 都为 `0/3`。

### 距离与预算

72 arm 聚合：

| 指标 | 最小 | 均值 | 最大 |
|---|---:|---:|---:|
| B2 `d0` | `1.0782476291624683m` | `2.279013066240849m` | `3.400694272089616m` |
| B2 `d100` | `0.7512362489141097m` | `2.122923739164829m` | `3.405736256383822m` |
| actual applied command budget | `0.3486734392248524m` | `0.39061773300557145m` | `0.43271220081555894m` |
| initial command margin | `-3.010679833816681m` | `-1.8883953332352779m` | `-0.7254330230916336m` |
| preposition→contact distance | `0.17698870020427832m` | `0.1789022161469772m` | `0.17951323071016292m` |
| B3+B4 nominal transition margin | `-0.08451323071016292m` | `-0.08390221614697718m` | `-0.08198870020427831m` |

所有 72 arm 的 initial command margin 与 nominal contact-transition margin 都为负。

必须保留限定：actual applied command budget 是命令积分支持代理，不是 actual tool path、
严格 reachable set 或 collision-free path；B3+B4 transition margin 是名义预算，没有执行
B3/B4 物理。

### 不对称反例

P51 原先报告的双臂平均距离下降会掩盖左右臂不对称：

- 仅左臂改善：`15/36` pair；
- 仅右臂改善：`11/36` pair；
- 两臂都改善：`10/36` pair；
- 单臂改善合计：`46/72` arm；
- 最大单臂改善 `d100-d0 = -0.47502736704390225m`；
- 最大单臂恶化 `d100-d0 = +0.10029644948933081m`。

因此不能用总体 mean distance 下降替代双臂同步 readiness；正式 runner 使用同一步的
左右臂 `<=0.10m`，实际为 `0/36`。

### 判定

冻结 `precontact_support_deficit_supported` 门为：

1. `ever_bilateral_ready <=6/36`；
2. 两臂 initial command margin 都负的 pair `>=30/36`；
3. 每 task `ever_bilateral_ready <=4/12`。

实际分别为：

1. `0/36`；
2. `36/36`；
3. living/dining/kitchen 均 `0/12`。

因此：

`accepted as bilateral pre-contact reachability measurement evidence`

且：

`precontact_support_deficit_supported`

允许的结论只有：

> 在固定 P51 frame-fixed cohort 中，现有 B2 没有建立同步双臂 pre-contact readiness，
> 且 actual-applied command support 相对起始 target distance 明显不足。

不允许外推为接触、抓取、任务能力、泛化、安全改善或 deployment。

### 独立审计

独立审计从 raw P51 distance 与 applied action 重算：`PASS`，blocker/major/minor 均为 0。

- 最大数值复算误差：
  `8.881784197001252e-16`
- 3,600 个 applied action vector 无 bounds violation；
- 36/36 P51 hard guard 通过；
- B3/B4 target 公式与正式 primitive 一致，重建误差不超过 `4.72e-16`；
- proposed/applied budget 在 72 arm 上全部不同，证明正式结果没有误用 proposed action；
- raw trace 为连续 B2 step `1..100`、50ms 间隔，无 off-by-one 或左右臂异步；
- 36 个 pair、environment seed、policy seed、planned episode 与 bank-pair hash 唯一。

### Artifact

目录：

`runs/research-loop/0011/r0011-p57-precontact-reachability-s20265701`

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `pairs.json` | 1,954,854 | `d2cae19d21492bea39972bbe3e50e06aa7f19eedf617fbd8dd6c708d8543ddb3` |
| `report.json` | 54,041 | `15fcfb3a0e9b706f9fbfdd9e2dd69a50d096db3435b6cb73100a69fb9697a39d` |
| `manifest.json` | 5,926 | `21ca8837232ac715e209884f355d37a64298d2f4115dc45c81c12f3a7e812e40` |

资源：
- wall time：`8.866050041979179s`；
- peak RSS：`293,273,600 bytes`；
- tracemalloc peak：`72,394,026 bytes`；
- artifact 总计：`2,014,821 bytes`。

## 评测维护

实现前相关测试发现 P51 frozen-document gate 同时要求：

- `docs/research-loop/0010/03-experiment.md` content/blob 不变；以及
- 整个 `docs/research-loop/0010/` tree 永远等于冻结实验提交。

后者会把 R0010 后续合法加入的 `04-results.md`、`05-summary.md` 误判为实验合同漂移，
使 P51 runner 在完成本轮后自我失效。修复：

- 保留 frozen commit ancestry；
- 保留 `03-experiment.md` content 与 Git blob fail-closed；
- 保留 round tree hash 作为 manifest 审计字段；
- 不再把 sibling result/summary 文件造成的 tree drift 当成冻结文档失败。

提交：

`13766f44f23eb463d852a0a94ffb94aefeea38f5`

该修复不改变 P51 输入、指标、算法、artifact 或历史判定，不计入任何能力对比。

## 验证

- P50-E3 preflight + P50 acquisition 回归：37 passed。
- P57 focused：17 passed。
- P51 tree-lock 修复 + 本轮联合 focused：101 passed。
- Python size：424 files，文件 `<=800` 行、函数 `<=200` 行。
- architecture：通过。
- physics integrity：通过。
- compileall：通过。
- `git diff --check`：通过。
- 沙箱全量 pytest：因 macOS `invalid CoreGraphics connection` 产生 MuJoCo renderer
  环境失败，不属于逻辑回归。
- 宿主 `MUJOCO_GL=glfw .venv/bin/python -m pytest -q`：全量通过；11 skipped；
  18 条 warning 均为既有 `torch.jit.script` deprecation。
- development-ready：在隔离 worktree 中因多个历史 ignored run artifact 不随 Git checkout
  出现而失败；包括 R0003 decoder、foundation replay、P36 与 P40/P41 输入。该门禁失败
  已在实现前出现，本轮没有修改这些历史 artifact 依赖。

## 当前能力结果

- 正式训练：未启动；
- 参数更新/checkpoint：无；
- policy inference：未执行；
- post-selection primitive：未执行；
- closed-loop task capability Episode：未执行；
- 新家务任务成功：0；
- qualified deployment：无；
- 世界模型、Actor、闭环成功率、泛化、数据效率或硬件安全改善：无声明。
