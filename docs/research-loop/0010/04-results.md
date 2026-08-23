# R0010 结果

## 结果总览

| ID | 判定 | 最强允许结论 |
|---|---|---|
| `R0001-P50-E1` | `accepted as immutable acquisition evidence contract` | 24 个 planned Episode 的 policy-visible acquisition bytes 可完整、确定性、无行为扰动地封存与重放 |
| `R0001-P50-E2` | `accepted as candidate-funnel measurement evidence` | 冻结 P41 generator 的 anchor/component/ranking 三层损失可守恒、确定性地离线测量 |
| `R0001-P51-E1` | `rejected` | frame-fixed 在冻结 eligible cohort 上有小幅正向趋势，但远低于结果前 MDE，不能称为有意义的物理收敛改善 |
| `R0001-P50-E3` | `deferred` | 未实施 |
| `R0001-P56` | `deferred` | 未实施 |

本轮没有正式训练、policy inference、closed-loop task capability Episode、新家务任务成功
或 deployment。P50 是测量合同；P51 只执行 acquisition、B0/B1 与 B2 Cartesian
preposition，没有进入接触、gripper、抓取或任务成功阶段。

## 实现与验证

### 冻结和实现谱系

- 初始冻结提交：
  `5fad6cec27e8f797c31a202497745a5616ab220b`
- 结果前 replay 资源澄清：
  `2d1752f2c0c8b9e39d7f3ebaa8e9ff0ec1d13f38`
- P50-E1 初始实现：
  `3dbe18503479b86c9b8d55bb7e5176a4d24d2ff7`
- P50-E2 初始实现：
  `356d777374754ea5b3a8d480c51b0c94fa0c39af`
- P50 独立审查修复：
  `981eebe254560943aa81c3cc801becafe3cbe114`
  `5ac14804688588b2762a60e9a226f677934cc22d`
  `d635b2c5d299fc7d3d4029ee10678373edb1d9c9`
  `91a8030a3a352f140f81d44c0421ab97cf84c7ba`
  `0cb5b4a6a2566ad9d10c3fb9da798fb5d820fcad`
- P51-E1 初始实现：
  `294e82debf59cc46c68578f6e383cadb798dbaa0`
- P51 独立审查、source-size 与验证修复：
  `470780e668b3dac8229d0e0d8b56cb8268b44515`
  `255862790feb73bb84dcde96ac3a14f8e8868101`
  `56cff266a45729675f243ac14e296bc9a8990bff`
  `2a59a3940b59ae02fefa49c7a4b58316c3d16b77`
  `601601f36da899b9bd041a2ddd6f4b7042aeae4a`
  `d67791a53491ce37cddaef4bd7d6b71ad3e66ac2`
- P51 candidate-empty validator 修复：
  `38daff6631c32af2f34ae0cd77578004c0a0768e`
- 首版 bank 提交：
  `3ab93c8959bd2ab0c84c04d7c5b0144d5b772481`
- JSON role-order 重放修复：
  `656c759235f1a736b11e8cc2a75c69e2c0b8a3f6`
- 首版 bank 归档提交：
  `a413feb47110c598285a9095020e0de7157dc866`
- 最终 bank 提交：
  `63b6c2dc0221f37b71b4356d0b154ec3e8e97c0a`

全部正式 source commit 都以 `2d1752f` 为祖先。`docs/research-loop/0001/`～`0009/`
tree 与本轮起始记录完全一致。

### 代码审查

P50 与 P51 初版实现均经过独立只读审查、修复和终审。高风险 finding 包括：

- P50 首版把同一 payload stream 的第二个 controller state 错称为 same-seed physical
  replay；
- P50 首版 capture-disabled 没有真正关闭持久化 side effect；
- P50 首版未从 runtime audit 核对实际 latency，failure Episode 仍可能生成正式
  candidate；
- P50 首版漏斗 pre-top64 来自第二次重算，而非同一次正式 generator trace；
- P50 plan 与 record 可同步篡改，未从 salt 和正式 latency sampler 重建；
- P51 首版 evaluator target 未使用 B2 起点当前 base pose；
- P51 首版 bank validator 可把早期 eligible 改成任意 ineligible 后用后序 seed 补位；
- P51 首版可信任自报 AUC、endpoint、terminal reason 与 hard-guard summary；
- P51 首版没有绑定完整 protected source/config/assets；
- P51 首版 JSON `sort_keys=True` 后把 `arms` mapping 键序误当作显式 role order。

全部中高 finding 在正式结果前修复，并由极窄终审返回
`no medium/high findings`。最终 runner：

- 从 raw bytes/step trace 重算接受指标和 hard guards；
- 从 salt 重建 seed、自然 latency 与 cohort；
- 将 frozen document、历史 tree、protected source/config/assets、bank source 与当前
  source 绑定；
- 不信任 artifact 自报的派生统计或 mapping iteration order。

### 测试与项目门禁

- P50 最终 focused suite：85 passed。
- P51 最终 focused suite：69 passed。
- 主 Agent 联合新功能与 P41/P51/P52 回归：195 passed。
- 主 Agent宿主全量 pytest：
  - 976 collected；
  - 11 skipped；
  - 其余全部通过；
  - 18 条 warning 均为既有 `torch.jit.script` deprecation warning。
- 沙箱全量 pytest 中所有失败均为
  `mujoco.cgl.cgl.CGLError: invalid CoreGraphics connection`；同一全量 suite 在宿主图形
  环境通过。
- Python size：418 files 通过，file `<=800`、function `<=200`。
- architecture、physics integrity、compileall、`git diff --check` 均通过。
- `verify_development_ready.py` 的 isolated Git snapshot 因 `runs/` 被 `.gitignore`
  排除，缺少历史冻结 artifact 而失败；真实 checkout 的 focused、全量、静态与
  provenance 门均通过。本轮未修改该无关基础设施。

## `R0001-P50-E1`

### 正式运行

- source commit：
  `d67791a53491ce37cddaef4bd7d6b71ad3e66ac2`
- 输出：
  `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_candidate_acquisition \
  --mode acquisition \
  --output runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --salt-file runs/research-loop/0010/.host/p50-e1-salt.txt
```

- 运行环境：Python 3.11.0、NumPy 2.4.6、MuJoCo 3.10.0、CPU。
- wall time：`2706.6186038340093s`。
- peak tracemalloc：`775,703,863 bytes`。
- planned：24 个独立 Episode；
- validation replay：24 个独立 backend/reset；
- 总 acquisition control step：47,760；
- post-selection：未执行。

### 完整性与安全

- seed candidate：246；
  - planned：24；
  - natural latency mismatch：222；
  - environment seed 246/246 唯一；
  - policy seed 246/246 唯一；
  - 跨域交集 0；
  - replacement 0。
- 24/24 planned Episode、12/12 cell、每 cell 2 replicate。
- primary/validation 均为 24/24 × 995 step，无 runtime terminal。
- 384 个输入：
  - A1：181；
  - A3：179；
  - A4 final：24。
- 384/384 policy blob deserialize→serialize bit-identical。
- 24/24 saved capsule 离线调用正式 `generate_candidate_set` 后：
  - canonical bytes/hash 一致；
  - candidate count 一致；
  - score hash 一致；
  - selected index 一致。
- primary/validation 的 randomization、observation、policy-input、capture payload、
  physical trace、proposed/applied action 与 candidate identity 全部一致。
- capture persistence：
  - primary：24/24 enabled；
  - validation replay：24/24 disabled。
- 48 次物理 acquisition 合计：
  - action-bound failure 0；
  - stale applied action 0；
  - severe collision 0；
  - invalid force 0；
  - safety intervention 0；
  - P40 conservation 非零 0。

### 候选概况与判定

- 总 final candidate：39；
- 空候选 Episode：5/24；
- acquisition failure：0。

全部不可变证据、重放、无行为扰动、seed、provenance 与安全测量门通过，正式判定：

`accepted as immutable acquisition evidence contract`

该接受只证明 evidence capsule 合同成立，不证明候选覆盖改善、感知泛化、数据效率、
交互或任务能力。

### Artifact

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `plan.json` | 138,145 | `5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab` |
| `capsules.json` | 794,512 | `223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf` |
| `report.json` | 2,218 | `b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0` |
| `manifest.json` | 186,310 | `cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86` |

manifest 还绑定 792 个 binary blobs；E1 目录共 796 files、303,660,083 bytes。

## `R0001-P50-E2`

### 正式运行

- source commit：
  `d67791a53491ce37cddaef4bd7d6b71ad3e66ac2`
- 输入：P50-E1 immutable capsule。
- 输出：
  `runs/research-loop/0010/r0010-p50-e2-funnel-s20265001`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_candidate_acquisition \
  --mode funnel \
  --capsules runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --output runs/research-loop/0010/r0010-p50-e2-funnel-s20265001
```

- wall time：`155.2085381669749s`。
- peak tracemalloc：`143,373,342 bytes`。
- 24 Episode 全部分析两次，report bit-identical。

### 漏斗守恒

Anchor 总计：

```text
876,960 enumerated
= 258,291 center/ring validity rejection
+ 613,377 prominence rejection
+ 4,837 center depth spread rejection
+ 0 patch support rejection before self-mask
+ 5 support rejection after self-mask
+ 150 height/range rejection
+ 0 planarity rejection
+ 75 width rejection
+ 225 raw candidates
```

按各 stage 自身输入计算的拒绝率：

- center/ring validity：`29.4530%`；
- prominence：`99.1446%`；
- center depth spread：`91.4021%`；
- support after self-mask：`1.0989%`；
- height/range：`33.3333%`；
- width：`25.0%`。

Component 与 ranking：

```text
149 connected components
= 110 view_count<2 rejection
+ 0 aggregate-normal-zero rejection
+ 39 pre-top64 candidates

39 pre-top64
= 39 retained
+ 0 truncated
```

### Cell 结果

格式为 `raw/component/final`：

| Cell | Task | 数量 | 重复描述性 loss stage |
|---|---|---:|---|
| `o1-a1` | living | `11/9/2` | prominence、depth spread、view count |
| `o1-a2` | living | `33/16/6` | prominence、depth spread |
| `o2-a1` | living | `14/13/0` | prominence、depth spread、view count |
| `o2-a2` | living | `10/6/3` | prominence、depth spread |
| `o1-a1` | dining | `13/7/3` | prominence、depth spread |
| `o1-a2` | dining | `27/16/4` | prominence、depth spread |
| `o2-a1` | dining | `13/8/2` | prominence、depth spread |
| `o2-a2` | dining | `11/6/3` | prominence、depth spread |
| `o1-a1` | kitchen | `21/15/4` | prominence、depth spread、view count |
| `o1-a2` | kitchen | `21/15/4` | prominence、depth spread、view count |
| `o2-a1` | kitchen | `24/19/3` | prominence、depth spread、view count |
| `o2-a2` | kitchen | `27/19/5` | prominence、depth spread、view count |

- prominence 与 center depth spread 在 12/12 cell 的两个 replicate 中均达到冻结的
  `>=60%` 描述性损失标签。
- view-count 在 living `o1-a1/o2-a1` 与全部四个 kitchen cell 重复达到该标签。
- 5 个空 Episode 全部已有 raw candidate 与 connected component，但在
  component→view-count-qualified 阶段归零。
- 最弱 cell 为 living `o2-a1`：
  - 2/2 Episode 为空；
  - raw 14；
  - connected component 13；
  - pre-top64/final 0。

### Unique-observation shadow

- candidate keyframe：360 identity、360 unique identity、360 unique payload；
- 加 A4 final：384/384/384；
- 重复 observation identity：0；
- ordinal retained：39；
- shadow retained：39；
- 24 个 Episode 的 shadow delta 均为 0。

因此本 cohort 否定“重复 observation identity 是当前空集合原因”。它支持的最具体描述是：
存在真实 raw/component 候选，但 view-count gate 会在部分 cell 清空全部 component；同时
prominence 与 depth spread 是普遍的大比例描述性拒绝阶段。该结果**不证明**放宽任一 gate
会改善候选质量或物理交互。

全部 source、single-generator-call、三层守恒、identity、candidate bit-identity 与
determinism 门通过，正式判定：

`accepted as candidate-funnel measurement evidence`

### Artifact

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 1,163,653 | `4c7f36d20356d2f0f9c83d024412da5ec3a95dea8714e9a04d91d0cd686d0e39` |
| `manifest.json` | 12,684 | `a2f2498ac1d23f22dae337b5ad0836ae72230861d4f33284cdff19c1b46e268e` |

## `R0001-P51-E1`

### Final bank

- bank source：
  `a413feb47110c598285a9095020e0de7157dc866`
- bank artifact commit：
  `63b6c2dc0221f37b71b4356d0b154ec3e8e97c0a`
- 输出：
  `runs/research-loop/0010/r0010-p51-e1-bank-s20265101`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_cartesian_convergence \
  --mode build-bank \
  --output runs/research-loop/0010/r0010-p51-e1-bank-s20265101 \
  --salt-file runs/research-loop/0010/.host/p51-e1-salt.txt
```

- wall time：`3025.6101250830106s`；
- peak RSS platform units：`959,709,184`；
- raw seed：405；
  - natural latency mismatch：361；
  - latency matched：44；
  - candidate empty：5；
  - relative yaw below `pi/6`：3；
  - eligible：36；
- 12/12 cell × 3 pair；
- 三任务各 12 pair；
- candidate count：
  - 1 candidate：10 pair；
  - 2：11；
  - 3：6；
  - 4：4；
  - 5：2；
  - 7：3；
- 最小冻结 relative yaw：`0.5745802670785696rad`，高于 `pi/6`。

最终 bank 与 superseded 旧 bank 的 `cells`、`pairs`、`seed_audit`、salt、eligible count
和 infeasible cells 完全一致；只改变 source/provenance identity，没有换 seed、candidate
或 cohort。

Final bank artifact：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `bank.json` | 7,160,016 | `09d2fe4e05f2bd8d23ebfe6886fe260d1b34b41771da42992f0f432a8a04f3d3` |
| `seed-audit.json` | 3,996,474 | `1b8ca0d93dd4324d0f6e14a1932bbce69b24964bc0aa08a53a9b4bdf2eb69407` |
| `report.json` | 1,009 | `084029d0c4eda55b10c885856dc19f2356ade2c31b30b71e070de91cd9d7653d` |
| `manifest.json` | 53,605 | `7e0d5f9c7757b59ceb8d4dfe3ddcba38cc1d1037c43c358e7168d700310d5e45` |

### Final evaluate

- source/bank commit：
  `63b6c2dc0221f37b71b4356d0b154ec3e8e97c0a`
- 输出：
  `runs/research-loop/0010/r0010-p51-e1-convergence-s20265101`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_cartesian_convergence \
  --mode evaluate \
  --bank runs/research-loop/0010/r0010-p51-e1-bank-s20265101/bank.json \
  --output runs/research-loop/0010/r0010-p51-e1-convergence-s20265101
```

- wall time：`5058.993814750022s`；
- peak RSS platform units：`1,091,125,248`；
- 36 pair、72 branch；
- 每 branch 完整执行 B2 100 control step；
- 无 ordinary terminal、carry-forward、unresolved 或 replacement。

### 连续主结果

```text
delta_i = normalized_AUC_legacy - normalized_AUC_fixed
```

- 总体 12-cell 等权 point estimate：
  `0.023449928237828013`
- paired stratified bootstrap：
  - 10,000 replicates；
  - seed `20265102`；
  - linear `q=0.05`；
  - one-sided lower：`0.0200166364588308`。
- point estimate 仅为冻结 MDE `0.10` 的约 `23.45%`。

分账：

| 维度 | delta |
|---|---:|
| living | `0.04254507447650997` |
| dining | `0.023053844667296614` |
| kitchen | `0.004750865569677455` |
| observation latency 1 | `0.019063909142789517` |
| observation latency 2 | `0.027835947332866515` |
| action latency 1 | `0.022525365325697957` |
| action latency 2 | `0.024374491149958075` |

12 个 cell mean 全为正；36 pair 中 30 个 delta 为正、6 个为轻微负值；最大单 pair delta
`0.07591725404801863`，仍低于 MDE。

### 稳健性与安全

- `delta_i >=0.10`：0/36；
- endpoint 有利于 fixed：33/36；
- 冻结 `frame_fixed_win`：0/36；
- 每 task：0/12；
- 每 latency combination：0/9；
- 总 `>=24/36`、每 task `>=6/12`、每 latency combination `>=4/9` 均失败。

72 branch、7,200 raw control step、14,400 proposed/applied action vector：

- action bounds violation 0；
- stale applied action 0；
- safety intervention 0；
- severe collision 0；
- invalid force 0；
- P40 conservation violation 0；
- nonfinite metric/runtime 0；
- reported hard failure 0；
- safety/cap/gripper/phase/target/FK/backend identity failure 0；
- 36/36 首个 treatment step 只在左右臂 linear xy indices `[2,3,8,9]` 产生差异；
- 两臂 action 均非塌缩。

### 判定

连续方向在全部聚合分账中为正，bootstrap lower 也高于 0，但改善幅度显著低于结果前
`0.10` MDE，且 0/36 pair 达到冻结 win 定义。按结果前判定顺序：

`rejected`

该结论不是“frame transform 完全无效”，而是：

> 在冻结 eligible natural-latency cohort 上，frame-fixed 相对 legacy 的 B2
> tool-to-preposition normalized AUC 只有小幅正向 `+0.02345`，不足以达到预设有意义改善。

P51 不进入新的能力基线，也不授权 P41 selector 正式对照、Replay 采集或训练。

Final evaluate artifact：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `terminals.json` | 26,314,706 | `1c54f93a95bfbf4e08076b3c633b22dce295990a6808a48f0f10de18a2b3c2c7` |
| `report.json` | 4,643 | `3fcac95c2362923d9eb94ef4d7121d5bcb31ea859a308ed352321dfa93771cc9` |
| `manifest.json` | 53,753 | `821f3cf6fea922a86b4096ee5d0ba9c64b9d8f444eacc98dcfc1f164da1328d2` |

落盘 JSON 由当前 validator 重新分析后 `analysis == report`，独立审计也逐项复算全部 raw
distance、action、bootstrap、win 与 hard guard，最终签字为合同有效的 `rejected`。

## Superseded 与中断记录

1. `runs/research-loop/0010/r0010-p51-e1-bank-s20265101-superseded-d67791a`
   - source：`d67791a`；
   - wall time：`2208.4496269169904s`；
   - failure：`incomplete prefix lacks failure evidence`；
   - 根因：validator 在读取冻结允许的 `candidate_set_empty` 之前错误拒绝未执行 B0/B1 的
     prefix；
   - 修复：`38daff6`；
   - `failure.json`：
     `a32a1be6eda72681450f851e03bbc822d17d039a9efaa73ed5d82fa19eb3e0f5`；
   - `manifest.json`：
     `e1d41be36934268c20fe08156ced1f31836199f34b2eba475839751ff5463ec1`。
2. 修复后第一次 bank 重跑被电脑重启中断：
   - 没有残留进程；
   - atomic output 未发布；
   - 没有中间结果可读；
   - 重启后用同 salt/seed 顺序原样重跑。
3. `runs/research-loop/0010/r0010-p51-e1-bank-s20265101-superseded-38daff6`
   - 完整 36-pair bank；
   - source：`38daff6`；
   - cohort/seed audit 与 final bank 完全相同；
   - 后因 JSON role-order validator 修复产生 source drift，不能继续作为正式 evaluate
     输入。
4. `runs/research-loop/0010/r0010-p51-e1-convergence-s20265101-superseded-3ab93c8`
   - source：`3ab93c8`；
   - 数值结果与 final evaluate 完全相同；
   - 首版 report 在内存分析为 `rejected`，但 `sort_keys=True` 使落盘 `arms` mapping 键序
     与显式 role order 不同，旧 validator 无法重放；
   - 修复：`656c759`；
   - 原 artifact 不作为最终证据。

必须同时记录实验时序偏差：第一次 failed bank 已读取冻结 salt，之后才修复
candidate-empty validator；第一次 formal evaluate 已产生结果，之后才修复 JSON
role-order roundtrip。两次重跑都使用相同 salt、相同 seed audit 与相同 cohort，v1/v2
数值完全一致，但不能声称整个流程从未发生结果后评测修复。

## 当前能力结果

- 正式训练：未启动；
- 参数更新/checkpoint：无；
- policy inference：未执行；
- closed-loop task capability Episode：未执行；
- 新家务任务成功：0；
- qualified deployment：无；
- P41 selector 正式对照：未运行；
- 世界模型、Actor、闭环成功率、泛化、数据效率或硬件安全改善：无声明。
