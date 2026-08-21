# R0007 结果

## `R0001-P32-E1`：`inconclusive`

### 完整性

- 冻结文档提交：`4ef5f3b728e3e13aa18552ea6cb744121ccce71f`
- P32 实现提交：`63032f54b5e926511cfd4530d6da483611b3573d`
- 集成/正式运行提交：`7bd995e9720ff7196b72a1111f26cef1377e1c75`
- 冻结提交是实现与正式运行提交祖先。
- 正式 run：
  `runs/research-loop/0007/r0007-p32-replay-conditional-e1-s20263201`
- 稳定命令：

```text
.venv/bin/python -m hwr.apps.evaluate_replay_conditional_information \
  --input-run runs/foundation-world-model/r0001-p01-baseline-v4-s20260812 \
  --output runs/research-loop/0007/r0007-p32-replay-conditional-e1-s20263201
```

- mode：formal；
- device：CPU；
- wall time：7.01 秒；
- 不更新 production 参数，不运行 MuJoCo Episode，不创建 checkpoint；
- run 磁盘占用：440 KiB。

产物：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 427,551 | `f2e2af4fa2db982c970a3882b45459e6afb22f103caf55865bfd84280eec6174` |
| `folds.json` | 14,022 | `df7a135f0c4fee4a948a05079c3657a9fda243d594c5a41dc2a57592e88d1729` |
| `manifest.json` | 2,696 | `0ebb4ce5bb9efbef7a248f50091f9130732a3949efb2aafb5b1401803b4db4af` |

report 与 manifest 的 `source_commit` 均为集成提交。manifest 记录冻结文档祖先、稳定命令、
输入 manifest identity、正式常量、直接产物 hash/bytes 和禁止能力声明位。

### 输入与切分核验

- 输入 Replay manifest：
  - bytes：450,509；
  - SHA-256：
    `c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985`。
- 168 个 shard 的直接文件 hash 全部通过。
- 24 个 source Episode：
  - 餐桌 6；
  - 厨房 6；
  - 客厅 12。
- 每 source 7 个 16-transition shard，无绝对时间范围重叠。
- 共 2,688 transition。
- outer 3-fold 每折均为餐桌 2、厨房 2、客厅 4 个 test source。
- 每个 outer-train 的 inner 2-fold 完整覆盖且 source 不跨 train/validation。
- 正式 fold manifest 与 frozen hash rule 唯一重建；未扫描切分。

### 主点估计

主 16-D rate target 的 source/task 等权结果：

| 分区 | control MSE | candidate MSE | ratio |
|---|---:|---:|---:|
| 餐桌 | 0.896691 | 0.830222 | 1.080062 |
| 厨房 | 0.857706 | 0.757650 | 1.132061 |
| 客厅 | 0.852191 | 0.757276 | 1.125336 |
| 三任务等权 | 0.868863 | 0.781716 | 1.111481 |

- aggregate source mean log-ratio：`0.10763176917575612`
- 5,000-replicate source-block bootstrap p05：`0.08909454222505013`
- 三任务点估计均大于 1。
- minimum outer-fold action-residual effective rank：
  `14.416167277510429`，高于冻结门槛 6。

17-D configuration target：

| 分区 | control MSE | candidate MSE | ratio |
|---|---:|---:|---:|
| 餐桌 | 0.399709 | 0.366492 | 1.090633 |
| 厨房 | 0.516658 | 0.467329 | 1.105556 |
| 客厅 | 0.424662 | 0.392216 | 1.082725 |
| 三任务等权 | 0.447010 | 0.408679 | 1.093791 |

- configuration bootstrap p05：`0.06787049665904979`
- 三任务 configuration 点估计均大于 1。

其他机制分区：

- no-rewrite：
  - ratio：`1.117756531036662`
  - bootstrap p05：`0.09472526385373697`
- shard interior：
  - ratio：`1.1181768793692342`
  - bootstrap p05：`0.09091801724827651`
- safety rewrite：
  - ratio：`0.6301523022621444`
  - 只有 6 个 source 含正例，多个 fold-task cell 为 0 或 1 个 source；
  - 0 个有限 bootstrap replicate；
  - 按冻结合同只报告，不作为接受依据。

这些点估计表明 state-only nuisance 下 action residual 看似含有增量预测信息，但不能脱离
功效和 controller-history 守护作结论。

### Controller-history 守护

加入 current actor proposal、过去 4 步 proposal/executed action 和 availability mask 后：

| 分区 | control MSE | candidate MSE | ratio |
|---|---:|---:|---:|
| 餐桌 | 0.867409 | 0.943364 | 0.919484 |
| 厨房 | 0.797232 | 0.880372 | 0.905563 |
| 客厅 | 0.852050 | 0.951061 | 0.895894 |
| 三任务等权 | 0.838897 | 0.924932 | 0.906982 |

- aggregate source mean log-ratio：`-0.09416025964212754`
- bootstrap p05：`-0.11092589471031235`
- 三任务 ratio 全部小于 1。

因此主 state-only 信号在 controller history 后完全消失并反向。若 exact-pipeline power
充分，该结果会触发冻结的 `rejected`；但本轮功效门没有通过，不能越过预注册判定顺序。

### Exact-pipeline 功效

冻结设计完整执行：

- 200 trial；
- 每 trial 1,000 source/task-stratified bootstrap replicate；
- 两个 null 和一个 planted 条件；
- 每 trial 使用实际 source/task/shard/history mask 和完整 nested pipeline；
- planted 方向与 scale 只由对应训练折生成；
- 600 个 fold-level planted training oracle ratio 全部为 1.10；
- outer-test 未参与 effect calibration。

结果：

| 条件 | pass / trial | empirical rate | exact 95% bound | 门槛 |
|---|---:|---:|---:|---:|
| zero action residual null | 0 / 200 | 0.000 | upper `0.014867039231272056` | `<=0.05`，通过 |
| random target null | 0 / 200 | 0.000 | upper `0.014867039231272056` | `<=0.05`，通过 |
| planted 10% oracle effect | 1 / 200 | 0.005 | lower `0.0002564335872234741` | `>=0.80`，失败 |

planted trial 的 observed outer-test ratio：

- minimum：`1.0117787091057537`
- maximum：`1.0507042615896183`
- mean：`1.0330017239625235`

这说明设计的假阳性控制良好，但以当前 24 个 source、嵌套拟合和冻结主门，无法可靠检出
训练折 oracle MSE ratio 1.10 的 planted effect。不得因为真实数据主点估计为 1.111 而忽略
功效失败。

### 判定

冻结判定顺序要求 rank 或 exact-pipeline power 不足时优先标记 `inconclusive`。

正式结果：

`inconclusive`

归因：

1. rank 通过；
2. 两个 null FPR 通过；
3. 10% planted power 严重不足；
4. controller-history 守护在观察结果上失败；
5. safety-rewrite stratum 缺乏 source-level 测量功效。

因此本轮既不接受“普通 Replay 含稳定增量条件信息”，也不在统计意义上拒绝该假设。
P32-E1 不授权 P31、P43、P33、Actor 解锁或任何训练。后续若重提，必须先增加独立 source
或降低设计方差，而不是修改冻结门槛、去掉 controller guard 或挑选 target。

## `R0001-P40-E1`：`accepted as safety measurement contract evidence`

### 完整性

- 冻结文档提交：`4ef5f3b728e3e13aa18552ea6cb744121ccce71f`
- P40 实现提交：`7bd995e9720ff7196b72a1111f26cef1377e1c75`
- 冻结提交是实现提交祖先。
- 正式 run：
  `runs/research-loop/0007/r0007-p40-contact-ledger-e1-s20264001`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_contact_ledger \
  --output runs/research-loop/0007/r0007-p40-contact-ledger-e1-s20264001
```

- device：CPU；
- wall time：2.63 秒；
- 每任务 fixed hold 32 control step，共 96 control step；
- 每任务 800 physics substep；
- camera rendering 在 reset 后禁用；
- 不运行 policy inference，不更新参数，不创建 checkpoint；
- run 磁盘占用：592 KiB。

产物：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 598,449 | `73a1a6e0a65f55b5d6d170219d847e84723c8ae01c2e876233efae44dd5a864c` |
| `manifest.json` | 2,221 | `7da768a3edbdea7707534d87edc888c7935f0eabfad7fe0812630ef8d1d32294` |

binding identity：

- path：`configs/adapters/mujoco/formal_3d_v1.json`
- bytes：3,051
- SHA-256：
  `7984ef2544bb618269681d274257a598b02621371a26de002bfdd8bbf7decab6`

### 合同结果

全部冻结检查通过：

- 显式角色键完整：
  - `floor_support`
  - `manipulated_object`
  - `target_container`
  - `articulation`
- 每任务角色互斥；
- 角色并集严格等于 legacy allow-list；
- 所有 geom 在加载模型时可解析；
- robot–environment contact 逐 contact point 读取 normal force；
- 同一无序 geom pair 先求和，再进入唯一类别；
- 每 substep、control period 和 Episode 的 peak/impulse/duration 语义通过单元 fixture；
- missing、nonfinite 和 negative force 均 fail-closed；
- 三任务 ledgers 的 missing/nonfinite/negative count 均为 0；
- 五类总账全部发布，包括零值类别、forbidden 和 ignored contact counts；
- measurement enabled/disabled 的 applied action、proprioception、reward、termination、
  success、reason、legacy severe-collision/forbidden-force 和 safety intervention trace
  bit-identical；
- 新总账不进入 runtime safety、reward、termination 或 success。

MuJoCo physics identity：

- timestep：0.002 秒；
- control_hz：20Hz；
- 每 control period：25 substep；
- solver：2；
- iterations：100；
- tolerance：`1e-8`。

timestep-halving fixture：

- dt：0.002 / 0.001；
- 非零类别：`floor_support`；
- relative cumulative-impulse difference：
  `1.810761304179668e-16`；
- 通过冻结 `<=10%` 门。

### 固定 trace 的描述性结果

| 任务 | severe collision | legacy max forbidden force | floor/support category peak | floor/support pair peak | floor/support impulse |
|---|---:|---:|---:|---:|---:|
| 客厅 | 0 | 0N | 1744.089N | 924.795N | 751.993N·s |
| 餐桌 | 0 | 0N | 1743.897N | 923.023N | 751.993N·s |
| 厨房 | 0 | 0N | 1798.404N | 900.515N | 751.993N·s |

三任务均出现：

- legacy severe collision = 0；
- legacy maximum forbidden force = 0；
- floor/support category peak > 220N；
- floor/support pair peak > 220N。

这直接证明旧 severe-collision 总账不覆盖 allowed floor/support 负载。该结果主要由机器人
静态支撑接触构成；220N 是旧 forbidden-contact 内部阈值，不是地面支撑限制、硬件危险
阈值或安全验收线。因此不得把上述数值解释为机器人不安全，也不得据此修改 runtime safety。

manipulated object、target container 和 articulation 在 fixed hold trace 中均为 0；本轮只
验证测量合同，没有主动制造抓取接触或用 scripted trajectory 伪造安全能力。

### 判定

P40-E1 全部冻结接受门通过，正式标记：

`accepted as safety measurement contract evidence`

该接受项：

- 修复安全测量可观测性；
- 不改变现有行为；
- 不改变安全决策；
- 不建立硬件阈值；
- 不代表安全、泛化或家务能力改善。

P41-E1 的安全测量前置现已满足一部分，但其 blind-control、MDE 和功效合同仍未修订，不能
在本轮自动启动。

## 集成验证

实施与正式 run 前后验证：

- P32 focused tests：14 passed；
- P40 pure-logic/app focused tests：19 passed；
- P40 + formal MuJoCo backend focused tests：39 passed；
- P32/P40 focused integration：33 passed；
- 全量 pytest：
  - 11 项既有 skip；
  - 18 条 warning 均为 `torch.jit.script` deprecation；
  - 无失败；
- training semantics：通过；
- physics integrity：通过；
- Python size：387 files 通过；
- architecture、compileall、`git diff --check`：通过；
- 历史 `docs/research-loop/0001/`～`0006/` tree 零差异。

## 本轮能力结果

- 正式训练：未启动；
- policy 闭环能力 Episode：未运行；
- 新家务任务成功：0；
- qualified deployment：无；
- 世界模型、Actor、闭环成功率、泛化或安全能力改善：无声明。
