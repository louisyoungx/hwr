# R0003 实验结果

## `R0001-P16`：`rejected`

### 完整性

- 源码提交：`03de3fcfaa151efc6dd92ae2081ca3ad08732798`
- report SHA-256：
  `b81615e80fe10bfb7314d6519d182e3da88449c8948c3adaca192e870e662821`
- 每种条件 500 trial；
- 每 trial 200 次同步 Episode bootstrap；
- 墙钟 65.02 秒；
- 500 个 trial 全部保留；
- source commit、P09 输入 hash、report 和 manifest 一致；
- 无重复启动、无残留进程。

### 三臂结果

| 设计 | null FPR | permutation FPR | planted power |
|---|---:|---:|---:|
| fragmented 7×16 | 0.0% | 0.0% | 0.0% |
| continuous same 7 starts | 0.0% | 0.0% | 0.0% |
| continuous all starts | 0.0% | 0.0% | 5.8% |

没有设计达到 80% 功效，因此 `qualified_arms=[]`，P01 路由 blocked。

### 失败指纹

- rho=0.50 连续全起点几乎满功效：三任务 16-step 为 99.8%、99.8%、100%。
- rho=0.96 长时功效只有约 39.8%～43.0%。
- 连续全起点 h16 有 776 行、54 列、数值秩 53，condition number 约 `4.7e6～5.35e6`。
- 7-start 设计只有 56 行、54 列，整体功效为 0。

### 结论

现有 Action Probe 设计对零效应保守，但对冻结动作效应检出功效不足。当前 data probe
失败只能标记为测量 `inconclusive`；不得降低 1.05/1.01 门槛，也不得启动 P01/P02。

下一阶段使用不依赖 state nuisance 拟合的训练前物理因果证据，并继续在 R0003 内推进。

## `R0001-P17`：`accepted as training-data causality evidence`

### 预检

- 源码提交：`9a748de1ea0acf83a397e0bf903516f6b1ae8c6f`
- run：
  `runs/research-loop/0003/r0003-p17-paired-action-s20261017-preflight`
- 64 seed/任务，三个任务共 192 Episode artifact。
- 193 个 artifact 的 SHA-256 和字节数全部复核通过。
- sham：三任务均 0/64 不一致，单任务 Clopper-Pearson 单侧 95% 上界 4.57%。
- actual action-difference RMS 最低 0.2567。
- 方向 cosine 最低近 1.0。
- first-stage 不对称为 0。
- 安全干预、严重碰撞和提前终止均为 0。
- 1,000 null trial：0 次 family-wise 假通过，上界 0.299%。
- 1,000 planted trial：1,000 次通过，power 下界 99.70%。
- 结论：`preflight_passed`。

### 正式确认

第一次 formal 启动在采集前因源码从 preflight 的 `9a748de` 仅漂移到一个 `AGENTS.md`
提交而被 lineage gate 拒绝，0 Episode，不是实验结果。

formal-v2 从 detached `9a748de` worktree 运行，并复用字节完全相同的 preflight 报告：

- run：
  `runs/research-loop/0003/r0003-p17-paired-action-s20261017-v2-formal`
- 64 个全新 seed/任务，共 192 Episode artifact；
- 193 个 artifact hash 全部通过；
- report SHA-256：
  `382c156690c6c748100fe805613048f7080ba687350e1c49a452f7847ee9a5c2`；
- preflight report SHA-256：
  `d6a5a0c0c4b2e93755485820e16cd35d905bb635055f4a340cc9dd5f3aacd610`；
- sham、first-stage、安全和终止守护全部通过；
- 12 个任务×horizon 分区的 permutation p-value 均为 0.001；
- Holm adjusted p-value 均为 0.012；
- 12 个分区全部通过。

| 任务 | horizon | cross-moment | first-stage RMS | outcome RMS |
|---|---:|---:|---:|---:|
| 餐桌 | 1 | 0.0674 | 0.2665 | 0.0413 |
| 餐桌 | 4 | 0.0848 | 0.2665 | 0.0606 |
| 餐桌 | 8 | 0.0790 | 0.2665 | 0.0642 |
| 餐桌 | 16 | 0.0832 | 0.2665 | 0.0793 |
| 厨房 | 1 | 0.0673 | 0.2665 | 0.0421 |
| 厨房 | 4 | 0.0821 | 0.2665 | 0.0587 |
| 厨房 | 8 | 0.0756 | 0.2665 | 0.0635 |
| 厨房 | 16 | 0.0819 | 0.2665 | 0.0794 |
| 客厅 | 1 | 0.0657 | 0.2665 | 0.0432 |
| 客厅 | 4 | 0.0866 | 0.2665 | 0.0618 |
| 客厅 | 8 | 0.0834 | 0.2665 | 0.0668 |
| 客厅 | 16 | 0.0883 | 0.2665 | 0.0800 |

### 结论

P17 建立了不依赖低功效 state-nuisance ridge 的训练前物理因果证据：在三个正式任务、四个
horizon 上，实际 plant action 对后续可控状态具有稳定、可重复、经安全层后的增量因果效应。

该结论只接受为 `training-data causality evidence`：

- 可以解除“自主随机数据没有动作物理效应”的阻断；
- 不能证明当前世界模型已经利用动作；
- 不能证明 Actor 或家务任务能力提升；
- 下一步仍需独立解决动作执行时序、Replay batch 组成和世界模型 action-shuffle 失败。

## `R0001-P11`：head-only 开发与 smoke

### P09 开发集

- 输入：P09 全部 96 Episode，manifest、report 与 96 个 artifact hash 全部校验。
- 三任务×rho 0.50/0.96×action latency 0/1，共 12 个分区全部通过。
- 稳定阶段 normalized RMSE：`0.000189～0.000351`。
- current proposal baseline：`0.00702～0.31082`。
- 每个分区 lag 识别准确率均为 100%。
- latency0 相对 current baseline 没有退化。
- 结论：开发门通过，但 P09 已参与机制设计，不能作为正式接受证据。

### MuJoCo smoke

- run：
  `runs/research-loop/0003/r0003-p11-causal-plant-s20261101-smoke`
- 条件：餐桌任务、rho 0.50、强制 action latency 1、seed `720261101`、64 transition。
- stable 45 transition normalized RMSE：`0.001300`。
- current proposal baseline RMSE：`0.307877`。
- lag=1 识别准确率：100%。
- proposal derangement RMSE：`0.370054`，相对候选恶化 `0.368754`。
- gain 均值：`1.047746`。
- out-of-bounds、安全干预、严重碰撞和提前终止均为 0。
- action-latency-only provenance 完整，artifact hash 与 manifest 一致。
- 结论：采集和分析链路通过 smoke；单 Episode 不进入正式结论。

正式 144 Episode 确认只能在实现提交后从干净源码运行。

### 正式确认：`rejected`

- 源码提交：`ef86971ecd9528c00022e3f944e7878f66665f4a`
- run：
  `runs/research-loop/0003/r0003-p11-causal-plant-s20261101`
- report SHA-256：
  `79195b94f48e8b59ce04bb7a9a3a680f8717f6484da69b2f28d54b3a9b842cda`
- manifest SHA-256：
  `509f1b7f11caa2aa0dced3324bcc6ea9af9b045071c85b1f6a2bc66a7160da3a`
- 144 Episode、18 个任务×rho×latency 分区、9,216 transition。
- 145 个 manifest artifact 的 SHA-256 与字节数全部通过。
- 无严重碰撞、提前终止、越界、缺失 artifact 或 provenance 漂移。

正式结果有 2,196 次安全改写，全部集中在 seed `720365830`、`720784746`：

- 两个 seed 的 evaluation observation latency 都为 3；
- 三任务、两个 rho、三个 action latency 共 36 Episode；
- 每个 Episode 从 step 3 到 63 连续 61 次安全改写；
- 动作 frame 使用延迟 observation timestamp，100ms validity window 小于 3 step
  observation latency 的 150ms；
- 安全层因此以 `outside_validity_window` 拒绝动作，proposal 与 applied action 的 motion
  和 gripper 都被改写；
- 这 36 个 Episode 没有任何稳定非干预 transition，所有分区因此含非有限聚合值并失败。

按照冻结合同，不能删除这些 Episode、放宽稳定定义或事后延长 validity 后改判，P11 正式
结论为 `rejected`。

机制子集仍保留以下反例信息，但不覆盖正式拒绝：

- 其余 108 Episode 共 4,860 个稳定 transition；
- lag1/2/3 的 selected lag 全部正确；
- stable normalized RMSE 为 `3.29e-17～0.004865`；
- current baseline RMSE 为 `0.09409～0.43491`；
- derangement RMSE 为 `0.33859～0.43071`。

该指纹同时暴露独立 stale-frame 运行时问题；它需要另立评测/运行时修复，不能与下一候选
P05 首次捆绑。按预注册路由，下一步执行 P05 冻结 Replay 三臂。

## `R0001-P05`：三臂 smoke

- 源码提交：`9f22b638129a1e06c35da1f26cd3b189e602d771`
- run：
  `runs/research-loop/0003/r0003-p05-batch-arms-s20261205-smoke`
- report SHA-256：
  `42459d19cc74e80480a53a8d33e642453261f29b8655acb7fa3388d49c129ce8`
- 输入 hash、168 个窗口、24 个 source Episode 均与冻结合同一致。
- schedule audit：161 个合格窗口、7 个单 source 餐桌 timeout 窗口同步排除。
- 三臂各完成 2 次真实 MPS update：
  - A：`source_episodes_per_batch=1`，`unique_windows_per_batch=1`；
  - B：`source_episodes_per_batch=1`，`unique_windows_per_batch=2`；
  - C：`source_episodes_per_batch=2`，`unique_windows_per_batch=2`。
- 三臂 strata 全部匹配，visual update anchor 均有 frozen teacher cache。
- 三臂 Actor、value、exploration Actor/value 和 slow value hash 均保持不变。
- 三臂 visual student、visual objective 和 world model hash 均发生变化。
- 梯度与 loss 全部有限；单臂 2 update 墙钟约 9.34～10.82 秒。

结论：`smoke_passed`。该结果只证明 schedule、缓存、梯度和冻结组件链路有效，不比较
2-update loss，不形成候选优劣结论。下一步按冻结合同串行运行 9 个 1,600-update formal。

### 正式三臂：`rejected`

- 源码提交：`c0053872eb7887905e2c9730e1d98b466a596ea2`
- aggregate：
  `runs/research-loop/0003/r0003-p05-batch-arms-s20261205-aggregate.json`
- aggregate SHA-256：
  `8fbd004092d86da680f1a3a4359d9ab92630a9d6a9e8d79b0bdb49c3b2bed2d4`
- 3 个 seed×3 个 arm，共 9 个 1,600-update run。
- 每 run 8 个冻结 audit，共 72 个 audit。
- 九份 report/manifest 各含 14 个 artifact；全部 SHA-256 与字节数验证通过。
- 无 failure artifact；所有 run 的 Actor/value/slow-value hash 均保持不变。

| seed | arm | aggregate ratio | ratio p05 | visual ratio | proprio ratio | true error | 墙钟秒 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20261205 | A duplicate | 0.99981 | 0.99769 | 0.99828 | 0.99981 | 1.79745 | 6472 |
| 20261205 | B same-source | 0.99983 | 0.99772 | 0.99843 | 0.99983 | 1.50261 | 7436 |
| 20261205 | C cross-source | 1.00059 | 0.99809 | 0.99625 | 1.00059 | 1.99741 | 5899 |
| 202716734 | A duplicate | 1.00019 | 0.99944 | 0.99865 | 1.00019 | 1.04932 | 5864 |
| 202716734 | B same-source | 1.00083 | 0.99974 | 0.99784 | 1.00083 | 1.36053 | 5734 |
| 202716734 | C cross-source | 0.99990 | 0.99786 | 0.99629 | 0.99990 | 1.46467 | 5831 |
| 202821463 | A duplicate | 1.00041 | 0.99762 | 0.99810 | 1.00041 | 1.64985 | 5776 |
| 202821463 | B same-source | 0.99969 | 0.99606 | 0.99279 | 0.99969 | 1.97474 | 5849 |
| 202821463 | C cross-source | 0.99946 | 0.99588 | 0.99629 | 0.99946 | 1.50649 | 5720 |

C-B 的预注册最弱任务/模态 ratio 差为：

- seed `20261205`：`-0.002786`；
- seed `202716734`：`-0.002807`；
- seed `202821463`：`+0.003612`；
- 中位数：`-0.002786`，未达到 `+0.02`。

正式判定：

- `candidate_all_physical_ratios_at_least_1_05`：三个 seed 全失败；
- `candidate_all_shuffle_families_pass`：三个 seed 全失败；
- C 最弱 ratio 高于 B：仅一个 seed 成立；
- 真实动作绝对误差非回归：三个 seed 全失败；
- action execution 非回归：三个 seed 全失败；
- collision 非回归、墙钟增幅和冻结组件守护：通过。

因此跨 source batch 不是当前 action utilization 失败的有效修复，P05 为 `rejected`。
不得从单个任务或 seed 的局部方向宣称收益。按预注册路由重审 P06，但原 shuffled-margin
设计因评测泄露被放弃，只允许先验证真实动作 posterior overshooting 的离线预检。

## `R0001-P06`：`rejected without training`

- 源码提交：`40cadf2`
- run：
  `runs/research-loop/0003/r0003-p06-posterior-overshooting-s20261306`
- report SHA-256：
  `639b6f260b522c844a331f5dc5ee88f126dbdf320442773390df6d2f8ddc6a51`
- manifest SHA-256：
  `cb6d1d42fe39d236b4fdb1b26014222249a1ae763393c0929394b8fda26c5128`
- 24 个 source Episode、24 个确定性窗口、25 个 artifact 全部 hash 通过。
- checkpoint manifest/artifact hash 与冻结合同一致。
- action gradient norm：`0.397753`，有限且 `>1e-6`。
- posterior target 无梯度。

| condition | h1 | h2 | h4 | h8 | mean |
|---|---:|---:|---:|---:|---:|
| true action | 10.4032 | 13.5121 | 19.4607 | 27.5470 | 17.7308 |
| zero action | 10.1448 | 13.3864 | 19.7262 | 28.2961 | 17.8884 |
| shifted action | 10.4098 | 13.5171 | 19.4774 | 27.5617 | 17.7415 |

- true/zero ratio：`0.99119`，只改善 0.88%，未达到 5%。
- true/shifted ratio：`0.99940`，只改善 0.06%，未达到 5%。
- true 只在 2/4 horizon 同时优于两个负控。

因此 P06 预检失败，不启动训练、不扫描权重或 horizon。下一步检查标准 RSSM
dynamics KL 是否因 `free_nats=1.0` 长期处于梯度死区。

## `R0001-P19`：`diagnostic_failed`

- 源码提交：`d514ad8`
- run：
  `runs/research-loop/0003/r0003-p19-free-nats-deadzone-s20261319`
- report SHA-256：
  `6aa49e763fc557822fbd0aede6ef5f71aeede7e67728ae92a539b4a3c05691c6`
- manifest SHA-256：
  `8b0aaa2119b0cf2f2205608a2cb1d2702c9e7853039ace3e84db746020292b7a`
- 24 Episode、384 transition、25 artifact 全部 hash 通过。

raw dynamics KL 分布：

- minimum `0.3379`；
- p05 `1.1411`；
- median `8.0475`；
- p95 `24.2462`；
- maximum `58.6060`；
- `<1.0` 比例 3.91%，`<0.1` 比例 0%。

梯度：

| 条件 | loss | prior 参数梯度 norm | action 梯度 norm |
|---|---:|---:|---:|
| current clamp 1.0 | 10.4170 | 56.0456 | 0.6970 |
| candidate clamp 0.1 | 10.4032 | 56.1668 | 0.6984 |
| raw | 10.4032 | 56.1668 | 0.6984 |

candidate 与 raw 参数梯度 cosine 为 1.0。current 梯度并未接近零，且绝大多数 KL 已高于
1.0；因此 free-nats 不是主要梯度死区，0.1 不进入训练。

下一步 P20 检查未归一化物理 action 在 1024+16 维 RSSM transition 输入中的实际
preactivation 贡献，仍只做冻结训练 Replay 诊断。

## `R0001-P20-E1`：`invalid execution`

- 源码提交：`be678cceaade0a4ccc36bf81bcce0320f64d25c4`
- run：
  `runs/research-loop/0003/r0003-p20-action-input-contribution-s20261320`
- failure SHA-256：
  `52c9ad040cf6749df5467fcf77e07f9388508326a45e9b675f14e26ac578b20a`
- manifest SHA-256：
  `2f4f0361873858ab6fd551e71408b72db192e259749434637e6326fc0e88c616`
- manifest 中唯一 artifact 为 `failure.json`，其 SHA-256 与字节数验证通过。
- 完成 Episode：`0/24`；没有 `episodes/*.json`，也没有 `report.json`。
- 失败发生在首个 Episode 的 contribution RMS 统计：MPS 不支持设备侧 Tensor 转
  `float64`，`value.double()` 抛出 `TypeError`。

该执行没有产生任何冻结指标，不能按 P20 阈值归因。失败目录永久保留，不删除、不覆盖、
不复用。

首次恢复计划只修复 MPS 搬运顺序，但在 R1 启动前的三份创新审查和两份独立筛选中发现：

- aggregate 对 Episode RMS 做算术平均，不是平方池化的 pooled RMS；
- canonical 的仿射平移会把 gripper DC 偏移计入未中心化 contribution gain；
- bounds、双侧 column norm、bias 和完整窗口血缘未全部落盘。

两位筛选均给出 `changes_required`。R1 路径从未创建，也没有看到任何 R1 指标；因此允许在
结果前修订评测实现和冻结合同。修订后判定只使用 Episode 内去均值 variation RMS，绝对/
DC RMS 仅作描述，aggregate 使用平方池化；checkpoint、24 个窗口、action、canonical
公式和原阈值保持不变。

## `R0001-P20-R1`：`diagnostic_failed`

- 评测实现提交：`5fca360`
- source commit：`c6493cfb32d4738ed8c624a73ebb0461034348bf`
- run：
  `runs/research-loop/0003/r0003-p20-action-input-contribution-s20261320-r1`
- report SHA-256：
  `1e94233f14c54fcc8beda7943aa5db9d243d05908aff9d2cc74cf67c8b3dc77c`
- manifest SHA-256：
  `1c071f74687afe5c471806321bb714cb07070f0b6d743ec471abb380748b3d3c`
- 24 个 Episode report、aggregate report，共 25 个 manifest artifact；全部 SHA-256
  与字节数验证通过，无 `failure.json`。
- Replay manifest、24-window selection、checkpoint manifest/artifact hash 与冻结值一致：
  - Replay：
    `c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985`；
  - windows：
    `ecb75110942b7411de483265181fa732b1dbafccf06527d94388315fd372375f`；
  - checkpoint：
    `72f9361762d7ff5086f086b9ae1db05396caa3cf91822ece20686095df4ad75b` /
    `ef24bdfcca3cc46274bdfebc1d8b1a4afc81c73abff3aa4128e393e6da2109c6`。

### Aggregate

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| raw/stochastic variation ratio | 0.16974 | `<0.20` | 通过 |
| canonical/raw variation gain | 2.37899 | `>=1.50` | 通过 |
| canonical finite | 6144/6144 | 全部 | 通过 |
| canonical in bounds | 6144/6144 | 全部 | 通过 |
| Episode 同时通过 | 17/24 | `>=20/24` | 失败 |

variation contribution RMS：

- stochastic：`0.0814234`；
- raw action：`0.0138211`；
- canonical action：`0.0328803`。

描述性 absolute/DC 结果：

- absolute stochastic/raw/canonical：`0.138415 / 0.0280656 / 0.0592188`；
- DC stochastic/raw/canonical：`0.111933 / 0.0244265 / 0.0492520`。

### Episode 一致性

- raw ratio `<0.20`：`17/24`，范围 `0.079955`～`0.339840`，中位数
  `0.174389`；
- canonical/raw gain `>=1.50`：`24/24`，范围 `2.19342`～`3.04170`，中位数
  `2.32266`；
- 两条件同时通过：`17/24`。

任务分层通过数：

- `clear_dining_table_3d/v1`：`2/6`；
- `store_kitchen_items_3d/v1`：`5/6`；
- `tidy_living_room_3d/v1`：`10/12`。

结论：canonical normalization 的量级增益稳定存在，但 raw action 相对 stochastic 偏弱并未
在至少 20 个 Episode 中成立，尤其 clear dining table 只有 2/6 通过。按照预注册门槛，
P20 为 `diagnostic_failed`，action-scale 假设被拒绝；不进入 normalization smoke，不扫描
阈值，不选择任务子集。下一步检查 posterior state shortcut。

## `R0001-P21`：`diagnostic_failed`

- 实现提交：`55515dc`
- source commit：`d1e13d22f68ba3d37a7c0a8c24541ffe270aff65`
- run：
  `runs/research-loop/0003/r0003-p21-layerwise-action-effect-s20261321`
- report SHA-256：
  `3fed03ab1235c211749563f3fecfabb750cd4700a526c9784fdcaafce250ddde`
- manifest SHA-256：
  `437a925e556817fca2673ec24971b964c41d0005782c783b4a9945c3d4f94f65`
- 24 个 Episode report 与 aggregate report，共 25 个 manifest artifact；全部 SHA-256
  与字节数验证通过，无 `failure.json`。
- device 为 `mps`，frozen invocation、Replay/window/checkpoint hash 与 P20-R1 冻结值一致。

### Aggregate 判定

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| Episode 通过 | 8/24 | `>=20/24` | 失败 |
| shift=1 通过 | 7/24 | `>=18/24` | 失败 |
| shift=5 通过 | 8/24 | `>=18/24` | 失败 |
| shift=9 通过 | 8/24 | `>=18/24` | 失败 |
| clear dining table | 2/6 | `>=5/6` | 失败 |
| store kitchen items | 6/6 | `>=5/6` | 通过 |
| tidy living room | 0/12 | `>=10/12` | 失败 |
| 首次低 retention 集中 | 无 | `>=16 Episode` | 失败 |

三个 aggregate core check 和 concentration 均失败，因此为 `diagnostic_failed`，不是
`inconclusive`。

### Shift 机制分解

72 个 shift 的共同守护：

- transition activation effect 全部 `>=0.05`；
- main stage active 维、action/h input active 维全部满足；
- retention 分母、local sensitivity 全部有效；
- 所有 stage effect 有限；
- 四次 GRU error 全部有限且 `<=1e-5`。

判定分歧只来自：

- `first_low_retention_exists`：`23/72` 通过；
- 对应 sensitivity ratio `<0.50`：相同 `23/72` 通过。

通过 shift 的首次低 retention 位置：

- `activation -> h_next`：7；
- `h_next -> prior probability`：16。

描述性分布：

| 指标 | minimum | median | maximum |
|---|---:|---:|---:|
| transition activation effect | 0.10460 | 0.33606 | 0.75298 |
| activation→h retention | 0.38525 | 0.60434 | 0.90340 |
| h→prior probability retention | 0.22042 | 0.67735 | 1.13392 |
| h_next action/deterministic sensitivity | 0.04106 | 0.10428 | 0.24126 |
| prior probability action/deterministic sensitivity | 0.01131 | 0.06523 | 0.15530 |

GRU update gate 的 24-Episode 描述统计：

- median 的 Episode 均值：`0.80435`；
- p05 的 Episode 均值：`0.21708`；
- p95 的 Episode 均值：`0.94509`。

结论：action effect 能稳定进入 transition activation，且相对 deterministic 的局部
sensitivity 很低，但大多数 shift 的 effect 并未在任一相邻层下降到 0.50 以下；该联合
shortcut 假设只在 store kitchen items 稳定成立，在 tidy living room 完全不成立。P21
正式拒绝，不进入 GRU/prior preservation smoke，不把 23 个通过 shift 或低 sensitivity
ratio 单项挑选为正证据。下一步重新审查 decoder/output insensitivity 或目标定义。

## `R0001-P23`：`diagnostic_failed`

- 实现提交：`a009515`
- source commit：`a2c11a17a686eb529ee22901dd7edf56d42eda5d`
- run：
  `runs/research-loop/0003/r0003-p23-prior-argmax-s20261323`
- report SHA-256：
  `eff6fbf62fe77ef956ee20a05de87115873a65c8055c9d99ed74af3ee0851fca`
- manifest SHA-256：
  `42c7e7cd65ab6c280ac50e3087180607253915e5d14bf97c88a4cf6b5a1a9279`
- 24 个 Episode report 与 aggregate report，共 25 个 manifest artifact；全部 SHA-256
  与字节数验证通过，无 `failure.json`。
- device 为 `mps`，frozen invocation、Replay/window/checkpoint hash 与冻结值一致。

### P23 机制判定

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| Episode 通过 | 5/24 | `>=20/24` | 失败 |
| shift=1 通过 | 5/24 | `>=18/24` | 失败 |
| shift=5 通过 | 4/24 | `>=18/24` | 失败 |
| shift=9 通过 | 7/24 | `>=18/24` | 失败 |
| clear dining table | 4/6 | `>=5/6` | 失败 |
| store kitchen items | 0/6 | `>=5/6` | 失败 |
| tidy living room | 1/12 | `>=10/12` | 失败 |

72 个 shift 的守护：

- active probability coverage：`72/72`；
- flip fraction `<=0.10`：`72/72`；
- near tie count 为 0：`72/72`；
- flip/crossing 一致：`72/72`；
- independent hard code 与 sample=False 一致：`72/72`；
- 全部值有限：`72/72`。

主判定分歧：

- probability effect `>=0.05`：`61/72`；
- probability-to-code retention `<0.50`：`20/72`。

描述性分布：

| 指标 | minimum | median | maximum |
|---|---:|---:|---:|
| probability effect | 0.01470 | 0.13324 | 0.37282 |
| hard code effect | 0.00000 | 1.47370 | 10.72086 |
| probability→code retention | 0.00000 | 12.53875 | 105.04652 |
| argmax flip fraction | 0.00000 | 0.001953 | 0.027344 |

argmax flip 虽然稀少，但 one-hot jump 相对同一 probability natural scale 通常很大，并未在
大多数 Episode 中形成 `<0.50` retention。正式 deterministic argmax 不是普遍 action-effect
抹除点，P23 为 `diagnostic_failed`；不得从低 flip fraction 单项宣称 argmax 瓶颈。

### P24 准入守护

- hard feature effect Episode：`24/24`；
- clear/store/tidy：`6/6、6/6、12/12`；
- 72 个 shift 全部过线，effect 范围 `0.06066`～`0.42971`，中位数 `0.22209`。

因此 P23 阴性后，action-conditioned `[h_next,z_hard]` effect 仍稳定到达 decoder 输入，满足
预注册的 P24 重审条件。该结果只准许重新筛选 P24，不直接证明 decoder 低 gain。

## `R0001-P24-E1/R1`：`measurement invalid`

E1：

- source commit：`107b4c7e68fe407b79910daed3c62e0dc2ecee3e`
- report/calibration/manifest SHA-256：
  - `fcf1b5dad3b93316054a5c884e13c8c35d0417d83a6ae745cabe3dda988f6cb5`
  - `16d4d6be2390415e215c5f02a61325171d38cfbebad4e0da67ab26c90b085337`
  - `2ce984233c4ce3a3c3aa932f9e8d3f1765fe5a315c5e4e13cc0a17928a521dda`
- 完成 24/24 Episode，但手工 LayerNorm 分段与 MPS fused head 的 float32 次序差触发过严
  endpoint 门，不能形成机制结论。

R1：

- source commit：`97cbdcfef8505d8f6c8b7c24e520a281e5e7df8b`
- report/calibration/manifest SHA-256：
  - `619c4f2d7749555768937899e8acad6ffbc1e3f82a859bd392086a8425ea891e`
  - `16d4d6be2390415e215c5f02a61325171d38cfbebad4e0da67ab26c90b085337`
  - `9b285ba522a3c00a062a8927f63e10a28f6c572c2864874d49ace28c4e880723`
- endpoint 修复成功：official/direct 144/144 exact，manual/direct 最大 `3.81e-6`；
- 仍因 4 个低于 `0.05` 的有效 feature effect 被误分类为 `jvp_invalid`，不能归因。

E1/R1 均永久保留，不重跑、不删除、不参与 P24 机制判定。

## `R0001-P24-R2`：`diagnostic_complete`

- 状态分类修复提交：`cf572d7`
- source commit：`cc5f2dd34176a52e3d867f34871b0353336d87c8`
- run：
  `runs/research-loop/0003/r0003-p24-decoder-gain-s20261324-r2`
- report SHA-256：
  `45cdc4be0c2120f1ec372fb2f6b37a3ad7a61a565643b9136e95c41b367871e8`
- calibration SHA-256：
  `16d4d6be2390415e215c5f02a61325171d38cfbebad4e0da67ab26c90b085337`
- manifest SHA-256：
  `40771d46d7d2549f0d55e7bc8dd6f2421576b1fb6b1298600145d5e9d799c19b`
- 24 Episode report、calibration、aggregate，共 26 个 manifest artifact；全部 hash/字节
  验证通过，无 failure artifact。
- E1+R1 recovery chain、Replay/window/checkpoint hash 与 frozen invocation 全部有效。

### Visual head

| 指标 | 结果 |
|---|---:|
| head state | `not_localized` |
| valid branch | 72/72 |
| output guard Episode | 24/24 |
| output guard clear/store/tidy | 6/6、6/6、12/12 |
| localized branch | 2/72 |
| localized Episode | 1/24 |
| branch state | 66 not_localized / 4 feature_guard_failed / 2 localized |

描述性分布：

- feature effect：`0.02421 / 0.09542 / 0.31565`（min/median/max）；
- output effect：`0.03782 / 0.17639 / 0.31703`；
- 两个 localized `feature_to_linear` actual retention：`0.33557 / 0.43416`；
- path retention：`0.33557 / 0.43416`；
- path reconstruction cosine `>=0.9999999998`，relative error `<=1.63e-5`。

### Proprioception head

| 指标 | 结果 |
|---|---:|
| head state | `not_localized` |
| valid branch | 72/72 |
| output guard Episode | 24/24 |
| output guard clear/store/tidy | 6/6、6/6、12/12 |
| localized branch | 5/72 |
| localized Episode | 1/24 |
| branch state | 63 not_localized / 4 feature_guard_failed / 5 localized |

描述性分布：

- feature effect：`0.02421 / 0.09542 / 0.31565`；
- output effect：`0.02982 / 0.12449 / 0.23805`；
- 五个 localized `feature_to_linear` actual retention：
  `0.28208`～`0.48340`，中位数 `0.46190`；
- path retention：`0.28208`～`0.48340`，中位数 `0.46190`；
- path reconstruction cosine `>=0.9999999998`，relative error `<=1.44e-5`。

结论：hard decoder feature effect 与 physical-head output effect 均稳定存活，但任一 head 都没有
系统性的相邻层 `<0.50` retention；P24 decoder low-gain 假设拒绝。按冻结决策表，两头都
是 `not_localized` 且 output guard 全过，因此 P25 可分头重筛 target scale/gradient 奖励；
不得指定少量 localized branch、不得合并 visual/proprio。
