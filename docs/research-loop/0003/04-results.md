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
