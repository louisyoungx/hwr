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
