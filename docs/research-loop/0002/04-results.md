# R0002 实验结果

## `R0001-P09`：`rejected`

### 完整性

- run ID：`r0001-p09-observation-lag-s20260901`
- 源码提交：`3d199c5bfe9ff17634ac423113e85beedc5b2be5`
- Episode：96/96
- artifact：97 个，逐文件 SHA-256 与字节数通过
- report SHA-256：
  `21d385fe4c39072652e1571a0e1254ab029b08548ad204651a50b9981c53a8cb`
- 任务、rho、lag、split 完全平衡。

### 聚合 Probe

| rho | 任务 | 最弱 ratio | 同步 p05 |
|---:|---|---:|---:|
| 0.96 | 餐桌 | 1.151361 | 1.022206 |
| 0.96 | 厨房 | 1.337591 | 1.200115 |
| 0.96 | 客厅 | 1.183184 | 1.050999 |
| 0.50 | 餐桌 | 1.707457 | 1.533979 |
| 0.50 | 厨房 | 1.510770 | 1.267329 |
| 0.50 | 客厅 | 1.702504 | 1.480154 |

### 守护失败

冻结合同要求 lag=1 的每个 horizon MSE 至少下降 10%，但：

- rho=0.96 的多数分区仅改善约 -2%～12%；
- rho=0.50 的 8/16-step 在三任务均恶化约 20%～34%。

因此“单一 observation lag 修复会跨 horizon 稳定改善”被否定。

该 run 第一次发布完整结果后，旧 Host runner 被 launchd 自动重启 9 次；后续启动均因 run
目录已存在立即失败，没有修改正式产物。后续 runner 已改为回调前显式移除自身服务。

## `R0001-P12`：`accepted as platform repair`

- 首个物理 Episode 前物化 9 条 evaluation instruction。
- evaluation-only cache 与训练 run 隔离。
- 已见 embedding 和确定性动作不变。
- focused tests 18 项、架构和尺寸门通过。
- 不计语言泛化提升。

## `R0001-P13`：`accepted as evaluation repair`

- acceptance schema 升级到 v3。
- 增加 Episode 安全干预负担的 p95、bootstrap upper 和 max 三重门。
- 高撞盾、零干预、单极端 Episode、重现性和不运动负对照均覆盖。
- 不修改安全层或策略。

## 路由

P01 拒绝后，完整 128-transition 轨迹与 R0001 的 7×16 Replay 出现明显证据池差异。下一轮
转向 Replay 连续性与 Action Probe 设计功效，见 `docs/research-loop/0003/`。
