# R0003 独立评审与筛选

## 独立评分

| Agent | ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| D | `R0001-P14` | 5 | 5 | 5 | 3 | 5 | 5 | 4 | 评测修复 |
| D | `R0001-P15` | 3 | 2 | 5 | 4 | 4 | 4 | 4 | `defer` |
| D | `R0001-P16` | 5 | 4 | 5 | 5 | 5 | 5 | 5 | 评测修复 |
| D | `R0001-P11` | 5 | 5 | 5 | 4 | 5 | 3 | 3 | 条件 `select` |
| D | `R0001-P05` | 4 | 4 | 5 | 5 | 4 | 3 | 4 | 条件 `select` |
| D | `R0001-P10` | 4 | 4 | 5 | 3 | 4 | 4 | 3 | `defer` |
| E | `R0001-P14` | 5 | 5 | 5 | 4 | 5 | 4 | 4 | 评测修复 |
| E | `R0001-P15` | 4 | 3 | 5 | 4 | 4 | 4 | 3 | `defer` |
| E | `R0001-P16` | 5 | 4 | 5 | 5 | 5 | 5 | 4 | 评测修复 |
| E | `R0001-P11` | 5 | 5 | 5 | 4 | 5 | 3 | 3 | 条件 `select` |
| E | `R0001-P05` | 4 | 4 | 5 | 5 | 4 | 3 | 4 | 条件 `select` |
| E | `R0001-P10` | 4 | 4 | 5 | 3 | 4 | 4 | 3 | `defer` |

## 共同反驳

- `R0001-P14` 恢复更多合法起点，本质上提高有效回归行数；只能算测量修复，不能宣称数据效率。
- 重叠起点不能作为独立 bootstrap 样本。
- P09 数据已被查看，只能用于机制发现。
- `R0001-P15` 均匀起点可能删除困难交互样本。
- `R0001-P16` 合成生成必须保留 Episode 自相关、窗口重叠和真实设计维度。
- `R0001-P11/P05/P10` 不得与测量修复首次捆绑。

## 主 Agent 决策

1. 先执行 `R0001-P16`。
2. 只有连续全起点设计功效合格，才允许 `R0001-P14` 新 seed 确认。
3. `R0001-P14` 后 h1 仍失败才执行 `R0001-P15`。
4. 测量合同稳定后，`R0001-P11` 与 `R0001-P05` 从同一父基线独立启动。
5. `R0001-P10` 最后触发。

## P16 后独立筛选

| Agent | ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 建议 |
|---|---|---:|---:|---:|---:|---:|---|
| D | `R0001-P17` | 5 | 3 | 5 | 4 | 4 | 条件 `select` |
| D | `R0001-P11` | 4 | 3 | 4 | 3 | 4 | `defer` |
| D | `R0001-P18` | 2 | 2 | 4 | 2 | 2 | `reject` |
| E | `R0001-P17` | 5 | 3 | 5 | 3 | 4 | 条件 `select` |
| E | `R0001-P11` | 4 | 3 | 4 | 3 | 4 | `defer` |
| E | `R0001-P18` | 2 | 2 | 3 | 1 | 2 | `reject` |

### 共同硬门

1. 主实验只从 Episode 初始 reset snapshot 分叉。
2. horizon 从 actual plant action 差首次非零开始。
3. 不删除安全裁剪、方向偏转或零响应样本。
4. 主 estimand、状态子集、动作方向、幅值、seed 和检验 family 在正式结果前冻结。
5. Episode 是随机化单位；多个 horizon 是簇内重复测量。
6. sham 报告 FPR 单侧置信上界；blind injection 报告 power 单侧置信下界。
7. 物理 snapshot/state 不得进入策略、训练或动作选择。

### 决策

- `R0001-P17`：选择，先做 snapshot、sham、injection 和 first-stage 预检。
- `R0001-P11`：等待 P17 建立稳定物理因果证据。
- `R0001-P18`：拒绝。

## P17 后主 Agent 筛选

三名创新 Agent 的有效结论一致指向：

1. `R0001-P11` 最便宜且机制最直接；
2. `R0001-P05` 需要 9 次固定重放，成本更高且不能修复动作时序；
3. `R0001-P10` 只处理安全正例稀疏，不能修复非干预 RMSE。

简单历史线性 probe 的反例：

| rho | 输入 | aggregate RMSE | lag0 RMSE | lag1 RMSE |
|---:|---|---:|---:|---:|
| 0.96 | current | 0.137 | 0.033 | 0.191 |
| 0.96 | history | 0.108 | 0.133 | 0.074 |
| 0.50 | current | 0.402 | 0.195 | 0.534 |
| 0.50 | history | 0.317 | 0.383 | 0.234 |

因此不选择直接拼接历史；先验证显式 causal plant FIFO estimator。

### 顺序

1. P11 head-only 因果 plant estimator；
2. P11 通过后实施正式 plant/safety 分解；
3. 若世界模型 action-shuffle 仍失败，执行 P05 三臂；
4. 若仅安全 recall/PR-AUC 失败，执行 P10。

## P11 正式确认后决策

- 144 Episode、18 分区和所有 artifact 完整，实验没有缺失或 lineage 漂移。
- 36 个 Episode 因 evaluation observation latency=3，从 step 3 起连续 61 次触发
  `outside_validity_window` 安全拒绝，没有 16 条共同非干预 feedback，按冻结合同导致
  18 个分区均失败。
- 不允许删除这 36 个 Episode、修改稳定阶段定义或延长 action validity 后重算 P11。
- 其余 108 Episode 的 4,860 个稳定 transition 对 lag1/2/3 全部识别正确，
  stable RMSE 最大 0.004865；这是机制子集证据，不能覆盖正式拒绝。
- 主 Agent 将 P11 标记为 `rejected`，按预注册路由选择 P05 三臂。
- stale-frame 是独立运行时问题；后续可单独提案，但不能与 P05 首次捆绑。

## P05 正式结果后决策

- 9 个 run、72 个 audit、9 个 checkpoint/report/manifest 与 aggregate 全部完整。
- 三个 seed 的 C-B 最弱任务/模态 ratio 差为
  `-0.00279, -0.00281, +0.00361`，中位数 `-0.00279`。
- C 在任一 seed 都没有使所有任务×物理模态达到 1.05，也没有通过五次 shuffle。
- C 相对 B 在前两个 seed 变差，第三个 seed 的微小正差也远低于预设 0.02。
- action execution 相对 B 回归，真实动作绝对误差守护也未通过；仅 collision 和墙钟守护
  通过。
- 因此 P05 标记为 `rejected`；不得把单 seed 或单任务的局部方向作为接受证据。

### P06 重审

- 原 P06 的 shuffled-action margin 直接复刻正式 audit，评测泄露风险不可接受。
- 保留“多步 prior 必须跟随真实动作”的机制问题，但将候选改为真实动作 posterior
  overshooting：
  - 只使用训练 Replay；
  - 未来 posterior latent 停止梯度；
  - 不生成 shuffled action，不读取正式 holdout；
  - 正式 action-shuffle 只作为实施完成后的独立评测。
- 主 Agent 只选择低成本离线预检；预检未证明 action 梯度与时间对齐时，不启动正式训练。
