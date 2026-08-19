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

## P06 预检后决策

- 24 个 source Episode、25 个 artifact 与 checkpoint/input hash 全部有效。
- action gradient norm 为 `0.39775`，有限且非零；posterior target 正确停止梯度。
- 真实动作 aggregate loss：
  - 相对 zero action 只低 0.88%，未达到 5%；
  - 相对 shifted action 只低 0.06%，未达到 5%；
  - 只在 2/4 horizon 同时优于两个负控。
- P06 标记为 `rejected without training`；不得扫描权重、horizon 或放宽负控门。

### 下一诊断

- P05 九个 run 的 dynamics/representation loss 长期精确落在 free-nats=1.0。
- 先选 P19 只读梯度诊断，区分：
  1. raw posterior-prior KL 本身没有 action 梯度；
  2. raw KL 有梯度，但被 free-nats 截断为零。
- P19 只比较冻结的 `1.0/0.1/raw` 三个预注册门，不训练、不读取正式 holdout。

## P19 诊断后决策

- 384 个 transition 的 raw dynamics KL：
  - median `8.0475`；
  - p05 `1.1411`；
  - 只有 3.91% 低于 1.0，0% 低于 0.1。
- current free-nats=1.0 的 prior 参数梯度 norm `56.05`，raw 为 `56.17`；
  current 并未处于梯度死区。
- candidate=0.1 与 raw 完全一致只是因为所有 KL 都高于 0.1，不支持改阈值。
- P19 标记为 `diagnostic_failed`，free-nats 0.1 不进入训练。

### P20 选择

- RSSM 首层输入为 1024 维 stochastic + 16 维物理单位 action。
- action 权重单元素 RMS 不低于 stochastic，但 action 维数少 64 倍，且连续 action 未按
  canonical bounds 归一化。
- 先做 P20 只读 preactivation 贡献诊断；不修改 checkpoint、不读取正式 holdout。

## P20 首次执行后实现审计

`R0001-P20-E1` 在 `0/24` Episode、尚未产生任何指标时因 MPS 设备侧 `float64`
转换失败。主 Agent 保留失败 run，并在 R1 执行前组织三份独立创新审查和两份独立筛选。

### 共同发现

1. MPS 修复本身成立：统计张量先搬到 CPU，再用 `float64` 归约，不改变模型输入或公式。
2. aggregate 不能对 Episode RMS 做算术平均，必须使用
   `sqrt(sum(n_i * rms_i^2) / sum(n_i))`。
3. canonical 映射含平移项，尤其 gripper `[0,1] -> [-1,1]`；未中心化 RMS 会把 DC
   偏移误当作 action variation，可能虚增 gain。
4. 判定必须基于 Episode 内去均值的 variation contribution；绝对 RMS 与 DC RMS 只作
   描述，不参与阈值。
5. 必须补齐 canonical bounds 比例/计数、非有限计数、stochastic/action column norms、
   Linear bias 和完整窗口血缘。

### 独立筛选

两位筛选 Agent 在互不交流的情况下均给出 `changes_required`。

| 筛选 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 4 | 2 | 3 | 2 | 3 | 5 | 4 | `changes_required` |
| B | 4 | 3 | 5 | 2 | 4 | 2 | 2 | `changes_required` |

筛选 A 的成本/风险高分表示更有利；筛选 B 的成本/风险高分表示成本或风险更高，因此这两列
不做横向总分。两者共同否决“只修 MPS 后直接执行 R1”，但没有否决 P20 假设。

### 主 Agent 决策

- E1 标记为 `invalid execution`，不计作 seed 或假设结果。
- 在看到任何 R1 指标前修订诊断定义和血缘合同。
- 修订仍是评测实现修复，不是能力改动；不得改 checkpoint、窗口、action、模型权重或
  原阈值。
- 修订实现再次通过独立筛选后，才允许唯一执行 R1。

### v2 最终复审

修订实现提交：`5fca360`。

- 判定使用每 Episode 沿时间轴去均值的 variation RMS；
- aggregate 使用按 transition count 加权的平方池化；
- absolute/DC、bias、stochastic/action 权重只作审计；
- 非有限、越界、重复 source/window 均确定性失败；
- Replay manifest 与 24-window hash 在创建 output 前硬校验；
- 参数 norm 统一先搬到 CPU，再用 `float64` 归约。

专项测试共 16 项，覆盖真实 Replay/window hash、MPS、常量 gripper、非有限、越界、
重复 identity、血缘失败不创建 output、不等长度平方池化、`None` 传播和阈值边界。
正确入口全量测试、354 文件尺寸门、架构门和物理完整性门均通过。

两位 v2 筛选 Agent 在补齐回归测试后均给出 `approve`，共同确认阻断项清零，允许在
干净已提交源码上唯一执行 R1；E1 不得重启，R1 不得重复。

## P20 正式诊断后决策

- R1 使用 source commit `c6493cfb32d4738ed8c624a73ebb0461034348bf`，24 个
  Episode、25 个 manifest artifact 与冻结输入血缘全部有效。
- aggregate raw action/stochastic variation ratio 为 `0.16974`，canonical/raw gain
  为 `2.37899`，canonical action 也全部有限且在界内。
- 但只有 `17/24` Episode 同时满足两个贡献条件，低于冻结的 `20/24`：
  - raw ratio `<0.20`：`17/24`；
  - canonical/raw gain `>=1.50`：`24/24`；
  - 两者同时通过：`17/24`。
- 七个失败 Episode 的 raw ratio 为 `0.227998`～`0.339840`；失败不是非有限值、越界、
  hash 漂移或执行异常。
- 任务分层通过数：
  - clear dining table：`2/6`；
  - store kitchen items：`5/6`；
  - tidy living room：`10/12`。

因此 raw action 偏弱和 canonical gain 在 aggregate 上成立，但跨 Episode 一致性不足；
不得把 aggregate 过线替代预注册的 `20/24` 门槛。P20 标记为 `diagnostic rejected`，
canonical normalization 不进入 2-update smoke。按冻结路由继续检查 posterior state
shortcut，不扫描 action-scale 阈值，也不按任务挑选子集。

## Posterior shortcut 创新编排

- 主 Agent 按 Agent 配置先后启动两批、共 6 个创新 Agent：第一批允许只读代码审查，
  第二批只要求基于完整事实直接输出文本提案。
- 两批 Agent 均在多次长等待和明确中断收束指令后保持 `running`，没有返回任何提案；
  主 Agent关闭全部线程，未采用或伪造其观点。
- 为避免研究循环因调度异常停滞，主 Agent仅基于已验证证据提出两个不重复的低成本诊断：
  - P21：逐级定位 action effect 在 transition、GRU、prior 哪一级衰减；
  - P22：比较 posterior 对 observation 与 prior deterministic 的经验干预敏感度。
- 两项都不得直接进入训练，必须先交给两个新建、互不交流的筛选 Agent独立评分。

## Posterior shortcut 独立筛选

两位筛选 Agent在互不交流的情况下均返回相同路由：P21 `approve/优先`，P22
`defer/依赖 P21`。

| 提案 | 筛选 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| P21 | E | 5 | 4 | 5 | 4 | 4 | 2 | 1 | `approve` |
| P21 | F | 5 | 4 | 5 | 4 | 3 | 4 | 5 | `approve` |
| P22 | E | 4 | 3 | 5 | 4 | 3 | 2 | 1 | `defer` |
| P22 | F | 4 | 3 | 5 | 4 | 3 | 4 | 5 | `defer` |

筛选 E 的成本/风险分数越高表示越不利；筛选 F 的成本/风险分数越高表示越有利，因此这两列
不跨筛选求和。

### 共同反驳

- P21 不重复 P06：P06 只证明端到端 posterior target loss 对 action 条件不敏感，P21
  定位内部 effect 首次在哪一层衰减。
- 原 P21 单一 derangement 可能 OOD，standardized denominator 接近零会虚高，有限差分
  尺度也可操纵结论。
- P22 的 obs/deterministic 置乱不一定同等 OOD；即使通过，也只证明 posterior target
  dominance，不能单独证明 action path 根因。

### 主 Agent 决策

- 不按总分机械选择；P21 直接回答冻结路由中的“posterior state shortcut 在哪一级”，
  且其结果可导向单一 GRU/transition/prior 候选，故优先。
- P21 按筛选意见改为三组 Episode 内经验循环置换、5% 凸组合局部干预、自然 variation
  分母保护和任务配额，不扫描 shift、epsilon 或阈值。
- P22 延后；只有 P21 证明 transition effect 存在但下游 retention/sensitivity 系统性不足，
  才允许继续判断 posterior observation dominance。

## P21 实现复审

- 实现提交：`55515dc`。
- 首轮实质复审发现并修复：
  - Episode 通过不得额外要求低 retention 位置共识；
  - evaluator/aggregate 必须严格拒绝非 16 transition；
  - 两级 retention 分母均需 `>=1e-6`；
  - 四次 GRU 一致性 error 必须全部有限；
  - 专用入口锁定冻结 input/checkpoint/output/device，禁止换路径或 CPU 重跑；
  - success/failure 产物补齐 criteria、device、invocation、stage、异常与全部 hash；
  - true GRU reset/update/new gate 分布作为描述指标落盘。
- 定向测试覆盖不同位置的 2/3 shift、inactive/subminimum denominator、GRU NaN、非16、
  结构漂移、非冻结 invocation、checkpoint manifest/artifact mismatch、Replay/window
  mismatch、重复 source/window、aggregate pass/inconclusive/fail，以及 success/failure
  manifest 血缘。
- 最终：
  - P21 专项测试 27 项通过；
  - P21/P20/P06/P19 相关测试 61 项通过；
  - 正确入口全量测试通过；
  - 357 文件尺寸门、架构门和物理完整性门通过；
  - 两位独立复审 Agent 均给出 `approve`，确认阻断项清零。

主 Agent据此批准提交并在干净、已 push 且冻结 run 路径不存在时唯一执行一次 P21。
