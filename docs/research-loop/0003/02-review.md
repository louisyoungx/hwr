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

## P21 正式诊断后决策

- source commit：`d1e13d22f68ba3d37a7c0a8c24541ffe270aff65`。
- 24 个 Episode、25 个 manifest artifact、冻结 invocation 和输入血缘全部有效。
- 所有 72 个 shift：
  - transition activation effect 均 `>=0.05`；
  - stage/local sensitivity 全部有限有效；
  - GRU 复刻与 `nn.GRUCell` 全部一致；
  - action 与 deterministic active 维全部满足要求。
- 但只有 `23/72` shift 同时出现首次低 retention 且对应 sensitivity ratio `<0.50`：
  - `activation -> h_next`：7 个通过 shift；
  - `h_next -> prior probability`：16 个通过 shift；
  - 其余 49 个 shift 没有 `<0.50` 的相邻 retention。
- Episode 通过数仅 `8/24`，任务分层为：
  - clear dining table：`2/6`；
  - store kitchen items：`6/6`；
  - tidy living room：`0/12`。
- shift 通过数为 `7/24、8/24、8/24`，均低于冻结的 `18/24`。

描述性结果显示 action/deterministic local sensitivity ratio 确实偏低：

- `h_next` 中位数 `0.1043`；
- prior probability 中位数 `0.0652`。

但 P21 的核心假设不是“action sensitivity 低”单项，而是 transition effect 存在、在固定相邻
层系统性衰减、且相对 deterministic sensitivity 低的联合条件。绝大多数 shift 没有低
retention，且 tidy living room 为 0/12，因此不得用 sensitivity ratio 单项覆盖硬门。

主 Agent将 P21 标记为 `diagnostic rejected`。不提出 GRU 或 prior preservation smoke；
P22 不自动启动。按冻结路由重新审查 decoder/output 对 latent action effect 的不敏感，或
posterior target/训练目标定义。

## Decoder/output 创新提案

三位创新 Agent 独立审查后形成相同的串行因果链：

1. P23：先判断连续 prior probability effect 是否在 deterministic argmax one-hot 处丢失；
2. P24：只有 hard decoder feature effect 存活时，才定位 visual/proprio decoder 逐层 gain；
3. P25：只有 decoder output effect 存活但物理 error 仍不区分 action 时，才诊断 target
   scale、residual 与 loss gradient 奖励。

### 合并与保留意见

- 两位 Agent 提议在 P23 加 coupled sampling 或 Actor JVP；主 Agent未采纳：P23 的主问题
  是正式 `sample=False` 路径的 deterministic argmax，加入采样/Actor 会同时改变推理语义
  和下游模块，不利于单变量归因。
- P24 保留 actual finite effect 与 local JVP 双重校验，防止 LayerNorm 非线性让单一指标
  误导；visual/proprio 分头判定。
- P25 保留 raw/whitened 指标与 gradient projection，但严格依赖 P24；whitening 只作诊断，
  不得直接改写训练 loss。
- P22 继续延后，不与 P23 并行；P21 未支持普遍 deterministic shortcut 后，先检查正式
  argmax 和 decoder 链条更直接。

P23/P24/P25 均不得直接进入训练；先交两个新建、互不交流的筛选 Agent 独立评分。

## Argmax/decoder 独立筛选

| 提案 | 筛选 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| P23 | I | 4 | 3 | 3 | 5 | 3 | 5 | 2 | `defer, revise` |
| P23 | J | 4 | 4 | 5 | 5 | 3 | 5 | 2 | `approve` |
| P24 | I | 3 | 2 | 4 | 4 | 3 | 4 | 3 | `defer` |
| P24 | J | 4 | 3 | 4 | 4 | 4 | 3 | 2 | `defer` |
| P25 | I | 3 | 1 | 3 | 2 | 2 | 4 | 2 | `defer` |
| P25 | J | 3 | 2 | 4 | 4 | 3 | 3 | 3 | `defer` |

筛选 I 的成本/风险高分表示更有利；筛选 J 的风险高分表示风险更高，因此不跨筛选汇总。

### 共同反驳

- 原 P23 的 probability 与 hard code effect 若各用自身自然尺度，retention 没有共同量纲。
- 原 `hard code active >=25%` 的分母未定义，并可能与 `flip <=10%` 结构冲突。
- margin 必须对 true winner 定义有向消耗；near tie 与 backend tie-break 需 fail-closed。
- 24 Episode 是统计单位，三个 shift 只是 Episode 内重复证据，不得当作 72 个独立样本。
- P24/P25 严格依赖上游；不得因 P23 阴性而绕过门槛继续扩大诊断链。

### 主 Agent 决策

- 不按一票 `approve` 直接执行 P23；先修订共同量纲和边界语义，再交原两位筛选复审。
- 修订版统一使用 1024 categorical 坐标、true probability natural variation scale 和同一
  active mask；删除 hard-code active 门。
- top-1 margin 固定为 true winner 对最佳竞争类的 signed margin；margin `<=1e-8` 的
  near tie 使该 shift 失效。
- P24/P25 继续 defer；P23 修订版未获两位筛选放行前不冻结、不实施。

### P23 修订复审

- 筛选 J 在第一次修订后 `approve`。
- 筛选 I 要求把 active mask、effect、retention、margin/crossing、near tie 和统计层级写成
  可执行公式；主 Agent已全部冻结到 `03-experiment.md`。
- 最终筛选 I/J 均 `approve P23`，共同确认：
  - probability/hard code 使用同一 1024 坐标、true probability scale 和 active mask；
  - retention 分母 fail-closed，不使用 epsilon；
  - margin/crossing 固定 true winner，near tie fail-closed；
  - Episode 是独立单位，任务和 shift 配额不能被 aggregate 覆盖；
  - sample=False 一致性只作实现门，不算机制证据。

主 Agent不按初始总分直接选择，而是在消除两份筛选共同指出的结构矛盾后批准 P23。P24/P25
继续 defer，严格等待 P23 结果。

## P23 实现复审

- 实现提交：`a009515`。
- 首轮复审发现并修复：
  - app 不得调用 `world_model.observe()`，只允许 observation encoder + `rssm.observe`，
    防止 decoder/aux heads 越界执行；
  - hard feature 只作 P24 准入守护，不得污染 P23 主判定；
  - 固定正式 `16×32×32`、1024 probability 坐标和至少 256 active 维；
  - flip/crossing 或 sample=False 一致性失败升级为全局 `diagnostic_invalid`，不能被
    Episode 2/3 或 aggregate 配额掩盖；
  - failure 记录 criteria、invocation、当前 source/window、异常和全部 hash，Episode
    artifact 成功写盘后才增加完成计数。
- 定向测试覆盖共同 scale/mask、margin/crossing、near tie、subminimum denominator、
  independent one-hot oracle、sample=False 一致性、hard-feature 独立守护、全局 invalid、
  frozen invocation、checkpoint/Replay/window hash 和 success/failure manifest。
- 最终：
  - P23 专项测试 25 项通过；
  - P23/P21/P20/P06/P19 相关测试通过；
  - 正确入口全量测试通过；
  - 360 文件尺寸门、架构门和物理完整性门通过；
  - 两位独立实现复审 Agent 均 `approve`。

主 Agent批准在实现与本复审记录均 push、工作区干净、冻结 run 路径不存在时唯一执行 P23。

## P23 正式诊断后决策

- source commit：`a2c11a17a686eb529ee22901dd7edf56d42eda5d`。
- 24 个 Episode、25 个 manifest artifact、frozen invocation 与全部输入血缘有效。
- 所有 72 个 shift：
  - active probability 覆盖均通过，active count 为 1022～1024/1024；
  - argmax flip fraction 均 `<=0.10`；
  - near tie count 均为 0；
  - flip/crossing 与 sample=False 实现一致性全部通过；
  - 全部值有限。
- 但 P23 联合机制只通过 `5/24` Episode：
  - shift 通过数 `5/24、4/24、7/24`；
  - clear dining table `4/6`；
  - store kitchen items `0/6`；
  - tidy living room `1/12`。
- probability effect `>=0.05` 在 `61/72` shift 成立，但 probability-to-code retention
  `<0.50` 仅 `20/72`；其 retention 中位数为 `12.5388`，hard one-hot effect 通常不是被
  抹除，而是相对同一 probability natural scale 被放大。
- P23 因此标记为 `diagnostic rejected`，不得提出 sampling/argmax 修改。

P24 准入守护独立通过：

- hard feature `[h_next,z_hard]` effect 的 Episode 守护 `24/24`；
- clear/store/tidy 为 `6/6、6/6、12/12`；
- 72 个 shift hard feature effect 全部 `>=0.05`，范围 `0.06066`～`0.42971`。

这只证明 action-conditioned hard feature 稳定到达 decoder 输入，不证明 decoder 已经低 gain。
按预注册路由允许重审 P24；P24 仍需独立筛选与完整冻结，不能因守护通过直接实施。

## P24 独立重筛

| 筛选 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施可行性 | 回归风险 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M | 4 | 3 | 3 | 3 | 3 | 4 | 4 | `revise` |
| N | 4 | 4 | 3 | 2 | 3 | 4 | 5 | `revise` |

两份评分均以 5 为更有利。

### 共同反驳

- hard one-hot true→shift 是有限跳变，单点 JVP 不得作为通过门。
- LayerNorm 需要拆分去均值/方差归一化与 affine；其抑制只能称为幅度定位，不能直接称
  信息丢失。
- stage/output scale 必须在比较 shift 前由冻结 true branch 全局校准，不能按 Episode
  自适应或结果后重算。
- 原“某一相邻层”允许结果后挑层；必须按固定顺序选择首个低 retention 边界。
- visual/proprio 必须独立满足 Episode/任务配额，不能池化补足；定位不同形成两个结论。
- P25 必须使用 P24 冻结的分头/分层结果，不能选择有利子集。

### 主 Agent 决策

- P24 保留重审资格，但当前版本不实施。
- 修订版用 24 true branch 的 384 transition 预先冻结每个 head/stage/coordinate scale。
- 有限跳变使用固定 16 段 midpoint path-integrated JVP 重建；单点 JVP仅作描述。
- 按 `feature→linear preactivation→LN normalized→LN affine→SiLU hidden→output` 固定顺序
  选择首个 actual retention `<0.50` 的边；同一边 path-JVP retention 和重建一致性必须
  同时通过。
- P24通过也只证明 decoder 网络链路幅度衰减，不能宣称 hard feature 已编码有效物理状态。

### P24 修订复审

- 两位筛选 Agent 都要求把 calibration、branch→Episode→head 聚合和 P25 路由完全机械化。
- 主 Agent补充冻结：
  - 384 个 true transition 的 exact mean/scale、一次性 active mask 与 calibration artifact；
  - branch 固定为 `head × Episode × shift`；
  - 固定顺序的首个低 retention 边不可跳过；
  - actual/path 使用同一 mask、scale、active RMS 与分母；
  - Episode 2/3 branch 同边，head 独立满足 20/24、任务、shift 与16-Episode集中度；
  - P25 按 `passed(edge)/not_localized/output_guard_failed/jvp_invalid/异头` 五状态预注册。
- 最终筛选 M/N 均 `approve`，共同确认：P24 只支持 decoder 幅度衰减位置，不支持 hard
  feature 物理语义或能力提升结论。

主 Agent批准 P24 进入实现；P25 继续 defer，等待 P24 分头状态。

## P24 实现复审

- 实现提交：`34a29ab`。
- 首轮复审发现并修复：
  - calibration 严格两遍：第一遍只生成 true hard feature，calibration 写盘/hash/重载后
    第二遍才计算任何 shift/P23 guard；
  - P23 hard-code endpoint 与正式 `decode_features` endpoint 均独立逐元素校验；
  - 全局 feature effect `>=0.05` 使用 P24 calibration scale，不复用 Episode 自适应 guard；
  - 固定首低边后只计算该边 path-JVP，不让后续无关边污染结论；
  - actual retention 不可计算立即 branch invalid，不得跳后边或进入 `not_localized`；
  - selected JVP 不合格 branch 直接 `jvp_invalid`；
  - Episode 允许 2 个有效同边 branch 加 1 个 invalid branch；但 `not_localized` head
    要求 72/72 branch 全部有效；
  - head/LN/output checkpoint 维数、首遍 failure 血缘与 calibration artifact 均锁定。
- 为满足 800 行门，calibration/path-JVP 数学拆到独立模块；行为接口不变。
- 最终：
  - P24 core/app 专项测试 24 项通过；
  - P24/P23/P21/P20 相关测试通过；
  - 正确入口全量测试通过；
  - 365 文件尺寸门、架构门和物理完整性门通过；
  - 两位独立实现复审 Agent 均 `approve`。

主 Agent批准在实现与本复审记录均 push、工作区干净、冻结 run 路径不存在时唯一执行 P24。

## P24 E1 与 R1 测量复审

- E1 source commit：`107b4c7`，完成 24/24 Episode 与 calibration，但 aggregate 为
  `diagnostic_invalid`。
- 根因不是 decoder endpoint 真漂移：
  - 正式 `decode_features` 与直接 head 逐元素完全相等；
  - 手工 LayerNorm 分段与直接 head 在 MPS float32 的 maximum absolute difference 为
    `7.15e-7`～`1.91e-6`；
  - 原 `atol=1e-7` 对接近零输出形成假失败。
- R1 修复提交：`6b3fdc4`，只修改 endpoint 实现门与 recovery 路径：
  - official/direct 继续要求 exact；
  - manual/direct maximum absolute difference `<=5e-6`；
  - 报告 mean absolute difference并拒绝 NaN/Inf；
  - 入口只允许 `...s20261324-r1`，并在创建 output 前硬校验 E1 report/calibration/
    manifest/source hash；
  - success/failure 均记录 `recovery_of`。
- calibration、stage、retention、path-JVP、配额和 P25 五状态决策表保持不变。
- 最终：P24-R1 专项测试 34 项、相关测试、全量测试、365 文件尺寸门、架构门和物理完整性
  门通过；两位独立复审 Agent 均 `approve`。

主 Agent批准在修复与本复审记录 push 后唯一执行 P24-R1；E1 永久保留，不得重跑。

## P24 R1 与 R2 状态复审

- R1 source commit：`97cbdcf`，endpoint 修复成功：
  - official/direct 144/144 exact；
  - manual/direct maximum absolute difference 最大 `3.81e-6`，全部低于 `5e-6`。
- R1 仍 `diagnostic_invalid` 的原因不是 endpoint 或 JVP，而是 4 个 Episode 的 shift=1
  feature effect `<0.05` 被错误归类为 `jvp_invalid`。
- R2 修复提交：`cf572d7`，只重排 branch 状态优先级与 recovery chain：
  - 基础测量失效优先 `jvp_invalid`；
  - 测量有效但 P23/global feature guard 未过为
    `feature_guard_failed(valid=true, passed=false)`，不扫描 retention/JVP；
  - `not_localized` 可以包含 `feature_guard_failed`，但不得包含任何 `jvp_invalid`；
  - 入口锁定 `...s20261324-r2`，同时硬校验 E1+R1 report/calibration/manifest/source。
- 边界测试覆盖 effect `0/1e-8/0.049`、P23 finite/active 失效优先、aggregate
  `feature_guard_failed`/`jvp_invalid` 路由、双 recovery chain 与 exact R2 path。
- 最终：P24-R2 专项测试 41 项、相关测试、全量测试、365 文件尺寸门、架构门和物理完整性
  门通过；两位独立复审 Agent 均 `approve`。

主 Agent批准在 R2 修复与本复审记录 push 后唯一执行 P24-R2；E1/R1 均永久保留。

## P24-R2 正式诊断后决策

- R2 source commit：`cc5f2dd34176a52e3d867f34871b0353336d87c8`。
- 24 个 Episode、calibration、26 个 manifest artifact、E1+R1 recovery chain 与全部输入
  血缘有效；aggregate 为 `diagnostic_complete`。
- endpoint、calibration、active、retention denominator 和 path-JVP 测量已全部有效：
  - official/direct 144/144 exact；
  - manual/direct maximum absolute difference 最大 `3.81e-6`；
  - visual/proprio 的 72/72 branch 全部 valid。
- visual：
  - head 状态 `not_localized`；
  - output guard `24/24`，任务 `6/6、6/6、12/12`；
  - localized branch `2/72`，Episode `1/24`，只在 `feature_to_linear`；
  - branch 状态：66 `not_localized`、4 `feature_guard_failed`、2 `localized`。
- proprioception：
  - head 状态 `not_localized`；
  - output guard `24/24`，任务 `6/6、6/6、12/12`；
  - localized branch `5/72`，Episode `1/24`，只在 `feature_to_linear`；
  - branch 状态：63 `not_localized`、4 `feature_guard_failed`、5 `localized`。

两头均没有达到 20/24、任务、shift 和16-Episode集中度联合门槛，因此 P24 decoder
low-gain 假设正式拒绝；不能从少量 `feature_to_linear` branch 宣称 decoder 瓶颈。

按结果前冻结的五状态决策表，两头同时满足 `not_localized`：

- output effect guard 24/24 与任务配额全部通过；
- 所有 branch 测量有效；
- 无系统低 retention。

因此 P25 对 visual 与 proprio 分头获得重审资格。P25 必须固定使用全部24 Episode、P24
calibration 和全部有效 branch，不指定有利边；visual/proprio 不得池化或捆绑成一个结论。
