# R0004 独立评审

## 评审过程

- 两个筛选 Agent 使用本机 `traex exec --ephemeral --sandbox read-only` 独立启动。
- 模型均为 GPT-5.6 Sol、极高推理强度。
- 两者只读 `AGENTS.md`、`00-context.md` 和 `01-proposals.md`，没有修改文件或启动训练。
- S1 完成前没有看到 S2 的评分；S2 完成前没有看到 S1 的评分。

## 评分维度

每项按 1–5 分独立评估：

1. 目标价值；
2. 证据强度；
3. 可检验性；
4. 因果可归因性；
5. 通用性；
6. 实施成本；
7. 回归风险。

## 强制拒绝条件

- 没有明确指标或无法证伪；
- 与 R0003 已拒绝假设重复且没有新证据；
- 依赖评测泄露、结果后选样或降低安全约束；
- 首次实验同时改变多个不可独立归因的主变量；
- 只优化训练代理量而没有预注册的物理闭环验证路径。

分数顺序统一为：

`目标价值 / 证据强度 / 可检验性 / 因果可归因性 / 通用性 / 实施成本 / 回归风险`

其中实施成本 5 表示低成本，回归风险 5 表示低风险。

## S1 独立评分

| ID | 分数 | verdict |
|---|---|---|
| `R0001-P26` | `4/3/4/3/4/4/3` | `changes_required` |
| `R0001-P27` | `4/3/4/2/4/2/2` | `changes_required` |
| `R0001-P28` | `5/3/3/4/4/4/2` | `changes_required` |
| `R0001-P29` | `4/5/5/5/3/5/5` | `approve, audit only` |
| `R0001-P30` | `4/2/5/5/4/4/5` | `approve, diagnostic only` |

### S1 反驳

- P26：
  - 同 Episode derangement 可能暴露时序或策略相关性；
  - 正 margin 只说明匹配 action 更符合 posterior，不证明模型学到可控物理方向；
  - “已有 margin 才训练 margin loss”会反向筛掉真正缺失 margin 的情况。
  - 修订要求：负动作按预注册距离分层，排除时间位置指纹；证明新增梯度与原 KL
    非完全共线，禁止扫描 shift、权重和负例。
- P27：
  - successor posterior 已看到结果 observation，inverse dynamics 可读取执行后的本体
    瞬态，却未必改善 action-conditioned prior；
  - 新 paired 数据、辅助头和 loss 容易成为多变量捆绑；
  - 必须证明信息进入 transition/prior，并封死 branch order、初态残差和生成器泄露。
- P28：
  - 现有 head 只在 posterior feature 上训练，冻结后直接喂 prior feature 是 OOD；
    失败不能否定候选；
  - 比值方向写反：若 true error 更低，应使用
    `negative_error / true_error >= 1.05`；
  - 替换全部 reconstruction 输入可能同时损害不可控视觉表征。
  - 修订后是 S1 的首选训练机制候选。
- P29：
  - latency=3 是预注册未见动力学，不能用 latest bundle 删除难度；
  - 只批准合同分类，若发现 bug 必须单独重建评测基线。
- P30：
  - loader 与 shape 合同已有索引正确的静态证据；
  - action latency 不等于 transition action 错位；
  - offset 只能由 provenance 给出，禁止按预测误差选择。

S1 的依赖顺序：

`P30 -> P29 -> 修订后的 P28 -> P26 -> P27`

## S2 独立评分

| ID | 分数 | verdict |
|---|---|---|
| `R0001-P26` | `4/3/4/4/3/4/3` | `changes_required` |
| `R0001-P27` | `3/2/3/2/3/2/2` | `reject` |
| `R0001-P28` | `4/3/2/4/4/4/3` | `changes_required` |
| `R0001-P29` | `4/4/5/5/4/5/5` | `approve, audit only` |
| `R0001-P30` | `5/2/5/5/5/5/5` | `approve, diagnostic only` |

### S2 反驳

- P26：
  - 固定 derangement 可能留下时距、动作幅值或位置指纹；
  - 若训练与验收共享负例族，等价于把 audit 形状写入目标；
  - 修订要求：训练与验证负例生成器隔离，且位置/幅值/时间 probe 不得识别负例来源。
- P27：
  - 当前判定逻辑不支持结论：如果冻结 posterior 已能高准确解码，所称表征缺口不成立；
    如果不能解码，也不能推出辅助 loss 可学；
  - 数据生成器和辅助 loss 是两个变化，容易泄露分支顺序、初态残差和安全 rewrite；
  - 缺少到 prior 和闭环控制的直接路由。
- P28：
  - 同样发现 error ratio 方向写反；
  - 冻结 OOD head 的失败不能作硬拒绝门；
  - 训练数据、负例族与正式 audit 必须隔离；
  - 最低成本门应是固定预算 head-only probe，并按 Episode/seed 隔离验证。
- P29：
  - 只批准时间线审计；真正 stale observation 仍必须被安全层拒绝；
  - 影子 latest bundle 不得进入正式 action chain。
- P30：
  - 唯一真值必须来自 timestamp/FIFO provenance；
  - provenance 缺失即 `inconclusive`，不得相关性猜 offset 或重标。

S2 的依赖顺序：

`P30 -> P29 -> P26 -> 修订后的 P28`，P27 不进入本轮训练。

## 主 Agent 汇总决策

### `R0001-P30`：筛选后历史去重，拒绝

- 两位筛选者都批准其作为只读数据合同诊断。
- 主 Agent 随后复查 R0002，发现其与 `R0001-P09` 的显式 observation-lag / actual
  plant-action 对齐实验实质重复。
- P09 已使用 96 个新 Episode、完整前置 action、lag 0/1、rho 0.96/0.50 和四个 horizon
  做正式对照，并否定“单一对齐跨 horizon 稳定改善”。
- P01 v4 数据源提交的 collector/writer/window/loader 与当前 HEAD 字节一致，静态血缘
  也支持 runtime applied action 与同次 outcome transition 成对存储。
- 因此不实施 P30，不重跑 P09，不搜索 offset，不重标 Replay。
- 结论：`rejected as duplicate of R0001-P09`。

### `R0001-P29`：改为第一候选

- 只作为 runtime timestamp contract 审计，不是能力改进。
- 不能移除 evaluation observation latency、延长 validity window 或让 latest bundle
  替代正式输入。
- P29 与 P09 不重复：P09 检验 visible-state/action 统计对齐，P29 检验 action validity
  对有意延迟 observation 的时间语义是否自洽。
- 先冻结纯合成 runtime 时间线和真实 P11 artifact 的只读审计；不修改 action chain。

### `R0001-P28`：修订后保留

- 纠正主要比值为 `negative_error / true_error`。
- 删除“冻结 posterior-trained head 失败即可否定训练候选”的 OOD 硬门。
- 后续最低成本门改为固定预算、Episode/seed 隔离的 head-only prior-feature 可学习性
  试验；visual/proprio 分头，不得相互补足。
- P30/P29 未完成前不冻结 P28。

### `R0001-P26`：保留但排在 P28 后

- 必须使用两个互斥负例族，并证明负例来源不能由位置、时距或动作幅值识别。
- 必须报告与原 KL 的梯度非冗余性。
- 不允许使用正式 action-shuffle audit 的 derangement family 训练或选择权重。

### `R0001-P27`：本轮拒绝

- 诊断门与“表征缺口”结论方向不一致。
- 新配对数据、辅助头和 loss 难以形成单变量因果对比。
- successor posterior 的结果可见性使 probe 容易成为 inverse-dynamics 捷径，且到 prior
  与闭环控制的路由最弱。
- 保留提案与反例，但不采集数据、不实现、不训练。

## P28 修订合同独立复审

P29 完成后，主 Agent 把 P28 修订为三折 source-level、fresh-head、true-action-only
训练草案。两个新的干净只读 Agent 独立复审，均未看到对方意见，均给出
`changes_required`。

### S1-R

阻断项：

1. 只有 split/fold hash，没有逐 source/window 可读清单、规范序列化和完整 batch
   schedule，无法唯一重建执行。
2. 未强制 visual student/world model `eval()` 与 inference path，posterior sampling
   可能使两臂差异不再只有 feature source。
3. successor posterior 已读取 target observation，只能作 head/data/预算上界；
   `prior/posterior <=1.25` 不能作公平硬门。
4. reverse 与 slot rotate 仍可能由时间位置、时距、动作幅值、相关性和 reservoir stage
   指纹区分。
5. 37-D proprio 的可控 16 维映射、单位和 innovation 幅值门未冻结；visual 缺少
   current-state/action-blind innovation 守护。
6. seed 内聚合和 bootstrap 必须明确 source 是唯一单位，并按任务/折保留依赖结构。

S1-R 结论：在补齐可重建 manifest、确定性 feature path、负例可识别性守护、精确 metric
和统计合同前不得执行。

### S2-R

阻断项：

1. successor posterior 的结果可见性使 oracle 对 prior 不公平；应删除 prior/oracle
   硬比较，或增加信息匹配 control。
2. `sample=False` 与正式训练 `sample=True` 分布不同；硬门必须匹配拟议训练采样路径并
   冻结 latent RNG。
3. 两个负例族仍是经验 action 时间重配，与正式 global derangement 同质；“算法不同”
   不足以证明无 audit 泄露。
4. 需要 head-independent label probe 证明时间、幅值、delta、时距不能识别负例来源，
   并至少加入一个支持内、非置换 counterfactual family。
5. constant baseline 与 visual cosine 的维度、epsilon、零 norm、reduction 和聚合顺序
   未完全定义。
6. `<30分钟` 的 MPS 时间、显存和 CPU/MPS parity 没有 smoke 证据。
7. 即使 head-only 通过，也只证明 frozen feature 可解码；没有冻结正式 world-model
   单变量 objective、梯度路径、对照和未见分布闭环合同。

S2-R 结论：P28 目前最多是诊断草案，不能授权 world-model 训练。

### 主 Agent 最终决策

- 两份复审均指出的是结构性因果与泄露风险，不是可在同轮内修补的实现细节。
- 继续修改负例、oracle、采样和 metric 会在看到多轮设计反馈后扩大变量空间，增加
  研究者自由度。
- 本轮停止 P28：不实现、不运行、不扫描预算或负例。
- 状态：`rejected before implementation after independent re-review`。
- P26 依赖同类负例可识别性问题，也不在本轮启动。
- 下一轮若继续目标函数方向，应先提出不依赖正式 audit 形状、具有信息匹配 control 的
  新稳定 ID 候选；不得直接复活当前 P28 草案。
