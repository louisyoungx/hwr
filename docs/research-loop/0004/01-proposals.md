# R0004 改进提案

## 生成过程

- 三个内置创新 Agent 及三个禁止工具的替补 Agent 均持续卡在运行状态，没有返回文本；
  主 Agent 关闭这些会话，没有伪造其意见。
- 随后使用本机 `traex exec --ephemeral --sandbox read-only` 启动三个干净独立会话，
  模型均为 GPT-5.6 Sol、极高推理强度，分别检查目标函数、时序合同和模型机制。
- 三个会话只读 `AGENTS.md` 与项目证据，没有修改文件或启动训练。
- 独立来源标签为 `A`、`B`、`C`；主 Agent 去重但保留分歧。

## 提案总表

| 稳定 ID | 名称 | 类型 | 独立来源 | 状态 |
|---|---|---|---|---|
| `R0001-P26` | 真实动作反事实 prior margin | 训练候选 | A | 延后 |
| `R0001-P27` | 配对因果逆动力学表征 | 诊断后训练候选 | A、C | 拒绝 |
| `R0001-P28` | prior-feature 物理 successor 解码 | 训练候选 | C | 复审拒绝，未实现 |
| `R0001-P29` | stale observation validity 合同 | 评测/运行时修复 | B | 接受为合同诊断 |
| `R0001-P30` | observation-action 执行时刻对齐 | 数据合同诊断 | B | 拒绝，重复 P09 |

## 提案要求

每项提案必须包含：

- 瓶颈证据；
- 可证伪的改进假设；
- 影响范围与唯一主变量；
- 最小低成本验证；
- 主要指标与守护指标；
- 失效条件；
- 成本、风险和依赖；
- 与 R0003 已拒绝假设的差异。

任何依赖评测泄露、结果后选样、降低门槛、同时修改多个不可归因变量，或只改善 loss
而无法路由到未见分布闭环物理结果的提案，均不得进入实验冻结。

## 共同证据与反例

- `WorldModelLoss` 的 visual/proprio reconstruction 直接解码包含当前 observation 的
  posterior feature；实际 action 只进入 RSSM prior transition，主要通过 prior/posterior
  KL 间接获得物理预测梯度。
- P17 已证明训练数据中的 actual plant action 对三任务、`1/4/8/16` 步可控状态具有稳定
  物理因果效应。
- 当前 checkpoint 的正式 one-step physical action-shuffle ratio 约为 `1.00128`。
- P21 的 72/72 shift 都能把 action effect 送入 transition activation，但没有系统性的
  GRU/prior 相邻层低 retention。
- P23/P24 表明 hard decoder feature 与 decoded output 的 action effect 在 Episode
  aggregate 上存活，argmax 与 decoder 系统低 gain 不是统一瓶颈。
- 更严格的 shift=1 output guard 失败 3 个 Episode。其 decoder feature standardized effect
  只有约 `0.0366/0.0242/0.0253`，至少两个在 decoder 前已经很弱。
- 冻结 24 窗口的探索性只读统计中，shift=1 归一化 action difference 与 proprio output
  effect 的 Pearson 相关约 `0.62`；三个失败 Episode 的 action difference 约为
  `0.340/0.275/0.220`。其中一个失败的动作差不弱，因此“干预幅值不足”也不是统一解释。

## `R0001-P26`：真实动作反事实 prior margin

### 证据与假设

- 来源 A 指出，当前目标只要求 true action prior 接近 posterior，没有显式要求错误 action
  prior 更差；绝对重建和 KL 可在 action-discriminative margin 很小时下降。
- 假设：在相同起始 posterior state 上，加入 true action prior 相对固定经验负动作的
  posterior-distance margin，可直接奖励 action-conditioned prior 的判别方向。

### 唯一主变量

- 新增一个 prior margin loss 及其预注册固定权重。
- 模型结构、原 KL/reconstruction、Replay、sampler、学习率、更新数和正式评测不变。

### 最小低成本验证

1. 只用冻结训练 Replay 与 checkpoint，不更新参数。
2. 负动作只从同 Episode 的经验 action 序列产生，保持边际分布；不得使用正式 holdout、
   P24 失败标签、任务结果或物理 target 选择负例。
3. 比较 true 与固定 derangement 的 posterior KL margin、对 transition/prior 参数的梯度
   norm、梯度 cosine，以及 24 Episode 和任务分层。
4. fixed derangement 的 shift/置换规则必须在计算指标前一次冻结，不能使用结果选择。

### 主要指标与守护

- true posterior KL 必须低于经验负动作，且 margin 在至少 `20/24` Episode、任务配额
  `5/6、5/6、10/12` 上为正。
- Episode-block bootstrap 的 mean margin 95% 下界必须 `>0`。
- margin 对 RSSM transition/prior 参数的梯度必须有限且 norm `>1e-6`。
- 负控：同一 action 的复制分支 margin 必须为 0；随机 target 标签不得通过。
- 不要求或查看正式 action-shuffle ratio，避免用最终评测直接设计 loss。

### 失效条件、成本与风险

- true/negative margin 不稳定、只在单任务成立、梯度近零、复制负控非零或血缘/hash 漂移，
  则诊断拒绝，不扫描负例、margin 或权重。
- 成本为冻结 checkpoint 的额外 prior 前向和 backward，不训练。
- 主要风险是高相关动作使经验负例过近，或模型只学习识别置换指纹。

### 与旧候选区别及路由

- 不同于 P06：不做多步 posterior overshooting，也不把正式 audit ratio作为训练排序目标。
- 不同于 P21：不假设特定 GRU/prior 层发生普遍衰减。
- 诊断通过后才另行冻结一个单权重、三 seed 的等预算训练候选；训练后仍需用训练前冻结的
  未见任务、物体、布局、语言、动力学和硬件分布做闭环物理评测，安全与计算守护不回归
  才能接受。

## `R0001-P27`：配对因果逆动力学表征

### 来源分歧、证据与假设

- 来源 A 建议从普通相邻 posterior feature 变化预测 actual action，成本低，但存在策略
  自相关与时序指纹捷径。
- 来源 C 建议只使用同初始物理状态的配对分支，让 feature difference 识别随机分配的
  actual action；因果更强，但需要新训练配对数据。
- 主 Agent 合并后只保留 C 的同状态配对版本，普通 Replay 逆动力学不进入正式候选。
- 假设：现有 posterior 没有稳定保留 actual action 所引起的可控状态差；同状态随机配对
  的逆动力学辨识可检验这一缺口，并可能作为独立辅助目标。

### 唯一主变量

- 诊断阶段只增加一个冻结表征上的线性 paired inverse-dynamics probe。
- 若诊断通过，训练候选只新增该配对辅助 loss；部署时移除辅助头。
- 不同时修改 prior、decoder、Replay sampler、stale-frame 或安全头。

### 最小低成本验证

1. 使用与正式能力评测 seed 互斥的新 P17 式同 snapshot `+d/-d` 训练诊断集。
2. probe 输入只含两分支的 successor posterior feature difference；标签是随机分配的
   actual action sign/direction。
3. 按 source seed 留出，报告三任务、四 horizon、全部零响应和安全改写样本。
4. 负控包括 branch label permutation、相同 action sham、交换输入顺序和去掉 successor
   difference；不得输入任务 ID、seed、时间索引或 proposal generator 状态。

### 主要指标与守护

- 主要指标为留 seed action-sign balanced accuracy 与连续 actual-action direction cosine。
- 三任务分别 balanced accuracy `>=0.65`，Holm 校正 permutation `p<=0.05`，且 cosine
  在每任务为正。
- sham family 不得高于预注册 5% FPR；任一任务或 horizon 配额失败即整体失败。
- probe 必须只用冻结 training diagnostic 数据；不读取最终闭环评测。

### 失效条件、成本与风险

- 只在单任务成立、sham 可解码、去掉 successor difference 后仍通过、跨 seed 不泛化，
  或实际梯度不能到达 RSSM，则拒绝。
- 诊断成本为新小规模配对物理采集和线性 probe；训练候选成本中等。
- 风险是动作生成器、分支顺序或初态残差泄露标签；必须逐项负控。

### 与旧候选区别及路由

- 不同于 P17：P17 证明 plant action 改变物理 state；P27 检验 learned posterior 是否编码
  该因果差异。
- 不同于 P06/P21/P23：不拟合未来 posterior、不定位层 retention、不依赖 argmax。
- 诊断通过后仍需先冻结 loss 权重和等预算训练，再走全 seed 未见分布闭环物理评测；
  probe accuracy 或训练 loss 不能作为能力结论。

## `R0001-P28`：prior-feature 物理 successor 解码

### 证据与假设

- 来源 C 指出，visual/proprio/reward/continue head 训练时看到当前 posterior observation，
  可以依赖 filtering shortcut；部署 imagination 却依赖 open-loop prior feature。
- 假设：让现有物理 head 在 `t>=1` 只从一步 prior feature 预测 successor target，会直接
  奖励 action-conditioned physical prediction，而不需要改变 RSSM 或 decoder 结构。

### 唯一主变量

- physical successor reconstruction 的输入从 posterior feature 改为一步 prior feature。
- head 架构、target、原 loss 权重、RSSM、Replay、优化预算和安全/action-execution 分支
  保持不变。

### 最小低成本验证

1. 冻结 checkpoint 与训练 Replay。
2. 从相同 posterior `(h_t,z_t)` 分别用 true actual action 与预注册经验负动作生成一步
   prior feature，经现有 visual/proprio head 预测真实 successor target。
3. 报告 true/negative error ratio、预测差分与实际 controllable-state innovation cosine、
   24 Episode、任务和 shift/负动作分层。
4. 先验证当前冻结 head 是否具备非零 prior-feature physical gradient；不更新权重。

### 主要指标与守护

- visual 与 proprio 分头判断，不得相互补足。
- true/negative error ratio 的 Episode-block bootstrap 95% 下界 `>=1.05`，至少
  `20/24` Episode，任务配额 `5/6、5/6、10/12`。
- proprio prediction difference 与可控 successor difference cosine 在三任务都 `>0`。
- posterior reconstruction、KL、action execution、collision 和安全路径不参与调参且不得
  因实现改变。

### 失效条件、成本与风险

- 冻结 head 对 prior feature 已没有 true advantage、梯度近零、只在单头/单任务成立，
  或 target 对齐不完整，则拒绝，不训练。
- 成本为冻结前向/backward；通过后训练成本与基线近似。
- 风险是一步 prior 噪声使 reconstruction 退化，或把视觉不可控变化错误归因给 action。

### 与旧候选区别及路由

- 不同于 P06：直接监督可观测 physical successor，不匹配 posterior latent 多步 rollout。
- 不同于 P24：不定位 decoder 增益，而是改变训练时 decoder 所见的时序状态。
- 诊断通过后冻结单变量训练，最终仍以未见分布闭环成功、安全、稳定性、数据与计算效率
  判定，不以 error ratio 或 reconstruction loss 判定能力。

## `R0001-P29`：stale observation validity 合同

### 证据与假设

- P11 正式集有 36 Episode 的 evaluation `observation_latency=3`，动作 frame 使用约
  150ms 旧 observation timestamp，超过 100ms validity window，step 3 后连续被安全层
  `outside_validity_window` 拒绝。
- 来源 B 假设这是“错误选择旧帧”的评测/运行时合同问题，并建议使用推理截止前最新完整
  多模态 observation bundle。
- 主 Agent 反例：evaluation latency 1–3 是配置中明确的未见动力学条件。若直接取最新帧，
  将删除预注册域随机化而不是修复实现。

### 唯一主变量

- 只审计 observation capture timestamp、delivery timestamp、policy inference timestamp、
  action valid-from/until 和 applied timestamp 的语义。
- 在审计证明契约错误前，不修改 validity window、安全层、模型、动作或评测随机化。

### 最小低成本验证

1. 不执行物理任务，构造 latency `0/1/2/3` 的确定性 runtime contract 测试。
2. 验证“传感器旧数据”与“评测错误回退旧 bundle”两种情况能否被时间线区分。
3. latest-complete bundle 只能作为影子分支；不得替代正式 evaluation 输入。
4. 检查生产语义：action validity 应绑定观测采集时刻、推理完成时刻还是控制计划时刻，
   并与独立安全层威胁模型一致。

### 主要指标与守护

- 主要结果是唯一可复现的时间线分类：真实 sensor latency 或错误 bundle selection。
- 100% 合成 case 分类正确；无 future frame、跨模态时间倒流或 action validity 延长。
- 原 safety 层对真正 stale observation 必须继续拒绝。

### 失效条件、成本与风险

- 若当前实现与配置语义一致，P29 作为修复候选拒绝；observation latency=3 暴露的是系统
  能力边界，不能通过移除域随机化“修复”。
- 若证明 bundle selection 与配置不符，才另立纯评测修复；修复结果不能与能力候选做同一
  因果对比。
- 成本低；风险是把安全失败误分类为评测 bug。

### 与 P11 区别及闭环路由

- P11 估计 plant action FIFO；P29 只澄清 observation/action timestamp contract。
- 任何修复后必须重新建立不含能力改动的评测基线，再在保留 latency=3 难度和原安全阈值
  的未见分布上运行闭环物理评测。

## `R0001-P30`：observation-action 执行时刻对齐

### 证据与假设

- 来源 B 怀疑 Replay 按同索引 observation/action 配对，而 deployed action 在未来执行，
  建议按实际执行时刻重建监督标签。
- 主 Agent 反例：`FoundationSequenceBatchLoader` 明确将 `executed_action[t]` 作为
  observation `t -> t+1` 的 transition action；RSSM shape 合同也是 actions 数量比
  observations 少 1。当前没有索引错位的正证据。
- 假设：NPZ 中实际 timestamp、action latency FIFO 与数组索引可能仍不满足这一语义，
  尤其在 nonzero latency Episode。

### 唯一主变量

- 只诊断 Replay transition identity；不重新标注、不训练。
- 比较 index contract 与 timestamp/FIFO reconstructed plant-action contract。

### 最小低成本验证

1. 对全部训练 source Episode 读取 observation/frame timestamp、proposal、executed action、
   runtime randomization provenance 和可控 proprio successor。
2. 对 latency `0/1` 分层，验证 `executed_action[t]` 是否确为 `state_t -> state_t+1` 的
   actual plant action；使用 runtime 日志/因果 plant 规则，不以预测 loss 选择 offset。
3. 固定 offset `-3..3` 只作合同单元测试；正确 offset 必须由 provenance 预先给出，
   不能按最小误差后选。
4. 报告缺失 timestamp、FIFO warmup、安全 rewrite 与 terminal step，不删除样本。

### 主要指标与守护

- 正确 index/timestamp 匹配率必须为 100%；array hash、transition count、task/seed
  lineage 全部保持。
- 同动作重放与 P17-style first-stage direction 必须和记录的 actual action 一致。
- 不读取 causality holdout 或最终闭环结果。

### 失效条件、成本与风险

- 若 index contract 已正确，P30 拒绝，不做重标注训练。
- 若 provenance 不足以唯一确定执行动作，结论为 `inconclusive`，不得用相关性猜 offset。
- 成本低；风险是把 action latency 与 transition action identity 混淆，或后选 offset
  制造 action-shuffle 改善。

### 与 P11 区别及路由

- P11 从 proposal history 估计 plant transformation；P30 只验证存储的 actual action 是否
  连接正确 observation pair。
- 只有确认系统性错位后才单独建立数据修复基线；之后的能力训练必须重新冻结 Replay 与
  未见分布闭环评测，不能把数据修复本身记为能力改善。

### 历史去重结论

主 Agent 在筛选后复查 R0002，确认 P30 与 `R0001-P09` 实质重复：

- P09 的问题正是“后端返回当前物理 step 的 actual plant action，但可见 observation
  可能来自更早物理状态；旧 probe 默认 action[t] 连接相邻可见 observation”；
- P09 使用显式 observation lag、完整前置 actual action、新 seed、96 个 128-transition
  Episode，对 old-index 与 lag-aligned index 做了正式对照；
- P09 结果为 `rejected`：lag=1 的每个 horizon state-action MSE 没有稳定改善 10%，
  rho=0.50 的 8/16-step 在三任务反而恶化约 20%～34%；
- R0002 已冻结结论：“observation lag 必须显式记录，但单一 lag 对齐不是跨 horizon
  失败的充分解释”。

此外，P01 v4 数据源提交 `d6d9a43` 的 collector、trajectory writer、window slicer 和
batch loader 与当前 HEAD 字节一致：

- collector 在 `backend.apply(frame)` 后直接读取 runtime `info["applied_action"]`；
- 同一循环把该 applied action 与 `outcome.observation` 追加成一条 transition；
- writer 固定 observations 数量为 actions+1；
- window 按 observation `[start:stop+1]` 与 transition `[start:stop]` 成对切片。

因此 P30 不再实施、不重跑 P09、不搜索 offset、不重标 Replay，正式标记
`rejected as duplicate of R0001-P09`。
