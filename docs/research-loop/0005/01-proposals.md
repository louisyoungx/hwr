# R0005 提案

## 生成过程

- 三个创新 Agent 独立只读检查 `AGENTS.md`、R0001～R0004 证据、当前代码和产物。
- 三者未修改文件、未启动正式训练，也未在提交前看到其他创新 Agent 的提案。
- Agent A 聚焦不依赖 action shuffle/derangement 的信息匹配诊断。
- Agent B 假设 latency=3 必须成为支持域，聚焦安全的 latency-aware scheduling。
- Agent C 假设 latency=3 当前不可安全执行，聚焦不删除挑战域的能力声明和更接近任务成功的
  下一变量。
- 主 Agent 只做稳定编号、重复项核查和文字归并，保留 B/C 的互斥立场。

## 提案总表

| ID | 名称 | 类型 | 来源 | 状态 |
|---|---|---|---|---|
| `R0001-P31` | 同状态配对前向响应三臂审计 | 目标诊断 | A | `changes_required` |
| `R0001-P32` | 正交化动作创新条件信息探针 | 数据诊断 | A | `changes_required` |
| `R0001-P33` | 同状态配对物理响应辅助损失 | 训练候选 | A | `deferred` |
| `R0001-P34` | 双时钟延迟域动作租约 | runtime 合同变更 | B | `rejected` |
| `R0001-P35` | 固定四步 action-chunk 延迟槽调度 | 能力训练/调度候选 | B | `deferred` |
| `R0001-P36` | 时效支持域与完整挑战域双账本 | 评测合同 | C | E1 已冻结 |
| `R0001-P37` | 任务无序的前接触进展势能 | 能力训练候选 | C | `rejected` |
| `R0001-P38` | P17 与 Actor 准入证据角色拆分 | 准入合同诊断 | C | `changes_required` |

## 共同证据与不可违反边界

- P01 v4 为完整负基线：24 Episode、1,600 update、0 成功，世界模型 action causality
  aggregate ratio `1.0012791539504238`，action execution 三任务 recall 均为 0。
- P17 已证明真实 plant action 有稳定物理效应，但不证明 production world model 已学习该
  效应。
- P28 已因 successor-posterior oracle、负例指纹、采样不一致和自由度过高被拒绝；本轮
  不复活其 action 时间重配负例。
- P29 已证明 latency=3 是有意延迟的真实旧观测流，不是 latest-bundle 选择 bug。
- 当前 `dual_arm_action_frame()` 把动作租约绑定 visible observation timestamp：
  `valid_until = observation timestamp + 100ms`；safety 在当前 runtime 时钟执行。
- 正式后端 observation queue 保留原始 timestamp，latency=3 稳态 age 为 150ms。
- 任何候选均不得延长 100ms 动作租约、读取 future/latest observation、删除 latency=3、
  降低碰撞/安全门槛，或把修复与能力训练放在同一首次对比。
- action shuffle、derangement、reverse、slot rotate 或同质时间重配不得作为训练标签或
  候选选择依据。

## `R0001-P31`：同状态配对前向响应三臂审计

### 证据与假设

- 当前 visual/proprio reconstruction head 主要从已看到结果 observation 的 posterior
  解码；actual action 通过 prior/posterior 路径间接获得物理梯度。
- P17 证明 plant 可控，P24 证明 action-conditioned feature effect 到达 decoder output，
  但尚未证明该方向与真实 plant response 对齐。
- 假设：训练分布中的真实动作—物理响应可被公平 fresh probe 学到，而 production prior
  没有稳定学到；若成立，才说明当前 objective 没有充分奖励正确动作方向。

### 唯一主变量与最小验证

- 只新增只读 evaluator，不修改 production 模型。
- 在训练专用、与正式闭环 seed 隔离的 P17-style 同 snapshot `+d/-d` 和 same-action sham
  数据上建立三臂：
  1. 冻结 production prior；
  2. fresh action-conditioned forward probe；
  3. 等容量 fresh action-blind control。
- 两个 fresh head 共用 pre-action state、successor target、source split、初始化、参数量、
  batch schedule 和预算；control action slot 只含训练折内由 pre-action state 预测的条件
  均值，唯一差异为是否获得 actual action innovation。
- 输入截止在动作执行前，禁止 successor posterior、reward、task ID、seed、时间索引、
  正式 holdout 和闭环成功标签。
- 预测 `1/4/8/16` 步可控 16-D proprio response，并比较 plant、production prior 和 fresh
  probe 的 response matrix。

### 指标、守护与失效

- 主要：
  - action-conditioned probe 相对 blind control 的 source-held-out normalized MSE 至少
    改善 5%，Episode bootstrap 95% 下界 `>0`；
  - response matrix 与 plant 的 Frobenius cosine 在三任务和四 horizon 全为正，并通过
    Holm 校正；
  - 只有 probe 通过而 production prior 不通过，才支持 objective-routing 缺口。
- 守护：
  - same-action sham 不得产生响应或显著梯度；
  - blind injection family-wise FPR `<=0.05`，功效 `>=0.80`；
  - conditioned 绝对误差必须改善，不能只放大 branch difference。
- 任一任务/horizon 不成立、sham 通过、source 泄露、probe 不优于 control，或 production
  prior 已与 plant 对齐，则拒绝对应缺口假设。
- 成本为小规模配对采集、冻结前向和小头训练；结论只允许作为 objective-routing evidence。

### 去重与路由

- 不生成错误动作，不同于 P26/P28 的负例 margin。
- 不从结果可见 posterior 反推动作，不同于 P27。
- 不拟合 posterior overshooting，不同于 P06。
- 若通过，只允许冻结 P33 的独立训练合同；不能直接宣称能力改善。

## `R0001-P32`：正交化动作创新条件信息探针

### 证据与假设

- 普通 Replay 动作高度相关，state history 可预测大量当前动作；P16 已证明普通
  state-only/state-action ridge 在该设计下功效不足。
- 假设：actual action 中不能由 pre-action state 预测的 residual 仍含 successor physical
  information；该问题可不用错误动作或新物理采集回答。

### 唯一主变量与最小验证

- 冻结 checkpoint、训练 Replay 和 features，只训练两个等容量 forward head。
- 三折 source-Episode cross-fit；同一 source 不跨折。
- 先由训练折 pre-action posterior 预测 `â_t = E[a_t | state_t]`。
- control 输入 `(state_t, â_t)`；candidate 输入 `(state_t, actual_action_t)`；唯一差异为
  residual `a_t-â_t`。
- 使用全部合法连续 transition，target 固定为下一步可控 16-D proprio innovation。
- 两臂共享初始化、优化器、更新数、batch schedule、标准化、target 和统计单位。

### 指标、守护与失效

- held-out `control_MSE / candidate_MSE >= 1.05`；
- Episode-block mean log-ratio bootstrap 95% 下界 `>0`，三任务分别为正；
- candidate 绝对误差优于训练折 target-mean baseline；
- action residual 有效秩 `>=6`，否则标记数据功效不足；
- 条件均值模型不得读取 successor、reward、task ID、source identity 或正式 holdout；
- planted residual 功效 `>=0.80`，零 residual 和随机 target FPR `<=0.05`。
- residual 低秩、candidate 不优于 control、单任务收益或绝对误差恶化即拒绝。
- 该项只证明普通训练 Replay 是否含条件动作信息，不能证明 production RSSM 已利用。

### 去重与路由

- 不使用 shuffle/derangement，不重复 P16 的普通高共线性 Probe。
- 若通过，只支持“训练 Replay 有可用信息”；仍需 P31 或等价 production-path 证据才能
  路由训练目标。

## `R0001-P33`：同状态配对物理响应辅助损失

### 依赖与假设

- 仅当 P31 形成“fresh probe 可学、production prior 不会”的预注册缺口后才可重新冻结。
- 假设：对训练专用同 snapshot 真实动作分支增加 physical response-difference loss，可
  迫使 prior 学习正确动作方向，而无需构造错误动作。

### 唯一主变量

- 基线与候选使用同一普通 Replay、配对数据、更新数和 batch schedule。
- 唯一变量为固定 `lambda_pair`：基线 0，候选为结果前按量纲而非结果扫描确定的非零值。
- loss：
  `Huber((prediction_plus - prediction_minus) - stopgrad(target_plus - target_minus))`。
- horizon 固定为 `1/4/8/16`，原 loss、模型、Actor、安全和采样保持不变。

### 最小验证、指标与守护

- 先验证 loss 有限，梯度到达 transition input、recurrent、prior 和 proprio head；
  same-action sham loss 接近 0；两次 update smoke 只改变新增 loss。
- 三个训练 seed 均需在独立 paired holdout 上改善 response-matrix alignment。
- 最终主要指标仍为冻结未见分布闭环成功率：candidate-baseline Episode bootstrap 95%
  下界 `>0`，三任务均不退化。
- true-action open-loop error、普通 reconstruction、KL、action execution、安全干预、
  严重碰撞和提前终止不得回归；内存与墙钟增幅预注册上限为 20%。
- P31 未通过、梯度不到 prior、只改善 paired proxy 或任一安全守护回归即拒绝。
- 成本高：训练专用 paired bank 加三 seed 等预算正式训练。

### 泄露边界

- 只用训练专用随机方向和真实 plant branch outcome；
- 不读取错误动作、shuffle ratio、正式 causality holdout、成功或安全标签。

## `R0001-P34`：双时钟延迟域动作租约

### 类型、证据与假设

- 类型：runtime 合同变更，不是评测修复或能力改善。
- 当前 action frame 没有独立 source-observation provenance，单一时间判断无法区分：
  - 刚调度、但依据预先支持的延迟观测生成的命令；
  - 命令本身已经过期、冻结、回放或乱序。
- 假设：把命令租约新鲜度和 source observation 支持域拆成两个独立安全门，可保持
  100ms 命令租约，同时让可信、持续前进的 latency=3 流可执行，并拒绝真正 stale 输入。

### 唯一主变量

- action `created_at/valid_from/valid_until` 改为 scheduler 生成与执行时间，租约仍不超过
  100ms。
- 另携带不可变的 source observation timestamp、source sequence、target control tick 和
  可信 capture/delivery provenance。
- safety 同时检查：
  - 命令租约未过期；
  - source 不在未来；
  - sequence 单调前进且无 replay/乱序；
  - source age 在结果前冻结的 0～3 步支持域；
  - target tick 与实际执行 tick 一致；
  - 推理或调度超时仍拒绝；
  - 原 clamp、预测碰撞、双臂停止和 gripper hold 保持。
- latency queue 未填满时只能发 hold，禁止重复初始帧驱动运动。
- 不修改 policy action value、模型、Actor、评测 latency 或碰撞阈值。

### 最小验证、指标与守护

- 先执行纯 runtime 合同测试，不训练：
  - latency 0/1/2/3 的确定性 20Hz 时间线；
  - latency=3 前三步 hold，稳态 61 个调度动作不得因旧 observation timestamp 被拒绝；
  - adversarial latency=4、age>150ms、timestamp/sequence 停滞、replay、乱序、跳号、
    future timestamp、target tick 错位、命令 lease 超时、跨任务和跨模态不一致。
- 主要：
  - latency=3 稳态 validity rejection `0/61`；
  - 所有 adversarial stale case rejection recall `100%`；
  - 支持域正常输入误拒绝率 0；
  - action lease 最大长度仍为 100ms。
- latency0/1/2 proposal vector 与旧路径一致；warm-up 只输出 hold；严重碰撞为 0；
  predictive safety、clamp 和 gripper hold 零回归。
- 任一 replay/停滞/乱序/超预算输入被接受，需读取未公开 evaluation latency 标签，需延长
  lease，或 latency0～2 回归，均拒绝。

### 风险与去重

- 若没有可信 capture/delivery provenance，“持续前进且固定落后 3 步的故障流”与正式
  latency=3 在可见信息上不可区分；此时只能声明最大支持延迟，不能宣称识别所有 stale。
- 不重复 P29 诊断、P11 plant FIFO 或 P09/P30 Replay 对齐。

## `R0001-P35`：固定四步 action-chunk 延迟槽调度

### 依赖与假设

- 只有 P34 独立成为 runtime 基线后才可重新冻结。
- 当前已有 `ActionChunk` 合同，但正式 foundation policy horizon 为 1，执行链只取 slot0。
- 假设：同一固定 horizon=4 policy 输出中，按可信 source-to-target offset 选择 slot
  0/1/2/3，可在不读 future/latest observation 的前提下改善 latency=3 闭环能力。

### 唯一主变量

- 同一冻结 horizon=4 checkpoint 上：
  - control 始终执行 slot0；
  - candidate 执行 `slot[source_to_target_steps]`。
- checkpoint、任务、seed、P34 runtime、安全和预算相同；唯一差异为调度槽位。
- horizon=4 由最大支持 latency=3 推导，不扫描；其训练是共同前置条件，不计入槽位收益。

### 验证、指标与风险

- 先冻结 chunk label 语义和 `valid_steps`；offset 只由可信 timestamp/control tick 计算，
  不输入 policy；slot 不存在时 hold/reject，不回退 slot0。
- 同 checkpoint 做短闭环 smoke，通过后运行全部未见任务、物体、布局、语言、动力学和
  latency seed。
- latency=3 paired success 绝对提升至少 10 个百分点，Episode-block 95% CI 下界 `>0`。
- latency0 回归不超过 5 个百分点；latency1/2 无系统回归；严重碰撞为 0；安全负担不增；
  p99 推理与调度时间不超过 50ms；P34 adversarial suite 保持 100% 通过。
- label/slot 语义不能唯一重建、收益只来自单任务/单 seed/少运动、或安全实时性回归则拒绝。
- 风险是共同 horizon=4 checkpoint 本身需要训练，必须防止其与 slot selection 构成不可
  归因捆绑。

## `R0001-P36`：时效支持域与完整挑战域双账本

### 类型、证据与假设

- 类型：评测合同，不改变能力。
- 当前评测没有显式区分“已支持、任务失败、安全不可执行域”，而 latency=3 的合同拒绝会
  混入策略失败。
- 假设：结果前按时间合同冻结支持域，并同时保留完整挑战域总账，可以诚实声明当前能力
  边界而不删除任何难度。

### 唯一主变量与最小验证

- 只改 benchmark manifest、report schema 和聚合逻辑；不改 policy、模型、观察、动作、
  safety、任务成功条件或 evaluation randomization。
- `S_supported`：visible observation age `<=100ms`，当前 latency 1/2。
- `U_challenge`：age `>100ms`，当前 latency 3。
- 结果前生成平衡 factorial manifest：task × observation latency 1/2/3 × action latency
  1/2/3；其他 evaluation randomization 仍使用原宽范围。
- domain label 只供 evaluator 聚合，禁止输入 policy，不能由成功、干预或模型输出反推。
- 用 P11/P29 artifact 和合成时间线验证：latency1/2 与 latency3 必须 100% 分入唯一域，
  Episode 不得遗漏、重复或结果后改域。

### 指标、守护与失效

- 支持域报告每任务成功率、Wilson 下界、P13 safety burden 和双臂消融。
- 完整域报告所有支持/挑战 Episode 的任务完成率；挑战域失败仍进入完整分母。
- 同时报告支持 strata 数和占完整 benchmark 的比例。
- latency=3 必须全量展示；过期动作实际应用率 0；严重碰撞 0。
- 固定声明文本：
  “支持 visible observation age <=100ms 的 evaluation 子域；完整 evaluation profile
  尚未支持。”
- 若域划分依赖结果、latency3 被删除、完整分母缩小，或条件成功被表述为全域能力，则
  拒绝。
- 成本低，最大风险是选择性展示支持域成绩。

### 与 P34 的分歧

- P36 接受 latency3 当前不支持，保持安全拒绝并显式记入完整挑战账本。
- P34 尝试建立可信延迟流的双时钟支持合同。
- 两者是互斥 runtime/评测路线，首次实施不得同时进行。

## `R0001-P37`：任务无序的前接触进展势能

### 依赖、证据与假设

- 当前 task potential 主要依赖物体到目标距离和部分 articulation；在导航、接近和首次
  抓取前，物体不动，任务 reward 近似平坦。
- 假设：加入排列不变、左右臂对称的末端到未完成物体最近距离 potential，可提供接近阶段
  任务信号，比 latent proxy 更直接路由到闭环成功。
- 必须在可信 runtime/评测基线与 causality-qualified 训练父提交后启动；不能直接把 P01
  v4 当合格 deployment。

### 唯一主变量

- 只改变训练期 potential：
  `Phi_new = Phi_old + lambda * Phi_precontact`。
- `Phi_precontact` 对所有未完成物体聚合
  `min(left_eef_distance, right_eef_distance)`，不指定对象顺序、左右臂或脚本动作。
- lambda 按几何量纲在结果前冻结，不扫描。
- model、Replay、sampler、policy input、safety、termination 和正式 success 保持不变。

### 最小验证、指标与守护

- 先复用 P17 同状态分支和 P01 Replay，不训练：
  same-state sham potential 一致；三任务接触前存在可判别正负进展；observation-only 扰动
  不改变 potential；严重碰撞不能获得正净收益。
- 通过后才做同父提交三 seed 等 Episode/updates 对照。
- 最终主要指标仍为支持域每任务闭环成功率和 Wilson 下界；早期只报告首次安全接触率、
  受控物体运动 Episode 比例和首次正任务进展时间。
- 完整域成功率同步报告；latency3 不得偷偷转入支持域；P13 safety、零严重碰撞、双臂
  消融、数据和计算预算不回归。
- 只改善 reward/contact、不改善闭环成功，单任务收益，碰撞或固定对象/手臂捷径即拒绝。
- 风险是 privileged geometry reward hacking；需审查其是否构成任务专属捷径。

## `R0001-P38`：P17 与 Actor 准入证据角色拆分

### 证据与假设

- R0003 已确认旧 Action Probe 功效不足，并接受 P17 为 training-data plant causality
  evidence。
- 当前 actor readiness 仍保留 data Action Probe 硬门，可能继续把低功效测量失败解释为
  “数据没有物理因果性”。
- 假设：拆分 data/plant causality 与 model action utilization 的证据角色，可修复错误
  归因，但不自动解锁 Actor 或 deployment。

### 唯一主变量与最小验证

- 只修改准入报告和证据角色：
  - 与当前 lineage 绑定的 P17-style paired physical gate 只证明 data/plant causality；
  - world-model action utilization、action execution、collision、interaction coverage 等
    继续作为独立硬门。
- 先对 P01 v4 做 shadow admission，不启动训练。
- 预期只移除“Probe 失败等于无 plant causality”的错误解释；P01 v4 仍因 model/action
  execution 失败而不能产生 qualified deployment。

### 指标、守护与失效

- 准入报告必须分别显示 data causality、model utilization 和 deployment readiness。
- 任一模型、安全或交互门不得删除，P17 不得自动解锁 task Actor/deployment。
- 若 P17 provenance 不能绑定当前 plant/config，结论为 `inconclusive`；若 shadow
  admission 意外解锁 Actor/deployment，立即拒绝。
- 成本低；该项是平台归因修复，不是能力改善。

## 历史去重结论

- P31/P32/P33 不使用 action-shuffle/derangement 训练，也不复活 P26/P28。
- P31 不读取 successor posterior 反推动作，不重复 P27。
- P34/P35 不重标 Replay，不重复 P09/P30；P34 不估计 plant action FIFO，不重复 P11。
- P36 不删除 latency3，也不重复 P13；P13 只定义 safety burden，未定义支持域双账本。
- P37 与 P25b 不同：P25b 检查 reconstruction gradient 对 shifted decoder direction 的
  投影；P37 改变训练期物理任务 progress potential。
- P38 不替换 model/deployment 门，只修正 P16 低功效结论后的 evidence-role 漂移。

## 禁止项

- 直接复活 `R0001-P28` 的 successor-posterior oracle 或经验 action 时间重配负例；
- 用 action shuffle、derangement 或与正式 audit 同质的负例形状作为训练监督；
- 通过延长 100ms validity、删除 latency=3、读取 future/latest observation 或降低安全门槛
  制造可执行性；
- 把 runtime/评测修复与世界模型、Actor 或安全模型训练放进同一因果对比；
- 只优化 loss、probe、训练回报或主观视频，而没有冻结闭环物理判定。
