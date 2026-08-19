# R0003 改进提案

## 提案总表

| 稳定 ID | 名称 | 类型 | 状态 |
|---|---|---|---|
| `R0001-P14` | 等 transition 预算的连续 Probe 证据合同 | 测量修复 | 阻断 |
| `R0001-P15` | 结果盲的 Replay 起点选择 | 数据保留修复 | 阻断 |
| `R0001-P16` | Action Probe 设计功效门 | 测量修复 | 拒绝 |
| `R0001-P17` | 同状态配对实际动作干预 | 物理因果诊断 | 接受 |
| `R0001-P11` | 因果 latent proposal-history gate | 训练候选 | 拒绝 |
| `R0001-P05` | 跨 source batch 三臂归因 | 训练候选 | 拒绝 |
| `R0001-P06` | 真实动作多步 posterior overshooting | 训练候选 | 预检拒绝 |
| `R0001-P19` | RSSM free-nats 梯度死区诊断 | 训练候选 | 诊断拒绝 |
| `R0001-P20` | RSSM action 输入量级诊断 | 训练候选 | 诊断拒绝 |
| `R0001-P21` | RSSM 逐级 action-effect 衰减定位 | 训练候选 | 诊断拒绝 |
| `R0001-P22` | Posterior observation/deterministic 支配诊断 | 训练候选 | 延后，依赖 P21 |
| `R0001-P23` | Prior probability 到 argmax code 离散化诊断 | 训练候选 | 诊断拒绝 |
| `R0001-P24` | Visual/proprio decoder 逐层 gain 诊断 | 训练候选 | 诊断入选 |
| `R0001-P25` | Physical target scale/gradient 奖励诊断 | 训练候选 | 延后，依赖 P24 |
| `R0001-P10` | 安全正例窗口分层采样 | 训练候选 | 延后 |

## `R0001-P14`：等预算连续 Probe

- 每 Episode 固定相同 112 transition。
- 三臂：
  - 7×16 硬切；
  - 连续 112 但固定相同 7 个起点；
  - 连续 112 全部合法起点。
- 只改变测量视图，不增加 Episode、原始 transition 或动作。
- bootstrap 单位保持 source Episode。
- 只有新 seed 上连续全起点全任务/horizon 过线，且绝对 state-action MSE 非劣，才能接受。

## `R0001-P15`：结果盲起点

- physical-salience selector 使用未来运动、动作创新、安全干预和交互结果选窗。
- 仅当 P01 后 h1 仍失败时，对比 salience 与预注册均匀起点。
- 接触、安全、碰撞和交互覆盖不得下降。

## `R0001-P16`：设计功效门

- 使用 P09 固定 state/action 设计矩阵生成 null、planted 和 permutation target。
- 三个设计臂、两个 rho、三个任务、四个 horizon。
- 每种效应 500 trial；每 trial 200 次同步 Episode bootstrap。
- null FPR `<=5%`、permutation FPR `<=5%`、planted power `>=80%`。
- 功效不足只能标记 `inconclusive`，不能把失败改为通过。

## `R0001-P11`：因果 latent proposal-history gate

- P09 分层诊断显示 action latency=1 时 previous proposal 可将 normalized RMSE 从
  0.197/0.529 降到约 0.0205/0.0157。
- 未知 lag 混合下直接拼接历史会伤害 lag=0，因此候选使用不读取真实 latency 的因果 gate。
- 等待可信数据因果门后再实施。

## `R0001-P05`：跨 source batch 三臂归因

- A：重复同一窗口；
- B：同 source 不同窗口；
- C：跨 source 不同窗口。
- 只有 C 稳定优于 B，才支持原跨 source 假设。
- 仅在数据因果门可信且世界模型动作利用仍失败时触发。

## `R0001-P10`：安全正例分层

- 仅在 P04 后执行 RMSE 可表示、但安全 recall/PR-AUC 仍失败时触发。
- P04 若已解决动作执行失败，则 `rejected without run`。

## `R0001-P17`：同状态配对实际动作干预

- 类型：训练前物理因果诊断，不是能力改进。
- 证据：
  - P16 证明现有 state-nuisance ridge 在高相关长 horizon 下功效不足；
  - 三个正式任务从同 seed、同初始 `PhysicalStateSnapshot` 重置后，同动作分支在
    proposal、实际 action、state、reward、event 和最终 snapshot 上逐元素一致；
  - 任务盲 `+d/-d` Rademacher 运动动作经原 plant 和安全层后，实际归一化动作差 RMS
    约 0.267、方向余弦 1.0，三个任务均无安全干预或严重碰撞，并在 1/4/8/16 步产生
    非零物理状态差。
- 假设：从相同初始物理状态出发，随机分配的 `+d/-d` 动作符号通过实际 plant action
  差稳定改变后续可控状态，可直接证明动作的增量物理因果效应。
- 最小验证：
  - 只从 Episode 初始 reset 状态分叉，避免未保存的中途 FIFO 或安全历史；
  - plus、minus、sham 使用相同 seed、snapshot、随机化和预算；
  - horizon 从实际 plant action 差首次非零开始；
  - snapshot 和物理 state 只作离线 outcome；
  - Episode/seed 是随机化单位。
- 主要门：
  - snapshot/sham 逐元素重放一致；
  - actual first-stage 非弱且方向对称；
  - sham family-wise FPR 单侧 95% 上界 `<=5%`；
  - blind-injection family-wise power 单侧 95% 下界 `>=80%`；
  - 三任务×四 horizon 的确认性 family 经 Holm 后全部通过；
  - 不删除安全改写或零响应样本。
- 通过后只解锁可信训练前数据因果证据，再单独路由 P11/P05；不得宣称任务能力。

## `R0001-P18`：动作序列编码可解码性

- 从同一 snapshot 执行边际匹配、时序编码不同的动作序列并解码标签。
- 两名筛选 Agent 认为该设计易利用时序生成器、安全裁剪和动作路径指纹，不能直接证明
  任务价值或因果控制。
- 状态：`rejected`，仅可作不影响决策的探索性负控。

## P17 后候选路由

### `R0001-P11`：因果 plant FIFO 与安全 rewrite 分解

- P17 已证明实际 plant action 对三任务、四个 horizon 都有稳定物理因果效应。
- P09 显示 action latency=1 时，当前 proposal 无法表示当前 actual action；使用前一
  proposal 可把 normalized RMSE 从 0.197/0.529 降到约 0.0205/0.0157。
- 直接拼接 proposal history 虽改善 lag1，却显著伤害 lag0，说明需要显式识别 plant
  latency，而不是一个共享线性 residual。
- 假设：用过去 proposal 与过去 applied feedback 因果估计固定 Episode 的 actuator gain
  和 lag，再从 proposal FIFO 产生 plant action baseline；学习头只负责安全 rewrite，可将
  确定性 plant 变换与稀有安全事件解耦。
- 最小验证先只评估非干预动作，不修改正式世界模型：
  - P09 训练 latency 0/1；
  - 独立短物理集确认 latency 1/2/3；
  - 不把真实 latency 或 actuator scale作为输入；
  - 报告冷启动与稳定阶段。
- 通过后再单独实现正式模型；安全 recall/PR-AUC 仍失败时才触发 P10。

### `R0001-P05`：跨 source batch 三臂归因

- P11 正式确认已按预注册门槛拒绝，因此触发本候选。
- 冻结 Replay 有 24 个 source Episode、每 source 7 个窗口；当前常规 sampler 的 batch
  size 为 2，通常重复同一 shard/window。
- A 为重复同一窗口，B 为同 source 不同窗口，C 为跨 source 不同窗口。
- 三臂共享相同 anchor schedule、模型初始化、更新数和留出，只替换 batch 的第二个样本。
- B/C 保持 anchor 的任务、结果类别和视觉监督 strata；只有至少有两个 source Episode 的
  strata 才进入三臂对照，全部排除项在训练前发布。
- 只有 C 在三个初始化 seed 上稳定优于 B，且真实动作绝对误差、action execution、
  collision 和数据 probe 不回归，才支持跨 Episode 假设。

### `R0001-P06`：真实动作多步 posterior overshooting

- P05 的跨 source batch 不能使 action-shuffle ratio 脱离约 1.0，说明 batch 内 Episode
  多样性不是主要瓶颈。
- 当前训练只用相邻一步 prior/posterior KL，视觉/本体重建主要解码已经看到当前
  observation 的 posterior feature，存在状态连续性捷径。
- 原 P06 的“真实动作与打乱动作 margin 排序”直接复刻正式 audit，存在评测目标泄露，
  本轮不采用。
- 改进假设：从训练 Replay 的 posterior 起点，仅用真实 executed action rollout，
  在 1/2/4/8 步匹配停止梯度的未来 posterior latent；该目标不读取正式 audit 的
  derangement、ratio、任务分区或阈值。
- 最小验证先做低成本预检：
  - 在冻结 Replay 上测量未训练 overshooting loss 的有限性、action 梯度、时间对齐；
  - action 全零与时间错位负控必须显著恶化；
  - 不更新正式模型，不查看正式 holdout 结果。
- 预检通过后才冻结权重、预算和独立训练 seed；不得与 stale-frame 修复、P10 或数据改动
  首次捆绑。

### `R0001-P19`：RSSM free-nats 梯度死区

- P05 九个正式 run 的 `world/dynamics` 与 `world/representation` 长期精确等于 1.0，
  即当前 `free_nats` 下限。
- P06 中真实动作相对 zero/shifted 只有 0.88%/0.06% 优势，说明当前 prior 尚未形成可用于
  overshooting 的动作对齐结构。
- 假设：当前 `_categorical_kl(...).clamp_min(1.0)` 使绝大多数 transition 的 dynamics KL
  梯度归零，RSSM prior 主要靠间接重建/ensemble 信号训练。
- 最小验证只读取冻结 checkpoint 与 P06 的 24 个训练窗口：
  - 报告 raw dynamics KL 分布和被 1.0 截断的比例；
  - 比较 current free-nats 1.0、预注册候选 0.1 与 raw KL 对 transition/prior 参数的梯度；
  - 不更新参数、不读取正式 holdout、不扫描其他阈值。
- 只有当前梯度近零、raw 梯度非零且 0.1 恢复有限梯度时，才允许把 free-nats 作为单变量
  训练候选。

### `R0001-P20`：RSSM action 输入量级

- P19 显示 raw posterior-prior KL 中位数 8.05，只有 3.9% transition 低于 1.0；
  free-nats=1.0 没有消除 prior/action 梯度。
- 当前 RSSM 首层直接拼接 1024 维 categorical stochastic 与 16 维物理单位 action；
  action 没有按正式上下界归一化。
- 冻结 Replay 连续 action RMS 约 0.12～0.35，canonical 归一化后约 0.60～0.71；
  checkpoint 的 stochastic/action 单元素权重 RMS 相近，但 stochastic 维数大 64 倍。
- 假设：未归一化物理单位使 action 对 transition preactivation 的贡献过小，模型更容易沿
  stochastic state shortcut。
- 最小验证只读比较：
  - 当前 raw action；
  - 按 `2*(a-min)/(max-min)-1` 映射到 `[-1,1]` 的 canonical action；
  - 同一 posterior stochastic 对 RSSM 首层 preactivation 的 RMS 贡献。
- 只有 raw action/stochastic 贡献比低、canonical normalization 至少提升 1.5 倍且不产生
  非有限值时，才允许把“仅 RSSM dynamics 输入归一化”冻结为单变量 smoke。

### `R0001-P21`：RSSM 逐级 action-effect 衰减定位

- 证据：
  - P06 中 true action 相对 zero/shifted 的多步 posterior target loss 只改善
    0.88%/0.06%；
  - 正式 teacher-forced one-step true/shuffled 物理预测 ratio 为 `1.00104`；
  - P20 中 canonical/raw variation gain 为 `2.37899`，但只有 17/24 Episode 的 raw
    action/stochastic ratio 低于 0.20，action scale 不是充分解释；
  - `step_prior` 还有 P20 未覆盖的路径：512 维 previous deterministic 作为 GRU hidden；
    checkpoint 的 GRU input/hidden Frobenius norm 为 `22.998/23.369`，参数量级不能定位
    shortcut。
- 假设：在固定同一 posterior `(h_t, z_t)` 时，true 与预注册 deranged action 的差异进入
  transition embedding 后，主要在 GRU gate/hidden retention 或 prior head 被衰减；previous
  deterministic 对下一状态的敏感度显著高于 action。
- 最小验证：
  - 只使用 P20 相同 checkpoint、训练 Replay、24 个窗口和 selection hash；
  - 对每个 transition 固定 `(h_t, z_t)`，比较 true 与三组 Episode 内经验 action
    循环置换，固定 shift 为 `1/5/9`；置换保持精确经验边际分布且无位置固定点；
  - 逐级记录 transition preactivation、LayerNorm/SiLU 输出、GRU reset/update/new gate、
    `h_{t+1}`、prior logits/probability 的 paired effect；
  - 以 Episode 内逐维自然 variation RMS 标准化，只纳入 scale `>=1e-4` 的活跃维度；
    action/h 各至少一半维度活跃，否则该 Episode 无效；
  - 局部敏感度使用冻结的 5% 凸组合：
    `x_eps=0.95*x_true+0.05*x_shifted`；action 始终位于经验物理 bounds 的凸包内，
    previous deterministic 也只做局部经验方向干预；
  - action 与 previous-deterministic 分支分别以自身标准化输入差异归一化，再比较到
    `h_{t+1}` 和 prior probability 的标准化输出 gain；
  - 不读取正式 holdout，不执行 decoder，不更新参数，不扫描 shift、epsilon 或阈值。
- 主要指标：
  - 各层 standardized paired effect 及相邻层 retention；
  - GRU update gate 分布；
  - action/deterministic sensitivity ratio；
  - 24 Episode 一致性、任务分层和全部有限性。
- 通过条件：
  - 每个 Episode 至少 2/3 shift 满足 transition activation standardized effect
    `>=0.05`；
  - `transition -> h_next` 或 `h_next -> prior probability` 的首次相邻 retention
    `<0.50`；
  - 对应输出的 action/deterministic local sensitivity ratio `<0.50`；
  - 总计至少 20/24 Episode 通过，且 clear/store/tidy 分别至少 `5/6、5/6、10/12`。
- 失效条件：effect 在 transition 前已近零、下游 retention 不低、action sensitivity 不弱、
  活跃维度不足、Episode/任务一致性不足，或任何 pairing/hash 漂移。
- 成本：一次冻结 checkpoint 前向与有限差分；不训练。
- 风险：gate 级 hook 容易误复刻 GRU 公式；必须用 `nn.GRUCell` 原输出做逐 transition
  一致性校验。有限差分方向必须来自同一冻结窗口，不能扫描扰动幅度。
- 依赖：P20 v2 的 MPS/血缘/中心化统计基础设施；P06 selector 与完整 window identity。

### `R0001-P22`：Posterior observation/deterministic 支配诊断

- 证据：
  - posterior 首层直接拼接 512 维 prior deterministic 与 512 维 observation embedding；
  - checkpoint 两支权重 Frobenius norm 为 `9.63/9.91`，单看参数没有 observation
    dominance 证据；
  - one-step 与 P06 都显示 prior action 对物理 target 的有效影响很弱，但尚未区分 posterior
    是否被当前 observation 覆盖。
- 假设：posterior logits 对当前 observation embedding 的经验干预远强于对 prior
  deterministic 的干预，导致 latent target 主要表示当前观测 nuisance，action-conditioned
  memory 对 posterior target 的约束过弱。
- 最小验证：
  - 复用 P21 相同冻结输入；
  - 在每个 transition 固定另一分支，分别对 observation embedding 与 prior deterministic
    使用预注册的同任务确定性 derangement；
  - 比较 posterior 首层 centered contribution、posterior probability KL/JS、argmax flip
    fraction；保留 true posterior 为共同 anchor；
  - 不使用 zero branch 作为主要判定，避免 OOD 零向量制造结论。
- 主要指标：
  - observation/deterministic centered contribution ratio；
  - observation-deranged / deterministic-deranged posterior divergence ratio；
  - categorical flip ratio、24 Episode 一致性和任务分层。
- 通过条件：两种 ratio 均 `>=2.0`，且 observation derangement 的 categorical flip
  fraction 更高，至少 20/24 Episode 同时成立。
- 失效条件：两支 effect 同量级、deterministic 更强、只在单任务成立，或 posterior
  divergence 近零导致 ratio 不稳定。
- 成本：一次冻结 checkpoint posterior 前向；低于 P21。
- 风险：posterior 本来就应强依赖当前 observation；即使通过，也只能证明 target
  dominance，不能单独证明 action path 是失败根因。需与 P21 级联解释，不能直接进入训练。
- 依赖：稳定的同任务 derangement、完整 window identity 和 posterior categorical
  divergence 实现。

### `R0001-P23`：Prior probability 到 argmax code 离散化

- 证据：
  - P21 的 72/72 shift 均有 transition action effect，prior probability effect 也全部有限；
  - 只有 23/72 shift 在 GRU/prior 相邻层出现 `<0.50` retention，不能解释 one-step
    physical true/shuffled ratio `1.00104`；
  - 正式 one-step 与 deployment 都使用 `sample=False`，每个 categorical variable 对 prior
    probability 直接 `argmax` 成 one-hot stochastic code；
  - decoder 接收 `[h_next, z_next]`，因此 probability 有 effect 不代表 hard code 有 effect。
- 假设：action 已改变 prior probability，但多数变化没有跨越 top-1 类别边界，stochastic
  action effect 在 deterministic argmax 处被抹除。
- 影响范围：只诊断 `prior probability -> hard stochastic code`；固定 P21 的同一
  `h_t/z_t/action shift`，不执行 decoder、Actor、target 或 loss。
- 最小验证：
  - 复用 P21 checkpoint、24 个窗口、shift `1/5/9` 和同一 true/shift prior probability；
  - 对每个 32-variable categorical 分布记录 true/shift top-1、top-2 margin、argmax code；
  - probability 与 hard one-hot code 都使用相同的 1024 categorical 坐标、true probability
    Episode natural variation scale 和 active mask；scale `>=1e-4` 的 active probability
    维至少 25%；
  - probability standardized effect 与 hard code standardized effect 均在同一 active mask
    上计算，retention 才允许取比；
  - true top-1 margin 定义为 `p_true[top1]-max(other)`；shift 后的 signed margin 固定保留
    true winner：`p_shift[true_top1]-max(other)`；报告有向 margin consumption、crossing
    fraction 与 argmax flip fraction；
  - top1/top2 margin `<=1e-8` 视为 near tie；任一 shift 有 near tie 则失效，禁止依赖
    backend tie-break 制造 flip；
  - 不做 coupled sampling，不把 soft probability 输入 decoder，不读取 target。
- 主要指标：
  - prior probability standardized effect；
  - 32 个 categorical variable 的 argmax flip fraction；
  - hard stochastic code standardized effect；
  - probability-to-code retention；
  - top-1 margin / counterfactual top-1 probability displacement；
  - 24 Episode、三任务、三个 shift 一致性。
- 通过条件：单 shift 同时满足：
  - probability standardized effect `>=0.05`；
  - probability active 维至少 25%；
  - argmax flip fraction `<=0.10`；
  - probability-to-code retention `<0.50`；
  - near tie count 为 0；
  - 全部有限且 hard code 与 RSSM `sample=False` 输出逐元素一致。
  Episode 至少 2/3 shift 通过；aggregate 至少 20/24，任务配额 `5/6、5/6、10/12`，
  每个 shift 至少 18/24。
- 失效条件：probability effect 已低、argmax 经常翻转、code retention 不低、probability
  active 维不足、存在 near tie、任务一致性不足或任何血缘/逐元素校验漂移。
- 成本：一次冻结前向，无训练、decoder、backward 或采样。
- 风险：one-hot jump 相对 probability variation 可能很大；两者必须使用同一 probability
  scale/active mask，并同时报告 raw flip 与 margin crossing，禁止只看 retention ratio。
- 依赖：P21 的内部 stage helper、固定 shift、完整血缘和 MPS 入口。

### `R0001-P24`：Visual/proprio decoder 逐层 gain

- 证据：
  - P21 显示 action effect 能进入 transition 且多数没有相邻层 `<0.50` 衰减；
  - one-step decoded physical true/shuffled ratio 仍为 `1.00104`；
  - visual/proprio head 均为 `Linear -> LayerNorm -> SiLU -> Linear`，首层 deterministic/
    stochastic 单元素权重 RMS 也相近，参数范数不能定位低 gain。
- 假设：P23 后 hard feature 仍有非零 action effect，但至少一个 physical decoder 在 hidden
  映射或最终输出层系统性压低该方向。
- 最小验证：固定 P23 true/shift hard feature，visual/proprio 分头记录 feature、Linear
  preactivation、LayerNorm 去均值/方差归一化、LayerNorm affine、SiLU hidden、output 的
  standardized effect 与相邻 retention；
  - 所有 stage scale 在比较 shift 前，由 24 个 true branch、384 transition 全局冻结，
    visual/proprio 分头分 stage/coordinate 计算，不使用 Episode 自适应 scale；
  - 对相同 full `delta stage input` 使用固定 16 段 midpoint path-integrated JVP 重建有限
    output jump；单点 JVP 不作判定；
  - 不读取 target。
- 主要指标：分头逐层 actual effect/retention、path-JVP retention、finite-jump 重建 cosine/
  relative error、LayerNorm true mean/variance 与 gamma/beta 描述统计、首个低 gain 边界。
- 通过条件：输入 feature effect `>=0.05`；按固定顺序找到首个 actual retention `<0.50`
  的边界，且同一边界 path-JVP retention `<0.50`、重建 cosine `>=0.90`、relative error
  `<=0.10`；Episode 2/3 shift 命中同一边界，aggregate 20/24 和任务配额。
- 失效条件：feature effect 已低、两头 retention/JVP 均不低、线性校验失败或两头定位不同。
- 成本：冻结 decoder 前向/JVP，无训练。
- 风险：one-hot 是有限跳变、LayerNorm 耦合坐标；路径积分只作网络链路幅度定位，不代表
  hard feature 具有物理语义。visual/proprio 不得合并掩盖或相互补足。
- 依赖：P23 hard feature 与 P21 血缘；P23 未完成前不实施。

### `R0001-P25`：Physical target scale/gradient 奖励

- 证据：
  - visual 使用原尺度 MSE+cosine，proprio 使用原尺度 MSE，没有逐维 target 标准化；
  - posterior absolute reconstruction 可由当前 observation 主导，而 prior 仅通过 KL 间接
    对齐 action；
  - 若 P24 证明 decoder effect 存活但 output error 仍不区分 action，需检查 target/loss。
- 假设：大尺度/高残差 target 维主导 loss 与梯度，或标准目标对 action-discriminative
  output direction 的梯度份额过低。
- 最小验证：冻结模型，逐 visual/proprio 维统计 target scale、innovation、decoder residual、
  action effect；比较 raw 与固定 train-only scale-whitened 诊断 loss，分解 objective 对
  decoder/latent 的梯度平方份额和 action-direction projection；不更新参数。
- 主要指标：raw/whitened shuffle ratio、effect/target scale、effect/residual、top 10% target
  维 loss/gradient 占比、action-aligned gradient fraction。
- 通过条件：P24 output effect 存活；raw ratio `<1.05` 而 whitened `>=1.05`，且 top 10%
  大尺度维贡献 `>=80%`；或 raw/whitened advantage 均 `<5%` 且 action-aligned gradient
  fraction `<5%`。
- 失效条件：分项不能复原原 loss、P24 effect 未到 output、scale whitening 不改变判定，
  或 target/noise 证据不足。
- 成本：一次冻结 forward/backward，无 optimizer。
- 风险：whitening 只作诊断，不能直接推出应修改 loss；visual latent 坐标相关性未建模。
- 依赖：P24 先证明 decoder output effect 的位置；P24 未完成前不实施。

### `R0001-P10`：安全正例分层

- 只在 P11 已使非干预/干预动作可表示、但自然 holdout 的安全 recall/PR-AUC 仍失败时启动。
- 不与 P11 或 P05 首次捆绑。
