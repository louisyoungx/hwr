# R0008 提案

## 生成过程

- 三个创新 Agent 分别从学习与交互控制、评测与物理安全、数据与训练系统方向独立只读
  检查 `AGENTS.md`、R0007、当前源码、测试和正式结果。
- 三者未修改正式实现、未启动正式训练，完成前未查看其他创新 Agent 的结果。
- 主 Agent 只做稳定编号、重复项合并、证据复核和文字归并，不按首选建议提前筛选。
- 延续观点保留原 ID 并增加修订号；本轮新观点从 `R0001-P47` 递增。
- 创新 Agent C 做过一次 hypothesis-generation 用 synthetic ridge 扫描；该结果只用于形成
  P32-E2，不是正式实验，不允许翻转 R0007 结论或在新数据结果后继续调参。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P40-E2` | 任务实体—机器人部位接触图与实体级冲量总账 | 测量合同 | A、B | 待筛选 |
| `R0001-P41-E2` | 共享 RGB-D 候选集上的 target-index-only 交互诊断 | 无训练行为诊断 | A、B、C | 待筛选 |
| `R0001-P47` | 固定相位外生动作脉冲门 | 物理因果采集诊断 | A | 待筛选 |
| `R0001-P48` | 正式能力声明的统一 deployment 资格防火墙 | 评测可信度 | B | 待筛选 |
| `R0001-P44-E1` | 严格执行器双尾单因子 override 合同 | 动力学评测合同 | B | 待筛选 |
| `R0001-P32-E2` | 合成功效预承诺的统一强 ridge P32 门 | 离线统计合同 | C | 待筛选 |
| `R0001-P49` | 128-step task-balanced 独立 source 补采 | 数据采集合同 | C | 待筛选 |
| `R0001-P46-E1` | 50-update 原子子周期恢复 | 训练系统可靠性 | C | 待筛选 |

## 共同证据与禁止边界

- 当前正式三维基线为 24 Episode、1,600 update、0 成功；Actor 从未解锁。
- 16 维动作均活跃，完整 Replay 动作有效秩约 `15.63`；问题不是简单的动作维度覆盖缺失。
- 三任务没有 bilateral contact 或 controlled motion。当前随机探索为 task-blind、
  observation-blind 的 `rho=0.96` 相关随机过程。
- R0007 P32-E1：
  - state-only rate-target ratio `1.1114809121271965`；
  - 加 controller history 后 `0.9069818836808229`；
  - 10% planted exact-pipeline power `1/200`；
  - 只能维持 `inconclusive`，不能据此启动 P31、P43 或训练。
- 当前普通 Replay 只有 24 个独立 source，任务分布餐桌 6、厨房 6、客厅 12。原始 source
  约包含 97,096 个控制 transition，最终只保留 2,688；transition 数不能替代 source 数。
- safety-positive source 只有 6 个，任务分布餐桌 2、厨房 1、客厅 3；P43 仍不能做
  source-disjoint 训练/选择/测试。
- P40-E1 只按 environment role 聚合 robot–environment contact；robot self-contact 与
  world–world contact 被忽略，且没有机器人部位、具体对象或接触—运动绑定。
- 当前 formal backend 分别对每臂所有对象执行 `any(contact)`：
  - 左右臂接触不同对象也可能增加同一步 simultaneous contact；
  - 任一接触存在时，所有对象总目标距离的改善都可能进入 controlled target progress；
  - 因此现有聚合指标不足以作为 P41-E2 的唯一主 outcome。
- P39 只隔离标准 `policy.reset()` 的 environment seed 与 policy RNG seed；不阻止同进程
  恶意策略读取 backend、audit、文件系统或 evaluator 内存。
- 没有 qualified deployment。P36-E2 的公开 planned ledger 明确
  `formal_capability_plan_usable=false`；不得运行正式 1,944-slot capability benchmark。
- 所有无训练 smoke、数据诊断、测量或评测合同都不得称为学习、家务能力、泛化或硬件安全
  改善。

## `R0001-P40-E2`：任务实体—机器人部位接触图与实体级冲量总账

### 瓶颈证据与假设

- 来源：A、B 独立提出；作为已接受 P40-E1 的测量扩展。
- P40-E1 能区分 floor/support、manipulated object、target container、articulation 和
  forbidden，但不能区分：
  - base、left arm、right arm；
  - 具体 manipulated object；
  - object–container 与 object–support world–world 接触；
  - 接触对象与随后运动对象是否相同。
- 假设：增加严格 evaluator-private 的
  `robot part × task entity × physics substep` 接触图和接触同期实体运动账本，可以排除
  floor contact、跨对象双臂接触、自由沉降和环境推动造成的假交互正例。

### 唯一主变量与影响范围

- 唯一变量是 report-only 接触证据归因粒度。
- 允许修改：
  - contact ledger 的纯测量数据结构；
  - formal binding 中显式 robot part / task entity 角色；
  - formal backend 的只读 audit observer；
  - 独立 evaluator、fixture 和测试。
- 不改变 policy 输入、动作、安全、reward、termination、success 或任何训练数据。

### 最小验证与主要指标

- 冻结 `base/left_arm/right_arm × floor/object/container/articulation/forbidden` 映射。
- 记录 task-relevant：
  - robot–environment pair；
  - object–container pair；
  - object–support pair；
  - arm–object 接触与同对象运动的控制周期对齐。
- fixture 覆盖：
  - 同一对象双臂；
  - 不同对象双臂；
  - 单臂；
  - floor-only；
  - 无接触物体运动；
  - 接触结束后的惯性运动；
  - articulation handle；
  - object–container impact。
- 主要门：
  - 所有 task-relevant pair 唯一分类；
  - fixture 分类 precision/recall 均为 1；
  - 新总账聚合回旧 P40 robot–environment 类别后的 force/impulse 误差为 0；
  - controlled motion 只允许同一实体的有效接触周期与实体位姿增量绑定；
  - enabled/disabled legacy trace bit-identical；
  - missing、nonfinite、negative force fail-closed。

### 失效、成本、风险与依赖

- 失效：
  - 机器人部位归属歧义；
  - pair 漏计、重复计数或新旧总账不守恒；
  - 跨对象接触被记为 same-object；
  - 无接触或接触结束后的运动被记为 controlled；
  - 开启测量改变 runtime 行为。
- 成本低至中，CPU fixture 与短 deterministic MuJoCo trace，无训练。
- 风险：
  - task entity graph 含 evaluator 语义，严禁进入 policy；
  - MuJoCo normal force 不覆盖切向力、力矩、压力、材料损伤或硬件风险。
- 依赖：已接受的 P40-E1。
- 禁止声明：不得声称安全、控制、任务能力或泛化改善；220N 仍不是 allowed-contact 或
  硬件安全阈值。

## `R0001-P41-E2`：共享 RGB-D 候选集上的 target-index-only 交互诊断

### 瓶颈证据与假设

- 来源：A、B、C 三者独立首选；修订 R0007 `changes_required` 的 P41-E1。
- 当前全维 observation-blind 随机动作没有进入三任务的可操作交互流形；继续增加同类
  transition 可能只会更精确地确认零交互。
- P41-E1 的 blind target matching、candidate generation、surface ranking、双臂分配和
  primitive 没有唯一化，也没有 MDE/power。
- 假设：在完全相同的 policy-visible RGB-D 候选集、固定 primitive 与预算下，只把目标索引
  从独立 policy RNG 的 hash-uniform index 改为冻结的通用几何可操作性评分 index，可以
  提高 task-contact-graph-qualified interaction Episode rate。

### 唯一主变量与影响范围

- 唯一变量：共享候选集上的 `target_index_selector`。
  - candidate：冻结几何评分的最高 index；
  - control：预提交 hash 与独立 policy RNG 决定的 uniform index。
- 每次 reset 只从 policy 实际可见的、经过 latency 的 RGB-D、动态标定、本体和历史已执行
  动作生成一次候选集。
- candidate/control 必须共享：
  - candidate-set bytes/hash、有效 mask 和顺序；
  - base/arm Cartesian primitive；
  - 速度、动作幅值、左右臂分配、双臂耦合；
  - gripper schedule、planned horizon、撤回和停止逻辑；
  - environment seed、物理随机化和初态。
- selector 运行在只接收序列化 policy input 的纯接口中，不持有 backend、audit 或 evaluator
  对象。
- 不修改世界模型、Actor、reward、task、success、安全层、Replay selector 或训练配置。

### 最小验证与主要指标

- 禁止输入：
  - instruction、task/object/target ID；
  - 颜色标签；
  - geom/body/site ID；
  - simulator object pose；
  - reward、stage、success；
  - expert waypoint/action。
- fixture 必须证明：
  - 相同 index 时两臂逐 step proposed action bit-identical；
  - candidate/control 候选集 hash 相同；
  - selector 只能访问冻结字段；
  - condition/role 不进入 environment seed。
- 先运行不据此判定优劣的小型 smoke，确认候选集非空率、primitive 可执行与产物完整性。
- 正式决策前用 exact paired binary simulation 冻结 MDE、样本量和 bootstrap/permutation
  方案；不得看到正式 outcome 后修改。
- 主要指标：
  - P40-E2 定义的 task-contact-graph-qualified controlled interaction Episode rate；
  - 事件必须包含同一 manipulated object 或 articulation 的有效机器人接触，以及接触同期
    `>=0.01m` 对象/关节位移；
  - 三任务等权 paired 改善达到冻结 MDE；
  - paired 95% lower bound `>0`；
  - 每任务方向非负并报告最弱任务。
- 描述性分账：
  - same-object dual-arm；
  - distinct-object dual-arm；
  - single-arm；
  - manipulated-object、articulation、container 与 floor/support；
  - supported observation latency 与 latency-3 challenge。
- 守护：
  - severe collision 为 0；
  - stale action 实际应用为 0；
  - safety intervention burden 不劣；
  - P40-E2 force/impulse 完整发布；
  - executed RMS、有效秩和活跃维度不塌缩；
  - planned horizon 内提前终止按失败计，不删除或补 seed。

### 失效、成本、风险与依赖

- 失效：
  - 候选集、primitive 或 termination 在两臂间不同；
  - blind 使用仿真真值或 candidate 结果作后验匹配；
  - 收益来自 floor/support、固定家具、容器、跨对象误计或单臂接触；
  - 接触增加但 contact-coupled motion 不增加；
  - 只有单任务改善；
  - 几乎不动换取低风险；
  - 严重碰撞、安全 burden、stale action 或动作塌缩守护失败；
  - 功效不足。
- 成本中等，短 paired MuJoCo Episode，无训练、无 foundation feature 物化、无 checkpoint。
- 风险：
  - 公共 primitive 仍是实验仪器，不能冒充 autonomous skill；
  - 几何评分可能偏向大平面；
  - 任务集不足以证明跨家庭泛化。
- 依赖：P40-E2、P39 seed 合同和现有动态 RGB-D 标定。
- 禁止声明：只允许称为“target selection 对交互数据产率的诊断”；不得称为学习、任务
  成功、策略能力、泛化或安全改善。

## `R0001-P47`：固定相位外生动作脉冲门

### 瓶颈证据与假设

- 来源：A。
- P17 证明 reset 状态下 plant 可控，但 P32 的普通 Replay signal 被 controller history
  解释；需要在交互流形附近产生一个 history 不能预测、又确实经过 safety 与 plant 的动作
  innovation。
- 假设：P41-E2 固定 primitive 到达结果前定义的 policy-visible phase 后，注入
  `z∈{-1,0,+1}` 的小幅任务盲双臂脉冲，会形成可识别 first stage，并方向性改变 P40-E2
  绑定的对象/关节运动。

### 唯一主变量与最小验证

- 唯一变量：固定时刻、固定幅值、固定持续时间的 pulse coefficient `z`。
- 固定 P41 selector、target index、prefix、历史和所有 primitive。
- pulse 方向只由 policy-visible local frame 决定，符号由独立 policy RNG 预先生成。
- 每 seed 从同一 reset 重放；pulse 前 observation、proposal、executed action、事件和状态
  必须逐元素一致。
- 运行 plus/minus/sham；保存 actual action、P40-E2 接触绑定和 evaluator-private outcome。
- 运行 exact permutation、sham FPR 与 planted power 门。

### 指标、失效与边界

- 主要指标：
  - pulse 对 actual action 的冻结 first-stage RMS；
  - pulse sign 与 contact-coupled entity motion 的 signed cross-moment；
  - 三任务经多重校正后方向一致；
  - `>=0.01m` contact-coupled motion 达冻结 MDE。
- 守护：sham 一致、零严重碰撞、P40-E2 force/impulse 不越预注册边界、结果不能只来自
  safety rewrite。
- 失效：prefix 不一致；pulse 被 FIFO/safety 完全消除；只有 proprio 变化；仅单任务有效；
  需要接触真值决定 pulse 时机；功效不足。
- 成本中等，paired MuJoCo 短 Episode，无参数更新。
- 依赖：P40-E2 与 P41-E2 先通过；否则 defer。
- 禁止声明：不得称为 production RSSM 使用动作、普通 Replay 已可训练、Actor 已解锁、
  learned skill 或任务成功；通过最多授权新的 source 采集合同。

## `R0001-P48`：正式能力声明的统一 deployment 资格防火墙

### 瓶颈证据与假设

- 来源：B。
- foundation evaluator 会在读取正式 salt、创建输出和启动环境前拒绝不 qualified 的
  deployment；但旧共享 bimanual evaluator 只验证旧训练 artifact 与 no-expert lineage。
- 未来 P36/P42/P44 或 P41 runner 若错误复用共享 evaluator，可能把随机、scripted 或未
  qualified policy 的 success 输出误当正式能力。
- 假设：把“可运行诊断”和“可发布能力结论”显式分离，要求所有正式聚合器消费 hash-bound
  qualification identity，可以关闭绕过 foundation gate 的报告路径。

### 最小验证与指标

- 唯一变量：报告的 capability authorization provenance。
- 不改变 policy、环境、随机化或成功判据。
- 未 qualified 的 P01 v4、随机策略、scripted primitive 和 legacy bimanual run 必须在：
  - 读取私有 salt；
  - 创建输出；
  - 构建环境；
  - 执行 action；
  之前拒绝正式模式。
- 诊断模式固定 `capability_claim_allowed=false`。
- 合成 qualified fixture 必须绑定 deployment、checkpoint、causality report、Actor
  readiness、训练 update、P39 seed lineage 和 deployment hash 固定后的私有 salt。
- 主要门：拒绝矩阵 100% fail-closed；正式 report 唯一重建 qualification hash chain；
  diagnostic report 不能被正式 aggregator 接受。

### 失效、成本、风险与依赖

- 失效：任一入口仅凭 checkpoint、policy ID、`passed=true` 或 scripted trace 进入正式
  聚合；salt 提前暴露；旧 report 静默升级。
- 成本低，主要为 provenance schema 与测试。
- 风险：影响 legacy 工具兼容性，应保留显式 diagnostic/legacy mode。
- 依赖：foundation registry、P39、P36-E2；不依赖训练。
- 禁止声明：只提升评测防误用，不代表能力、泛化或安全改善。

## `R0001-P44-E1`：严格执行器双尾单因子 override 合同

### 瓶颈证据与假设

- 来源：B；修订 deferred 的 P44。
- evaluation actuator scale `[0.88,1.12]` 与训练 `[0.96,1.04]` 重叠，宽区间结果不能称
  严格未见动力学；实际 scale 只作用于 action 前 14 维，不包括两个 gripper target。
- 假设：evaluator-private nominal/strict-low/strict-high override 能保持其他 randomization、
  初态、语言、相机和 RNG 消耗不变，为未来执行器增益 OOD 对比建立单因子合同。

### 最小验证与指标

- 唯一变量：前 14 维 base/arm action 的 actuator scale。
- 同 planned Episode 建立：
  - nominal `1.0`；
  - low-tail `[0.88,0.96)`；
  - high-tail `(1.04,1.12]`。
- 去除 scale 后 randomization、state、calibration、instruction hash 必须相同。
- 固定 pulse action 验证前 14 维缩放和两个 gripper target bit-identical。
- condition 不进入 environment/policy seed。
- 主要门：100% 严格尾部、0 个训练域样本、所有非 scale 字段 hash 相同、动作缩放误差
  在冻结数值容差内、provenance 完整。

### 失效、成本与边界

- 失效：tail 落入训练域；scale 改变 RNG 消耗或其他随机化；gripper 被缩放；condition
  对 policy 可见。
- 成本低，CPU reset 与 short apply，无能力 Episode。
- 风险：只覆盖执行指令 gain，不代表饱和、摩擦、背隙、时延或完整 sim-to-real。
- 依赖：P39；未来能力结果仍依赖 qualified deployment 与私有 salt。
- 禁止声明：本轮最多接受为动力学评测合同，不能给出 OOD 成功率或硬件迁移结论。

## `R0001-P32-E2`：合成功效预承诺的统一强 ridge P32 门

### 瓶颈证据与假设

- 来源：C；修订 R0007 inconclusive 的 P32-E1。
- controller-history 特征为 185-D，而最弱 inner-train task 只有 2 个 source；固定
  `ridge=1e-3` 的 planted power 仅 `1/200`。
- hypothesis-generation synthetic 扫描显示 `ridge=100` 时 visible-state 与
  controller-history 两条 planted pass 均为 `199/200`，两个 null 仍为 `0/200`；但
  `ridge=1000` 又令 controller power 降为 0。
- 这只是事后形成假设的证据，不能重解释 R0007；真实判断必须使用结果前冻结 estimator 和
  fresh cohort。
- 假设：对 action nuisance、target nuisance 和 residual map 统一固定 `ridge=100`，能
  降低高维小-source 方差，同时维持 null 校准。

### 最小验证与指标

- 唯一变量：P32 全部线性映射统一 ridge `1e-3 → 100`。
- 第一阶段只用历史 source/state/action 设计运行 synthetic zero/random/planted：
  - 不重算或发布历史真实 effect；
  - visible-state 与 controller-history 分开；
  - 200 trial × 1,000 source bootstrap。
- 接受 estimator 合同要求：
  - 两种 feature family planted-power exact 95% lower `>=0.80`；
  - 两个 null exact 95% upper `<=0.05`。
- 通过后冻结 ridge；只能在 P49 fresh cohort 上一次性运行真实 P32。
- fresh cohort 保留 P32-E1 原有：
  - `1.05` 主门；
  - `1.02` controller-history 门；
  - configuration target；
  - 三任务方向；
  - source/task 等权 bootstrap；
  - source-disjoint nested fitting。

### 失效、成本与边界

- 失效：任一 feature family 功效不足或 null 膨胀；fresh outcome 后继续扫描 ridge；删
  controller/configuration guard；只报告有利 target。
- 成本低，CPU 分钟级 synthetic contract；不运行 MuJoCo，不更新参数。
- 风险：强 ridge 可能过度收缩真实非线性信号。
- 依赖：真实判断依赖 P49 的全新 source；与 P49 必须在采集 outcome 前共同冻结。
- 禁止声明：不得翻转 R0007，或声称 plant causality、production utilization、训练或
  闭环能力改善。

## `R0001-P49`：128-step task-balanced 独立 source 补采

### 瓶颈证据与假设

- 来源：C。
- 当前 24 个独立 source 为餐桌 6、厨房 6、客厅 12；约 97,096 个原始 transition 最终
  只保留 2,688。
- 当前 task sampler 前 6 Episode 轮转平衡，之后为 outcome-adaptive，不能保证统计诊断
  所需的 task-balanced source quota。
- 假设：使用现有随机 action process、安全层和 selector，固定
  `12 source/task × 3 task × 128 transition` 的紧凑 cohort，可以用 4,608 个 control
  transition 获得 36 个 task-balanced 独立 source。

### 唯一主变量与最小验证

- 唯一变量：预提交的 compact balanced source schedule。
- 新建 collection-only run：
  - 不加载或更新模型；
  - 不运行 Actor warmup、feature materialization 或能力评测；
  - 每任务预提交 12 个 source；
  - seed 全局唯一且与历史不重叠；
  - 每 source 最多 128 transition；
  - 使用现有 `random_rl_exploration`：
    `rho=0.96`、gripper flip `0.05`；
  - 使用现有 runtime、安全和 7×16 selector。
- 所有 planned attempt 均记录；不按 safety、contact、motion 或 target outcome 补采、替换
  或删除。

### 指标、失效与边界

- 主要完整性门：
  - 36 个唯一 source ID；
  - 每任务 12；
  - seed 全局唯一；
  - 每个完整 source 128 transition、7 个不重叠 shard；
  - planned、valid、unresolved 恒等；
  - manifest 与 shard hash 完整；
  - source 不跨 task/fold。
- 守护：完整报告短 Episode、safety rewrite、P40-E2 contact、controlled motion 和 selector
  起点；统计继续 source/task 等权；不得把 4,032 retained transition 当独立样本。
- 失效：任一任务 quota 不足；复用 seed/source；按结果替换；action/safety 合同漂移；
  为达到 P32 功效临时追加 source；大量早停使冻结 cohort 不可分析。
- 成本低至中，4,608 control transition；初步存储估计小于 1GiB，必须记录实测 wall time
  与 bytes。
- 风险：128-step source 偏向 Episode 早期；同一 random process 仍可能完全没有物体交互；
  不保证足以支持 P43。
- 依赖：若用于真实 P32-E2，estimator、fold、MDE 和判定必须在采集 outcome 前冻结。
- 禁止声明：不得称为数据效率或能力改善，不得把更多 source 自动解释为更好 Replay。

## `R0001-P46-E1`：50-update 原子子周期恢复

### 瓶颈证据与假设

- 来源：C；修订 R0007 `approved, not selected` 的 P46。
- 每周期 200 update 后才发布 recovery-complete checkpoint；P01 v4 最后一周期 update
  约 758.54 秒，而 checkpoint 约 1.91 秒。
- progress 每 10 update 发布，但没有绑定 durable trainer、optimizer、RNG、sampler 与
  partial metrics state。
- 假设：每 50 update 原子覆盖一个独立 mid-cycle recovery slot，可把最大重复计算从
  199 降到 49 update，同时保持恢复轨迹一致。

### 最小验证与指标

- 唯一变量：recovery checkpoint cadence。
- 不改变 Replay、sampling probability、batch 顺序、模型、loss、优化步数、正式
  `latest.json` 或 deployment 暴露。
- 200-update CPU fixture 在 update 49/50/51/99/100/149/150/199 及 atomic rename 前后
  注入故障；再用冻结 P01 Replay 做 100–200 update smoke。
- 恢复后比较：
  - batch identity trace；
  - NumPy/Torch RNG；
  - model 与所有 optimizer；
  - update count；
  - partial metric accumulator；
  - Replay manifest。
- 主要门：
  - CPU 最终状态 bit-identical；
  - batch identity 完全一致；
  - 最大重复 update `<=49`；
  - 不重复采集或记账；
  - checkpoint overhead `<=2%`；
  - 瞬时额外磁盘 `<=1.5GiB`；
  - 损坏或不完整 snapshot fail-closed。

### 失效、成本与边界

- 失效：任一 RNG/batch/parameter/optimizer/metric 漂移；恢复需重新采集；mid-cycle pointer
  覆盖正式 deployment/latest；写放大超限。
- 成本低至中，无正式训练；MPS 可只按结果前冻结数值容差验证。
- 风险：MPS 未必 bit-exact；频繁写盘可能抖动；不得混淆 mid-cycle recovery 与周期末
  causality-qualified checkpoint。
- 依赖：现有 training checkpoint、runner recovery、atomic manifest 和 progress metrics。
- 禁止声明：不得计作能力、样本效率或计算效率改善；看门狗不得在 calibration/evidence
  gate 的预期失败后自动重启，也不得重复启动同一 run。

## 创新 Agent 首选与分歧

- A 首选 P41-E2，但要求先做臂—对象绑定测量；P47 只在 P41-E2 通过后执行。
- B 首选先实施 P40-E2，再执行 P41-E2；另强调 capability evaluator 的资格防火墙。
- C 首选 P41-E2；同时建议先冻结 P32-E2 synthetic power，再补 P49 fresh source。
- 三名 Agent 一致认为：
  - 当前最高价值能力瓶颈是没有 task-relevant 物理交互；
  - P41-E2 必须严格做到 target-index-only；
  - 不应在当前 24-source Replay 上直接继续 objective/head 训练；
  - P43 暂不具备独立 positive source；
  - P46-E1 只能作为可靠性侧车。
- A/B 独立发现当前 bilateral/controlled 指标缺少 arm–entity–motion 绑定；主 Agent 将其
  合并为 P40-E2。
- B 认为 P40-E1 本身不足以作为 P41-E2 主指标前置；A 也要求先补测量，二者在依赖顺序上
  一致。
- C 提出的 P32-E2/P49 是另一条数据可识别性路径；它不应与 P41-E2 混成一个因果实验。
