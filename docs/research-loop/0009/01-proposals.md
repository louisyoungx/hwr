# R0009 提案

## 生成过程

- 三个创新 Agent 分别从因果学习与世界模型、数据效率与泛化、评测安全与闭环物理方向
  独立只读检查 `AGENTS.md`、R0008、当前源码、测试与正式产物。
- 三者在完成前未查看其他创新 Agent 输出，未修改正式实现，未启动正式训练。
- 主 Agent 只做稳定编号、重叠项合并、证据复核和实验变量拆分，不按首选建议提前筛选。
- 历史观点重提保留原 ID 并增加修订号；本轮新观点从 `R0001-P50` 递增。
- 创新 Agent C 对现有 P41 terminal 做过 evaluator-private 只读复算，创新 Agent A/C
  做过小型只读诊断。它们只用于形成假设，不是正式结果，不得直接改变历史结论或接受门。

## 共识证据

1. 当前可信能力基线仍是 24 Episode、1,600 update、0 成功；Actor 未解锁，三任务无
   bilateral contact 或 controlled motion。
2. P17 已证明 plant 会响应动作；当前主要瓶颈不是“环境完全不可控”。
3. 现有普通 Replay 只有 24 个独立 source，P32 中 action information 被 controller
   history 解释，10% planted exact-pipeline power 仅 `1/200`。
4. P41 smoke 的候选数为 `4/0/1/3/5/3`。客厅 latency `(2,2)` 虽有 18 个 keyframe、
   完整 1,655 step 和通过的安全守护，最终 candidate set 仍为空。
5. 其余五个非空 P41 cell 也没有 single-arm、same-entity dual-arm 或
   same-object dual-arm grasp；所有主事件为 false。
6. 当前 P41 candidate generator 只读取 depth；RGB 被序列化但未用于生成。流程包含逐帧
   validity、prominence、局部厚度、self-mask、范围、平面性、宽度及至少双视角合并，
   但正式产物只报告最终 candidate 数。
7. P41 primitive 在 acquisition frame 计算末端误差，实际 dual-arm backend 将 3D
   Cartesian command 当作当前 base frame 向量再旋转到 world frame；现有测试没有验证
   非零底盘 yaw 下两者语义一致。
8. P41 用手写 policy FK 估计工具位置，plant 使用 MuJoCo grasp-center site 与 Jacobian。
   现有证据没有冻结两者的真实位置误差门。
9. 创新 Agent C 的假设形成复算认为，现有 6 个 smoke cell 的 top-1 几何候选只有厨房
   2 个 cell 接近 task entity；客厅和餐桌候选更接近沙发或椅背。该复算不是正式结果，
   只能支持“非空不等于可操作实体覆盖”的风险。
10. 三个 Agent 一致反对当前直接修改 RSSM loss 或启动正式世界模型训练。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P50` | policy-visible 候选留存漏斗与视觉身份诊断 | report-only 测量诊断 | A、B、C | 待筛选 |
| `R0001-P51` | Cartesian primitive acquisition→base 坐标合同 | 单变量行为修复 | C，A 间接支持 | 待筛选 |
| `R0001-P52` | policy FK—plant tool-site 一致性门 | report-only 运动学诊断 | A | 待筛选 |
| `R0001-P53a` | 位姿补偿的多帧 RGB-D 几何候选融合 | 候选生成候选 | B | 待筛选 |
| `R0001-P53b` | 冻结基础视觉特征的 RGB-D 实例候选 | 候选生成候选 | C | 待筛选 |
| `R0001-P54` | 冻结视觉—语言联合空间的指令 target selector | 无训练语义选择候选 | B | 待筛选 |
| `R0001-P49-E1` | 任务×语言×布局×动力学平衡独立 source | 数据采集合同 | B | 待筛选 |
| `R0001-P47-E1` | 固定预接触相位的外生动作创新门 | 物理因果采集诊断 | A | 待筛选 |
| `R0001-P55` | 逐实体操纵链能力成功证书 | report-only 评测收紧 | C | 待筛选 |

## `R0001-P50`：policy-visible 候选留存漏斗与视觉身份诊断

### 瓶颈证据与假设

- 来源：A 首提，B/C 的候选覆盖证据支持。
- 当前只知道最终候选数，不知道空集合来自：
  - depth 无效或视野覆盖不足；
  - center/ring validity；
  - prominence；
  - center depth spread；
  - patch support；
  - robot self-mask；
  - height/range；
  - planarity；
  - width；
  - 跨帧 graph merge；
  - `view_count >= 2`。
- 自然 observation latency 可能使多个 capture ordinal 对应重复的 observation identity。
  当前 merge 用 `frame_ordinal` 计算 view count，未区分新的 capture ordinal 与新的视觉证据。
- 假设：空候选与低任务实体覆盖由少数可定位阶段主导；其中重复 observation identity 和
  `view_count >= 2` 的组合是客厅 latency-2 空集合的主要可证伪解释之一。

### 唯一主变量与范围

- 唯一变量：增加不影响输出的 report-only 候选漏斗和视觉身份审计。
- 正式 P41 candidate bytes、候选顺序、selector、primitive、action、安全、termination、
  success 和 Replay 均不得改变。
- 每个 keyframe 同时记录：
  - capture ordinal；
  - observation timestamp/sequence；
  - depth payload hash；
  - 有效 depth 比例；
  - 各冻结 gate 的输入、通过与首次拒绝计数。
- 允许 shadow 计算“按唯一 observation identity 计 view”的结果，但该 shadow 不得进入
  runtime candidate set 或动作。

### 最小验证与主要指标

- 先用纯合成 fixture 验证每一 gate 的唯一 first-rejection 分类、计数守恒和输入顺序不变。
- 再用新的、结果前 seed 运行 acquisition-only cohort；R0008 的 6 个 seed 只能用于
  regression，不进入接受判定。
- 样本单位是完整 acquisition Episode，不是 frame、anchor 或 raw candidate。
- 主要指标：
  - 每 Episode capture ordinal 数、唯一 timestamp/sequence 数和唯一 depth hash 数；
  - 每一级 first-rejection Episode 分账及 candidate survival funnel；
  - 空集合 Episode 中最后一个非空 stage；
  - ordinal-view 与 unique-identity shadow merge 的 source-level 配对差；
  - 三任务、latency 1/2 完整分账和最弱 cell。
- 测量合同接受门：
  - 所有 anchor 恰好进入一个 terminal stage，计数守恒；
  - enabled/disabled 的正式 candidate canonical bytes/hash 完全一致；
  - 同输入重复运行 report bit-identical；
  - 禁止字段无法进入诊断算法；
  - 全部 planned Episode 保留，无补 seed。
- 假设失效：
  - 空集合在 depth/raw-anchor 之前已经形成；
  - 不存在重复视觉 identity；
  - unique-identity shadow 与正式 merge 无差异；
  - 没有单一或少数阶段解释主要损失。

### 成本、风险与依赖

- 成本低，CPU acquisition-only MuJoCo Episode，无训练、无 post-selection primitive。
- 风险：leave-one-gate-out 容易诱导后验调参；本提案只定位，不授权修改任何 P41 gate。
- 依赖：P29、P39、P41-E2 已实现 acquisition。
- 与历史关系：是 P41-E2 的新前置测量，不重跑 selector 对照，不覆盖其 `inconclusive`。
- 禁止声明：不得称为候选能力、交互改善、学习、泛化或安全改善。

## `R0001-P51`：Cartesian primitive acquisition→base 坐标合同

### 瓶颈证据与假设

- 来源：C 首提；A 的“非空候选仍零 arm contact”证据间接支持。
- P41 在 acquisition frame 中计算 `target - tool`，随后直接将该三维误差编码为 arm
  command；backend 将 command 视为当前 base frame twist 并旋转到 world frame。
- acquisition 执行两次 panorama 和 base forward，post-selection 时 base yaw 通常不为
  acquisition yaw；因此存在明确坐标语义风险。
- 假设：只把 acquisition-frame 平移误差旋转到当前 base frame，再按原幅值生成 canonical
  action，可恢复末端朝冻结 acquisition-frame target 的收敛。

### 唯一主变量与范围

- 唯一行为变量：arm Cartesian linear command 的
  `acquisition frame -> current base frame` 旋转。
- 不改变：
  - candidate generator、candidate bytes、selected index；
  - acquisition、base navigation、phase、target points；
  - velocity cap、gripper、时长、撤回和停止；
  - backend、IK、task、reward、success、安全和 P40。
- 禁止使用 MuJoCo site、object pose、entity ID、contact 或 evaluator 状态生成动作。

### 最小验证与主要指标

1. 解析 fixture：
   - 覆盖 base yaw `0, ±pi/2, pi` 和随机 yaw；
   - runtime 旋转后 world/acquisition-frame 速度必须与目标误差方向一致；
   - yaw=0 时新旧动作 bit-identical。
2. evaluator-private kinematic fixture：
   - 固定 policy-visible joints、base pose 与 synthetic candidate；
   - 一步和多步末端目标误差单调下降；
   - 不允许用 fixture 结果改 velocity 或 target。
3. 若上述门通过，再以新 salt 做小型固定候选 paired MuJoCo smoke；不能与候选生成修订
   捆绑。

主要指标：

- 每臂 acquisition-frame target error 的 signed directional derivative；
- pre/contact phase 的 minimum tool-target distance；
- grasp-pad-qualified task-entity contact Episode rate；
- 执行动作 RMS、活跃维、安全干预、stale action、P40 守恒。

### 失效、成本、风险与依赖

- 失效：
  - 解析方向仍不一致；
  - 修复后 target error 不下降；
  - 接触只来自 base 撞物或错误家具；
  - 需要同时改 FK、候选、phase 或幅值才有效；
  - 任一安全守护失败。
- 成本低，无训练；解析/运动学 fixture 后只有少量 paired Episode。
- 风险：更准确地执行错误候选也会增加错误接触；因此不能将接触增加直接称为能力改善。
- 依赖：P40-E2、P41-E2、P39。
- 与历史关系：新问题。P41-E2 冻结了动作 bit-identity，但未验证跨 frame 执行语义。
- 禁止声明：最多接受为 Cartesian primitive correctness evidence；不授权 P41 正式对照、
  Replay 收集或训练。

## `R0001-P52`：policy FK—plant tool-site 一致性门

### 瓶颈证据与假设

- 来源：A。
- P41 用手写 `_arm_chain` 根据可见 joints 估计工具位置；实际 plant 用 MuJoCo
  grasp-center site 和 Jacobian 解算 twist。
- 当前没有 policy FK 与真实 tool site 的误差分布，也没有证明该误差小于接触 primitive
  的空间容差。
- 假设：手写 FK 存在足以解释零 arm contact 的系统误差；若如此，使用部署机器人模型的
  精确 FK 能在不改变其他变量时消除该误差。

### 唯一主变量与范围

- 第一阶段唯一变量是 report-only FK 误差测量，不改变动作。
- evaluator-private MuJoCo site 只作为标签，绝不能进入 policy 输入。
- 只有当前 FK 预注册误差门失败且可部署 FK 通过，才允许下一轮另行冻结行为修复；本提案
  不自动实施该修复。

### 最小验证与主要指标

- 结果前冻结 joint-grid 与自然轨迹 state，两臂等权。
- 对每个 state 比较 policy FK 与 MuJoCo grasp-center site 在当前 base frame 的位置。
- 主要指标：每臂和 aggregate 的 median/p95/max 位置误差及方向偏差。
- 建议判别门：
  - 当前 FK p95 `>0.03m`：支持“误差足以影响接触”的假设；
  - 可部署替代 FK p95 `<=0.01m`：才具备后续修复前置；
  - 当前 FK p95 `<=0.01m`：拒绝该瓶颈假设。
- fixture 覆盖关节上下限、随机内部状态、左右臂镜像和 base yaw；缺失/非有限 fail-closed。

### 成本、风险与依赖

- 成本低，CPU kinematic fixture 与短自然 state 采样，无训练。
- 风险：MuJoCo site 是 evaluator-private ground truth；测量代码必须与 policy 实现隔离。
- 依赖：机器人模型与 grasp-center binding identity。
- 与历史关系：新运动学测量，不重复 P17 plant action 因果或 P51 frame contract。
- 禁止声明：不得称为控制、接触、任务或硬件迁移改善。

## `R0001-P53a`：位姿补偿的多帧 RGB-D 几何候选融合

### 瓶颈证据与假设

- 来源：B。
- 当前逐帧硬阈值后再做跨视角 component merge，可能在 latency、dropout、遮挡和小物体
  视角变化下丢失候选。
- 假设：先用 policy-visible camera calibration 与 base pose 将全部 acquisition depth
  融合到 acquisition-frame 3D evidence，再生成几何候选，可提高任务实体 proposal recall
  和支持域非空率。

### 唯一主变量与范围

- 唯一变量：candidate generator；与 `P53b` 互斥。
- acquisition、输入白名单、selector、primitive、安全、budget 与 outcome 固定。
- 只允许 policy-visible RGB-D、动态标定、base pose 和 joints；禁止 evaluator pose/mask。

### 最小验证与主要指标

- 必须依赖 P50 定位结果，先冻结体素/融合、动态点处理、surface grouping 和全部阈值。
- 使用新 `3 task × 2 supported latency × N seed` acquisition-only paired cohort；N 由
  source-level power 结果前决定。
- evaluator-private task-entity geometry 只用于评分，不反馈 generator。
- 主要指标：
  - visibility-conditioned task-entity proposal recall；
  - 三任务等权 supported-cell 非空率；
  - paired source 改善置信下界；
  - floor/support/fixed furniture/self false-proposal rate；
  - candidate count、bytes/hash 可重建性与计算成本。
- 失效：只增加数量不增加 entity recall；只改善一个任务；ghost surface 或 false proposal
  超门；依赖 evaluator truth；后验调参。

### 成本、风险与依赖

- 成本低至中，acquisition-only MuJoCo 与 CPU 几何处理，无训练。
- 风险：动态物体 ghost、里程计误差、内存与阈值复杂度。
- 依赖：P50、P29、P39。
- 与历史关系：P41 的候选生成分支，不比较 target index。
- 禁止声明：不得称为策略、任务成功或泛化能力改善。

## `R0001-P53b`：冻结基础视觉特征的 RGB-D 实例候选

### 瓶颈证据与假设

- 来源：C。
- 当前 RGB 未参与候选生成，几何 top-1 可能偏向沙发、椅背等大结构。
- 假设：冻结的任务无关视觉 patch 特征聚类配合 depth 几何一致性，可比纯几何生成器提高
  小型可动物体的候选覆盖。

### 唯一主变量与范围

- 唯一变量：candidate generator；与 `P53a` 互斥。
- 模型权重、revision、预处理、特征层和聚类规则全部结果前锁定；不微调。
- 禁止 task/object ID、instruction、颜色规则、simulator segmentation、geom/body ID、
  真实 pose、contact 或 evaluator mask。
- selector、primitive、acquisition、安全和预算固定。

### 最小验证与主要指标

- 在新预提交 acquisition cohort 上对相同 policy-visible frames 同时运行旧/新 generator。
- evaluator-private entity surface distance 仅用于评分。
- 主要指标：
  - 三任务等权 task-entity candidate recall；
  - top-1 entity hit rate；
  - supported cell 非空率与最弱任务；
  - 家具/容器/地面 false-target rate；
  - 每 Episode 候选数、推理 wall time、峰值内存；
  - 未见布局、材质、相机扰动分账。
- 失效：只增加候选数量；依赖 benchmark 纹理捷径；最弱任务不改善；成本超过冻结预算；
  物理接触守护恶化。

### 成本、风险与依赖

- 成本中等，无训练但需要冻结基础视觉模型推理。
- 风险：域偏移、过分割、纹理捷径、计算成本。
- 依赖：P50、foundation model locks、P29、P39。
- 与历史关系：P41 的互斥候选生成分支，不重复 P42。
- 禁止声明：离线 entity recall 不能称为 affordance、交互或任务能力。

## `R0001-P54`：冻结视觉—语言联合空间的指令 target selector

### 瓶颈证据与假设

- 来源：B。
- P41 不读取 instruction，几何 score 无法区分同一布局中的任务相关与无关实体。
- 仓库已有冻结 SigLIP2 patch-grid 与 multilingual text embedding。
- 假设：在一个已证明覆盖任务实体的共享候选集上，只用冻结视觉—语言相似度选择 index，
  可提高任务实体 top-1 命中并对未见中文改写保持一致。

### 唯一主变量与范围

- 唯一变量：共享 candidate set 上的 target-index selector。
- candidate generator、acquisition、primitive、预算、environment seed、安全和终止固定。
- 不训练或微调视觉—语言模型，不允许 task/object ID、simulator truth 或颜色规则。

### 最小验证与主要指标

- 只有 P53a/P53b 之一先证明 task-entity candidate recall 后才可执行。
- 第一阶段只离线比较：
  - matched instruction；
  - 同目标未见 paraphrase；
  - 同场景 mismatched/counterfactual instruction。
- 主要指标：
  - matched top-1 entity hit rate 相对几何 selector 的 source-level paired 改善；
  - 三条 evaluation paraphrase 的最弱表现和选择一致性；
  - matched 相对 mismatched 的因果选择差；
  - instruction permutation sensitivity。
- 离线通过后，物理交互仍需下一轮重新冻结 P41-style paired 诊断。
- 失效：对 instruction permutation 不敏感；只依赖背景/颜色；训练原句有效但改写失效；
  entity hit 上升而交互产率不升。

### 成本、风险与依赖

- 成本低至中，冻结 SigLIP 推理，无训练。
- 风险：语义相关不等于可操作性；多对象指令含混。
- 依赖：P53a/P53b 中一个候选覆盖方案通过，基础模型锁可用。
- 与历史关系：不重复 P12 的语言 embedding 物化或 P42 的 deployment 成功门。
- 禁止声明：不得称为任务理解、闭环成功或语言泛化能力。

## `R0001-P49-E1`：任务×语言×布局×动力学平衡独立 source

### 瓶颈证据与假设

- 来源：B；修订 R0008 `changes_required` 的 P49。
- 当前只有 24 个独立 source，任务分布 `6/6/12`；增加 transition 不能替代独立 source。
- 假设：在相同 source 数和固定 outcome-blind retention 下，结果前平衡任务、训练指令、
  初始布局和训练域动力学/传感器因素，可降低 P32 类 source-level 估计方差。

### 唯一主变量与范围

- 唯一变量：source seed/condition schedule。
- 动作过程、Episode 长度、retention、模型、训练预算与安全均不变。
- 先纯 CPU 生成并锁定 factor-balanced 与 hash-uniform control 设计，不先采 outcome。

### 最小验证与主要指标

- 先比较设计 discrepancy 与 exact-pipeline synthetic power。
- 只有 source 数由功效预注册且显著低于简单平衡设计，才允许 collection-only run。
- early termination 保留且不补 seed；source Episode 是唯一独立单位。
- 主要指标：
  - 每任务、每训练 instruction 配额；
  - 连续随机化 marginal discrepancy 与 pairwise coverage；
  - 预注册 MDE 所需 source 数；
  - planted power、null FPR、nuisance OOF；
  - 真实采集后 interaction-positive source 只报告，不据此补样。
- 失效：功效不改善；扭曲自然训练分布；任一 quota 因早停缺失；仍无 interaction-positive
  source；真实信号继续被 controller history 消除。

### 成本、风险与依赖

- 设计验证低；采集低至中，无参数更新。
- 风险：36 source 仍可能不足，连续因素离散分层可能人为。
- 依赖：P39、P40-E2、重新冻结的 P32-E2 estimator 与 nonlinear confounded null。
- 与历史关系：P49 修订，不是新提案；区别于 P03 任务配额和 P45 retention topology。
- 禁止声明：不得称为数据效率、学习或能力改善。

## `R0001-P47-E1`：固定预接触相位的外生动作创新门

### 瓶颈证据与假设

- 来源：A；修订 R0008 deferred 的 P47。
- P17 证明 reset 附近 plant 可控；P32 表明普通 Replay action signal 被 controller history
  解释。未来需要在可操作表面附近形成 history 不可预测的 actual-action innovation。
- 假设：在结果前固定的 policy-visible 预接触 phase 注入
  `z in {-1,0,+1}` 的小幅双臂 pulse，可形成强 first stage，并方向性改变实体距离、接触或
  contact-associated motion。

### 唯一主变量与范围

- 唯一变量是 pulse coefficient `z`。
- candidate、target index、prefix、phase、时刻、持续时间、幅值和 primitive 固定。
- phase 只能由 pulse 前 policy-visible 状态决定；禁止按后续是否接触筛选。

### 最小验证与主要指标

- 依赖已证明非空、命中任务实体且 Cartesian/FK 合同可信的共同前缀。
- 先运行少量 plus/minus/sham：
  - pulse 前 trace bit-identical；
  - actual-action first stage；
  - sham identity；
  - 全部 planned unit 按 ITT 保留。
- 主要指标：
  - pulse 对 actual action 的归一化 RMS 与方向服从；
  - pulse sign 对 task-entity minimum distance、接触、contact-associated motion 的
    signed cross-moment；
  - sham FPR、planted power、三任务方向一致性；
  - 安全与 P40-E2 完整分账。
- 失效：pulse 被 FIFO/safety 消除；只改变 proprio；只单任务成立；依赖 contact truth
  决定 pulse 时机；功效不足。

### 成本、风险与依赖

- 成本中等，paired 短 Episode，无参数更新。
- 风险：固定 primitive 是实验仪器，数据不能冒充 autonomous skill。
- 依赖：P40-E2、P39，以及 P50/P51/P52/P53 中必要前置。
- 与历史关系：直接修订 P47，明确禁止 post-treatment mediator selection。
- 禁止声明：不得直接授权 Replay、世界模型训练、Actor 或能力 benchmark。

## `R0001-P55`：逐实体操纵链能力成功证书

### 瓶颈证据与假设

- 来源：C。
- 当前 formal success 分别检查最终稳定 placement 与 Episode 累积双臂接触，没有把每个
  required object 的 arm contact、对象运动和最终目标状态绑定成同一证据链。
- P40-E2 已能按实体和机器人部位记录接触、双 pad grasp 与接触同期运动。
- 假设：增加 evaluator-private、report-only 的逐实体证书可消除底盘推物、接触无关对象、
  先放置后补接触、自由沉降或 reset settling 的潜在 false accept。

### 唯一主变量与范围

- 唯一变量：能力报告的附加授权谓词。
- legacy reward、termination、EpisodeResult、policy、训练和安全行为保持不变。
- 每个初始位于目标外的 required object 必须有该实体的 grasp-qualified arm contact
  associated displacement，并最终稳定在目标；articulation 需要 arm–handle contact
  associated joint motion。
- 保留整体双臂参与要求，但不强制两臂同时抓同一对象。

### 最小验证与主要指标

- evaluator-only 反例：
  - 底盘推入；
  - 双臂接触不同无关实体；
  - 先放置后补接触；
  - 自由沉降；
  - reset settling；
  - 接触后惯性运动；
  - 合法的单臂/双臂分工抓取放置。
- 主要指标：
  - false accept `0`；
  - 合法 fixture recall `1`；
  - required entity 证据覆盖 `100%`；
  - enabled/disabled runtime trace bit-identical；
  - manifest hash lineage 完整。
- 未来只有 qualified deployment 才同时发布 legacy 与 certificate-qualified 成功率。

### 成本、风险与依赖

- 成本低，无训练。
- 风险：过度规定操纵方式；fixture 必须允许两臂分别操作不同对象。
- 依赖：P40-E2。
- 与历史关系：不重复 P48。P48 是 deployment/provenance 防火墙，本项收紧成功语义。
- 禁止声明：不得称为能力或安全改善；它只减少错误能力声明。

## 创新 Agent 首选与分歧

- Agent A：`P50 -> P52 -> P47-E1`，认为先定位候选损失，再校验 FK，最后建立动作创新。
- Agent B：`P53a -> P49-E1 -> P54`，认为多帧几何融合最直接，数据设计次之，语言选择需等
  候选覆盖成立。
- Agent C：`P51 -> P53b`，`P55` 可作侧车；认为 Cartesian frame 错配比空候选更直接解释
  非空 cell 的零 arm contact。
- 共识：
  - 不修改世界模型 loss；
  - 不启动正式训练；
  - 不把候选非空当任务实体覆盖；
  - 不把 contact-associated motion 当 controlled motion；
  - 候选生成修订与 primitive 修订不能捆绑成一个因果实验。
- 主要分歧：
  - 先完成 P50 测量，还是先修复 P51 的解析坐标合同；
  - 候选生成优先采用纯几何多帧融合还是冻结基础视觉特征；
  - P55 是否应占用当前交互瓶颈的实施预算。
