# R0007 提案

## 生成过程

- 三个创新 Agent 分别从学习与控制、评测与安全、数据与计算效率方向独立只读检查
  `AGENTS.md`、R0006、当前源码、测试和正式结果。
- 三者未修改正式实现、未启动正式训练，完成前未查看其他创新 Agent 的结果。
- 主 Agent 复核了 P01 v4 Replay manifest：
  - SHA-256：
    `c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985`；
  - 24 个 source Episode；
  - 168 个 shard；
  - 每 source 7 个 shard；
  - 每 shard 16 transition；
  - 共 2,688 transition；
  - source task 分布为餐桌 6、厨房 6、客厅 12；
  - `proprioception` 为 17×37，`actor_proposal`、`executed_action` 为 16×16。
- 主 Agent 只做稳定编号、重复项合并、证据复核和文字归并，不按首选建议提前筛选。
- 延续观点保留原 ID；明确修订使用扩展 ID；本轮新观点从 `R0001-P43` 递增。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P32-E1` | 双重正交普通 Replay 条件信息门 | 离线数据诊断 | A | 待筛选 |
| `R0001-P40-E1` | 分类型 allowed-contact 力—冲量安全总账 | 安全测量合同 | B | 待筛选 |
| `R0001-P41-E1` | 单变量 RGB-D 几何目标选择 smoke | 无训练行为诊断 | A | 待筛选 |
| `R0001-P42-E1` | 同 task ID 反事实语言—目标绑定门 | 泛化评测合同 | B | 待筛选 |
| `R0001-P43` | 安全改写正例分层的执行头判别 | 离线训练诊断 | A、C | 待筛选 |
| `R0001-P44` | 严格双尾执行器迁移门 | 动力学泛化评测 | B | 待筛选 |
| `R0001-P45` | 等 transition 预算的连续 Replay 视图 | 数据采集诊断 | C | 待筛选 |
| `R0001-P46` | 50-update 原子子周期恢复 | 训练系统可靠性 | C | 待筛选 |

## 共同证据与禁止边界

- 当前完整 3D 负基线是 24 Episode、1,600 update、0 成功；Actor 从未解锁。
- P17 只证明 plant action 可控，不证明普通 Replay 条件可识别，也不证明 production RSSM
  使用动作。
- P36-E2 只接受为平衡能力 benchmark 合同证据；不存在 qualified deployment，本轮不能
  用 scripted policy、P01 v4 或放宽准入门制造能力数值。
- 普通 Replay 由 task-blind physical-salience selector 保留；selector 使用 successor
  motion、action innovation、安全和交互结果，任何离线正结果最多外推到当前 retained
  distribution。
- 24 个 source Episode 才是统计独立单位；不得把 168 shard 或 2,688 transition 当独立
  重复。
- evaluation 的宽随机化区间与训练支持域重叠，不等于严格 OOD；执行器增益 evaluation
  `[0.88,1.12]` 中有三分之一位于训练区间 `[0.96,1.04]`。
- 当前每个正式 task ID 固定绑定场景、对象和目标映射；未见语言改写不能单独证明语言
  grounding。
- allowed-contact 集合在现有 forbidden-force 扫描中先被跳过，零严重碰撞不等于已测量
  抓取对象、容器、支撑面和 articulation 的接触负载。
- 当前安全改写正例极稀疏：约 37/2,688 transition，只来自少量 source；任何分层训练必须
  以 source 为单位验证，不能靠重复 transition 伪造样本量。
- 任何提案不得延长 100ms validity、删除 latency 3、读取 future/latest observation、
  放宽碰撞或安全门槛、将 scripted action 当能力、读取正式成功标签训练，或按结果选择
  seed、fold、cell、target、权重、门槛。

## `R0001-P32-E1`：双重正交普通 Replay 条件信息门

### 证据、假设与结论边界

- 来源：A；延续 R0006 未实施草案，本轮重新筛选。
- production action-causality ratio 为 `1.0012791539504238`；旧普通 probe 点估计虽为
  `1.0647`，保守 p05 仅 `0.9116`。
- P16 已证明旧 state/state-action ridge 设计功效不足；P17 又证明 plant 本身可控。
- 假设：source-Episode 完全 out-of-fold、action 与 successor target 双重残差化后，
  `u_t = a_t - E[a_t|S_t]` 仍能预测
  `v_t = y_t - E[y_t|S_t]`。
- 正结果只表示当前 salience-retained Replay 在给定 pre-action visible state 后含增量
  条件预测信息；不称为 plant causality、production utilization、能力、安全或泛化改善。

### 唯一主变量与影响范围

- 唯一比较变量是共享 state baseline 上是否加入 OOF action-residual map。
- 只新增 CPU 离线 evaluator，不修改 Replay、模型、Actor、runtime、安全层或评测。
- 输入固定为 P01 v4 Replay；主 state 为原始 37-D `proprioception[t]`，action 为
  `executed_action[t]`。
- 主 target 为一步 16-D rate-like controllable visible-proprio innovation。
- configuration guard 为一步 17-D delta：12 joint position、2 gripper position、3 base
  pose；yaw 使用冻结 wrap。

### 最小验证

1. 按 task 分层 outer 3-fold，每折留出餐桌 2、厨房 2、客厅 4 个 source。
2. 每个 outer-train 内按 source 做 inner 2-fold；fold 由 source identity 和固定 hash
   唯一生成，不扫描分折。
3. inner OOF 分别拟合固定 ridge `m_a(S)` 与 `m_y(S)`；标准化、均值、scale、rank 和
   ridge 解只使用训练折。
4. residual map `B` 只拟合 inner OOF action/target residual；outer-test nuisance 由全部
   outer-train 重拟合。
5. 同报告运行：
   - controller context：current actor proposal、过去 4 步 proposal/executed action 和
     availability mask；
   - safety rewrite / no rewrite；
   - shard 首四步 / interior；
   - 三任务和全部 planned source；
   - rate 与 configuration target；
   - exact-pipeline zero residual、random target 和 empirical planted signal。

### 指标、失效、成本与依赖

- 主指标：
  - source-Episode 等权 `MSE_control/MSE_candidate >=1.05`；
  - Episode-block mean log-ratio bootstrap 95% 下界 `>0`；
  - 三任务 point estimate 分别为正；
  - candidate 绝对 target MSE 改善。
- 守护：
  - outer-OOF action residual 有效秩 `>=6`；
  - empirical planted power `>=0.80`；
  - zero-residual 和 random-target FPR 均 `<=0.05`；
  - controller context 后 ratio 仍 `>=1.02` 且下界 `>0`；
  - no-rewrite、shard interior 和 configuration target 不得消失。
- `accepted as retained-Replay conditional information evidence`：全部主门和守护通过。
- `rejected`：功效充分但主指标失败；收益只来自 rewrite/边界；controller history 消除
  信号；configuration target 不支持；任一 source 泄露或绝对误差恶化。
- `inconclusive`：rank/power 不足、visible lag 使 configuration target 无测量功效、
  artifact 缺失或 lineage 无法重建。
- 成本低，CPU，预计分钟至小时；不更新 production 参数、不启动物理 Episode。
- 通过后最多授权修订 P31；失败时应优先改善数据采集支持，不能直接换 action objective。

## `R0001-P40-E1`：分类型 allowed-contact 力—冲量安全总账

### 证据、假设与唯一主变量

- 来源：B；修订 R0006 `changes_required` 的 P40。
- allowed-contact 集合包含 floor/support、被操作对象、目标容器和 drawer/articulation；
  现有扫描在计算 forbidden normal force 前跳过这些接触。
- 假设：当前 `severe_collision_count=0` 可与较大的 allowed-contact 峰值力或冲量共存，
  因而现有安全总账存在测量盲区。
- 唯一主变量是新增 report-only 接触测量；不改变 policy、action、reward、白名单、
  220N forbidden 阈值、预测停止、终止或成功条件。

### 最小验证

1. 冻结互斥且完备的 geom 分类：
   floor/support、manipulated object、target container、articulation、forbidden。
2. 每 physics substep 先按无序 geom pair 汇总全部 contact-point normal force，再计算
   `ΣF_normal × Δt`；冻结 pair 去重与跨 substep 累加语义。
3. 固定测量 fixture 覆盖 pair 顺序、多个 contact point、missing 值、类别冲突和 timestep
   减半后的冲量一致性。
4. 使用既有 deterministic trace 验证旧 action、状态、结果、终止、success 和 forbidden
   collision 报告 bit-identical。
5. 首轮只作描述性内部一致性报告；220N 只能作为旧 forbidden 阈值参照，不能冒充真实
   硬件安全阈值。

### 指标、失效、成本与依赖

- 主要指标：各类别 Episode peak force、最大控制周期 impulse、累计 impulse、接触时长，
  以及 allowed peak 超过旧 220N 但 severe collision 仍为 0 的 Episode 数。
- 守护：旧 severe collision、safety intervention、成功率与动作完全不变；缺失力不得
  记为 0；每个 contact pair 必须唯一分类。
- `accepted as safety measurement contract evidence`：fixture、去重、timestep 稳定性、
  分类完备性和行为 bit-identity 全部通过。
- `rejected`：漏记/重复记账、类别歧义、预注册 timestep 容差失败或测量改变轨迹。
- `inconclusive`：MuJoCo API 无法稳定提供所需 substep force 或旧 trace 缺乏可重建输入。
- 成本低至中，CPU，无训练。
- 风险：MuJoCo contact force 对 solver、timestep 和接触模型敏感；未来硬门仍需材料与
  硬件标定。
- P41-E1 必须以本提案通过为前置。

## `R0001-P41-E1`：单变量 RGB-D 几何目标选择 smoke

### 证据、假设与唯一主变量

- 来源：A；按 R0006 要求将被拒绝的多变量 P41 拆为单变量。
- 当前 random RL exploration 不读取 observation/task；24 个 source 在三任务均没有合法
  双臂接触或受控运动，但有严重碰撞正例。
- 假设：固定速度、动作幅值、双臂耦合、轨迹 primitive、gripper schedule 和撤回逻辑，
  仅使用 deployment-visible RGB-D 选择可达表面点，会比 geometry-matched blind target
  selection 更常进入双臂可操作接触流形。
- 唯一变量是 target selection；不改变训练、reward、任务、世界模型、安全阈值或成功
  条件。

### 最小验证与指标

- candidate/blind 使用完全相同的低速 Cartesian primitive。
- blind target 在深度、左右间距、法向和可达性上与 candidate 匹配。
- candidate 只能读取 RGB-D、标定、本体和已执行动作；禁止语言、task/object/target ID、
  颜色语义、reward、success、仿真真值和专家轨迹。
- 在结果前冻结的未见 3D seed 上做 paired smoke；完整发布 supported 与 latency-3
  challenge，不作能力结论。
- 主要指标：每万 physics step 的 bilateral-contact Episode、至少 0.01m controlled
  object/articulation motion Episode 和三任务最弱分区。
- 守护：severe collision 为 0；安全干预 burden 不劣；左右臂接触均衡；动作有效秩和
  executed RMS 不塌缩；接触不得主要来自 floor、固定家具或支撑面。
- 失效：仅增加单臂/支撑面接触、无受控运动、只单场景改善、几乎不动换取低风险、身份
  泄露或退化为 scripted trajectory。
- 成本中等，配对 MuJoCo smoke，无训练。
- 依赖：P40-E1 通过；否则不允许为优化接触启动该候选。

## `R0001-P42-E1`：同 task ID 反事实语言—目标绑定门

### 证据、假设与唯一主变量

- 来源：B；修订 R0006 deferred 的 P42。
- 当前每个 task ID 固定绑定 scene/object/target mapping；现有 language holdout 仅为相同
  目标的未见改写，策略同时持有 task ID。
- 假设：真正使用语言的策略应在同一 task ID 和相同物理场景中同时完成 canonical 与
  swapped object-target assignment，并对 matched instruction 显著优于 mismatched。
- 唯一变量是 instruction 指定的 object-target assignment；模型、task ID、初态、对象、
  目标体积、视觉、动力学和 safety 不变。

### 最小验证与指标

- canonical/swapped 配对 reset，除 evaluator-private instruction/goal mapping 外，RGB-D、
  proprio、标定和 randomization hash 一致。
- evaluator 私有保存目标映射；policy 不接收 variant/mapping ID、目标坐标或成功标签。
- 先验证两个映射几何可容纳且静态稳定；scripted policy 只能验证可完成性，不能形成能力
  成绩。
- qualified deployment 后运行 matched/mismatched paired closed-loop。
- 主要指标：`min(S_canonical,S_swapped)`、matched 相对 mismatched 的 paired success
  差与置信下界、同 seed 两种 matched assignment 均成功比例。
- 守护：canonical 不回归；完整挑战、supported ledger、零严重碰撞、双臂消融、稳定保持
  和 safety burden 不变。
- 失效：映射通过 task ID/seed/path/color/额外 observation 泄露；两映射物理难度不等；
  swapped 不可完成；matched 与 mismatched 无差异。
- 成本：合同低，正式闭环中高。
- 依赖：qualified deployment、P36-E2、私有正式 salt；当前最多完成合同，不能验收能力。

## `R0001-P43`：安全改写正例分层的执行头判别

### 证据、假设与唯一主变量

- 来源：A、C 独立重复提出，已合并。
- P01 v4 retained Replay 约有 37/2,688 safety intervention transition，只来自少量 source；
  当前 sampler 显式分层 severe collision 和视觉监督，但没有 safety-positive 层。
- 三任务 action-execution recall 均为 0，intervention normalized RMSE 约
  `0.302–0.357`；该验证是 exploration Actor 解锁硬门。
- 假设：固定比例的 safety-positive source/window 分层能在不改变模型和 loss 的前提下，
  改善独立 holdout 上的安全改写与 executed-action 建模。
- 唯一主变量是执行头训练 batch 中 safety-positive window 的固定比例；首轮冻结视觉
  student 和 RSSM，只重训现有 safety/action-execution heads。

### 最小验证

1. 先精确审计 positive transition、window 和 source 数及任务分布。
2. P01 Replay 按 source Episode 做 outer split；不得让同 source 进入训练与选择/测试。
3. baseline 使用自然分布，candidate 结果前固定 25% safety-positive window；三优化 seed。
4. 两臂共享初始化、posterior feature、优化步数、任务/result reason/collision/visual
   supervision strata；positive 内按 task/source 均衡。
5. 现有独立 action-execution holdout 只在冻结 candidate 后使用一次，不训练或调参。

### 指标、失效、成本与依赖

- 主要门：三任务 recall `>=0.80`、PR-AUC `>=0.50`，候选相对自然分布三 seed 均改善。
- 守护：Brier `<=0.10`、intervention RMSE `<=0.15`、identity RMSE `<=0.05`、越界率 0；
  production action-causality、visual/proprio error、collision validation 不劣化；冻结模块
  hash 不变。
- 失效：任一任务/seed 不改善；收益由单个 source 或重复 transition 主导；负例或 identity
  action 回归；共享 backbone 实际变化。
- `inconclusive`：独立 safety-positive source 太少，无法形成无泄露且有功效的 split。
- 成本低至中，head-only CPU/MPS 更新，无物理 Episode；正式多 seed 世界模型训练不在
  首轮范围。
- 风险：正例 source 极少、容易过拟合；balanced holdout 与自然 Replay 分布不同。
- 若 P32-E1 在功效充分时失败，不应继续在同一 Replay 上扩大本候选训练预算。

## `R0001-P44`：严格双尾执行器迁移门

### 证据、假设与唯一主变量

- 来源：B。
- training `actuator_scale=[0.96,1.04]`，evaluation `[0.88,1.12]`；当前均匀 evaluation
  有三分之一样本仍落在训练支持域，宽区间不等于严格未见动力学。
- 假设：对相同初态仅把执行器增益置于严格低尾 `[0.88,0.96)` 或高尾 `(1.04,1.12]`，
  闭环成功率相对 nominal `1.0` 至少下降 10 个百分点。
- 唯一变量是 `actuator_scale`；policy、checkpoint、任务、语言、初态、相机、质量、
  摩擦、latency 和 safety 固定。

### 最小验证与指标

- 建立 nominal/low-tail/high-tail 三元配对 manifest。
- reset-only 和短 apply smoke 验证除 actuator scale 外 canonical hash 一致。
- 结果前运行 paired null/planted power；qualified deployment 后才运行正式闭环。
- 主要指标：`S_tail=min(S_low,S_high)` 与
  `Δdrop=S_nominal-mean(S_low,S_high)`；完整挑战和 supported ledger 分开发布。
- 守护：零严重碰撞、零过期动作应用；原 27-cell latency 权重、成功条件、双臂消融和
  安全门不变；失败与 unresolved fail-closed。
- 失效：tail 样本进入训练支持域、非 actuator 字段漂移、功效不足或无 qualified
  deployment；功效充分且 `Δdrop<10pp` 否定该主要瓶颈。
- 合同成本低、正式闭环成本高。
- 只检验执行器增益代理，不得外推为完整硬件 sim-to-real。
- 依赖：P36-E2、qualified deployment、部署冻结后的私有 salt。

## `R0001-P45`：等 transition 预算的连续 Replay 视图

### 证据、假设与唯一主变量

- 来源：C。
- 当前每 source 只保留 7×16 transition；selector 强制非重叠且使用 successor outcome。
- 同一连续数据切为短 shard 后，历史长 horizon ratio 曾降至 `0.160–0.400`；但恢复全部
  起点后的旧线性 probe 功效仍只有 5.8%，这是必须保留的反例。
- 假设：相同每 Episode 112 transition 原始预算下，outcome-blind 连续 shard 比 7 个
  salience-selected 离散 shard 保留更多条件动作—successor 信息。
- 唯一变量是 Replay retention topology；模型、loss、动作过程、安全、总 transition
  预算、更新数和视觉监督窗口数不变。

### 最小验证与指标

- 新 seed、三任务各 8 条、每条 128 transition 的普通随机 Replay，无训练。
- 每 source 构造等预算两视图：当前 7×16 selector，以及由预提交 hash 决定起点的连续
  112-transition block。
- 使用 P32-E1 的双重 OOF residual、source-Episode outer split 估计 rate/configuration
  target 条件信息；旧数据只作 smoke。
- 主要指标：两目标 aggregate OOF MSE 相对下降至少 5%，source-bootstrap p05 为正，
  三任务方向一致。
- 守护：每 source 恰 112 transition；视觉监督仍 2 窗；安全/交互事件保留率、task 与
  latency 分布完整报告；不使用 action shuffle。
- 失效：exact-pipeline 功效不足；只单任务改善；连续视图减少安全或交互证据；P32 类
  条件信息仍不可辨识。
- 成本低至中，约 3,072 MuJoCo step 加 CPU 诊断，无 production 更新。
- 依赖：先有通过实现与审计的 P32-E1 evaluator。

## `R0001-P46`：50-update 原子子周期恢复

### 证据、假设与唯一主变量

- 来源：C。
- 当前每周期采集 3 Episode、更新 200 次，只在周期末 checkpoint；历史中断曾丢失约
  90 个未持久化 update。
- P01 v4 最终周期 update 约 758.54 秒，而 checkpoint 约 1.91 秒。
- 假设：每 50 update 保存 trainer/RNG/sampler/metrics 原子 mini-checkpoint，可把最大
  重算从 199 降到 49 update，同时保持训练轨迹完全一致。
- 唯一变量是恢复持久化频率与原子提交；不改变 Replay、采样、模型、目标或更新顺序。

### 最小验证与指标

- fixture 20-update 周期，在 update 7/13、原子 rename 前后强制异常。
- 再用冻结 P01 Replay 做 100-update smoke。
- 恢复后与 uninterrupted 比较模型、optimizer、RNG、batch identity、metrics 和 manifest
  hash。
- 主要门：全部 failpoint 最终状态 bit-identical；最大重复 update `<=49`；无重复采集或
  重复记账。
- 守护：checkpoint wall time 开销 `<=2%`、瞬时额外磁盘 `<=1.5 GiB`；损坏或不完整
  snapshot fail-closed。
- 失效：任一参数/RNG/batch trace 漂移；恢复需重新采 Episode；写放大超过 5%；无法原子
  绑定 Replay manifest。
- 成本低至中，不启动正式训练。
- 结论只涉及有效预算保留和可复现性，不计作能力改善；不得挤占当前能力瓶颈诊断。

## 创新 Agent 首选与分歧

- A 首选 `R0001-P32-E1`：它能最低成本决定“同一 Replay 继续优化”还是“先换数据支持”。
- B 首选 `R0001-P40-E1`：它是不依赖 qualified deployment 的真实安全测量盲区前置。
- C 首选 `R0001-P43`：它直接针对探索 Actor 的 action-execution 硬门，且可先做
  head-only 低成本判别。
- A 与 C 独立重复提出 P43 类正例分层，增强其问题真实性，但两者均承认正例 source
  过少和过拟合风险。
- A 明确把 P41-E1 置于 P40-E1 之后；B 认为 P42-E1、P44 均需等待 qualified
  deployment；C 认为 P46 不应挤占能力实验变量。
