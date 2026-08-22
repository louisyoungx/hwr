# R0010 提案

## 生成过程

- 三个创新 Agent 分别从因果控制与物理交互、感知覆盖与数据效率、评测有效性与安全方向
  独立只读检查 `AGENTS.md`、R0008/R0009、当前源码、测试与正式产物。
- 三者完成前未查看其他创新 Agent 输出，未修改正式实现，未启动正式训练。
- 主 Agent 只做证据复核、稳定编号、重复项合并与多变量拆分，不按首选建议提前筛选。
- 历史观点重提保留原 ID 并增加实验后缀；本轮新观点从 `R0001-P56` 递增。
- 三名 Agent 对核心顺序形成共识：P51 的 closed-loop physical convergence 与 P50 的
  candidate coverage/identity 是正交问题；不得把它们捆绑成一个因果比较。

## 共识证据

1. 当前可信能力基线仍是 24 Episode、1,600 update、0 成功；Actor 未解锁，三任务无
   bilateral contact 或 controlled motion。
2. P51 的 144-cell 解析合同、48 个 legacy 反例与实际 `primitive_action` 集成门均通过，
   但正式 artifact 明确记录 `closed_loop_physics_executed=false`。
3. P52 的 918 个 terminal 得到 aggregate p95
   `4.652682298944613e-16m`、max `7.021666937153402e-16m`；厘米级 policy FK—plant
   tool-site mismatch 不是当前支持解释。
4. R0008 P41 smoke 的独立样本是 6 个 environment-seed pair，不是 12 个 branch；
   candidate 数为 `4/0/1/3/5/3`，全部 arm interaction 为 0。
5. 客厅 `(observation latency=2, action latency=2)` 有 18 个 keyframe 和完整
   1,655-step route，但 candidate set 为空；其余五个非空 cell 也未证明 candidate 属于
   task-relevant entity。
6. P41 只持久化 candidate canonical bytes；其中 acquisition 输入是 SHA-256，而原始
   policy-input bytes 只存在于运行时内存，历史 smoke 无法做可信逐 gate 离线复算。
7. 当前 merge 按 `frame_ordinal` 计算 `view_count`，而 latency warmup 允许重复
   `(timestamp_ns, sequence_id)`。重复视觉 identity 只能虚增 ordinal-view 支持，不能
   解释空集合。
8. 现有 P40 `contact_associated_motion` 是接触同期运动，不是受控运动；P41 safety 的
   target-contact intensity 聚合未按 robot part 排除 base contact。
9. 三名 Agent 一致反对现在恢复 P41 selector 正式对照、Replay 采集、Actor 或世界模型
   训练，也反对同时修改 candidate、phase、幅值、FK、gripper 与坐标旋转。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P51-E1` | fixed-candidate paired physical Cartesian convergence | 单变量闭环物理诊断 | A、C，B 支持正交执行 | 待筛选 |
| `R0001-P50-E1` | 12-cell immutable acquisition evidence capsule | report-only 证据采集合同 | B，C 支持 | 待筛选 |
| `R0001-P50-E2` | 离线有序候选漏斗与 unique-observation 审计 | report-only 测量诊断 | B、C | 待筛选 |
| `R0001-P50-E3` | evaluator-private candidate—entity association | report-only 身份诊断 | A、B、C | 待筛选 |
| `R0001-P56` | arm-qualified phase-resolved contact attribution | report-only 评测收紧 | C | 待筛选 |

## `R0001-P51-E1`：fixed-candidate paired physical Cartesian convergence

### 瓶颈证据与假设

- P51 只证明数学公式和实际 `primitive_action` 集成正确，没有执行 physics step。
- backend 仍包含 action latency、actuator scaling、Jacobian damping、关节速度裁剪和
  servo-error 限制；解析正确不保证真实 grasp-center site 朝目标收敛。
- P52 已排除 FK/tool-site 数值错配，因此可以隔离检验 P51 frame transform。
- 假设：在 candidate、初态、acquisition、B0/B1 前缀、phase、速度、gripper、backend
  与安全均固定时，当前 frame-fixed 公式相对 legacy 公式，会降低 B2 阶段真实双臂
  grasp-center site 到冻结 preposition target 的距离 AUC。

### 唯一主变量与范围

唯一 treatment 是：

```text
frame_fixed:
v_B = Rz(theta_A - theta_B) * clip_norm(2 * e_A, velocity_max)

frame_legacy:
v_B = clip_norm(2 * e_A, velocity_max)
```

- 两个 role 必须在同一 source commit、同一 runner 和同一 `primitive_action` 路径执行，
  不能用两个 checkout 对比。
- candidate 来自 treatment 前 acquisition，随后按 canonical bytes/hash、selected index
  和完整 record 固定；不得依据实体、contact、reward、success 或 treatment 结果纳入。
- evaluator-private MuJoCo site 只用于评分，不得进入 policy、candidate 或动作。
- 不改变 generator、selector、acquisition、B0～B7 phase、target、velocity cap、
  gripper、FK、IK、latency、安全或 termination。

### 结果前 candidate plan

采用两阶段冻结：

1. 设计提交只公开新 salt 的 SHA-256 commitment，同时冻结 seed derivation、eligibility、
   最大搜索量、样本量、指标、MDE、守护与判定顺序；
2. 在不知道 treatment 结果时，按 seed ordinal 做自然 latency rejection 和
   treatment-free acquisition/B0/B1 prefix，为完整
   `3 task × observation latency {1,2} × action latency {1,2}` 冻结 candidate bank。

eligibility 只允许：

- acquisition 无失败；
- candidate set 非空且冻结 selector index 有效；
- B2 起点 `abs(wrap(theta_B-theta_A)) >= pi/6`，确保 treatment 实际暴露；
- candidate bytes/hash、acquisition input hash 和 B0/B1 prefix state/trace 可重建；
- 不读取 B2 后结果、contact、entity、reward 或 success。

每 cell 最多检查 64 个 latency-matched acquisition。若无法填满冻结样本，整个设计标记
`inconclusive_design_infeasible`；不得减少 cell、放宽门或替换为已看过的 R0008 seed。

### 样本量分歧与待筛选问题

- 创新 Agent C 建议每 cell 2 个 pair，共 24 pair；`17/24` wins 的单侧 exact
  binomial p-value 为约 `0.032`，在真实 win probability `0.8` 时功效约 `0.91`。
- 创新 Agent A 建议每 cell 3 个 pair，共 36 pair；总 `>=24/36` wins 且每 task
  `>=6/12`，估计全局假阳性率约 3%，在 win probability `0.75` 时功效约 0.89。
- 筛选 Agent 必须在冻结前明确选择一个设计；不得先跑 24 个再按结果扩到 36。
- 一个 environment seed pair 是一个独立样本；两个 role、两臂、frame、step、contact
  point 与 physics substep 都不是独立样本。

### 主要指标、守护与判定

主窗口固定为 B2 的 100 个 control step，避免 target 切换、gripper 与接触混杂：

```text
d_t = 0.5 * (
  ||left_site_A - left_pre_target_A|| +
  ||right_site_A - right_pre_target_A||
)

normalized_AUC = mean_t(d_t / max(d_0, 0.05m))
```

一个 pair 的 `frame_fixed_win=1` 仅当：

1. `normalized_AUC_legacy - normalized_AUC_fixed >= 0.10`；
2. B2 末端 `d_fixed <= d_legacy`；
3. pair 完整且全部 hard guard 通过。

必须完整报告：

- 每 pair、arm、task、latency cell 的 B2 起点、末点、最小距离与 normalized AUC；
- 前 10 个实际 applied、非零 arm step 的 signed directional derivative；
- treatment 是否产生非零 action byte 差；
- B3～B6 如执行时的 arm/entity contact 只作描述性次要结果。

硬守护：

- pair 两 role 的 candidate、seed、初态和 B0/B1 prefix bit-identical；
- 首个 treatment step 除左右臂线速度 xy 外，其余 action 字段一致；
- arm action 非塌缩，action bounds 有效；
- supported stale action applied、severe collision、invalid force、P40 conservation
  violation 均为 0；
- safety、cap、gripper、phase、target、backend identity 不变；
- 缺失、正常 early terminal 或未激活 treatment 按 ITT 记 fixed 失败，不补 seed；
- hard safety failure 立即停止正式 run 并保留已完成 terminal；
- 禁止 efficacy/futility sequential peek。

### 成本、风险、依赖和失效

- 成本中低：24 或 36 个 eligibility/prefix run，加 48 或 72 个短 MuJoCo branch；无训练。
- 风险：冻结 candidate 可能是错误家具或空气中的几何点。因此只检验控制收敛，不检验
  perception、selector 或 task interaction。
- 依赖：P39、P40-E2、P51、P52；P56 只在接触进入判定时才是硬依赖。
- 以下任一使假设失败或证据无效：
  - 无法填满冻结 bank；
  - prefix/candidate identity 不同；
  - treatment 没有实际 action 差；
  - 未达到结果前选定的 win 门；
  - 效果只来自一个 task 或一个 latency cell；
  - evaluator-private site 泄漏到动作；
  - 任一 hard safety guard 失败。
- 通过时最多声明
  `accepted as paired physical Cartesian convergence evidence`；不得声明交互、抓取、任务
  成功、学习、泛化、deployment 或硬件安全改善。

## `R0001-P50-E1`：12-cell immutable acquisition evidence capsule

### 瓶颈证据与假设

- R0008 只覆盖每任务 `(1,1)` 与 `(2,2)`，未覆盖 `(1,2)`、`(2,1)`。
- action latency 会改变 acquisition 底盘实际执行，observation latency 会改变可见观测，
  两者是不同机制，不能只测对角 cell。
- 历史 artifact 只保留 input hashes，没有原始 RGB-D、标定、base pose、关节与
  self-mask 所需 serialized bytes。
- 假设：不改变 acquisition、candidate 或动作即可建立完整 12-cell、可离线重放的不可变
  输入证据；若不能，后续漏斗和实体关联均不可审计。

### 唯一主变量与范围

唯一变量是增加候选输入证据的持久化：

- `3 task × observation latency {1,2} × action latency {1,2}`；
- 每 cell 结果前固定 2 个新 environment seed，共 24 个 acquisition-only Episode；
- 自然 latency rejection 按预提交顺序填充，记录所有 rejected seed，不用 reset override；
- 原子保存 A1/A3 keyframe 与 A4 final input 的原始 serialized bytes；
- 每项记录 capture ordinal、`(timestamp_ns, sequence_id)`、完整 input SHA-256、
  candidate-visible observation subpayload SHA-256 与 byte length；
- acquisition 完成后停止，不执行 post-selection primitive。

### 最小验证与指标

- 每个 blob deserialize→serialize bit-identical；
- 离线调用未修改的 `generate_candidate_set`，canonical bytes/hash 与在线结果完全一致；
- 同 seed 重放的输入 identity 序列、bytes/hash、candidate bytes bit-identical；
- enabled/disabled capture 的 acquisition proposed/applied trace 与 candidate bytes
  bit-identical；
- 24/24 planned Episode 完整，按 12 cell 报告 keyframe、unique identity、unique
  candidate-visible payload 与 candidate count；
- 同 identity 对应不同 observation-derived payload 时 fail-closed；phase/history 导致
  完整 policy-input bytes 不同不应误判为视觉冲突；
- 不补 seed、不删除失败 Episode。

### 成本、风险、依赖和失效

- 成本低至中：24 个 995-step CPU acquisition，无训练、无 primitive。
- 风险：artifact 体积和控制时序干扰；应先在内存封存，acquisition 结束后原子落盘。
- 依赖：P29、P39、P41 acquisition 与 canonical serializer。
- 任一 planned bytes 缺失、hash 不符、离线不可重建、capture 改变 action/candidate，
  均为 `invalid`。
- 通过时最多声明
  `accepted as immutable acquisition evidence contract`；不得声明候选覆盖、感知泛化、
  数据效率、交互或任务能力改善。

## `R0001-P50-E2`：离线有序候选漏斗与 unique-observation 审计

### 瓶颈证据与假设

当前生成器依次执行：

1. center/ring validity；
2. prominence；
3. center depth spread；
4. patch support；
5. self-mask 后 support；
6. height/range；
7. planarity；
8. width；
9. connected-component merge；
10. nonzero normal；
11. `view_count>=2`；
12. top-64。

正式 artifact 只暴露最终候选数，无法定位损失。假设：空集合由一个或少数可重复阶段
主导；重复 observation identity 可能制造虚假多视角支持，但不会恢复空集合。

### 唯一主变量与范围

- 唯一变量：对 P50-E1 immutable bytes 增加不影响输出的离线描述性漏斗。
- 每个 anchor 只记录正式顺序下的 first rejection。
- merge 后记录 component、zero-normal、ordinal-view、unique-identity-view、
  `view_count>=2` 与 top-64。
- unique identity 固定为 `(timestamp_ns, sequence_id)`；depth/RGB hash 仅作载荷一致性
  审计。
- unique-identity merge 是 shadow，不进入正式 `CandidateSet`。

### 最小验证、指标与守护

- 每个 gate 有只命中一个 first-rejection 的合成边界 fixture；
- 每 frame：
  `enumerated anchors = sum(first rejection) + raw candidates`；
- merge 输入与 terminal component 完全守恒；
- 对 24 个 Episode 分析两次，report bit-identical；
- enabled/disabled 时正式 candidate bytes、顺序、score 与 selector index 完全一致；
- unique-view count 不得大于 ordinal-view count，shadow candidate 增加即实现错误；
- 按 Episode/cell 报告最后非空阶段、全部 funnel、component 数、view 差和最弱 cell；
- 若同一 cell 两个 seed 均在同一阶段损失至少 60% 的该阶段输入，只能标记“可重复描述性
  主导阶段”，不能解释移除该 gate 的因果效果；
- raw stage 已空、无重复 identity、无可重复主导阶段都必须原样报告，不得换 seed。

### 成本、风险、依赖和失效

- 成本低：P50-E1 后纯离线 CPU 分析。
- 风险：复制 generator 会逻辑漂移；必须共享冻结 gate 实现并以 output identity 防护。
- 依赖：P50-E1。
- 输入不完整、计数不守恒、正式 candidate 漂移或报告不确定时为 `invalid`。
- 通过时最多声明
  `accepted as candidate-funnel measurement evidence`；不得据此直接修改阈值、移除 gate，
  或声明候选、交互、学习与泛化改善。

## `R0001-P50-E3`：evaluator-private candidate—entity association

### 瓶颈证据与假设

- 五个非空 P41 smoke cell 仍全部 arm interaction 为 0；“非空”不等于任务实体覆盖。
- 当前 generator 只读 depth 几何，candidate 没有正式 entity association。
- observation queue 会返回延迟状态；不能用“当前 MuJoCo state”给旧 RGB-D 近似打标签。
- 假设：部分 Episode 属于“task entity 可见，但 final candidate 只对应错误家具”，这与
  “task entity 未进入视野”和“几何 gate 删除 task entity”是不同瓶颈。

### 唯一主变量与范围

唯一变量是 evaluator-private 事后实体标签：

- 使用与每个 observation identity 同物理时刻的 segmentation/geom mask 或等价快照；
- 对 raw patch 保存 entity pixel histogram，对 merged component 聚合 raw provenance；
- 角色至少分为 manipulated object、articulation、target container、floor/support、
  other furniture、robot、mixed/unknown；
- candidate bytes、顺序、score、selector 与动作完全不变；
- 私有 entity truth 不得反馈给 P50-E1/E2、P51-E1 的 bank 纳入或 policy。

### 最小验证、指标与守护

- 合成覆盖遮挡、mixed patch、重复 identity、错位 identity 和多实体边界；
- evaluator replay 的 policy input hash 必须与 immutable capsule 匹配；
- identity、camera calibration 或 state 对齐不一致即 fail-closed；
- enabled/disabled candidate 和 acquisition trace bit-identical；
- Episode 级报告：
  - evaluator-confirmed task-entity visible rate；
  - task-entity raw/component/final recall；
  - distractor-only final candidate rate；
  - 当前 top-1 的 task-entity association rate；
  - 空集合分型：not visible、visible/no eligible anchor、raw 后被 gate 删除、view gate 删除；
- mixed/unknown 不得强制归入有利类别；
- 不把 pixel、anchor、candidate 或 geom 当独立样本。

### 成本、风险、依赖和失效

- 成本中等，可与 P50-E1 同 cohort，但必须是单向隔离的独立 sidecar。
- 最大风险是 delayed-observation 标签错配和 evaluator leakage。
- 依赖：P50-E1；若要定位 raw/component 阶段则依赖 P50-E2。
- 时间对齐、mapping 或重放不确定，或标签影响 candidate/action，即为 `invalid`。
- 通过时最多声明 candidate identity measurement evidence，不得声明物体识别、语义理解、
  affordance、交互、任务成功或泛化。

## `R0001-P56`：arm-qualified phase-resolved contact attribution

### 瓶颈证据与假设

- P40 `contact_associated_motion` 只要求同一 period 内某机器人部位接触实体并发生位移，
  不是 controlled motion。
- P41 target-contact intensity 汇总所有 target-role edge，base contact 可作为“阳性”
  safety sample；R0008 客厅两个 pair 已出现 `base→football` 接触，但 arm interaction 为 0。
- P41 持久化 `_graph_summary` 删除 period/substep，无法区分 acquisition、base approach
  与 arm phase 接触。
- 假设：report-only phase×robot-part×entity ledger 能排除 base-only 假阳性，并明确
  arm-pad contact 与同期位移的声明边界。

### 唯一主变量与范围

- 唯一变量：增加 evaluator-only 输出；不改变 P40 分类、policy、action、安全、
  termination 或 legacy success。
- 输出 `phase × robot_part × entity` 的 onset、duration、force、impulse、pad
  qualification 与 same-period displacement。
- 保留 `contact_associated` 命名，禁止输出 `controlled_motion=true`。

### 最小验证、指标与守护

反例 fixture 至少覆盖：

1. base 推 required object；
2. arm 接触错误实体；
3. 单 pad 或单臂接触；
4. 合法双 pad arm contact 与同周期位移；
5. 接触后惯性运动；
6. acquisition 阶段接触；
7. reset settling；
8. 自由沉降；
9. 合法单臂/双臂分工。

要求：

- enabled/disabled runtime trace bit-identical；
- P40 conservation 精确；
- period ledger 可重放且完整持久化；
- base contact 永不进入 arm-qualified sample；
- reset settling 与 post-contact inertia 分开；
- 缺失 phase、part、entity、force 或 motion 时 fail-closed；
- control-step/period 计数守恒。

### 成本、风险、依赖和失效

- 成本低，以 fixture 与 report schema 为主。
- 风险：过度规定抓取形式、phase 边界过硬、把同期运动误称因果。
- 依赖：P40-E2、P41 phase identity；若 P51-E1 只以 tool-target convergence 为主门，
  P56 不是其硬依赖。
- 运行时序或 observation age 被改变、base 仍进入 arm-qualified sample、motion floor
  未冻结，均使证据无效。
- 最多声明 `accepted as phase-resolved arm-contact attribution evidence`，不得声明安全
  改善、受控操纵、接触因果或任务能力。

## 待独立筛选的关键问题

1. P51-E1 是否应先于正式 P50 cohort，还是与 P50-E1/E2 分文件并行实施、串行运行；
2. P51-E1 使用 24 pair 还是 36 pair，哪个在成本与功效之间更合理；
3. P51-E1 是否只执行 B2 主窗口，完全避免让 P56 成为硬依赖；
4. P50-E1 与 P50-E2 是否应本轮连续选择，还是先只接受 immutable capsule；
5. P50-E3 的 observation-time private truth 对齐能否在不保存完整 MuJoCo state 的前提下
   可靠实现；
6. P56 是否解决本轮主瓶颈，还是只应作为以后接触型主指标前的侧车。

## 明确拒绝的组合

- P51 rotation、P50/P53 generator 修改和 selector 修改放进同一物理实验；
- 用 contact/entity outcome 后验选择 fixed candidate、seed、task、latency 或阈值；
- 把 P41 的 12 branch、P51 的 arm/step、P50 的 frame/anchor/candidate 当独立样本；
- 用旧 P41 的 54-pair binary-event power 为 P51 连续距离 endpoint 背书；
- 把 candidate nonempty、几何关联、base-object contact、arm contact、
  contact-associated motion 与 controlled motion 视为同一成功链；
- 在 P50 漏斗中一边诊断一边移除 gate；
- 同时实现 P53a/P53b 后按结果择优；
- 在本轮诊断完成前恢复 P41 selector 正式对照、P47 pulse、Replay 采集或世界模型训练。
