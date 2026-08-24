# R0011 提案

## 生成过程

- 创新 Agent A 从实体可见性与候选覆盖方向提出行为型 segmentation 候选增强。
- 创新 Agent B 从因果控制与物理交互方向提出 P57 离线双臂可达性审计、P50-E3
  entity-conditioned coverage 与条件式 P58。
- 创新 Agent C 从评测有效性与安全方向提出 P50-E3 的 evaluator-only 安全版本，并明确
  P56 不是本轮 acquisition-only 测量的硬依赖。
- 首轮 A/C 与首批替代线程发生调度层长时间无响应；这些线程未返回正文，不计作有效创新
  意见。主 Agent 关闭异常线程后，串行获得两份不使用工具、基于冻结证据摘要的独立意见。
- 三份有效意见生成时均未查看其他创新 Agent 输出；未修改实现，未启动正式训练。
- 主 Agent 只做证据复核、稳定编号、重复项合并与约束冲突标注，不按偏好提前筛选。
- 历史观点重提保留原 ID；本轮新观点从 `R0001-P57` 递增。

## 共识证据

1. 当前可信能力基线仍是 24 Episode、1,600 update、0 成功；Actor 未解锁。
2. P50-E1 的 24 个 acquisition Episode 只封存 RGB-D、动态标定、proprioception 与动作
   证据，没有 observation-time entity truth。
3. P50-E2 得到 225 个 raw candidate、149 个 connected component、39 个 final
   candidate；5 个空 Episode 都已有 raw candidate/component，最后在
   `view_count<2` 阶段归零。
4. “candidate set 非空”不等于 task entity coverage；当前不能区分 task entity 未进入
   视野、未形成 raw candidate、在某 gate 被删除或 final 只覆盖 distractor。
5. observation latency queue 返回旧 observation，不能用后续或当前 MuJoCo state 给旧
   RGB-D 近似标注。
6. 主 Agent 的只读 API 与小型宿主探针确认：
   - MuJoCo `3.10.0` 支持 segmentation rendering；
   - segmentation 与 `head_depth` 同为 `192×256`；
   - 输出是 `(object_id, object_type)`，可包含 geom、site 与 background；
   - RGB 可见实体常对应 visual geom，而 binding 常只列透明 collision geom；
   - 因此 entity association 必须沿 geom→body 归一，并对 site、background、无映射、
     visual/collision 冲突保守处理。
7. P51-E1 已以冻结结果排除 frame transform 是零交互主要瓶颈：normalized-AUC 改善
   `+0.023449928237828013`，远低于 MDE `0.10`。
8. 三名创新 Agent 都没有授权当前启动 selector 正式对照、Replay、Actor 或世界模型
   训练。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P50-E3` | observation-time entity-conditioned candidate coverage | evaluator-only 测量诊断 | B、C | 待筛选 |
| `R0001-P57` | bilateral pre-contact reachability budget audit | 既有 artifact 离线测量诊断 | B | 待筛选 |
| `R0001-P58` | entity-hit/reachability-qualified B3 action-vs-hold | 条件式闭环物理诊断 | B | 待筛选 |
| `R0001-P59` | segmentation-assisted candidate augmentation | 行为型 candidate 修订 | A | 待筛选 |
| `R0001-P56` | arm-qualified phase-resolved contact attribution | 历史 report-only 评测收紧 | C 评估依赖 | 待筛选依赖状态 |

## `R0001-P50-E3`：observation-time entity-conditioned candidate coverage

### 瓶颈证据与假设

- P50-E1/E2 已建立不可变输入和完整漏斗，但没有 entity truth。
- 假设：task entity 的可见性与 candidate funnel 留存率显著低于“任意几何候选非空率”；
  部分 final set 只包含 distractor，部分 task-entity raw/component 在特定 gate 被删除。
- 该假设也允许被否定：
  - 若 task entity 大多不可见，主要瓶颈在 acquisition coverage；
  - 若可见 task entity 已稳定进入 final set，主要瓶颈转向动作支持链；
  - 只有 entity-conditioned 单一 loss stage 可重复成立，才支持下一轮提出一个 generator
    主变量。

### 唯一主变量与范围

唯一变量是启用或关闭 evaluator-private observation-time segmentation sidecar：

- segmentation 在原始 observation 完整生成后、进入 `_delay_observation` queue 前采集；
- 与同一原始 observation 的 `(timestamp_ns, sequence_id)`、head RGB、head depth、动态
  calibration 和 candidate-visible SHA-256 绑定；
- delayed policy observation 只能按 identity 取回历史 sidecar，不得读取当前/latest
  simulator state；
- sidecar 只进入独立 evaluator 与 artifact，不进入 `DualArmObservation`、policy input、
  acquisition state、candidate generator、score、selector、动作、安全、reward、
  termination 或 seed eligibility；
- enabled 与 disabled role 使用相同冻结 Episode、动作与 candidate 路径；不是两个不同
  candidate generator 的比较。

### 冻结实体映射

映射以 MuJoCo body identity 为主，不以 collision geom 名称替代 RGB 可见实体：

1. segmentation `objtype=GEOM` 时，`geom_id → geom_bodyid → body_rootid`；
2. binding 中 manipulated-object body 与 articulation body 为明确任务角色；
3. target-container 由绑定的 target-container geom 所属 body 闭包定义；
4. robot 由 robot root body 闭包定义；
5. floor/support 与 other-furniture 分开；
6. site、background、无名 geom、body 多角色冲突、不可映射 visual/collision identity
   一律进入结果前冻结的 `unknown` 或 `mixed`，不得乐观归入 task entity。

角色至少为：

- `manipulated_object`；
- `articulation`；
- `target_container`；
- `floor_support`；
- `other_furniture`；
- `robot`；
- `background`；
- `mixed`；
- `unknown`。

### 与 funnel 的关联

- visible：一个角色在同 identity 的 depth-valid segmentation 上有结果前冻结的最小像素
  支持；像素只用于 Episode 内测量，不是独立样本。
- raw anchor/patch：按冻结 patch footprint 统计 entity histogram；若首位角色未达到冻结
  purity，标为 `mixed`。
- component：必须沿组成它的 raw-candidate provenance 聚合，不允许只按 final center
  单像素贴标签。
- final：保留完整 raw/component provenance 后聚合角色。
- top-1 只作描述性 association；不执行 selector 对照。
- 每个 task 有两个 manipulated object；不得把“任意一个对象可见”偷换为“全部所需对象
  覆盖”。同时报告 any-object 与 all-required-object。

### 最小验证

1. renderer 单元测试证明 RGB/depth/segmentation 使用相同 camera name、模型状态、
   分辨率、时间戳与 frame index。
2. fixture 覆盖 visual geom/collision geom 同 body、geom/site/background、无名 geom、
   robot self-occlusion、遮挡、多角色边界、mixed patch、unknown、缺帧、错位 identity、
   重复 identity 和不同 payload。
3. sidecar 缺失、重复、identity/hash/calibration 不一致时 fail closed。
4. private truth isolation 用 AST/import graph 与实际集成测试双重证明；禁止自报布尔 flag。
5. sidecar enabled/disabled 的 policy-input trace、candidate canonical bytes、selected index、
   proposed/applied action、runtime terminal 与 physics trace bit-identical。
6. 同一 artifact 重复分析 bit-identical；实体映射、source 与场景递归 XML identity 进入
   manifest。
7. 复用 R0010 冻结 24 个 Episode seed 只允许建立 measurement contract 和 cohort
   诊断，不称未见分布泛化；不得按 sidecar 结果补 seed。

### 主要指标

独立样本单位为 Episode，按完整
`3 task × observation latency {1,2} × action latency {1,2}` 分账：

- any/all task-entity visible Episode rate；
- visible 条件下 task-entity raw/component/final Episode recall；
- task-entity 在各 anchor/component/ranking stage 的首次删除 Episode rate；
- distractor-only final-set Episode rate；
- selected-index association rate，仅描述性；
- 空集合分型：
  - `task_entity_not_visible`；
  - `visible_no_eligible_raw`；
  - `task_entity_raw_rejected_at_<stage>`；
  - `task_entity_component_rejected_at_view_count`；
  - `task_entity_retained_but_final_empty_invalid`；
- mixed/unknown Episode rate和像素支持只作质量守护。

像素、point、anchor、raw candidate、component、final candidate、frame、geom 与 control
step 均不得作为独立统计样本。

### 接受、拒绝与失效

`accepted as entity-conditioned candidate coverage measurement evidence` 仅当：

1. 24/24 planned Episode sidecar 与 capsule 完整；
2. identity、mapping、source、determinism、isolation 与 enabled/disabled identity 全通过；
3. entity-conditioned funnel 在 anchor/component/final 三层守恒；
4. mixed/unknown 不进入有利类别；
5. 报告全部 Episode、task 与 latency cell，不挑结果。

`rejected`：

- 合同有效，但预设的 entity-conditioned failure 假设不成立；例如 task entity 多数不可见，
  或 visible task entity 已稳定进入 final set。

`invalid`：

- 时间错配、当前状态回标、renderer mode 污染、mapping 漂移、private truth 泄漏、
  candidate/action 变化、计数不守恒、缺失 Episode 或后验改规则。

### 成本、风险、依赖和声明边界

- 成本中等：24 个 acquisition Episode 加 24 个 sidecar-disabled validation replay；无训练。
- 风险：共享 renderer mode 污染正式 RGB/depth；visual/collision geom 映射错配；site
  泄漏；mixed/unknown 被后验有利归类。
- 依赖：P50-E1、P50-E2；P56 不是 acquisition-only coverage 的硬依赖。
- 通过时不得声明物体识别、语义理解、affordance、交互、抓取、任务成功、泛化、安全改善
  或 deployment。

## `R0001-P57`：bilateral pre-contact reachability budget audit

### 瓶颈证据与假设

- 创新 Agent B 对 P51 final raw artifact 的只读审查认为：
  - B2 起点双臂平均距离约 `2.279m`；
  - B2 终点平均距离约 `2.123m`；
  - 只有 `10/36` pair 两臂都靠近，`26/36` 为一臂靠近、一臂远离；
  - B2 名义最大 Cartesian 路径为 `100/20×0.08=0.40m`；
  - preposition→contact target 位移至少约 `0.177m`，B3+B4 名义预算约 `0.095m`。
- 上述数字是创新提案证据，尚未由筛选 Agent 和正式 validator 签字。
- 假设：固定 primitive 在进入接触阶段前没有建立双臂同时可达的 preposition state；
  phase 切换时至少一臂仍明显超出剩余路径预算。

### 唯一主变量与范围

只新增对受保护 P51 final artifact 的 evaluator-only 离线可达性预算分析：

- 不重跑物理；
- 不改变 candidate、动作、速度、phase、gripper、安全或环境；
- 不把平均双臂距离作为唯一结果，必须逐臂和 pair-level 报告。

### 最小验证与指标

- 校验 P51 manifest、bank、terminal、source 与 artifact hash；
- 从每 step `tool_distances` 与 `applied_actions` 重算：
  - 每臂 `d0`、`d_end`、`min_distance`；
  - applied Cartesian path budget；
  - `preposition_budget_margin = path_budget - d0`；
  - `both_arms_improved`；
  - pair-level `max(left_distance, right_distance)`；
  - preposition→contact target displacement 与 B3/B4 nominal budget；
- fixture 覆盖双臂可达、单臂不可达、双臂不可达、action-latency 前缀动作和 early
  terminal；
- 重复分析 bit-identical。

主指标：

- `both_arm_preposition_ready_rate`；
- 每臂 budget margin；
- pair-level maximum-arm endpoint distance；
- `both_arms_improved_rate`；
- contact-transition budget margin；
- task 与 latency 分账。

### 守护、判定与边界

- 样本单位是 pair，不把 arm 或 step 当独立样本。
- 不读取 entity/contact/reward/success 后验，不改变 action bytes。
- artifact 不能从 raw step 重算、arm/target identity 缺失或依赖自报汇总则 `invalid`。
- 合同有效且绝大多数 pair 至少一臂负预算或 B2 末未 ready，则可
  `accepted as bilateral pre-contact reachability measurement evidence`。
- 若绝大多数 pair 两臂均有非负预算且 B2 末接近目标，则假设 `rejected`。
- 成本低、纯 CPU；风险是 command budget 不是严格动力学/碰撞可达集。
- 不得据此后验增加 phase、速度或改 gripper；不得声明接触、抓取或能力改善。

## `R0001-P58`：entity-hit/reachability-qualified B3 action-vs-hold

### 假设与唯一主变量

在 candidate 已由 P50-E3 确认命中 task entity，且 P57 证明 B3 起点双臂满足结果前冻结
readiness 的 continuation 上，现有 B3 arm command 相对 arm hold 会降低双臂到实体表面的
距离并提高 arm-qualified contact onset。

唯一主变量：

- treatment：B3 未修改 arm command；
- control：B3 arm linear command 乘 `0`，其余 action 保持一致。

candidate、prefix、base、phase、时长、velocity cap、gripper、安全与环境均固定。

### 前置、指标与停止条件

- P50-E3 与 P57 都接受前，本轮不允许实施。
- treatment 前 continuation 必须包含完整 MuJoCo state、servo targets、action/observation
  latency queues 与 policy history。
- P56 或等价 phase×robot-part×entity ledger 是 contact 指标的硬依赖。
- 主指标为双臂 pad-to-entity surface distance 的 pair-level max normalized AUC；
  contact onset 是守护/次要指标。
- coverage/readiness 不足时为 `inconclusive_support`，不得放宽 eligibility 或换 seed。
- private truth 进入 action、prefix 按后续 contact 选择、action latency 抹去 treatment、
  safety/P40 守护失败或未达到结果前 MDE 时拒绝或 invalid。
- 成本中等、短程 paired physics；最多声明
  `accepted as local B3 arm-action physical efficacy evidence`，不得声明抓取、selector、
  任务能力或泛化。

## `R0001-P59`：segmentation-assisted candidate augmentation

### 来源观点

创新 Agent A 主张将 class-agnostic segmentation sidecar 产生的实体候选合并进现有
candidate set，并在冻结未见场景子集用盲标 entity mask 测 `Recall@IoU>=0.5`、漏检实体
和 queue recall。

### 唯一主变量

- baseline：现有 depth-only candidate generator；
- treatment：额外把 segmentation sidecar 候选并入正式 candidate set；
- queue 容量、排序、策略、控制器、场景、种子和预算不变。

### 指标、守护和风险

- 指标：盲标子集 candidate recall、每帧漏检、queue recall；
- 守护：candidate count、false-candidate rate、queue overflow、latency、memory 和闭环
  success；
- future/action/task-answer/failure/evaluation mask 泄漏，或评测标注参与调参、训练、排序
  时 invalid。

### 当前协议冲突

该提案把 MuJoCo evaluator-private segmentation 直接进入正式 candidate generator，与
`00-context.md` 的单向隔离硬约束冲突，也把 measurement sidecar 和行为修订捆绑。当前
没有部署时可获得的等价 segmentation 来源、独立训练方案或未见闭环物理预算。

主 Agent 不在提案阶段删除该独立意见；筛选 Agent 必须明确判断它应：

- 因 evaluator leakage 与多阶段捆绑直接拒绝；或
- 仅保留为未来“deployable RGB-D segmentation model”新提案的启发，不能使用 simulator
  private truth。

## `R0001-P56`：当前依赖状态

- P56 能排除 base-only contact 和 phase 混淆，是任何以 contact yield 为指标的后续 P58
  的硬依赖。
- P50-E3 只测 acquisition-time visibility 与 candidate association，不执行 post-selection
  primitive，不使用 contact 指标，因此 P56 不是本轮 P50-E3 的硬依赖。
- 不应为“顺便完善评测”而把 P56 与 P50-E3 捆绑成一个实现或因果结论。

## 待独立筛选的关键问题

1. P50-E3 是否直接解决 R0011-C01，且 evaluator-private isolation 能否被独立证明；
2. 复用 R0010 的 24 个冻结 Episode 是否足够建立 measurement contract，还是需要新的
   result-blind cohort；两者允许的声明边界必须区分；
3. visual geom→body→role 映射、site/background 和 mixed/unknown 门是否足够保守；
4. P57 是否应作为低成本并行侧车，还是会把当前主瓶颈从 entity coverage 上移开；
5. P58 是否必须 defer，直到 P50-E3、P57 与 P56 前置证据成立；
6. P59 是否因 evaluator truth 泄漏和 measurement/behavior 捆绑而直接拒绝；
7. 本轮是否完全不训练，并把所有接受结果限制为 measurement evidence。
