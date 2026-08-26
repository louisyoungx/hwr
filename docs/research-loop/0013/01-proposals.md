# R0013 提案

## 生成过程

- 创新 Agent A 从生产 safety predicate 同构证据与预测分支可观测性提出三项意见。
- 创新 Agent B 从 B0/B1 路径、action-latency 后的 plant action 与 candidate-conditioned
  counterfactual 提出三项意见。
- 创新 Agent C 作为红队主动反驳“单个 P60 拒绝等于通用 B1 routing 缺陷”，并从
  selector relevance、independent feasibility 与因果证据充分性提出三项意见。
- 三名 Agent 均先阅读 `AGENTS.md`，只读检查仓库、历史文档、实现与已有 artifact；
  完成前未查看彼此输出；没有修改文件、启动训练或运行正式物理实验。
- 主 Agent 只做证据复核、稳定编号、重复合并与约束冲突标注，不按偏好提前筛选。
- 本轮新观点从 `R0001-P66` 递增。

## 共识证据

1. P60 没有测到 phase-entry geometry estimand：
   - 只执行 2 个 latency-matched physical prefix；
   - 一个 candidate set 为空；
   - 一个在 B1、总 prefix step `1279` 被预测安全层拒绝；
   - selected Episode、B2 action、geometry measurement 均为 `0/null`。
2. 拒绝事件是 `action_rejected / predicted_severe_collision`：
   - authoritative physics 没有执行该危险动作；
   - safety 把动作改写为 hold；
   - actual severe collision count 为 `0`；
   - 因而不得称为实际碰撞、安全能力失败或 phase-entry geometry 结果。
3. 生产预测器：
   - 复制 authoritative `MjData`；
   - 在 clone 上执行最多两个 control step；
   - 每个 control boundary 检查当前 forbidden contact point 的最大 normal force；
   - 判据是单个 contact point `>=220N`，不是同一 geom pair 多点 force 求和；
   - 触发后丢弃 clone，并在 authoritative 分支执行 hold。
4. 当前事件只保存 `reason`，没有保存：
   - 预测 clone 的最大 forbidden contact point force；
   - 对应 geom/body pair；
   - 首次越阈 control boundary/substep；
   - 预测器实际收到的 delayed/scaled plant action；
   - P60 trace 中的 authoritative `physics_advanced`。
5. P60 trace 的 policy `proposed_action` 不是预测器的直接输入。正式 backend 会先执行
   actuator scale 和 action-latency FIFO，再把 plant frame 交给基础 safety 与 predictor。
   P60 拒绝 Episode 的 action latency 为 1。
6. P60 拒绝步的 policy proposal 为
   `(base_linear=0.12, base_angular=-0.1822977766101701)`；双臂命令为零。该信息不足以
   区分 translation、rotation、既有接触状态或 delay queue 中上一动作的贡献。
7. 当前测试通过 monkeypatch 强制 boolean violation，只证明拒绝后 hold 与
   `physics_advanced=false`，没有验证真实 force、pair、阈值边界、action lineage 或
   observer 不改变行为。
8. 反方历史证据：
   - P51 bank 有 44 个 latency-matched physical prefix、39 个 non-empty candidate；
   - 后续 36 pair、72 branch、7,200 control step 均无 safety intervention；
   - P51 到 P60 的相关 selector/backend/task 配置没有对应行为变化；
   - 因而单个 P60 event 当前更像新 seed/cohort 下的事件，不能外推为通用 B1 缺陷。
9. B0/B1 都以 selected candidate center 计算 heading；B1 在 heading error 进入窗口后
   同时旋转和前进。selector score 不含 task role、通行性或 swept clearance。
10. 对 P60 拒绝 Episode 的只读静态复算把 selected candidate 世界坐标落在
    `sofa_seat_collision` AABB 边缘内。这是 candidate-induced route 的假设证据，不是
    contact-pair 或碰撞归因。
11. P50-E4 只证明 evaluator mapping 可构造；P61 只证明有限静态 direct-call 信息缺口；
    两者都不能自动证明 selected candidate 与当前 microinteraction 相关。
12. 相关源码尺寸已接近门禁：
    - `dual_arm_backend.py`：727 行；
    - `formal_household_backend.py`：800 行；
    - 新诊断不得继续堆入 800 行核心文件，必须使用薄 hook 与独立模块。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P66` | production-isomorphic predictive safety witness | evaluator instrumentation 合同 | A+B | 待筛选 |
| `R0001-P67` | fresh-cohort predictive rejection recurrence | 新鲜前缀测量 | B+C | 待筛选 |
| `R0001-P68` | initial-microinteraction association stopping gate | evaluator-only association | C | 待筛选 |
| `R0001-P69` | same-state base-action component ablation | 预测 clone 因果诊断 | A+B+C | 待筛选 |
| `R0001-P70` | candidate-conditioned B0/B1 paired routing | 配对前缀诊断 | A+B | 待筛选 |
| `R0001-P71` | independent bilateral feasibility witness | 静态/短预测可行性下界 | C | 待筛选 |
| `R0001-P72` | P61 exact-reference anti-self-certification audit | 静态审计 | C | 待筛选 |

## `R0001-P66`：production-isomorphic predictive safety witness

### 瓶颈证据与假设

P60 只记录 boolean 拒绝。假设：可以在完全不改变生产 decision、阈值、horizon、动作队列、
authoritative physics 与 observation 的前提下，导出与生产 predicate 同构的预测分支
witness，并确定地解释 P60 anchor 为什么被拒绝。

### 唯一主变量与范围

唯一变量是 evaluator-only predictor observer 的启用状态。observer 必须记录：

- policy command；
- delayed/scaled plant action 与 queue source step；
- 基础 safety 后真正进入 predictor 的 action；
- 每个 control boundary 的最大 forbidden **contact point** normal force；
- 冻结 tie-break 后对应的 canonical unordered geom pair、body pair、robot/environment side；
- 首次越过 `220N` 的 boundary/substep；
- `predictive_trial_physics_advanced`；
- authoritative `physics_advanced`、最终 applied hold 与实际 severe collision count。

禁止把同一 pair 的多个 contact point 求和后与 `220N` 比较。pair-summed force 若保留，只能
作为名称明确的次要描述量。

本项只建立 instrumentation 与已知 anchor 的可重放合同，不用单个事件估计总体发生率。

### 最小验证

1. 结果前冻结后重放 P60 raw ordinal `35`：
   - 拒绝前 policy input、candidate bytes、policy proposal 与 authoritative trace hash
     必须与原 artifact 一致；
   - step `1279` 的 event/rewrite 必须复现。
2. 同一 replay 在 observer off/on 下运行：
   - policy、plant、applied action；
   - event 与拒绝 step；
   - authoritative `qpos/qvel/time`、queue、arm targets 与 runtime counters；
   - observation 与 final hard-stop；
   必须 bit-identical。
3. 正负 fixture：
   - 同 pair 两个 `130N` point：pair sum `260N`，仍不得拒绝；
   - 单点 `219.999N` 不拒绝，`220N` 拒绝；
   - multiple pair、equal-force tie、allowed pair、robot-self、world-world；
   - nonfinite force fail closed。
4. 拒绝事件必须证明：
   - trial physics 确实推进到冻结 boundary；
   - authoritative physics 未推进；
   - actual severe collision count 保持 `0`。

### 主要指标与判定

- production event 与 `max contact-point force >=220N` 一致率 `100%`；
- rejection witness 的 force/pair/boundary/action lineage 完整率 `100%`；
- observer off/on authoritative trace 与 state identity 一致率 `100%`；
- P60 anchor 拒绝位置、event 与 rewrite bit-identical；
- actual severe collision、invalid force、P40 conservation drift 均为 `0`。

全部通过：

`accepted as predictive-safety witness contract`

任一 witness 与生产判据不一致、observer 改变行为或 authoritative dangerous physics 前进：

`invalid`

无法复现历史 anchor：

`inconclusive_anchor_not_reproduced`

### 实施边界、成本与风险

- 建议薄 hook 位于 `dual_arm_backend.py`，默认 no-op；详细 force/pair 扫描与保存位于新的
  `src/hwr/adapters/mujoco/predictive_safety_diagnostic.py`。
- evaluator 分析与 app 分别放在新的 `src/hwr/eval/`、`src/hwr/apps/` 文件；不得向
  `formal_household_backend.py` 堆入实现。
- witness 不进入 `RuntimeStepOutcome.info`、`EpisodeEvent.details`、observation、history、
  selector、训练数据或 policy；只能由显式 evaluator observer 消费。
- 成本低：一个约 1,280-step 历史 anchor replay 与 synthetic tests，无训练、无 B2。
- MuJoCo solver force 只解释模拟器预测器的拒绝，不等同硬件碰撞或真实未来碰撞。

## `R0001-P67`：fresh-cohort predictive rejection recurrence

### 瓶颈证据与假设

历史 P51 的大量 prefix 没有 safety intervention，而 P60 只有一个拒绝 Episode。假设：
在结果前冻结的新鲜、平衡 task/latency cohort 中，B0/B1 预测拒绝可跨独立 Episode
稳定复现，而非单 seed 离群。

### 唯一 estimand 与范围

- 样本单位：独立 Episode。
- estimand：
  - `predicted_rejection_episode_rate`；
  - 首次拒绝 phase/step；
  - witness force margin 与 pair 分类；
  - task/latency 完整分账。
- 只运行 acquisition+B0/B1；不生成或执行 B2 action。
- P66 必须先 accepted。

### 最小验证与冻结 cohort

1. 新 salt 在实现和正式运行前 commitment。
2. `3 task × 2 observation latency × 2 action latency = 12 cell`。
3. 每 cell 取前 2 个 natural latency match，共 24 个 prefix：
   - candidate empty 作为完整 Episode 结果保留，不换 seed；
   - safety 拒绝只停止当前 Episode，不绕过安全；独立 Episode 可继续；
   - 每 cell latency match 上限、raw seed 上限与 wall/RSS/disk 预算结果前冻结。
4. 每个拒绝必须有 P66 完整 witness；非拒绝也必须记录最大检查 force 与 margin。
5. 旧 P60 anchor 只作回归，不计入 24 个新样本。

### 主要指标与判定

提案阶段不预设“拒绝多就是能力失败”。建议三向判定：

- 至少 3 个独立拒绝、覆盖至少 2 个 task/cell：
  `accepted as recurrent predictive-rejection measurement evidence`；
- 0/24 拒绝：
  `rejected recurrence hypothesis`；
- 1–2/24 或 cohort 不完整：
  `inconclusive`。

所有比例同时给出 Wilson 区间；contact point、pair、boundary 或 step 不得作为独立样本。

### 失效条件、成本与风险

- P66 不通过、witness 缺失、结果后换 salt、替换 candidate-empty seed、降低阈值、继续
  rejected action 或发生 actual severe collision：`invalid`。
- 成本低到中，最多 24 个 prefix，无训练。
- 风险：24 Episode 只支持冻结三任务与 latency cells，不足以声明泛化或 controller 根因。

## `R0001-P68`：initial-microinteraction association stopping gate

### 瓶颈证据与假设

P61 已冻结每个任务 reset 后首个合法 microinteraction role，P50-E4 已建立 evaluator-only
exact-geom mapping；但当前 selector 完全 role-blind。假设：P50 的 selected candidate 在多数
Episode 中与 initial microinteraction role 兼容，因而继续诊断 B1 route 具有优先价值。

### 唯一 estimand 与范围

- 对冻结 P50 24-Episode acquisition cohort做 evaluator-only association；
- candidate bytes、顺序、selected index 与 P50 artifact 必须 bit-identical；
- 不执行 B0/B1/B2；
- 不建立 action-chain positive control，不冒充完整 P64。

Episode 分类：

- `stage_compatible_selected`；
- `relevant_exists_but_distractor_selected`；
- `no_relevant_final_candidate`；
- `mixed_or_unknown`。

### 最小验证

1. 使用 P50-E4 role mapping 与 P61 initial microinteraction annotation。
2. 只消费与原 RGB-D observation identity 同步的 evaluator-private segmentation。
3. 结果前冻结 candidate-to-geom association：
   - source pixel/point 与 component/candidate 聚合口径；
   - mixed 与 unknown 上限；
   - 不以名称、颜色、task ID 或结果后 alias 选择 candidate。
4. synthetic fixture 验证：
   - relevant candidate；
   - relevant exists 但 selector 选 distractor；
   - mixed candidate；
   - unknown/background/site；
   - candidate empty。
5. instrumentation on/off 必须保持 P50 candidate artifact bit-identical。

### 主要指标与判定

- `stage_compatible_selected >=18/24` 且每任务过半：
  `accepted as initial-association stopping-gate evidence`，允许提高 B1 诊断优先级；
- `stage_compatible_selected <=6/24`：
  `accepted as selector-relevance stopping evidence`，停止 B1 行为修订；
- 其余：
  `inconclusive`。

### 失效条件、成本与风险

- sidecar 改变 observation/candidate/action；
- 结果后新增 alias 或改 association；
- target site/private stage进入行为；
- pixel、frame、candidate 被当成独立样本；
- candidate provenance 无法绑定。
- 成本低到中，无 post-acquisition action、无训练。
- 风险：只覆盖三个冻结场景，不证明识别或泛化；association 本身不证明可达或可交互。

## `R0001-P69`：same-state base-action component ablation

### 瓶颈证据与假设

P60 拒绝步同时有 forward 与 angular command，但 predictor 实际输入经过 delay/scale。假设：
在相同拒绝前状态中，B1 translation 是跨过 `220N` 的必要或充分即时因素，而不是 hold
状态已超阈、rotation 或组合项。

### 唯一主变量与范围

只改变预测 clone 的 base action 分量；所有分支从同一 authoritative snapshot 开始：

1. exact delayed/scaled plant action；
2. hold；
3. `v=0`，保留 exact `ω`；
4. 保留 exact `v`，`ω=0`。

arm/gripper、solver state、targets、model、horizon、threshold 与 boundary 全部固定。所有
counterfactual 只在 clone 中运行，不提交 authoritative physics。

### 最小验证与指标

- P66 必须先 accepted。
- exact branch 必须复现 production decision、force、pair 和 first crossing。
- 每分支报告最大 contact-point force、margin、pair 与 crossing boundary。
- 分类：
  - `state_already_unsafe`；
  - `translation_necessary`；
  - `rotation_necessary`；
  - `component_combination_required`；
  - `unresolved`.
- P60 anchor 只允许事件级结论：
  - exact `>=220N` 且 zero-linear `<220N`，支持“forward component 对本次拒绝必要”；
  - zero-linear 仍 `>=220N`，拒绝该局部假设。
- 若要外推 controller 根因，必须依赖 P67 至少 8 个拒绝、覆盖至少 2 个 task，且结果前
  冻结聚合门；本项不自行创建新 cohort。

### 失效条件、成本与风险

- 任何分支改变 queue、scale、非目标 action 分量、初始状态或进入真实 physics；
- exact branch 不复现；
- 根据结果选择分支或 contact pair；
- 把局部充分性称为完整路径根因。
- 成本很低；每个拒绝状态四个两步 clone。
- 风险：局部消融不覆盖此前 B1 累积路径。

## `R0001-P70`：candidate-conditioned B0/B1 paired routing

### 瓶颈证据与假设

B0/B1 直接由 selected candidate 决定 heading，selector 不检查 role 或可通行性。假设：
冻结 candidate identity 会改变 B0/B1 path 与预测拒绝，而拒绝不是所有 candidate 下都会
由同一 generic controller 产生。

### 唯一主变量与范围

从 acquisition 完成后的相同逻辑状态开始，只改变 candidate index：

- baseline：当前 selector index；
- control：由 `policy_rng_seed + candidate_set_sha256` 在其他 index 中均匀派生；
- 可选 null/hold branch 只能作为结果前冻结的机制对照，不得替代正式 paired control。

candidate generator、controller、400-step 总预算、latency、safety 与环境均不变。预测拒绝
仍触发 hold 并停止该分支；不执行 B2。

### 最小验证与指标

1. P66 必须 accepted；P68 未证明 selected relevance 时不得把更安全 alternate candidate
   称为能力改善。
2. 新 salt commitment；每 cell 取前两个 `candidate_count>=2` 的 natural latency match。
3. 每个 acquisition 独立复现，要求 branch 前 observation、queue、arm target、candidate
   bytes 与 trace identity 一致。
4. 样本单位是 paired Episode；至少 24 个 informative pair。
5. 主要指标：
   - paired `predicted-reject-free B2-entry`；
   - first rejection step、witness margin/pair；
   - 结果前定义的 swept base clearance。
6. 至少 `8/24` outcome discordance 且覆盖至少两个 task，才支持 candidate-conditioning；
   `<=2/24` 拒绝该假设；中间为 `inconclusive`。

### 失效条件、成本与风险

- 结果后挑“最安全” candidate；
- branch 前 state/queue 不同；
- 替换 candidate-empty seed；
- unsafe action 进入真实 physics；
- 把 pair 内 branch 或 control step 当独立样本。
- 成本低到中，最多 48 个 B0/B1 branch，无训练。
- 风险：所有 candidate 可能来自同一家具，导致无判别力；不评价 entity correctness。

## `R0001-P71`：independent bilateral feasibility witness

### 瓶颈证据与假设

P57 只证明现有 finite-horizon command support 不足；P62 因缺少 joint-limit、collision 与
safety 独立 feasibility witness 而延期。假设：在冻结 B2 entry base pose 下，多数
preposition target pair 至少存在一个满足双臂 joint limit、self/environment collision
与预测 safety 的同步配置。

### 唯一 estimand 与范围

- 使用已提交 36 个 P51 continuation snapshot；
- 固定 base/candidate/preposition target，不改变任何 task path；
- 独立多起点 IK/约束搜索只在复制状态中运行；
- 用 MuJoCo site truth 验证 endpoint；
- optimizer restart、arm、constraint 不是独立样本。

### 最小验证与指标

- 当前 tool pose 是正控制；
- outer-envelope 外 target 是负控制；
- FK 生成与验证实现必须独立，不能同源自证；
- 每 Episode 报告：
  - bilateral witness 是否存在；
  - 每臂 endpoint residual；
  - joint-limit margin；
  - minimum self/environment clearance；
  - predictor safety margin。
- `>=24/36` 且每任务 `>=6/12` 有 witness，才解锁 P62；
- 其他结果只表示“未证明”，不得因 optimizer 失败声称设计 infeasible。

### 失效条件、成本与风险

- 移动 base/candidate；
- 忽略碰撞、降低阈值或用同一 FK 实现自我验证；
- 把 restart 当样本；
- 把 endpoint witness 称为动态路径可达。
- 成本低到中，无正式 task action、无训练。
- 风险：逆解与碰撞搜索可能不完备，negative 结果难解释。

## `R0001-P72`：P61 exact-reference anti-self-certification audit

### 瓶颈证据与假设

红队发现 P61 虽已增加可翻转 fixture，但 `frozen_reference` 的 exact schema/signature
matches 未明确成为核心 verdict 的必要输入，且 `planner_call_state_available` 与 role field
存在耦合。假设：当前 accepted verdict 可能在 schema/signature 漂移时仍错误保持 accepted。

### 唯一主变量与范围

只对 P61 verdict dependency graph 做反事实静态审计：

- 独立扰动 exact schema/signature、role field、interaction field、destination field 与
  planner call state；
- 不修改 runtime、candidate、动作或历史 P61 artifact；
- 新结果只能补充或修正静态合同可信度，不改变能力基线。

### 最小验证与指标

- 每个冻结 requirement 必须有独立 perturbation 能改变相应 check；
- 任一 exact-reference mismatch 必须 fail closed 或有结果前明示的非关键理由；
- planner availability 必须有独立证据，不能仅由 role field 存在推断；
- mutation sensitivity `100%`；
- 重复审计 canonical bytes bit-identical。

全部通过：

`accepted as P61 anti-self-certification audit`

任一关键 mutation 无法翻转：

`accepted as residual P61 contract gap evidence`

### 失效条件、成本与风险

- 用固定 claim flag 代替 executable evidence；
- 修改历史 P61 artifact 或后验弱化 requirement；
- 把有限静态 audit 外推为 whole-program proof。
- 成本低，纯静态，无物理运行、无训练。
- 风险：与本轮首要 safety blocker 相关性较低，只适合作为低成本 sidecar。

## 分歧与依赖图

### 共识

- 不直接修改 B1 controller；
- 不降低阈值、不绕过 safety；
- P66 是任何 P67/P69/P70 的前置条件；
- 单个 P60 event 只可作 deterministic anchor，不可充当总体样本；
- 当前不启动 selector、Replay、Actor、世界模型训练或 capability evaluation。

### 未决分歧

- A/B 倾向 P66 后优先做 P69，以最低成本解释瞬时 action 分量；
- C 倾向先做 P68，防止更好地路由到错误 candidate；
- B 倾向 P67 收集跨 cell recurrence，再决定是否值得做 P69/P70；
- C 认为历史零拒绝对照使 P67 的期望价值有限，更建议 P71 先关闭 P62 positive-control
  前置条件。

### 依赖

- `P66 → P67`
- `P66 → P69`
- `P66 + P68 → P70`
- `P71 → P62`
- `P50-E4 + P61 → P68`
- P72 独立，但不直接解除物理实验门。

本文件冻结后，不得依据后续结果新增 proposal、改阈值、改样本单位或调整判定方向。
