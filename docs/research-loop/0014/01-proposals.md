# R0014 提案

## 生成过程

- 创新 Agent A 独立审计 P68 的 capture、相机渲染与 deadline 路径，并做不落盘、
  不计算 association classification 的单 Episode 探索性计时。
- 创新 Agent B 独立审计 P71、P57、P52、P66 与当前 B2/backend 约束，并做不落盘的
  运动链必要条件复算。
- 创新 Agent C 作为红队独立审计 P61/P72、P50 generator 与 P68/P71 的前置关系，
  主动寻找自证、评测泄露、伪独立样本、历史 provenance 破坏和不可执行预算。
- 三名 Agent 均先阅读 `AGENTS.md`，只读检查仓库和历史证据；完成前没有查看彼此输出，
  没有修改文件、启动训练或运行正式 cohort。
- 主 Agent 只做证据复核、稳定编号、重复合并和约束冲突标注，不按偏好提前筛选。
- 本轮新观点从 `R0001-P73` 递增；重提历史 P68 时保留原 ID 并使用实验后缀。

## 共识证据

1. P68 的 `inconclusive_budget_exceeded` 是执行失败，不是否定 association 假设：
   - 固定 24-Episode cohort 没有正式或 partial artifact；
   - 当前 selector relevance 仍未知；
   - 原 `0.80` support ratio、`18/24` 高关联门、`6/24` 低关联门没有被检验。
2. 当前 `CandidateAssociationBackend._observation()` 在每个 source observation 渲染
   segmentation，但 association 只消费 capture identity 对应的帧：
   - 每 Episode 约 996 次 source observation；
   - 每 Episode 只有 12–21 个 capture；
   - 固定 cohort 共 384 个 capture，而当前 treatment 约渲染 23,904 次
     segmentation。
3. 仅把 segmentation 改为 capture 时从 `MjData` source-state snapshot 渲染，设计上
   更精确，但 Agent A 的单 Episode探索探针只得到约 `1.028×` 加速：
   - eager：约 `31.41s`；
   - snapshot/capture-only：约 `30.56s`；
   - 27-field identity、candidate bytes、capture identity 与 segmentation sequence
     hash 均相同。
   该探索值已被看到，只能用于筛选和预算设计，不能作为正式确认性结果。
4. 同一探索探针显示通用 observation 每步渲染 head RGB、head depth、left wrist RGB、
   right wrist RGB；P68 serializer/candidate path 只消费 head RGB-D、标定与
   proprioception。禁用 P68 不消费的两路 wrist RGB 后，单 Episode 约
   `31.45s → 18.80s`，加速约 `1.67×`，已有 identity summary 与 candidate bytes
   未变。该值同样只作设计证据。
5. P57 已证明固定 P51 cohort：
   - `ever_bilateral_ready=0/36`；
   - `endpoint_bilateral_ready=0/36`；
   - 36/36 pair 两臂 command margin 都为负；
   - B2 起始 arm-to-preposition distance 为 `1.078–3.401m`；
   - 100 step actual-applied arm command budget 为 `0.349–0.433m`。
6. P52 已把 policy FK 误差排除到 aggregate P95 `4.65e-16m`；P66 已证明至少一个
   B1 action 会被 production predictor 因 robot-base/tea-table `356.9928N` witness
   拒绝。
7. Agent B 对 P51 36 pair 做的非正式、只读外包络复算得到：
   - 35/36 pair 的双臂 target 都至少有一臂超出固定 base 的严格链长外包络；
   - 70/72 arm 被排除；
   - living/dining/kitchen 分别为 11/12、12/12、12/12 pair；
   - 唯一未被外包络排除的 pair，以 `0.08m/s` 到 `0.10m` readiness 的名义直线下界
     仍约 `13.39s`，超过 B2 的 `5s`。
   这些值尚未经过正式 provenance、控制和独立实现验证，不是 R0014 结果。
8. P71 旧提案称“已提交 36 个 continuation snapshot”，但 P51 bank 实际只保存可重放
   continuation identity/hash，没有可直接加载的完整 `qpos/qvel/ctrl/queue/solver`
   snapshot；任何物理 witness 必须重放 prefix 并验证 identity。
9. P61 主 auditor 仍有两个已知完整性缺口：
   - 五个 exact-reference match flag 被计算，但未进入最终 fail-closed gate；
   - `planner_call_state_available` 与 role availability 仍有错误耦合；
   P72 只测量并收缩结论，没有修复 P61 producer。
10. P50 generator 中 `patch_valid` 是全局 `valid` mask 的 slice view，并原地执行
    `&=`；后续窗口会读取被前序扫描污染的 mask。
11. Agent C 的探索性离线 A/B 只增加 `.copy()` 后，旧 24 capsule 中：
    - 22/24 candidate-set hash 改变；
    - 16/24 candidate count 改变；
    - 22/24 selected candidate canonical identity 改变；
    - 3/24 empty/non-empty 状态改变；
    - 总 candidate 数 `39 → 36`。
    这些已观察数字只能证明缺陷可能重大，不能再作为结果前确认样本。
12. P50 顺序修复会产生新 candidate bank；不能把旧 P51/P57/P60/P66 的 target、
    continuation 或 safety witness 无条件解释为修复后 pipeline 的证据。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P73` | capture-identity source-state snapshot | evaluator instrumentation/正确性 | A+C | 待筛选 |
| `R0001-P74` | P68 head-RGBD observation projection | evaluator 执行优化 | A | 待筛选 |
| `R0001-P75` | precommitted extended-budget P68 rerun | 预算重冻结 | A | 待筛选 |
| `R0001-P76` | fixed-base bilateral outer-envelope certificate | 静态必要条件测量 | B | 待筛选 |
| `R0001-P77` | production-isomorphic bilateral clone path witness | 短 clone 存在性诊断 | B | 待筛选 |
| `R0001-P78` | P61-v2 verdict-integrity repair | evaluator 修复 | C | 待筛选 |
| `R0001-P79` | deterministic candidate-mask ownership correction | generator 正确性修复 | C | 待筛选 |

## `R0001-P73`：capture-identity source-state snapshot

### 瓶颈证据与假设

当前 segmentation 在 `_observation()` 内逐步渲染，虽与 source observation 对齐，但
绝大多数帧不被 P68 使用。假设：每个 source observation 只复制完整 `MjData` 到固定
环形缓冲，在 P68 的 `capture=True` 或 final seal 时才从匹配 identity 的 snapshot
渲染，可保持 segmentation 与全部正式 trace bit-identical。

### 唯一主变量与范围

- 唯一变量：segmentation 取证时机，从 eager source render 改为 source-state snapshot
  加 capture-time render。
- 仅修改 P68 专用 `candidate_association.py` 及测试。
- 不修改共享 backend、candidate generator、mapping、selector、动作、安全或物理。

### 最小验证

1. 使用预分配 `MjData` ring，以 `(timestamp_ns, sequence_id)` 绑定 source state。
2. 覆盖三任务、observation latency 1/2；保存 source state 后先推进 authoritative
   state，再渲染 snapshot，避免测试退化成“当前状态相同”。
3. 对 eager 与 deferred 路径逐 capture 比较 segmentation bytes/hash。
4. 比较 P68 既有 27-field identity、policy input、candidate-visible bytes、
   candidate bytes/order/hash、selected index 与 physical/action trace。
5. 断言 segmentation render count 等于 unique capture identity count，未 capture
   identity 不渲染。
6. 增加 ring eviction、latency warm-up 重复 identity、错误 identity fail-closed、
   snapshot 非推进的测试。

### 指标与判定

- 主要指标：
  - segmentation byte equality `100%`；
  - candidate byte equality `100%`；
  - observer off/on identity `100%`；
  - render count 从约 23,904 降至 384。
- 守护：
  - authoritative trace、queue、action 与 observation bytes 不变；
  - RSS `<8GiB`；
  - snapshot 不调用物理 step，不进入 action path。
- 全部 identity 和 capture-only 绑定通过：
  `accepted as capture-identity instrumentation contract`。
- 等价成立但固定 pair wall-time 改善 `<10%`：
  `rejected as standalone performance remedy`，但可保留为正确性重构。
- 任一 bytes/trace 漂移、渲染当前 state、ring 错绑或 snapshot 推进：
  `invalid`。
- GL 输出不稳定或 timing 波动无法判定：
  `inconclusive`。

### 成本、风险、依赖与边界

- 低成本，无训练；依赖 MuJoCo `mj_copyData`。
- 风险是重复 identity、队列保留窗口和 renderer state 未正确同步。
- segmentation 只进 evaluator sidecar；capture、frame 和 pixel 不是独立样本。
- 该提案本身不发布 association classification，也不授权 P68 正式重跑。

## `R0001-P74`：P68 head-RGBD observation projection

### 瓶颈证据与假设

通用 backend 每步生成四路相机，而 P68 的 policy-input 与 candidate generator 只消费
head RGB-D。假设：P68 专用 backend 对 baseline/treatment 都投影为 head RGB-D
observation，并以运行时访问守护禁止 wrist 消费，可在保持冻结 P68 estimand 与所有
identity 的前提下把固定 cohort 运行时间降到 30 分钟以内。

### 唯一主变量与范围

- 唯一变量：P68 replay 的 observation render 通道集合，从四路相机改为 head RGB-D。
- baseline 与 treatment 使用同一 P68 专用 projection；两者唯一差异仍是 segmentation
  observer。
- 不修改共享 `dual_arm_backend.py`、P50 generator、P50/P51 历史 artifact、
  selector、primitive 或 task。
- P73 可作为独立先决正确性重构，但必须分提交、分 microbenchmark，不能把两项性能
  增益合并归因。

### 最小验证

1. 静态调用图与运行时 access guard 证明 wrist frames 不进入：
   - P68 serializer；
   - `_input_failure()`；
   - candidate generator；
   - selector；
   - safety/action/runtime state。
2. 在不计算 association classification 的冻结 sentinel 上比较：
   - current full-camera；
   - P73-only；
   - P74 head-only；
   每项独立计时。
3. sentinel 必须覆盖三任务、observation latency 1/2，并在任务内选结果前定义的
   capture-count 上界 Episode。
4. 比较 head RGB/depth/calibration、policy input、candidate-visible、candidate、
   selected index、27-field identity、physical/action trace 与 segmentation bytes。
5. 增加 Episode 内 deadline；覆盖 baseline、treatment、support reconstruction 与
   artifact commit 前检查。
6. 用 sentinel P95 总和加至少 10% 余量预测 24-Episode runtime；在看到任何正式
   classification 前冻结正式预算。

### 指标与判定

- 固定 sentinel pair wall-time 降低 `>=30%`；
- 预测完整 cohort `<=27min`，给 30 分钟门留至少 10% 余量；
- 所有冻结 identity、candidate 和 segmentation bytes `100%` 相同；
- wrist access guard 零违规；RSS 与 artifact 门不变。

全部通过：

`accepted as P68 execution-ready`

若等价成立但改善 `<15%` 或预测仍超过 30 分钟：

`rejected`

改善 `15%–30%` 或 timing 方差越过预冻结界限：

`inconclusive`

任一输入、candidate、trace、segmentation 漂移或发现隐藏 wrist consumer：

`invalid`

### 成本、风险、依赖与边界

- 低到中成本，无训练。
- 风险是当前静态检查漏掉未来动态 wrist consumer；运行时 guard 必须 fail closed。
- 通道集合由 P68 schema 预先决定，不能按 task、seed、candidate、label 或结果变化。
- Episode 仍是样本单位；通过只表示执行就绪，不是 association 或能力证据。

## `R0001-P75`：precommitted extended-budget P68 rerun

### 瓶颈证据与假设

上一轮 30 分钟门可能只比实际完整运行少数分钟。假设：在不改变任何执行与科学条件时，
结果前把预算一次性重冻结为 45 分钟，并加入精确 deadline，即可完成 P68。

### 唯一主变量与范围

- 唯一变量：wall-time budget `30min → 45min`。
- 保持原始 eager segmentation、双 replay、四路 observation、固定 cohort、所有阈值
  和 artifact schema。
- 必须写入新的 R0014 output，不能补写 R0013。

### 最小验证、指标与判定

1. fake-clock 覆盖 baseline、treatment、support reconstruction 和写 artifact 前的
   deadline。
2. 超时不写 partial artifact 或残留 `.tmp`。
3. 正式 24/24 Episode 在 45 分钟内原子完成后，科学结论仍按原 P68 门独立判定。

- 完成：执行方案 `accepted`，P68-E2 另按冻结门判定。
- 仍超时：`rejected`。
- 看到 classification 后修改预算、复用旧内存值、替换 Episode 或产生 partial：
  `invalid`。
- 外部中断、GL 失败或资源争用：`inconclusive`。

### 成本、风险、依赖与边界

- 实现最低，计算浪费最高；约 47,760 control step 加大量无关渲染。
- 依赖稳定前台图形环境。
- 预算变化不改变样本，但可能掩盖根因；若 P74 可行，本项信息价值较低。

## `R0001-P76`：fixed-base bilateral outer-envelope certificate

### 瓶颈证据与假设

P57 证明现有 command support 不足，但没有区分 controller 不足与 target 在固定 B2 base
pose 下结构性不可达。探索复算提示 35/36 pair 可能被运动链外包络硬排除。假设：可以从
MuJoCo 模型运动树独立导出严格必要条件，并对旧 36 pair 建立不依赖 optimizer 的
fixed-base endpoint impossibility certificate。

### 唯一 estimand 与范围

- estimand：每个冻结 P51 pair 的 exact preposition target pair 是否满足固定 base 下
  的必要外包络条件。
- 不改变 base、candidate、target、phase、速度、IK、模型或 safety。
- 纯 evaluator-only 静态诊断；不运行 capability Episode。

### 最小验证

1. 绑定 P51 bank、P57 artifact、candidate bytes、target identity、base pose、
   primitive source 与 robot XML provenance。
2. 从 MuJoCo kinematic tree 独立提取肩点、link translations、site offset 与 joint
   ranges；不得硬编码探索结论。
3. 以 bank 中已提交 target 为被测对象，并用黑盒 `primitive_action()` target-capture
   校验当前 B2 endpoint identity。
4. 计算每臂
   `outer_margin = maximum_chain_length - shoulder_to_target_distance`。
5. 名义 command-time 只作描述量；除非另证 backend site-speed 上界，不称严格动态
   不可达。
6. 正控制为当前 MuJoCo grasp-center；负控制为外包络外 epsilon target；
   target/shoulder/link mutation 必须翻转相应 verdict。

### 指标与判定

- 主要指标：
  - `hard_outer_excluded_pair_count`；
  - 每任务 hard-excluded 数；
  - `maximum_possible_positive_witness_count`；
  - 描述性 nominal time/support margin。
- 守护：
  - target/base/model/source identity；
  - 两次重算 canonical bytes 一致；
  - target capture 与 bank 一致；
  - 正负控制与 mutation sensitivity `100%`；
  - 不读取 entity role、task success、destination 或其他语义 truth。

若 `>=30/36` pair hard-excluded 且每任务 `>=8/12`：

`accepted as fixed-base outer-envelope deficit evidence`

并直接关闭旧 P71 的 `>=24/36` positive-witness 解锁门。

若 `<=12/36` 且每任务 `<=6/12`：

`rejected`，结构几何不足以解释 P57，才考虑 P77。

中间范围：

`inconclusive`，P77 只处理未硬排除 pair。

artifact/source 漂移、target 不能绑定当前 primitive、运动树提取失败、控制不翻转或
硬编码 verdict：

`invalid`

### 成本、风险、依赖与边界

- 极低成本，CPU 秒到分钟级，无训练、无正式 rollout。
- 外包络只能给出充分的“不可能”证书；margin 非负不代表 joint-limit、collision、
  dynamics 或路径可行。
- Episode/pair 是样本单位；arm、constraint、mutation 不是样本。
- 这是对旧 P51 cohort 的确认性重分析，只覆盖当前 XML、三个任务和旧 candidate/base
  分布，不证明修复后 generator、未见布局或任务能力。

## `R0001-P77`：production-isomorphic bilateral clone path witness

### 瓶颈证据与假设

静态 endpoint IK 正例不能证明 5 秒内通过 latency、actuator scale、DLS servo、joint
limit、路径碰撞和 predictive safety 到达。假设：对 P76 未硬排除 pair，受约束的
arm-only action-sequence search 能找到并在 fresh clone 上盲重放 production-isomorphic
路径。

### 唯一主变量与范围

- 同一 B2-entry clone 上的 arm command generator：
  - baseline：当前 `primitive_action()`；
  - treatment：独立受约束 action-sequence search。
- base command 固定为零，gripper、candidate、target、phase 与 horizon 不变。
- evaluator-private search 不得进入 authoritative physics、policy、Replay 或训练数据。

### 最小验证

1. 从冻结 seed 重放 prefix，逐项匹配 continuation identity；不能把 hash 当 snapshot。
2. clone 完整 `MjData`、model randomization、servo target、action/observation queue 和
   runtime counter。
3. 搜索最多 100 个 B2 action，仅允许当前 B2 arm linear velocity 空间，并保留
   normalization、actuator scale、latency、joint velocity/limit 与 servo clipping。
4. 每个候选 action 通过真实 `backend.apply()` 和 production two-step predictor；
   不直接写 qpos、清空 queue 或绕过 safety。
5. 找到路径后在 fresh rebuilt clone 盲重放；验证器独立读取 MuJoCo site、joint range、
   raw contact 和 clearance，不信任 optimizer 自报 residual。
6. 正控制：小位移可达 target；负控制：外包络外 target；P66 anchor 必须仍触发
   `220N` 门。

### 指标与判定

- 同一步 bilateral readiness `<=0.10m`；
- first-ready step、endpoint residual、accepted command 数和 elapsed time；
- minimum joint-limit margin；
- path minimum robot-robot/robot-environment clearance；
- maximum forbidden contact-point force 与 safety margin。

至少 `24/36` pair 且每任务 `>=6/12` 有 fresh-clone 可复现完整路径：

`accepted as bounded-horizon path-existence evidence`

只有独立硬证书把最大可能正例数压到门以下时才可 `rejected`；optimizer 找不到路径只能：

`inconclusive_not_proven`

直接改 qpos/base、降低 safety threshold、忽略 latency/scale、污染 authoritative state、
或 P66 负控制失效：

`invalid`

### 成本、风险、依赖与边界

- 中等成本，无训练；必须在 P76 后，仅运行未硬排除 pair。
- 搜索不完备，只能可靠证明存在，不能由失败证明不存在。
- pair 是样本；branch、restart、step 与 arm 不是额外样本。
- 通过也不证明 candidate 属于正确实体、当前 primitive 能发现路径或任务成功。

## `R0001-P78`：P61-v2 verdict-integrity repair

### 瓶颈证据与假设

P72 证明 P61 主 auditor 的 exact-reference 和 planner evidence 仍可自证。假设：把冻结
重放的 exact flags 接入 fail-closed gate，并把 interface expressivity 与 independent
external-planner evidence 拆开，可形成 mutation-sensitive 的 P61-v2 evaluator。

### 唯一主变量与范围

- 唯一变量：P61-v2 verdict dependency graph。
- 不修改 candidate、runtime、planner、动作、task config 或旧 P61/P72 artifact。
- 纯 evaluator 修复，不能与 planner 能力实现或物理实验共享因果结论。

### 最小验证

1. 五类 exact-reference mutation 在 frozen replay mode 分别使报告 `invalid`。
2. role-only 与 contract-flag-only 反事实均不得产生 validated external planner。
3. 拆分 `direct_call_interface_expressive` 与 `external_planner_evidence`；后者允许
   `unknown`，不得从缺字段直接推断 planner 不存在。
4. 旧 14 项 mutation 加新增 planner controls 全部重复执行，canonical bytes 一致。
5. initial annotation 三任务与四类反事实的既有结果不漂移。

### 指标与判定

- exact mutation fail-closed `5/5`；
- planner anti-self-certification controls `100%`；
- initial annotation `3/3` 和 P68 dependency gate 语义不变。

全部通过：

`accepted as P61-v2 evaluator integrity repair`

任一 role/config 常量仍可自证 planner，或 exact mismatch 仍有效：

`invalid`

若新 schema 允许扩展，必须版本化，不能后验把旧 exact requirement 改成非关键。

### 成本、风险、依赖与边界

- 预计一分钟内、低于 1GiB。
- 风险是把无害扩展误判为漂移，因此 exact gate 只用于冻结重放模式。
- 不得宣称 whole-program planner absence、任务能力或泛化。

## `R0001-P79`：deterministic candidate-mask ownership correction

### 瓶颈证据与假设

`target_selection.py` 的 `patch_valid` 是 `valid` slice view，原地 `&=` 会让后续扫描依赖
遍历顺序。假设：只修正局部 mask ownership，就能使 raw support 和 final candidate
对结果前冻结的遍历顺序保持不变。

### 唯一主变量与范围

- 唯一变量：局部 patch 是否持有独立 copy。
- 阈值、扫描网格、geometry、merge、ranking、selector 全部固定。
- 这是 generator 正确性修复，不是 P68 association、能力提升或旧 artifact 重解释。

### 最小验证

1. 构造重叠深度窗口，证明旧实现的顺序依赖。
2. 修复后每次 probe 前后 parent `valid` bytes 相同，mutation count 为 0。
3. row-major、reverse row-major、column-major 三种预注册 traversal 的 raw-support
   multiset 与 final canonical candidate bytes 完全一致。
4. 对旧 24 capsule 发布 paired regression ledger；Episode 是单位，pixel/window/
   candidate 不是。
5. 不把已观察的探索性 22/24 hash drift 当确认性接受门；正式判定只依赖顺序不变量
   和 input ownership。

### 指标与判定

- 原缺陷 fixture 可复现；
- 修复后所有 traversal 的 candidate bytes 相同；
- parent validity mask mutation count 为零；
- 不引入 segmentation、task ID、entity role 或 evaluator truth；
- 历史 P50/P51/P57 artifact 原字节保持。

全部通过：

`accepted as deterministic candidate-generator correction`

修复后仍有遍历顺序影响：

`rejected`

不能证明 parent mask 不变、同时修改阈值/排序/selector、或覆盖历史 artifact：

`invalid`

### 成本、风险、依赖与边界

- 纯离线，预计数分钟。
- 接受后必须建立版本化的新 candidate bank；旧 P50/P51/P57/P60/P66 继续作为旧实现的
  历史证据，但不得冒充新 generator 基线。
- 因为会改变 P68 的被测 candidate，P79 与旧-bank P68-E2 不能放在同一因果运行中；
  是否先修 P79 是本轮筛选的核心依赖决策。

## 关键冲突与筛选问题

1. **先完成旧-bank P68，还是先修 P79？**
   - 前者保留已冻结 estimand并尽快回答旧 selector relevance；
   - 后者修复会改变 22/24 旧 candidate hash 的根本正确性问题，但使旧 P68/P71
     candidate lineage不再代表新 pipeline。
2. **P73 是否值得独立实施？**
   - 正确性更清晰、render count 大幅下降；
   - 探索计时却表明它不是 30 分钟预算的主要解法。
3. **P74 是否为合法单变量？**
   - 两路 wrist RGB 当前不被 P68消费；
   - 必须用静态和运行时 guard 证明没有隐藏 consumer，并避免修改共享 backend。
4. **P76 是否应优先于 P68？**
   - 它可能用极低成本直接解释 P57 的 0/36；
   - 但只覆盖旧 candidate/base cohort，而且 P79 可能使这些 target 失去新基线意义。
5. **P78 的位置**
   - 是明确、低成本的 evaluator 完整性修复；
   - 不直接改善物理能力，也不再是旧 P68 的必要前置。
6. 本轮不得把 P73+P74+P79 捆成一个不可归因性能/生成器改动；不得在看到 P68
   classification 后决定修哪一项。
