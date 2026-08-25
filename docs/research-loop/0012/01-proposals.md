# R0012 提案

## 生成过程

- 创新 Agent A 从 evaluator-only geom/local-region mapping 方向提出两项意见。
- 创新 Agent B 从 entity-blind phase-entry geometry 与 controller transition 方向提出
  两项意见。
- 创新 Agent C 主动反驳“直接恢复完整 P50 entity funnel”，提出 stage contract、
  synthetic primitive positive-control 与 final-set-only 价值门。
- 三名 Agent 均先阅读 `AGENTS.md`，只读检查仓库与历史文档；完成前未查看彼此输出；
  没有修改文件、启动训练或运行正式实验。
- 主 Agent 只做证据复核、稳定编号、重复合并和约束冲突标注，不按偏好提前筛选。
- 历史 P50 mapping 修订保留 `R0001-P50` lineage，编号为 `R0001-P50-E4`；本轮新观点
  从 `R0001-P60` 递增。

## 共识证据

1. 当前可信能力基线仍是 24 Episode、1,600 update、0 成功，Actor 未解锁。
2. `R0001-P50-E3` 没有测得 entity coverage；它只证明 body-exclusive role table
   无法表达 kitchen articulated container。
3. kitchen 同一 `kitchen_drawer` body 内：
   - `drawer_handle_visual` 与 `drawer_handle` 是 articulation；
   - `drawer_bottom/front/back/left/right/divider` 是 target-container surface；
   - exact geom 可区分角色，body-exclusive role 不可区分。
4. 纯 body 扩张还会造成尚未触发的误标：
   - dining 的 `cup_holder`、`plate_holder` 与 `sideboard_top/body` 共用 body；
   - floor、墙和部分家具 visual geom 可共用 world body；
   - 因而不能把 exact task geom 的角色传播到整个 body。
5. collision-only exact geom 也不足：
   - manipulated object 的正常 segmentation 主要返回 `*_visual`；
   - kitchen handle 返回 `drawer_handle_visual`，不是透明 collision geom；
   - visual/collision identity 必须显式、结果前绑定。
6. 主 Agent 的启动期宿主探针确认：
   - segmentation 是 exact `(object_id, object_type)`；
   - kitchen handle visual、drawer container geoms 与两个 cleaner visual geoms 可分；
   - dining target sites 会以 `SITE` 出现，必须排除；
   - living 单个初始帧没有 task entity，不能用静态模型替代正式 acquisition visibility。
7. P57 已证明固定 P51 cohort 的 36/36 pair 从未同步 ready，且 36/36 pair 两臂
   initial command margin 均为负；这使“继续精化 entity attribution 是否优先”成为真实
   分歧，而不是默认答案。
8. 创新 Agent B 的只读探索复算（尚未正式签字）认为：
   - P51/P57 cohort 的 candidate-base range 与较差臂 B2 起始距离高度相关；
   - 35/36 pair 的至少一臂 preposition target 甚至落在保守 arm-chain 外包络之外；
   - 当前 selector 在多数 pair 已选到 set 内几何较优项；
   - 因而 phase-entry geometry 可能比 selector 更接近当前动作瓶颈。
9. 创新 Agent C 的只读粗 AABB 反例（不是正式 entity association）认为 living/dining
   selected candidate 很可能落在 sofa/chair distractor；这支持先做 final-set 价值门，
   但 AABB 不能替代 segmentation 或精确表面关联。
10. 三名 Agent 均拒绝当前恢复 selector、Replay、Actor 或世界模型训练，也不支持后验
    增加 phase 时长、速度或降低安全门。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P50-E4` | exact-geom first + explicit visual alias mapping | evaluator-only mapping 合同 | A | 待筛选 |
| `R0001-P60` | entity-blind phase-entry geometry certificate | 物理前缀测量诊断 | B | 待筛选 |
| `R0001-P61` | stage-conditioned interaction-target contract audit | 静态设计审计 | C | 待筛选 |
| `R0001-P62` | oracle-candidate primitive competence fixture | synthetic 物理能力下界诊断 | C | 待筛选 |
| `R0001-P63` | unnamed body-local region alias audit | evaluator-only 替代映射研究 | A | 待筛选 |
| `R0001-P64` | final-set-only exact-geom value gate | 条件式 entity sidecar 测量 | C | 待筛选 |
| `R0001-P65` | fixed-budget readiness-triggered B0 exit | 条件式行为对照 | B | 待筛选 |

## `R0001-P50-E4`：exact-geom first + explicit visual alias mapping

### 瓶颈证据与假设

- P50-E3 的 body-exclusive table 在 kitchen 必然冲突，并可能把同 body 的无关 geom
  错标为任务角色。
- binding exact collision geom 在正常 segmentation 中通常不可见。
- 假设：以 exact geom claim 为第一层、以结果前显式冻结的一跳 same-body visual alias
  为第二层，可以对三个正式场景构造完整、确定、保守且不发生角色传播的 evaluator-only
  role table。

### 唯一主变量与影响范围

唯一变量是 evaluator mapping key：

- 从 `body_id → one role`；
- 改为 `exact geom_id → one role`，并允许冻结的
  `visual geom → canonical collision/exact role geom` 一跳 alias。

body 只校验 alias 两端属于同一刚体，不再给同 body 的所有 geom 传播角色。同一 body
可承载多个角色；同一 exact geom 被两个角色声明仍为 `invalid`。

建议结果前冻结的最小 alias：

- living：
  - `toy_duck_visual → object:duck`
  - `toy_football_visual → object:football`
  - `storage_basket_visual → target_container`
- dining：
  - `dining_cup_visual → object:cup`
  - `dining_plate_visual → object:plate`
- kitchen：
  - `cleaner_yellow_visual → object:cleaner_yellow`
  - `cleaner_pink_visual → object:cleaner_pink`
  - `drawer_handle_visual → articulation:drawer`

本项只修改：

- `src/hwr/adapters/mujoco/entity_candidate_mapping.py`
- evaluator-only alias 配置或等价冻结常量
- `tests/test_entity_candidate_coverage.py`
- `src/hwr/apps/evaluate_entity_candidate_coverage.py` 的 mapping-only preflight

不得在本项中恢复 observation sidecar、运行 Episode 或修改 candidate/selector/action。

### 最小验证

1. 三场景 exhaustive geom preflight，报告每个 geom 的 exact claim、alias source、body、
   最终 role 与 table hash。
2. 所有冻结 alias 必须：
   - 两端存在；
   - 同 body；
   - 一跳；
   - 无循环、链式或跨 body；
   - 一个 source/target 不产生角色歧义。
3. synthetic fixture 覆盖：
   - 同 body 不同 geom、不同 role 合法；
   - 同 exact geom 双角色失败；
   - 缺失、无名、跨 body、链式 alias 失败；
   - site/background/未知 object type 保持 unknown。
4. 负守护：
   - `sideboard_top/body` 不得成为 target-container；
   - wall/rug 不得成为 floor-support；
   - `drawer_handle_visual` 不得成为 target-container；
   - drawer wall/bottom 不得成为 articulation。
5. 固定 scene/camera/reset 探针只验证 observed task visual geom 有唯一 mapping；不报告
   visibility rate。
6. 重复构表 canonical bytes/hash 必须 bit-identical。

### 主要指标与判定

- 3/3 正式场景 preflight 通过；
- 冻结 alias 全部通过；
- exact-geom role conflict `0`；
- unrelated same-body geom 误标 `0`；
- 预注册 task-visible visual geom unknown `0`；
- table deterministic；
- source/XML/binding/alias provenance 完整。

全部成立：

`accepted as exact-geom evaluator mapping contract`

任一场景或负守护失败：

`rejected_design_not_expressive`

source、alias、identity 或 determinism 不可审计：

`invalid`

### 失效条件、成本、风险与依赖

- 失效：
  - 根据 candidate/coverage 结果追加 alias；
  - 使用角色优先级覆盖 exact conflict；
  - 任意 body 角色传播；
  - 把 target site 或空 container interior 当可见实体；
  - preflight 通过就冒充 entity coverage evidence。
- 成本：低，静态与固定快照 preflight，分钟级，无训练。
- 风险：显式 alias 有场景维护成本；XML 名称存在但语义漂移；alias 若进入行为路径会泄露。
- 依赖：P40 exact-contact role resolver、P50-E1/E2 provenance、递归 XML identity。
- 泄露防护：
  - alias 只能由 evaluator mapping/app/tests 导入；
  - AST/import graph 禁止 candidate、policy、backend action 路径引用；
  - 不向 `DualArmObservation` 写 role、geom ID 或 sidecar handle；
  - 不读取 seed、frame、candidate、selected index、task stage 或结果。

## `R0001-P60`：entity-blind phase-entry geometry certificate

### 瓶颈证据与假设

- P57 证明 command support deficit，但没有区分底盘入场错误、arm workspace 外目标和
  当前工具位姿离 target 过远。
- 现有 primitive 固定按步数切换 phase，而不是按几何 readiness 切换。
- 假设：B2 入场时 selected candidate 派生的双臂 preposition target，在大多数新鲜
  prefix 上不满足与 entity identity 无关的必要几何条件；这比“B2 时间不够”更早。

### 唯一变量与范围

只新增 geometry evaluator：

- 不改变 candidate、selector、phase、动作、速度、gripper、安全或环境；
- 不读取 segmentation、geom/body/site identity、task entity、contact、reward、success
  或 B2 outcome；
- 新 cohort 只运行 acquisition+B0/B1，在 B2 入口停止。

历史 P51/P57 复算只能作为探索性 sanity check；确认性结论必须来自结果前冻结的新 salt、
新鲜 `12 cell × 3 pair = 36 pair` prefix-only cohort。

### 最小验证

1. 从 immutable candidate、base/tool pose 与 P51 target formula 重建：
   - `candidate_base_range_m`
   - `candidate_heading_error_rad`
   - B1 residual command
   - 每臂 shoulder-to-preposition distance
   - arm-chain conservative outer reach margin
   - `0.40m + 0.10m - tool_to_preposition_d0` nominal B2 support margin
2. P52-bound robot model与 link length identity 必须一致。
3. 输出显式 `role_order=["left","right"]`，不得依赖 JSON map 顺序。
4. fixture 覆盖 outer-envelope 之外、包络内但不能证明可达、单臂失败、双臂通过、非有限值、
   incomplete prefix 与 role-order serialization。
5. 36/36 pair、12/12 cell、每任务 12 pair 完整报告。

### 主要指标与判定

pair 是样本单位；arm、step、candidate、cell 不是独立样本。

`bilateral_phase_entry_ready` 要求同一 B2 入口两臂同时满足全部**必要**几何条件。

建议门：

- `phase_entry_geometry_deficit_supported`：
  - ready `<=6/36`；
  - 至少 `30/36` pair 两臂均违反 outer-reach 必要条件；
  - 每 task ready `<=4/12`。
- `phase_entry_geometry_deficit_rejected`：
  - ready `>=24/36`；
  - 每 task ready `>=6/12`。
- 其余：`diagnostic_inconclusive`。

合同有效需 raw reconstruction、provenance、finite、bounds、完整性、determinism 和分账全过。

### 失效条件、成本、风险与依赖

- 失效：用 B2 outcome 选 seed/门；模型漂移；prefix 不完整；结果后换 cohort；把外包络内
  误称为 IK/collision reachable。
- 成本：低到中；36 个 acquisition+B0/B1 prefix，无 B2、接触或训练。
- 风险：几何必要条件不是充分条件；candidate 可能任务无关。
- 依赖：P50 immutable candidate、P51 target contract、P52 FK agreement、P57 readiness。
- 允许声明仅为 phase-entry necessary-geometry measurement，不是 task interaction 能力。

## `R0001-P61`：stage-conditioned interaction-target contract audit

### 瓶颈证据与假设

- “task entity hit”不等于“当前阶段可行动目标”：
  - kitchen 要先开 drawer，再放两个 cleaner；
  - living/dining 各有两个 required object；
  - selector 一次只选一个 candidate；
  - 当前 candidate schema 不含 stage/interaction role；
  - 固定 B0–B7 primitive 被统一应用。
- 假设：至少一个正式任务存在
  `required transition → candidate representation → primitive interaction type`
  不能唯一对应的结构缺口；若不先定义它，entity coverage 指标会混合语义相关、阶段正确
  与 primitive 兼容三件事。

### 唯一变量与范围

本项是静态穷举审计，不运行 MuJoCo Episode、不修改 policy，也不把 task ID 或 success
truth 写入正式行为。

对三个 task 枚举：

- 初始状态；
- 必须满足的有序状态转移；
- 每阶段允许与禁止的 interaction role；
- 当前 candidate schema 可表达的信息；
- 当前 primitive 可造成的状态变化；
- “完整任务 target”与“单次 controlled-interaction microbenchmark target”的边界。

### 最小验证、指标与判定

主要指标：

- required transition 总数；
- 当前 candidate schema 可唯一表达的 transition 数；
- 一个 entity role 对多个 stage 的一对多歧义数；
- primitive 与 required interaction type 不兼容数；
- 无法在不读取 private task state 下定义的 transition 数。

三任务与 transition 是穷举设计项，不做显著性检验。

- 若每个 proposed measurement 都明确限定为一个 microinteraction，且 role/primitive 唯一：
  假设 `rejected`。
- 若存在至少一个未消解的一对多或 primitive mismatch：
  `accepted as interaction-contract gap evidence`。
- 若审计依赖隐含专家动作或未版本化配置：`invalid`。

### 成本、风险、依赖与泄露防护

- 成本：极低，纯静态，只读分析。
- 风险：只能证明表示/合同缺口，不能证明物理执行失败。
- 依赖：task config、binding、success predicate、candidate schema、primitive。
- 泄露防护：审计结果只能限制 evaluator/microbenchmark 定义；不得用 task stage/private
  success state给正式 policy 选目标。

## `R0001-P62`：oracle-candidate primitive competence fixture

### 瓶颈证据与假设

- P57 在真实 P51 cohort 上为 0/36 bilateral readiness、72/72 arm negative margin。
- evaluator mapping 本身不会改善动作。
- 假设：当前固定 primitive 即使收到已知正确、无歧义、几何上满足必要条件的 synthetic
  candidate，也不能可靠建立双臂 pre-contact readiness；若成立，继续扩展 entity funnel
  不是当前最有价值的工作。

### 唯一变量与范围

只新增 synthetic diagnostic fixture：

- 当前机器人模型与 `primitive_action` 原样运行；
- 注入结果前冻结、经独立 FK/outer-workspace 检查的 candidate；
- 不使用正式场景 task entity、语言、reward、success、contact truth 或训练；
- 不修改 phase 时长、速度、target formula、IK、gripper 或 safety。

### 最小验证

1. 冻结少量 candidate center/height/width 设计点与正负例。
2. positive control 必须由独立 P52-consistent geometry check 支持；否则 fixture invalid。
3. 运行 B0–B4，逐步记录 proposed/applied action、左右 tool-target distance、phase entry、
   safety rewrite 与 terminal。
4. 同时包含：
   - 双臂必要几何成立的 positive controls；
   - 单臂超界；
   - 双臂超界；
   - heading/base range 边界。
5. 不把 control step 或左右臂作为独立样本；完整报告冻结设计点。

### 主要指标与判定

- 每设计点 `ever_bilateral_ready`；
- B2 entry/exit 左右距离；
- actual-applied command budget 与距离下降；
- B3 entry readiness；
- action bounds、安全 rewrite、碰撞和 terminal 守护。

- 所有 positive controls ready，负例按预期失败：
  `primitive_necessary_competence_supported`。
- positive controls 系统性不 ready：
  `primitive_necessary_competence_rejected`。
- positive control 本身未独立可达或 fixture 需改 primitive 才通过：`invalid`。

### 成本、风险、依赖与声明

- 成本：低，短程 synthetic MuJoCo，分钟级，无训练。
- 风险：synthetic pass 不覆盖障碍、实体 identity、接触动力学或真实任务；synthetic fail
  只证明当前 primitive 下界不足。
- 依赖：P52 FK、P57 readiness、当前冻结 primitive 与 safety。
- 不得把 oracle candidate 或 synthetic success 称为家务能力。

## `R0001-P63`：unnamed body-local region alias audit

### 瓶颈证据与假设

- P50-E4 的显式 alias 最保守，但依赖人工维护场景名称。
- kitchen handle visual 与 collision handle 在 body-local geometry 中高度重合。
- 假设：对 unresolved visual pixel，仅用 observation-time body-local geometry，可以在
  不读取名称的条件下唯一匹配 canonical role geom；区域外为 unknown，多个区域重叠为
  mixed。

### 唯一变量与范围

唯一变量是 unresolved visual pixel：

- baseline：`unknown`；
- treatment：body-local unique-containment role。

禁止 nearest-role、后验 dilation、target site、semantic name 或 role priority。exact geom
仍优先。该项只做固定扫描 audit，不进入正式 24-Episode P50 cohort。

### 最小验证与指标

1. 以 P50-E4 explicit alias 仅作离线 oracle；算法不可读取 alias 名称。
2. 三场景固定 camera/reset，加 kitchen drawer qpos `{0,0.21,0.42}` 扫描。
3. 同 identity depth 反投影到 world，再用同 identity body transform 转 local。
4. synthetic 覆盖角色体积重叠、边界、遮挡、错 body pose、current/latest pose 回标和
   invalid depth。
5. 指标：
   - 冻结 alias role consistency；
   - task-role pixel precision；
   - unrelated geom 误标；
   - overlap→mixed；
   - outside→unknown；
   - drawer translation 前后稳定性。

只有 precision `1.0`、unrelated false positive `0`、所有 overlap/outside 守护通过才可
`accepted as local-region alias feasibility evidence`；否则 `rejected`。像素只是合同质量
检查，不是能力样本。

### 成本、风险、依赖与失效

- 成本：中低，宿主渲染扫描，无正式 Episode、无训练。
- 风险：visual mesh 与 collision proxy 偏离；sideboard holder 相交；一个 visual geom
  跨多个局部角色；container interior 是空体积。
- 依赖：P50-E4 exact mapping、精确 depth/calibration、observation-time body transform。
- 失效：读取 current/latest body pose、为 recall 使用 nearest role、查看 P50 result 后
  调 region、将 body pose/role 写入行为路径。

## `R0001-P64`：final-set-only exact-geom value gate

### 瓶颈证据与假设

- 完整 P50-E3 原计划关联 876,960 anchors；但当前决策首先只需要区分：
  - final set 无 stage-compatible candidate；
  - final set 有正确 candidate但 selector 选 distractor；
  - selected candidate 正确，转向 action support。
- 假设：在实现全 anchor first-deletion attribution 前，只对 225 raw、149 component、
  39 final 的 provenance 做 exact-geom association 已足以选择下一步。

### 唯一变量、范围与依赖

唯一变量是启用 evaluator-only observation-time segmentation sidecar；正式 candidate 与
行为不变。

- exact geom + P50-E4 alias；
- 只关联 raw/component/final，不关联全部 enumerated anchor first-rejection；
- 24 个历史 P50 Episode 是固定重分析 cohort，不能称为新独立泛化证据；
- 同 seed replay 只做行为 identity，不把样本数从 24 增加到 48；
- `R0001-P61` 必须先给出 stage-compatible role 定义；
- `R0001-P50-E4` 必须先接受；
- 若 `R0001-P62` 证明 primitive positive control 不成立，默认不执行本项。

### 最小验证

1. sidecar 在原 observation RGB-D/calibration 完成后、进入 latency queue 前采集。
2. segmentation 与 `(timestamp_ns,sequence_id)`、RGB-D、calibration hash 一一绑定。
3. site/background/unknown/mixed fail closed。
4. raw 继承同 identity anchor patch association；component 按完整 raw provenance 聚合；
   final 继承 pre-top64 component label。
5. policy input、candidate bytes/order、selected index、proposed/applied action、terminal、
   physics trace 与 P50 historical capsule bit-identical。
6. private truth AST/import/integration isolation 可执行，不依赖自报 flag。

### 主要指标与判定

Episode 是唯一统计单位：

- final set 含 stage-compatible entity；
- selected candidate stage-compatible；
- `relevant_exists_but_distractor_selected`；
- `no_relevant_final_candidate`；
- mixed/unknown；
- task 与 12 cell 完整分账；
- raw/component/final provenance 守恒。

合同全部通过时：

`accepted as final-set entity-association measurement evidence`

合同有效但无法区分 generator/selector：

`diagnostic_inconclusive`

identity、mapping、isolation、behavior 或 conservation 任一失败：

`invalid`

### 成本、风险与失效

- 成本：中，24 acquisition + 必要 identity replay，无训练。
- 风险：不能定位 task entity 首次在哪个 anchor gate 消失；仅当本项证明这一信息确实是
  下一决策所需，才重新筛选完整 funnel。
- 失效：根据结果添加 alias/role；使用 target site；同 cohort 调参后宣称改善；把
  raw/component/frame/pixel 当独立样本。

## `R0001-P65`：fixed-budget readiness-triggered B0 exit

### 瓶颈证据与假设

- 创新 Agent B 的探索复算认为多数 prefix 在固定 B0 后仍有较大 heading error。
- 假设：只把 B0→B1 固定时钟改为 policy-visible `|heading_error|<=0.35` 触发，同时保持
  B0+B1 总预算 400 step 不变，可提高 step 400 的 bilateral phase-entry geometry
  readiness。

### 唯一主变量

- baseline：固定 `B0=100, B1=300`；
- treatment：首次满足既有 `0.35rad` 条件时从 B0 进入 B1；
- 两者总 prefix 都为 400 step。

B1 控制律、速度上限、candidate、target、arm hold、安全、latency 与环境全部固定；不运行
B2/contact。

### 最小验证与指标

- fresh 36-pair、12-cell paired prefix；
- candidate selection 后完整 continuation clone 分叉；
- 两臂始终 hold；
- 记录 branch/continuation hash、每步 phase、geometry、proposed/applied action 与停止原因。

主要指标：

- step 400 `bilateral_phase_entry_ready_rate`；
- pair-level maximum arm tool-to-preposition distance；
- outer-reach violation count；
- candidate-base range、heading error；
- hard safety、action bounds、terminal 与 candidate identity。

建议接受门：

- treatment ready `>=24/36`；
- 每 task `>=6/12`；
- baseline `<=6/36`；
- 所有守护无不可接受回归。

### 失效、成本、风险与依赖

- 失效：总预算增加、arm 非 hold、修改速度/target/B1 公式、branch identity 不同、结果后
  删除 pair。
- 成本：中低，约 `72 × 400` control step，无训练。
- 风险：可能只是更有效地接近错误 candidate；即使 ready 也不证明 B2/contact/task。
- 依赖：P60 先接受且证明有可由 transition 改善的 geometry，不自动继承。
- 泄露防护：controller 只能读取 candidate center、base pose、policy-visible proprio/FK
  与 safety state；禁止 entity/contact/reward/success/private outcome。

## 初始依赖图

```text
P50-E4 ──> P64 ──> 必要时才重提完整 P50 entity funnel
   └─────> P63（替代显式 alias 的独立可行性研究，不与 P50-E4 捆绑）

P61 ─────> P64 的 stage-compatible role 定义
P62 ─────> P64 是否值得执行的 action-chain positive-control

P60 ─────> P65（唯一行为改动；必须重新冻结 paired experiment）
```

提案集不预设依赖链全部在本轮执行。筛选需优先考虑低成本停止门、主瓶颈价值、结论边界和
文件写集；不得仅按总分机械选择。
