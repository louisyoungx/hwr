# R0017 提案

## 生成过程

- 创新 Agent A 独立审计 v2 unique-coordinate association、12-cell 执行预算与
  legacy P68；建议优先完整 22-Episode label-blind 单重放 preflight。
- 创新 Agent B 独立审计 authoritative-prefix bridge、runtime/snapshot/safety
  边界；建议优先 same-process v2 B2-entry coverage。
- 创新 Agent C 作为反方独立审计两条路线的判定可达性、分母、泄露与自证风险；
  建议先建立零物理成本的合同 oracle，并在修订后优先 prefix。
- 三名 Agent 均先读取 `AGENTS.md`、`R0017/00-context.md`、R0016 提案/筛选/总结及
  相关代码和 artifact；完成前没有查看彼此输出，没有修改文件、启动训练或运行正式
  cohort。
- 主 Agent 只复核证据、合并重复、修正编号冲突并记录分歧，不按偏好提前选择。
- 历史提案保留原 ID 并增加实验后缀。两个 Agent 都临时使用 `R0001-P87`：
  主 Agent 将先出现且更基础的“判定合同可达性 oracle”固定为 `R0001-P87`，将
  “support-coordinate lineage”固定为 `R0001-P88`。

## 共识证据

1. P79/P83 的 24 个 planned Episode 身份一致：
   - 24 Episode、384 capture、36 final candidate；
   - 22 nonempty，任务分账为 living `6/8`、dining `8/8`、kitchen `8/8`；
   - 14 个 singleton、8 个 multi-candidate；
   - candidate、score hash、selected index `24/24`，nonempty selected identity
     `22/22` 可复核。
2. 12 个 `task × observation latency × action latency` cell 每格固定 2 个
   Episode。4 个 latency pair 的 nonempty 分母分别为：
   - `(1,1)`：`5`；
   - `(1,2)`：`6`；
   - `(2,1)`：`5`；
   - `(2,2)`：`6`。
3. legacy P68 不能直接消费 v2：
   - 显式调用 v1 generator 并拒绝 v2 schema；
   - 使用 raw support multiplicity 投票；
   - 每 Episode 同时执行 observer-off baseline 与 observer-on treatment；
   - segmentation 在约 996 个 source observation 上逐步渲染，而最终只消费
     12–21 个 capture；
   - 固定 24-Episode 正式 run 超过 30min，未发布正式或 partial artifact。
4. 22 个 nonempty Episode 共 21,890 acquisition control steps、354 个 capture。
   旧 P50 的 48 次 acquisition replay 总 wall 约 `2706.62s`，线性换算 22 次
   single replay 约 `20.68min`，但这不包含当前 P68 eager segmentation 成本，
   不能当作执行上界。
5. 旧 P85 的 6-sentinel 设计遗漏 action latency，且以 capture count 代理
   eager segmentation wall；该预算门无效。
6. v2 support 的只读、production-assisted 探针：
   - 24/24 candidate canonical bytes 与 P79 一致；
   - 36/36 candidate 的 raw multiplicity 为 `58,351`，unique
     `(capture ordinal,row,column)` 为 `23,616`；
   - 28/36 candidate 有重复坐标；
   - selected raw multiplicity `48,776`，unique coordinate `18,627`，
     比例约 `0.38189`；
   - 19/22 selected candidate 有重复。
   这些已见数值只能用于设计，不是 source-disjoint 确认性结果。
7. R0016 的 P68-E3 `selector_negative >=18/22` 门数学不可达：
   - selector-negative 至少需要同一 Episode 有两个 candidate；
   - 冻结 bank 仅 8 个 multi-candidate Episode；
   - 任务上限分别为 living `1`、dining `1`、kitchen `6`。
   因此旧 P68-E3 不得按原判定冻结。
8. candidate canonical record 不保存完整 component membership 或 support-coordinate
   ledger。P83 只证明 selection lineage，不证明 association 投票坐标。
9. legacy P60：
   - 使用 v1 generator；
   - 995-step acquisition 后执行 B0 100 step、B1 300 step；
   - 在 B2 action 前停止，但把 prefix 与 geometry 捆绑；
   - safety stop 使整个 cohort invalid；
   - 没有 P50→P79/P83→v2 prefix 的显式双根/三根 lineage。
10. 当前 `PhysicalStateSnapshot` 只保存 MuJoCo integration state、
    qpos/qvel/qacc、ctrl 与 solver state，不保存 latency queues、policy history、
    servo targets、runtime/task/contact/safety counters、cached cameras 或完整 RNG
    状态；reset 还会清空计数与队列并把时间重置。它不是 authoritative continuation。
11. R0016 P76-E3 的 pooled coverage 门可隐藏 latency-pair 崩溃：总计 `19/22` 且
    task 门全部通过时，`(2,2)` 仍可能只有 `3/6` safe entry。
12. legacy P60 的 `b2_action_generated=false` 与 `b2_action_executed=false` 主要由
    同一 producer 自报；新 bridge 需要 parent-observed 调用级 tripwire。
13. 创新阶段已看到三任务各一个 `obs=1/action=1` nonempty sentinel：
    - 三者都到达 1,395-step safe B2 entry；
    - wall 约 `43.25–57.50s/Episode`；
    - P50 trace、P79 candidate/index、P40 conservation 与安全守护通过。
    这些 sentinel 未覆盖 12 cell，只可用于可执行性和预算设计，不能进入正式
    coverage 结果或接受判定。
14. capture-only segmentation 的历史探索加速约 `1.028×`，已被拒绝为独立性能
    remedy；head-only observation projection 曾有约 `1.67×` 探索信号，但没有
    12-cell 确认性门，不能在 P85 运行中临时叠加。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P87` | frozen verdict reachability and denominator oracle | 实验合同静态验证 | C | 待筛选 |
| `R0001-P85-E1` | full-cohort 12-cell single-replay execution certificate | 执行预算/身份门 | A | 待筛选 |
| `R0001-P88` | source-disjoint v2 support-coordinate lineage | 测量 lineage | A+C | 待筛选 |
| `R0001-P68-E4` | sealed v2 unique-coordinate selected-route association | association 测量 | A+C | 待筛选 |
| `R0001-P76-E5` | stratified no-B2 same-process authoritative-prefix receipt | physical prefix 测量 | B+C | 待筛选 |
| `R0001-P76-E6` | compiled-model fixed-base outer-envelope certificate | 条件式 geometry 测量 | B+C | 硬依赖 P76-E5 |
| `R0001-P86-E1` | event-complete full-runtime continuation parity gate | restore 合同 | B+C | 延后候选 |

## `R0001-P87`：frozen verdict reachability and denominator oracle

### 瓶颈证据与假设

R0016 的 P68 selector verdict 不可达，P76 pooled gate 又能隐藏 latency pair 的
50% coverage collapse。假设：只使用冻结 cohort 结构与待审判定公式，通过穷举或
约束求解，可在任何物理运行前识别不可达 verdict、分母混用和 accepted 区域中的
分层崩溃。

### 唯一主变量与范围

- 唯一主变量：是否对实验判定合同执行外部结构可达性验证。
- 输入只含 Episode identity、task/cell、candidate cardinality 与合同阈值。
- 不读取真实 association label、prefix outcome、segmentation、mapping、contact、
  force、reward 或 success。
- 不修改候选合同本身；只输出可达性、反例和最坏分层覆盖。

### 最低成本验证

1. 锁定 P79/P83 cohort identity 与 24/22/14/8 分账。
2. 对每个 categorical verdict 穷举结构允许的 Episode 分类：
   - 检查 verdict 是否存在至少一个满足 assignment；
   - 给出每个因果标签的结构上限；
   - 验证 route/nonempty/eligible denominator 守恒。
3. 对每个 accepted 区域求 task、observation latency、action latency 与 latency
   pair 的最坏 coverage。
4. 负控：
   - P68 selector gate `18→8` 时应由不可达变为可达；
   - P76 增加 latency-pair floor 后应拒绝 `19/22` 且 `(2,2)=3/6` 的反例；
   - 删除 empty/nonempty 分账必须 fail closed；
   - candidate/frame/pixel 充当独立样本必须 fail closed。

### 指标与判定

- 指标：
  - verdict reachability；
  - denominator conservation；
  - causal-label structural maximum；
  - accepted 区域各 stratum 的 minimum coverage；
  - mutation/control pass count。
- 全部门通过：
  `accepted as frozen experiment-contract oracle`。
- 任一声明 verdict 不可达、分母不守恒或 accepted 区域可隐藏禁止的 stratum collapse：
  对受测合同给出 `rejected_contract`，oracle 自身通过。
- 读取真实结果、后验改阈值、把非独立单位当样本或 oracle/contract 同源自证：
  `invalid`。

### 成本、风险、依赖与边界

- CPU 秒级，RSS `<256MiB`，artifact `<8MiB`，无 MuJoCo、训练或物理 action。
- 依赖 P79/P83 结构 identity，不依赖结果标签。
- 风险是只证明逻辑一致，不证明科学阈值合理；必须限制声明边界。
- 若入选，本轮不同时运行 P85/P68/P76 正式 cohort，保持单一主假设。

## `R0001-P85-E1`：full-cohort 12-cell single-replay execution certificate

### 瓶颈证据与假设

旧 P68 双 replay 超预算，旧 P85 sentinel 又不能界定真实 eager segmentation 成本。
假设：以冻结 P50 receipt 替代新跑 observer-off baseline，只运行一次 observer-on
replay，可在不改变 physics、action、candidate 或 selection lineage 的前提下，让
固定 22-Episode、12-cell label-blind Phase A 在 24min 内完成。

### 唯一主变量与范围

- 唯一主变量：identity comparator 从同次 observer-off replay 改为冻结 P50 receipt。
- 保持 full-camera、eager segmentation、995 steps、自然 latency、动作、安全与物理
  配置不变。
- 不同时引入 capture-only snapshot、wrist suppression、并行化、generator 修改或
  association classification。
- mapping、interaction 与 classification 模块不得加载。

### 最低成本验证

1. 固定 P79 的 22 个 nonempty Episode，不补 seed，覆盖全部 12 cell。
2. 每 Episode 只运行一次 observer-on replay。
3. 对 P50/P79/P83 逐项核验 randomization/latency、observation/capture identity、
   policy/candidate-visible bytes、proposed/applied action、physical trace、
   v2 candidate/score/index/identity。
4. 直接测量完整 process wall，不按 capture count 或少量 sentinel 外推。
5. 全部 replay 完成后原子封存 354 帧 raw segmentation、trace 与 lineage receipt；
   任何超时不得发布 partial artifact。

### 指标与判定

- 主要指标：完整 22-Episode process wall。
- 守护：
  - 12/12 cell；
  - lineage/trace `100%`；
  - segmentation count `354`；
  - process-tree RSS `<8GiB`；
  - artifact `<256MiB`；
  - label/mapping/interaction read count `0`。
- wall `<=24min` 且全部守护通过：
  `accepted as P68 Phase-A execution-ready`。
- wall `(24min,30min]`：
  `inconclusive_budget_margin`。
- 有效完整执行需要 `>30min`：
  `rejected_budget`。
- identity/lineage 漂移、cell 缺失、加载 label、发布 partial、动态切换优化或后验改门：
  `invalid`。

### 成本、风险、依赖与边界

- 22×995 control steps，无 B0–B7 post-selection action、无训练。
- 依赖 P50 bytes、P79/P83 identity。
- GL 冷启动、任务模型和 eager render 方差是主要风险。
- raw segmentation 是 evaluator-private truth；本提案只封存未解释 bytes，不发布
  association label，也不能仅凭“未 import mapping”声称完整恶意代码隔离。

## `R0001-P88`：source-disjoint v2 support-coordinate lineage

### 瓶颈证据与假设

P83 证明了 candidate/score/selection lineage，但 P79 canonical candidate 不保存完整
component membership 或 support-coordinate ledger。假设：只使用 P50 policy-visible
bytes 的 source-disjoint worker，可重建 36/36 v2 candidate 的 post-self-mask raw
component membership 与完整 coordinate ledger，并在三种 anchor traversal 下得到相同
multiset。

### 唯一主变量与范围

- 唯一主变量：新增隐藏 support-coordinate lineage。
- 票据固定为 `(planned_episode_id,capture_ordinal,row,column)`。
- 不运行 MuJoCo，不读取 segmentation/entity label，不改变 candidate、selector 或
  历史 artifact。
- raw multiplicity 只作描述；后续 association 的主要票数必须 unique。

### 最低成本验证

1. blind worker 只接收净化 P50 blob、acquisition pose 与 capture composite identity。
2. worker 不得 import `hwr`，不得读取 P79/P83 expected candidate/selected metadata。
3. 原子封存后揭盲比较：
   - candidate exact `24/24`；
   - support-count conservation `36/36`；
   - component membership；
   - unique-coordinate ledger hash。
4. row-major、reverse-row-major、column-major 坐标 multiset 一致。
5. mutation 覆盖 v2 `.copy()`、capture ordinal、component edge、pre/post-self-mask
   坐标混淆与 rows/columns 不同步过滤；合成 robot-overlap 正控必须能触发 self-mask。

### 指标与判定

- 全部 exact、双重构 bit-identical、三遍历一致、全部 controls 通过：
  `accepted as v2 support-coordinate lineage evidence`。
- 独立重建稳定但无法匹配冻结 v2 candidate/component：`rejected`。
- worker 读取揭盲 metadata、调用 production helper、坐标 control 无判别力、
  输入漂移或自报 ledger 未经父进程验证：`invalid`。
- 平台数值差异只发生在 component 边界且无法定位：`inconclusive`。

### 成本、风险、依赖与边界

- CPU-only，预算 `3min/<2GiB/<32MiB`，无 MuJoCo 或训练。
- 依赖 P50/P79/P83。
- 风险是 source-disjoint oracle 复制相同错误；沿用 P83 的 blind-root、fixed-worker
  blob、source similarity 与真实 mutation 门。
- 只证明坐标 lineage，不证明实体相关性。

## `R0001-P68-E4`：sealed v2 unique-coordinate selected-route association

### 瓶颈证据与假设

22 个 selected v2 candidate 是否对应 initial microinteraction 允许实体仍未知；
legacy multiplicity 会显著重复计票。假设：以唯一 capture-coordinate 计票后，多数
selected candidate 仍与冻结 allowed entity 相符。

### 唯一主变量与范围

- 唯一主变量：对冻结 selected v2 candidate 执行 evaluator-only entity association。
- 只消费已接受的 P85 Phase-A segmentation/trace、P88 coordinate ledger、P50-E4
  mapping 与 P61/P72 initial annotation。
- 不运行新 physics，不改变 candidate、score、selector、action 或 safety。
- 不再声明全局 selector verdict；冻结 bank 只有 8 个 choice-opportunity Episode，
  不足以支撑旧 `selector_negative >=18/22`。

### 分类与分账

Candidate：

- `compatible`：allowed unique-coordinate ratio `>=0.80`；
- `explicit_incompatible`：known-forbidden unique-coordinate ratio `>=0.80`；
- 其他为 `ambiguous`；
- background、unknown 与 site 保留在总分母。

Episode：

- `selected_positive`：selected candidate compatible；
- `selected_explicit_negative`：selected candidate explicit incompatible；
- `ambiguous`：其余；
- generator availability 与 choice-opportunity selector accuracy 只作单独描述，
  不由 mixed negative 合并归因。

必须分别报告：

1. route availability：`22/24`；
2. relevant candidate availability：`compatible candidate exists / 22`；
3. selected relevance：`selected_positive / 22`，唯一主要指标；
4. choice-opportunity selector accuracy，只在同时存在 compatible 与
   explicit-incompatible candidate 的 Episode 上描述；
5. task 分账：living/dining/kitchen 固定 `6/8/8`。

### 判定

- `selected_positive >=18/22`，living `>=5/6`、dining/kitchen 各 `>=5/8`，
  ambiguous `<=2/22` 且每任务 `<=1`：
  `accepted as v2 initial-association evidence`。
- `selected_explicit_negative >=18/22` 且同样满足任务/ambiguity 门：
  `rejected as selected-route relevance hypothesis`，只停止当前 selected route。
- 其他有效结果：`inconclusive`。
- 发布全局 selector verdict、multiplicity 投票、semantic truth 影响 Phase A、
  lineage/分母/阈值漂移或预算失败后发布 partial label：`invalid`。

### 成本、风险、依赖与边界

- P85 与 P88 接受后为纯离线 Phase B，预算 `<3min/<2GiB/<64MiB`。
- Phase B 必须是新进程，只能在输入 artifact 原子封存后加载 mapping/interaction。
- unique coordinate 仍可能跨 capture 重复同一物理表面；样本单位始终是 Episode。
- 只允许冻结 P50 capture 上的 candidate-conditioned association 声明，不证明识别、
  可达、安全、控制、泛化或家务能力。

## `R0001-P76-E5`：stratified no-B2 same-process authoritative-prefix receipt

### 瓶颈证据与假设

P79/P83 已确定 selection lineage但没有 physical continuation；legacy P60 使用 v1，
把 prefix 与 geometry/全局 hard-stop 捆绑，且 no-B2 主要由 producer 自报。假设：
冻结 22 个 nonempty Episode 可在不实现 restore、不计算 geometry、不生成 B2 action
的条件下，形成跨 task 与 latency strata 均充分覆盖的 same-process authoritative
B2-entry receipt。

### 唯一主变量与范围

- 唯一主变量：新增显式绑定 P50/P79/P83 的 v2 prefix consumer。
- 不改变 generator、score/tie-break、B0/B1 primitive、backend、安全阈值、latency、
  task、seed、模型、runtime restore 或 geometry 判据。
- worker 从 live 995-step acquisition 的 policy-visible capture 重建 full-precision
  v2 candidate/score/selection；P79/P83 expected selected metadata 只在 worker
  原子封存后由 comparer 揭盲。
- 两个 empty Episode 只做 lineage 核验并归类，不执行 B0/B1。

### 最低成本验证

1. 24/24 acquisition 匹配 P50 physical/policy/capture/randomization lineage。
2. candidate/score/index/identity 匹配 P79/P83，v1 generator 调用为 0。
3. 22 个 nonempty Episode 执行 B0 100 step 与 B1 300 step。
4. parent-side wrapper 记录每次 `primitive_action(post_selection_step)`：
   - 只允许 `0..399`；
   - 循环扩为 401 的 mutation 必须在生成 B2 action 前 fail closed。
5. entry 只生成 phase-index 7 policy-visible payload；不调用 B2 primitive。
6. 保存 identity-only、不可恢复声明的 same-process entry capsule：
   - MuJoCo integration/ctrl、qpos/qvel/qacc/warmstart；
   - authoritative base pose/twist；
   - randomization、latency queues、current observation、policy history；
   - servo targets、runtime/task/contact/safety counters、graph/ledger；
   - full-precision candidate/score/target 与 entry payload identity。
7. 穷尽分类顺序：
   `lineage_invalid`、`input_invalid`、`safety_stopped`、`runtime_terminal`、
   `safe_b2_entry`；两个固定空路由为 `empty_candidate`。

### 指标与判定

- 主要指标：
  - `safe_b2_entry/22`；
  - task coverage；
  - 4 个 latency-pair coverage；
  - 12 个 task×latency cell 至少一个 safe entry；
  - first-stop reason/step；
  - lineage exact-match、wall/RSS/artifact。
- 接受门：
  - safe entry `>=19/22`；
  - living `>=4/6`、dining `>=6/8`、kitchen `>=6/8`；
  - latency pair floors：`4/5、5/6、4/5、5/6`；
  - 12/12 cell 至少一个 safe entry；
  - 全部守护通过。
- lineage/合同有效但 coverage 未达门：
  `inconclusive_prefix_coverage`。
- mutation-free、来源一致的预注册重复 sentinel 有可复现 B0/B1 非确定性：
  `rejected_nondeterministic_prefix`。
- metadata 提前泄露、P50/P79/P83 mismatch、v1 调用、生成/执行 B2 action、
  instrumentation 改 trace、safety stop 冒充 entry、restore/geometry 混入或分母不穷尽：
  `invalid`。

### 成本、风险、依赖与边界

- 22×1,395=`30,690` control steps，加 2 个 empty 的离线 lineage 核验。
- 创新期三任务单 sentinel wall 为 `43.25–57.50s`；线性设计参考约 23min，
  不是确认性上界。建议 hard wall `35min`，RSS `<=2GiB`，artifact `<=256MiB`。
- 无训练，前台运行，不休眠。
- worker 不能读取 expected P79/P83 selected metadata；private truth 只允许在 receipt
  封存后用于安全分账，不得影响 candidate、selection 或 B0/B1 action。
- accepted 只证明当前 selected-route safe-prefix coverage，不证明 association、
  reachability、geometry、restore 或能力。

## `R0001-P76-E6`：compiled-model fixed-base outer-envelope certificate

### 瓶颈证据与假设

旧 P60 硬编码链长与 compiled model 相差约 `1.316mm`，且不在 v2 authoritative
entry 上。假设：在 P76-E5 safe-entry Episode 中，至少 80% 的 selected target pair
在 authoritative B2-entry 的瞬时固定基座下，至少一臂被 compiled-model 保守外包络
严格排除。

### 唯一主变量与最小验证

- 唯一主变量：compiled-model fixed-base endpoint 必要条件。
- 仅消费已接受 P76-E5 的 sealed safe entry，不与 E5 同轮实施。
- 在 fresh `MjData` 加载 qpos 并 `mj_forward`，不 step physics。
- 从 compiled model shoulder ancestor 到 grasp-center site 导出保守链长。
- 使用 entry payload、acquisition pose 与 full-precision candidate 重建 target；
  shoulder/target/site 统一 frame。
- 当前 grasp-center 正控不得被排除；shoulder 外 `L+ε` 负控必须被排除；
  shoulder/site/ancestor/model/target mutation 必须翻转或 fail closed。

### 指标与判定

设 safe-entry pair 数为 `n`：

- excluded `>=ceil(0.80n)`，且每任务 `>=ceil(2n_task/3)`：
  `accepted as conditional fixed-base outer-envelope deficit evidence`。
- excluded `<=floor(n/3)`，且每任务均 `<=floor(n_task/2)`：`rejected`。
- 中间区域：`inconclusive`。
- E5 coverage 未接受、分母混入 safety stop、硬编码链长、量化 target、frame/control
  失败：`invalid`。

### 成本、风险、依赖与边界

- CPU-only `<2min/<512MiB/<16MiB`，无 rollout、action 或训练。
- fixed-base 只是不可能性的瞬时必要条件，不证明 IK、collision-free path 或 free-base
  动态不可达。

## `R0001-P86-E1`：event-complete full-runtime continuation parity gate

### 瓶颈证据与假设

当前 snapshot restore 会丢失 queue、time、sequence、counter、history 等状态；P66
又证明 predictive rejection 在 qpos/qvel/time 不推进时仍会改变 runtime。假设：
evaluator-local versioned envelope 可对安全非零 B1 尾段、predictive rejection、
terminal 边界与 multi-clone 隔离产生和不中断分支相同的 continuation。

### 唯一主变量、验证与判定

- 唯一主变量：continuation 是否经过 serialization + fresh-backend restoration。
- 依赖 P76-E5 定义 entry，但不是 P76-E6 前置。
- baseline 不间断执行预注册 actions；treatment 在 fresh backend restore 后执行相同
  bytes。
- 覆盖非零 B1 action、最大 latency queue flush、P66 rejection anchor、terminal
  fixture 与两个 fresh clone 的 mutation isolation。
- envelope 至少包括 mutable model/data、servo targets、queues、full observation、
  history、step/sequence/time/result、task/contact/safety counters、ledger/graph、
  cached cameras、RNG/randomization 与 candidate/target identity。
- 全部 case 每步 action/event/observation/physics/queue/counter/final identity
  bit-identical且所有 deletion/reorder/mutation fail closed：
  `accepted as full-runtime continuation restoration contract`。
- 有效来源下稳定 divergence：`rejected`。
- 只测 hold、跳过 rejection/terminal/multi-clone 或通过清空 queue/重置 counter 获得
  一致：`invalid`。

### 成本、风险、依赖与边界

- 中等到高实现风险，无训练。
- 只证明冻结 probe suite，不是通用 runtime serialization。
- 主要服务未来 P77 clone/search；当前不应为 P76-E5/E6 或 association 提前承担。

## 路线分歧与依赖

### 创新 Agent 的独立优先级

- Agent A：`P85-E1 → P88 → P68-E4 → P76-E5`。
  理由是先确认 association 不可离线替代的 segmentation replay 是否可执行；若
  selected relevance 失败，继续 current selected route 的 B0/B1 决策价值低。
- Agent B：`P76-E5 → P76-E6 → association`。
  理由是 prefix 已有三任务 safe sentinel 与约 23min 设计参考，泄露面更小，且
  coverage 失败能直接停止 geometry/search。
- Agent C：`P87 → P76-E5 → P68-E4`。
  理由是先阻止不可达或分层失真的合同进入正式运行；修订后 prefix 风险低于
  association 的语义真值与预算组合风险。

### 硬依赖

- P68-E4 依赖 P85-E1 与 P88 accepted。
- P76-E6 依赖 P76-E5 accepted。
- P86-E1 依赖 P76-E5 提供具体 continuation 边界，但不是 P76-E6 前置。
- P77 仍为 no-go；必须先有明确 association、prefix、geometry、restore 与 bounded
  positive witness 门。
- selector、默认 v2 migration、Replay、Actor、世界模型训练与 capability evaluation
  仍不授权。

### 不可捆绑项

- P87 合同 oracle 与被审合同的物理结果；
- P85 执行预算与 P68 association label；
- P88 coordinate lineage 与 P68 classification；
- P76 prefix coverage 与 P76 geometry；
- P76 prefix coverage 与 P86 restore；
- 评测修复与能力改进；
- 任一 measurement result 与训练。
