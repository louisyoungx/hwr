# R0016 提案

## 生成过程

- 创新 Agent A 独立审计 producer-side v2 score/selection receipt 与下游真实依赖。
- 创新 Agent B 独立审计 v2 initial association、unique support coordinate、
  label-blind replay 和执行预算。
- 创新 Agent C 独立审计 v2 authoritative-prefix bridge、fixed-base feasibility 与
  free-base search 的 stopping gate。
- 三名 Agent 均先阅读 `AGENTS.md`、`R0016/00-context.md` 和相应历史证据，只读检查
  仓库、Git 历史与已提交 artifact；完成前没有查看彼此输出，没有修改文件、启动训练
  或运行正式 cohort。
- 主 Agent 只做证据复核、稳定编号、重复合并和冲突标注，不按偏好提前筛选。
- 新观点从 `R0001-P83` 递增；历史 P68、P76、P77 重提时保留原 ID 并增加实验后缀。

## 共识证据

1. P79 v2 bank 固定为 24 Episode、384 capture、36 final candidate：
   - route nonempty 为 22/24；
   - living 为 6/8，dining 为 8/8，kitchen 为 8/8；
   - 2 个 empty Episode 只能算 route unavailable，不能进入 selected relevance 或
     physical feasibility 分母。
2. P79 bank 保存 candidate canonical bytes/hash、`score_bytes_sha256`、selected index
   和 selected canonical identity，但没有 exact score bytes，也没有可独立复核的
   score→selection receipt。
3. candidate canonical record 对 center/width/prominence 等字段量化，而 score 使用
   full-precision candidate：
   - 只从 canonical JSON 反量化，22/22 nonempty Episode 的 exact score hash 均无法
     恢复；
   - 但当前 22/22 selected index 仍相同；
   - 8 个多候选 Episode 的 full-precision 最小 top-2 margin 约
     `0.012365174520925004`。
4. 使用冻结 P50 capture、v2 generator 与独立写出的 score 公式做只读内存探针，可在
   24/24 Episode 恢复 P79 score hash 与 selected index。该探针不是正式独立 oracle，
   但反驳“缺少新 producer receipt 必然阻塞具体 P68/P76 consumer”。
5. P79 producer 对每个 Episode 先调用 `candidate_scores()` 生成 hash，再由
   `select_candidate_index()` 内部第二次计算 scores；当前 deterministic，但没有原子
   证明 selected index 来自被 hash 的同一 score tuple。
6. legacy P68：
   - 明确调用 v1 generator并拒绝 v2；
   - support tracer 保留 legacy `patch_valid &= ...` view mutation；
   - 把 raw support multiplicity 当作 pixel vote；
   - 同一 Episode 执行 observer-off baseline 与 observer-on treatment 两次 replay；
   - 正式 24-Episode run 在 30 分钟门失败且没有 artifact。
7. v2 support 的只读内存探针显示：
   - 24/24 candidate canonical records 与 P79 一致；
   - 36/36 candidate 的 multiplicity support 守恒；
   - 28/36 candidate 有同一 capture 内重复 coordinate；
   - 19/22 selected candidate 有重复；
   - selected raw multiplicity 合计 `48,776`，unique capture-coordinate 合计
     `18,627`，后者仅为前者 `38.19%`；
   - selected Episode 的 unique/multiplicity ratio 中位数约 `0.531`，最低约
     `0.209`。
8. 384 个 capture 只有 254 个唯一 `(timestamp_ns, sequence_id)`；可靠 join key
   至少包含 Episode、capture ordinal、timestamp、sequence 与输入 hash。
9. P79 cohort 与 P57 36 Episode、P60 两个已执行 prefix Episode、P66 anchor 均无
   Episode ID 交集。P57/P60/P66 只能提供机制和边界证据，不能提供 v2 发生率。
10. P79 没有可恢复的 authoritative B2-entry continuation；当前
    `PhysicalStateSnapshot` 也不含 latency queues、policy history、servo targets、
    runtime/task/safety counters，且 restore 会重置 MuJoCo time。
11. P66 证明 predictive rejection 不推进 authoritative qpos/qvel/time，也不提交危险
    action；但 rejection step 仍改变 runtime counters、queues、servo targets 与
    `ctrl`。因此 safety-stopped 状态不得冒充 B2-entry continuation。
12. 描述性反例表明：
    - acquisition base 下，22/22 v2 selected target pair 至少一臂被 fixed-base
      外包络排除；
    - 把 base 理想放到 candidate 前方 `0.85m` 后，22/22 都不再被同一外包络排除。
    这不证明路径安全或动态可达，但足以否定“瞬时 fixed-base exclusion 等于 free-base
    动态不可达”。
13. 从 compiled model 的 shoulder ancestor 到 grasp-center site 按刚性段长度求和，
    保守上界约 `1.1402562418976663m`；旧 P60 硬编码值约
    `1.138940147524481m`，相差约 `1.316mm`。新证书必须从 compiled model 导出，
    不得沿用旧常量。
14. P50 capture bytes 受 `.gitignore` 排除，但 P79 manifest 绑定了其 size/hash；
    当前 checkout 可消费，跨 checkout 恢复性仍是风险，不得把 Git hash commitment
    冒充 bytes 的持久可用性。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P83` | consumer-local v2 selection-lineage oracle | dependency/measurement ablation | A+B | 待筛选 |
| `R0001-P84` | atomic producer-side v2 score/selection receipt | producer evidence contract | A | 待筛选 |
| `R0001-P85` | label-blind single-replay execution preflight | evaluator execution budget | B | 待筛选 |
| `R0001-P68-E3` | v2 unique-coordinate initial association | physical association measurement | B | 待筛选 |
| `R0001-P76-E3` | v2 authoritative-prefix bridge | physical prefix measurement | C | 待筛选 |
| `R0001-P76-E4` | compiled-model fixed-base outer-envelope certificate | conditional geometry measurement | C | 待筛选 |
| `R0001-P86` | full-runtime continuation restore parity gate | evaluator state contract | C | 待筛选 |
| `R0001-P77-E3` | bounded free-base existence search | physical existence witness | C 反方门 | 当前不具备筛选资格 |

## `R0001-P83`：consumer-local v2 selection-lineage oracle

### 瓶颈证据与假设

P79 exact score bytes 不可由量化 candidate document 恢复，但具体 P68/P76 consumer
仍可访问 P79 manifest 绑定的 P50 policy-visible captures。假设：若在看不到 P79
score hash、selected index 与 selected identity 的支路中，source-disjoint oracle
仍能恢复相同 full-precision v2 candidate、score 与 selection，则新 producer receipt
不是这两个具体 consumer 的硬阻塞。

### 唯一主变量与范围

- 唯一主变量：重建支路在揭盲前是否可读取 P79 score/selection metadata。
- control：允许读取 metadata。
- treatment：runtime read guard 禁止读取。
- 不修改 v2 generator、score 权重、tie-break、candidate schema、P79 bank、P68/P76
  estimand、默认 runtime、动作、安全或训练。

### 最低成本验证

1. 锁定 P79 bank、P50 captures、v2 producer blob、selector blob和两个具体 consumer
   入口；显式解析双 artifact root。
2. treatment 不导入或调用 production `generate_candidate_set_v2()`、
   `candidate_scores()`、`select_candidate_index()`。
3. 独立实现 v2 anchor/mask-copy/self-mask/merge/ranking、score 公式与
   `max(score, -index)` tie-break。
4. 对 24 Episode 重建两次；只在 sealed receipt 生成后揭盲比较 P79 commitments。
5. 负控至少覆盖：
   - v1/v2 schema 互换；
   - candidate order；
   - final base pose；
   - score 权重；
   - tie-break；
   - selected metadata；
   - capture root/path/hash；
   - canonical-only candidate 冒充 full-precision input；
   - 提前读取 receipt 字段。

### 指标与判定

- 主要指标：
  - candidate bytes/hash exact match `24/24`；
  - score hash exact match `24/24`；
  - selected index exact match `24/24`；
  - nonempty selected identity exact match `22/22`；
  - receipt field read before reveal `0`；
  - legacy v1 generator call `0`。
- 守护：
  - 24/24 capture lineage；
  - 双重构 bit-identical；
  - 所有负控 fail closed；
  - private geom/body/contact/segmentation/label/force truth 读取为 0；
  - P79/P50 artifact 零修改。
- 全部门通过：
  `accepted as consumer-local v2 selection-lineage evidence`，并允许声明新 producer
  receipt 不是冻结 P68/P76 的共同硬前置。
- 输入与 provenance 有效但任一 Episode 无法恢复，且 receipt-visible control 可恢复：
  `rejected`，分 consumer 记录阻塞范围。
- 仅 exact score bytes 存在绑定平台的浮点差异、selection 一致：
  `inconclusive_platform_score_bytes`。
- oracle 复用 producer helper、提前读取 receipt、调用 v1、读取 private truth、猜测
  双根、修改历史 artifact 或 source/input 漂移：
  `invalid`。

### 成本、风险、依赖与边界

- CPU-only，wall `3min`，process-tree RSS `<2GiB`，artifact `<16MiB`。
- 无 MuJoCo、训练、segmentation 或 B0/B1。
- 风险是“独立 oracle”复制同一逻辑错误；必须由 source-disjoint 实现与针对公式各项的
  mutation controls 共同约束。
- 该提案只回答具体依赖，不证明 candidate quality、association、reachability、
  safety、能力或未来任意 consumer 完备。
- 不重复 P79：不再次验证 mask ownership。
- 不重复 P80：不声称 whole-program consumer completeness，只验证两个冻结 consumer
  所需 lineage 能否在看不到 receipt 时恢复。

## `R0001-P84`：atomic producer-side v2 score/selection receipt

### 瓶颈证据与假设

P79 producer 两次计算 score，只保存 hash 与 selected metadata，没有原子证明两者来自
同一 tuple。假设：在不改变 selector 语义的前提下，可由同一次 score 计算产生
versioned、bit-replayable、独立可审计的 selection receipt。

### 唯一主变量与范围

- 唯一主变量：是否生成原子 receipt。
- producer 每 Episode 只计算一次 scores，并直接从该 tuple 选择 index。
- 不改变 generator、score 权重、base transform、tie-break、candidate order或默认
  runtime；不覆盖 P79 bank，只写新的 R0016 artifact。

### 最低成本验证

receipt 至少绑定：

- P79 candidate schema/path/hash；
- P50 capture复合 identity与 final-input hash；
- acquisition/final base pose IEEE bytes；
- full-precision scoring-input ledger；
- `<f8` score bytes、长度与 hash；
- selected index、tie-break ID、selected canonical identity；
- producer、selector、独立 oracle source blob与环境版本。

对 24 Episode 生成两次并比较 bit identity；独立 oracle 不得调用 production
score/selection 函数。负控覆盖 score byte、candidate order、final base、weight、
tie-break、schema和“双调用 scorer 返回不同值”。

### 指标与判定

- score entry `36`，raw score payload `288 bytes`；
- P79 score hash `24/24`、selected index `24/24`、selected identity `22/22`；
- production score call `1/Episode`；
- independent oracle `24/24`；
- 双重构 bit-identical、所有负控 fail closed。
- 全部门通过：
  `accepted as atomic v2 score-selection receipt`。
- 有效冻结环境下无法保持 P79 selection，或 oracle 发现真实 score→selection 不一致：
  `rejected`。
- 只有平台数学库造成 exact-byte 差异但 interval/tie-break selection 稳定：
  `inconclusive_platform_score_bytes`。
- 修改 selector 语义、覆盖 P79、读取 private truth、调用 v1、source/artifact 漂移或
  partial output：
  `invalid`。

### 成本、风险、依赖与边界

- CPU-only，分钟级，artifact KB 级，无 MuJoCo/训练。
- full-precision ledger 只用于审计，不成为新的 geometry authority。
- 若 P83 accepted，P84 仅是 provenance 改善，不能继续阻塞 P68/P76，也不应占用本轮
  唯一高价值科学候选。
- 不重复 P79/P80：补的是同一 producer event 的原子 selection relation，不是 generator
  mask 修复或任意 consumer 完备性。

## `R0001-P85`：label-blind single-replay execution preflight

### 瓶颈证据与假设

legacy P68 每 Episode 先跑 observer-off baseline，再跑 observer-on treatment；正式
24 Episode 因 30 分钟门失败。假设：只运行 observer-on replay，并直接与冻结 P50
historical trace receipt 比较，可以不改变物理轨迹和 association estimand而显著降低
执行预算。

### 唯一主变量与范围

- 唯一主变量：每 Episode replay 数量由 2 次降为 1 次。
- 保持 full-camera、eager segmentation、candidate/score/selector、动作、安全和物理
  配置不变。
- 本提案只做 timing/identity preflight，不加载 mapping/interaction，不产生
  classification。

### 最低成本验证

1. 结果前按三任务×observation latency 1/2 固定 6 个 sentinel；每格选择 capture count
   最大者，平局按 Episode ID。
2. 交替运行 legacy 双 replay 与 single replay，预热后重复。
3. single replay 直接匹配 P50 capture sequence、policy/candidate-visible bytes、
   proposed/applied actions、physical trace、randomization与 failure identity。
4. 使用保守上界投影 22 个 nonempty Episode。
5. mapping、interaction和 classification 模块不得导入；完成后也不得发布任何
   association label。

### 指标与判定

- identity/trace `100%` 一致；
- paired wall-time 降幅；
- 22-Episode conservative projection。
- identity 全过、降幅 `>=30%` 且 projection `<=24min`：
  `accepted as P68 execution-ready`。
- identity 等价但降幅 `<20%` 或 projection `>30min`：
  `rejected`。
- 降幅 `20%–30%`、projection `24–30min` 或 GL 波动跨门：
  `inconclusive`。
- trace/candidate 漂移、加载 label 模块、结果后改预算或动态按 task/seed 选路径：
  `invalid`。

### 成本、风险、依赖与边界

- sentinel wall `15min`，RSS `<8GiB`；无正式 cohort、无训练、不休眠。
- 依赖 P83 accepted 或等价 selection lineage。
- GL 抖动可能影响小样本 timing；capture-only segmentation 过去仅约 `1.028×`，不得
  混入本提案。
- 若仅性能不足，再独立考虑 P74；不得与本提案捆绑。

## `R0001-P68-E3`：v2 unique-coordinate initial association

### 瓶颈证据与假设

legacy P68 把重叠 anchor 对同一 capture pixel 的重复 support 当作独立票，v2 探针显示
selected unique support 仅为 raw multiplicity 的 `38.19%`。假设：在 22 个 nonempty
Episode 上，以唯一 capture-coordinate 为投票单位，selected v2 candidate 多数与冻结的
initial microinteraction allowed entity 相符。

### Estimand 与唯一主变量

- route availability：`nonempty/24`，独立分账。
- relevant candidate availability：22 个 nonempty Episode 中至少一个 compatible
  candidate 的 Episode 数。
- selected relevance：22 个 nonempty Episode 中 selected candidate compatible 的
  Episode 数，唯一主要指标。
- candidate 内票据单位：
  `(planned_episode_id, capture_ordinal, row, column)`。
- 唯一主变量：对冻结 selected candidate 做 evaluator-only entity association。
- multiplicity-weighted ratio仅作预注册敏感性描述。

### 最低成本验证

1. P83 与 P85 先通过。
2. source-disjoint support oracle 重建 36/36 final candidate：
   v2 mask-copy、self-mask 后坐标、component membership、raw multiplicity、
   unique-coordinate ledger与 final bytes/order。
3. Phase A 完全 label-blind，运行 22 个 single observer-on replay，预计 354 个
   capture segmentation；全部 trace、segmentation 与 coordinate ledger 封存后，
   Phase B 才加载 frozen mapping/interaction离线分类。
4. private-truth mutation只能改变 Phase B 报告，不得改变 candidate、selection、action
   或 physics trace。
5. empty Episode 单列，不进入 22 分母。

### 分类与判定

candidate：

- `compatible`：allowed unique-coordinate ratio `>=0.80`；
- `explicit_incompatible`：known-forbidden unique-coordinate ratio `>=0.80`；
- 其他：`ambiguous`；
- background/unknown 保留在总分母。

Episode：

- `selected_positive`：selected candidate compatible；
- `generator_negative`：没有 compatible candidate，且全部 candidate explicit
  incompatible；
- `selector_negative`：存在 compatible candidate，但 selected explicit incompatible；
- 其他：`ambiguous`。

接受门：

- positive evidence：`selected_positive >=18/22`，living `>=5/6`、dining `>=5/8`、
  kitchen `>=5/8`，ambiguous `<=2/22` 且每任务 `<=1`；
- generator stopping evidence：`generator_negative >=18/22`，同样任务级和
  ambiguous 门；
- selector stopping evidence：`selector_negative >=18/22`，同样任务级和
  ambiguous 门；
- mixed stopping evidence：两类 negative 合计 `>=18/22`，但各自未单独过门；
- 其余：`inconclusive`；
- lineage、oracle、source lock、预算、label 隔离、分母或 immutable trace 失败：
  `invalid`。

### 成本、风险、依赖与边界

- 正式 22 次 replay；P85 projection 必须 `<=24min`，hard gate `30min`；
  RSS `<8GiB`，artifact `<2GiB`。
- P50 是修复前采集的冻结 captures，不是新鲜 v2 cohort；结论只允许是
  frozen-capture candidate-conditioned evidence。
- unique coordinate 仍可能跨 capture 重复同一物理表面；最终样本单位始终是 Episode。
- 无训练、无 post-selection B0–B7 capability execution、无能力或泛化声明。

## `R0001-P76-E3`：v2 authoritative-prefix bridge

### 瓶颈证据与假设

P79 与 legacy P57/P60/P66 cohort 不相交，P79 又没有 B2 continuation。假设：在具体
v2 selection lineage 成立后，冻结 P79 cohort 的 22 个 nonempty Episode 可按同 seed、
同随机化、同 acquisition 和当前 B0/B1 primitive 重放，形成可审计的 authoritative
B2-entry，且 safe-prefix coverage 足以支持后续条件式几何分析。

### 唯一主变量与 Prefix 定义

- 唯一主变量：新增一个具体 consumer 锁定、版本化的 v2 prefix bridge。
- 不修改 generator、selector、B0/B1 primitive、backend、安全阈值、latency、task、
  seed 或模型。
- bridge 必须：
  1. 显式连接 P79 与 P50 双根；
  2. 校验 Episode、capture、seed、latency、randomization与 trace lineage；
  3. 从 P50 captures 重建 full-precision v2 candidate和 selection；
  4. 从 fresh backend完整重放 995-step acquisition并匹配 P50；
  5. 注入 receipt-bound v2 selected candidate，只执行 100-step B0 与 300-step B1；
  6. safety intervention、terminal、stale action、nonfinite、invalid force 或
     provenance drift 时立即停止；
  7. B2-entry 必须位于任何 B2 action 生成前；
  8. 保存 authoritative MuJoCo state、base pose/twist、policy-visible payload、
     queues、servo targets、history、counters、randomization与 full-precision
     candidate/target identity。

### 分母、指标与判定

- lineage denominator：24；
- route nonempty：22/24；
- safe-prefix denominator：固定 22，任务分账 6/8/8；
- 每个 Episode 唯一归类为 `empty_candidate`、`safe_b2_entry`、
  `safety_stopped`、`runtime_terminal`、`input_invalid` 或 `lineage_invalid`。
- 主要指标：`safe_b2_entry_count/22`、每任务 coverage、失败类别与 first stop。
- `safe_b2_entry >=19/22`，living `>=4/6`、dining `>=6/8`、kitchen `>=6/8`，
  且全部 lineage/safety/identity 守护通过：
  `accepted as v2 authoritative-prefix bridge evidence`。
- lineage有效但 coverage不足：
  `inconclusive_prefix_coverage`，不得运行 P76-E4 正式比例。
- selection lineage缺失、任何 mismatch、instrumentation改变 trace、
  safety-stopped冒充 B2 entry、生成/执行 B2 action或 complete-case删除：
  `invalid`。

### 成本、风险、依赖与边界

- 上界 `24×995 + 22×400 = 32,680` control steps；
- wall `45min`，process-tree RSS `<=2GiB`，artifact `<=256MiB`；
- 无训练、无 B2/contact phase；当前属于小型正式物理测量，前台等待，不休眠。
- 依赖 P83 或等价 concrete selection lineage。
- low coverage 不能表述为 fixed-base exclusion 或动态不可达。

## `R0001-P76-E4`：compiled-model fixed-base outer-envelope certificate

### 瓶颈证据与假设

旧 P60 硬编码链长与 compiled model 相差约 `1.316mm`，且未在 v2 authoritative B2
entry 上测量。假设：在 P76-E3 的 safe-entry Episode 中，至少 80% 的 v2 target pair
在各自 authoritative B2-entry 瞬时 base 下，至少一臂被 compiled-model 保守外包络
严格排除。

### 唯一主变量与最小验证

- 唯一主变量：compiled-model fixed-base endpoint 必要条件。
- 不改变 prefix、base、candidate、target、IK、controller、安全或 dynamics。
- 从 compiled model 的 shoulder ancestor 到 grasp-center site 自动提取保守链长；
  对每个 B2-entry qpos 执行 `mj_forward`。
- target 使用 B2 policy-visible delayed base 与 full-precision v2 candidate按冻结
  primitive公式重建；shoulder 使用 authoritative qpos；二者变换到同一 frame。
- 正控为当前 grasp-center not-excluded，负控为 shoulder 外 `L+ε` target excluded；
  shoulder/target/site/ancestor/model mutation必须翻转预期结果。

### 分母、指标与判定

- 只消费 P76-E3 `safe_b2_entry`，且 E3 必须已 accepted；
- pair 是样本，arm 不是独立样本；
- safety-stopped、empty、terminal 不进入 geometry denominator。
- 指标：pair minimum outer margin、至少一臂 hard-excluded比例、双臂描述比例、
  每任务分账、base delta、compiled chain ledger。
- `E >= ceil(0.80n)` 且每任务 `E_task >= ceil(2n_task/3)`：
  `accepted as conditional fixed-base outer-envelope deficit evidence`。
- `E <= floor(n/3)` 且每任务 `E_task <= floor(n_task/2)`：
  `rejected`。
- 中间区域：`inconclusive`。
- coverage未通过、硬编码链长、量化 target、delayed base冒充 physical shoulder pose、
  control/mutation失败或错误分母：
  `invalid`。

### 成本、风险、依赖与边界

- CPU-only `<2min`，RSS `<512MiB`，artifact `<16MiB`，无 rollout/训练。
- 依赖 P76-E3 accepted。
- accepted 也只证明瞬时 fixed-base 外包络必要条件，不证明 joint-limit IK、
  collision-free path 或 free-base 动态不可达。

## `R0001-P86`：full-runtime continuation restore parity gate

### 瓶颈证据与假设

当前 snapshot 不含完整 Python/runtime continuation，P66 又证明 safety rejection 会改变
非 qpos/qvel 状态。假设：一个只用于具体 evaluator、同时保存 MuJoCo 与 Python runtime
state 的版本化 envelope，可从 fresh backend恢复 safe B2-entry，并在连续三步冻结 hold
下与不中断 continuation逐字段一致。

### 唯一主变量、验证与判定

- 唯一主变量：continuation serialization/restoration completeness。
- 输入仅来自 P76-E3 safe B2-entry。
- envelope 至少含 integration/control state、servo targets、action/observation queues、
  full observation bytes、policy history、step/sequence/time、task/contact/safety counters、
  active task、randomization与 candidate/target identity。
- baseline为不中断进程，treatment为fresh backend restore；执行相同三步 hold，覆盖最大
  两步 latency再多一步。
- 全部 safe-entry Episode 的 pre-state 与三步 outcome bit-identical，且所有 component
  deletion/reorder/mutation fail closed：
  `accepted as full-runtime continuation restoration contract`。
- 任一有效 Episode 可复现 divergence：`rejected`。
- P76-E3 coverage不足：`inconclusive_prefix_coverage`。
- 清空队列、重置计数、直接改 qpos、引入 future truth 或 safety-stopped冒充入口：
  `invalid`。

### 成本、风险、依赖与边界

- 最多 22 fresh backend×3 steps，wall `5min`，RSS `<1GiB`，artifact `<64MiB`。
- 强依赖 P76-E3；只对当前 source/model有效，不是通用 serialization API。
- 不是 P76-E4 前置，只是未来 P77 clone/search 的强制门。

## `R0001-P77-E3` 的反方 stopping gate

当前不具备正式筛选资格。只有以下条件全部满足后才允许重新提案：

1. concrete v2 selection lineage accepted；
2. P76-E3 safe-prefix coverage accepted；
3. P76-E4 表明足够多 fixed-base exclusion，存在重布置 base 的信息价值；
4. P86 full-runtime restore parity accepted；
5. v2 association至少证明被操作 candidate 不是系统性 distractor；
6. action generator、horizon、lattice、tie-break、seed、restart与总预算结果前冻结；
7. ranking只读 policy-visible candidate/proprioception/允许观测，不读取 geom/body/contact/
   force 或 future safety truth；
8. 所有动作经过真实 latency、actuator scale、`backend.apply()` 与 production predictor。

未来即使搜索不到，也只能判冻结 controller 在冻结预算内无 witness；动态可达性仍为
`inconclusive_not_found`。直接设置 base/qpos、清队列、绕过 predictor、结果后扩预算
或使用 private truth 均为 `invalid`。

## 提案间依赖与冲突

- A/B 共识：P83 应先于“直接实现 producer receipt”，因为 receipt 可能不是具体下游
  的真实阻塞。
- A 主张 P84 可作为独立 provenance 改善；B 主张 P68 的 scientific blocker 是 support
  与预算；两者不冲突，但资源优先级不同。
- B 建议 P83→P85→P68-E3；C 建议 P83→P76-E3→P76-E4。
- P68-E3 与 P76-E3 都需要具体 selection lineage，但互不依赖：
  - P68回答“candidate 是否对应 initial microinteraction”；
  - P76回答“选定 candidate 的 prefix 是否形成足够 authoritative B2 entry”。
- P84、P85、P68-E3、P76-E3、P76-E4、P86 不得在同一因果实验中捆绑。
- 本轮原则上只选择一个主假设；未选项必须重新创新与筛选，不自动继承。
