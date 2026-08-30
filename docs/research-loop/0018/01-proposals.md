# R0018 提案

## 生成过程

- 创新 Agent A 独立审查 `R0001-P87-E2`，只读检查 R0017 冻结合同、失败实现分支
  `exp/R0001-P87-contract-oracle` 与提交 `485367f`；建议把 P87 收缩为不接入真实
  artifact 的合成单调下界合同内核资格门。
- 创新 Agent B 独立审查 `R0001-P88` 与 association 前置，只读检查 P79/P83
  generator、worker、tests 与 artifact；建议先做人工精确 coordinate/component
  真值资格门，再单独重建真实 v2 bank ledger。
- 创新 Agent C 独立审查 `R0001-P76` authoritative-prefix 与反方路线，只读检查
  phase-entry、runtime、snapshot、安全、tests 与 P50/P79/P83 cohort；建议先建立
  外部 no-B2 action-service 能力边界，再做 full-precision authorization，最后才运行
  新鲜 holdout prefix。
- 三名 Agent 均先读取 `AGENTS.md` 与 `R0018/00-context.md`，完成前未查看彼此输出，
  未修改文件、启动训练或运行正式物理 cohort。
- 主 Agent 只合并重复、统一命名、保留分歧与依赖，不按偏好提前选择。

## 共识证据

1. 当前主基线没有 P87 实现；候选提交 `485367f` 仅保留供审计。
2. R0017 P87-E1 的两项核心 control 不是有效单变量对照：
   - C01 把 overall target `18→8` 后，living/dining task floor 仍各要求 `5`，
     而各自 eligible choice-opportunity 仅 `1`，整个合同仍不可达；
   - C02 删除原 `latency_pair` claim scope 后再增加 latency floor，同时改变了
     claim scope 与 floor。
3. P87-E1 Solver B 只检查空 assignment，却把其余状态计为 pruned 并声称
   exhaustive；contradiction verifier 没有验证 `required > available`。
4. P79/P83 当前只证明 candidate generation 与 selection lineage：
   - P79 canonical candidate 不保存完整 component membership 或 coordinate ledger；
   - P83 worker 在 back-project 后丢弃 row/column identity；
   - candidate bytes、support count 与遍历一致均不能排除等量坐标替换、row/column
     错配、跨 component 交换或 pre/post-self-mask 混淆。
5. production-assisted 设计期探针显示 coordinate lineage 不是形式问题：
   - 197,961 个 pre-self-mask support pixel；
   - self-mask 删除 863 个；
   - 6 个 anchor 因 self-mask 后不足 24 点消失；
   - 533 个 raw anchor 形成 110 个 component，其中 36 个成为 final candidate；
   - selected raw multiplicity `48,776`，unique coordinate `18,627`；
   - 19/22 selected candidate 存在重复坐标。
   这些数值只用于设计，不是 source-disjoint 确认性结果。
6. legacy P68 显式绑定 v1，并保留会原位修改 parent mask 的 slice-view 逻辑，不能
   作为 v2 coordinate oracle。
7. 当前 generic `primitive_action()` 覆盖 B0–B7，B2 从
   `post_selection_step=400` 开始；legacy P60 只是由调用方 `range(400)` 避免 B2，
   现有 sequence/string tests 不能证明 runner 没有 B2 callable，也不能证明 step
   400 在 action bytes 生成前被拒绝。
8. P83 已证明 candidate/score/index/identity，但现有 receipt 没有面向 P76 的
   full-precision selected-candidate authorization。quantized canonical candidate
   不能代替 action 所需的 full-precision bytes。
9. 当前固定 bank 的三个 `(observation_latency=1, action_latency=1)` sentinel
   safe-entry outcome 已暴露：
   - 保留它们时，固定 bank 只能作为 descriptive；
   - 排除它们后只有 19 个未暴露 nonempty Episode；
   - living `(1,1)` cell 为零，未暴露部分只有 11/12 cell；
   - 因而当前固定 bank 不能支持完整 12-cell 的未见确认性 P76。
10. 当前 `PhysicalStateSnapshot` 不是 full-runtime continuation；P76 prefix 不应捆绑
    snapshot/restore。prefix coverage、geometry 与 restore 也必须保持三个独立实验。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P87-E2` | manual truth-table monotone-contract qualification | 合同内核资格门 | A | 待筛选 |
| `R0001-P88-E1` | manual exact coordinate-oracle qualification | coordinate oracle 资格门 | B | 待筛选 |
| `R0001-P88-E2` | fixture-qualified blind v2 coordinate ledger | 真实 bank lineage | B | 硬依赖 P88-E1 |
| `R0001-P76-E7` | external B0/B1 action-service capability gate | no-B2 能力边界 | C | 待筛选 |
| `R0001-P83-E2` | full-precision selected-candidate authorization handoff | lineage 授权 | C | 待筛选 |
| `R0001-P76-E8` | precommitted fresh-holdout authoritative prefix | 物理 prefix 测量 | C | 硬依赖 P76-E7/P83-E2 |

## `R0001-P87-E2`：manual truth-table monotone-contract qualification

### 瓶颈证据与假设

R0017 证明完整五合同 oracle 的冻结 control 与所谓独立 solver 无效。现有目标合同只含
total/stratum 下界；对 eligible 集合 `E` 和 accepted subset `S⊆E`，若判定只有：

- `|S| >= target_minimum`；
- 对每个分层 `|S∩G(g,k)| >= floor(g,k)`；

则谓词关于 `S` 单调。假设：一个不读取历史 outcome、仅支持该单调下界语义的极小
predicate kernel，能在预冻结人工 fixture 上与独立完整真值表逐状态一致，并正确验证
真实不等式方向。

### 唯一主变量与范围

- 每个 paired control 只改变一个合同数值叶子：
  - overall `target_minimum`；
  - 一个 task floor；
  - 一个 latency-pair floor。
- Episode 集、claim scope、denominator、eligibility、其他 floors、candidate kernel 与
  verifier均不变。
- canonical deep-diff 必须精确为一个 JSON Pointer。
- 不接入 P50/P79/P83、历史五合同、MuJoCo、candidate、selector、association、
  prefix、geometry、restore、安全、训练或 formal publisher。

### 冻结前手工 fixture

| Pair | 固定结构 | 唯一变化 | 不可达侧 accepted assignments | 可达侧 accepted assignments |
|---|---|---|---:|---:|
| `F-O` | 4 个 eligible Episode | overall `5→4` | 0 | 1 |
| `F-T` | L=2、D=2；total=2、D floor=1 | L floor `3→2` | 0 | 3 |
| `F-L` | `o1-a1=2`、`o2-a2=1`；total=2、前者 floor=1 | 后者 floor `2→1` | 0 | 3 |

完整状态总数为 `16×2 + 16×2 + 8×2 = 80`。80 是穷尽软件状态，不是独立科学样本。

另冻结 schema guard：claim scope 声明某分层却缺 minimum 时，必须
`invalid: claim_without_minimum`；不能删除 claim scope 以通过 control。

### 最小验证与指标

1. 实现前冻结六份合同、三个 single-leaf diff 与静态 expected accepted-ID set。
2. candidate process 只输出每个 assignment 的 pass/fail，不读取 expected truth。
3. verifier 不 import candidate module、不调用 candidate helper，直接比较完整
   assignment-ID set。
4. verifier 独立重算 eligible、stratum available 与 `required > available`。
5. equality 与 reversed inequality 必须被拒绝为 contradiction。
6. control inventory 缺失、重复或额外均失败。
7. 两次运行 bit-identical。

主要指标：

- truth-table agreement `80/80`；
- reachability verdict `6/6`；
- accepted cardinality 精确 `0/1、0/3、0/3`；
- valid inequality witness `3/3`；
- single-leaf diff `3/3`；
- exact control inventory通过。

守护指标：

- private outcome 与 P50/P79/P83 read count均为 0；
- shared scientific helper count 0；
- sample-unit violation 0；
- CPU-only，wall `<1s`、RSS `<64MiB`、artifact `<1MiB`。

### 判定、成本与风险

- `accepted`：全部精确门通过；只接受为单调下界合同内核资格证据。
- `rejected`：fixture与独立 verifier有效，但任一状态、cardinality或不等式错误。
- `invalid`：truth由 candidate生成、deep-diff不止一个叶子、verifier共享 helper、
  inventory不精确、读取真实 outcome或把 assignment冒充科学样本。
- `stop`：冻结前数学证明不成立；实现引入上界、互斥、预算、依赖等非单调约束；或
  试图在本候选直接审计历史五合同。
- 成本约半日至一日，纯 CPU。
- 风险：合成 fixture覆盖有限；accepted也不证明任何真实 P68/P76合同有效。

### 旧代码处置

- 可复用：命名、schema概念、canonical JSON基础设施和红队反例。
- 必须重写或丢弃：
  - 伪 exhaustive `solve_enumeration()`；
  - 不检查不等式方向的 `verify_contradiction()`；
  - 旧 C01/C02；
  - 不验证 exact inventory 的 report acceptance；
  - 把五合同、publisher、provenance与科学 kernel混在近 800 行模块的结构。

## `R0001-P88-E1`：manual exact coordinate-oracle qualification

### 瓶颈证据与假设

当前没有人工可知的 post-self-mask coordinate、raw-anchor membership、component 与
merge 后 unique ledger真值。仅比较 candidate bytes、support count、hash或遍历一致，
无法排除数量保持型坐标错误。

假设：一个不依赖 production helper 的 blind worker，能在实现前冻结、人工逐项可推导
的 fixture上精确恢复上述 ledger，并杀死保持 count/hash外观的错误。

### 唯一主变量与范围

- 唯一主变量：是否加入外部冻结的人工 exact truth。
- 不改变 production generator、阈值、candidate、selector、association或历史 artifact。
- 只允许新增 R0018 fixture、worker、comparer和 focused tests。

### 最小验证

1. 实现 worker 前冻结独立 truth pack；expected只能手写，不能由 production、
   P83 worker或待测 worker生成。
2. 至少包含：
   - base box、左右 arm capsule 的明确 self-mask内外点，全部远离阈值边界；
   - 30 像素坐标同步 fixture：rows `94..99`、columns `198..202`、depth `0.8`、
     identity camera/base、`fx=20, fy=80, cx=189.5, cy=96`；人工期望只有
     column `198` 的 6 个像素被 base self-mask删除，其余24个保留；
   - merge fixture：`A-B-C` 中 A-B、B-C有边而A-C无边，验证传递闭包；包含单视角
     component丢弃和“raw multiplicity 6、unique coordinate 5”的重叠账本。
3. blind worker以 `python -I -S` 执行、清空 `PYTHONPATH`、禁止 import `hwr`；
   blind root不得包含 truth。
4. worker退出并原子封存后，父进程才读取 truth并比较 exact set/partition。
5. mutation至少覆盖：
   - retained/masked coordinate等量互换；
   - row/column交换；
   - duplicate-one/drop-another；
   - 删除 capture ordinal导致跨帧错误去重；
   - component member等量交换；
   - 直接邻接误写为 clique；
   - 忽略 dropped component；
   - pre-self-mask冒充post-self-mask。

### 指标与门

主要指标：

- exact post-self-mask coordinate-set cases；
- exact component partition cases；
- exact retained/dropped cases；
- exact raw-multiset与unique-ledger cases；
- count-preserving mutation kill count。

守护指标：

- fixture/truth hash在 worker实现前冻结；
- worker无 `hwr` import或production helper；
- whole-source token/AST similarity对production均 `<=0.45`；
- 两次 blind output bit-identical仅作为复现门；
- candidate bytes、support count与遍历一致只作守护项，不算truth命中。

判定：

- `accepted`：全部人工 fixture exact，全部 mutation被 exact comparer拒绝，隔离与原子
  封存通过；只授权进入 P88-E2。
- `rejected`：输入与truth有效，但固定worker稳定地产生错误ledger。
- `invalid`：truth由待测代码生成、truth进入blind root、用count/hash替代坐标、
  fixture落在浮点边界、mutation未进入真实路径或worker调用production helper。
- `stop`：非accepted时不得运行真实bank P88，P68继续hard defer。

成本：CPU `<30s`、RSS `<512MiB`、artifact `<4MiB`。主要风险是人工 truth写错；
冻结文档必须展开坐标表、边关系和阈值裕量，并由筛选或红队独立复算。

## `R0001-P88-E2`：fixture-qualified blind v2 coordinate ledger

### 假设与唯一主变量

假设：通过 P88-E1 的固定 worker，可从冻结 P50 bytes重建当前 v2 bank完整、可复核的
coordinate/component lineage。

唯一主变量是新增 evaluator-private lineage sidecar；candidate生成、merge语义、
ranking、score、selection和历史 artifact全部不变。

### 最小验证、指标与门

1. P88-E1必须先accepted，worker与fixture hash保持冻结。
2. blind plan只暴露opaque Episode ID、acquisition pose、capture identity和P50 bytes；
   不提供P79/P83 expected metadata。
3. 每个raw anchor输出：
   - `(planned_episode_id,capture_ordinal,row,column)` identity；
   - sorted post-self-mask coordinate multiset；
   - full-precision raw candidate；
   - component identity与成员。
4. 每个component输出：
   - anchor identities；
   - retained/dropped状态与原因；
   - raw multiplicity；
   - sorted unique coordinate ledger；
   - retained时绑定final ordinal与canonical identity。
5. 两次blind build；独立proof verifier从申报坐标重读depth/valid，独立back-project、
   self-mask、raw candidate和另一种graph traversal。
6. 封存后才揭盲P79 raw/final candidate与P83 selection。
7. 复用P88-E1 mutation，并增加跨candidate交换、edge翻转、capture ordinal碰撞、
   只发布hash、final observation误入generation ledger。

主要指标为coordinate agreement、raw membership、component partition、retained/dropped
mapping和36个final candidate ledger完整率。守护指标为P79 candidate `24/24`、
P83 binding `24/24`、support conservation `36/36`、双运行bit-identical、mutation全过、
private truth read count 0。

- `accepted`：24 Episode与36 final candidate全部完整且exact，只解除P68的coordinate
  lineage前置。
- `rejected`：两条合格source-disjoint路径一致，但无法绑定冻结P79语义。
- `invalid`：E1未accepted、验证路径不一致、提前读取expected/private truth、只比较
  hash/count、坐标缺Episode/capture identity或修改历史artifact。
- `stop`：失败时转向producer-time显式receipt并重建版本化bank，不从candidate bytes
  猜坐标。

成本：CPU `<3min`、process-tree RSS `<2GiB`、artifact `<32MiB`。P68仍需独立
segmentation feasibility、Phase A/B隔离与重新冻结判定；coordinate不是独立样本。

## `R0001-P76-E7`：external B0/B1 action-service capability gate

### 瓶颈证据与假设

legacy P60只靠调用方循环停在399；runner仍可持有覆盖B0–B7的generic callable。假设：
独立、固定blob的action service可把prefix runner能力严格限制在step `0..399`，使B2
action不可请求、不可返回、不可提交。

### 唯一主变量与范围

- action生成从runner内直接调用generic `primitive_action()`，改为调用只授权B0/B1的
  外部service。
- 只新增sidecar service、外部supervisor/IPC receipt与focused tests。
- 不修改production primitive、backend、安全阈值、candidate、selector、geometry、
  snapshot或restore。

### 最小验证与指标

1. synthetic payload/candidate上，step `0..399`返回bytes与production逐字节一致。
2. runner不得import、持有或动态解析generic primitive callable。
3. supervisor记录request、return、proposed、applied与backend apply ordinal。
4. 将循环单变量变异为401次：step 400必须在调用primitive、序列化响应与
   `backend.apply()`之前拒绝。
5. 覆盖负数、400、401、跳号、重复、重连重置ordinal与伪造session/Episode token。

主要指标：

- legal action byte equality `400/400`；
- forbidden response count 0；
- forbidden backend-apply count 0；
- request/return/apply ledger守恒。

守护指标：

- service固定hash、只读staging、source前后稳定；
- runner generic primitive import/call count 0；
- 无MuJoCo、物理action、geometry或restore。

- `accepted`：合法bytes全同，所有越界/绕行mutation在生成bytes前fail closed。
- `rejected`：边界有效，但合法B0/B1不能byte-identical。
- `invalid`：只靠producer自报、runner仍持有generic callable、tripwire位于action
  生成之后或instrumentation改变动作。
- `stop`：任何step `>=400`返回action bytes或进入backend，立即停止且不运行物理cohort。

成本：CPU秒级、RSS `<256MiB`、artifact `<8MiB`。action call/fixture是合同测试单位，
不是Episode或科学样本。

## `R0001-P83-E2`：full-precision selected-candidate authorization handoff

### 瓶颈证据与假设

P83已证明canonical candidate、score hash、selected index与identity，但没有供P76消费
的full-precision authorization。假设：blind worker可在揭盲前封存完整candidate和score
bytes；父进程验证后只发放一个immutable selected-candidate authorization。

### 唯一主变量与范围

- P76 selected candidate来源从隐式重算或直接读bank，改为parent-verified capsule。
- 只新增P83 successor sidecar与schema；不生成action、不运行MuJoCo、不修改generator
  或selector。

### 最小验证、指标与门

1. 在现有24-Episode bank上重复closed-world blind build。
2. worker揭盲前封存所有full-precision candidate bytes、score bytes与selected index。
3. 父进程揭盲后重算canonical、score hash与selection。
4. 22 nonempty发放授权；2 empty必须无授权。
5. mutation覆盖sub-mm center drift、candidate/index swap、acquisition pose、
   Episode/cell、worker hash与authorization replay。

主要指标：canonical/score/index均 `24/24`，authorization `22/22`，empty unauthorized
`2/2`，两次blind build bit-identical。守护指标为揭盲前expected metadata读取0、
private truth/v1/action/backend调用0，且capsule只暴露单个selected candidate。

- `accepted`：全部exact且mutation fail closed。
- `rejected`：source-disjoint重建稳定，但无法形成与冻结score/index一致的full-precision
  authorization。
- `invalid`：quantized candidate驱动动作、提前读取P79 expected、父进程不验证、
  self-signed hash或empty获授权。
- `stop`：揭盲/验证前产生授权，或capsule可跨Episode/session重放。

成本参考P83约25秒，RSS `<2GiB`、artifact `<32MiB`。样本单位为24个Episode；
candidate、score和capture只作Episode内证据。

## `R0001-P76-E8`：precommitted fresh-holdout authoritative prefix

### 假设与唯一主变量

假设：在结果前冻结、与历史seed不相交的新鲜holdout上，v2 nonempty route能以足够比例
在同一backend continuation中安全到达B2-entry。

唯一主变量是cohort identity从已暴露固定bank改为新的预提交Episode seeds。只执行
acquisition+B0+B1；不生成/执行B2、不计算geometry、不snapshot/restore、不修改动作、
安全、candidate或selector。

### 最小验证

1. 结果前提交salt commitment、seed derivation、threshold、source hash及每cell最多
   4个natural-latency-matched attempt。
2. 覆盖12个`task×observation×action latency` cell；每cell选择前两个v2 nonempty
   Episode，形成固定24-Episode route-conditioned prefix denominator。
3. 全部attempt包括empty都进入route-availability ledger，不得替换或删除。
4. 每个nonempty Episode在同一runner PID/backend instance中连续执行995+400 steps；
   中间无reset/snapshot/restore。
5. candidate只来自accepted P83-E2；action只来自accepted P76-E7。
6. output封存前不暴露分层结果。

### 指标与门

主要指标：

- safe entry `>=21/24`；
- 每任务 `>=6/8`；
- 每latency pair `>=5/6`；
- 12/12 cell各至少 `1/2`；
- first-stop reason/step完整。

守护指标：

- route Episode和全部empty attempt与历史ID/seed严格不相交；
- service request精确 `0..399`，B2 request/return/apply均0；
- lineage、randomization、latency、proposed/applied action与P40守恒完整；
- predictive rejection计为有效`safety_stopped`，不删除Episode；
- actual severe collision、nonfinite或invalid force仍为硬停止。

- `accepted`：全部coverage与守护门通过。
- `rejected`：24-Episode route cohort有效完成，但任一预设coverage门未达。
- `invalid`：seed重叠/替换、提前暴露、lineage漂移、任何B2 bytes、reset/restore、
  geometry混入、complete-case删除或instrumentation改变trace。
- `stop_insufficient_route_availability`：任一cell在4个预提交attempt内不足2个nonempty，
  不计算prefix verdict。
- B2 tripwire、actual severe collision或invalid force触发时立即停止并保留证据；
  predictive rejection只结束该Episode。

最大 `48×995 + 24×400 = 57,360` control steps；hard wall `60min`、RSS `<=2GiB`、
artifact `<=256MiB`。这是unseen-Episode证据，不是未见任务/布局OOD泛化。

## 依赖与互斥

- P87-E2、P88-E1与P76-E7均无科学前置，可独立作为本轮候选；三者不能捆绑成一个
  verdict。
- P88-E2严格依赖P88-E1 accepted。
- P76-E8严格依赖P76-E7与P83-E2 accepted。
- P88-E2 accepted只解除P68 coordinate/component lineage前置；不解除segmentation、
  execution-feasibility或统计合同前置。
- P83-E2不依赖P76-E7，但两者都必须在P76-E8前单独accepted。
- 固定bank的22-Episode P76只允许descriptive排障，不得与新鲜holdout混合。
- prefix、geometry与restore互斥，必须分轮或分候选归因。
- P77、selector、默认v2 migration、Replay、Actor、世界模型训练与capability
  evaluation继续no-go。

## 创新 Agent 优先级分歧

- Agent A：`P87-E2` 首选；先修复所有后续实验都会使用的合同内核。
- Agent B：`P88-E1 → P88-E2` 首选；该路线最直接解除association的独立真值缺口。
- Agent C：`P76-E7 → P83-E2 → P76-E8` 首选；先消除B2能力越界，再运行新鲜prefix。
- 主 Agent不在本文件裁决；由两名独立筛选 Agent 对全部提案评分、反驳与排序。

## 冻结声明

本文件提交后视为提案冻结。筛选可以拒绝、延后或要求在 `03-experiment.md` 收窄，
但不得在看到实现或结果后改变提案假设、样本单位、阈值或判定方向。
