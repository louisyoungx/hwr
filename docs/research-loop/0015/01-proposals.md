# R0015 提案

## 生成过程

- 创新 Agent A 独立审计 v2 bank 的 consumer/version contract、默认调用面与迁移风险。
- 创新 Agent B 独立审计 v2 initial association 的 estimand、support reconstruction、
  执行预算与 evaluator-private truth 边界。
- 创新 Agent C 独立审计 v2 bilateral feasibility、legacy P57/P66 外推边界和
  free-base 搜索的反方 stopping gate。
- 三名 Agent 均先阅读 `AGENTS.md` 与 `R0015/00-context.md`，只读检查仓库、历史证据和
  正式 artifact；完成前没有查看彼此输出，没有修改文件、启动训练或运行正式 cohort。
- 主 Agent 只做证据复核、稳定编号、重复合并和冲突标注，不按偏好提前筛选。
- 本轮新观点从 `R0001-P80` 递增；历史 P68、P74、P76、P77 重提时保留原 ID 并使用
  新实验后缀。

## 共识证据

1. 默认 production candidate generator 仍是 legacy v1：
   - `target_selection.CANDIDATE_SCHEMA` 为 `hwr.p41-target-candidates/v1`；
   - v2 仅由独立 `generate_candidate_set_v2()` 暴露；
   - 现有 association、acquisition、target-selection diagnostic、P51/P60 类路径仍
     直接依赖 legacy generator。
2. 内存 `CandidateSet` 不携带 producer/version；score、selector 与 control-index
   函数不检查 schema。仅重标 candidate document schema 并重算自带 hash，现有 legacy
   guard 无法恢复真实 producer lineage。
3. P79 bank 有双根解析要求：
   - `candidate_set.path` 相对 v2 bank 根；
   - `old_candidate_set.path` 和 capture blobs 相对 P50 `source_acquisition` 根；
   - 两类 candidate path 在 24/24 Episode 中同名，不能靠“哪个路径存在”猜测；
   - 当前仓库没有严格的 P79 bank consumer。
4. P79 manifest 已绑定 generator/helper source 与 producer commit，当前 hash 匹配；
   但未来 consumer 尚未强制验证这些字段。
5. 旧 P68 没有可发布 classification：
   - 正式运行在 30 分钟门触发后退出；
   - output 与 `.tmp` 均不存在；
   - v1→v2 后 24/24 candidate-set hash、22/24 selected identity 改变；
   - 旧 P68 runner/reconstructor 明确调用 legacy generator并拒绝 v2。
6. v2 bank 只保存 final candidate、selected index 与 P50 capture 引用，没有 raw support
   pixel。v2 support 必须按 `patch_valid.copy()` 语义独立重建，并以 final canonical
   bytes 证明等价。
7. 384 个 capture 跨 Episode 只有 254 个唯一 `(timestamp_ns, sequence_id)`；capture
   join 必须至少包含 Episode、capture ordinal、clock identity 与输入 hashes。
8. `Candidate.support_count` 是 raw patch pixel multiplicity，不是独立观测；直接 pooled
   计票会重复计算重叠 anchor pixel。association 的统计单位必须是 Episode。
9. 旧 P68 的低门只约束 `stage_compatible_selected <=6`，可能在其余样本大多 ambiguous
   时误发 stopping evidence；也混淆“没有 relevant candidate”和“有 relevant candidate
   但 selector 选错”。
10. v2 的 24 个 `planned_episode_id` 与 P57 的 36 个 ID 交集为 0，P66 anchor 也不在
    v2 bank。P57/P66 只能作为机制证据或守护，不能作为 v2 样本。
11. v2 bank 有 22 个非空 selected Episode，2 个 living Episode candidate empty；
    empty 是 no-target subtype，不能伪装成 bilateral physical negative。
12. v2 bank 不保存 B2-entry authoritative base、full-precision selected target、
    continuation state 或 preposition target。canonical candidate 是量化整数，不能反量化
    后冒充运行时 full-precision target。
13. 探索性只读复算显示：
    - acquisition 时的 policy-visible base 下，22/22 selected pair 至少一臂被球外包络
      排除；
    - 把 base 反事实放到 B1 理想 `0.85m` 对准距离后，22/22 均不再被该外包络排除。
    该已见 sanity bracket 不得作为确认性门，但证明“所有 base pose 下静态不可达”不成立；
    真正未知量是安全、动力学与延迟后的 authoritative B2-entry base pose。
14. B2 base command 为零不等于 base 被物理固定：robot root 是 freejoint，MuJoCo
    dynamics 仍可移动基座。旧 P60 还使用 delayed policy-visible base 与源码常量链长，
    不足以成为 authoritative physical certificate。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P80` | version-sealed candidate artifact resolver | artifact/consumer 合同 | A+B+C 共识依赖 | 待筛选 |
| `R0001-P81` | explicit-generator isolated v2 replay runner | evaluator replay 隔离 | A | 待筛选 |
| `R0001-P82` | default-migration call-site completeness gate | 迁移前审计 | A | 待筛选 |
| `R0001-P68-E2` | v2 initial selected-support association stopping gate | 物理 association 测量 | B | 待筛选 |
| `R0001-P74-E1` | P68 evaluator-only wrist-render suppression | evaluator 执行优化 | B | 待筛选 |
| `R0001-P76-E2` | authoritative B2-entry fixed-base outer-envelope certificate | 物理必要条件测量 | C | 待筛选 |
| `R0001-P77-E2` | precommitted free-base safe staging positive witness | 有界物理存在性搜索 | C | 待筛选 |

## `R0001-P80`：version-sealed candidate artifact resolver

### 瓶颈证据与假设

当前没有 P79 bank 的严格 reader；同名路径必须按角色使用不同 artifact root；内存
`CandidateSet` 不携带 producer/version；schema 可被重标后重算自带 hash。假设：若所有
v2 consumer 只能通过一个 fail-closed resolver 获得带 schema、producer lineage 与显式
根角色的 typed envelope，则静默 v1/v2 混用、版本洗白和错误根解析会在消费前被拒绝。

### 唯一主变量与范围

- 唯一变量：artifact 到 typed candidate envelope 的解析/验证合同。
- 新增独立 reader/contract 与测试。
- 不修改 v1/v2 generator、score、selector、association semantics、默认 runtime、
  历史 artifact 或物理行为。

### 最低成本验证

1. 对正式 P79 bank 的 24 个 v2 candidate、24 个 legacy candidate 和 768 个 capture
   blob 做只读完整验证。
2. 显式验证 outer bank schema、inner candidate schema、manifest size/hash、producer
   commit/source identity、selected index、canonical hash 与 root role。
3. 使用内存或临时 fixture 注入：
   - unknown schema；
   - outer/inner mismatch；
   - v2→v1 重标并重算 candidate hash；
   - 错误 root、同名 path fallback 与 path traversal；
   - size/hash drift；
   - selected-index 越界；
   - producer source/commit drift。
4. 禁止自动转换，禁止在多个 root 中选择“恰好存在”的文件。
5. 对正式 bank 重复解析两次，typed envelope canonical receipt 必须 bit-identical。

### 指标与判定

- 主要指标：
  - 24/24 正式 v2 Episode 通过；
  - 所有预注册负控 `100%` fail closed；
  - 每个引用唯一解析为 `(artifact_root_role, schema, producer_identity)`；
  - 两次 typed receipt bit-identical。
- 守护：
  - P50/P68 legacy 24/24 reconstruction hash 不变；
  - P79 bank/blob byte diff 为 0；
  - 默认 `generate_candidate_set()` 仍为 v1；
  - 不调用 MuJoCo、generator、selector、policy 或 classifier；
  - 历史文档 tree 不变。
- 全部门通过：
  `accepted as version-sealed candidate artifact consumer contract`。
- 任一重标、错根、未知版本、path traversal 或 producer drift 仍通过：
  `rejected`。
- 正式 artifact 必须靠路径猜测或缺少不可恢复 lineage：
  `inconclusive_artifact_contract_insufficient`。
- provenance/history 漂移或 consumer 写回 artifact：
  `invalid`。

### 成本、风险、依赖与边界

- 低成本，CPU-only，分钟级，无物理 rollout、训练或正式 capability Episode。
- 风险是把 resolver 过度硬编码到 R0014 绝对路径；应绑定 schema、manifest 字段和显式
  root role，不绑定单个工作区绝对路径。
- 依赖 P79 manifest/bank 与 P50 source manifest。
- 只证明 artifact 可被安全识别和消费；不证明 candidate relevance、selector 改善、
  feasibility、训练价值或家务能力。

## `R0001-P81`：explicit-generator isolated v2 replay runner

### 瓶颈证据与假设

现有 runtime/diagnostic runner 直接依赖 legacy generator；v2 只在 P79 evaluator 内可达，
而旧 P68 support reconstructor 仍保留 v1 mask 行为。假设：显式接收 generator
implementation、candidate schema 与 producer identity 的离线 v2 replay runner，可在不
触碰默认 production 的情况下精确复现 P79 bank并拒绝 legacy fallback。

### 范围、验证与指标

- 只新建 v2-only replay adapter/runner；止于 candidate generation、score 和 selected
  identity，不进入 association classification、动作或物理执行。
- 从 384 个冻结 capture 重建 24 个 v2 Episode。
- 24/24 candidate bytes/hash、score hash、selected index 和 canonical identity 必须
  与 bank 精确一致。
- runtime spy 要求 legacy generator 调用次数为 0；反向 v1 runner也不得调用 v2。
- v1 runner/v2 artifact、v2 runner/v1 artifact、缺失 explicit dependency 全部
  fail closed。
- 守护 legacy replay 24/24 原 hash，默认 import、shared backend、历史 artifact不变。
- 任一 implicit fallback、per-Episode 混版、错误 support provider 或重建不一致：
  `rejected`；provenance 漂移为 `invalid`。

### 成本、风险、依赖与边界

- 低到中成本，无新物理采样；依赖 P80 或等价的 version-sealed envelope。
- 风险是复制旧 runner 形成双份路径验证；应复用统一 resolver。
- 只证明 runner/version 隔离和离线 replay，不授权 association、routing、B0–B7 或训练。

## `R0001-P82`：default-migration call-site completeness gate

### 瓶颈证据与假设

默认 generator 有多个直接调用点，v1/v2 selected identity 已在 22/24 Episode 改变；直接
切换默认 alias 会同时改变历史 evaluator 与动作路径。假设：mutation-sensitive AST
call graph 加 runtime import spy 可在迁移前证明每个 candidate producer 都声明版本，并
捕获新增或别名化的未登记调用。

### 范围、验证与指标

- 本提案只新增迁移就绪审计，不切换默认 generator。
- 枚举当前 producer 调用并分类为 frozen-v1、isolated-v2 或 forbidden。
- 临时 fixture 注入 direct call、import alias、wrapper call 和 dynamic binding；
  审计必须翻转为失败。
- 主要指标：已知调用点 100% 分类；预注册 mutation 100% 捕获；不存在无版本声明但可
  生成 candidate 的路径。
- 守护：默认 v1 symbol、历史 source blob 与 artifact replay 不变。
- 漏掉 alias/dynamic call、依赖硬编码 `passed=True` 或需真实迁移才能通过：
  `rejected`；静态分析不完备且 runtime spy 也不能覆盖：`inconclusive`。

### 成本、风险、依赖与边界

- 低成本；可独立实施。
- 风险是 AST 白名单自证，应以 mutation 与 runtime spy 提供独立反证。
- 通过只表示迁移影响面可枚举，不表示默认迁移安全或能力改善。

## `R0001-P68-E2`：v2 initial selected-support association stopping gate

### 瓶颈证据与假设

v2 selector relevance 未知；旧 P68 无正式结果、只支持 v1，且 v2 已改变 candidate 被测
对象。假设：在固定 v2 bank 中，多数 Episode 的 initial selected candidate，其唯一
capture-pixel support 与冻结 initial microinteraction allowed entity 相符。

### 唯一 estimand 与范围

- Episode 是唯一统计单位。
- 对 selected candidate 构造唯一
  `(planned_episode_id, capture_ordinal, row, column)` support set。
- allowed-label ratio `>=0.80` 为 positive；known-incompatible ratio `>=0.80` 为
  negative；其余为 ambiguous。
- candidate empty 计入 negative，但保留 `candidate_set_empty` subtype。
- multiplicity-weighted ratio只作预注册敏感性描述，不参与主门。
- 仅新增 v2 association evaluator/runner/tests；不改 generator、selector、mapping、
  action、安全或默认 runtime。

### 最低成本验证

1. 先做无标签离线 support reconstruction，24/24 candidate bytes/hash/index 与 v2 bank
   一致，36/36 candidate multiplicity support 守恒。
2. 使用双根路径与完整 capture composite key。
3. 做 label-blind 6-Episode timing sentinel；程序不得加载 mapping/interaction模块。
4. 只有预测完整 cohort `<=24min` 才可在冻结 `30min` 门内启动。
5. 固定 24 Episode 每 Episode 只做一次 observer-on replay，与历史 P50 trace receipt
   比较；全部 replay/provenance 完成后才统一 classification。
6. private-truth taint mutation 只能改变报告，policy input、candidate bytes/index、
   action 与 physics trace必须不变。

### 指标与判定

- 高关联：
  - positive `>=18/24`；
  - 每任务 `>=5/8`；
  - ambiguous `<=2/24` 且每任务 `<=1/8`。
- 低关联 stopping evidence：
  - explicit negative `>=18/24`；
  - 每任务 `>=5/8`；
  - ambiguous 使用相同上限。
- 其他：`inconclusive`。
- 守护：
  - 24 个唯一 Episode、每任务 8；
  - v2 schema/hash/index 与 support完整；
  - sidecar on/off 或历史 receipt identity一致；
  - mapping/interaction hashes冻结；
  - 不落 partial artifact。
- 任一静默转换、错根、capture 错绑、candidate drift、deadline 越界、private truth 影响
  行为或结果后调门：`invalid`。

### 成本、风险、依赖与边界

- 无训练；约 24×995 step，wall `30min`、RSS `8GiB`，正式运行前需 label-blind preflight。
- 依赖 P80/等价 consumer guard、P79 bank、P50-E4 mapping、P61/P72 initial annotation。
- 风险是 GL 抖动、support tracer 漂移，以及旧 P50 capture不能代表新鲜 v2分布。
- 低门只能称 `v2 candidate-conditioned route stopping evidence`，不能把 generator
  coverage 与 selector错误混为一谈。
- 不得称物体识别、selector因果改善、可达、可交互、安全、闭环成功或泛化。

## `R0001-P74-E1`：P68 evaluator-only wrist-render suppression

### 瓶颈证据与假设

通用 backend 每步渲染四路相机，而 P68 projection只消费 head RGB-D。R0014 探索值显示
禁用 wrist render 可能约 `1.67×` 加速，但该值不是确认性结果。假设：P68 专用路径固定
禁用两路 wrist GPU render，可保持 P68 projected input与物理 trace不变并满足运行预算。

### 范围、验证与指标

- 唯一变量：是否执行 left/right wrist render；不同时改变 segmentation时机、并行度、
  generator 或 association规则。
- 新增 P68 专用 projection/backend与测试，不改共享 production backend。
- 结果前固定三任务×latency 1/2 的 6 个 sentinel；full-camera/head-only成对运行，
  classification 与 mapping模块不得加载。
- runtime guard 对任何 wrist payload访问立即失败。
- 接受门：
  - sentinel paired wall-time 降低 `>=30%`；
  - 预测完整 cohort `<=27min`；
  - head RGB/depth/calibration、policy input、candidate-visible、candidate bytes/index、
    capture identity、action/physics trace `100%` 一致。
- 隐藏 wrist consumer、任一 projected byte/trace漂移、结果依赖 task/seed动态切换或
  预测仍超预算：`rejected`；timing噪声无法判定为 `inconclusive`。

### 成本、风险、依赖与边界

- 低到中成本，无训练；依赖冻结 P68 projection字段和 v2 consumer合同。
- 风险是 evaluator observation 分叉与静态扫描漏动态访问。
- 只能称 `P68 execution-ready`；不得声称完整 observation bit-identical、association
  或能力改善。

## `R0001-P76-E2`：authoritative B2-entry fixed-base outer-envelope certificate

### 瓶颈证据与假设

v2 physical feasibility 未知；legacy P57/P66 与 v2 Episode identity不相交。假设：在冻结
v2 selected Episode 的现有 B0/B1 安全到达 authoritative B2-entry 姿态下，多数 target
pair 至少一臂违反由 compiled model 推导的严格 arm-chain 外包络必要条件。

### 唯一主变量与范围

- 唯一 estimand：同一 authoritative B2-entry 瞬时物理基座姿态下，双臂 target 是否
  被 fixed-base 最大链长外包络排除。
- 新增版本化 evaluator/bridge/tests；不修改默认 generator、selector、primitive、
  backend、安全或动作。
- 不执行 B2；不把 fixed-base certificate 外推到 free-base dynamics。

### 最低成本验证

1. 确定性重放 24 个 P50 acquisition，capture identity 与 v2 candidate bytes/hash
   24/24 匹配；full-precision candidate 必须由绑定输入和 v2 generator重建。
2. 22 个非空 Episode 执行冻结 B0/B1；2 个 empty单列，不计 physical negative。
3. 在 B2 action 生成前同时保存 authoritative state、policy-visible input/queue、
   base qpos/qvel、actual shoulder 与 full-precision target。
4. 最大链长从 compiled MuJoCo ancestor tree保守提取。
5. 正控制、外包络外 epsilon负控制及 shoulder/target/model mutation必须翻转 verdict。

### 指标与判定

- 主要指标：
  - `safe_b2_entry_count`；
  - `hard_outer_excluded_pair_count/22`；
  - 每任务分账与 pair minimum margin。
- 建议支持门：
  - total `>=19/22`；
  - living/dining/kitchen 至少 `4/6、6/8、6/8`。
- 建议拒绝门：
  - total `<=7/22`；
  - 各任务不超过 `3/6、4/8、4/8`。
- 中间为 `inconclusive`。
- 守护：
  - 24/24 lineage；
  - v1 generator调用 0；
  - target双重重算；
  - authoritative/policy-visible base差异显式报告；
  - safety、nonfinite、source drift、缺失 continuation均 fail closed。
- 任一 selected Episode 无安全 B2 entry、compiled chain无法保守提取或使用 delayed
  base冒充 physical base：实验 `invalid`，不得发布比例。

### 成本、风险、依赖与边界

- 中等；最多约 `32,680` control steps，无训练、无 B2/contact。
- 依赖 v2 consumer合同；可能被 P66 类 B1 safety 拦截。
- 仍是旧 P50 seed，不是修复后新鲜未见 cohort。
- 最多证明冻结 v2 lineage 在实际瞬时固定基座下被/未被外包络排除；不证明 IK、
  collision-free path、free-base 动态不可达、entity relevance、接触或任务能力。

## `R0001-P77-E2`：precommitted free-base safe staging positive witness

### 瓶颈证据与假设

探索 bracket 表明 base repositioning 可改变 outer-envelope verdict。若 P76 证明多数 pair
需要重布置基座，假设：只改变 B1 base-command generator，在 production free-base
dynamics 与现有安全层下，可找到并在 fresh clone重现安全 staging path，使双臂 target
均不再被 outer envelope排除。

### 范围、验证与指标

- 必须在结果前冻结 action lattice/连续参数化、horizon、objective、beam width、
  tie-break、seed、restart与总 expansion。
- 所有动作经真实 latency、actuator scale、`backend.apply()` 与 two-step predictor；
  禁止直接写 qpos、清 queue或绕过 safety。
- 主要指标：fresh-clone reproducible witness Episode 数、首次安全 staging step、
  末端 base speed、双臂 outer margins、最大 forbidden force。
- P66 anchor 必须继续触发原 `220N` 门；candidate/target不变；branch/restart不是样本。
- 找到预注册数量 fresh-clone witness：
  `accepted as bounded free-base staging existence evidence`。
- 没找到一律 `inconclusive_not_found`；不得称动态不可达。
- 任一未冻结搜索自由度、结果后加预算、绕过 predictor 或直接设 base state：
  `invalid`。

### 成本、风险、依赖与边界

- 中高成本；依赖 P76 显示重布置需求且有可恢复安全 B2-entry snapshot。
- 搜索空间可能预算失控，必须有 hard node/time gate。
- 只证明某个冻结 controller/search 可到达几何允许 staging pose；不证明双臂 IK/B2
  成功、candidate relevance、接触、抓取、任务成功或泛化。

## 提案间依赖与冲突

1. P80 是 P81、P68-E2 和 P76-E2 的共同前置合同；没有等价 fail-closed consumer 时，
   后三者不能正式启动。
2. P81 只解决离线 replay隔离；P68-E2 还需版本匹配的独立 support provider，不能复用
   legacy `_raw_support_at()`。
3. P74-E1 只有在 P68-E2 的 label-blind single-replay preflight仍不能保留足够预算时
   才有必要；不得把 P74与正式 classification同一次归因。
4. P76-E2 与 P68-E2 回答正交问题：几何必要条件不证明 entity relevance，association
   不证明可达。
5. P77-E2 仅能在 P76-E2 后作为 positive witness 路线；bounded search miss永远不能
   成为动态不可达证据。
6. P82 是默认迁移前审计；本轮没有证据授权默认 production migration。
