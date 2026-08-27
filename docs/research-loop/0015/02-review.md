# R0015 独立筛选

## 筛选过程

- 两名筛选 Agent 均先阅读 `AGENTS.md`、`R0015/00-context.md` 与已提交的
  `R0015/01-proposals.md`，再核对代码与 artifact。
- 两者独立工作，完成前没有查看彼此评分。
- 全程只读，没有修改实现、运行正式 cohort 或训练。
- 评分维度均为 1–5：
  - 目标价值；
  - 证据强度；
  - 可检验性；
  - 因果可归因性；
  - 通用性；
  - 实施成本，5 表示成本低；
  - 回归风险，5 表示风险低。
- 总分只作参考；主 Agent 根据依赖、边界和信息价值做非机械选择。

## 独立评分

### 筛选 Agent 1

| 提案 | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 总分 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P80` | 4 | 5 | 5 | 5 | 5 | 5 | 4 | 33 | 接受，本轮唯一候选 |
| `R0001-P81` | 2 | 5 | 5 | 4 | 3 | 4 | 3 | 26 | 拒绝独立立项 |
| `R0001-P82` | 2 | 4 | 2 | 4 | 3 | 4 | 3 | 22 | deferred |
| `R0001-P68-E2` | 5 | 4 | 4 | 4 | 3 | 2 | 3 | 25 | deferred |
| `R0001-P74-E1` | 3 | 4 | 5 | 5 | 2 | 4 | 3 | 26 | 条件 deferred |
| `R0001-P76-E2` | 4 | 3 | 3 | 4 | 3 | 2 | 2 | 21 | deferred，需重写失效门 |
| `R0001-P77-E2` | 3 | 2 | 2 | 2 | 2 | 1 | 1 | 13 | 拒绝当前形式 |

### 筛选 Agent 2

| 提案 | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 总分 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P80` | 2 | 5 | 5 | 5 | 4 | 5 | 4 | 30 | 接受，仅工程前置 |
| `R0001-P81` | 2 | 4 | 5 | 5 | 3 | 4 | 4 | 27 | 拒绝独立立项 |
| `R0001-P82` | 2 | 4 | 3 | 4 | 3 | 4 | 4 | 24 | deferred |
| `R0001-P68-E2` | 5 | 3 | 3 | 4 | 3 | 2 | 3 | 23 | deferred |
| `R0001-P74-E1` | 1 | 3 | 4 | 4 | 1 | 4 | 3 | 20 | deferred |
| `R0001-P76-E2` | 4 | 3 | 2 | 4 | 3 | 2 | 3 | 21 | deferred |
| `R0001-P77-E2` | 3 | 2 | 2 | 3 | 2 | 1 | 2 | 15 | 拒绝当前形式 |

## 共同反驳

### `R0001-P80`

两名筛选者都确认真实 blocker：

- v2/legacy candidate path 在 24/24 Episode 中同名，但根角色不同；
- `CandidateSet` 不携带 producer/version；
- 现有 score/selector 不检查 schema；
- P68/P76 的旧路径仍硬编码 legacy v1。

两者也独立指出同一关键反例：

> artifact 内部自报的 `source_commit`、source hash、manifest 与 nested hash 只能证明内部
> 自洽；若攻击者同时重写 artifact、manifest 和自带 hash，纯内部验证仍可通过。

因此 P80 必须从 artifact 外部取得冻结的 Git trust anchor，例如：

- 已提交的 bank/manifest Git blob identity；
- 起始提交；
- 提案与实验文档中结果前冻结的 schema、producer source blob、artifact root role。

接受结论必须收窄为“与冻结可信 receipt 一致”，不得声称证明 artifact 的实际执行历史。
还必须覆盖：

- outer 与 inner schema 同时重标并重算所有自带 hash；
- 同名双根 fallback；
- path traversal 与 symlink root escape；
- producer source drift；
- consumer 绕过 resolver 的 architecture/import gate。

### `R0001-P81`

- P79 已经从同一 384 capture 重建 24 个 v2 Episode并做两次完整 bank build。
- 再实现 standalone replay runner 高度重复，且可能制造第二套路径/provenance验证。
- runtime spy只能证明覆盖路径没有 legacy fallback，不能证明未来 consumer不会绕过。
- `explicit generator/schema dependency` 原则应成为未来 P68/P76 的局部接口要求，不单独
  占用本轮实验。

结论：拒绝作为独立候选。

### `R0001-P82`

- 当前多个 legacy direct call真实存在，但没有默认迁移授权。
- AST 加 runtime spy不能证明 Python whole-program completeness；反射、registry、
  closure、dynamic import和未覆盖分支可逃逸。
- mutation fixture容易只捕获审计者预先设计的形式，造成自证。

结论：deferred。未来重提应先通过 API 收口与 producer registry缩小状态空间，结论只能
覆盖冻结 source tree 和注册集合。

### `R0001-P68-E2`

两名筛选者均认为科学价值高，但当前不可执行：

1. P80 或等价 consumer合同尚未验证；
2. legacy support reconstructor使用 v1 generator和原地 mask行为；
3. final candidate bytes与 support count不能证明 raw support坐标正确；
4. 需要独立 support-coordinate oracle、support-ledger hash和坐标 mutation；
5. candidate empty不能同时充当“selected support negative”：
   - 24 Episode route availability；
   - 22 nonempty selected-support association；
   应拆账；
6. unique-pixel ratio是新 estimand，不能声称旧 `0.80` 门无变化继承；
7. single-replay 虽比旧 baseline+treatment双 replay合理，但逐 step segmentation仍是
   已知成本，应先做 capture-only cloned-state label-blind preflight。

结论：deferred。P80 后重新冻结，不沿用当前未通过评审的门。

### `R0001-P74-E1`

- P68 projection确实只消费 head RGB-D，共享 backend却每步渲染 wrist RGB。
- 但 single-replay可能已足以满足预算；当前先优化 wrist可能解决一个不存在的 blocker。
- 即使实施也只能证明 P68 projection与 trace一致，不能称完整 observation一致。
- timing需 AB/BA、warm-up与重复，不得把 6 个 timing cell冒充独立统计样本。

结论：条件 deferred。只在修订 P68 的 label-blind preflight仍超预算后立项。

### `R0001-P76-E2`

- 改为 authoritative physical base和 compiled-model chain有价值。
- 但当前 P76合同把 `safe_b2_entry_count` 列为指标，同时规定任一 selected Episode无
  safe entry就使整轮 invalid，内部矛盾。
- P66 已证明 B1安全 stop不是边缘风险；不能让一个 unsafe prefix删除其余 Episode证据。
- 应分层报告：
  - safe-prefix coverage；
  - 达到结果前冻结最小 coverage 后，才报告 eligible denominator上的条件式 geometry；
  - coverage不足为 `inconclusive_prefix_coverage`；
  - safety-stopped既不是 outer-excluded，也不能 complete-case静默删除。
- P80 单独通过仍不足以启动，还需 v2 prefix bridge与 full-precision reconstruction。

结论：deferred，重写分母与失效门后再筛选。

### `R0001-P77-E2`

- 没有具体 witness数量、任务分账或明确 rejection门；
- search miss永远是 `inconclusive_not_found`，当前假设不可证伪；
- action lattice、objective、beam、restart、预算与新 command generator变量过多；
- 若 search ranking读取 authoritative geom/contact/force/future safety truth，会成为
  simulator-oracle scripted action，违反 private truth边界；
- fresh clone不是 independent unseen Episode。

结论：拒绝当前形式。

## 主 Agent 决策

### 本轮科学候选

`no-go`

本轮没有任何 proposal 同时满足：

- 前置合同已验证；
- 单一科学假设；
- 可执行且可证伪；
- evaluator-private truth隔离；
- 可解释的 Episode分母；
- 不与工程修复捆绑。

因此本轮不启动 association、feasibility、selector、Replay、Actor、世界模型训练或
capability evaluation。

### 本轮唯一实施项

`R0001-P80`

将其明确归类为：

`engineering/evidence-hygiene prerequisite`

选择原因不是总分最高，而是：

1. P68-E2 与 P76-E2 共同依赖它；
2. 它只改变 artifact消费边界，不改变行为或科学 estimand；
3. 成本最低，失败也能明确暴露 lineage不足；
4. 若跳过它，后续物理预算可能消耗在错根、混版或可洗白 artifact上。

### 本轮不实施

- P81：与 P79 replay重复；其接口原则留给未来 consumer局部实现。
- P82：当前没有默认迁移授权。
- P68-E2：待 P80 后重写 support oracle、capture-only执行与双 estimand。
- P74-E1：待修订 P68 timing preflight证明需要。
- P76-E2：待 P80 后重写 safe-prefix coverage与 geometry分母。
- P77-E2：当前形式拒绝。

## P80 冻结前修订要求

1. 期望 schema、commit、Git blob与 root role必须来自实验文档冻结的外部 trust anchor，
   不能从被验证 artifact自身读取。
2. 复用既有 `read_bound_blob()` 的 size/hash/path traversal能力，不重复实现通用 blob
   reader。
3. resolver必须返回 typed、immutable、带 schema/producer/root role的 envelope；
   下游不得只拿裸 `CandidateSet` 后丢失版本。
4. architecture/import gate至少约束本轮新增 consumer只能通过 resolver读取 P79 bank；
   不声称证明整个 Python程序不存在绕过。
5. 正控覆盖正式 24 Episode，负控至少覆盖：
   - unknown schema；
   - outer/inner mismatch；
   - inner+outer共同重标并重算自带 hash；
   - 错根与存在性 fallback；
   - traversal 与 symlink escape；
   - size/hash drift；
   - selected-index越界；
   - producer commit/source blob drift；
   - trust anchor drift。
6. 接受边界只能是版本密封的 artifact consumer合同，不能宣称 candidate质量、相关性、
   feasibility、默认迁移、训练或能力改善。
