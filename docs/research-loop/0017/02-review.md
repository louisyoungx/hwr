# R0017 独立筛选

## 评审过程

- 筛选 Agent D：独立评审科学价值、统计、合同可达性与因果归因。
- 筛选 Agent E：独立评审工程执行、安全隔离、资源、provenance 与失败模式。
- 两名 Agent 均只以冻结提交 `6f34c51` 下的 `00-context.md`、`01-proposals.md`
  为提案合同，并只读核查相关代码、历史文档与 artifact；完成前没有查看彼此输出。
- 两名 Agent 均未修改文件、运行正式 cohort 或启动训练。
- 评分维度均为 1–5；实施成本与回归风险按 `5=低成本/低风险` 计。
- 主 Agent 不按总分机械选择，而是结合硬依赖、结果已见污染、可归因性与本轮单一
  主假设约束作出决策。

## 筛选 Agent D

| 提案 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P87` | 4 | 5 | 5 | 5 | 5 | 5 | 4 | 唯一首选，修订后冻结 |
| `R0001-P85-E1` | 3 | 3 | 5 | 3 | 2 | 2 | 3 | deferred |
| `R0001-P88` | 4 | 4 | 5 | 4 | 4 | 5 | 4 | deferred，次优先 |
| `R0001-P68-E4` | 5 | 3 | 4 | 3 | 4 | 5 | 3 | 当前 no-go |
| `R0001-P76-E5` | 5 | 3 | 4 | 3 | 5 | 2 | 3 | deferred，先处理已见结果 |
| `R0001-P76-E6` | 4 | 3 | 5 | 4 | 3 | 5 | 3 | 当前 no-go |
| `R0001-P86-E1` | 4 | 5 | 3 | 4 | 4 | 1 | 1 | deferred，当前 no-go |

### D 的关键意见

1. P87 无硬依赖、成本最低，并已由当前两个结构反例证明有现实用途；它能在任何正式
   物理运行前阻止不可达门或分层坍缩。
2. P68-E4 的总体/任务门可达，但可在某个 latency pair 仅 `1/5` positive 时通过：
   - 当前合同若不增加 latency floor，就必须显式禁止 latency-robustness 声明；
   - 不能把总体/任务结果解释为跨 latency 稳健。
3. P76-E5 的总量、task、latency-pair 与 12-cell 门联合可达。D 对最多三个失败
   Episode 做穷举，存在 `585` 个 accepted assignment。
4. P76-E5 在冻结前已看到三个 `obs=1/action=1` safe-entry sentinel：
   - `45b8ab11...` living；
   - `7e039594...` dining；
   - `c8a3a55e...` kitchen。
   当前合同没有登记其完整 Episode ID、已见字段、是否进入分母或重复次数，因此不能
   继续把完整 22 Episode 当作未见确认性 cohort。
5. P85-E1 相对 legacy P68 同时改变 v1→v2、24→22、双 replay→单 replay、同次
   baseline→历史 P50 receipt；可测绝对执行可行性，但不能把 wall 改善因果归于单一
   comparator 变量。
6. P88 没有外部 coordinate ledger 真值；candidate bytes、support 总数与三遍历一致
   仍可能稳定复制同一个坐标归属错误，必须补手工可知的合成 component/self-mask
   controls。
7. P86 动机充分但范围接近完整 runtime serialization，当前成本与遗漏风险过高。

### D 的范围偏离

D 的返回中额外列出未出现在冻结提案总表的 `R0001-P89` 与 `R0001-P84-E2`，并给出
缺失合同惩罚分。主 Agent 将这两行视为范围外输出，不纳入本轮评分、筛选或后续状态。

## 筛选 Agent E

| 提案 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P87` | 3 | 5 | 5 | 5 | 5 | 5 | 4 | deferred，作为冻结审查工具 |
| `R0001-P85-E1` | 3 | 4 | 4 | 4 | 3 | 2 | 3 | deferred |
| `R0001-P88` | 4 | 4 | 3 | 4 | 4 | 5 | 3 | deferred，先补独立真值 |
| `R0001-P68-E4` | 5 | 4 | 4 | 4 | 3 | 2 | 3 | hard defer |
| `R0001-P76-E5` | 5 | 4 | 4 | 4 | 4 | 2 | 3 | 条件首选 |
| `R0001-P76-E6` | 4 | 3 | 4 | 5 | 2 | 5 | 4 | hard defer |
| `R0001-P86-E1` | 3 | 5 | 3 | 3 | 4 | 1 | 1 | reject current form |

### E 的关键意见

1. E 条件选择 P76-E5：prefix coverage 无论高低都能直接缩小后续 geometry/search
   路线；P87 只做合同 lint，不产生物理 prefix 证据。
2. P76-E5 当前存在 live cohort 账本矛盾：
   - 一处要求 24/24 acquisition identity；
   - 一处又规定两个 empty 只离线核验，总 live steps 写为 `22×1395=30,690`。
   若 empty 不 live replay，应冻结为 22/22 live acquisition 与 24/24 offline
   selection lineage；若坚持 24/24 live，则总步数应为 `32,680`。
3. parent 只包装 `primitive_action` 仍不足以独立证明 no-B2。冻结前需要：
   - 同 PID、同 backend、无 reset/snapshot/restore；
   - action 服务只接受 post-selection step `0..399`；
   - runner 不持有能生成 B2 的 callable；
   - supervisor 观察 action request/return/proposed/applied；
   - 401-step mutation 在 B2 action bytes 产生前失败。
4. P76 selection 揭盲应复用 P83 已接受的 fixed worker blob，而不是再写近似 selector；
   prefix runner 只接收通过验证的 candidate 授权，不读取 expected index/identity。
5. Phase A 需要文件系统级 blind root。只检查 import/read count 不足以隔离
   mapping、interaction、segmentation label 与任务实体真值。
6. wall 必须由外部 supervisor 从进程启动测至 `fsync + atomic rename`；RSS 使用
   parent-observed process-tree 峰值；evaluator、focused tests 与 full pytest 分账。
7. P88 必须加入父进程独立重建与手工精确 coordinate/component fixtures。
8. P86-E1 缺少明确资源上限、case 数、continuation horizon 与数值容差，当前不可执行。

## 主 Agent 综合决策

### 入选

`R0001-P87`

选择理由：

1. P87 不是按总分机械入选。D 将其列为唯一首选；E 虽倾向 P76-E5，也确认 P87 的
   证据强度、可检验性、因果归因与通用性均为 `5`。
2. P76-E5 在冻结前仍有四个实质 blocker：
   - 三个已见 safe sentinel 没有 exposure policy；
   - 22/24 live acquisition 与 step budget 自相矛盾；
   - no-B2 仍可能由 producer 自证；
   - metadata/semantic truth 的文件系统隔离与外部资源观测尚未冻结。
   这些不是实施细节，不能边实现边补合同。
3. P87 能在秒级、零 MuJoCo 风险下把上述缺口转为机器可执行证据，并审计 P68/P76
   当前和修订合同的 verdict 可达性、分母守恒、分层最坏值与结果暴露处置。
4. 当前已出现两个由人工审计才发现的结构错误：
   - P68 selector-negative `18/22` 在 8 个 multi-candidate Episode 下不可达；
   - P76 pooled `19/22` 可隐藏 latency pair `3/6`。
   若不先固化该类门，继续正式物理 cohort 会重复消耗资源并产生不可解释结果。
5. P87 不修改 candidate、selector、动作、安全、runtime 或训练，能严格维持本轮单一
   主假设。它的结果只决定实验合同是否具备进入后续轮次的资格。

### 冻结前收缩

P87 实施时必须：

1. 使用在实现前提交的机器可读 contract registry；阈值、分母、claim strata 与
   exposure policy 不得重写在 oracle 源码中。
2. 将 registry SHA-256 与 `03-experiment.md` blob 固定为实现外部 trust anchor。
3. 至少使用两条独立检查路径：
   - 解析式 structural upper-bound/reachability；
   - assignment enumeration 与 witness validation。
4. formal 输入只含 P50/P79/P83 身份与结构，不读取真实 association/prefix outcome。
5. result exposure ledger 只登记 Episode ID、已见字段、来源与分母政策；oracle 不读取
   或重放已见 outcome value。
6. 明确 sample unit 与 denominator type；candidate/frame/pixel/arm/control step
   冒充 Episode 必须 fail closed。
7. 同时测试：
   - 不可达 selector gate；
   - pooled gate 的 latency collapse；
   - 加入 latency floors 后反例被排除；
   - empty/nonempty denominator 漂移；
   - 缺失 exposure policy；
   - contract/oracle 同源或输入漂移。
8. P87 acceptance 依赖 solver agreement、witness validity、mutation coverage 与
   provenance，不依赖某个 formal contract 被接受还是拒绝，避免把预期结论写成
   自证通过门。
9. 只允许 `accepted as frozen experiment-contract oracle` 声明；不得产生
   association、prefix、安全、reachability、泛化或能力结论。

### 其他候选

| ID | 主 Agent 决策 | 理由 |
|---|---|---|
| `R0001-P85-E1` | deferred | 只能称绝对执行可行性；需重命名并冻结外部计时/负载/GL/近门重复规则 |
| `R0001-P88` | deferred | 需手工精确 coordinate/component oracle 与 parent verification |
| `R0001-P68-E4` | hard defer | 依赖 P85/P88，且需明确 latency claim scope 与 negative task 门 |
| `R0001-P76-E5` | deferred | 价值高，但先修 exposure、cohort、no-B2、blind root 与资源合同 |
| `R0001-P76-E6` | hard defer | 严格依赖 P76-E5 accepted，并需另轮冻结 compiled-chain 定义 |
| `R0001-P86-E1` | rejected in current form | 范围过宽，缺 case/horizon/tolerance/resource 上限 |

## 本轮 no-go

- 不实施 P85/P88/P68/P76/P86 的代码或正式 cohort。
- 不运行 MuJoCo physical acquisition、B0/B1/B2、geometry、restore 或训练。
- 不启动 selector、默认 v2 migration、Replay、Actor、世界模型训练或 capability
  evaluation。
- P77 继续 no-go。
