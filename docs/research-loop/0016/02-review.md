# R0016 独立筛选

## 评审过程

- 筛选 Agent D：独立评审科学价值、证据强度、可检验性与因果归因。
- 筛选 Agent E：独立评审工程可执行性、安全隔离、资源预算与失败模式。
- 两名 Agent 均只读取冻结提交 `315d684` 下的 `00-context.md`、
  `01-proposals.md`、相关代码与已提交 artifact；完成前没有查看彼此评分。
- 两名 Agent 均未修改文件、运行正式 cohort 或启动训练。
- 评分维度均为 1–5；实施成本与回归风险按 `5=低成本/低风险` 计。
- 主 Agent 不按总分机械排序，而是结合未满足依赖、结果可改变的决策数量和本轮单一
  主假设约束选定候选。

## 筛选 Agent D

| 提案 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 合计 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P83` | 4 | 4 | 5 | 5 | 3 | 5 | 4 | 30 | select，唯一首选 |
| `R0001-P84` | 2 | 5 | 5 | 5 | 3 | 5 | 4 | 29 | defer |
| `R0001-P85` | 3 | 4 | 3 | 4 | 2 | 3 | 4 | 23 | reject current form |
| `R0001-P68-E3` | 5 | 4 | 4 | 4 | 3 | 2 | 3 | 25 | defer |
| `R0001-P76-E3` | 5 | 4 | 4 | 4 | 3 | 2 | 3 | 25 | defer |
| `R0001-P76-E4` | 4 | 4 | 5 | 5 | 3 | 5 | 4 | 30 | defer，硬依赖未满足 |
| `R0001-P86` | 3 | 5 | 3 | 3 | 2 | 3 | 3 | 22 | reject current form |

### D 的关键意见

1. P83 以最低成本判定 P68/P76 的共同 lineage 前置是否真实存在，且 candidate、
   score、index、identity 都有 exact gate；它本身不产生 association、feasibility
   或能力结论。
2. P84 能补新的 producer event，但不能追溯性证明旧 P79 的两次 score 计算原子一致；
   当前 scorer 又没有 divergence witness，决策价值低于 P83。
3. P85 的 6 sentinel 只按 task×observation latency 分层，遗漏 action latency；
   capture count 也不是当前 eager per-observation segmentation 的 wall-time 上界，
   不能按当前设计宣称 execution-ready。
4. P68-E3 直接回答高价值语义问题，但必须先有 P83 和有效的预算门；其 mixed stopping
   evidence 还需明确每任务门，不能在同轮把 lineage、执行优化、support 与 classification
   捆绑。
5. P76-E3 是 v2 physical prefix 的必要基础，但实现和运行范围大；P76-E4 设计清晰、
   成本低，却硬依赖 E3 accepted，不能因总分相同而提前选择。
6. P86 的三步 hold 只能覆盖 latency queue flush；未触发的 servo、predictor、
   contact、task counter 字段即使遗漏也可能假通过，当前不能称 full-runtime restore。

### D 对 P83 的冻结前要求

- oracle、测试和配置不得嵌入正式 score hash、selected index 或 selected identity。
- oracle 不得调用 production generator、merge、score、selection helper。
- 共用二进制格式或纯数学原语若不可避免，必须列为共享可信边界，不能暗称完全独立。
- metadata 隔离只允许表述为冻结执行路径未读取，不得升级为恶意代码安全沙箱。
- 正式运行前验证 P79 manifest 绑定的 796 个 P50 输入文件；输入缺失或 hash 漂移必须
  fail closed。

## 筛选 Agent E

| 提案 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P83` | 5 | 4 | 4 | 5 | 3 | 3 | 4 | select，唯一首选，需收紧合同 |
| `R0001-P84` | 3 | 5 | 5 | 4 | 4 | 4 | 4 | defer |
| `R0001-P85` | 4 | 4 | 4 | 5 | 2 | 3 | 4 | defer，P83 后优先 |
| `R0001-P68-E3` | 5 | 5 | 4 | 4 | 3 | 2 | 3 | defer |
| `R0001-P76-E3` | 5 | 4 | 4 | 4 | 3 | 2 | 2 | defer |
| `R0001-P76-E4` | 4 | 4 | 5 | 5 | 2 | 5 | 4 | defer |
| `R0001-P86` | 3 | 5 | 3 | 4 | 3 | 2 | 2 | defer，需重设计 |

### E 的关键意见

1. 当前 P50 根约 `292MiB`，796 个 P79-bound input 共 `303,660,083 bytes`：
   - 本次核验缺失 0、hash 不符 0；
   - 但整个 P50 目录受 `.gitignore` 的 `runs/` 规则排除；
   - Git 中的 hash commitment 不保证未来 checkout 仍能取得 bytes。
2. P83 的 `3min/<2GiB` 只能限定正式 oracle evaluator本身；最近完整 pytest约
   `246.77s`、process-tree RSS约 `2.443GiB`，必须与 evaluator预算分账。
3. P83 应新增隔离 evaluator/app/tests，不修改已接近 800 行的 P79 app，也不修改
   legacy v1 与共享 selector。
4. blind oracle应在独立进程中只收到净化 plan 与 P50 capture副本/链接；进程原子封存
   oracle output 后再由 comparer揭盲。该约束仍只证明冻结路径，不是安全沙箱。
5. P68-E3 的 Phase A 与 Phase B 应进程级分离；P76-E3 不得直接复用会先生成 B2 action
   的 legacy prefix流程。
6. P86还遗漏 runtime RNG state、mutable model randomization、placement/contact ledger、
   cached cameras与 branch isolation 等边界。

## 主 Agent 综合决策

### 入选

`R0001-P83`

选择理由：

1. 两名筛选 Agent 独立同意其为唯一首选。
2. 它以最低物理风险直接裁决一个共同依赖：
   - 若 accepted，P84 从硬前置降为可选 provenance 增强，下一轮可在
     P85→P68-E3 与 P76-E3→P76-E4 之间重新创新和筛选；
   - 若 rejected，可明确具体 consumer 缺少何种不可恢复 lineage，并重新考虑 P84；
   - 若因 P50 bytes 缺失或平台 exact score 不可复现而 inconclusive，也能阻止未经
     证实的下游消费。
3. P83 不需要 MuJoCo、segmentation、B0/B1、训练或高风险 runtime 修改，能维持单一
   主假设和清晰归因。
4. 选择 P83 不是因为总分最高：P76-E4 在 D 的合计同为 30，但其 E3 硬依赖未满足；
   P68-E3 与 P76-E3 科学价值更高，却都会把尚未通过的 lineage 与其他实现变量捆绑。

### 收缩后的 P83 声明

若 accepted，只允许声明：

> 在当前 checkout 可用、由 P79 manifest 绑定的 P50 capture bytes 和冻结的 P68/P76
> 数据需求上，不读取 P79 score/selection metadata 的 source-disjoint oracle可以
> 重建相同 v2 candidate、score 与 selection lineage；因此新增 producer receipt
> 不是这两个具体 consumer 的硬前置。

不得声明：

- P79 artifact 自包含或可从 Git 独立恢复；
- 任意未来 consumer 都不会绕过合同；
- Python 路径构成恶意代码安全沙箱；
- candidate 与任务相关、可达、安全、能被控制或能完成家务；
- score/selector 更优；
- 默认 runtime 已迁移到 v2。

### 其他候选

| ID | 主 Agent 决策 | 理由 |
|---|---|---|
| `R0001-P84` | deferred | provenance 有价值，但 P83 应先回答它是否是具体下游硬阻塞 |
| `R0001-P85` | rejected in current form | sentinel 分层与 wall-time 上界设计不充分 |
| `R0001-P68-E3` | deferred | 依赖 P83 与重新设计的执行预算门，不能同轮捆绑 |
| `R0001-P76-E3` | deferred | 依赖 P83，且真实物理 prefix 实现与风险面较大 |
| `R0001-P76-E4` | deferred | 严格依赖 P76-E3 accepted |
| `R0001-P86` | rejected in current form | 三步 hold 不足以证明 full-runtime continuation完整 |
| `R0001-P77-E3` | not eligible | stopping gate仍不充分，继续 no-go |

## P77 反方结论

冻结提案中的 stopping gate 方向正确，但仍缺少：

1. P68-E3/P76-E4 的具体 accepted decision，而非模糊“不是 distractor/足够多”；
2. 结果前冻结的 eligible cohort、positive witness数、任务分账、node/step/wall/RSS/
   artifact预算；
3. 不参与搜索的 fresh restore对同一 action bytes做盲重放；
4. action ranking 的完整输入白名单与冻结路径 read guard；
5. 非零 latency-sensitive action、predictive rejection、多 clone隔离的 P86 parity；
6. branch/restart/node不得冒充独立样本；
7. timeout、crash、partial witness与 replay divergence 的预注册判定。

因此 P77-E3 本轮不进入实验冻结。搜索 miss 即使未来发生，也只能是
`inconclusive_not_found`，不能证明 free-base动态不可达。
