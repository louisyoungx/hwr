# R0011 独立筛选

## 过程与输入

- 冻结提案：`docs/research-loop/0011/01-proposals.md`
- 筛选输入 SHA-256：
  `6eef0f248c9d93bf17172fdd4a19e5fd09217f9c5096a13db954a3b5e138c54c`
- 两名筛选 Agent 收到相同摘要，完成前未查看另一人的输出。
- 两者均未修改文件、未运行实验、未启动训练。
- 评分维度：目标价值、证据强度、可检验性、因果可归因性、通用性、实施成本、回归风险；
  每项 1–5，实施成本与回归风险的 5 分表示低成本、低风险。

## 筛选 Agent 1

| ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P50-E3` | 5 | 5 | 5 | 4 | 4 | 4 | 4 | accept |
| `R0001-P57` | 4 | 3 | 4 | 3 | 3 | 5 | 5 | accept，限诊断 |
| `R0001-P58` | 4 | 2 | 4 | 5 | 4 | 3 | 4 | defer |
| `R0001-P59` | 2 | 1 | 3 | 2 | 2 | 2 | 1 | reject |
| `R0001-P56` | 4 | 3 | 4 | 4 | 4 | 4 | 5 | accept，report-only |

主要反驳：

- P50-E3 即使证明 entity coverage 不足，也不能证明提高 coverage 会提高成功率。
- entity class 不能冒充实例；遮挡碎片、mixed/unknown、visual/collision 映射错误都可能
  制造虚假漏斗损失。
- P57 的双臂距离与 command budget 只是几何代理，不证明 IK、碰撞、法向或接触可行。
- P58 若从结果中挑有利 continuation、没有完整状态克隆或把 hold 当天然中性，会破坏
  因果归因。
- P59 将 simulator-private segmentation 放入正式 generator，是不可部署 oracle 与
  评测泄漏风险；不能把 P50-E3 的 evaluator 顺势升级为行为输入。
- P56 的正确实体接触仍可能只是擦碰或被动碰撞，不等于抓取或受控运动。

冻结前修订：

1. P50-E3 冻结实例 identity、可见阈值、分母、Episode 聚合、stage 顺序与
   mixed/unknown 规则。
2. P57 readiness 必须是同一时刻或结果前时间窗内双臂同时成立，不能分别挑最佳帧。
3. P56 如执行，必须冻结 phase、部位、力/持续时间、多接触与被动碰撞规则。
4. P58 必须冻结快照、共同随机性、hold 语义、重复次数、主要指标与安全守护。
5. private segmentation 必须有可执行的硬隔离检查。

## 筛选 Agent 2

| ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 实施成本 | 回归风险 | 总分 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P50-E3` | 5 | 4 | 5 | 5 | 4 | 4 | 5 | 32 | accept，主实验 |
| `R0001-P57` | 4 | 3 | 5 | 3 | 4 | 5 | 5 | 29 | accept，辅助离线分析 |
| `R0001-P56` | 3 | 3 | 4 | 4 | 4 | 4 | 5 | 27 | defer/条件执行 |
| `R0001-P58` | 4 | 2 | 4 | 5 | 3 | 3 | 4 | 25 | defer |
| `R0001-P59` | 3 | 1 | 3 | 2 | 2 | 3 | 1 | 15 | reject |

主要反驳：

- P50-E3 的映射错误或重跑轨迹漂移可能被误判为 generator 漏斗错误；unknown 必须与真实
  删除分开。
- P57 必须使用 actual applied command，不能把 requested path 当执行预算；距离与预算
  也不证明姿态、IK、碰撞或接触法向 ready。
- P56 与 P50-E3 的 entity 维度、P57/P58 的 part/bilateral 维度部分重叠；本轮默认
  defer，只有零行为影响、近零额外成本时才可作探索附表。
- P58 的 hold 有多种语义，状态克隆、接触 solver 和残余 controller state 都可能破坏
  paired identity。
- P59 同时改变 segmentation 候选源和 merge/conflict resolution，不能归因；部署不可得
  的 simulator segmentation 是 oracle shortcut。

冻结前修订：

1. P50-E3 明确定义 task-entity 分母、visible、raw/component/final coverage、首次删除和
   mixed/unknown 独立统计。
2. 冻结 24 Episode、seed、初态、配置和轨迹指纹；复现漂移使实验 invalid。
3. body/geom/entity 映射先独立验证，并报告 mapping coverage、unknown 与冲突率。
4. P57 冻结距离几何、actual-applied budget、双臂同步窗口、readiness 数值门和缺失规则。
5. 结果前写明 P50-E3/P57 结论允许触发的下一步；禁止看结果后补门。

## 共识

两名筛选 Agent 独立达成：

1. `R0001-P50-E3` 是最直接解决 `R0011-C01` 的候选，应为主实验；
2. `R0001-P57` 可作为低成本、无行为变化的辅助诊断，但不能替代 entity coverage；
3. `R0001-P58` 当前依赖未满足，应 defer；
4. `R0001-P59` 当前形态使用 evaluator-private truth 修改行为，应 reject；
5. 本轮不得启动 selector、Replay、Actor 或世界模型训练。

分歧：

- Agent 1 建议并行实施 P56 report-only 合同；
- Agent 2 建议默认 defer，只在零行为影响、近零额外成本时作探索附表。

## 主 Agent 裁决

主 Agent 不按总分机械选择，而按主瓶颈、变量隔离、依赖与实现面裁决：

| ID | 裁决 | 理由 |
|---|---|---|
| `R0001-P50-E3` | **selected，唯一主实验** | 直接回答 task entity 从可见到 final candidate 的首次损失位置 |
| `R0001-P57` | **selected，独立辅助实验** | 复用受保护 artifact、低成本、与 P50-E3 文件和结论解耦 |
| `R0001-P56` | **deferred** | acquisition-only P50-E3 不使用 contact；并行实现会扩大变量与验收面 |
| `R0001-P58` | **deferred** | P50-E3、P57、P56 前置均未通过，不能冻结 continuation 或 contact 指标 |
| `R0001-P59` | **rejected** | simulator-private segmentation 进入正式 candidate，违反单向隔离并捆绑测量与行为修订 |

P50-E3 与 P57 是两个独立结论：

- P50-E3 回答候选是否覆盖 task entity；
- P57 回答既有 P51 cohort 的双臂 preposition 几何/执行预算是否同时 ready；
- 不用任一结果补写或改变另一实验的门槛；
- 任一通过都不授权训练、selector、Replay 或行为修订。

## 未入选项的重新进入条件

- P56：只有后续候选使用 contact/onset/yield 为指标时重新筛选。
- P58：P50-E3 证明 entity-hit 支持、P57 证明 readiness 支持、P56 接受后重新冻结。
- P59：必须替换为部署可得的 RGB-D segmentation 模型，隔离训练数据与未见评测，并把
  candidate source 与 merge strategy 拆成单变量提案；不得使用 simulator private truth。
