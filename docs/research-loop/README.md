# Research Loop 状态

## 当前状态

- 当前研究机制：能力优先，规则见仓库根目录 `AGENTS.md`。
- 当前能力等级：`L0 未通过`。
- 最新能力基线：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`，24 Episode、
  1,600 update、0 success、Actor 未解锁。
- 最新轮次：`0019`，已结束，结论为 `inconclusive_capability`。
- R0019 结果：paired development seeds `19001`～`19006` 上 baseline 与 teacher 均为
  `0/6` success；teacher 在 `1/6` seed 形成双臂同步接触，最长 `83` step，但 transport
  丢失接触。全部 Episode 为 `0` actual severe collision、`0` safety intervention。
- 100-seed confirmation 未启动，因为 development teacher 已知为 `0/6` success。
- 当前最早稳定阻塞：跨 seed 联合双臂抓取不稳，以及抓稳后的 transport support 不足。
- 下一步：等待用户决定；不得自动启动 L1 或新研究轮。

## 历史归档

`0001`～`0018` 属于旧的 evidence-first 研究机制，已于 2026-08-31 原位归档。由于多项
历史评测把这些路径和 Git tree 当作冻结证据，目录不做物理移动，也不批量改写。

详细索引：`archive/legacy-evidence-loop-0001-0018.md`。

这些历史材料可以用于：

- 复现旧实验和失败；
- 提取已验证的物理、运动学、安全和测量事实；
- 避免重复已否定的假设。

不得用于：

- 把 measurement/contract/oracle 的 `accepted` 当作能力提升；
- 把反复查看、outcome 已暴露的 24-Episode bank 当作密封确认集；
- 自动继承旧候选的批准状态；
- 继续创建多层 oracle、lineage 或 contract 前置链。

## 新轮次导航

新轮只保留四份必要文档：

- `00-context.md`
- `01-experiment.md`
- `02-results.md`
- `03-summary.md`

同一能力假设的诊断、修复和复跑留在同一目录。每轮结束后停止，不自动创建下一轮。
