# Research Loop 状态

## 当前状态

- 当前研究机制：能力优先，规则见仓库根目录 `AGENTS.md`。
- 当前能力等级：`L0 未通过`。
- 最新能力基线：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`，24 Episode、
  1,600 update、0 success、Actor 未解锁。
- 活跃轮次：无。
- 下一轮：`0019`，必须由用户明确启动。
- 下一目标：在 `carry_living_room_basket/v1` 上建立正常物理和独立安全层下的
  privileged teacher/oracle ceiling；该结果属于 `development`，不冒充可部署能力。

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
