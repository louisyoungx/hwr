# Research Loop 状态

## 当前状态

- 当前研究机制：能力优先，规则见仓库根目录 `AGENTS.md`。
- 当前能力等级：`L0 未通过`。
- 最新能力基线：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`，24 Episode、
  1,600 update、0 success、Actor 未解锁。
- 最新轮次：`0020`，已按 `f7b27a3` 后的新规则重开并结束；结论为 `abandoned`，仅适用于
  behavior entry 后冻结的当前实现。
- R0020 使用联合底盘/双臂抓取构型规划、联合关键帧路径与 payload-relative 闭环跟踪，
  不沿用 R0019 的独立逐臂 CEM 或固定 transport twist。
- 固定 development seed `19001` 的 3 个历史 physics 运行均为 `0 success`，只到达
  `approach/acquire/failed_hold`，没有形成任一完整双 pad contact；全部为 `0` actual
  severe collision、`0` safety intervention。
- 这三次运行分别发生在代码修改之后，现归类为 implementation iteration 1～3，不是独立
  重复证据；由于未形成连续 10-step 双臂接触并进入 `secure`，它们没有达到差异化机制的
  behavior entry，不构成联合规划路线的可判别失败。
- 静态 planner 能找到四 pad 接近或进入接触的联合末态，但动态 joint waypoint tracker
  原先未把静态解转化为真实抓取。重开实现修复了非 pad–handle 篮子穿透，并用在线
  pad/handle 几何、signed distance 与 contact feedback 完成最后接近和闭爪。
- smoke 011 达到 behavior entry：seed 19001 上进入 `secure`，最大连续双臂接触 `11` step，
  `0` safety intervention、`0` actual severe collision。
- 冻结实现的完整 seed 19001 Episode 执行到 `secure` 后丢失接触，以 `secure_timeout`
  失败；`0 success`、`0` 抬升、`0` 受控目标进展、`0` severe collision。
- 4-seed development cohort、confirmation 和 sealed final 均为 `not_run`。
- 已撤销“R0018～R0020 连续三轮无能力进展”的判断：R0018 是旧机制归档，R0019 为
  `invalid`，R0020 历史 attempt 1～3 均不计数；只有重开后达到 behavior entry 的冻结实现
  计为一个未推进能力的可计数轮次，尚未触发三轮停止。
- 下一步等待用户明确启动；不得自动创建 R0021 或运行 confirmation。

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
