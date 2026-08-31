# R0020 上下文

## 起点

- 用户于 2026-08-31 明确授权启动 R0020。
- 起始提交：`2ed7edd13b3edb003ba9fd53e04a1b693485eda8`。
- 分支：`feat/research-loop`；启动时 worktree 干净，本地领先远端 4 个提交。
- 当前能力等级：`L0 未通过`。
- 唯一目标：为 `carry_living_room_basket/v1` 建立完整 privileged teacher/oracle
  ceiling。

## R0019 边界

- R0019 已结束且结论为 `invalid`；本轮不修改或续写 `docs/research-loop/0019/`。
- R0019 的独立逐臂 CEM 只优化抓取末态，后续使用固定 transport twist；它只在
  development seed 19001 形成过双臂接触，随后失去接触，未实现完整成功状态机。
- 本轮不复制 R0019 runner 的 source hash、clean-worktree、qualification report 或
  confirmation provenance 门禁。
- R0019 的接触观察可作为开发线索，但不能作为 R0020 成功或能力证据。

## 当前最短板与证据边界

- 最短板仍是最简单正式任务没有完整 L0 oracle ceiling。
- 本轮只做 `development`：允许读取 simulator-private 的机器人、篮子、把手、目标和接触
  状态。
- teacher 必须经正式 16 维动作接口、原始 MuJoCo 物理、`DualArmSafetySupervisor` 和两步
  predictive collision filter 执行；不得 teleport、直接改写权威 `MjData`、绕过安全层、
  修改任务成功条件或改变物理参数。
- confirmation 与 sealed final 均为 `not_run`；本轮不会开启。

## 2026-08-31 重开

- 用户在提交 `f7b27a38b1b2ccbbeba8a8f3783c7feed646b203` 后明确授权重新打开 R0020；
  主假设不变，不创建 R0021。
- `f7b27a3` 明确 attempt 1～3 是代码变化后的 implementation iteration，未达到
  behavior entry，不是独立重复证据，也不足以判定联合规划路线失败。
- 撤销“R0018～R0020 连续三轮无能力进展”的判断：R0018 是旧机制归档，R0019 为
  `invalid`，旧 R0020 尚未 behavior-ready。
- 重开阶段只处理 `acquire`：联合 planner 生成无非法 robot–basket 穿透的可执行近场路径，
  在线 tracker 使用 pad/handle 几何和真实接触反馈完成对中与闭爪。
