# R0019 上下文

## 起点

- 用户于 2026-08-31 明确授权启动 R0019。
- 起始提交：`0df823abb602c528b7b138ab17fb6bc9097b3048`。
- 分支：`feat/research-loop`。
- 当前能力等级：`L0 未通过`。
- 最新完整三维学习基线：24 Episode、1,600 update、0 success，Actor 未解锁。
- 唯一目标：为 `carry_living_room_basket/v1` 建立正常 MuJoCo 物理和独立安全层下的
  privileged teacher/oracle ceiling。

## 历史证据

- `R0001-P57`：旧 B2 preposition 阶段 36/36 pair 的两臂 command margin 都为负，
  `ever_bilateral_ready=0/36`；100 step applied command budget 约 `0.349–0.433m`，
  明显小于起始目标距离。该证据只说明旧 B2/B3/B4 时域不是可行控制上界。
- `R0001-P61`：generic `Candidate` 与 B0–B7 primitive 缺少 `entity_role`、
  `interaction_type` 和 `destination_target`，不能唯一表达完整任务转移。
- 本轮不继续 P88、P76、P68，也不增加新的 oracle、contract 或 lineage 资格链。

## 当前系统

- 动作：16 维，底盘线/角速度、左右臂各 6D base-frame tool twist、左右夹爪目标。
- 控制：MuJoCo backend 用 Jacobian damped least squares 将 tool twist 转成关节速度目标。
- 安全：所有动作先经过 `DualArmSafetySupervisor`，再经过两控制步预测碰撞检查；
  forbidden robot contact 达 `220N` 时动作被替换为 hold。
- 成功：需要曾连续双臂接触至少 10 step、双臂接触控制下完成目标位移，并在目标支撑上
  满足位姿/速度门连续 40 step；actual severe collision 必须为 0。

## 证据边界

- 本轮是 `development`，允许读取 simulator-private state、对象/目标位姿和接触状态。
- teacher 只能通过正常 16 维动作接口进入同一 MuJoCo backend；不得 teleport、改写状态、
  关闭或削弱安全层、放宽成功条件或修改物理。
- development seeds 可反复查看；confirmation seeds 在实现、门槛和预算冻结后才运行。
- R0001–R0018 的 24-Episode bank 只作历史开发证据，不进入本轮确认集。
