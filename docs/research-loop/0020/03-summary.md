# R0020 总结

状态：结束。

- 当前能力等级：`L0 未通过`。
- 结论：`abandoned`。
- confirmation：`not_run`。
- sealed final：`not_run`。

## 结论

R0020 实现了不同于 R0019 的联合双臂路线：

- 在复制的 MuJoCo state 上联合优化底盘接近位置和两臂 12 个关节；
- 同时优化四个 finger-pad 到两侧 handle 的 signed distance，并检查联合插值路径；
- 在线使用关键帧 joint tracking 和抓取后的 payload-relative Cartesian feedback；
- controller 明确实现
  `approach/acquire/secure/lift/target_transport/place/release/stabilize`，
  不再使用固定 transport twist。

但固定 development seed 19001 的 3 个完整 Episode 均为 `0 success`。三次都只到达
`approach/acquire/failed_hold`，没有形成任一完整双 pad contact，最早失败阶段为
`acquire_timeout`。全部 Episode 为 `0` safety intervention、`0` actual severe collision。

静态 planner 能找到四 pad signed distance 接近或进入接触的联合末态，focused test 也确认
planner 不修改权威状态；但 joint-space waypoint tracker 没有把该静态解转化为真实动态抓取。
因此当前路线没有资格评价后续 lift、transport、place、release 或 stabilize，也不能建立
L0 oracle ceiling。

按预注册 3-Episode 预算停止本 candidate。单 seed 升级门未通过，未扩到 4-seed
development cohort，也未启动 confirmation。

## 允许声明

- 当前联合静态 planner 能在复制 state 中找到满足四 pad 距离目标的候选构型。
- 当前 joint-space waypoint closed-loop tracker 在固定 development seed 19001 的三次正式
  physics Episode 中均未形成完整双 pad 接触。
- 三次 Episode 没有 safety intervention 或 actual severe collision。

## 不允许声明

- 不允许声明 `carry_living_room_basket/v1` 已有可行 L0 teacher ceiling。
- 不允许用本轮结果否定联合 motion planning、轨迹优化或 demonstration-backed tracking
  整体路线；本轮只否定当前“静态 joint CEM + joint waypoint tracker”实现。
- 不允许把 attempt 1 的非受控高度变化称为抓取或抬升成功。
- 不允许归因尚未真正执行的 lift、target transport、place、release 或 stabilize。
- 不允许声明任何可部署状态策略、视觉策略、泛化或硬件能力。

## 保留与回退

- 保留 development-only planner、controller、runner、focused tests 与三个原始 artifact，
  用于复现静态解到动态接触之间的缺口。
- 不修改 `docs/research-loop/0019/` 或 `0001/`～`0018/`。
- 不保留任何 confirmation 门禁；本轮 confirmation 与 sealed final 均为 `not_run`。
- 不自动启动下一路线或 R0021。

## 用户决策

R0018～R0020 已连续三轮没有能力阶梯进展，触发研究循环强制停止：

- R0018：coordinate oracle 资格链在正式 physics run 前被终止，没有能力结果；
- R0019：独立逐臂 CEM 能在 `1/6` seed 形成双臂接触，但未实现完整状态机且 transport
  丢失接触；
- R0020：联合静态抓取规划能找到几何末态，但 `0/3` 固定 seed Episode 形成完整双 pad
  接触。

未经用户明确决定不得启动 R0021。下一次也不能继续只细化同一抓取 artifact；建议用户在
以下方向中选择：

1. 缩小任务或加入更明确的 grasp fixture/末端执行器约束；
2. 引入 demonstration/遥操作轨迹，以 task-space trajectory optimization 或 MPC 跟踪替代
   当前 joint waypoint controller；
3. 改变篮子抓取机构或机器人架构；
4. 终止该正式任务路线。

## 资源分配

主要资源用于 planner/controller 实现、三次完整 MuJoCo behavior Episode 与一次最短控制时序
诊断，行为/实验占比超过 70%；没有新增 evaluator provenance 或 confirmation 基础设施。
