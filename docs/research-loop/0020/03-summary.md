# R0020 总结

状态：重开工作已结束。

- 当前能力等级：`L0 未通过`。
- 当前结论：`abandoned`，只适用于 behavior entry 后冻结的当前实现。
- confirmation：`not_run`。
- sealed final：`not_run`。

## 重开说明

提交 `f7b27a3` 明确：R0020 attempt 1～3 是三次代码变化后的 implementation iteration，
未进入候选差异化机制，不能作为独立重复证据或可判别路线失败。用户已明确授权在原目录
重开 R0020；不创建 R0021。

当前只修复 `acquire` 的实现就绪缺口。behavior entry 为 seed 19001 在正式 16 维动作、
正常 MuJoCo 物理和原安全层下形成至少 10 个连续 control step 的真实双臂接触并进入
`secure`。达到 entry 前不扩写或评价 lift、transport、place、release、stabilize。

重开实现已在 smoke 011 达到 behavior entry：进入 `secure`，连续双臂接触 `11` step，
`0` safety intervention、`0` actual severe collision。随后冻结 controller 源码 hash 并启动
候选判别预算。

## 历史结论边界

R0020 实现了不同于 R0019 的联合双臂路线：

- 在复制的 MuJoCo state 上联合优化底盘接近位置和两臂 12 个关节；
- 同时优化四个 finger-pad 到两侧 handle 的 signed distance，并检查联合插值路径；
- 在线使用关键帧 joint tracking 和抓取后的 payload-relative Cartesian feedback；
- controller 明确实现
  `approach/acquire/secure/lift/target_transport/place/release/stabilize`，
  不再使用固定 transport twist。

固定 development seed 19001 的 3 个历史 implementation iteration 均为 `0 success`。三次都只到达
`approach/acquire/failed_hold`，没有形成任一完整双 pad contact，最早失败阶段为
`acquire_timeout`。全部 Episode 为 `0` safety intervention、`0` actual severe collision。

静态 planner 能找到四 pad signed distance 接近或进入接触的联合末态，focused test 也确认
planner 不修改权威状态；但 joint-space waypoint tracker 没有把该静态解转化为真实动态抓取。
这些结果只定位旧实现的 behavior readiness 缺口，没有实际执行 payload-relative 机制，
因此不能否定联合规划主假设、不能评价后续阶段，也不能建立 L0 oracle ceiling。

## 重开结果

重开阶段修复了旧实现的两个根因：

1. 旧静态 planner 允许 palm/wall 等非 pad–handle 篮子接触，产生不可执行的穿透末态；新
   planner 只允许四个 pad–对应 handle 接触，并把联合路径穿透纳入搜索。
2. 最后接近不再用 joint-error 门触发一次性闭爪，而是在线追踪 pad/handle 相对几何，根据
   signed distance、单/双 pad 接触和最大 3mm 横向平衡修正渐进闭爪。

冻结实现的完整 seed 19001 判别 Episode 实际到达
`approach/acquire/secure/failed_hold`，最大连续双臂接触 `11` step，证明差异化 acquire
机制已经进入正常 physics。但接触在 `secure` 中丢失，最终 `secure_timeout`、`0 success`，
没有抬升或受控目标进展；全部安全守护无回归。

因为完整 seed 19001 未端到端成功，按冻结条件停止，不运行 19001～19004 development cohort。
该结果是当前冻结实现的可判别失败，但只覆盖其已执行的 acquire/secure 行为；不否定联合
规划 + payload-relative closed-loop tracking 主假设。

## 允许声明

- 当前联合静态 planner 能在复制 state 中找到满足四 pad 距离目标的候选构型。
- 历史 joint-space waypoint tracker 在固定 development seed 19001 的三次 implementation
  iteration 中均未形成完整双 pad 接触。
- 三次 Episode 没有 safety intervention 或 actual severe collision。
- 重开实现经正式动作、物理和安全层达到 behavior entry，形成 11-step 真实双臂接触并进入
  `secure`。
- 当前冻结实现的完整 seed 19001 Episode 在 `secure` 丢失接触并失败，未达到 cohort 门。

## 不允许声明

- 不允许声明 `carry_living_room_basket/v1` 已有可行 L0 teacher ceiling。
- 不允许用历史 attempt 1～3 否定联合 motion planning、轨迹优化或 payload-relative
  tracking 路线；它们只表明当时实现尚未 behavior-ready。
- 不允许用重开完整 Episode 否定联合规划路线；它只淘汰当前冻结的 acquire/secure 实现。
- 不允许把 attempt 1 的非受控高度变化称为抓取或抬升成功。
- 不允许归因尚未真正执行的 lift、target transport、place、release 或 stabilize。
- 不允许声明任何可部署状态策略、视觉策略、泛化或硬件能力。

## 保留与回退

- 保留 development-only planner、controller、runner、focused tests 与三个原始 artifact，
  用于复现静态解到动态接触之间的缺口。
- 保留 11 个重开 smoke、behavior-entry freeze manifest 和完整 seed 19001 判别 artifact；
  `runs/` 不写入 Git，但文档记录路径与 hash。
- 不修改 `docs/research-loop/0019/` 或 `0001/`～`0018/`。
- 不保留任何 confirmation 门禁；本轮 confirmation 与 sealed final 均为 `not_run`。
- 不自动启动下一路线或 R0021。

## 计数修订

撤销“R0018～R0020 连续三轮无能力进展”的判断：

- R0018 属于旧机制归档，不计数；
- R0019 为 `invalid`，不计数；
- R0020 历史 attempt 1～3 不计数；重开后的冻结实现达到 behavior entry 后完成一次可判别
  Episode，因此 R0020 现在只计为一个未推进能力的可计数轮次。

当前没有“三轮无进展”强制停止。R0020 已按预算结束；不自动创建新轮或运行 confirmation。

## 资源分配

重开阶段共运行 11 个有原始记录的短 smoke，未达到 24-smoke 或 2-hour debug 上限；达到
entry 后立即冻结并只运行一个完整判别 Episode。主要资源仍用于真实 physics behavior。
