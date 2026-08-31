# R0020 总结

状态：第二次重开工作已结束。

- 当前能力等级：`L0 未通过`。
- 当前结论：`abandoned`，只淘汰第二次重开冻结的精确定义候选。
- confirmation：`not_run`。
- sealed final：`not_run`。

## 重开说明

提交 `f7b27a3` 明确：R0020 attempt 1～3 是三次代码变化后的 implementation iteration，
未进入候选差异化机制，不能作为独立重复证据或可判别路线失败。用户已明确授权在原目录
重开 R0020；不创建 R0021。

当前只修复 `acquire` 的实现就绪缺口。behavior entry 为 seed 19001 在正式 16 维动作、
正常 MuJoCo 物理和原安全层下形成至少 10 个连续 control step 的真实双臂接触并进入
`secure`。达到 entry 前不扩写或评价 lift、transport、place、release、stabilize。

smoke 011 已验证 acquire 子目标：进入 `secure`，连续双臂接触 `11` step，
`0` safety intervention、`0` actual severe collision。但提交 `fc8938c` 明确该 entry 定义
过早：至少一个 payload-relative lift action 必须经正式接口执行，且其后继 observation 仍保持
双臂接触。本轮此前没有满足该条件。

第二次重开在 smoke 012 达到修正后的 behavior entry：`secure` 延续在线几何/contact
feedback，保持连续双臂接触至 `26` step；首个 payload-relative lift action 已经正式执行，
其后继 observation 仍保持左右双臂接触，且 `0` safety intervention、`0` severe collision。

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

此前所谓冻结实现的完整 seed 19001 Episode 到达
`approach/acquire/secure/failed_hold`，最大连续双臂接触 `11` step。但 `secure` 立即切回
静态 joint target 与较低闭爪预载，接触在首个 payload-relative lift action 前丢失，最终
`secure_timeout`、`0 success`、`0` 抬升、`0` 受控目标进展。

因此该完整 Episode 重新归类为 pre-entry implementation result：它验证 acquire 子目标并
淘汰当时的 acquire→secure 交接实现，但没有执行主假设的差异化 payload-relative 机制，不能
淘汰当前冻结候选、不能否定联合规划路线，也不计为无进展轮次。

## 第二次重开结果

第二次重开只修复 acquire→secure 控制连续性：

- secure 继续使用 acquire 的在线 pad/handle 几何、signed-distance、contact feedback 和
  `1.0` 闭爪预载；
- 完成 secure 连续接触后，执行现有 payload-relative lift tracker 的动作；
- 没有修改 transport、place、release 或 stabilize。

smoke 012 在 seed 19001 上达到修正后的 entry，随后冻结三个 controller 源码文件。冻结 v2
完整 Episode 实际执行 `approach/acquire/secure/lift/failed_hold`：

- 最大连续双臂接触 `26` step；
- payload-relative lift action 实际执行 `9` 个 control step；
- 接触随后丢失，`lift_contact_lost`；
- `maximum_lift_m=0`、`maximum_controlled_target_progress=0`；
- `0` safety intervention、`0` actual severe collision。

因为完整 seed 19001 未成功，按冻结条件不运行 development cohort。本结果可淘汰当前精确定义
的“collision-aware joint planner + 在线 acquire/secure feedback + 当前 payload-relative
lift tracker”冻结候选；不得外推否定整个 motion-planning/trajectory-optimization 家族。

## 允许声明

- 当前联合静态 planner 能在复制 state 中找到满足四 pad 距离目标的候选构型。
- 历史 joint-space waypoint tracker 在固定 development seed 19001 的三次 implementation
  iteration 中均未形成完整双 pad 接触。
- 三次 Episode 没有 safety intervention 或 actual severe collision。
- 重开实现经正式动作、物理和安全层达到 behavior entry，形成 11-step 真实双臂接触并进入
  `secure`；按新规则这只算 acquire 子目标达成。
- 旧完整 seed 19001 Episode 证明 acquire→secure 交接会立即丢失接触，未执行 lift action。
- 第二次重开 smoke 012 满足修正后的 entry：执行 payload-relative lift action 后的 observation
  仍保持双臂接触。
- 冻结 v2 候选在 lift 中丢失接触，未产生可测抬升或目标进展。

## 不允许声明

- 不允许声明 `carry_living_room_basket/v1` 已有可行 L0 teacher ceiling。
- 不允许用历史 attempt 1～3 否定联合 motion planning、轨迹优化或 payload-relative
  tracking 路线；它们只表明当时实现尚未 behavior-ready。
- 不允许把 smoke 011 或旧完整 Episode 称为修正后 behavior entry 已达到。
- 不允许用旧完整 Episode 淘汰冻结候选或否定联合规划路线；它只淘汰当时的
  acquire→secure 交接实现。
- 不允许把 v2 候选失败外推为 motion planning、trajectory optimization 或
  payload-relative tracking 家族失败。
- 不允许把 attempt 1 的非受控高度变化称为抓取或抬升成功。
- 不允许归因尚未真正执行的 lift、target transport、place、release 或 stabilize。
- 不允许声明任何可部署状态策略、视觉策略、泛化或硬件能力。

## 保留与回退

- 保留 development-only planner、controller、runner、focused tests 与三个原始 artifact，
  用于复现静态解到动态接触之间的缺口。
- 保留 11 个重开 smoke、历史 behavior-entry freeze manifest 和完整 seed 19001 artifact；
  `runs/` 不写入 Git，但文档记录路径与 hash。
- 保留 smoke 012、`behavior-entry-freeze-v2.json` 和
  `reopened-v2-candidate-seed-19001.json` 及其 hash。
- 不修改 `docs/research-loop/0019/` 或 `0001/`～`0018/`。
- 不保留任何 confirmation 门禁；本轮 confirmation 与 sealed final 均为 `not_run`。
- 不自动启动下一路线或 R0021。

## 计数修订

撤销“R0018～R0020 连续三轮无能力进展”的判断：

- R0018 属于旧机制归档，不计数；
- R0019 为 `invalid`，不计数；
- R0020 历史 attempt 1～3 不计数；smoke 011 与旧完整 Episode 没有执行 payload-relative
  lift action，也不计数。
- 第二次重开 v2 候选执行了差异化 payload-relative 机制，但结论明确不允许否定所研究的路线
  家族；按 `fc8938c` 规则，本轮仍不累计为路线级无进展轮次。

当前没有“三轮无进展”强制停止。R0020 第二次重开已按预算结束；不自动创建新轮或运行
confirmation。

## 资源分配

第一次重开共运行 11 个有原始记录的短 smoke；第二次重开只运行 smoke 012 后达到 entry，
未耗尽剩余 13-smoke / 约 100 分钟 debug 预算。随后只运行一个冻结完整 Episode。
