# R0019 总结

状态：结束。

- 当前能力等级：`L0 未通过`。
- 结论：`inconclusive_capability`。

## 结论

本轮没有达到冻结 L0 门。最终 development paired cohort 中：

- generic B0–B7 baseline：`0/6` success、`0/6` 双臂接触；
- privileged teacher：`0/6` success、`1/6` 双臂接触；
- 两组均为 `0` actual severe collision、`0` safety intervention。

seed 19001 证明正常 MuJoCo 物理、正式 16 维动作接口与独立安全层下可以形成真实双臂接触，
最长连续 `83` step。后续更直接的开发探针把同步接触延长到 `160` step，并将篮子最高抬至
约 `0.582m`，但在 transport 过程中一侧抓取滑脱；没有 Episode 同时完成抓取、搬运、
目标放置和 40-step 稳定门。其余 5 个独立 development seed 甚至未稳定进入双臂接触。

最早且稳定的阻塞是：

1. **跨 seed 抓取规划不稳**：独立逐臂 CEM 在 5/6 seed 未产生同步双垫接触；
2. **transport support 不足**：唯一成功抓取 seed 在移动载荷时丢失一侧接触，最大受控目标
   进展仅 `0.0035074962m`；
3. 安全层不是当前阻塞：无 action rejection、无 actual severe collision。

因此没有启动预注册的 100-seed confirmation。development `0/6` success 已知不可能满足
`>=80/100`，继续运行只会消耗资源并暴露独立 seed。
runner 已加固为只从干净、已提交的 worktree 启动 confirmation，并严格核对冻结 seed 域与
完整 paired Episode；这只保护未来确认结论，不改变本轮失败结果。

## 允许声明

- 现有 generic B0–B7 在该正式任务上没有进入双臂接触。
- privileged CEM + inverse-DLS 控制能在至少一个 development seed 上形成持续真实双臂接触。
- 当前模型/控制接口下的下一个工程瓶颈是跨 seed 的联合双臂抓取与载荷 transport support。

## 不允许声明

- 不允许声明 `carry_living_room_basket/v1` 已有可行 L0 teacher ceiling。
- 不允许声明任何可部署状态策略、视觉策略、泛化或硬件能力。
- 不允许把局部双臂接触、抬升高度、测试通过或 loss/距离改善当作能力等级推进。

## 保留与回退

- 保留 R0019 runner、teacher 原型与测试，作为下一次用户决策时的最短复现入口。
- 不修改 `docs/research-loop/0001/`～`0018/`。
- 不运行 world model、Actor 或大规模训练。
- 不自动启动 L1 或下一轮；等待用户决定是缩小任务、加强 grasp fixture/末端执行器、引入
  demonstration/motion-planning 库，还是停止该任务路线。

## 资源分配

本轮主要工作是实际 MuJoCo action/contact/Episode 与 controller 实现调试，超过 70%；
静态审计与文档低于 20%，评测 runner/测试低于 10%。未创建新的递归 oracle/lineage 门。
