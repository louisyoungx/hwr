# R0019 总结

状态：结束。

- 当前能力等级：`L0 未通过`。
- 结论：`invalid`。

## 结论

本轮 candidate 未实现预先声明的完整 teacher 状态机：只有
`approach/acquire/secure/transport_probe`，缺少
`lift/target_transport/place/release/stabilize`。因此端到端实验属于实现偏离，不能用于
判定环境、机器人、控制时域或完整 teacher 路线是否可解。

最终 development paired cohort 中：

- generic B0–B7 baseline：`0/6` success、`0/6` 双臂接触；
- privileged teacher：`0/6` success、`1/6` 双臂接触；
- 两组均为 `0` actual severe collision、`0` safety intervention。

seed 19001 证明该原型经正常 MuJoCo 物理、正式 16 维动作接口与独立安全层可以形成真实双臂
接触，最长连续 `83` step，随后在原型 transport 动作中失去接触。其余 5 个 development
seed 没有形成双臂同步接触。没有 Episode 完成搬运、目标放置、释放和 40-step 稳定门。

可归档的子目标观察是：

1. 当前 acquire 原型跨 seed 不稳：5/6 seed 未产生同步双臂接触；
2. 当前 transport 原型在唯一同步接触 seed 上丢失一侧接触，最大受控目标
   进展仅 `0.0035074962m`；
3. 这 12 个 Episode 中没有 safety intervention 或 actual severe collision。

这些观察只能覆盖已实现的子目标，不能宣称它们是完整任务的最早或唯一瓶颈。100-seed
confirmation 状态为 `not_run`，没有查看冻结 seed。runner v2 已拒绝当前不完整 teacher，
并要求未来 confirmation 提供同提交、干净、至少一个 teacher 端到端成功的 development
资格报告。

## 允许声明

- 现有 generic B0–B7 在该正式任务上没有进入双臂接触。
- 当前 privileged CEM + inverse-DLS 原型能在一个 development seed 上形成持续真实双臂接触。
- 当前原型的 acquire 和 transport 子阶段存在上述可复核失败。

## 不允许声明

- 不允许声明 `carry_living_room_basket/v1` 已有可行 L0 teacher ceiling。
- 不允许用本轮端到端 `0/6` 判定环境、机器人、控制时域或完整 teacher 技术路线不可解。
- 不允许把当前子阶段失败称为完整任务的最早或唯一瓶颈。
- 不允许声明任何可部署状态策略、视觉策略、泛化或硬件能力。
- 不允许把局部双臂接触、抬升高度、测试通过或 loss/距离改善当作能力等级推进。

## 保留与回退

- 保留 R0019 runner v2、teacher 子目标原型与测试，作为失败复现和防止错误 confirmation
  的入口；旧 v1 判定不得复用。
- 不修改 `docs/research-loop/0001/`～`0018/`。
- 不运行 world model、Actor 或大规模训练。
- 不自动启动 L1 或下一轮；等待用户决定是缩小任务、加强 grasp fixture/末端执行器、引入
  demonstration/motion-planning 库，还是停止该任务路线。

## 资源分配

本轮主要工作是实际 MuJoCo action/contact/Episode 与 controller 实现调试，超过 70%；
静态审计与文档低于 20%，评测 runner/测试低于 10%。未创建新的递归 oracle/lineage 门。
