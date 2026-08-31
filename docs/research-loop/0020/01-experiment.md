# R0020 实验

## 单一主假设

不同于 R0019 的独立逐臂 CEM 与固定 transport twist，一个**联合双臂关键帧 motion
planner + payload-relative closed-loop tracker** 可以在正常接口和物理约束下完成：

`approach → acquire → secure → lift → target_transport → place → release → stabilize`。

planner 在复制的 `MjData` 上联合求解两臂和底盘关键帧，维持两只 grasp site 相对篮子把手的
几何关系，并检查插值路径；在线 tracker 根据当前 payload/handle/tool 位姿闭环修正，不执行
长时间固定常量动作。

## 对照与不变量

- 历史对照：R0019 seed 19001 teacher，`0 success`，虽有最长 83-step 双臂接触，但 transport
  时丢失接触。
- R0020 candidate：独立新 controller；不修改 R0019 teacher。
- 保持同一 `carry_living_room_basket/v1`、seed、任务时域、物理、随机化、安全阈值和正式
  16 维动作接口。
- 主要指标：完整 Episode `success=true`。
- 阶段指标：抓取、连续双臂接触、抬升、受控目标搬运、目标支撑、释放及 40-step 稳定。
- 守护指标：actual severe collision `=0`；报告全部 safety intervention 和失败阶段。

## 最小开发实验

- 固定 development seed：`19001`。
- 每个 Episode 最多 `1200` control step。
- candidate 调试预算：最多 3 个完整 physics Episode；实现级单元测试或短 smoke 不计为额外
  candidate，但不得用它们替代完整 Episode。
- 原始产物目录：`runs/research-loop/0020/development/`。
- 单一命令入口：

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_joint_basket_teacher \
  --seed 19001 \
  --output runs/research-loop/0020/development/seed-19001.json
```

## 升级与停止

- 只有 seed 19001 在正式成功状态机下完整成功且 actual severe collision 为 0，才扩到固定
  development cohort：`19001, 19002, 19003, 19004`。
- cohort 只用于检查开发稳定性；不属于 confirmation。
- 若固定实现预算或 3 个完整 Episode 内没有完整成功，停止该 candidate，结论为
  `abandoned`，报告最早失败阶段和已覆盖阶段。
- 不调换 seed、不删除失败、不自动启动另一技术路线、下一轮或 confirmation。

## 判定

- 单 seed 完整成功：`validated_development` 的单点可解性证据，但尚不足以推进 L0。
- 小型 cohort 达到 `>=3/4` 完整成功、全部 severe collision 为 0：R0020
  `validated_development`，允许把 L0 记为通过；仍不允许声明可部署策略能力。
- 否则为 `abandoned`；若因实现偏离、物理/安全绕过或数据污染，则为 `invalid`。
