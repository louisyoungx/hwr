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

以下“3 个完整 Episode”预算属于初次执行合同，已被文末
“2026-08-31 重开修订”取代；保留它只为解释 attempt 1～3 的历史来源。

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

## 2026-08-31 重开修订

用户依据提交 `f7b27a3` 后的新规则明确授权重新打开 R0020。主假设保持不变，不创建 R0021。
此前 attempt 1～3 分别发生在代码修改之后，属于同一 candidate 的三次 implementation
iteration；它们没有进入本轮相对 R0019 的差异化机制，不是独立重复证据，也不构成联合规划
路线的可判别失败。

### 当前实现范围

- 当前只处理最早未通过的 `acquire`：把静态联合接触构型转化为动态最后接近、闭爪和持续
  双臂接触。
- 不继续扩写尚不可达的 `lift/target_transport/place/release/stabilize`；已有后续代码不作为
  当前实现就绪证据。
- 最后接近与闭爪必须使用在线 pad/handle 几何和真实 contact feedback；不再只依赖
  `<0.018rad` joint error，不进行无边界常量动作或增益扫参。

### 差异化机制与 behavior entry

差异化机制仍为“联合双臂 planner + payload-relative closed-loop tracker”。其候选判别前的
behavior entry condition 冻结为：

- seed `19001`；
- 正式 16 维动作接口；
- 正常 MuJoCo 物理；
- 原 `DualArmSafetySupervisor` 与两步 predictive collision filter；
- 形成至少 `10` 个连续 control step 的真实双臂接触；
- controller 实际进入 `secure`。

类字段、静态 joint 解、pad 距离阈值、未执行状态机或单侧/非连续接触均不算达到 entry。

### Implementation/debug 预算

- 自本次重开开始，主动实现调试 wall time 最多 `2` 小时，或最多 `24` 个有原始记录的短程
  physics smoke，以先到者为准。
- 每个 smoke 使用 seed `19001`，只运行到 behavior entry、明确 acquire 失败或冻结的短程
  step 上限；保存命令、源码版本、配置、stage/contact/pad 几何、安全结果和原始 artifact。
- 影响结论的关键版本至少保留 commit、patch 或最小源码 hash。
- 达到 behavior entry 后立即冻结当前实现，debug 预算结束，候选判别预算才开始。
- 若 debug 预算耗尽仍未达到 entry，停止当前实现；结论只适用于该实现，不否定联合规划
  路线，不累计“三轮无进展”，不自动启动新轮或 confirmation。

### 冻结候选判别预算

1. behavior entry 达到后，先运行一个完整 seed `19001` Episode。
2. 只有该 Episode 端到端成功且 actual severe collision 为 `0`，才运行原冻结 development
   cohort：`19001, 19002, 19003, 19004`。
3. cohort 达到 `>=3/4` 完整成功且全部 actual severe collision 为 `0`，才以
   `validated_development` 推进 L0。
4. 本轮不运行 confirmation 或 sealed final。

## 2026-08-31 第二次重开修订

用户依据提交 `fc8938c` 后的新规则再次明确授权重新打开 R0020。主假设仍为“联合双臂
planner + payload-relative closed-loop tracker”，不创建 R0021。

smoke 011 已证明 acquire 子目标可以形成 `11` step 连续双臂接触并进入 `secure`，但第一次
重开冻结的 behavior entry 定义过早：controller 在 `secure` 立即从在线 pad/handle feedback
切回静态 joint target 与 `GRASP_GRIPPER`，完整 Episode 随即失去接触；没有任何
payload-relative lift action 通过正式接口执行。因此 smoke 011、旧
`behavior-entry-freeze.json` 和 `reopened-candidate-seed-19001.json` 均重新归类为
pre-entry implementation evidence，不构成路线失败或可计数无进展轮次。

### 当前实现范围

- 只修复 `acquire → secure` 的控制连续性。
- `secure` 必须延续已成功的在线 pad/handle 几何、contact feedback、目标姿态和闭爪预载，
  不再切回静态 joint target。
- 不扩写 `transport/place/release/stabilize` 或其他后续功能；只允许执行现有
  payload-relative lift tracker 的首个动作来验证阶段交接。
- 不进行无边界常量动作或增益扫参。

### 修正后的 behavior entry

以下条件必须在 seed `19001` 的同一短程 physics smoke 中全部满足：

- 正式 16 维动作接口、正常 MuJoCo physics、原 `DualArmSafetySupervisor` 与两步
  predictive collision filter；
- 保持双臂接触完成 `secure` 所需的连续接触；
- controller 实际进入 `lift`；
- 至少一个由 payload-relative lift tracker 生成的动作已通过正式接口执行；
- 已记录该动作之后的 observation；
- 后继 observation 中双臂接触仍保持，actual severe collision 为 `0`。

仅进入 `secure`、仅生成但未执行 lift action，或首个后继 observation 已丢失接触，都不算
entry。

### 剩余 debug 预算

- 沿用第一次重开的原预算，不重置：最多再运行 `13` 个有原始记录的短程 physics smoke，
  即 smoke 012～024；主动实现调试 wall time 余额约 `100` 分钟，以先到者为准。
- smoke 运行到修正后的 entry、明确 acquire/secure 失败或冻结短程 step 上限；不得在 stage
  刚切换时停止。
- 继续保存命令、源码版本或最小源码 hash、配置、动作后继 observation、contact 与安全结果。
- 若预算耗尽仍未达到 entry，只停止当前实现，不否定主假设、不累计无进展轮次。

### 修正后的候选判别预算

达到 entry 后立即冻结源码和配置，再运行且只运行一个完整 seed `19001` Episode：

1. 端到端成功且 actual severe collision 为 `0`，才运行原冻结的 `19001`～`19004`
   development cohort；
2. 差异化 payload-relative 机制已真实执行但完整 Episode 失败时，只淘汰当前精确定义的
   冻结候选，不外推否定 motion-planning/trajectory-optimization 家族；
3. 不运行 confirmation 或 sealed final，不自动启动其他路线或新轮。
