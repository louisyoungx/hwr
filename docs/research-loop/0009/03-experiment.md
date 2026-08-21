# R0009 冻结实验

## 冻结状态

- 冻结候选：`R0001-P51`、`R0001-P52`。
- 不执行：P50、P53a、P53b、P54、P49-E1、P47-E1、P55。
- 本轮无训练候选，不启动正式训练，不使用 MPS，不休眠。
- 冻结文档必须先形成 Git 提交；实现提交必须以该提交为祖先。

## 共同边界

1. P51 只改变 P41 primitive 的 acquisition-frame linear error 到 base-frame command 的
   坐标转换。
2. P52 只测量当前 policy FK 与 MuJoCo grasp-center site 的位置一致性，不改变动作。
3. 禁止修改 candidate generator、candidate bytes、selector、acquisition、phase、target、
   velocity cap、gripper、backend IK、task、reward、termination、success、安全或 P40。
4. 禁止 task/object/target ID、instruction、颜色、simulator object pose、contact、reward、
   stage、success 或 evaluator output 进入 policy 动作。
5. 所有 MuJoCo site、model qpos 和 evaluator transform 仅用于 P52 标签与 P51 测量。
6. 所有产物必须记录 source commit、命令、配置/模型 identity、直接 artifact hash/bytes
   与禁止声明位。

## `R0001-P51`：Cartesian primitive acquisition→base 坐标合同

### 研究问题

> 在 candidate target、policy FK、phase、速度和安全全部固定时，只把 acquisition-frame
> 线速度正确转换到当前 base frame，能否恢复 P41 arm command 的坐标语义？

### 唯一变量与精确公式

设：

- `e_A = target_A - tool_A`；
- `theta_A` 为 acquisition base yaw；
- `theta_B` 为当前 base yaw；
- `R_B_from_A = Rz(theta_A - theta_B)`。

candidate：

```text
v_B = R_B_from_A * clip_norm(2 * e_A, velocity_max)
```

baseline：

```text
v_B_legacy = clip_norm(2 * e_A, velocity_max)
```

冻结语义：

- 只旋转 `x/y`；`z` 保持原值；
- 角速度三维继续为 0；
- 旋转发生在 norm clipping 后，因旋转保范数；
- 最终 canonical action bounds 与 clipping 不变；
- relative yaw 为 0 时 candidate 与 legacy action float64 bytes 完全一致；
- hold、安全 fail-closed 与 gripper 行为不变。

### 实现负责人和文件所有权

- 唯一负责人：P51 实施 Agent。
- 允许修改：
  - `src/hwr/eval/target_selection.py`
  - `tests/test_target_selection.py`
- 允许新增：
  - `src/hwr/apps/evaluate_cartesian_frame_contract.py`
  - `tests/test_cartesian_frame_contract_app.py`
- 不得修改 P52 文件、backend、配置或历史文档。
- 如需越过上述范围，停止并交主 Agent重新冻结。

### 冻结合同矩阵

- acquisition yaw：
  `0, pi/3, -pi/2`
- relative yaw：
  `0, pi/6, -pi/6, pi/2, -pi/2, pi`
- acquisition-frame vector：
  - `(+1,0,0)`
  - `(0,+1,0)`
  - `(-1,+1,0)`
  - `(+0.3,-0.4,+0.5)`
- 左右臂都必须覆盖。

每个 cell 计算：

- legacy/candidate base-frame command；
- 转回 acquisition frame 的 realized vector；
- angular error；
- norm error；
- z error；
- float64 byte identity。

该矩阵是确定性合同覆盖，不把 cell 当统计独立样本。

### 接受标准

`accepted as Cartesian primitive correctness evidence` 仅当：

1. 所有 relative-yaw=0 cell candidate/legacy bytes 完全一致；
2. 所有 candidate cell 的 realized acquisition-frame vector 与 expected vector：
   - 最大绝对误差 `<=1e-12`；
   - angular error `<=1e-12 rad`；
   - norm error `<=1e-12`；
   - z error `<=1e-12`；
3. 预设 legacy 反例：
   - relative yaw `±pi/2`、非零水平 vector 的 angular error `>=pi/2-1e-12`；
   - 证明测试能够拒绝旧语义；
4. primitive 的 target、phase、velocity cap、gripper、hold 和 action bounds 回归测试通过；
5. 修改仅限冻结文件和公式；
6. source tree 干净且冻结文档提交为祖先。

任一公式、identity、误差或回归门失败：`rejected`。实现/lineage/禁止字段污染：
`invalid`。

P51 的接受只证明坐标合同正确，不证明 tool FK、candidate 质量、物理接触、任务成功、
泛化或硬件安全改善。

### 稳定命令与产物

```text
.venv/bin/python -m pytest -q \
  tests/test_target_selection.py \
  tests/test_cartesian_frame_contract_app.py

.venv/bin/python -m hwr.apps.evaluate_cartesian_frame_contract \
  --output runs/research-loop/0009/r0009-p51-cartesian-frame-s20265101
```

产物：

- `report.json`
- `manifest.json`
- 失败时 `failure.json`

## `R0001-P52`：policy FK—plant tool-site 一致性门

### 研究问题

> 在 latency-free 的同一物理 state、同一 current-base frame 下，P41 手写 policy FK 与
> MuJoCo 实际控制的 grasp-center site 是否在接触 primitive 所需尺度上充分一致？

### 唯一变量与时间对齐

- P52 是 report-only 测量，没有 treatment 行为。
- 每个 deterministic state：
  1. 将同一 joint vector 写入 MuJoCo qpos；
  2. 调用 `mujoco.mj_forward`；
  3. 将同一 joint vector输入 policy `_tool_position`；
  4. 读取同一时刻 MuJoCo grasp-center site；
  5. 将 site 从 world frame 转到同一 current-base frame；
  6. 计算欧氏位置误差。
- 不调用 observation latency queue，不把 stale observation joints 与 current site 比较。
- base yaw/position 只用于验证 frame invariance，不计入 FK 误差来源。

### 实现负责人和文件所有权

- 唯一负责人：P52 实施 Agent。
- 只允许新增：
  - `src/hwr/eval/tool_kinematics.py`
  - `src/hwr/apps/evaluate_tool_kinematics.py`
  - `tests/test_tool_kinematics.py`
  - `tests/test_tool_kinematics_app.py`
- 不得修改 P51 文件、backend、模型、配置或历史文档。
- app 可读取现有 P41 `_tool_position` 与 MuJoCo backend 私有 site，后者严禁进入 policy。

### 冻结 state grid

- 三个正式 task/binding 全部运行，验证机器人模型与 site identity 一致。
- 每臂 6 个 joint。
- state 集合：
  - model `qpos0`；
  - 每个 joint 分别位于有效 range 的 `20%` 与 `80%`，其他 joint 保持 qpos0；
  - 固定 seed `20265201` 的 128 个 scrambled-Halton central-range vector；
  - 每一维只取 joint range 的 `[10%,90%]`，避免极限奇异与非法边界。
- 左右臂同时报告；三任务不得合并掉最弱 task/arm。
- state grid 是确定性数值覆盖，不计算 p-value，不把 joint state 当随机独立样本。

### 冻结指标与判定

每个 task × arm 报告：

- count；
- mean、median、p95、max Euclidean position error；
- x/y/z absolute-error p95/max；
- finite 与 deterministic replay；
- robot model、joint names/ranges、tool-site name、task binding identity。

aggregate 取全部 task/arm state，同时发布最弱 task/arm。

测量合同有效门：

1. 三任务的 robot joint/site mapping 完整且一致；
2. 每个 planned state/arm 都有唯一 finite terminal；
3. 相同输入两次运行 report payload bit-identical；
4. base pose 变换 fixture 在平移和 yaw 下保持 base-frame site 坐标不变，误差 `<=1e-12`；
5. evaluator site、qpos、object truth 不进入 policy action或 candidate；
6. source tree 干净且冻结文档提交为祖先。

假设判定：

- aggregate p95 `<=0.01m` 且 aggregate max `<=0.02m`：
  `accepted as FK agreement contract evidence`；
- aggregate p95 `>0.03m`：
  `accepted as material FK mismatch evidence`；
- 其他有限结果：
  `inconclusive`；
- 任一有效门失败：
  `invalid`。

阈值依据：P41 最小合格 object width 为 `0.035m`、接触阶段目标偏置为 `0.015m`，因此
`<=0.01m` 视为小于接触几何尺度的 agreement，`>0.03m` 视为足以与最小物体尺度同量级的
material mismatch；灰区不作强结论。

无论结果如何，P52 不自动替换 FK，不证明零接触全部由 FK 导致。

### 稳定命令与产物

```text
.venv/bin/python -m pytest -q \
  tests/test_tool_kinematics.py \
  tests/test_tool_kinematics_app.py

.venv/bin/python -m hwr.apps.evaluate_tool_kinematics \
  --output runs/research-loop/0009/r0009-p52-tool-kinematics-s20265201
```

产物：

- `report.json`
- `manifest.json`
- 失败时 `failure.json`

## 顺序与物理 smoke

1. P51/P52 实现可由不同 Agent 并行，但必须是独立原子提交。
2. 主 Agent 合并后先运行 focused tests、Python size、architecture、physics integrity、
   compileall 和 `git diff --check`。
3. 先执行 P52 正式测量。
4. P52 为 FK agreement 时：
   - 允许对 P51 做新的固定候选 MuJoCo smoke；
   - smoke 只能报告 tool-target distance、directional derivative、arm/entity contact 和
     现有安全字段；
   - 不参与 P51 解析接受判定，不得称为 task success。
5. P52 为 material mismatch 或 inconclusive 时：
   - 不运行或不解释 P51 接触 smoke；
   - P51 只按解析合同判定；
   - 下一轮必须把 FK 行为修复作为独立提案，不能在本轮追加。
6. 本轮不启动 P41 正式 selector 对照、世界模型训练、Actor、Replay 采集或能力 benchmark。

## 全局门禁

```text
.venv/bin/python scripts/check_python_size.py
.venv/bin/python scripts/check_architecture.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m compileall -q src tests scripts
git diff --check
```

正式结果前还必须确认：

- `git status --short` 干净；
- 冻结文档提交是实现提交祖先；
- `docs/research-loop/0001/`～`0008/` tree 与 R0009 起始记录一致；
- 所有 artifact identity 与 manifest 一致；
- `capability_claim_allowed=false`
- `task_success_claim_allowed=false`
- `generalization_claim_allowed=false`
- `hardware_safety_claim_allowed=false`
- `training_executed=false`
