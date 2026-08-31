# R0019 结果

## 运行总账

### 1. Generic B0–B7 baseline

命令：

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_bimanual_teacher \
  --mode development --controller paired \
  --seed 19001 --seed 19002 --seed 19003 \
  --seed 19004 --seed 19005 --seed 19006 \
  --output runs/research-loop/0019/development/paired-seeds-19001-19006-v2.json
```

baseline 使用 simulator-private payload 几何构造唯一 candidate，因而刻意移除了 P61 的
感知/角色歧义；其余 B0–B7 时域与动作逻辑不变。结果：

- `0/6` success；
- `0/6` Episode 产生双臂同步接触；
- `0` safety intervention；
- `0` actual severe collision；
- 6 个 Episode 均以 `bimanual_task_timeout` 结束。

seed 19001 的阶段终点显示 B2/B3/B4 仍未进入接触：最小左右 handle reach distance
分别为 `0.1138905511m` 和 `0.1167015719m`，最终 payload target distance
`0.8968926532m`。最早失败阶段为 `B3_contact_approach`，与 P57 的 command-support
deficit 一致；B5/B6 之后距离再次增大。

### 2. Privileged teacher

实现：

- 在复制的 `MjData` 上对每只手运行固定预算 CEM，以四个 finger-pad 到对应 handle 的
  signed distance 为目标；
- planner 不写权威 `MjData`；
- 用 backend Jacobian/DLS 的逆映射生成正常 16 维 Cartesian action；
- 状态机为 `approach_base → acquire → secure → transport_probe/failed_hold`；
- 未实现目标引导的完整 transport、`place`、`release` 或 `stabilize`；
- 所有 action 都经过原 `DualArmSafetySupervisor` 与 predictive collision filter。

结果：

| seed | success | 首次双臂接触 | 最长同步接触 | 最早失败阶段 | safety | severe |
|---|---:|---:|---:|---|---:|---:|
| 19001 | 0 | 123 | 83 | `transport_contact_lost` | 0 | 0 |
| 19002 | 0 | 无 | 0 | `acquire_timeout` | 0 | 0 |
| 19003 | 0 | 无 | 0 | `acquire_timeout` | 0 | 0 |
| 19004 | 0 | 无 | 0 | `acquire_timeout` | 0 | 0 |
| 19005 | 0 | 无 | 0 | `acquire_timeout` | 0 | 0 |
| 19006 | 0 | 无 | 0 | `acquire_timeout` | 0 | 0 |

汇总：

- `0/6` success；
- `1/6` Episode 产生真实双臂同步接触；
- seed 19001 的左右 contact step 为 `272/83`，最长同步窗口 `83` step；
- seed 19001 的最大 controlled target progress 仅 `0.0035074962m`；
- 其余 seed 中，19003 只有左手接触，19006 只有右手接触；
- 全部 teacher Episode 为 `0` safety intervention、`0` actual severe collision；
- 最大观察到的 forbidden force 为 `54.0003924N`，低于 `220N` severe 门。

探索阶段还观察过更长接触和局部上抬，但这些一次性探针没有保留命令、完整配置和原始产物，
按本轮收尾规则记为“未归档观察”，不用于正式瓶颈判断或路线决策。

### 3. 确认集决定

100-seed confirmation 状态为 `not_run`，其 `valid` 字段为 `null`。当前 teacher 缺少
`lift/target_transport/place/release/stabilize`，runner 在首个 confirmation Episode 前
拒绝该实现；development teacher 也为 `0/6` success，没有可用的同提交成功资格报告。未
消耗或查看任何 confirmation seed。

runner v2 同时执行以下约束：

- confirmation 必须使用 paired controller 和冻结的 100-seed domain；
- 必须从干净 worktree 启动，并提供同 commit、同 source hash、至少 1 个 teacher success
  的完整 development 资格报告；
- teacher 必须声明覆盖完整任务主要阶段；
- 已存在的 confirmation 输出不能被覆盖；
- wall-time 或基础设施错误会写出部分报告并使 confirmation 无效；
- 单个 controller 失败仍只计为该 Episode 失败，不中断普通 cohort；
- `invalid` 命令退出码为 `2`，不能被自动化误判为通过。

### 4. 原始产物

- 最终 paired development artifact（schema v2）：
  `runs/research-loop/0019/development/paired-seeds-19001-19006-v2.json`
- SHA-256：
  `629629437ff262bb523ca1e61260393e440a586f82c42a7960c0c31063057c8c`
- artifact 大小：`61,276` bytes；
- 运行耗时：`44.5654s`；
- `run_status.completed=true`，12/12 都是有效 Episode 结果，无基础设施失败；
- `decision=invalid`，`l0_gate_passed=false`；
- `confirmation_evidence.status=not_run`、`valid=null`；
- `implementation_evidence.valid=false`，缺少
  `lift/target_transport/place/release/stabilize`；
- qualification hash 覆盖 19 个直接依赖文件，包括 teacher/runner、task/backend/safety、
  动作封装、任务配置、场景 XML 及其共享机器人 XML；
- artifact 记录运行时源码提交 `5102d1411c23e3465dc11bf8891bfbc8505a43a7` 和
  `source_worktree_dirty=true`，因此只属于 development，不能作为未来 confirmation 资格报告。
- runner 同时写出 `.sha256` sidecar；`runs/` 受 `.gitignore` 管理，不纳入 Git。
- 旧 v1 artifact 的 `decision=validated_development` 和 confirmation `valid=true` 语义错误，
  已被 v2 取代，不得用于结论。

### 5. 验证

```text
48 passed in 8.09s
Python size check passed: 461 files, file <= 800 lines, function <= 200 lines
Architecture check passed: engine, foundation, and core boundaries are intact
```
