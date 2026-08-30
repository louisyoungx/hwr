# R0019 结果

## 运行总账

### 1. Generic B0–B7 baseline

命令：

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_bimanual_teacher \
  --mode development --controller paired \
  --seed 19001 --seed 19002 --seed 19003 \
  --seed 19004 --seed 19005 --seed 19006 \
  --output runs/research-loop/0019/development/paired-seeds-19001-19006.json
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
- 状态机为 `approach_base → acquire → secure → transport/failed_hold`；
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

开发阶段的补充物理探针还证明：

- CEM 可在复制状态找到四个 pad 同时接触的构型；
- inverse-DLS 执行可在 seed 19001 上连续保持双臂接触超过 100 step；
- 小幅水平/垂直搬运会因夹持构型不稳而丢失一侧接触；
- 最大上抬实验曾将 payload 从约 `0.435m` 抬到约 `0.582m`，但在到达目标支撑前失去
  双臂接触，不能计为任务成功。

开发探针总账：

| 探针 | 结果 |
|---|---|
| 直接 grasp-site Cartesian servo | 最小 reach 约 `0.055–0.067m`，0 双垫接触 |
| 外侧偏置与基座前移 | 右手短暂双垫接触；前移在 `x≈0.077m` 被 predictive safety 拒绝 |
| 6D 姿态约束 | 可求位置姿态 IK，但在线路径碰撞/漂移，0 双臂同步接触 |
| pad-midpoint feedback | 仍受接触扰动和局部可达性限制，0 双臂同步接触 |
| 180 组合局部参数网格 | 固定快照上 0 个组合形成同步接触，未扩大搜索空间 |
| 独立逐臂 CEM | 复制状态找到四 pad 接触构型；在线 inverse-DLS 首次双臂接触 step 187 |
| CEM pose waypoint | 进入高力接触并触发 predictive safety hold，0 actual severe collision |
| inverse-DLS grasp | 首次双臂接触 step 123，最长同步 50/83/141 step（不同 transport 探针） |
| 固定向上 twist | payload 最高约 `0.582m`，随后右侧接触丢失 |
| 5/10/15mm vertical preload | 5/10mm 可长期保持接触但不能脱离支撑；15mm 后接触丢失 |
| 刚体/IK lift waypoint | 规划残差 `<3e-5m`，在线执行在早期 waypoint 丢失右侧接触 |

两次一次性脚本错误也保留在会话记录中：一次缺少 `mujoco` import，一次未初始化局部
`first` 字典；二者均在产生科学结果前退出，修正后按原参数重跑。

### 3. 确认集决定

未运行 100-seed confirmation。原因不是计算资源不足：6 个 paired Episode 用时约
`44.65s`，按线性估计 100 paired Episode 在 `1,800s` wall-time 门内可运行。停止原因是
teacher 在 development set 为 `0/6` success，已知无法达到 `80/100` 门；继续运行确认集
只会浪费预算并污染独立 seed。

runner 已增加 confirmation 启动与判定硬约束：脏 worktree 在首个 Episode 前被拒绝，最终
报告必须精确覆盖冻结的 100 个 seed，且每个 seed 恰有一条 baseline 和一条 teacher 结果。
本轮在脏开发树上调用 confirmation 命令时退出码为 `1`，未创建输出文件、未运行 Episode。

### 4. 原始产物

- 最终 paired development artifact：
  `runs/research-loop/0019/development/paired-seeds-19001-19006.json`
- SHA-256：
  `dc61ccbfdd98e7fa8a5bde47f86f7b861385a8c21580e6a1e6c76c3fbd2a1816`
- artifact 大小：`57,402` bytes。
- runner 同时写出 `.sha256` sidecar；`runs/` 受 `.gitignore` 管理，不纳入 Git。

### 5. 验证

```text
42 passed
Python size check passed: 461 files, file <= 800 lines, function <= 200 lines
Architecture check passed: engine, foundation, and core boundaries are intact
```
