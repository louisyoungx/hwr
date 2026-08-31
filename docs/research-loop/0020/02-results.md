# R0020 结果

状态：结束。

## 运行总账

固定 development seed `19001` 共运行 3 个完整 physics Episode，达到预注册 candidate
预算。三次均经正式 16 维动作接口、正常 MuJoCo 物理、`DualArmSafetySupervisor` 与两步
predictive collision filter 执行。

共同命令形式：

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_joint_basket_teacher \
  --seed 19001 \
  --output runs/research-loop/0020/development/seed-19001-attempt-<N>.json
```

| attempt | success | stages reached | failure | 双臂接触 | 最大抬升 | safety | severe |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 0 | `approach/acquire/failed_hold` | `acquire_timeout` | 0 step | `0.060549m` | 0 | 0 |
| 2 | 0 | `approach/acquire/failed_hold` | `acquire_timeout` | 0 step | `0m` | 0 | 0 |
| 3 | 0 | `approach/acquire/failed_hold` | `acquire_timeout` | 0 step | `0m` | 0 | 0 |

三次均执行满 `1200` step，并以 `bimanual_task_timeout` 结束。三次最大 forbidden force
分别为 `25.329396N`、`22.219264N`、`0N`，均低于 `220N` severe 门。

attempt 1 的 planner 在 reset 后、篮子完全落稳前求解；执行中篮子被非任务有效接触扰动，
出现 `0.060549m` 高度变化，但 tracker 没有记录左右任一完整双 pad 接触，
`maximum_controlled_target_progress=0`。该高度变化不是抓取或抬升成功。

attempt 2 改为底盘到位、篮子落稳后重规划；静态末态预测的 maximum pad signed distance 为
`0.000582m`，但动态执行在进入最后 waypoint 时仍有约 `0.074rad` 关节误差并提前闭爪。
一次用于定位该控制时序的短诊断未保留独立 artifact，记为“未归档观察”，不支撑最终结论；
最终结论只使用三个完整 Episode。

attempt 3 在重规划收敛后才进入 acquire，并要求最终 waypoint 误差低于 `0.018rad` 才锁存
闭爪。静态末态 maximum pad signed distance 为 `0.000747m`，但正式动态执行仍为左右
`0/0` contact step，未进入 `secure`。这表明当前 joint-space waypoint tracker 没有把静态
接触解转化为动态可闭合抓取；本轮不能进一步归因 `lift/transport/place/release/stabilize`。

## 升级与停止

- 单 seed 升级门未通过：`0/3` 完整成功。
- 小型 development cohort `19001`～`19004`：`not_run`。
- confirmation：`not_run`，`valid=null`。
- sealed final：`not_run`，`valid=null`。
- 已达到预注册 3-Episode candidate 预算，停止该 candidate；不调换 seed、不扩 cohort、
  不自动启动另一技术路线。

## 原始产物

- attempt 1：
  `runs/research-loop/0020/development/seed-19001-attempt-1.json`，
  SHA-256 `2bd0f644194c8c30a2c83a298dbd797e1cbdd0f8cf1ef84a29742c6b0d009839`，
  `5,221` bytes，Episode wall time `6.5328s`。
- attempt 2：
  `runs/research-loop/0020/development/seed-19001-attempt-2.json`，
  SHA-256 `1694886ec73874529bc5d3470bea5f11e85bb472fecc297f8763c90eabc034d3`，
  `5,178` bytes，Episode wall time `6.6917s`。
- attempt 3：
  `runs/research-loop/0020/development/seed-19001-attempt-3.json`，
  SHA-256 `7fcb863c1d88bfdec6c21cf607f9bc78b21924e138f34835eb1258e441898a30`，
  `5,177` bytes，Episode wall time `7.8084s`。
- 每个 JSON 均有同目录 `.sha256` sidecar；`runs/` 受 `.gitignore` 管理。

## 实现与验证

- 新实现由 `joint_basket_planner.py` 的联合 13 维抓取构型搜索、插值路径检查，以及
  `joint_basket_teacher.py` 的关键帧跟踪与完整阶段状态机构成。
- planner 在复制的 `MjData` 上运行，focused test 验证其不修改权威 `qpos/qvel/ctrl`。
- focused tests：`4 passed`。
- bimanual 回归：`50 passed`。
- Python 尺寸检查：`465` 个文件通过，文件不超过 800 行、函数不超过 200 行。
- architecture check：通过。
