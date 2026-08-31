# R0020 结果

状态：结束。

## 运行总账

固定 development seed `19001` 共运行 3 个完整 physics Episode，达到预注册 candidate
预算。三次均经正式 16 维动作接口、正常 MuJoCo 物理、`DualArmSafetySupervisor` 与两步
predictive collision filter 执行。

依据 `f7b27a3` 后的新规则，以上“达到预注册 candidate 预算”的旧判读已撤销：attempt 1～3
分别发生在实现修改之后，现统一归类为 implementation iteration 1～3。它们保留原文件名、
原始 artifact、hash 与历史文字，不覆盖、不重命名；三次都未达到 2026-08-31 重开后冻结的
behavior entry，因此不是三次独立重复证据，也不构成联合规划路线的可判别失败。

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

上述停止决定属于旧规则下的历史记录。R0020 已于 2026-08-31 重新打开；新 debug 与候选判别
预算以 `01-experiment.md` 的重开修订为准。历史 attempt 1～3 不计入新 `24` smoke 上限。

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
- 以下初次收尾验证保留为历史记录：focused tests `4 passed`、bimanual 回归
  `50 passed`、Python 尺寸检查 `465` 个文件通过、architecture check 通过。

## 2026-08-31 重开调试

重开后共运行 11 个有原始记录的 seed 19001 短程 physics smoke，未达到 `24` smoke 上限；
artifact 内 wall time 合计 `72.1399s`，从 smoke 001 到 smoke 011 的本地时间跨度约 20 分钟，
未达到 2 小时主动调试上限。所有 smoke 都使用正式 16 维动作接口、正常 MuJoCo physics 与
原安全层。

| smoke | 关键实现变化 | steps | 最大连续双臂接触 | 最终阶段 | entry | safety / severe |
|---|---|---:|---:|---|---|---|
| 001 | 重开基线 | 406 | 0 | `failed_hold` | 否 | 0 / 0 |
| 002 | 在线 pad/handle 目标 | 406 | 0 | `failed_hold` | 否 | 0 / 0 |
| 003 | 未对中时主动张开 | 406 | 0 | `failed_hold` | 否 | 0 / 0 |
| 004 | 在线闭环提前接管 | 406 | 0 | `failed_hold` | 否 | 0 / 0 |
| 005 | 终端碰撞约束初版 | 50 | 0 | `approach` | 否 | 0 / 0 |
| 006 | 路径感知联合 planner | 390 | 0 | `failed_hold` | 否 | 0 / 0 |
| 007 | 对中后继续闭爪 | 390 | 0 | `failed_hold` | 否 | 0 / 0 |
| 008 | 轻微负 signed-distance 目标 | 390 | 0 | `failed_hold` | 否 | 0 / 0 |
| 009 | 正式 gripper 上限与固定 acquire 时域 | 550 | 2 | `failed_hold` | 否 | 0 / 0 |
| 010 | 在线双 pad 距离平衡 | 550 | 2 | `failed_hold` | 否 | 0 / 0 |
| 011 | 双 pad 接触后维持闭爪预载 | 204 | 11 | `secure` | 是 | 0 / 0 |

关键修复链：

1. 旧 planner 把所有篮子 geoms 视为允许 robot contact，静态四 pad 解同时包含最大约
   `45mm` palm/wall 穿透；重开实现只允许四个 pad–对应 handle 接触，并把四点插值路径穿透
   加入联合搜索的精英选择。
2. joint path 只运行到无接触近场；之后由实时 pad 中点、handle pose、两 pad signed
   distance 与接触对控制两臂目标和闭爪。
3. 在线闭爪在未对中时主动张开；对中后渐进闭合；两 pad 距离不平衡时施加最大 `3mm`
   横向修正；形成双 pad 接触后维持 gripper target `1.0`。

smoke 011 达到冻结 behavior entry：

- seed `19001`；
- 进入 `secure`；
- `maximum_concurrent_steps=11`；
- 四个 pad 均产生真实 handle contact；
- `0` safety intervention；
- `0` actual severe collision；
- artifact：
  `runs/research-loop/0020/debug/smoke-011-contact-preload.json`；
- SHA-256：
  `a2cc719233ae95dd8137dfe41a785ddcc73bd1a151ce63358d1b977044a3a053`。

冻结 manifest：

- `runs/research-loop/0020/debug/behavior-entry-freeze.json`；
- SHA-256：
  `2a59912d914d35655a454448505a1f9fd12b0ec8f9c228d3fbe3c92e68d6dea7`；
- 冻结的三个 controller 文件 hash 在完整 Episode 后复核一致。

## 重开候选判别

behavior entry 后运行且只运行一个完整 seed 19001 Episode：

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_joint_basket_teacher \
  --seed 19001 \
  --output runs/research-loop/0020/development/reopened-candidate-seed-19001.json
```

结果：

- `0/1` success；
- 执行 `1200` step，以 `bimanual_task_timeout` 结束；
- stages reached：
  `approach/acquire/secure/failed_hold`；
- `maximum_concurrent_steps=11`，确认差异化 acquire 机制在判别 Episode 中实际执行；
- `left_contact_steps=11`、`right_contact_steps=17`、`simultaneous_contact_steps=11`；
- 进入 `secure` 后接触丢失，`teacher_failure_stage=secure_timeout`；
- `maximum_lift_m=0`、`maximum_controlled_target_progress=0`；
- `0` safety intervention、`0` actual severe collision、最大 forbidden force `0N`；
- artifact：
  `runs/research-loop/0020/development/reopened-candidate-seed-19001.json`；
- SHA-256：
  `870321f7c935b84dc8899fb8bec34b7ae2405e38a58bba3b77e65004d733c170`；
  `5,671` bytes。

完整 seed 19001 未端到端成功，因此不满足 cohort 升级条件。development cohort
`19001`～`19004`、confirmation 和 sealed final 均为 `not_run`。

## 重开收尾验证

- R0020 focused tests：`9 passed`。
- bimanual 相关回归：`55 passed in 28.22s`。
- Python 尺寸检查：`466` 个文件通过，文件不超过 800 行、函数不超过 200 行。
- architecture check：通过。
- `git diff --check`：通过。
