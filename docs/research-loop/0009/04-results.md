# R0009 结果

## `R0001-P51`：`accepted as Cartesian primitive correctness evidence`

### 完整性

- 冻结文档提交：
  `4385ceee2fffcbd23788b498d258747dc273465c`
- 初始实现提交：
  `74ec4332830d93ecfb60e560450999a8ae917cf9`
- primitive integration gate 加固提交：
  `b810f73df6be05d25041ee64b3e898df08598c35`
- 最终 source commit：
  `0ad97bb7ee0d37a8f308c1ea9ffc705550891acb`
- 冻结提交是全部实现和加固提交的祖先。
- 最终正式 run：
  `runs/research-loop/0009/r0009-p51-cartesian-frame-s20265101`
- 稳定命令：

```text
.venv/bin/python -m hwr.apps.evaluate_cartesian_frame_contract \
  --output runs/research-loop/0009/r0009-p51-cartesian-frame-s20265101
```

- mode：deterministic analytic coordinate contract；
- device：CPU；
- policy inference：未执行；
- closed-loop physics：未执行；
- 参数更新/checkpoint：无。

最终产物：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 226,555 | `763270993a9199c7997c305f1f040794a631dc13fee961e022204b25c0b6c016` |
| `manifest.json` | 4,910 | `5f47bdddf829ba0b08d94fa499e495ffc542919f07ae48ed225dda9417301465` |

report 与 manifest 的 `source_commit` 均为
`0ad97bb7ee0d37a8f308c1ea9ffc705550891acb`。artifact hash/bytes 逐项复算一致。

### 坐标合同结果

冻结矩阵完整执行：

- 3 个 acquisition yaw；
- 6 个 relative yaw；
- 4 个 acquisition-frame error vector；
- 左右两臂；
- 共 144 个 deterministic cell。

candidate 将裁剪后的 acquisition-frame 线速度按
`Rz(acquisition_yaw - current_base_yaw)` 转到 current-base frame。backend 再转回
world/acquisition frame后的误差：

| 指标 | 最大值 | 门槛 |
|---|---:|---:|
| 最大绝对误差 | `1.7763568394002508e-17` | `<=1e-12` |
| 水平角误差 | `2.7105054312137616e-16 rad` | `<=1e-12 rad` |
| norm 误差 | `1.3877787807814457e-17` | `<=1e-12` |
| z 误差 | `0.0` | `<=1e-12` |

所有 relative-yaw 为 0 的 candidate/legacy float64 bytes 完全一致。冻结的 48 个
`relative yaw=±pi/2` legacy 反例全部产生至少 `pi/2-1e-12` 的水平角误差，证明合同对旧
语义具有判别力。

### `primitive_action` 集成守护

首次正式 runner 虽正确验证 helper 公式，但独立审查指出 helper 可能正确而
`primitive_action` 集成错误，仍会产生假接受。该 run 未被用作最终证据，原样保留在：

`runs/research-loop/0009/r0009-p51-cartesian-frame-s20265101-superseded-8fbfbfa`

加固后的正式 runner 直接调用 `primitive_action`：

- 20 个 phase × relative-yaw case；
- 28 次双臂 transform 调用；
- 2 个 hold/fail-closed case；
- 覆盖：
  - B0～B7 全部 phase；
  - target、velocity cap 与 transform 参数；
  - 左右臂 linear command；
  - arm angular command 保持为 0；
  - base command、gripper、hold、安全 fail-closed；
  - canonical action bounds；
  - relative-yaw 为 0 时 legacy byte identity。

14 项 primitive integration check 全部通过。负向测试证明：

- helper 被绕过；
- linear/angular/gripper 字段被篡改；
- phase、hold 或 bounds 被篡改；

都会令 report decision 变为 `rejected`。

### 唯一行为变化

`src/hwr/eval/target_selection.py` 只改变：

```text
acquisition-frame linear error
  -> clip_norm
  -> Rz(acquisition_yaw - current_base_yaw)
  -> current-base-frame arm command
```

最终 report 明确记录以下字段全部为 `false`：

- `candidate_generator_changed`
- `candidate_bytes_changed`
- `selector_changed`
- `acquisition_changed`
- `phase_changed`
- `target_changed`
- `velocity_cap_changed`
- `gripper_changed`
- `backend_changed`
- `safety_changed`

### 判定

全部冻结解析门和加固后的实际 primitive integration 门通过，正式标记：

`accepted as Cartesian primitive correctness evidence`

该接受只证明 P41 fixed primitive 的线速度 frame 语义正确。它不证明：

- candidate 对应任务实体；
- arm 能实际接触实体；
- target selection 提高交互产率；
- 家务任务成功；
- 学习、泛化或硬件安全改善。

## `R0001-P52`：`accepted as FK agreement contract evidence`

### 完整性

- 冻结文档提交：
  `4385ceee2fffcbd23788b498d258747dc273465c`
- 初始实现提交：
  `8fbfbfafc37fe8181d2f953a9a72aaf1435fa7ba`
- evaluator isolation/provenance 加固提交：
  `0ad97bb7ee0d37a8f308c1ea9ffc705550891acb`
- 最终 source commit：
  `0ad97bb7ee0d37a8f308c1ea9ffc705550891acb`
- 最终正式 run：
  `runs/research-loop/0009/r0009-p52-tool-kinematics-s20265201`
- 稳定命令：

```text
.venv/bin/python -m hwr.apps.evaluate_tool_kinematics \
  --output runs/research-loop/0009/r0009-p52-tool-kinematics-s20265201
```

- mode：latency-free evaluator-private kinematics measurement；
- device：CPU；
- observation latency queue：未使用；
- policy/action/physics step：未执行；
- 参数更新/checkpoint：无。

最终产物：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 1,264,273 | `af8b245ac4654dcebf0e0af57ceac9b7f5fc288b3cac42deb3c894186fafe3fb` |
| `manifest.json` | 12,821 | `8309a6366b8eeb277db237bfcfca06b6fe3601a3aa1a2e01eed9b67d2a65b0a2` |

report 与 manifest 的 `source_commit` 均为
`0ad97bb7ee0d37a8f308c1ea9ffc705550891acb`。artifact hash/bytes 逐项复算一致。

### 冻结 state grid

每个正式 task 使用相同 153-state grid：

- model `qpos0`：1；
- 左右臂每个 joint 的 20%/80% 单关节状态：24；
- seed `20265201` 的 12 维 scrambled-Halton central-range state：128。

三个 task、两臂全部完成：

- 153 states/task；
- 306 terminal/task；
- 918 terminal 总计；
- 无缺失、重复或非有限 terminal。

state grid 是确定性数值覆盖，不作为随机显著性样本。

### FK 误差

全部 918 个 terminal 的 aggregate：

| 指标 | 数值 |
|---|---:|
| mean Euclidean error | `1.9604804851278044e-16m` |
| median Euclidean error | `1.7663203618858437e-16m` |
| p95 Euclidean error | `4.652682298944613e-16m` |
| max Euclidean error | `7.021666937153402e-16m` |

最弱 task/arm 为客厅左臂：

- count：153；
- p95：`4.743512267633838e-16m`；
- max：`7.021666937153402e-16m`。

frame-invariance fixture 覆盖 identity、平移、正 yaw 和平移加负 yaw，最大误差
`4.440892098500626e-16m`，通过 `<=1e-12m` 门。

结果远低于冻结 agreement 门：

- aggregate p95 `<=0.01m`；
- aggregate max `<=0.02m`。

因此当前手写 policy FK 与 MuJoCo grasp-center site 在冻结机器人模型上数值一致；厘米级
FK mismatch 不是解释 P41 零 arm contact 的支持证据。

### 独立审查与加固

首次正式 runner 的数值结果正确，但独立审查发现：

1. evaluator-private isolation 由常量自证，存在未来假阳性路径；
2. clean source 未显式枚举 untracked files；
3. manifest 未直接绑定全部关键源码与递归 XML include；
4. deterministic replay 未按 task × arm 发布。

该 run 未作为最终证据，保留在：

`runs/research-loop/0009/r0009-p52-tool-kinematics-s20265201-superseded-8fbfbfa`

加固提交后：

- 可执行 AST audit 检查 P52 app/core 的：
  - action/action-frame import；
  - backend import；
  - `apply`、`step`、`mj_step*`；
  - selector/primitive；
  - 别名、包装引用和动态属性访问；
- 正式 audit 检查 2 个 source，violations 为 0；
- clean gate 显式使用 `--untracked-files=all`；
- manifest 直接绑定 8 个关键源码；
- 每个正式模型绑定 7 个递归 XML dependency；
- 6 个 task × arm replay payload 全部 bit-identical。

独立审查 Agent 在最终 HEAD 复核上述四项 finding，结论为无 remaining finding。

### 判定

全部冻结有效门、误差门和加固后的隔离/provenance 门通过，正式标记：

`accepted as FK agreement contract evidence`

该接受只证明冻结 state grid 上 policy FK 与 MuJoCo tool site 一致。它：

- 拒绝“厘米级 FK 错配是 P41 零接触主因”的当前假设；
- 不证明 primitive 实际收敛；
- 不证明 candidate 对应任务实体；
- 不证明接触、任务成功、泛化或硬件安全能力。

## 物理 smoke 停止决定

P52 的 agreement 结果使 P51 fixed-candidate MuJoCo smoke 成为允许项，但 R0009 冻结文档
没有唯一化该 smoke 的：

- environment seed/commitment；
- fixed candidate 集与 identity；
- paired baseline/candidate 样本量；
- tool-target distance 的时间窗和判定门；
- arm/entity contact 的接受与守护关系；
- early termination、缺失和多任务聚合规则。

在已看到 P51/P52 结果后补这些参数会形成后验实验设计。因此本轮不运行物理 smoke，不把
解析 correctness 外推为物理交互改善；下一轮如重提，必须重新创新、筛选并结果前冻结。

## 验证

- 实施 Agent：
  - P51 初始 focused tests：18 passed；
  - P51 加固后 focused tests：23 passed；
  - P52 初始 focused tests：12 passed；
  - P52 加固后 focused tests：25 passed；
  - 各自 size、compileall、diff check 通过。
- 主 Agent：
  - 最终 P51/P52 focused suite：48 passed；
  - P41/P51/P52 相关宿主回归：22 passed；
  - 全量宿主 pytest：822 collected，11 skipped，其余通过；
  - 18 条 warning 均为既有 `torch.jit.script` deprecation warning；
  - Python size：404 files 通过；
  - architecture、physics integrity、compileall、`git diff --check` 通过；
  - P51/P52 report/manifest source commit、artifact hash/bytes、冻结祖先、历史 tree 和
    claim flags 核验通过；
  - P52 source/XML provenance、isolation audit、task × arm replay 核验通过；
  - `docs/research-loop/0001/`～`0008/` tree 零差异。
- 沙箱中的 P41/dual-arm 渲染相关回归因
  `mujoco.cgl.cgl.CGLError: invalid CoreGraphics connection` 失败；相同 22 项在宿主环境
  全部通过。该错误与 P51/P52 逻辑无关。

## 当前能力结果

- 正式训练：未启动；
- policy inference：未执行；
- closed-loop capability Episode：未执行；
- 新家务任务成功：0；
- qualified deployment：无；
- P41 正式 selector 对照：未运行；
- 世界模型、Actor、闭环成功率、泛化或硬件安全改善：无声明。
