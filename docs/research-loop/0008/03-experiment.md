# R0008 冻结实验

## 冻结状态

- 本文件先冻结 `R0001-P40-E2`。
- `R0001-P41-E2` 仅为条件选择，尚未冻结、不得实现：
  - 只有 P40-E2 正式结果为 accepted；
  - 且 candidate set、selector、primitive、ITT、MDE、样本量和安全非劣门全部结果前冻结；
  - 才允许在本文件追加独立的 P41-E2 实验段。
- 本轮不启动 P47、P48、P44-E1、P32-E2、P49、P46-E1 或正式训练。

## `R0001-P40-E2`：任务实体—机器人部位接触图与实体级冲量总账

### 研究问题

在不改变 action、physics、safety、reward、termination、success 或 policy observation 的
前提下，能否建立一个 evaluator-private、可守恒、可反例验证的接触图，使未来交互诊断能
区分：

1. base、left arm 与 right arm；
2. 具体 manipulated object、target container、floor/support 与 articulation；
3. 左右臂接触同一对象、左右臂接触不同对象和单臂接触；
4. robot–environment contact 与 task-relevant world–world contact；
5. 同一控制周期内的 contact-associated entity motion 与无接触运动。

本实验只验证测量合同。即使通过，也不把接触同期运动称为 controlled motion、action
causality、任务能力或安全改善。

### 冻结实现范围与负责人

- 唯一负责人：P40-E2 实施 Agent。
- 工作分支：`feat/research-loop` 当前分支上的原子实现提交。
- 允许修改：
  - 新增 `src/hwr/adapters/mujoco/entity_contact_graph.py`；
  - `src/hwr/adapters/mujoco/__init__.py`；
  - 新增 `src/hwr/apps/evaluate_entity_contact_graph.py`；
  - 新增 `tests/test_entity_contact_graph.py`；
  - 新增 `tests/test_entity_contact_graph_app.py`。
- 原则上不修改：
  - `src/hwr/adapters/mujoco/formal_household_backend.py`；
  - `src/hwr/adapters/mujoco/contact_ledger.py`；
  - `src/hwr/adapters/mujoco/bindings.py`；
  - `configs/adapters/mujoco/formal_3d_v1.json`；
  - 任何训练、policy、reward、safety 或 success 文件。
- 若实现发现必须越过上述边界，停止并交主 Agent 重新冻结，不自行扩展。

### 冻结身份与命名

- proposal：`R0001-P40-E2`
- report schema：`hwr.entity-contact-graph-contract-report/v1`
- artifact schema：`hwr.entity-contact-graph-contract-artifacts/v1`
- measurement schema：`hwr.mujoco-entity-contact-graph/v1`
- robot part：
  - `base`
  - `left_arm`
  - `right_arm`
- robot part body root：
  - `base -> robot_base`
  - `left_arm -> left_shoulder_pan_link`
  - `right_arm -> right_shoulder_pan_link`
- 每个 robot geom 沿 body parent 链选择最近的上述 root；全部 robot geom 必须恰好解析到一类。
- environment entity：
  - `floor_support:<geom-name>`
  - `manipulated_object:<object-id>`
  - `target_container:<geom-name>`
  - `articulation:<articulation-id>`
  - `forbidden:<geom-name-or-id>`
- task-relevant world–world edge 只记录：
  - manipulated object–floor/support；
  - manipulated object–target container；
  - manipulated object–articulation；
  - 两个不同 manipulated object。
- 其他 world–world 与 robot self-contact 继续单独计数，但不伪装成任务实体边。

### 冻结聚合语义

- 每个 physics substep：
  1. 读取所有 contact point；
  2. 每个 contact point 调用 `mj_contactForce` 读取 normal force；
  3. 先按无序 geom pair 求和；
  4. robot–environment pair 唯一映射为
     `robot part × environment entity`；
  5. task-relevant world–world pair 唯一映射为无序 entity pair；
  6. 再更新 substep peak、control-period impulse 与 Episode cumulative impulse。
- `normal_force is missing/nonfinite/negative` 一律计数并 fail-closed。
- robot–environment 子账按 P40 role 聚合后必须与并行启用的 P40-E1 ledger 完全一致：
  - pair peak；
  - category peak；
  - cumulative impulse；
  - contact duration；
  - contact point count；
  - unique pair observation count。
- world–world 子账单独报告，不参与上述 P40 守恒式。
- 同一 entity 的左右臂接触：
  - `same_substep_dual_arm_contact` 只在同一 physics substep 内 left arm 和 right arm
    都接触该 entity 时成立；
  - `distinct_entity_dual_arm_contact` 单独报告，永不折算成 same-entity 正例。
- manipulated object grasp：
  - 复用现有 `GraspContactMonitor` 语义；
  - 每只臂必须该臂两个 gripper pad 同时接触同一 object 才为 grasp-qualified；
  - same-object dual-arm grasp 要求同一 physics substep 两只臂均对同一 object
    grasp-qualified。
- entity motion：
  - manipulated object 使用同一 control period 起止 collision geom 世界坐标的欧氏距离；
  - articulation 使用同一 control period 起止 joint position 的绝对差；
  - 只报告 `contact_associated_motion`，不使用 `controlled_motion` 命名；
  - motion 只有在同一 entity 的有效 robot contact 于该 control period 内出现时才关联；
  - 无接触运动、reset settling 和接触结束后下一 period 的惯性运动不得关联；
  - 报告原始位移，不在 P40-E2 中使用 `0.01m` 作为能力或接受阈值。

### 冻结纯逻辑 fixture

固定构造下列反例并要求精确分类：

1. left/right 同时接触同一 object；
2. left/right 同时接触不同 object；
3. 只有一臂接触 object；
4. base/floor-only contact；
5. object 在无 robot contact 时移动；
6. object 在上一 period 接触、本 period 无接触时惯性移动；
7. arm/articulation handle contact 与 articulation position change；
8. object/target-container impact；
9. object/floor-support contact；
10. 同一 geom pair 的多个 contact point；
11. missing、NaN、Inf 与 negative normal force；
12. 映射缺失、重叠、未知 root、未知 entity role。

接受要求：

- task-relevant edge classification precision/recall 均为 1；
- same-entity、distinct-entity、single-arm 互不混淆；
- contact-associated motion 正例、无接触和惯性反例全部正确；
- pair 去重后 force/impulse 数值与手算一致；
- 所有无效证据 fail-closed。

### 冻结正式 trace

- 任务：
  - `tidy_living_room_3d/v1`
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
- seed：
  - base `20264002`
  - task stride `104729`
- 每任务 32 control step。
- action：reset 后固定 hold，保持当时左右 gripper position；无 policy inference。
- camera：reset 取得初帧后关闭 rendering。
- 每任务运行：
  - P40-E1 与 P40-E2 均 disabled 的 control trace；
  - P40-E1 与 P40-E2 均 enabled 的同 seed trace。
- legacy trace 必须逐字段 bit-identical：
  - applied action；
  - proprioception；
  - reward；
  - terminated/truncated；
  - result success/reason；
  - severe collision；
  - maximum forbidden force/pair；
  - safety intervention。
- 正式结果必须发布：
  - 完整 mapping；
  - 每 task robot–environment graph；
  - 每 task task-relevant world–world graph；
  - 每 period contact 与 entity motion；
  - P40 聚合守恒差；
  - ignored/invalid counts；
  - physics timestep、solver、iterations、tolerance 与 substeps。

### 接受标准

全部满足才标记：

`accepted as entity-contact measurement contract evidence`

检查：

1. 三任务 robot geom 全部且唯一解析到 base/left/right。
2. 三任务 object、allowed role 和 articulation entity 全部可解析且无冲突。
3. 全部纯逻辑 fixture 通过。
4. 三任务 enabled/disabled legacy trace bit-identical。
5. P40-E2 robot–environment 子账聚合回 P40-E1 的全部冻结字段，绝对差 `<=1e-12`。
6. world–world 子账不进入 P40 守恒式。
7. formal trace 与 fixture 的 missing/nonfinite/negative count 均为 0。
8. same-entity dual-arm、distinct-entity dual-arm、single-arm 和 no-contact motion 分账完整。
9. report 固定：
   - `measurement_only=true`
   - `policy_inference_executed=false`
   - `closed_loop_capability_episode_executed=false`
   - `capability_claim_allowed=false`
   - `hardware_safety_claim_allowed=false`
   - `action_causality_claim_allowed=false`
   - `legacy_runtime_behavior_unchanged=true`
10. report、manifest、命令、source commit、binding identity 与所有直接产物 hash/bytes
    完整可重建。

若映射、fixture、守恒、fail-closed、bit-identity 或 provenance 任一失败：

- 可明确否定合同则 `rejected`；
- 环境/API 无法提供所需证据则 `inconclusive`；
- 均停止 P41-E2，不修改门槛。

### 稳定命令与产物

正式 run：

`runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002`

命令：

```text
.venv/bin/python -m hwr.apps.evaluate_entity_contact_graph \
  --output runs/research-loop/0008/r0008-p40-entity-contact-e2-s20264002
```

必须原子写入：

- `report.json`；
- `manifest.json`；
- 失败时 `failure.json`。

manifest 必须记录：

- proposal 与 schema；
- source commit；
- 冻结文档提交为祖先；
- 稳定命令；
- binding hash/bytes；
- robot body-root identity；
- MuJoCo version 与 physics 配置；
- 所有直接产物 hash/bytes；
- 禁止声明位。

## 实施后门禁

先运行 focused tests，再运行：

```text
.venv/bin/python scripts/check_python_size.py
.venv/bin/python scripts/check_architecture.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m compileall -q src tests scripts
git diff --check
```

正式 run 前必须：

1. 本冻结文档提交是实现提交祖先；
2. 实现已原子提交；
3. 工作区干净；
4. focused tests 与共同门禁通过；
5. binding hash 与冻结提交一致；
6. 历史 `docs/research-loop/0001/`～`0007/` tree 零差异。

正式 run 后、P41-E2 冻结前必须运行全量：

```text
.venv/bin/pytest -q
```

## 本阶段停止条件

- P40-E2 不是能力实验，不启动正式训练，不使用 MPS，不休眠。
- P40-E2 未接受：记录结果并停止 P41-E2。
- P40-E2 接受：只授权追加 P41-E2 的结果前冻结合同，不自动授权实现或正式 run。
