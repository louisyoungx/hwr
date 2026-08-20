# R0007 独立筛选

## 评审过程

- 两名筛选 Agent D、E 在同一版 `00-context.md` 与 `01-proposals.md` 上独立只读评审。
- 两者完成前未查看对方结果，未修改实现或文档，未启动训练。
- 每项按以下七维分别 1–5 分：
  - 目标价值；
  - 证据强度；
  - 可检验性；
  - 因果可归因性；
  - 通用性；
  - 实施成本，5 表示成本低；
  - 回归风险，5 表示风险低。
- 主 Agent 在收到两份完整评审后才汇总，不按总分机械选择。

## 筛选 Agent D

| ID | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P32-E1` | 5 | 4 | 4 | 4 | 3 | 4 | 5 | `select` |
| `R0001-P40-E1` | 4 | 5 | 4 | 5 | 4 | 4 | 4 | `select` |
| `R0001-P41-E1` | 4 | 3 | 2 | 2 | 3 | 2 | 2 | `changes_required` |
| `R0001-P42-E1` | 4 | 4 | 4 | 4 | 4 | 2 | 4 | `defer` |
| `R0001-P43` | 4 | 5 | 1 | 3 | 2 | 3 | 3 | `defer` |
| `R0001-P44` | 4 | 4 | 4 | 5 | 3 | 2 | 4 | `defer` |
| `R0001-P45` | 4 | 3 | 2 | 2 | 3 | 3 | 4 | `changes_required` |
| `R0001-P46` | 3 | 4 | 3 | 4 | 4 | 2 | 3 | `changes_required` |

### D 的主要反驳

- P32-E1：
  - 通过也只能证明 salience-retained Replay 条件信息，不能升级为 plant causality 或
    production utilization；
  - rate/configuration target 的逐列索引、单位、yaw wrap、ridge、标准化、planted
    scale、bootstrap 和多重判定层级必须在实现前唯一化；
  - 24 个 source 的统计功效有限，rank/power 失败时只能 `inconclusive`。
- P40-E1：
  - MuJoCo force 依赖 solver/timestep，旧 220N 只能作内部参照；
  - 必须冻结 robot–environment 范围、自接触与 world–world 处理、显式 geom 角色、无序
    pair 去重、substep 积分和非有限值语义；
  - 新测量不能进入安全决策。
- P41-E1：
  - geometry-matched blind control 仍不唯一；若使用仿真真值匹配则引入 privileged
    control，若复用 candidate 结果则发生后验泄露；
  - 缺少最小效应、样本量与 exact-pipeline power；
  - 必须等待 P40-E1。
- P42-E1：
  - 当前没有 qualified deployment；
  - 客厅没有互斥 target，餐桌 target 几何不等价；需先证明反事实 mapping 均可完成且
    近似等难。
- P43：
  - 只读复算确认 37 个 positive transition、7 个 positive window、6 个 positive
    source；task 分布为餐桌 2、厨房 1、客厅 3；
  - 厨房无法同时形成含正例的 source-disjoint train/test；
  - 当前 trainer 没有真正 head-only 的优化路径；
  - 与 R0002 P10 高度重复，必须先补独立正例 source。
- P44：
  - nominal success 很低时有 floor effect；
  - low/high tail 必须分别判定，不能用均值掩盖单尾崩溃；
  - 等待 qualified deployment。
- P45：
  - 128-step source 中比较连续 112 与离散 112，覆盖差异过小；
  - 连续视图会同时改变窗口重叠和训练 exposure，不能只归因为 topology；
  - 应等待 P32 evaluator。
- P46：
  - 20-update fixture 不能验证 50-update cadence；
  - mid-cycle snapshot 必须保存 RNG、sampler、partial metrics 和 batch identity，且不能
    冒充经过 causality gate 的 latest deployment。

D 的执行建议：

1. 主路径实施 P32-E1；
2. P40-E1 由独立负责人和独立文件集合并行实施；
3. P43 本轮不进入执行序列。

## 筛选 Agent E

| ID | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P32-E1` | 5 | 4 | 4 | 4 | 3 | 5 | 5 | `select` |
| `R0001-P40-E1` | 4 | 5 | 4 | 5 | 4 | 4 | 4 | `select` |
| `R0001-P41-E1` | 4 | 3 | 2 | 3 | 3 | 3 | 2 | `defer` |
| `R0001-P42-E1` | 4 | 5 | 3 | 4 | 3 | 3 | 4 | `defer` |
| `R0001-P43` | 5 | 3 | 2 | 3 | 3 | 3 | 3 | `defer` |
| `R0001-P44` | 4 | 5 | 3 | 5 | 3 | 2 | 4 | `defer` |
| `R0001-P45` | 4 | 3 | 3 | 2 | 4 | 3 | 4 | `changes_required` |
| `R0001-P46` | 3 | 4 | 4 | 4 | 4 | 3 | 3 | `approve` |

### E 的主要反驳

- P32-E1：
  - outer-train 每 task 只有 4/4/8 个 source；inner-train 进一步减半，高维 controller
    context 可能使 ridge 的 planted power 不足；
  - 16-D rate target 尚未在提案中逐列唯一化；
  - planted effect 不得从 outer-test residual 标定；
  - source 等权和 task 等权必须同时冻结，不能让客厅或长 Episode 主导。
- P40-E1：
  - allowed 列表当前混合 floor、对象、容器和 drawer 部件；
  - category peak、pair peak、contact-point peak 语义必须唯一化；
  - timestep 减半只应检查预注册冲量容差，不能要求 solver 轨迹逐步相同；
  - P01 Replay 没有完整 contact trace，不能追溯重建历史安全总账。
- P41-E1：
  - 缺少数值接受门和 paired power；
  - candidate-set 生成、surface ranking、左右臂分配与可达性求解仍可能捆绑多变量。
- P42-E1：
  - 客厅两个对象共享同一目标，交换无反事实意义；
  - evaluator-private success mapping 不得通过共享 task spec、audit 或进程对象泄露。
- P43：
  - 同样确认仅 6 个 positive source，厨房只有 1 个；
  - 三优化 seed 不增加统计独立性；
  - 当前 `world_optimizer` 会更新整个 world model，直接复用违反冻结 RSSM/其他 head；
  - 即使通过，Actor readiness 仍受 action-information 等其他门阻塞。
- P44：
  - 当前 actuator scale 只作用于 action 前 14 维，不包括两个 gripper target；
  - 不能外推为完整执行器或硬件迁移。
- P45：
  - 比较同时改变 outcome-aware selection、连续性、边界数量和时间覆盖；
  - P32 是一步诊断，当前设计无法单独识别连续性机制；
  - 应增加 outcome-blind 7×16 对照并冻结短 Episode unresolved 规则。
- P46：
  - 建议批准为后续系统候选，但不进入本轮前二；
  - 需独立 mid-cycle recovery pointer、durable flush、损坏回退、CPU bit-identity 和
    MPS 预注册容差。

E 的执行建议：

1. 主关键路径实施 P32-E1；
2. P40-E1 作为无 accelerator 竞争的独立侧车并行实施；
3. P43 不与 P32 并行，且 P32 通过也不自动批准 P43。

## 主 Agent 证据复核

### P32 数据与目标索引

- Replay manifest 实际 hash 与提案一致。
- 24 个 source、168 shard、2,688 transition、task/source 分布和每 source 7 shard 均经
  实际 manifest 与 NPZ 复核。
- 37-D `DualArmProprioception.vector()` 的顺序为：
  - `0:6`：left joint position；
  - `6:12`：left joint velocity；
  - `12:18`：right joint position；
  - `18:24`：right joint velocity；
  - `24`：left gripper position；
  - `25`：right gripper position；
  - `26:29`：base pose `(x,y,yaw)`；
  - `29:31`：base twist；
  - `31:37`：IMU。
- 现有 action probe 的 16-D controllable state 索引确为
  `6:12 + 18:26 + 29:31`，target 是 successor 与 current 的差，而不是绝对值。

### P43 独立性

- 主 Agent 只读复算与两名筛选 Agent 一致：
  - 37/2,688 positive transition；
  - 7/168 positive window；
  - 6/24 positive source；
  - 餐桌 2、厨房 1、客厅 3 个 positive source。
- 因此当前数据无法满足三任务均具有 source-disjoint positive train/test 的必要条件。
- P43 不是简单低成本训练：当前 optimizer 会更新整个 world model，需另建 detached
  head-only 路径，且仍不能修复独立 source 不足。

### P40 运行时证据

- 正式后端每 physics substep 已调用 `_after_physics_substep()`。
- `_scan_forbidden_contacts()` 对 robot–other pair 先检查 allow-list；allowed geom 在调用
  `mj_contactForce` 前直接跳过。
- 当前 binding 只有一个 flat `allowed_robot_contact_geoms` 集，没有语义角色。
- 因此 P40-E1 的测量盲区真实存在；修复可保持当前 safety decision 完全不变。

## 主 Agent 决策

### `R0001-P32-E1`：选择

- 两名筛选 Agent 均选择。
- 它直接回答 R0007-C01，成本低、无需 accelerator、不会改变 production 行为。
- 本轮先实现 exact-pipeline power 和泄露守护；若 rank/power 不足，接受
  `inconclusive`，不得降低门槛或转入训练。
- 正结果只授权后续修订 P31，不直接授权 P43、P33、Actor 解锁或正式训练。

### `R0001-P40-E1`：选择

- 两名筛选 Agent 均选择。
- 它是独立、直接、可归因的安全测量合同，不依赖 qualified deployment。
- 与 P32 的代码文件、算力和结论完全分离，可由唯一负责人并行实施。
- 首轮只接受测量合同与 deterministic fixture 证据，不形成硬件安全阈值或能力结论。

### 其他提案

| ID | 本轮决策 | 原因 |
|---|---|---|
| `R0001-P41-E1` | `changes_required` | 缺唯一 blind-control 合同、MDE 与 power；依赖 P40-E1 |
| `R0001-P42-E1` | `deferred` | 无 qualified deployment；部分任务无等价互斥 mapping |
| `R0001-P43` | `deferred` | 厨房仅 1 个 positive source，无法无泄露分割；需新数据 |
| `R0001-P44` | `deferred` | 无 qualified deployment；当前只覆盖前 14 维 gain |
| `R0001-P45` | `changes_required` | selection、连续性、边界和 exposure 多变量混淆 |
| `R0001-P46` | `approved, not selected` | 系统价值成立，但不回答当前能力瓶颈且修改训练核心 |

## 依赖与停止规则

1. P32-E1 和 P40-E1 可在独立文件所有权下并行实施。
2. P32-E1 rank 或 exact-pipeline power 不足：结果为 `inconclusive`，不启动 P43 或训练。
3. P32-E1 在功效充分时失败：不继续在同一 Replay 上扩大 action-objective 或 head
   训练，下一轮优先改善数据采集支持。
4. P32-E1 通过：仅提高修订 P31 的优先级；P43 仍需新增独立 positive source。
5. P40-E1 分类不完备、force 非有限、pair 重复计数、timestep 稳定性失败或改变旧行为：
   拒绝当前实现，不启动 P41-E1。
6. 本轮不启动正式训练，不运行 capability benchmark，不生成能力结论。
