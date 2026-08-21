# R0008 独立筛选

## 筛选过程

- 筛选 Agent D、E 同时收到冻结的 `00-context.md` 与 `01-proposals.md`。
- 两者在提交评分前互不查看结果，不修改实现，不启动训练。
- 评分均为 1–5：
  - 目标价值；
  - 证据强度；
  - 可检验性；
  - 因果可归因性；
  - 通用性；
  - 实施成本，5 表示成本低；
  - 回归风险，5 表示风险低。
- 主 Agent 不按总分机械选择，而按当前瓶颈、依赖、测量有效性和因果顺序综合决策。

## 筛选 Agent D

| ID | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P40-E2` | 5 | 5 | 5 | 4 | 4 | 4 | 4 | `changes_required` |
| `R0001-P41-E2` | 5 | 4 | 3 | 3 | 3 | 2 | 2 | `changes_required` |
| `R0001-P47` | 3 | 4 | 3 | 4 | 3 | 3 | 3 | `defer` |
| `R0001-P48` | 4 | 5 | 5 | 5 | 4 | 3 | 4 | `approve` |
| `R0001-P44-E1` | 3 | 5 | 5 | 5 | 3 | 4 | 4 | `approve` |
| `R0001-P32-E2` | 3 | 3 | 4 | 2 | 2 | 5 | 2 | `changes_required` |
| `R0001-P49` | 4 | 5 | 4 | 3 | 3 | 3 | 3 | `changes_required` |
| `R0001-P46-E1` | 3 | 5 | 5 | 5 | 4 | 2 | 3 | `approve` |

### D 的关键反驳

- P40-E2：
  - 同一控制周期内的 contact 与 entity displacement 仍可能来自重力、惯性、容器或另一
    物体推动；
  - 非干预证据只能命名为 `contact-associated motion`，不能称为 controlled motion；
  - robot–environment 与 world–world 总账必须分开，只有前者参与旧 P40 守恒。
- P41-E2：
  - 仓库没有现成的通用 candidate generator、几何评分器和固定双臂 primitive；
  - candidate bytes、排序、去重、阈值、空集合、每步动作、ITT、MDE、样本量和安全
    非劣门必须在正式结果前唯一化；
  - 所有 planned pair 都进入 ITT；空候选、同 index、提前终止均不得删除；
  - 诊断轨迹不得静默进入普通 Replay。
- P47：
  - 若只分析 pulse 后发生接触的 Episode，会按处理后的中介变量筛选；
  - 必须等待 P40-E2 与 P41-E2，并保留所有 planned plus/minus/sham。
- P48：
  - 当前 formal aggregator 仍可能过度信任输入 report 的通过位；
  - 应重算 qualification hash chain，但不应挤占当前交互瓶颈。
- P44-E1：
  - pulse 若触发 clip，low/high scale 可能产生相同 actual action；
  - 必须使用不饱和正负内部动作，并只接受为评测合同。
- P32-E2：
  - 强 ridge 可能故意欠拟合 controller history，使 action residual 重新代理 FIFO/history；
  - 需增加保留 controller confounding 但不存在 action effect 的 null 和 nuisance OOF
    质量守护。
- P49：
  - 128 step 可能只覆盖远离物体的 Episode 早段；
  - 当前 salience selector 使用 successor outcome，不能作为 fresh P32 的无偏窗口；
  - 应使用完整 128 transition 或结果前固定的 outcome-blind 窗口。
- P46-E1：
  - 批准为可靠性侧车，但 mid-cycle pointer 必须与正式 deployment/latest 分离；
  - resume 必须跳过本周期重复 collection/materialization。

D 的执行建议：

1. 修订并实施 P40-E2；
2. P40-E2 接受后，再实施严格冻结的 P41-E2。

## 筛选 Agent E

| ID | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P40-E2` | 5 | 5 | 4 | 4 | 4 | 3 | 4 | `approve` |
| `R0001-P41-E2` | 5 | 4 | 3 | 4 | 3 | 2 | 2 | `changes_required` |
| `R0001-P47` | 4 | 3 | 3 | 3 | 3 | 2 | 2 | `defer` |
| `R0001-P48` | 4 | 5 | 5 | 5 | 4 | 3 | 4 | `approve` |
| `R0001-P44-E1` | 3 | 5 | 5 | 5 | 3 | 4 | 4 | `approve` |
| `R0001-P32-E2` | 3 | 3 | 4 | 3 | 2 | 5 | 3 | `changes_required` |
| `R0001-P49` | 3 | 4 | 4 | 3 | 3 | 3 | 4 | `changes_required` |
| `R0001-P46-E1` | 3 | 4 | 4 | 4 | 4 | 3 | 3 | `approve` |

### E 的关键反驳

- P40-E2：
  - 同意现有 `any(contact)` 和全对象总 distance progress 存在跨对象误计；
  - 明确要求将主字段命名为 contact-associated motion；
  - 同一实体、同一 control period、有效 robot–entity contact 是最小关联条件；
  - world–world pair 不得混入旧 P40 守恒式。
- P41-E2：
  - 主事件必须明确为左右臂在同一实体上的接触加同期实体运动；单臂只能描述；
  - paired power 必须按 discordant pair 建模；
  - condition 不进入 environment 或 policy seed；
  - latency 3 只作 challenge 分账，不能混入支持域主效应；
  - smoke 若导致设计修改，必须重新冻结并更换正式 seed。
- P47：
  - phase 必须只由 pulse 前 policy-visible 信息决定；
  - 未到 phase、FIFO/safety 不服从均必须按 ITT 保留。
- P48：
  - 需建立单一可信 qualification verifier；
  - qualification 应先于私有 salt，evaluation manifest 再绑定 salt commitment，避免循环。
- P44-E1：
  - 只覆盖前 14 维 command gain，不能外推完整执行器或 sim-to-real。
- P32-E2：
  - 当前 synthetic null 与所选线性 nuisance 同构，不能发现 nonlinear/controller
    misspecification；
  - 旧 24-source power 不能直接证明 36 个 128-step source 的功效。
- P49：
  - 必须改为 outcome-blind 固定窗口或完整 source；
  - source 数应由 P32-E2 功效先确定；
  - early termination 不得补 seed。
- P46-E1：
  - 批准为侧车；
  - CPU 要求 bit-identical，MPS 容差需结果前冻结。

E 的执行建议：

1. 先执行 P40-E2；
2. P40-E2 接受后，再执行修订后的 P41-E2。

## 主 Agent 汇总决策

### `R0001-P40-E2`：选择

- 两名筛选 Agent 都认可问题真实、测量价值高和前置依赖已满足。
- D 的 `changes_required` 与 E 的 `approve` 不构成方向分歧；D 要求的命名、分账和
  evaluator-private 边界全部纳入冻结实验。
- 本轮首先实施 P40-E2：
  - 只建立 arm/entity contact graph 与 contact-associated motion；
  - 不修改现有 runtime 行为；
  - 不把时间关联升级为动作因果或 controlled motion；
  - 不把 world–world contact 混入 P40 robot–environment 守恒。

### `R0001-P41-E2`：条件选择

- 三名创新 Agent 均首选，两名筛选 Agent 均认为它最贴近当前零交互瓶颈。
- 当前不能立即实现；必须先：
  1. P40-E2 接受；
  2. 冻结 candidate generator、canonical ordering、score、empty handling；
  3. 冻结逐 step primitive 与相同 index 的 bit-identity；
  4. 冻结同实体双臂接触加同期位移的主事件；
  5. 用 discordant-pair synthetic power 冻结 MDE、样本量和统计；
  6. 固定 ITT、supported/challenge、安全与动作非劣守护。
- P40-E2 失败或证据不足时，P41-E2 不启动。

### 其他提案

| ID | 本轮决策 | 原因 |
|---|---|---|
| `R0001-P47` | `deferred` | 依赖 P40-E2/P41-E2；避免按接触中介筛选 |
| `R0001-P48` | `approved, not selected` | 真实评测防火墙缺口，但当前无 qualified deployment |
| `R0001-P44-E1` | `approved, not selected` | 单因子合同成立，但不解决当前零交互 |
| `R0001-P32-E2` | `changes_required` | 需 confounded null、nuisance 质量守护与 fresh-cohort power |
| `R0001-P49` | `changes_required` | 需 outcome-blind 窗口且 source 数应先由功效决定 |
| `R0001-P46-E1` | `approved, not selected` | 有价值可靠性侧车，但不回答当前能力瓶颈 |

## 执行顺序与停止规则

1. 冻结并实施 P40-E2。
2. 若 P40-E2 的映射、fixture、守恒、fail-closed 或 bit-identity 任一失败，记录
   `rejected` 或 `inconclusive`，停止 P41-E2。
3. P40-E2 接受后，主 Agent必须在任何 P41 行为代码前追加完整 P41-E2 冻结合同。
4. P41-E2 只运行无训练 paired 物理诊断；本轮不启动 P47、P32-E2、P49 或正式训练。
5. 本轮不运行 capability benchmark，不发布任务成功率、泛化或安全能力结论。
