# R0009 独立筛选

## 筛选过程

- 两名有效筛选 Agent 对相同的冻结快照独立评分，完成前互不查看结果。
- 快照硬门：
  - Git HEAD：
    `0fea5f3fce43a9d00ab902138ff1aea63015f1d0`
  - `00-context.md` SHA-256：
    `db99aec53fe774303c5ef359a9e381fd9dc24b800633adb6ce93bf4273017202`
  - `01-proposals.md` SHA-256：
    `a7a50778ea295a240170649fd96117bd54584853ce80766741cd868f50ce0c95`
- 一次更早的筛选子线程返回了不存在于当前仓库的路径、提交和 `15/30` 成功基线，且其读取的
  两份 R0009 文档首行与实际文件不同。该输出因 snapshot mismatch 整体作废，不进入评分、
  汇总或选择。
- 替代筛选 Agent 与另一筛选 Agent 均先通过上述 HEAD、首行和 SHA-256 硬门。
- 七维均为 1～5 分；实施成本与回归风险的 5 分分别表示低成本、低风险。

## 筛选 Agent D

| ID | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P50` | 5 | 5 | 5 | 4 | 4 | 5 | 4 | `changes_required` |
| `R0001-P51` | 5 | 5 | 5 | 5 | 4 | 5 | 3 | `approve` |
| `R0001-P52` | 4 | 4 | 5 | 5 | 4 | 5 | 4 | `approve` |
| `R0001-P53a` | 4 | 2 | 4 | 2 | 4 | 3 | 3 | `defer` |
| `R0001-P53b` | 4 | 2 | 3 | 2 | 3 | 2 | 2 | `defer` |
| `R0001-P54` | 3 | 2 | 3 | 3 | 2 | 3 | 1 | `reject` |
| `R0001-P49-E1` | 3 | 4 | 4 | 4 | 4 | 3 | 4 | `defer` |
| `R0001-P47-E1` | 4 | 3 | 4 | 4 | 3 | 3 | 2 | `defer` |
| `R0001-P55` | 4 | 5 | 5 | 4 | 4 | 4 | 3 | `changes_required` |

### D 的关键反驳

- P50：
  - 必须对已捕获、不可变的 policy-input bytes 离线插桩，不能在控制循环中加入可能改变
    observation age 或动作时序的工作；
  - `(timestamp, sequence)` 才是视觉身份，depth hash 只能审计重复载荷；
  - first-rejection 是有序流水线描述，不是移除该 gate 的因果效应；
  - 若未来用于选择 P53，还需隔离 evaluator-private entity association；
  - 新 cohort 必须覆盖完整
    `task × observation latency {1,2} × action latency {1,2}`，不能只看对角 cell。
- P51：
  - `DualArmAction` 的 arm command 是 base-frame tool twist；
  - 当前 primitive 在 acquisition frame 计算误差后直接写入 action；
  - backend 又按当前 base rotation 转到 world frame；
  - 因而相对 yaw 非零时存在源码可证明的坐标合同错误；
  - physical smoke 应以 tool-target directional derivative 和距离收敛为主，接触只作描述。
- P52：
  - MuJoCo site 只能作为 evaluator 标签；
  - 所有误差必须在相同当前 base frame、相同物理时刻比较；
  - joint-grid 是确定性覆盖，不得冒充独立随机样本；
  - 阈值必须绑定 primitive 空间尺度，不能看结果后修改。
- P53a：
  - 当前代码已经将逐帧点按动态标定和 base pose 转入 acquisition frame；
  - 真正新变量是“在局部 hard gate 前融合 raw 3D evidence”，而不是首次加入位姿补偿；
  - voxel、动态点、surface grouping 和阈值是多变量算法包，必须等待 P50 缩窄。
- P53b：
  - 当前依据主要来自 hypothesis-generation 复算；
  - 模型层、预处理、深度关联、聚类和阈值尚未唯一化；
  - 冻结模型不等于没有纹理或背景捷径。
- P54：
  - 与本轮禁止 policy/candidate 使用语言语义的硬约束冲突；
  - 多 required-object 指令没有唯一 top-1 标签；
  - 本轮拒绝。
- P49-E1：
  - 不解决当前零交互；
  - exact-pipeline power 必须含 nonlinear confounded null、完整 nuisance OOF 和
    early-termination 机制；
  - 当前只适合作为后备 design-only 分析。
- P47-E1：
  - 候选命中、坐标和 FK 前置均未成立；
  - `+/-/sham` 的独立单位是 environment-seed block；
  - 不得按后续 contact 或运动筛选。
- P55：
  - 当前 success 没有把每个 required entity 与 arm contact、运动和 placement 绑定；
  - 但 P40-E2 的 contact-associated motion 仍允许任意机器人部位接触，不能直接作为
    arm-grasp 因果链；
  - 需要逐实体时序自动机和噪声下限后才可实施。

## 筛选 Agent E

| ID | 价值 | 证据 | 可检验 | 归因 | 通用 | 成本 | 风险 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P50` | 5 | 4 | 5 | 5 | 4 | 4 | 5 | `approve` |
| `R0001-P51` | 5 | 5 | 5 | 5 | 4 | 5 | 3 | `approve` |
| `R0001-P52` | 4 | 3 | 4 | 5 | 4 | 5 | 5 | `changes_required` |
| `R0001-P53a` | 4 | 2 | 3 | 3 | 4 | 3 | 3 | `changes_required` |
| `R0001-P53b` | 4 | 3 | 3 | 3 | 3 | 2 | 2 | `defer` |
| `R0001-P54` | 4 | 2 | 3 | 4 | 3 | 3 | 2 | `reject` |
| `R0001-P49-E1` | 3 | 4 | 3 | 4 | 5 | 3 | 4 | `defer` |
| `R0001-P47-E1` | 5 | 4 | 3 | 5 | 4 | 3 | 2 | `defer` |
| `R0001-P55` | 4 | 4 | 4 | 5 | 5 | 4 | 3 | `changes_required` |

### E 的关键反驳

- P50：
  - 重复 capture ordinal 会让 `view_count >=2` 更容易通过，不会直接导致空集合；
  - unique-identity shadow 只能诊断虚假多视角支持，不能预设它会增加非空率；
  - 同一 observation identity 出现不同 payload 必须 fail-closed；
  - 接受目标只能是测量合同，而不是证明某 gate 为根因。
- P51：
  - 精确旋转应为 `Rz(acquisition_yaw - current_yaw)`；
  - 只旋转 acquisition-frame 线速度，保持 z、角速度、norm、clip、phase 与 target 不变；
  - bit-identity 条件应是相对 yaw 为零，而不是绝对 yaw 为零；
  - 物理解释需要先通过修订后的 P52。
- P52：
  - 必须解决 observation latency 的时间对齐，不能把 stale joints 与 current site 比较；
  - 允许在 latency-free deterministic joint states 上比较，或按 timestamp/sequence 缓存
    同时刻 evaluator site；
  - `1cm..3cm` 灰区和替代 FK 仍大于 1cm 的情况必须预注册为 `inconclusive`。
- P53a：
  - 应等待 P50 后只冻结一个被证据支持的变化；
  - 不能整套重写 generator 后仍称为单变量。
- P53b：
  - 只有 P50 证明几何表面存在但 task-entity 身份被大结构淹没时才有优先级；
  - 需加入材质、布局、相机和 RGB permutation/ablation 守护。
- P54：
  - 与 R0009 输入白名单直接冲突；
  - 本轮拒绝。
- P49-E1：
  - 因子平衡只能降方差，不能创造被 controller history 消除的 action innovation；
  - 当前只允许无采集 design/power 检查。
- P47-E1：
  - pulse 方向、轴、幅值、时刻和随机化单位仍未冻结；
  - safety 对正负 pulse 的非对称裁剪必须进入 ITT 分账。
- P55：
  - 需要冻结 arm-pad-qualified contact 到实体运动的时间窗、位移噪声下限和合法单臂/
    双臂分工；
  - 只能作为 report-only 侧车。

## 主 Agent 决策

### `R0001-P51`：选择

两名筛选 Agent 均给出 `approve`，并且源码合同本身足以构成强证据：

1. candidate target 与 policy FK tool position 都在 acquisition frame；
2. 两者之差当前未经旋转直接写入 base-frame tool-twist action；
3. backend 再把该 action 从当前 base frame 转到 world frame；
4. 相对 yaw 非零时会产生确定性的方向错误。

本轮选择 P51，先接受解析和确定性合同证据。物理接触解释必须等待 P52；即使 P51 的
坐标修复成立，也不得称为交互、任务或能力改善。

### `R0001-P52`：条件选择

两名 Agent 均认可问题价值，一名 `approve`、一名 `changes_required`。主 Agent 采纳全部
修订后选择：

- 只在 latency-free、相同物理 state 上比较 policy FK 与 MuJoCo grasp-center site；
- site 始终 evaluator-private；
- 坐标统一到当前 base frame；
- deterministic joint-grid 只作为覆盖，不做随机显著性；
- 误差门在结果前冻结，灰区为 `inconclusive`；
- 不在同一提案中替换 FK。

P52 是 P51 physical smoke 的前置，不改变行为。

### `R0001-P50`：批准但不选择

P50 是后续 P53 的必要测量前置，但需要新建完整 12-cell acquisition cohort、离线插桩、
entity association 与独立确认 seed。当前存在更直接、低成本且源码可证明的 P51 错误，
本轮不并行扩大到 P50。

### 其他提案

| ID | 决策 | 理由 |
|---|---|---|
| `R0001-P53a` | `deferred` | 等 P50 缩窄为单 gate 或单聚合变量 |
| `R0001-P53b` | `deferred` | 多自由度且当前正式 entity-coverage 证据不足 |
| `R0001-P54` | `rejected` | 与本轮禁止语言语义进入 candidate/policy 的硬约束冲突 |
| `R0001-P49-E1` | `deferred` | 只降方差，不解决零交互；暂不采集 |
| `R0001-P47-E1` | `deferred` | 依赖候选命中、P51、P52 和可信共同前缀 |
| `R0001-P55` | `changes_required, not selected` | 需逐实体 arm-pad 时序链；只作未来评测侧车 |

## 执行顺序与停止规则

1. 冻结 P51/P52 的代码边界、公式、测试、命令、门槛与声明边界。
2. P51 与 P52 由不同实施 Agent、互斥写集合实施。
3. 先运行 P52：
   - 当前 policy FK aggregate p95 `<=0.01m` 且 max `<=0.02m`：P52 支持 FK agreement；
   - p95 `>0.03m`：P52 支持 material mismatch；
   - 其他情况：P52 `inconclusive`；
   - 任一 mapping、time alignment、finite 或 determinism guard 失败：P52 `invalid`。
4. P51 必须先通过解析合同：
   - 相对 yaw 为零时 action bit-identical；
   - 非零 yaw 下 candidate command 的 acquisition-frame 方向误差 `<=1e-12`；
   - norm、z、角速度、clip 与 hold 行为不变；
   - legacy 在预设非零 yaw 反例上必须失败，证明测试有判别力。
5. 只有 P52 为 FK agreement，才允许 P51 固定候选 MuJoCo smoke 的接触/距离结果进入
   描述；P52 mismatch 或 inconclusive 时，P51 只接受/拒绝解析坐标合同。
6. P51 不得与 candidate、FK、phase、幅值、gripper、安全或 backend 修改捆绑。
7. 本轮不启动世界模型/Actor 训练，不运行 P41 正式 selector 对照，不采集新 Replay。

