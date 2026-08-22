# R0010 独立筛选

## 输入与独立性

- 两名筛选 Agent 独立阅读同一版 `01-proposals.md`。
- 两者核验的提案文件 SHA-256 均为
  `b7ee266b5456b0da111fcfafa5b09848aa8d8157efeb8f5c6d242137599bb84a`。
- 两者完成前未查看对方输出，未修改文件，未运行正式实验。
- 评分维度为目标价值、证据强度、可检验性、因果可归因性、通用性、实施成本和回归
  风险，均按 1～5 分；实施成本和回归风险为越低分越高。
- 主 Agent 不按总分机械选择，而按当前瓶颈、依赖、评测有效性与资源预算裁决。

## 筛选 Agent 1

| ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 成本 | 风险 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P51-E1` | 5 | 5 | 4 | 4 | 4 | 2 | 3 | `changes_required` |
| `R0001-P50-E1` | 5 | 5 | 5 | 5 | 4 | 3 | 4 | `approve` |
| `R0001-P50-E2` | 5 | 5 | 4 | 4 | 4 | 4 | 4 | `changes_required` |
| `R0001-P50-E3` | 5 | 4 | 3 | 4 | 4 | 2 | 2 | `changes_required` |
| `R0001-P56` | 3 | 5 | 4 | 4 | 4 | 4 | 4 | `defer` |

主要反驳：

1. P51-E1 必须使用 36 pair，而不是 24 pair。`17/24` 在真实胜率 0.75 时功效只有
   约 0.766；36 pair 的 task 分层联合门在同一假设下约 0.891。
2. P51-E1 只执行 B2；若进入 B3～B6，会把 target 切换、gripper、接触和实体质量重新
   混入 Cartesian convergence。
3. candidate bank 必须先提交设计 commitment，再在任何 treatment 前提交完整 bank。
   B2 起点 identity 必须覆盖 action/observation latency queue，而不只是 qpos。
4. P50-E2 提案中的 component gate 顺序与源码不一致。源码是 connected component 后先
   `view_count>=2`，再 aggregate normal 非零；必须分别做 anchor、component、ranking
   三套守恒，不能混用分母。
5. P50-E3 的 observation-time truth 尚未唯一化；禁止在延迟 observation 被读取时用
   当前 MuJoCo state 近似打标签。
6. P56 有价值但不解决本轮 B2 收敛与候选覆盖主问题。

## 筛选 Agent 2

| ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 成本 | 风险 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P51-E1` | 5 | 4 | 3 | 4 | 3 | 2 | 3 | `changes_required` |
| `R0001-P50-E1` | 5 | 5 | 5 | 5 | 4 | 4 | 4 | `approve` |
| `R0001-P50-E2` | 5 | 4 | 4 | 5 | 4 | 4 | 3 | `approve` |
| `R0001-P50-E3` | 5 | 4 | 3 | 3 | 4 | 3 | 2 | `changes_required` |
| `R0001-P56` | 3 | 5 | 4 | 5 | 4 | 4 | 3 | `defer` |

主要反驳：

1. 同意选择 36 pair；但二项 win gate 不能为连续 AUC endpoint 提供功效证明。
   `delta_i = normalized_AUC_legacy_i - normalized_AUC_fixed_i` 必须成为真正主分析，
   结果前冻结连续置信下界与 MDE。
2. candidate eligibility 虽在 treatment 前确定，仍改变 estimand。结论只能覆盖自然
   支持 latency 下、candidate 非空并满足 yaw/exposure 的 Episode，不得外推全部支持域。
3. B2 普通 terminal 必须对两个 role 使用对称 carry-forward，不能一律记 fixed 失败；
   infrastructure corruption 与有效的 efficacy failure 也必须分账。
4. P50-E1 的 24 Episode 足以验收 immutable capsule 合同，不足以估计候选覆盖率或
   latency 主效应。
5. P50-E2 可与 E1 连续选择，但必须复用正式 generator gate，而不是复制近似实现。
6. P50-E3 若实施，private segmentation 必须在原始 observation 生成时、进入 latency
   queue 前同步封存；当前提案未冻结像素阈值与隔离合同。

## 统计复核

主 Agent 使用独立脚本复算：

| 设计 | 零假设尾概率 | `p(win)=0.75` 功效 | `p(win)=0.80` 功效 |
|---|---:|---:|---:|
| `17/24` | `0.03195732831954956` | `0.7662041693168753` | `0.9108287412264922` |
| `24/36` | `0.03262266761157662` | `0.9077928557951302` | `0.9817832445244096` |
| `24/36` 且每 task `>=6/12` | `0.030402215372305363` | `0.8913700748231963` | `0.9744835904798549` |

这些只描述 Bernoulli win 守护的设计性质，不替代连续 endpoint 推断。

## 主 Agent 决策

### `R0001-P51-E1`：条件选择，修订后实施

采纳两名筛选者共同要求：

- 36 个独立 pair，每个 12-way task×observation-latency×action-latency cell 3 个；
- 只执行 B2 的 100 个 control step；
- treatment 前两阶段 candidate-bank 提交；
- 主分析是 36 个 paired continuous `delta AUC`；
- 二项 `>=24/36`、每 task `>=6/12`、每 latency combination `>=4/9` 仅作稳健性守护；
- 普通 terminal 对称 carry-forward；基础设施、identity、泄露或测量损坏与 efficacy
  failure 分账；
- 无 sequential peek、无 24→36 扩样、无 replacement。

原因：这是 P51 进入代码基线后尚未闭合的最直接行为因果链，且 P52 已排除 FK mismatch。

### `R0001-P50-E1`：选择

24 Episode 只验收 immutable acquisition evidence contract，不用于能力或 coverage
改善结论。补充 per-cell seed 搜索上限、candidate-visible subpayload、capture
enabled/disabled identity 和 planned Episode 不替换规则。

### `R0001-P50-E2`：选择

作为 P50-E1 后的独立离线提交与判定。按实际源码顺序冻结三层守恒：

1. anchor：first rejection 与 raw candidate；
2. component：`view_count` rejection、aggregate-normal rejection、pre-top64；
3. ranking：retained 与 truncated。

“同 cell 两个 seed 均损失至少 60%”只允许描述重复阶段，不授权修改 gate。

### `R0001-P50-E3`：延期

三个 Agent 均认为候选身份重要，但当前没有冻结 observation-time private segmentation
的生成、像素对齐、visible/association threshold、mixed/unknown 与单向隔离合同。不得
用当前 state 给延迟 RGB-D 近似打标签。本轮不实现、不运行。

### `R0001-P56`：延期

接触归因问题真实，但 P51-E1 停在 B2 且不使用接触 endpoint，P56 不是当前主实验依赖。
未来任何 B3～B6 或 contact-yield 对照前必须重新筛选。

## 本轮顺序

1. 结果前冻结 P51-E1 与 P50-E1/E2 的完整合同和独立 salt commitments。
2. 可按互斥文件所有权并行实现；正式 MuJoCo 运行串行。
3. 先执行最低成本的 focused tests、合同测试和 P50-E2 合成 funnel fixture。
4. 执行 P50-E1 的 24-Episode acquisition-only cohort，再在封存 bytes 上执行 P50-E2。
5. 使用独立 salt 构建 P51 treatment-free candidate bank；提交完整 bank 后运行
   36-pair B2-only 正式对照。
6. 本轮不启动 P50-E3、P56、P41 selector 正式对照、P47、Replay 采集、Actor 或世界
   模型训练。
