# R0006 独立评审

## 评审过程

- 两名筛选 Agent 在相同冻结文本上独立只读评分：
  - `00-context.md` SHA-256：
    `d4aff9c75702825dfea6f9dfeb8753e2cb5b8bd37495dffee30f2cb5a1cd372a`
  - `01-proposals.md` SHA-256：
    `3814bab5dba019bb2d9b37402feab0a69a4982d2aa7bdbf436184aa07a220691`
- 两者完成前未查看对方评分，均未修改文件、未启动训练。
- 两者可只读核查源码、历史文档和产物，但不得改变提案。

## 评分合同

分数顺序统一为：

`目标价值 / 证据强度 / 可检验性 / 因果可归因性 / 通用性 / 实施成本 / 回归风险`

其中实施成本 5 表示低成本，回归风险 5 表示低风险。

强制拒绝条件：

- 没有明确指标或无法证伪；
- 与历史已拒绝或完成项实质重复而没有新证据；
- 依赖评测泄露、结果后选择或削弱安全；
- 首次实验同时改变多个无法独立归因的主变量；
- 只优化代理量且没有预注册闭环路径；
- 把评测修复与能力改进捆绑。

## S1 独立评分

| ID | 分数 | verdict |
|---|---|---|
| `R0001-P36-E2` | `5/5/5/4/5/3/4` | `select` |
| `R0001-P39` | `4/5/5/5/5/5/4` | `select` |
| `R0001-P32-E1` | `4/4/5/4/3/5/5` | `approve` |
| `R0001-P40` | `5/5/3/3/4/4/4` | `changes_required` |
| `R0001-P41` | `5/4/3/2/4/3/2` | `reject` |
| `R0001-P42` | `4/4/4/4/5/3/4` | `defer` |

### S1 关键反驳

- P36-E2：
  - 三个训练 seed 不能被大量 Episode 伪装成大量独立训练重复；
  - infrastructure missing 与 candidate timeout 的分类必须结果前冻结；
  - 只选择合同、合成功效、故障注入和 reset smoke，不执行正式 capability Episode。
- P39：
  - policy RNG seed 即使经过 hash 仍是 Episode 标识；
  - 结论必须限制为关闭标准 `policy.reset()` 接口的 raw environment-seed 泄露；
  - 不能宣称隔离恶意 policy 对 evaluator 内存或文件系统的读取。
- P32-E1：
  - 24 个 source 才是独立单位，不能按 2,688 transition 统计；
  - positive residual 仍可能编码 controller history、FIFO、actuator scale 或
    observation lag；
  - rate/configuration target 索引、ridge、bootstrap 和缺失处理必须进一步冻结。
- P40：
  - `floor` 也在 allowed geometry 中，正常承重可能制造 220N 假阳性；
  - impulse 必须按 physics substep 的 `Σ F_normal × Δt` 计算；
  - 220N 只能作当前内部一致性参照，不能冒充真实硬件阈值。
- P41：
  - 同时改变 RGB-D 条件化、目标选择、轨迹 primitive、速度、双臂耦合和撤回过程；
  - 触发“多主变量”强制拒绝；
  - P40 尚未建立允许接触的力—冲量测量合同。
- P42：
  - swapped target 可能因容器尺寸、抓取方式而物理不可完成；
  - 两个 mapping 必须使用同一 task ID，且 success predicates 必须互斥；
  - 当前没有 qualified deployment，只能延后。

S1 建议：

`P39 -> P36-E2 低成本合同阶段 -> qualified deployment -> 正式 benchmark -> P42`

P32-E1 可独立执行；若 P36-E2 reset smoke 失败，它是本轮首选替代项。

## S2 独立评分

| ID | 分数 | verdict |
|---|---|---|
| `R0001-P36-E2` | `5/5/4/4/5/2/3` | `approve` |
| `R0001-P39` | `5/5/5/5/5/4/4` | `select` |
| `R0001-P32-E1` | `4/4/5/4/3/5/5` | `select` |
| `R0001-P40` | `4/5/4/5/4/4/4` | `changes_required` |
| `R0001-P41` | `5/4/3/2/3/2/2` | `reject` |
| `R0001-P42` | `5/4/4/4/4/3/4` | `defer` |

### S2 关键反驳

- P36-E2：
  - 联合 override 不能改变随机数消费顺序或任何非 latency 字段；
  - `n`、逐任务不可接受回归和训练 seed 配对必须在冻结实验中给出具体值；
  - P39 前不得生成正式 seed bank。
- P39：
  - `policy_seed = H(environment_seed, salt)` 仍是环境 seed 的确定函数；
  - 更稳妥的是从 opaque planned Episode identity 用独立 domain 分别派生 environment 与
    policy seed；
  - deployment 可见 seed bank/salt 时仍触发泄露强制拒绝。
- P32-E1：
  - controller-context 消除信号时必须拒绝，不能保留有利的 rate-only 结论；
  - configuration target 无功效时只能 `inconclusive`；
  - 不能扫描 fold、ridge 或 target 来补救少量 source。
- P40：
  - 现有 contact monitor 只覆盖 pad-object，不是完整 allowed-contact 总账；
  - 需冻结多接触点、geometry pair 去重、substep 积分和 solver/timestep 稳定性。
- P41：
  - 同时改变至少五项行为，收益不能归因于 observation conditioning；
  - 通用近表面 primitive 还可能退化为 scripted trajectory；
  - 当前版本强制拒绝。
- P42：
  - 同一 scene、同一 task ID、同一初态与随机化是硬门；
  - variant 身份不得从路径、seed、target token 或视觉改动泄露；
  - 等待 P36-E2 与 qualified deployment。

S2 建议：

- 本轮实施 P39；
- 独立实施 P32-E1；
- P36-E2 只批准合同/runner-integrity，不运行正式物理能力基准。

## 主 Agent 代码复核

### P39 真实泄露面

- `src/hwr/eval/bimanual.py` 当前连续执行：
  - `environment.reset(seed=seed, ...)`
  - `policy.reset(seed=seed, ...)`
- `FoundationWorldModelPolicy.reset()` 用该 seed 初始化动作生成器。
- 环境 seed 同时决定 evaluation instruction、随机化、latency 和传感器噪声。
- 因此 raw environment seed 直通 policy 是真实接口泄露，不只是理论威胁。
- P39 可以不改变全局 `Policy` 方法签名：只需给 `seed` 参数新的、独立的 policy RNG
  语义，并在 report 中分别记录 environment/policy seed。

### P32-E1 可执行性

- 冻结 Replay 确为 24 个 source、168 个 shard、2,688 transition；
- 每 source 恰好 7 个 16-transition 连续窗口；
- task source 数 6/6/12，可严格构造每 outer fold 留出 2/2/4 source；
- action 连接相邻 observation 的语义由 collector、loader、RSSM 和既有测试共同支持；
- safety rewrite 只覆盖部分 source，必须完整分层，不能事后删除。

### P36-E2 联合 latency 边界

- 当前单独 observation override 只允许 0/1；
- action override 不允许与 observation override 同时存在；
- 新联合 override 只能用于 evaluator/reset contract；
- 它不得进入 policy、training 或 runtime 自适应，也不得让 latency 3 的 stale 动作通过。

### P40/P41 安全边界

- forbidden contact scan 在计算 force 前跳过 allowed geometry，P40 的盲区证据成立；
- 但 allowed 集含地面、容器、目标物和 articulation，不能用单一 220N 阈值合并解释；
- P41 主动追求接触，却没有先完成上述分账与 impulse 合同，且同时改变多个行为变量。

## 主 Agent 非机械决策

### 选择 `R0001-P39`

- 两名筛选 Agent 一致选择；
- 它是所有未来 P36-E2 seed bank、baseline/candidate 配对和能力结论的硬前置；
- 当前代码存在已确认的 raw environment-seed 直通；
- 成本低、不训练、不改变环境或 policy action value。

本轮接受范围只包括：

- 标准 policy reset 接口 seed 隔离；
- opaque planned Episode identity 的双 domain seed 派生；
- commitment/reveal、deterministic replay 和 lineage；
- existing evaluator 集成与测试。

不包括：

- 隔离恶意 policy 对 evaluator 内存/文件系统的读取；
- 能力或安全改善结论。

### 选择 `R0001-P36-E2` 的合同阶段

- 两名筛选 Agent 均批准，S1 直接选择；
- 它对应 `R0006-C01` 的主要瓶颈；
- P39 通过后可以无能力行为变化地建立：
  - 27-cell 平衡 planned manifest；
  - 双账本；
  - paired hierarchical statistics；
  - fail-closed 缺失合同；
  - 联合 latency-only reset provenance。
- 本轮不运行正式 capability Episode，不绕过 deployment/action-causality 准入。

### `R0001-P32-E1`：批准但本轮不实施

- 两名筛选 Agent 均认为可实施，且它是低成本、可证伪的下一候选；
- 但本轮实施名额优先用于所有未来能力对比共用的 P39/P36-E2；
- P32-E1 不被否决，冻结为下一优先级；
- 若 P36-E2 联合 override 无法保持非 latency 随机化不变，则本轮停止 P36-E2，不放宽
  合同，也不临时转入未冻结实现；P32-E1 留到新轮次重新确认后执行。

### 其他提案

| ID | 决策 | 原因 |
|---|---|---|
| `R0001-P40` | `changes_required` | 缺 floor/support 分类、substep impulse、pair 去重和阈值语义 |
| `R0001-P41` | `rejected` | 多主变量且安全测量前置缺失 |
| `R0001-P42` | `deferred` | 依赖 P36-E2 与 qualified deployment |

## 本轮依赖顺序

1. 冻结并实施 P39；
2. P39 测试与正式诊断通过后，冻结的 P36-E2 seed bank 才能生成；
3. 实施 P36-E2 合同、合成功效、故障注入和 reset-only smoke；
4. 不运行正式训练或能力 Episode；
5. 下一轮重新确认 P32-E1、修订 P40 和未解决问题。
