# R0012 独立筛选

## 过程与输入

- 冻结提案提交：`b397c10a63d623096281e6c88ed0a3ac63755cfd`
- 冻结提案：`docs/research-loop/0012/01-proposals.md`
- 输入 SHA-256：
  `b555c98be651077b759fec459c5c0f826c885bbea484e9ef69d088eeab99b450`
- 两名筛选 Agent 均先核验提交内文件 hash，再独立评分；完成前未查看另一人的输出。
- 两者均未修改文件、未运行正式实验、未启动训练。
- 评分维度：目标价值、证据强度、可检验性、因果可归因性、通用性、实施成本、回归风险；
  每项 1–5，实施成本与回归风险的 5 分表示低成本、低风险。

## 筛选 Agent 1

| ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P50-E4` | 3 | 4 | 5 | 5 | 2 | 5 | 4 | accept，条件排程 |
| `R0001-P60` | 5 | 4 | 4 | 4 | 4 | 4 | 5 | accept，优先停止门 |
| `R0001-P61` | 5 | 4 | 4 | 5 | 5 | 5 | 5 | accept，第一停止门 |
| `R0001-P62` | 5 | 4 | 2 | 3 | 3 | 4 | 4 | defer，设计未闭合 |
| `R0001-P63` | 2 | 2 | 2 | 4 | 3 | 3 | 3 | reject current design |
| `R0001-P64` | 4 | 3 | 3 | 4 | 3 | 3 | 4 | defer |
| `R0001-P65` | 4 | 3 | 4 | 4 | 3 | 4 | 3 | defer |

主要反驳：

1. P50-E4 的 alias 示例必须落到 canonical exact-claimed geom，不能绕过 exact conflict
   直接写 role；task-visible geom 清单必须结果前冻结。
2. P60 把多种性质不同的指标混成一个 readiness：
   - 至少一臂违反 strict outer reach 即足以判 pair hard-impossible；
   - 两臂均违反只能作严重度分账；
   - nominal B2 command support、heading、range 与 residual 不能冒充严格结构可达性。
3. P61 的 role 一对多并不自动构成缺口；必须排除合法外部 planner 调度或相同 interaction
   semantics 的反例。
4. P62 的 outer-envelope/FK 必要条件不能构成 positive control 的充分可行性证明；
   synthetic fail 可能只是 fixture 不可达，不能归因于 primitive。
5. P63 没有 recall/coverage 下界，全 unknown 可平凡满足 precision=1 与 zero false
   positive；同场景 alias oracle 同时用于设计和评测也会泄露。
6. P64 缺少 association 规则、mixed/unknown 上限和三路决策数量门。
7. P65 实际是 heading-triggered B0 exit，不是 readiness trigger；P60 也必须先证明 deficit
   落在可由 B0/B1 分配干预的量上。

## 筛选 Agent 2

| ID | 目标价值 | 证据强度 | 可检验性 | 因果归因 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P50-E4` | 3 | 5 | 5 | 5 | 2 | 5 | 4 | accept，限 mapping 合同 |
| `R0001-P60` | 5 | 4 | 5 | 4 | 4 | 4 | 5 | accept，优先诊断 |
| `R0001-P61` | 4 | 4 | 4 | 4 | 4 | 5 | 5 | accept，限静态审计 |
| `R0001-P62` | 5 | 4 | 3 | 2 | 3 | 5 | 4 | defer，充分性缺口 |
| `R0001-P63` | 2 | 2 | 4 | 3 | 3 | 3 | 4 | reject，本轮不执行 |
| `R0001-P64` | 3 | 3 | 4 | 4 | 3 | 3 | 4 | defer，严格条件式 |
| `R0001-P65` | 5 | 3 | 5 | 5 | 3 | 4 | 4 | defer，等待 P60 |

主要反驳：

1. P50-E4 是已证伪 body-exclusive mapping 的实质修复，但只覆盖三个当前场景；它不解决
   P57 的动作支持瓶颈，preflight 不能推出 entity coverage。
2. P60 的 outer reach 只能产生 `hard-impossible` 或 `not-disproven`，不能产生 reachable、
   IK-feasible、collision-free 或 task-capable。
3. 新 salt 只能称为新 seed 确认性复现；设计与门槛已受 P51/P57 启发，不能称完全未触碰
   的独立分布。
4. P61 必须形成机器可审计 transition 表，且区分 evaluator annotation、policy-visible
   state、planner call state 与 primitive input。
5. P62 必须有 independent joint-limit、dual-arm IK、collision、safety-budget witness，
   或独立参考 controller；否则无法建立 positive control。
6. P64 的 `no_relevant_final_candidate` 只能说明 final set 上游缺失，不能定位 generator
   的具体 gate；`selected relevant` 也不代表可操作。
7. P65 需增加 paired win/loss、最大臂距离 effect 与每 task 守护，不能只用绝对比例。

## 独立共识

两名筛选 Agent 独立达成：

1. `R0001-P60` 是当前最高价值物理诊断，应优先于完整 entity funnel；
2. `R0001-P61` 是低成本语义停止门，防止把 entity presence、阶段正确和 primitive
   compatibility 混为一个标签；
3. `R0001-P50-E4` 可接受为 evaluator mapping 基础合同，但不能宣称 coverage 或能力；
4. `R0001-P62` 方向有价值，但当前 positive-control 设计把必要条件偷换为充分条件，应
   defer；
5. `R0001-P63` 当前设计应拒绝；
6. `R0001-P64` 与 `R0001-P65` 依赖未满足，应 defer；
7. 本轮不恢复 selector、Replay、Actor、世界模型训练或完整 P50 anchor funnel。

## 主 Agent 裁决

主 Agent 不按总分机械选择，而按停止门价值、依赖、成本和结论隔离裁决：

| ID | 裁决 | 理由 |
|---|---|---|
| `R0001-P61` | **selected，静态第一停止门** | 先冻结当前 microinteraction 的合法 target/role/primitive 边界 |
| `R0001-P50-E4` | **selected，独立 mapping 合同** | 低成本修复已知不可表达性，为未来 evaluator 提供可执行基础 |
| `R0001-P60` | **selected，唯一物理诊断** | 新鲜 prefix-only cohort 区分 strict outer-reach 与 finite-horizon support deficit |
| `R0001-P62` | **deferred** | 缺少独立 joint/collision/safety feasibility witness，不能构成 positive control |
| `R0001-P63` | **rejected** | 全 unknown 平凡解、同场景拟合和显式 alias 的竞争性重复 |
| `R0001-P64` | **deferred** | P50-E4/P61/P62 前置未满足，且 decision gate 未冻结 |
| `R0001-P65` | **deferred** | P60 尚未证明 deficit 可由 heading/time allocation 干预 |

三项入选工作保持独立：

- P61 只审计 measurement semantics；
- P50-E4 只验证 evaluator geom-role mapping；
- P60 只测 entity-blind B2 phase-entry necessary geometry；
- 任一项的结果不得修改另一项冻结门或后验补 seed/alias/transition。

## 冻结前强制修订

### P61

1. 每个 transition 使用稳定 ID，冻结：precondition、allowed/forbidden role、interaction
   type、expected state change 与 evaluator predicate。
2. 明确 evaluator-private annotation、policy-visible state、planner call state 与 primitive
   input 四个信息边界。
3. “歧义”只在合法外部调度不能消解，或同 role 需要不同 primitive semantics 时成立。
4. 本轮只决定 P50 初始 acquisition 的 microinteraction target，不扩张为完整 planner。

### P50-E4

1. alias 记录为 `source visual geom → canonical exact-claimed geom → role/instance`。
2. 冻结完整 exact claim inventory、alias inventory、task-visible inventory、XML/binding
   identity 与 serialization。
3. 同 body 多角色合法；同 exact geom 多角色 invalid；不得 body propagation 或 priority。
4. site、background、empty interior、无名/未知 object type 固定为 unknown/excluded。
5. preflight 通过只接受 mapping contract；本轮不恢复 sidecar 或 Episode。

### P60

1. 将指标分成三层：
   - strict structural outer-reach certificate；
   - finite-horizon nominal B2 support margin；
   - heading/base/residual 描述量。
2. `hard_bilateral_impossible` 定义为同一 pair **至少一臂** strict outer-reach margin `<0`；
   两臂均负只作严重度。
3. outer-envelope 内只称 `not_disproven`，不称 reachable。
4. 冻结 frame、shoulder origin、link-length derivation、target formula、boundary equality 与
   floating tolerance。
5. 新 salt/seed derivation、36 pair、12 cell、每 task 12 pair 在结果前落盘。
6. 结论仅为 phase-entry necessary-geometry 与 nominal support deficit，不是 IK、无碰撞、
   entity relevance、接触或能力结论。

## 未入选项重新进入条件

- P62：结果前获得独立 dual-arm joint-limit/collision/safety feasibility witness，并冻结
  positive/negative controls 与数值门。
- P63：显式 alias 在预注册新场景出现真实维护失败，且 derivation/evaluation 场景隔离、
  recall 下界和 held-out camera/qpos 完整。
- P64：P50-E4、P61 接受；P62 有效 positive controls 未否定 action chain；冻结 patch、
  mixed/unknown 与三路 direction gate。
- P65：P60 证明 deficit 主要落在 B0/B1 可干预 heading/time allocation，并重新冻结 paired
  win/loss 与 off-by-one 语义。
