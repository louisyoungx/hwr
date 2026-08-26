# R0013 独立筛选

## 过程与独立性

- 评分对象固定为提交
  `e21c01dea625ca3a4f032a4b9978f9f1849c21ef` 中的
  `00-context.md` 与 `01-proposals.md`。
- 两名有效筛选 Agent 均先阅读 `AGENTS.md`，独立只读核验；完成前未查看对方输出，
  未修改代码、未运行正式物理实验或训练。
- 第一名筛选 Agent 的首次回复讨论了不属于本仓库提案集的 flow/draining 内容，属于明显
  上下文错配，未进入任何评分或决策。主 Agent 在同一线程要求从磁盘重读固定提交，
  纠偏后的完整评分才记为 Reviewer 1。
- 原第二名筛选 Agent 在三次长等待与一次收束指令后仍未返回，记为执行失败并关闭；
  替补 Reviewer 2 在隔离线程对同一冻结提交独立评分。
- 实施成本与回归风险均使用 `5 = 低成本/低风险`。

## Reviewer 1

| ID | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P66` | 5 | 5 | 5 | 5 | 4 | 4 | 4 | accept，第一优先 |
| `R0001-P67` | 4 | 3 | 4 | 3 | 3 | 3 | 4 | defer |
| `R0001-P68` | 5 | 4 | 4 | 4 | 3 | 4 | 4 | accept，条件停止门 |
| `R0001-P69` | 4 | 4 | 5 | 4 | 2 | 5 | 5 | defer |
| `R0001-P70` | 4 | 3 | 4 | 4 | 3 | 3 | 4 | defer |
| `R0001-P71` | 5 | 4 | 3 | 4 | 4 | 3 | 4 | defer，高优先级后续 |
| `R0001-P72` | 4 | 5 | 5 | 5 | 4 | 5 | 5 | accept，静态完整性门 |

### 关键意见

1. `P66`
   - 直接命中本轮 hard-stop 的证据缺口，是 P67/P69/P70 的共同前置。
   - observer 若重写 contact predicate，会与 production path 同源自证；必须保留原
     decision path，只旁路采集其实际检查输入。
   - production 当前没有显式 nonfinite fail-closed；若 fixture 发现问题，只能报告并
     停止，不得在 P66 中顺手改变 safety 行为。
   - P60 anchor 是一个确定性回放，不是发生率样本。
2. `P67`
   - 只测 recurrence，不解释根因；历史 P51 零 intervention 使新 cohort 的先验较弱。
   - candidate-empty 必须同时进入全 Episode 分母，并从 B0/B1 at-risk 分母分开报告。
   - 暂缓，等待 P66 与 P68。
3. `P68`
   - 是继续 B1 路线前的必要 stopping gate。
   - candidate empty 必须固定归类并保留在 24 Episode 分母。
   - association 必须由原 RGB-D identity、source pixel/point provenance 与冻结 mapping
     建立；不得仅按 candidate world coordinate 或名称回填。
   - P72 若发现 role/initial annotation 依赖无效，应停止 P68。
4. `P69`
   - 对单次拒绝有局部因果价值，但四个 branch 不是四个独立样本，不能外推 controller。
   - 暂缓，P66 后才允许作为事件级 sidecar。
5. `P70`
   - 必须区分“candidate identity matters”和“baseline candidate 更差”两个 estimand。
   - alternate candidate 更安全不代表更正确；暂缓。
6. `P71`
   - 战略价值高，但 solver negative 不完备；低于门槛只能称未证明。
   - 若 P68 停止 B1 路线，下一轮应优先考虑 P71。
7. `P72`
   - 已看到具体风险：`frozen_reference` exact-match flags 未明确进入核心 checks，
     `planner_call_state_available` 与 role field 同源。
   - mutation 必须重新执行完整 auditor/verdict，不能直接翻转输出布尔值。
   - 关键 dependency 不敏感时，应收缩 P61 claim，而不是自动否定全部 gap。

Reviewer 1 选择：`P66 + P72 + 条件式 P68`。

## Reviewer 2

| ID | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P66` | 5 | 5 | 5 | 4 | 4 | 5 | 4 | accept |
| `R0001-P67` | 3 | 2 | 4 | 2 | 2 | 4 | 4 | defer |
| `R0001-P68` | 5 | 3 | 4 | 3 | 2 | 4 | 3 | accept |
| `R0001-P69` | 4 | 4 | 5 | 4 | 2 | 5 | 5 | defer |
| `R0001-P70` | 3 | 3 | 4 | 4 | 2 | 3 | 4 | defer |
| `R0001-P71` | 4 | 3 | 4 | 4 | 2 | 3 | 5 | accept |
| `R0001-P72` | 2 | 3 | 5 | 4 | 2 | 5 | 5 | defer |

### 关键意见

1. `P66`
   - 当前目标、证据与可检验性最强；只接受为 simulator predictive witness。
   - observer 必须从 raw contact point force 构造可审计记录，不能复述 production boolean。
2. `P67`
   - `3/24` 门缺少明确基线率或效应量依据，且不能支持通用 B1 缺陷；暂缓。
3. `P68`
   - 能防止“更精确地解释驶向错误对象”的自证，接受为三冻结场景 stopping gate。
   - sidecar truth 若影响 alias、candidate、index 或行为，立即 invalid。
4. `P69`
   - 对 anchor 局部机制有价值，但不覆盖此前路径积累；暂缓到 P66 后。
5. `P70`
   - paired 设计可检验 candidate-conditioning，但不能证明 alternate 更正确或能力改善；
     暂缓。
6. `P71`
   - 能补 P62 的 endpoint feasibility 前置；只允许 positive existence witness，
     optimizer failure 不得称 infeasible。
7. `P72`
   - 有实际静态审计价值但不直接解除 safety hard-stop，建议 defer。

Reviewer 2 选择：`P66 + P68 + P71`。

## 主 Agent 决策

| ID | 决策 | 原因 |
|---|---|---|
| `R0001-P66` | **selected，第一停止门** | 两名 reviewer 一致；关闭当前 safety witness 与 action-lineage 缺口 |
| `R0001-P72` | **selected，静态依赖门** | Reviewer 1 发现具体 P61 dependency 风险；成本极低，并决定 P68 的 annotation 是否可信 |
| `R0001-P68` | **selected，条件运行** | 两名 reviewer 一致；只有 P72 的 role/initial-annotation 相关门通过后运行 |
| `R0001-P67` | deferred | recurrence 先验弱，且 P66/P68 未成立前信息价值不足 |
| `R0001-P69` | deferred | 只解释单事件即时分量；P66 后才有可信 estimand |
| `R0001-P70` | deferred | 依赖 P66、P68，且 direction estimand 尚需拆分 |
| `R0001-P71` | deferred，高优先级下一路线 | 有战略价值，但不解除当前 safety/P68 依赖链；P68 停止 B1 时优先重提 |

选择不是按总分机械决定：

1. P66 是不可替代的生产判据证据门。
2. P72 的总分低于 P71，但它直接审计 P68 所依赖的 P61 initial-role 证据；不先验证就
   运行 P68 会把潜在自证继续传递。
3. P68 只在 P72 相关 dependency 通过后运行；若失败，本轮不以结果驱动方式临时补入
   P71/P67/P69/P70。
4. 本轮不修改 B1 controller，不启动训练。

## 总体停止门

- P66 anchor 不复现：P66 `inconclusive`，P67/P69/P70 保持 deferred。
- P66 witness 与生产判据不一致、observer 改变 authoritative trace/state 或危险动作进入
  authoritative physics：P66 `invalid`，停止所有依赖项。
- P72 mutation harness 无法证明 mutation 实际到达 auditor：P72 `invalid`。
- P72 发现 initial microinteraction uniqueness 或 role annotation 不进入可信 verdict：
  停止 P68；收缩 P61 对应 claim。
- P68 candidate provenance 不可绑定、sidecar 改变行为或需要结果后 alias：P68 `invalid`。
- P68 `stage_compatible_selected <=6/24`：停止 B1 行为修订。
- P68 为中间区间或 mixed/unknown 超限：保持 `inconclusive`，不解锁 P70。
- 任一已选项提前停止时，本轮不补选第四项。
