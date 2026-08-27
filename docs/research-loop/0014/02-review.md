# R0014 独立评审

## 评审过程

- 提案冻结提交：`5966e3f83d380384cf25d6b963f728b6da919299`。
- 两名筛选 Agent 在隔离线程读取同一冻结提交，完成前没有查看彼此输出。
- 两名 Agent 都只读检查源码、历史 artifact 和实验依赖，没有修改文件、训练或运行正式
  物理 cohort。
- 分数范围 1–5；实施成本与回归风险使用 `5 = 低成本/低风险`。
- 分数只辅助判断；最终选择同时考虑依赖、已知缺陷是否改变被测对象、结果前自由度和
  后续决策价值。

## Reviewer 1

| ID | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P73` | 2 | 4 | 5 | 4 | 3 | 3 | 3 | defer |
| `R0001-P74` | 5 | 4 | 4 | 3 | 2 | 4 | 3 | conditional accept |
| `R0001-P75` | 2 | 1 | 5 | 5 | 1 | 1 | 4 | reject |
| `R0001-P76` | 3 | 5 | 5 | 5 | 2 | 5 | 4 | defer |
| `R0001-P77` | 3 | 3 | 2 | 2 | 2 | 1 | 2 | reject |
| `R0001-P78` | 3 | 5 | 5 | 3 | 4 | 5 | 4 | defer |
| `R0001-P79` | 5 | 5 | 5 | 5 | 5 | 5 | 3 | accept，第一优先 |

### 关键意见

1. `P79`
   - `patch_valid` 是 frame-level `valid` 的 slice view，原地 `&=` 是确定的 ownership
     缺陷，不依赖统计或主观解释。
   - 探索性 22/24 selected identity drift 表明这不是无害实现细节。
   - 只修 generator ownership；不要顺便修改 P68 support reconstruction。
   - 新 generator/schema/bank 必须版本化，旧 artifact 保留且明确标为 legacy。
2. `P74`
   - 当前 P68 serializer、input failure、candidate generator 与 predictive safety
     没有 wrist RGB consumer，技术假设成立。
   - 但删除 wrist frame 会改变完整 `DualArmObservation.cameras`、calibration key set 与
     quality；不能宣称完整 observation 不变。
   - 只能在新 bank 建立后，收紧为 evaluator 专用未消费 render suppression，并用
     runtime access guard fail closed。
3. `P73`
   - eager source render 当前没有已知取证错误；snapshot ring 增加 eviction 和同步风险。
   - 已看到的约 `1.028×` 探索加速低于 standalone 价值门，延期。
4. `P75`
   - 历史 P50 本身已耗时约 45.11min；45min 预算没有可靠完成余量，且保留无关渲染，
     拒绝。
5. `P76`
   - 外包络必要条件科学问题明确，但旧 P51 candidate/target lineage 在 P79 后不代表
     新 pipeline；延期到新 bank。
6. `P77`
   - 搜索算法、objective、restart、seed、模型步数与预算未冻结，结果后自由度过大；
     P51 也没有可直接恢复 snapshot，拒绝当前设计。
7. `P78`
   - 缺陷真实，但 P72 已公开收缩 P61；当前没有后续实验消费 validated planner
     evidence，延期。

Reviewer 1 选择：

`P79 → 新 candidate bank → 条件 P74 → 另行冻结新-bank P68`

本轮最多建议 `P79 + 条件 P74`。

## Reviewer 2

| ID | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 建议 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P73` | 2 | 4 | 5 | 5 | 3 | 3 | 3 | reject |
| `R0001-P74` | 2 | 4 | 4 | 4 | 2 | 4 | 3 | defer |
| `R0001-P75` | 1 | 2 | 5 | 5 | 1 | 3 | 5 | reject |
| `R0001-P76` | 3 | 4 | 4 | 4 | 2 | 5 | 4 | defer，需收紧 |
| `R0001-P77` | 3 | 3 | 3 | 4 | 3 | 2 | 2 | defer |
| `R0001-P78` | 2 | 5 | 5 | 5 | 3 | 5 | 4 | defer |
| `R0001-P79` | 5 | 5 | 5 | 5 | 4 | 5 | 2 | accept，唯一入选 |

### 关键意见

1. candidate bank 仍可作为旧实现实际行为的历史证据，但不能继续充当未来 pipeline 的
   决策基线：
   - 22/24 selected identity 可能变化；
   - 3/24 empty/non-empty 可能翻转；
   - 继续旧-bank P68/P76 即使成功，也不能授权修复后 selector、routing 或训练。
2. `P79` 接受前必须：
   - 用独立 immutable-mask oracle 证明每个 probe 使用原始 frame validity；
   - parent validity mask 每个 probe 前后 byte-identical；
   - 三个预注册 traversal 的 raw-support multiset 和 final bytes 一致；
   - 若 `.copy()` 后仍有其他 traversal drift，不得现场扩展改动；
   - 不覆盖历史 artifact，建立版本化新 bank；
   - 明确阻止新 bank 被旧 P68 support reconstructor 当成 v1 消费。
3. `P73/P74/P75` 都服务于旧-bank P68；在 P79 前继续投入的决策价值不足。
4. `P76` 的严格性边界：
   - 只有 base pose 硬固定、肩根固定、最大链长为 compiled model 祖先路径全部刚性
     translation 模长之和时，球外包络才是严格不可达充分条件；
   - “零 base command”不等于 base pose 固定，自由基座仍可能受动力学移动；
   - joint range 不能在没有全局保守证明时用于收紧严格上界；
   - 因而旧 P76 不能直接关闭 production-dynamics P77。
5. `P78` 不是无意义清理，但 P72 已收缩结论，当前边际价值低。

Reviewer 2 只选择 `P79`。

## 主 Agent 决策

| ID | 决策 | 原因 |
|---|---|---|
| `R0001-P73` | rejected as standalone remedy | 两名 reviewer 均认为探索加速不足，且 snapshot ring 增加新的正确性风险 |
| `R0001-P74` | deferred | 技术上可行，但必须先有 P79 后的新 bank；本轮不把 generator 修复和执行优化捆绑 |
| `R0001-P75` | rejected | 45min 缺乏完成余量，只扩大低效旧-bank运行 |
| `R0001-P76` | deferred，需重设计 | 对旧 P57 有解释力，但应在新 bank 上重提，并严格区分 fixed-base geometry 与 free-base dynamics |
| `R0001-P77` | rejected in current form | 搜索自由度与预算未冻结，negative 不可解释，且依赖 P76 与可恢复状态 |
| `R0001-P78` | deferred | 静态完整性缺陷真实，但当前无直接 consumer，不解除主要瓶颈 |
| `R0001-P79` | **selected，唯一候选** | 确定的 generator 根因、单变量、低成本、高影响，且决定后续 bank 与所有 candidate-conditioned 诊断的有效性 |

本轮不按总分机械选择。虽然 P74、P76、P78 各有价值，但：

1. P79 会改变 candidate identity，是它们中与 candidate lineage 相关实验的上游依赖；
2. 同轮加入 P74 会把 generator correctness 与执行性能混成两个主变量；
3. P76 在旧 bank 上只能解释历史失败，不能指导修复后 pipeline；
4. P78 没有当前后续 consumer，且 exact-reference 与 planner evidence 最好拆成独立修复。

## 总体停止门

1. 本轮只实施 `R0001-P79-E1`，不实现 P73/P74/P76/P77/P78。
2. P79 若不能通过三种 traversal、parent-mask immutability 与独立 oracle，判为
   `rejected` 或 `invalid`，不得现场增加排序、merge、阈值或 selector 修复。
3. P79 accepted 后建立版本化 corrected candidate bank；不覆盖旧 P50 artifact。
4. 新 bank 只来自旧 P50 已冻结的 policy-visible capture bytes，不重新采样 Episode，
   不运行新的 physical cohort。
5. 新 bank 建立后停止本轮，不结果驱动地追加 P74、P68、P76 或训练。
