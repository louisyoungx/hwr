# R0017 冻结实验

## 实验身份

- 提案：`R0001-P87`
- 名称：frozen verdict reachability and denominator oracle
- 类型：实验合同静态验证
- 状态：实验设计冻结，尚未实施
- 起始提交：`26fb24ed2575bef1f268dc731323010607092fc8`
- 上下文提交：`8e1002ca0973003baad2427bc6f96884f52e69e9`
- 提案提交：`6f34c51`
- 筛选提交：`a34980864d81155d13bdd9113c66c6f0f5156548`
- 实施分支：独立 worktree `exp/R0001-P87-contract-oracle`
- 负责人：唯一实施 Agent；主 Agent只做审查、集成、门禁与归因

本轮只实施 `R0001-P87`。不得顺带实施 P85-E1、P88、P68-E4、P76-E5、P76-E6、
P86-E1、P77、默认 v2 migration、selector、runtime restore 或训练。

## 冻结问题

本实验只检验：

> 对冻结 P50/P79/P83 cohort 结构与预注册 experiment contract，两个计算路径能否一致
> 判定 verdict 可达性、denominator 守恒、causal-label 结构上限、accepted region 的
> 最坏 stratum coverage，以及 result-exposure policy 是否与 confirmatory 声明相容。

P87 不检验真实 association、prefix、安全、reachability、训练、泛化或能力。某个受审
合同被判 `rejected_contract` 不代表 oracle 失败；相反，可靠识别结构错误正是 P87 的
目标。

## 冻结输入

### Machine-readable registry

- 路径：`configs/eval/r0017_experiment_contracts.json`
- SHA-256：
  `c13ce840c4342772f46e77d87d0d856595f761bf02880266bad98ba6ffee6e8b`
- schema：`hwr.r0017-experiment-contract-registry/v1`
- sample unit：`Episode`

registry 在实现前创建并与本文件同时冻结。实现不得在源码中另写一套 formal threshold、
denominator、claim scope、exposure policy 或 expected verdict。

### Artifact identities

P50：

- root：
  `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
- `capsules.json`：
  `223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf`
- `plan.json`：
  `5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab`
- `report.json`：
  `b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0`
- `manifest.json`：
  `cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86`

P79：

- root：
  `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`
- `bank.json`：
  `888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e`
- `manifest.json`：
  `162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9`

P83：

- root：
  `runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`
- `report.json`：
  `0f09944aadb052d8b92f0b0c54cd40fb31f4835fdab1c44d2f24ef48fffd2513`
- `manifest.json`：
  `f149a586e30cb8d29156c685e084507b6a9adf490786993ffbaba589a04bd565`

### 冻结 cohort

- 24 个独立 Episode；
- 22 nonempty、2 empty；
- 14 singleton、8 choice-opportunity；
- task nonempty：living `6`、dining `8`、kitchen `8`；
- latency-pair nonempty：
  - `o1-a1=5`
  - `o1-a2=6`
  - `o2-a1=5`
  - `o2-a2=6`
- 12 个 task×observation×action cell，每格原始 cohort 为 2 Episode。

cohort 只能从 hash-bound P50/P79 文件独立 join。registry 中的 expected counts 只用于
交叉检查，不可替代 artifact 解析。

### Result exposure ledger

创新阶段已经看到三个 prefix sentinel 的 `safe_b2_entry`、wall、lineage、安全、
terminal、invalid-force 与 conservation 字段：

- living：
  `45b8ab118505cf7f834d59f9b6cad45a39ae7b25b076ff6157046d7ad620bdcd`
- dining：
  `7e039594072ca9a6aceda502e9cc69d2bcd697082906465b674bcb8b952674e8`
- kitchen：
  `c8a3a55e0f8f4f76c077d6b57366861f051be52e295329a1c3227adf1fc18d18`

oracle 不读取或编码这些 outcome value，只消费 registry 中的 Episode ID、field name、
source 与 policy：

- `include_exposed`：若 contract 为 confirmatory 且 outcome field 已见，合同
  `rejected_contract`；
- `exclude_matching_outcome_fields`：相应 Episode 从 confirmatory denominator 和
  strata 中排除，再重新计算门的可达性；不得仅减 threshold；
- `historical_design_audit`：只允许非 confirmatory 历史合同，结果用于回顾错误，不授予
  新正式运行资格；
- 缺失/未知 policy：`invalid`。

## Formal contracts

registry 冻结五个受审合同：

1. `r0016-p68-e3-selector-negative`
   - 历史设计审计；
   - denominator `nonempty=22`；
   - target eligibility `choice_opportunity=8`；
   - target minimum `18`；
   - task minimum living/dining/kitchen 均为 `5`。
2. `r0016-p76-e3-pooled-prefix`
   - 历史设计审计；
   - denominator `nonempty=22`；
   - total `19`；
   - task minimum `4/6/6`；
   - 声明含 latency pair，但没有 latency-pair floor。
3. `r0017-p68-e4-selected-positive`
   - confirmatory；
   - denominator `nonempty=22`；
   - total `18`，task minimum `5/5/5`；
   - claim scope 只允许 overall/task；
   - action/observation/latency-pair/cell 为 forbidden claim scope；
   - 若未来要声明 latency robustness，必须另建含 strata floor 的合同。
4. `r0017-p76-e5-include-exposed-draft`
   - confirmatory；
   - 含三个已见 safe-prefix Episode；
   - total `19`、task `4/6/6`、latency pair `4/5/4/5`、12 cell 各至少 1。
5. `r0017-p76-e5-exclude-exposed-draft`
   - confirmatory；
   - 三个已见 sentinel 从 confirmatory denominator/strata 排除；
   - 剩余 denominator `19`；
   - total `16`、task `4/6/6`、latency pair `2/5/4/5`、12 cell 各至少 1。
   - 该合同预期揭示：living `o1-a1` 唯一 nonempty Episode 已暴露，排除后 cell
     denominator 为 0，因此 12-cell confirmatory 门是否仍可达必须由 oracle判定。

这些合同是被审对象，不是 expected answer。实现和测试不得硬编码任一合同应为
`eligible` 或 `rejected_contract`。

## 双求解路径

### Solver A：解析式边界

从 cohort 与 contract 推导：

- denominator membership 与守恒；
- target eligibility 的结构上限；
- 每个 stratum 可用 denominator；
- total 与 strata minimum 必要条件；
- exposure exclusion 后的剩余 denominator；
- obvious impossibility witness，例如 `required > eligible` 或 stratum denominator
  小于 minimum。

Solver A 不枚举 outcome assignment。

### Solver B：assignment enumeration

- 对有效 target-eligible Episode 枚举二元 outcome assignment；
- 最多 22 个 Episode，即上限 `2^22=4,194,304`；必须使用剪枝或组合枚举使正式
  wall 满足资源门；
- 验证 total、task、latency、cell 与 exposure 约束；
- 若可达，返回 canonical accepted witness；
- 若不可达，返回穷尽计数与至少一个由 Solver A 给出的 contradiction witness。

### Agreement

- `reachable=true`：两者均判断所有必要条件可满足，且 Solver B 给出可独立验证的
  accepted assignment。
- `reachable=false`：Solver A 给出至少一个必要条件 contradiction，Solver B
  穷尽后无 assignment。
- 两者分歧：P87 `invalid_solver_disagreement`，不得发布 accepted oracle 结论。
- witness verifier 必须是与 Solver B 搜索循环分离的函数，并从 registry/cohort
  重算全部门。

## 输出

正式 output：

`runs/research-loop/0017/r0017-p87-contract-oracle-s20268701`

至少包含：

- `cohort.json`：从 P50/P79 join 后的最小结构记录，不含真实结果；
- `contracts.json`：registry 规范副本及 identity；
- `analysis.json`：每合同双求解结果、witness/contradiction、denominator 与最坏 strata；
- `controls.json`：全部 mutation/control 结果；
- `report.json`：P87 判定与 claim flags；
- `manifest.json`：source、input、command、runtime、预算与所有 artifact hash。

JSON 使用 sorted-key canonical bytes；先写 sibling staging，全部门通过后 atomic rename。
正式 output 或 staging 已存在时 fail closed。失败不得发布 partial scientific artifact；
可由测试或外部 supervisor 保存非科学失败日志。

## 预注册 controls

至少覆盖：

1. selector target `18→8` 后从结构不可达变为存在 assignment；
2. pooled P76 构造 `19/22` 且 `(2,2)=3/6` 的 witness：
   - 旧 pooled 合同可接受；
   - 加入 frozen latency floor 后必须拒绝；
3. 删除 empty/nonempty 分账；
4. denominator 名称互换；
5. choice-opportunity 冒充 nonempty；
6. candidate/frame/pixel/arm/control-step 冒充样本单位；
7. task、latency、cell key 缺失或未知；
8. duplicate/missing Episode；
9. P50/P79 Episode identity 不一致；
10. candidate count、task、latency 变异；
11. expected cohort count 自证但 artifact 解析值不同；
12. confirmatory contract 使用 `include_exposed`；
13. exposure Episode 或 fields 缺失、重复、未知；
14. exposure exclusion 只减 total threshold、不减 denominator/strata；
15. claim scope 含某 stratum但没有相应 minimum，也没有显式 forbidden scope；
16. reachable witness 不满足 threshold 或 strata；
17. unreachable 合同伪造 accepted witness；
18. registry schema/hash、冻结设计 blob、source scope、历史 tree 或输入 hash 漂移；
19. output/staging 预存在、symlink/path escape 或 partial write；
20. 两个 solver 人工注入分歧。

语义 controls 必须命中对应 category，不能只依靠顶层 hash 使全部 mutation 在 provenance
阶段失败。

## 主要指标

- `contract_count`
- `reachable_contract_count`
- `rejected_contract_count`
- `solver_agreement_count / contract_count`
- `valid_accepted_witness_count / reachable_contract_count`
- `valid_contradiction_count / rejected_contract_count`
- `denominator_conservation_count / contract_count`
- `exposure_policy_valid_count / contract_count`
- `mutation_control_pass_count / mutation_control_count`
- `private_outcome_read_count`
- `sample_unit_violation_count`

描述性报告：

- 每合同 effective denominator；
- target eligibility 上限；
- 每个 stratum denominator 与 minimum；
- accepted region 最坏 stratum coverage；
- canonical witness 或 contradiction；
- wall、process-tree RSS、artifact bytes。

## 接受、拒绝与无效门

### `accepted as frozen experiment-contract oracle`

必须同时满足：

1. P50/P79/P83 所有冻结文件 hash 匹配；
2. 24 Episode identity join 唯一且 cohort 分账匹配；
3. registry hash与本冻结文档 blob匹配；
4. 5/5 contract 完成双求解；
5. solver agreement `5/5`；
6. reachable contract 的 witness 全部通过独立 verifier；
7. rejected contract 均有有效 contradiction 且 enumeration 无 accepted assignment；
8. denominator conservation `5/5`；
9. exposure policy valid `5/5`；
10. private outcome read count `0`；
11. sample-unit violation count `0`；
12. 全部预注册 mutation/control 通过；
13. formal wall、RSS、artifact 在预算内；
14. source/provenance、allowed file scope 与历史 tree 全部通过；
15. 两次完整 analysis bit-identical。

### `rejected`

输入、registry 与 provenance 有效，但两条独立求解路径对至少一个合同稳定分歧，或
witness verifier 揭示 solver 的逻辑错误。

### `invalid`

任一输入/hash/schema/source/tree 漂移、读取真实 outcome、样本单位错误、同源阈值、
mutation 无判别力、partial output、预算越界或后验改 registry/门槛。

## 资源预算

- CPU-only；
- formal wall `<30s`；
- process-tree peak RSS `<512MiB`；
- artifact `<8MiB`；
- 数据卷剩余空间 `>=20GiB`；
- 无 MuJoCo、Torch、GPU、tmux、后台任务、休眠或 host-exec；
- focused tests、repository gates、full pytest 与 formal evaluator 分开计量。

## 实施范围

允许且只允许：

- `configs/eval/r0017_experiment_contracts.json`
- `src/hwr/eval/experiment_contract_oracle.py`
- `src/hwr/apps/evaluate_experiment_contracts.py`
- `tests/test_experiment_contract_oracle.py`
- `tests/test_experiment_contract_oracle_app.py`

实施要求：

1. `eval` 模块只做 schema、cohort、solver 与 witness 验证；
2. `app` 负责 artifact/provenance、资源、atomic output 与命令；
3. 不修改 P50/P79/P83、legacy P68/P60、candidate、runtime、安全或训练代码；
4. 每项行为变化有测试；
5. 唯一实施 Agent 在独立 worktree完成一个原子提交，提交信息引用 `R0001-P87`；
6. 正式运行前必须合入主分支、工作区干净、source committed、focused tests 与项目门
   通过；
7. 历史 `docs/research-loop/0001/`～`0016/` tree保持不变。

## 验证顺序

1. JSON/schema 与静态 source 审查；
2. focused unit tests；
3. mutation/control；
4. Python size、architecture、compileall、`git diff --check`；
5. 从冻结提交对比 allowed file scope；
6. full pytest failure-set 回归；
7. 独立只读红队审计；
8. 只有 0 blocker、0 major 时运行 formal evaluator；
9. 两次独立分析 bit identity 与 artifact/hash 复核；
10. 写入 `04-results.md`，按预注册门归因。
