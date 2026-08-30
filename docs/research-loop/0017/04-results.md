# R0017 实验结果

## 结果总表

| ID | 判定 | 结论边界 |
|---|---|---|
| `R0001-P87-E1` | `inconclusive (invalid_design)` | 冻结 control 自相矛盾，独立红队禁止 formal evaluator；没有正式 oracle artifact 或受审合同 verdict |
| `R0001-P85-E1` | `deferred` | 本轮未实施、未运行 |
| `R0001-P88` | `deferred` | 本轮未实施、未运行 |
| `R0001-P68-E4` | `hard defer` | P85/P88 前置未满足 |
| `R0001-P76-E5` | `deferred` | exposure、cohort、no-B2 与 blind-root 合同未满足 |
| `R0001-P76-E6` | `hard defer` | P76-E5 未 accepted |
| `R0001-P86-E1` | `rejected in current form` | 范围过宽，缺 case/horizon/tolerance/resource 上限 |

本轮没有训练、参数更新、checkpoint、policy inference、MuJoCo physical acquisition、
B0/B1/B2 action、contact phase、capability evaluation 或新家务任务成功。

## `R0001-P87-E1`

### 冻结目标

P87 原计划以两个独立计算路径审计五个 experiment contract：

- verdict 可达性；
- denominator 守恒；
- causal-label 结构上限；
- accepted region 的最坏 stratum coverage；
- result-exposure policy；
- 20 类 mutation/control。

正式接受门要求：

- Solver A/B 对 5/5 contract 一致；
- reachable contract 有独立验证的 assignment witness；
- unreachable contract 经过 Solver B 穷尽且有真实 contradiction；
- 全部 controls 有判别力；
- input/provenance/resource/atomicity 门通过。

### 实施

唯一实施 Agent 在独立 worktree：

`/Users/louis/Developer/AIWorkspace/50-housework-robot-r0017-p87`

完成单一候选提交：

`485367fe1a4901f69407329af1e25bd7cdf5498b`

只新增冻结允许的四个文件：

- `src/hwr/eval/experiment_contract_oracle.py`
- `src/hwr/apps/evaluate_experiment_contracts.py`
- `tests/test_experiment_contract_oracle.py`
- `tests/test_experiment_contract_oracle_app.py`

主分支曾以单一实现提交：

`9f1cf0da0386c0c4aa20f9b6af1b2160b2c6e9eb`

合入同一实现，独立红队阻止正式运行后，由：

`9eed083e0902c7f5ed4691b05cd8b145ee39eab6`

完整回退。当前基线不包含 P87 实现；候选分支保留可追溯提交。

### 实现已通过的门

候选最终修订版：

- focused pytest：`23 passed`；
- Python size：4 个目标文件通过，file `<=800` lines、function `<=200` lines；
- repository-wide Python size：462 files 通过；
- architecture：通过；
- compileall：通过；
- `git diff --check`：通过；
- changed file 集合精确等于冻结四文件；
- 单一实现提交；
- `docs/research-loop/0001/`～`0016/` tree 保持不变；
- P50/P79 cohort join、Episode sample unit 与基础 denominator 分账通过；
- registry SHA-256、Git blob 与冻结文档 blob 可绑定；
- output/staging 预存在、直接 symlink、path escape、partial write 有 fail-closed
  路径；
- post-rename resource 或 parent-fsync 失败会删除 final output；
- 未读取真实 association/prefix outcome；
- 没有在实现中硬编码五个 formal contract 的最终 verdict。

### 完整 pytest

候选实现提交的完整测试收集：

- 1,184 个节点；
- `1,172 passed`；
- `1 failed`；
- `11 skipped`；
- 18 个既有 Torch JIT deprecation warnings；
- wall：`224.15s`；
- maximum resident set size：`1,992,130,560 bytes`。

唯一 failure：

`tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete`

该 failure 与 R0016 记录的既有 failure ID 相同，P87 新增 23 个测试全部通过，没有新增
failure。

一次并行 baseline pytest 因临时 worktree 的 `runs/` 软链接错误覆盖 tracked tree，
产生大量输入缺失伪失败；该结果不参与回归判定。修正临时 worktree 后没有再用其结果
替代 R0016 已记录的同代码基线。

## 独立红队

独立只读红队审计：

- blocker：2；
- major：3；
- minor：1；
- 结论：**不允许运行 formal evaluator**。

### Blocker 1：Solver B 未真正穷尽，contradiction verifier 接受伪矛盾

候选最终实现中，Solver B 遇到独立重建的 structural contradiction 后：

- 只检查 1 个空 assignment；
- 把剩余 assignment 全部记为 pruned；
- 将 `exhaustive=true`；
- 没有满足冻结合同的“不可达时穷尽后无 assignment”。

当前四个不可达合同全部走该路径，不能把它称为独立 assignment enumeration。

更严重的是，独立探针在一个真实 reachable 的合同上伪造：

```text
required=18, available=22
```

以及：

```text
task required=5, available=8
```

当前 `verify_contradiction()` 均返回 `passed=true`。它只重建同一字段，没有检查
`required > available` 这一矛盾关系是否成立。因此 `valid_contradiction_count`
不构成独立正确性证据。

### Blocker 2：冻结 C01/C02 本身不可按单变量执行

#### C01

冻结 control 要求：

> 只把 P68 selector target 从 `18→8`，应由不可达变为可达。

独立复算：

```json
{
  "base_reachable": false,
  "target_18_to_8_only_reachable": false,
  "independent_rejections": [
    {
      "scope": "task",
      "key": "clear_dining_table_3d/v1",
      "required": 5,
      "available": 1
    },
    {
      "scope": "task",
      "key": "tidy_living_room_3d/v1",
      "required": 5,
      "available": 1
    }
  ]
}
```

因此 C01 的预期翻转在冻结合同下数学不可达。实现为了让 control 通过，又删除了 task
floors 并把 claim scope 缩为 overall，同时改变多个变量。这不能算冻结 control。

#### C02

冻结 control 要求：

> 同一 pooled P76 合同仅增加 latency floor 后，应拒绝 `19/22` 且 `(2,2)=3/6`
> 的反例。

实现先从旧合同删除 `latency_pair` claim scope，再把它作为 control；随后在另一份
合同增加 floor。删除 claim scope 是未预注册变量变化，不能构成单变量对照。

由于 P87 接受门要求全部 controls 通过，而 C01/C02 无法按冻结语义成立，实验设计本身
已经无效。按协议不得后验修改 `03-experiment.md` 或 registry 以迁就实现。

### Major

1. `build_report()` 只检查现有 controls 全部 passed，不验证 C01～C20 的精确必需
   ID/类别集合；缺失 C18/C19 仍可得到 accepted report。
2. finalized wall/RSS 在 rename+fsync 后只返回到 stdout，不写入 hash-bound artifact；
   artifact 无法独立复核最终资源值。
3. symlink control 只检查末级路径；root 内祖先目录 symlink 可被接受，未形成逐级
   no-symlink provenance。

### Minor

两个核心文件分别为 800/798 行，多处压缩语句影响独立审计性和后续维护，应在重新设计
时拆分 sidecar，而不是继续压线。

## 判定

`R0001-P87-E1`：

`inconclusive (invalid_design)`

理由：

1. formal evaluator 从未启动；
2. formal output 与 staging 均不存在；
3. 冻结 C01 的预期行为数学不可达；
4. 实现为通过 C01/C02 改变了未预注册变量；
5. Solver B/contradiction verifier 不满足冻结的独立证据门；
6. 独立红队明确禁止运行；
7. 因此不能把假设判为 `accepted` 或科学上的 `rejected`。

该结论允许声明：

> R0017 发现并保留了 experiment-contract oracle 的必要性，但 P87-E1 的冻结 control
> 与独立求解合同无效；正式 evaluator 未运行，候选实现未进入基线。

不得声明：

- oracle 已验证任何 P68/P76 合同；
- P68 或 P76 物理假设被否定；
- candidate、selector、association、prefix、safety 或能力发生变化；
- P87 implementation 已通过独立审计。

## Artifact

冻结 formal 路径：

`runs/research-loop/0017/r0017-p87-contract-oracle-s20268701`

状态：

- 不存在；
- 无 `.staging`；
- 无 report、manifest、checkpoint 或 partial scientific artifact。

## 后续重新设计要求

若下一轮重提 P87，必须使用新实验后缀并重新创新/筛选/冻结：

1. C01 必须选择真实单变量且数学可翻转的 mutation：
   - 只改 overall threshold 时，同时保留 task floors就不能预期可达；
   - 可以改为只验证 `required_gt_eligible` contradiction 消失，不要求整个合同可达；
   - 或为 overall-only 单独建立受审合同，但不能在 control 中临时删 task floor。
2. C02 必须在同一 claim scope 上只增加 latency floor；旧合同“声明 latency 但没有
   floor”应直接判 `claim_without_minimum`，不能先删 claim。
3. Solver B 必须真正枚举或明确更名为独立约束 solver，并相应重新冻结；不得用一次空
   assignment 声称 exhaustive。
4. contradiction verifier 必须检查不等式方向与数值关系。
5. report 必须要求精确 control inventory。
6. finalized wall/RSS 必须进入可哈希的最终 receipt。
7. 明确并测试祖先 symlink 边界。
8. 拆分两个压线的 800 行模块，提高审计性。
