# R0006 结果

## `R0001-P39-E1`：`accepted as evaluation leakage fix evidence`

### 完整性

- 冻结文档提交：`81b4d9cabdf5cd2749f9b75505bf4d0bd4b49cd6`
- 实现提交：`23cbcac53f86122df08bce114f75c945bf8ee2d7`
- 冻结提交是实现提交祖先。
- 正式 run：
  `runs/research-loop/0006/r0006-p39-seed-isolation-e1-s20263901`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_seed_isolation \
  --output runs/research-loop/0006/r0006-p39-seed-isolation-e1-s20263901 \
  --salt R0001-P39-E1-s20263901
```

- device：CPU；
- 不更新模型参数，不执行 MuJoCo 能力 Episode，不创建 checkpoint；
- run 磁盘占用：32 KiB；
- report：
  - bytes：21,027；
  - SHA-256：
    `89dc8aaf5c7288fe03f71eb64d1d908b61a6bb3272f9d7b4486d15b7671cbda0`；
- manifest：
  - bytes：5,970；
  - SHA-256：
    `224a1549a6d5905813bac0dc3132c92179d47448d8c8f790d2b8ad860972e539`。

report 与 manifest 的 `source_commit` 均为 P39 实现提交，manifest 记录稳定命令、
commitment/reveal、逐 Episode seed lineage 和 report identity。

### Seed 合同结果

- 诊断 salt：`R0001-P39-E1-s20263901`
- commitment：
  `a94db502b86fd2c83a9096eb856b110de53158f588cc7496a60e4264fc190237`
- planned Episode：8；
- planned Episode identity：8 个唯一；
- environment seed：8 个唯一；
- policy RNG seed：8 个唯一；
- 同一 Episode 的 environment/policy seed 相等次数：0；
- raw environment seed 标准接口直通次数：0；
- baseline/candidate seed-pair coverage：100%；
- 相同 plan+salt 的 report、seed 和 action trace：bit-identical。

两个现有 evaluator 均使用兼容模式保留原 environment seed 序列，只把
`policy.reset(task_id, seed)` 的 seed 改为独立 policy RNG domain。正式 foundation
deployment/action-causality 准入顺序保持不变；测试明确验证 gate 拒绝发生在 salt 读取
之前。

### 能力与威胁边界

报告固定：

- `formal_seed_bank=false`
- `capability_claim_allowed=false`
- `threat_model=standard_policy_reset_interface_only`

明确排除：

- closed-loop capability improvement；
- malicious same-process policy isolation；
- observation information hiding。

因此 P39-E1 只接受为标准 policy reset 接口的评测泄露修复证据，不代表恶意进程隔离、
机器人能力、安全或泛化改善。未来正式 capability salt 必须在 deployment hash 固定后
生成，不能复用本公开诊断 salt。

### 验证

提交前：

- P39 focused、evaluator、aggregate、repository constraints 与 architecture：66 passed；
- 全量 pytest 通过：
  - 11 项既有 skip；
  - 18 条 warning 均为 `torch.jit.script` deprecation；
  - 无失败；
- training semantics 通过；
- physics integrity 通过；
- Python size：375 files 通过；
- architecture、py_compile、`git diff --check` 通过；
- 实现提交只修改冻结的 11 个 P39 文件；
- 历史 `docs/research-loop/0001/`～`0005/` 零差异。

正式 run 后：

- P39 focused/evaluator/aggregate/repository/architecture：51 passed；
- Python size 与 architecture 再次通过；
- 独立逐字段核验 source commit、8 个 Episode、双域 seed、commitment/reveal、
  pass-through count、pair coverage、bit-identical replay 和 artifact hashes，全部通过。

### 判定

P39-E1 全部冻结接受门通过，正式标记：

`accepted as evaluation leakage fix evidence`

P39 的实现提交已成为后续 P36-E2 的必需祖先。P36-E2 仍只能使用公开诊断 salt 建立
benchmark contract，不得生成或发布未来正式能力 seed bank。

## `R0001-P36-E2`：`accepted as balanced benchmark contract evidence`

### 完整性

- 冻结文档提交：`81b4d9cabdf5cd2749f9b75505bf4d0bd4b49cd6`
- P39 前置实现提交：`23cbcac53f86122df08bce114f75c945bf8ee2d7`
- P39 结果提交：`9fdadf19c4ffb756acadff5982713403ee1806c8`
- P36-E2 实现提交：`fadaf59b79d16ddebc5f0666183fdb21c33a2c2b`
- 上述祖先链满足冻结依赖。
- 正式 run：
  `runs/research-loop/0006/r0006-p36-factorial-e2-s20263602`
- 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_factorial_benchmark_contract \
  --output runs/research-loop/0006/r0006-p36-factorial-e2-s20263602 \
  --salt R0001-P36-E2-s20263602
```

- device：CPU；
- wall time：54.99 秒；
- max RSS：610,418,688 bytes，约 582 MiB；
- run 磁盘占用：772 KiB；
- 不执行 policy inference、action、完整 Episode 或视频；
- 不更新模型参数，不创建 checkpoint。

产物：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `report.json` | 139,305 | `7f0174bc1250627d2f0d76ed8f47698b5ce86693429fc86f63f173706d60c419` |
| `planned-ledger.json` | 554,073 | `70f44328058afd38d0a7c194e0655b442022a63026b51d220628ccc8e47fd26c` |
| `reset-only-smoke.json` | 82,088 | `0c0e45034120bbcac92ce4116f8d574f337e4a5f1021170fd030f6cf2289f6bf` |
| `manifest.json` | 1,386 | `e46a7d469dd9ae71db8adb3cce505d8db925ce5c70dec57409c67bfdf77c9729` |

report 与 manifest 的 source commit 均为 P36-E2 实现提交，manifest 记录稳定命令、
诊断 seed lineage 和三个内容产物的 hash/bytes。

### 合成功效结果

冻结设计完整执行：

- candidate `n`：`4/8/12/16/24/32`；
- baseline probability：`0.10/0.30/0.50/0.70`；
- paired shared-randomness fraction：`0.0/0.5/0.9`；
- null/planted 两个 condition；
- 共 72 个 strata；
- 每 stratum 500 Monte Carlo trial；
- 每 trial 1,000 hierarchical paired bootstrap replicate；
- training-seed slot 作为外层重采样单位；
- cell 内同步重采样 paired Episode；
- complete/support cell 等权。

选择结果：

- `n=4`：未通过全部冻结 strata；
- `n=8`：未通过全部冻结 strata；
- `n=12`：第一个通过全部冻结 strata；
- selected n：12；
- `n=12` 最坏 null FPR 双侧 95% Clopper-Pearson 上界：
  `0.023181388069658915`，通过 `<=0.05`；
- `n=12` 最坏 planted power 双侧 95% Clopper-Pearson 下界：
  `0.839503609048341`，通过 `>=0.80`。

没有使用真实 policy 结果选择 n，也没有查看结果后扩大 MDE、seed 或预算。

### Planned ledger

诊断计划包含：

- 3 个 future training-seed slot；
- 27 个
  `task × observation latency × action latency` cell；
- 每 slot-cell 12 个 replicate；
- pair：`3 × 27 × 12 = 972`；
- future execution plan slot：`972 × 2 = 1,944`。

这里的 1,944 只是未来 baseline/candidate execution 身份，不是本轮实际运行的 Episode。

完整性：

- 972 个 pair ID 全部唯一；
- 972 个 environment seed 全部唯一；
- 972 个 policy RNG seed 全部唯一；
- 两个 seed domain 没有交集；
- role 不进入 seed derivation；
- policy visible fields 为空；
- replacement seed 与 complete-case deletion 均禁止；
- `formal_capability_plan_usable=false`。

### 双账本与缺失合同

- 首要账本：27-cell 等权 `complete_challenge`；
- supported conditional：observation latency 1/2 的 18-cell 等权；
- latency 3 保持完整 observation-latency 权重三分之一；
- `full_profile_supported=false`；
- planned/valid/unresolved 恒等式冻结；
- missing terminal 一律进入 unresolved，实验只能 `inconclusive`；
- policy invalid/NaN/exception 在有可验证 provenance 时仍是 valid failure；
- host kill、power loss、artifact corruption 和无法归因异常为 unresolved；
- 不允许 replacement seed 或 complete-case 删除。

全部 fault injection 通过：

- missing exit 保持 unresolved；
- policy failure 保持 valid failure；
- duplicate、corruption、out-of-range cell、replacement seed 和未知分类全部拒绝；
- infrastructure unknown 保持 unresolved；
- planned identity 恒等式成立。

### 27-cell reset-only smoke

- 三个正式任务；
- 每任务 observation latency 1/2/3 × action latency 1/2/3；
- 共 27 次 reset；
- 不推理 policy、不 apply action、不运行完整 Episode。

全部检查通过：

- exact 27-cell coverage；
- sampled randomization 每任务固定；
- 去除两 latency 后的 other-randomization hash 每任务固定；
- instruction、physical state 和 camera calibration 每任务固定；
- effective latency 与 cell 完全一致；
- `verified_only_latency_pair_changed=true`。

联合 reset 先完整采样所有随机化，再只覆盖两项 latency，不改变 RNG 消费顺序。旧单项
diagnostic 语义和测试保持。联合 diagnostic 被强制为 reset-only，调用 `apply()` 会拒绝，
因此不能用它放宽 latency 3 safety。

### 禁止能力声明

报告固定：

- `formal_seed_bank=false`
- `capability_claim_allowed=false`
- `closed_loop_success_available=false`
- `primary_ledger=complete_challenge`
- `full_profile_supported=false`
- `policy_inference_executed=false`
- `complete_episode_executed=false`
- `action_applied=false`

因此本结果只接受为未来能力评测的合同与 runner-integrity 证据。公开诊断 salt 不可用作
未来正式能力 seed bank；正式 salt 必须在 deployment hash 冻结后生成。

### 验证

- 实施 Agent：
  - focused/formal backend/repository/architecture：44 passed；
  - Python size、architecture、physics integrity、pycompile、diff check 通过。
- 主 Agent：
  - P36 focused、P39 前置、formal backend、repository、architecture：61 passed；
  - 独立核验 Clopper-Pearson、72 strata、selected n、最坏 FPR/power、972/1,944
    planned counts、双域 seed、27-cell reset、fault injection 和 artifact identities；
  - 正式 run 后全量 pytest 通过：
    - 11 项既有 skip；
    - 18 条 warning 均为 `torch.jit.script` deprecation；
    - 无失败；
  - training semantics、physics integrity、Python size（379 files）、architecture、
    pycompile 与 diff check 全部通过；
  - 实现提交只修改冻结的 6 个 P36-E2 文件；
  - 历史 `docs/research-loop/0001/`～`0005/` 零差异。

### 判定

P36-E2 全部冻结接受门通过，正式标记：

`accepted as balanced benchmark contract evidence`

该结论不授权立即运行 1,944 个正式 Episode。未来还需要三个 hash-bound qualified
deployment pair、每任务安全/消融门和 deployment 冻结后的正式私有 seed salt。

## 其他候选当前状态

| ID | 状态 |
|---|---|
| `R0001-P36-E2` | 接受为 balanced benchmark contract evidence |
| `R0001-P32-E1` | 已批准，下一优先级，本轮暂不实施 |
| `R0001-P40` | `changes_required` |
| `R0001-P41` | `rejected` |
| `R0001-P42` | `deferred` |

本轮尚未启动正式训练，没有新任务成功，也没有能力改善。
