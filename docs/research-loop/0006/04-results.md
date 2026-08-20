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

## 其他候选当前状态

| ID | 状态 |
|---|---|
| `R0001-P36-E2` | 已冻结，等待 P39 结果提交后实施合同阶段 |
| `R0001-P32-E1` | 已批准，下一优先级，本轮暂不实施 |
| `R0001-P40` | `changes_required` |
| `R0001-P41` | `rejected` |
| `R0001-P42` | `deferred` |

本轮尚未启动正式训练，没有新任务成功，也没有能力改善。
