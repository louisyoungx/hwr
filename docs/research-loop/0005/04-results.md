# R0005 结果

## `R0001-P36-E1`：`accepted as evaluation contract evidence`

### 完整性

- 冻结文档提交：`376c172`
- 冻结后共享规则提交：`52433e8`
  - 只在 `AGENTS.md` 新增“给子 Agent 充足排队和任务处理时间”；
  - 未改变 P36 代码、输入、门槛或评测合同；
  - 冻结提交仍是实现提交祖先。
- 实现提交：`998abb9ca63439ce0b09045c91f30f7d72fcca07`
- run：
  `runs/research-loop/0005/r0005-p36-support-domain-e1-s20263601`
- 命令：
  `.venv/bin/python -m hwr.apps.evaluate_support_domain --p11-run runs/research-loop/0003/r0003-p11-causal-plant-s20261101 --p29-run runs/research-loop/0004/r0004-p29-stale-validity-s20262901 --output runs/research-loop/0005/r0005-p36-support-domain-e1-s20263601`
- wall time：约 0.32 秒；
- device：CPU；
- 新增 run 磁盘占用：88 KiB；
- report：
  - bytes：84,698；
  - SHA-256：
    `b9648d9c82b081433bd7040b06c537526345a1bfa6ba5bb3813e1278c7d7db6d`；
- manifest：
  - bytes：1,447；
  - SHA-256：
    `74823721bb6a1aca08b9b1b93eb4ee4fd53f6cfa3e5e42ed41cd2ef17e1e8ceb`。

正式 run 的 manifest 记录源码提交、完整命令、四个输入身份和 report 身份；产物目录只含
`report.json` 与 `manifest.json`，没有临时文件、checkpoint、训练日志或后台任务。

### 输入与 lineage

四个冻结输入的 hash/bytes 全部一致：

| 输入 | SHA-256 | bytes |
|---|---|---:|
| P11 report | `79195b94f48e8b59ce04bb7a9a3a680f8717f6484da69b2f28d54b3a9b842cda` | 423,970 |
| P11 manifest | `509f1b7f11caa2aa0dced3324bcc6ea9af9b045071c85b1f6a2bc66a7160da3a` | 28,249 |
| P29 report | `7729eed53e034bef5a9d5c50bd1d87d6025370b290226f3055c480f3014274fb` | 71,794 |
| P29 manifest | `41495b1a1efa9f24b65e8c5359bc98cfa28ecee4dbdbc20a5c20d37aa7900cef` | 317 |

- P11 source commit、proposal、`contract_complete=true` 和 `decision=rejected` 一致；
- P29 source commit、proposal 与 manifest 一致；
- P29 report 内嵌的 P11 report/manifest hash 与冻结值一致；
- P11 manifest 145 个 artifact 和 P29 manifest 1 个 artifact 的 hash/bytes 全部验证；
- 144 个 P11 Episode 的 report-to-manifest artifact lineage 全部一致；
- P29 assessment：
  - `passed=true`；
  - `decision=contract_incompatible`；
  - 13 项 checks 全部为 true；
- P29 guardrails 仍显示：
  - 未延长 action validity；
  - 未删除 evaluation latency；
  - 未替换 latest bundle；
  - 未改变 P11 decision。

### 时效分类合同

| observation latency | 最大 source age | domain |
|---:|---:|---|
| 0 | 0ms | `supported` |
| 1 | 50ms | `supported` |
| 2 | 100ms | `supported` |
| 3 | 150ms | `challenge` |

- 20Hz、50ms 控制周期和 100ms validity 与 P29 一致；
- `age == 100ms` 仍为 supported，`age >100ms` 为 challenge；
- action latency 不改变 observation-age domain；
- latency3 Episode 没有因 warm-up 前几步较新而拆分或重标。

### 双账本结果

| ledger | Episode | safety intervention | severe collision | early termination |
|---|---:|---:|---:|---:|
| `complete_challenge` | 144 | 2,196 | 0 | 0 |
| `supported_conditional` | 108 | 0 | 0 | 0 |
| `challenge` | 36 | 2,196 | 0 | 0 |

- `complete_challenge` 是首要总账；
- 每个 Episode 恰好进入完整总账和一个 domain，144 个 identity 全部唯一；
- supported/challenge 每任务为 36/12；
- supported/challenge 在 action latency 1/2/3 下分别均为 36/12；
- 完整 27 个 task × observation latency × action latency cell 均报告；
- 每任务、每 action latency 下 observation latency 1/2/3 的 cell count 为 2/10/4；
- P11 observation-latency 分布为 18/90/36，不平衡，因此报告明确：
  `balanced_factorial_benchmark=false`。

### 能力边界

报告固定：

- `capability_claim_allowed=false`
- `closed_loop_success_available=false`
- `balanced_factorial_benchmark=false`
- `primary_ledger=complete_challenge`

限定声明为：

> 支持 visible observation age <=100ms 的 evaluation 子域；完整 evaluation profile
> 尚未支持。

P36-E1 使用的是 P11/P29 诊断证据，没有任务成功标签，不报告或推断闭环成功率。它接受的
只有评测合同证据：

- latency3 没有从完整分母中删除；
- 条件支持域与完整挑战域可以同时、唯一、可重建地报告；
- safety rejection 与 policy failure 可以在未来能力评测中明确区分。

它不代表：

- 世界模型、Actor、任务成功率或泛化改善；
- latency3 已成为安全可执行域；
- P01 v4 已成为 causality-qualified deployment；
- supported conditional metric 可以与旧完整域数值直接比较。

### 验证

实现 Agent 验证：

- `tests/test_support_domain.py`、repository constraints、architecture：22 passed；
- Python size：371 files passed；
- py_compile 与 `git diff --check` 通过。

主 Agent 独立验证：

- P36/P29/闭环 evaluator：28 passed；
- repository constraints 与 architecture tests：8 passed；
- py_compile、Python size 和 architecture script 全部通过；
- training semantics 与 physics integrity 通过；
- 沙箱内全量 pytest 有 65 项 MuJoCo 渲染测试因
  `CGLError: invalid CoreGraphics connection` 失败；所有失败均发生在 CGL context
  创建处，不经过 P36 实现；
- 按沙箱规则在沙箱外用同一提交重跑全量 pytest 后全部通过：
  - 11 项既有 skip；
  - 18 条 warning 全部来自 `torch.jit.script` deprecation；
  - 无测试失败；
- 正式 run 后逐字段核验：
  - report/manifest SHA-256 与 bytes；
  - source commit；
  - 四个输入 identity；
  - 144/108/36 Episode；
  - 0/2,196 safety intervention；
  - 27 cell；
  - 四个能力禁止字段；
  - 固定限定声明；
  均通过。
- 历史 `docs/research-loop/0001/`～`0004/` 零差异。

### 判定

P36-E1 全部冻结接受门通过，正式标记：

`accepted as evaluation contract evidence`

下一步不能直接把 P11 不平衡分布当 capability benchmark。需要独立冻结 P36-E2 的
balanced factorial seed、预算、统计和现有 benchmark integration，先建立无能力变化的
新评测基线。

## 其他候选

| ID | 当前结论 |
|---|---|
| `R0001-P31` | `changes_required`，排在 P32 后 |
| `R0001-P32` | `changes_required`，P36-E1 后的低成本诊断候选 |
| `R0001-P33` | `deferred`，依赖 P31 |
| `R0001-P34` | `rejected`，放宽 source-age safety |
| `R0001-P35` | `deferred`，缺安全 runtime 前置 |
| `R0001-P37` | `rejected`，与现有双臂 shaping 重复且引入单臂捷径 |
| `R0001-P38` | `changes_required`，只允许 shadow/report-only |

所有后续结果必须报告全部冻结 seed、失败和异常，不得挑选结果。
