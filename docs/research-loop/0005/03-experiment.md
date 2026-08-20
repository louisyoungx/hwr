# R0005 冻结实验

## `R0001-P36-E1`：时效双账本合同与历史证据重放

### 状态

`frozen before implementation`

P36-E1 是本轮唯一获批实施项。它只验证评测合同与历史证据聚合，不修改当前正式
benchmark runner、不执行新物理 Episode、不训练，也不产生能力改善结论。

### 唯一主假设

在不改变 runtime、policy、安全层、任务或已有 Episode 的前提下，可以用预先冻结的
source-age 合同把每个 Episode 唯一归入：

- `supported`：该 Episode 所属 observation-latency stratum 的最大 visible
  observation age `<=100ms`；
- `challenge`：最大 visible observation age `>100ms`。

同时保留：

1. `complete_challenge` 首要总账：包含全部 Episode，不删除 latency=3；
2. `supported_conditional` 条件账：只描述当前时效合同内的子域。

E1 只用 P11/P29 证据验证分类、覆盖和安全拒绝归因；P11 是诊断数据，没有闭环任务成功
标签，因此 E1 不报告或推断成功率。

### 冻结父提交与负责人

- 冻结前父提交：`cae86e48eebc45e170f91c633525f00e965cec98`
- 分支：`feat/research-loop`
- 实施负责人：一个 P36 实施 Agent
- 文件所有权：
  - `src/hwr/eval/support_domain.py`
  - `src/hwr/apps/evaluate_support_domain.py`
  - `tests/test_support_domain.py`
- 主 Agent 只集成、审查、运行门禁和记录结果。
- 禁止修改：
  - `src/hwr/eval/bimanual.py`
  - `src/hwr/apps/evaluate_foundation_world_model.py`
  - runtime、policy、safety、task、training 与 config 文件
  - `docs/research-loop/0001/`～`0004/`

### 冻结输入

| 输入 | SHA-256 | bytes |
|---|---|---:|
| P11 report | `79195b94f48e8b59ce04bb7a9a3a680f8717f6484da69b2f28d54b3a9b842cda` | 423,970 |
| P11 manifest | `509f1b7f11caa2aa0dced3324bcc6ea9af9b045071c85b1f6a2bc66a7160da3a` | 28,249 |
| P29 report | `7729eed53e034bef5a9d5c50bd1d87d6025370b290226f3055c480f3014274fb` | 71,794 |
| P29 manifest | `41495b1a1efa9f24b65e8c5359bc98cfa28ecee4dbdbc20a5c20d37aa7900cef` | 317 |

路径：

- `runs/research-loop/0003/r0003-p11-causal-plant-s20261101`
- `runs/research-loop/0004/r0004-p29-stale-validity-s20262901`

必须验证：

- P11 source commit：
  `ef86971ecd9528c00022e3f944e7878f66665f4a`
- P29 source commit：
  `b34716d32cccfe5e619b18aaecaaeda8954a765a`
- P29 report 内记录的 P11 report/manifest hash 与上表一致；
- 两个 manifest 对其声明 artifact 的 hash/bytes 验证仍通过；
- P29 assessment 为 `passed=true` 且 decision 为 `contract_incompatible`；
- P29 guardrails 显示没有延长 validity、删除 latency、替换 latest bundle 或改变 P11
  decision。

### 冻结时效合同

- control frequency：20Hz；
- control period：50,000,000ns；
- validity duration：100,000,000ns；
- 边界：`age == 100ms` 为 supported，`age >100ms` 为 challenge；
- P29 冻结映射：

| observation latency | 最大 source age | domain |
|---:|---:|---|
| 0 | 0ms | `supported` |
| 1 | 50ms | `supported` |
| 2 | 100ms | `supported` |
| 3 | 150ms | `challenge` |

- Episode 归域按预注册 latency stratum 的最大稳态 source age，不能按 warm-up 前几步的
  较小 age 把同一 latency=3 Episode 拆成 supported。
- action latency 只作为报告 strata，不改变 observation-age domain。
- classification 只存在于 evaluator/report，不能写入 policy、action frame 或模型输入。

### P11 冻结覆盖

- Episode 总数：144；
- 每任务：48；
- observation latency：
  - latency1：18；
  - latency2：90；
  - latency3：36；
- action latency 1/2/3：各 48；
- task × observation latency × action latency 共 27 个 cell；
- 每任务每 action-latency 下：
  - observation latency1：2 Episode；
  - observation latency2：10 Episode；
  - observation latency3：4 Episode。

因此预期：

- `complete_challenge`：144 Episode；
- `supported_conditional`：108 Episode；
- `challenge`：36 Episode；
- 每任务 supported/challenge：36/12；
- 每 action latency supported/challenge：36/12；
- supported safety intervention：0；
- challenge safety intervention：36×61 = 2,196；
- complete severe collision：0；
- 所有 Episode 均未提前终止。

P11 不是未来正式 balanced benchmark；E1 必须显式报告各 cell count 和 observation
latency 不平衡，不能把 144 Episode 称为新的 factorial capability benchmark。

### 实现合同

`support_domain.py` 必须：

1. 解析并验证冻结 P11/P29 report；
2. 从 P29 timeline 建立 latency -> maximum age 映射；
3. 逐一分类 P11 Episode，保留 task、seed、rho、observation latency、action latency、
   safety intervention、severe collision 和 early termination；
4. 拒绝未知 latency、重复 Episode identity、缺字段、hash/lineage 不匹配、future
   domain label、Episode 丢失或多次计数；
5. 生成首要 `complete_challenge`、条件 `supported_conditional` 和 `challenge` 三份聚合；
6. 按 task、observation latency、action latency 和完整 27-cell 报告 count；
7. 生成固定声明：
   `支持 visible observation age <=100ms 的 evaluation 子域；完整 evaluation profile 尚未支持。`
8. 明确：
   - `capability_claim_allowed=false`
   - `closed_loop_success_available=false`
   - `balanced_factorial_benchmark=false`
   - `primary_ledger=complete_challenge`

CLI 必须：

- 只接收 P11/P29 run path 和一个不存在的 output path；
- 原子写 `report.json` 与 `manifest.json`；
- manifest 记录 source commit、命令、输入/输出 SHA-256 与 bytes；
- 不覆盖既有目录；
- 不读网络、不启进程、不修改输入。

### 冻结命令与产物

正式 E1 命令：

```text
.venv/bin/python -m hwr.apps.evaluate_support_domain \
  --p11-run runs/research-loop/0003/r0003-p11-causal-plant-s20261101 \
  --p29-run runs/research-loop/0004/r0004-p29-stale-validity-s20262901 \
  --output runs/research-loop/0005/r0005-p36-support-domain-e1-s20263601
```

正式产物：

- `runs/research-loop/0005/r0005-p36-support-domain-e1-s20263601/report.json`
- `runs/research-loop/0005/r0005-p36-support-domain-e1-s20263601/manifest.json`

没有随机抽样 seed；run suffix `s20263601` 只是稳定运行身份，不得据此选择结果。

### 资源预算

- device：CPU；
- wall time：不超过 5 分钟；
- peak RSS：不超过 1 GiB；
- 磁盘新增：不超过 10 MiB；
- 不使用 MPS/GPU；
- 不启动 tmux、后台任务、host-exec、watchdog 或休眠；
- 不更新模型参数，不创建 checkpoint。

### 正式接受门

以下全部通过才接受 P36-E1 为 `evaluation contract evidence`：

1. 四个冻结输入 hash/bytes 完全一致；
2. P11/P29 lineage、P29 assessment 与 guardrails 完全匹配；
3. 144 个 Episode 恰好各出现一次；
4. supported/challenge count 为 108/36；
5. task 分层各为 36/12；
6. action latency 1/2/3 分层各为 36/12；
7. 27 个 cell 及其 2/10/4 计数完全匹配；
8. supported/challenge safety intervention 为 0/2,196；
9. severe collision 为 0，early termination 为 0；
10. latency0/1/2/3 最大 age 与 domain 映射和 P29 完全一致；
11. complete ledger 是首要总账，且 latency3 未删除、未重标、未拆 Episode；
12. 固定限定声明与四个禁止能力字段完全匹配；
13. 聚焦测试、Python size、architecture 和 py_compile 通过；
14. 历史 `docs/research-loop/0001/`～`0004/` 零差异。

### 判定

- `accepted as evaluation contract evidence`：全部接受门通过。
- `rejected`：聚合器丢 Episode、缩小完整分母、错误归域、允许结果后分类，或把条件账
  提升为能力结论。
- `inconclusive`：冻结 artifact 缺失/损坏、lineage 无法验证，或输入 schema 无法唯一
  重建。

E1 通过也不能：

- 宣称家务成功率、泛化、安全能力或模型能力改善；
- 让 P01 v4 成为 causality-qualified deployment；
- 改变当前 benchmark acceptance；
- 把 supported conditional metric 与旧完整域数值直接比较；
- 自动授权 P32/P31/P33/P38。

### 后续路由

- E1 通过后，下一步是在新稳定 ID 或 P36-E2 下独立冻结未来 balanced factorial
  capability benchmark 的 seed、预算、统计和 integration；不得沿用 P11 不平衡分布作为
  正式成功率 benchmark。
- E1 完成后可重新审查 P32 的 nested cross-fit 合同；不得与 P36 首次实现捆绑。
