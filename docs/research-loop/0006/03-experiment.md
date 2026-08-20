# R0006 冻结实验

## 冻结状态

`frozen before implementation`

- 冻结前父提交：`c1bf06f83400aabea9e63bfe57eceec1f8e85516`
- 分支：`feat/research-loop`
- 实施顺序：
  1. `R0001-P39-E1`
  2. P39 通过后实施 `R0001-P36-E2`
- 本轮不启动正式训练，不运行 qualified/unqualified policy 的完整 capability Episode。
- P39 与 P36-E2 的正式结果只能是评测合同证据，不是机器人能力改善。

## `R0001-P39-E1`：标准 policy 接口 seed 隔离

### 唯一主假设

当前 evaluator 把 raw environment seed 直接传给 `policy.reset()`。在不改变 environment
seed、environment observation、policy 权重、action value、runtime 或 safety 的前提下，
可以通过 opaque planned Episode identity 与独立 domain seed 派生，使标准 policy
接口只收到独立 policy RNG seed，并保持完全可重放的 commitment/reveal lineage。

E1 只声称关闭标准 `Policy.reset(task_id, seed)` 接口的 raw environment-seed 直通，不声称：

- 隔离同进程恶意 policy 对 evaluator 内存、模块全局或文件系统的读取；
- 隐藏 observation 本身携带的环境信息；
- 提升闭环成功率、安全或泛化。

### 负责人和文件所有权

- 唯一实施负责人：一个 P39 实施 Agent。
- 允许修改：
  - `src/hwr/eval/seed_contract.py`
  - `src/hwr/eval/bimanual.py`
  - `src/hwr/eval/__init__.py`
  - `src/hwr/apps/evaluate_bimanual_rl.py`
  - `src/hwr/apps/evaluate_foundation_world_model.py`
  - `src/hwr/apps/evaluate_seed_isolation.py`
  - `tests/test_seed_contract.py`
  - `tests/test_bimanual_evaluator.py`
  - `tests/test_bimanual_evaluation_app.py`
  - `tests/test_foundation_evaluation_app.py`
  - `tests/test_seed_isolation_app.py`
- 禁止修改：
  - policy/model/training/safety/task/config；
  - `docs/research-loop/0001/`～`0005/`；
  - P36-E2 文件；
  - deployment/action-causality 准入逻辑。

### 冻结 seed 合同

#### Identity

每个 Episode 使用：

- `plan_id`：由 evaluator invocation/benchmark plan 固定；
- `planned_episode_id`：
  `SHA256(schema || plan_id || task_id || ablation || episode_ordinal)`；
- 该 identity 不含结果、成功、安全、模型输出或运行时异常。

#### 双 domain 派生

从 256-bit salt 与 `planned_episode_id` 独立派生：

- `environment_seed = int63(SHA256(salt || "environment" || planned_episode_id))`
- `policy_rng_seed = int63(SHA256(salt || "policy" || planned_episode_id))`

要求：

- 两者均为 `[0, 2^63)`；
- 同一 Episode 两者不得相等；
- baseline/candidate 对应 Episode 的两类 seed 分别完全一致；
- policy 只接收 `policy_rng_seed`；
- environment 只接收 `environment_seed`；
- raw environment seed 不得写入 policy 可见 observation 或 reset 参数。

已有 evaluator 需要保留旧 environment-seed 序列时：

- planned identity 可以绑定既有 environment seed/ordinal；
- policy seed 仍只由独立 policy domain 派生；
- environment 行为必须逐项不变；
- P36-E2 的新正式计划不得使用该兼容模式，必须双 domain 独立派生。

#### Commitment/reveal

- deployment/checkpoint hash 固定前只发布：
  - seed schema；
  - plan identity；
  - `SHA256(salt)` commitment；
  - 派生算法。
- deployment hash 固定后才向 evaluator 提供 salt。
- 运行结束后 manifest reveal salt，以便重放。
- salt、seed plan 或私有 evaluator 对象不得通过标准 policy interface 传入。
- formal capability run 的 salt 必须在 deployment hash 固定后随机生成；不得使用仓库内
  公开常量。

P39-E1 诊断使用公开、不可用于正式能力评测的固定 salt：

`R0001-P39-E1-s20263901`

其 commitment 为：

`a94db502b86fd2c83a9096eb856b110de53158f588cc7496a60e4264fc190237`

该固定 salt 只验证算法、接口和重放；报告必须标记：

- `formal_seed_bank=false`
- `capability_claim_allowed=false`
- `threat_model=standard_policy_reset_interface_only`

### 行为兼容

- `BimanualEpisodeEvaluation` 和 report 必须记录：
  - environment seed；
  - policy RNG seed；
  - planned Episode identity；
  - seed commitment identity。
- policy action source、action frame、observation、success、安全和终止逻辑不变。
- evaluator app 使用随机 256-bit salt 或显式 salt 文件；不得把 salt 作为 policy 参数。
- app manifest 必须保存 commitment、reveal、plan ID 与逐 Episode seed pair。
- formal foundation evaluator 的 deployment/action-causality 验证顺序不得放宽。

### 最小验证

1. seed contract 单测：
   - domain separation；
   - deterministic replay；
   - commitment/reveal；
   - invalid salt、重复 identity、seed collision fail closed；
   - baseline/candidate plan 配对。
2. fake evaluator：
   - environment 记录收到的 environment seed；
   - policy 记录收到的 policy seed；
   - 两者不同且与计划一致；
   - report identity 完整。
3. compatibility：
   - 固定 environment seed 的旧新 evaluator environment audit/observation 相同；
   - deterministic policy 使用固定 policy seed 时 action trace 可重放；
   - 不假设不同 policy 消费相同 RNG 次数。
4. canary：
   - canary 只能接收 policy RNG seed，不读取 observation、文件系统或 evaluator 对象；
   - raw environment seed equality/pass-through 次数为 0；
   - 该 canary 只验证接口，不作信息论不可逆声明。
5. CLI 正式诊断：

```text
.venv/bin/python -m hwr.apps.evaluate_seed_isolation \
  --output runs/research-loop/0006/r0006-p39-seed-isolation-e1-s20263901 \
  --salt R0001-P39-E1-s20263901
```

### 资源预算

- CPU；
- wall time `<=5` 分钟；
- peak RSS `<=1 GiB`；
- 新增产物 `<=10 MiB`；
- 不使用 MPS/GPU、tmux、host-exec、后台任务或休眠；
- 不更新模型参数，不创建 checkpoint。

### 正式接受门

以下全部通过才接受为 `evaluation leakage fix evidence`：

1. seed schema、domain 和 commitment/reveal 与冻结合同完全一致；
2. raw environment seed 进入标准 policy reset 接口次数为 0；
3. 全部 planned Episode 的 environment/policy seed 不同；
4. baseline/candidate seed-pair coverage 100%；
5. 相同 plan+salt 重放 bit-identical；
6. environment randomization/instruction/observation 在兼容模式下不变；
7. report/manifest 记录每个 Episode 两类 seed、plan identity 和 lineage；
8. 无 API fallback 会把 environment seed 重新传给 policy；
9. 两个 evaluator app 集成且原 deployment 准入不变；
10. focused tests、Python size、architecture、py_compile 与 `git diff --check` 通过；
11. 历史 `docs/research-loop/0001/`～`0005/` 零差异；
12. 报告明确禁止能力声明和恶意进程隔离声明。

判定：

- `accepted as evaluation leakage fix evidence`：全部通过；
- `rejected`：仍存在 raw seed pass-through、plan 可结果后替换、environment 行为改变或
  baseline/candidate 不配对；
- `inconclusive`：artifact/lineage 无法重建。

## `R0001-P36-E2`：平衡双账本 benchmark 合同与 runner integrity

### 前置

- 只有 P39-E1 接受并原子提交后才实施。
- P36-E2 复用 P39 seed contract，但使用公开诊断 salt，不生成未来正式能力 seed bank。
- 不运行 policy inference，不执行完整 Episode，不产生 closed-loop success。

### 唯一主假设

在不改变 policy、runtime、安全、任务成功条件或非 latency 随机化的前提下，可以建立一个
结果前冻结、27-cell 平衡、baseline/candidate 配对、缺失 fail-closed、统计功效可审计的
未来能力 benchmark 合同。

### 负责人和文件所有权

- 唯一实施负责人：一个 P36-E2 实施 Agent；在 P39 提交之后开始。
- 允许修改：
  - `src/hwr/eval/factorial_benchmark.py`
  - `src/hwr/apps/evaluate_factorial_benchmark_contract.py`
  - `src/hwr/adapters/mujoco/formal_household_backend.py`
  - `tests/test_factorial_benchmark.py`
  - `tests/test_factorial_benchmark_app.py`
  - `tests/test_formal_household_dual_arm_backend.py`
- 允许只读使用 P39 seed contract。
- 禁止修改：
  - `src/hwr/eval/bimanual.py`
  - P39 文件；
  - policy/model/training/safety/task/config；
  - foundation deployment/action-causality 准入；
  - 历史轮次文档。

### 冻结 benchmark cell

- task：
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
  - `tidy_living_room_3d/v1`
- observation latency：`1/2/3`
- action latency：`1/2/3`
- 每个 training-seed slot、task、latency cell 使用相同 replicate 数 `n`。
- baseline/candidate deployment 对同一 pair identity 使用相同 environment/policy seed。
- training-seed slots：3；slot 只是未来 deployment pair 的层级位置，不是当前模型。
- ablation 不纳入主 27-cell power；未来正式 run 仍需单独保持既有左右臂消融门。

### 合成功效冻结

P36-E2 不用真实 policy 结果选择 `n`。正式 contract run 只运行预注册合成设计：

- candidate `n`：`4, 8, 12, 16, 24, 32`；
- outer training-seed slots：3；
- 3 task × 3 observation latency × 3 action latency；
- baseline success probability：`0.10/0.30/0.50/0.70`；
- paired shared-randomness fraction：`0.0/0.5/0.9`；
- null：candidate probability 等于 baseline；
- planted：candidate probability 为 `baseline + 0.10`；
- 每个 `n × probability × shared fraction × condition`：500 Monte Carlo trial；
- 每 trial：1,000 hierarchical paired bootstrap replicate；
- bootstrap：
  - cell 内同步重采样 paired Episode；
  - training-seed slot 为外层同步重采样单位；
  - cell 权重固定等权；
- 同时检验：
  - `Δcomplete_macro` 95% percentile lower `>0`
  - `Δsupported_macro` 95% percentile lower `>0`
  - planted point estimate `>=0.10` 不作为通过条件，只用于审计有限样本偏差；
- empirical null FPR 与 planted power 使用 95% Clopper-Pearson 区间；
- 每个 n 的最坏分层：
  - null FPR upper `<=0.05`
  - planted power lower `>=0.80`
- 选择满足全部分层的最小 n；
- 若 `n<=32` 均不满足，decision 为 `inconclusive_power`，不得扩大真实预算或查看模型
  结果后改设计。

随机数：

- Monte Carlo base seed：`20263602`
- bootstrap seed 由 base seed、n、probability index、shared index、condition、trial index
  通过固定 SHA-256 domain 派生；
- 不使用系统随机数。

### 计划总账与缺失合同

只有功效门选出 n 后，生成 diagnostic planned ledger：

- pair 数：`3 training slots × 27 cell × n`；
- future execution 数：每个 baseline/candidate 角色各一份，共 `162n` Episode；
- 每个 pair identity 唯一，角色不进入 environment/policy seed 派生；
- planned manifest 在任何 terminal record 前原子写入；
- terminal record append-only，hash-bound，不覆盖。

状态分类：

- `valid_success`：完整终止且成功；
- `valid_failure`：
  - task timeout；
  - task failure；
  - safety rejection/termination；
  - severe collision；
  - policy 明确返回 invalid/NaN/exception 且有可验证 policy provenance；
- `unresolved_infrastructure`：
  - host kill/power loss；
  - artifact corruption；
  - 无法唯一归因为 policy 或 environment 的异常；
- 不允许 replacement seed、complete-case 删除或按相似 cell 补抽。
- `planned = valid_success + valid_failure + unresolved_infrastructure`。
- unresolved 非 0 时实验只能 `inconclusive`。

### 双账本与未来统计

- 首要 `complete_challenge`：
  - 27 cell 等权；
  - latency 3 占 observation-latency 权重三分之一；
  - 所有 valid failure 保留。
- `supported_conditional`：
  - observation latency 1/2 的 18 cell 等权；
  - 不能替代首要账本。
- 未来共同主要门：
  - paired `Δcomplete_macro` 95% lower `>0`；
  - paired `Δsupported_macro >=0.10` 且 95% lower `>0`。
- 逐任务 supported point estimate 不得低于 baseline 超过 0.05。
- severe collision 必须为 0；过期动作实际应用率为 0；既有 P13 safety burden、稳定保持、
  并发双臂和消融门不变。
- 三个 training seed 必须作为外层层级，不得把 Episode 扁平当训练重复。
- latency 3 当前标记 `full_profile_supported=false`，不能用完整 macro 的旧 70% 门宣称
  全域支持。

### 联合 latency-only reset

新增 evaluator-only diagnostic reset：

- observation latency 只允许 `1/2/3`；
- action latency 只允许 `1/2/3`；
- 必须在同一次 reset 联合覆盖；
- sampled randomization 先完整生成，再覆盖两字段，禁止改变 RNG 消费顺序；
- provenance 同时记录：
  - sampled/effective latency；
  - sampled randomization SHA-256；
  - effective randomization SHA-256；
  - 去除两 latency 后的 `other_randomization_sha256`；
  - `verified_only_latency_pair_changed=true`。
- 不修改旧单项 diagnostic 的语义。
- 不允许 policy 读取 cell label；不允许 latency 3 stale 动作执行。

### 最小验证与命令

1. property tests：27-cell 覆盖、pair identity、双 domain seed、无重复/替换；
2. synthetic power：所有 candidate n、null/planted 分层、Clopper-Pearson；
3. fault injection：退出、损坏、重复、越界 cell/seed、policy failure/infrastructure unknown；
4. reset-only MuJoCo：
   - 每任务 × 9 latency cell 至少一个固定 diagnostic seed；
   - 不执行 action；
   - 同 seed 9 cell 的 `other_randomization_sha256` 完全一致；
   - instruction、object pose、camera、mass/friction/visual/actuator 字段除 latency 外一致；
5. 正式 contract command：

```text
.venv/bin/python -m hwr.apps.evaluate_factorial_benchmark_contract \
  --output runs/research-loop/0006/r0006-p36-factorial-e2-s20263602 \
  --salt R0001-P36-E2-s20263602
```

诊断 salt：

- reveal：`R0001-P36-E2-s20263602`
- commitment：
  `f094032ccc029cc15979be8ffd636d9566500398f356256bc23efc5d8f88cdc9`
- `formal_seed_bank=false`
- `capability_claim_allowed=false`
- `closed_loop_success_available=false`

### 资源预算

- CPU；
- contract/power wall time `<=15` 分钟；
- reset-only smoke wall time `<=10` 分钟；
- peak RSS `<=2 GiB`；
- 新增产物 `<=50 MiB`；
- 不使用 MPS/GPU、tmux、host-exec、后台任务或休眠；
- 不执行完整 Episode，不更新模型，不创建 checkpoint。

### 正式接受门

以下全部通过才接受为 `balanced benchmark contract evidence`：

1. P39 已接受且 P36 实现提交以其为祖先；
2. 27 cell、3 training slot 和 selected n 的 planned pair 数完全匹配；
3. synthetic null FPR upper 与 planted power lower 对全部冻结分层通过；
4. 若没有 n 通过，则不生成可用正式计划并标记 `inconclusive_power`；
5. 双账本权重、latency 3 完整分母和逐任务守护固定；
6. baseline/candidate pair identity 与两类 seed 100% 匹配；
7. planned/terminal/unresolved 恒等式和 fail-closed 聚合通过全部故障注入；
8. policy failure 不会被重标为 infrastructure missing；
9. 27 个 reset-only cell 只改变两项 latency，其他 hash 一致；
10. latency 1/2/3 source-age 合同与 P29 一致；
11. 不运行 policy inference、完整 Episode或视频；
12. report 固定：
    - `formal_seed_bank=false`
    - `capability_claim_allowed=false`
    - `closed_loop_success_available=false`
    - `primary_ledger=complete_challenge`
    - `full_profile_supported=false`
13. focused tests、Python size、architecture、py_compile 与 diff check 通过；
14. 历史 `docs/research-loop/0001/`～`0005/` 零差异。

判定：

- `accepted as balanced benchmark contract evidence`：全部接受门通过；
- `inconclusive_power`：n<=32 没有设计达到功效门；
- `rejected`：cell/seed/权重/缺失可被结果后改变，联合 override 改变其他随机化，或任何
  设计删除 latency 3/放宽 safety；
- `inconclusive`：artifact、MuJoCo reset 或 lineage 无法重建。

## 本轮停止条件

- P39 失败：停止 P36-E2，不尝试兼容性绕过。
- P36-E2 power 不足：记录 `inconclusive_power`，不扩大正式预算。
- reset-only smoke 发现非 latency drift：拒绝当前联合 override，不修订门槛。
- 本轮无论结果如何都不启动训练；P32-E1 留给下一轮重新确认。
