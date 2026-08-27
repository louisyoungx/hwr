# R0015 结果

## 结果总表

| ID | 判定 | 结论边界 |
|---|---|---|
| `R0001-P80-E1` | `rejected` | version-sealed consumer 的主假设被 mutation 反例否定，且冻结完整验证的 process-tree RSS 超过 2GiB；没有正式 output |
| `R0001-P81` | `rejected as standalone proposal` | 与 P79 已完成的双 bank replay 高度重复；显式 generator dependency 只保留为未来 consumer 的局部要求 |
| `R0001-P82` | `deferred` | 当前没有默认迁移授权；AST/runtime completeness 不能声称覆盖任意 Python 动态路径 |
| `R0001-P68-E2` | `deferred` | 需拆分 route availability 与 nonempty selected-support association，并建立独立 support-coordinate oracle |
| `R0001-P74-E1` | `deferred` | 只在修订后的 P68 label-blind single-replay preflight 仍超预算时考虑 |
| `R0001-P76-E2` | `deferred` | 需拆分 safe-prefix coverage 与 eligible geometry denominator，并建立 v2 prefix bridge |
| `R0001-P77-E2` | `rejected in current form` | 缺少具体 witness 门和 rejection 条件，且 search objective 存在 evaluator-private truth 泄露风险 |

本轮没有训练、参数更新、checkpoint、policy inference、MuJoCo physical acquisition、
B0–B7 action、contact phase、capability evaluation 或新家务任务成功。

## `R0001-P80-E1`

### 冻结目标

P80 试图证明：

> 使用外部 Git receipt、显式双根、typed immutable envelope、完整 artifact ledger 和
> consumer architecture gate，可以让 P79 v2 bank 的静默混版、schema 洗白、错误根、
> 路径逃逸和 producer drift 在下游消费前 fail closed。

冻结接受门要求所有正控、25 类负控、provenance、architecture 与资源门同时通过。

### 实现

最终压缩后的单一实现提交：

`3fff8389645f9cf24a0de2b934eb689de627b9f3`

只新增冻结允许的四个文件：

- `src/hwr/eval/candidate_artifact_contract.py`
- `src/hwr/apps/evaluate_candidate_artifact_contract.py`
- `tests/test_candidate_artifact_contract.py`
- `tests/test_candidate_artifact_contract_app.py`

实现包含：

1. P79 v2 bank 与 P50 legacy source 的显式双根角色；
2. frozen dataclass/enum typed envelope；
3. artifact commit、tree、bank/manifest blob、producer source blob 与 frozen document
   的 Git identity；
4. 24 Episode、384 capture、24 v2 candidate、24 legacy candidate、768 capture blob、
   28 P79 artifact、795 P50 artifact和 796 P50 input file 的完整读取与 hash/size
   校验；
5. candidate canonical bytes、capture composite identity、acquisition base pose、
   capture schema/phase、regression audit 与 selected metadata 的交叉绑定；
6. absolute/traversal/symlink、双根重叠、schema、candidate、capture、producer和 Git
   receipt 的 mutation-sensitive fail-closed 测试；
7. 完整 pytest、冻结既有失败复现、repository gates、总 wall-time、self/child RSS、
   staging-before-rename 与无 partial output 门；
8. claim flags 全部为 false。

该实现对一个重要证据缺口保持了明确边界：

- P79 bank保存 v2 `score_bytes_sha256`；
- P79 regression没有独立 v2 score hash或 score bytes ledger；
- 因此 v2 selected index、score和selection relation不能独立复核；
- 即使其他技术门全部通过，runner也只允许输出
  `inconclusive_artifact_contract_insufficient`，不得输出 accepted。

### 主 Agent 验证

压缩为单一提交后执行冻结 focused 命令：

```text
.venv/bin/python -m pytest \
  tests/test_candidate_artifact_contract.py \
  tests/test_candidate_artifact_contract_app.py \
  tests/test_candidate_mask_ownership_app.py \
  tests/test_initial_candidate_association.py
```

结果：

- `104 passed in 29.67s`；
- 25 个预注册负控全部有独立 node；
- semantic mutation使用 category断言，不再普遍被顶层 bank hash提前吞掉。

其他门：

- Python size：458 files通过，file `<=800` lines、function `<=200` lines；
- architecture：通过；
- compileall：通过；
- `git diff --check`：通过；
- 工作区干净；
- `f224149..3fff838` implementation commit count 为 1；
- changed file集合精确等于冻结四文件；
- `docs/research-loop/0001/`～`0014/` tree保持不变。

### 独立红队过程

独立审计不是一次性通过。初始实现虽然 focused 69 项通过，但红队发现：

1. `producer_sources=()` 会真空通过；
2. architecture gate可被 `getattr`、bound alias、`subprocess cat`和 import alias绕过；
3. 多数负控只在最外层 hash处失败，没有命中命名语义层；
4. report可只靠两个布尔值接受伪 pytest receipt；
5. self RSS没有计入 validation child；
6. selected/score关系与多个 capture/provenance字段没有明确绑定。

第二版将 focused增至95项并补齐这些问题，但红队继续发现：

1. dynamic reader与 dynamic schema仍有可枚举绕过；
2.真实 `pytest -vv` 的 failure ID没有被旧正则捕获；
3.完整 pytest可能使用 MPS，不能称 validation suite为严格 CPU-only；
4.实现历史有多个提交，不符合冻结单提交要求。

最终版将：

- app/evaluator import、完整 AST、call shape和 subprocess command收缩为冻结 allowlist；
- failure parser兼容真实 `FAILED <node>` 输出；
- validation accelerator边界如实记录；
- score/selection缺证转为 inconclusive；
- 实现压缩为单提交。

### 确认性反例

即使经过三轮收缩，P80 的 accepted 假设仍被否定。

#### 反例 1：default generator audit的数据流不完备

主 Agent在内存中对 legacy `generate_candidate_set()` 做 mutation：

```text
if False:
    unused = {"schema_version": CANDIDATE_SCHEMA}
document = {
    "schema_version":
        "hwr.p79-target-candidates/" + chr(118) + chr(50),
    ...
}
```

该 mutation：

- 保留一个不可执行的 v1 schema reference；
- 实际运行路径动态生成 v2 schema；
- 不出现可被简单字符串扫描捕获的 literal v2；
- `audit_default_generator_schema()` 仍返回 `passed=True`。

这直接满足冻结拒绝条件：

> consumer/default-version bypass mutation 仍通过。

因此 P80 不能接受为一般性的 version-sealed consumer合同。

#### 反例 2：architecture receipt仍与实现同源

最终 app使用严格 AST/call fingerprint，能够拒绝已知 alias、dynamic import、`cat`、
`sed`、`awk`和额外 subprocess路径。它能有效防止当前冻结 source意外漂移。

但 fingerprint常量和被审计 app仍位于同一允许修改文件，没有在 `f224149` 文档中独立
冻结。同步修改 app与 fingerprint仍是同源自证，不能证明未来 consumer只经 resolver。

该问题不否定当前 source与当前 receipt的一致性，但否定 P80 想要的通用 consumer
完整性声明。

#### 反例 3：冻结 RSS预算超限

独立红队按最终 full-pytest路径实测：

- `1186 passed`；
- `1 failed`；
- `11 skipped`；
- wall time：`246.77s`；
- 唯一失败：
  `tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete`；
- 该失败已在冻结 `f224149` detached clean worktree复现。

资源：

- self peak RSS：`36,569,088 bytes`；
- validation children peak RSS：`2,406,711,296 bytes`；
- 保守 process-tree上界：`2,443,280,384 bytes`；
- 冻结上限：`2,147,483,648 bytes`；
- 超出：`295,796,736 bytes`。

正式 runner会在原子 rename前触发 `P80 RSS budget exceeded` 并删除 staging。

### 其他已知边界

1. static symlink、同 bytes symlink和读前后可见 inode替换会 fail closed；但不是
   `openat/O_NOFOLLOW` 的原子路径绑定，极端 replace/restore TOCTOU仍可构造。
2. final output使用 staging并在 rename前检查 existing output；但 check与
   `os.replace()` 之间仍有并发空目录竞态。
3. complete pytest在当前机器可能执行已有 MPS测试；P80 resolver本身不导入或调用
   MuJoCo/Torch，不执行物理或科学计算。validation suite与evaluator compute必须分开描述。
4. fixed Git hashes证明当前 source与已提交 frozen document/artifact receipt一致，不
   抵御恶意改写整个仓库历史。

### 判定

`rejected`

理由：

1. default-version audit存在确认性 bypass；
2. architecture完整性仍依赖同源 mutable fingerprint；
3.冻结 full validation的 process-tree RSS超过2GiB；
4. P79 artifact本身缺少独立 v2 score/selection lineage，即使工程门全过也只能
   `inconclusive_artifact_contract_insufficient`。

## 为什么没有正式 artifact

正式命令没有启动：

```text
.venv/bin/python -m hwr.apps.evaluate_candidate_artifact_contract \
  --bank runs/research-loop/0014/r0014-p79-candidate-bank-s20267901 \
  --output runs/research-loop/0015/r0015-p80-artifact-contract-s20268001
```

原因不是隐藏失败，而是冻结流程要求先执行最低成本、具有判别力的验证。确认性 mutation
已触发 rejected门，且完整验证实测必然超过RSS门；继续启动正式 command只会重复约四
分钟验证后 fail closed，不能产生新的科学信息。

最终确认：

- 正式 output不存在；
- `.tmp` staging不存在；
- 没有 partial report；
- 没有把内存值或独立审计日志冒充正式 artifact。

## Git处理

- rejected实现提交：
  `3fff8389645f9cf24a0de2b934eb689de627b9f3`
- 从当前基线回退：
  `788a61c`
- 实现和测试仍可通过提交引用追溯；
- 当前 baseline不包含 rejected P80代码。
