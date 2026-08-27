# R0015 冻结实验

## 决策

- 本轮科学候选：`no-go`。
- 本轮唯一实施项：`R0001-P80`。
- 类型：`engineering/evidence-hygiene prerequisite`。
- 不实施 P81、P82、P68-E2、P74-E1、P76-E2、P77-E2。
- 本轮不运行 MuJoCo physical cohort、B0–B7 action、policy inference、训练或
  capability evaluation。

## `R0001-P80-E1`

名称：version-sealed candidate artifact resolver。

### 单一主假设

若 P79 v2 bank 的 consumer：

1. 从 artifact 外部冻结 Git trust anchor；
2. 显式区分 v2 bank root 与 legacy source root；
3. 联合验证 outer bank、inner candidate、manifest、producer source、blob identity 和
   selected metadata；
4. 只返回不可变且携带 version/producer/root-role 的 typed envelope；

则正式 24-Episode bank 可唯一解析，而 v1/v2 静默混用、schema 洗白、错误根、路径逃逸和
producer drift 可在任何下游消费前 fail closed。

本实验不证明 artifact 由某次真实执行产生，只证明当前工作树中的 bytes 与结果前固定的
Git receipt 一致。

## 冻结 trust anchor

### Artifact

- artifact commit：
  `93ea4e7afad8c52d83abd54f41a2d08d40a3cab4`
- artifact root：
  `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`
- root Git tree：
  `9a78c75e1f26b2c80399626042252b4e87404169`
- `bank.json`：
  - Git blob：`471d7fbc526ac1c73b1efdafd03c9f073bcf3e5c`
  - SHA-256：`888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e`
- `manifest.json`：
  - Git blob：`5d99ddc39e475c98e8eb3a64132f52192ed84061`
  - SHA-256：`162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9`
- bank schema：`hwr.p79-candidate-bank/v1`
- manifest schema：`hwr.p79-candidate-mask-ownership-artifacts/v1`
- candidate schema：`hwr.p79-target-candidates/v2`
- proposal：`R0001-P79-E1`
- Episode：24
- capture：384

### Producer

- producer commit：
  `9eef9953f8a8558228a5e8870d7d2d8f7499ee1e`
- source blobs：

| 路径 | Git blob | SHA-256 |
|---|---|---|
| `src/hwr/eval/candidate_mask_ownership.py` | `3d3839605eb290f9f2e0b77ec7db22ac7de15a31` | `9bcf9eaa45238f3053022010158188b642478185c39ab24976130e4cd4fd6c9a` |
| `src/hwr/eval/target_selection.py` | `d7e588ba76ce18882255e3e22b1f86459ab235dd` | `54961b5e84f29d58efe01ccfe24d04ffcf04b76a36697aac8d24a431ec4c9b4a` |
| `src/hwr/apps/evaluate_candidate_mask_ownership.py` | `30759570f978eb73e612515e4e0c256f3f374dcf` | `70cc24d20ba00a79f1694882005602b1cca8807c7e4f8fc5f082aa7e59e455a8` |

### Legacy source root

- root：
  `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
- manifest：
  - bytes：`186310`
  - SHA-256：`cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86`
- source commit：
  `d67791a53491ce37cddaef4bd7d6b71ad3e66ac2`
- manifest schema：`hwr.p50-acquisition-artifacts/v1`
- legacy candidate schema：`hwr.p41-target-candidates/v1`

P50 root 不在 Git tree 内；其信任链来自已由 Git 固定的 P79 `manifest.json` 中的完整
`input.files` identity，以上 P50 manifest SHA/size作为额外结果前锚，不替代该链。

### Frozen document

- 本文件必须在实现开始前单独提交。
- 实现中的 `FROZEN_DOCUMENT_COMMIT` 必须是包含本文件当前内容的提交。
- 正式 runner 必须：
  - 验证该提交是 source commit祖先；
  - 从该提交读取本文件并记录 Git blob与 SHA-256；
  - 验证上述 trust anchor常量与实现中的 typed anchor一致。

## 实现边界

唯一负责人：实施 Agent P80。

允许修改：

- `src/hwr/eval/candidate_artifact_contract.py`
- `src/hwr/apps/evaluate_candidate_artifact_contract.py`
- `tests/test_candidate_artifact_contract.py`
- `tests/test_candidate_artifact_contract_app.py`

主 Agent在正式运行后另行写入：

- `runs/research-loop/0015/r0015-p80-artifact-contract-s20268001/`
- `docs/research-loop/0015/04-results.md`
- `docs/research-loop/0015/05-summary.md`

禁止修改：

- v1/v2 generator、score、selector、default runtime；
- P68/P76/P77；
- `src/hwr/apps/__init__.py` 中既有 `read_bound_blob()`；
- 任何 `docs/research-loop/0001/`～`0014/`；
- 任一历史 artifact。

## Resolver 合同

### 输入

- repository root；
- 显式 `P79_V2_BANK` root role；
- 显式 `P50_LEGACY_SOURCE` root role；
- 冻结 `CandidateArtifactTrustAnchor`。

不接受：

- 自动探测版本；
- 自动转换版本；
- 在多个 root中选择“存在的文件”；
- 绝对 artifact path；
- `..` traversal；
- symlink文件或 symlink目录；
- artifact自报字段作为 root trust source。

### 输出

返回 frozen dataclass/enum组成的 immutable envelope，至少包含：

- bank schema、proposal、producer commit；
- Episode identity、task/cell/seed identity；
- root role；
- candidate schema、path、bytes、SHA-256；
- candidate canonical bytes；
- candidate count、selected index、selected canonical identity；
- ordered capture identities及其 policy/candidate-visible bound blobs。

下游不能只取得裸 `CandidateSet` 后丢失 schema、producer或root role。

### 验证

1. 当前 checkout必须包含 artifact commit与producer commit，且二者 ancestry符合冻结链。
2. 当前 `HEAD` 必须包含 artifact commit；正式运行前工作区干净且 source commit等于
   `HEAD`。
3. artifact root、bank和manifest在冻结 artifact commit的 tree/blob identity必须匹配。
4. 当前工作树 bank、manifest及全部 v2 artifacts必须匹配 Git固定的 P79 manifest。
5. P50 legacy root及全部引用输入必须匹配 Git固定 P79 manifest中的 `input.files`。
6. 每个 bank record：
   - Episode ID唯一，cohort与计数冻结；
   - root role显式且不可交换；
   - outer/inner schema与proposal一致；
   - candidate bytes/hash/count一致；
   - acquisition input hashes与 ordered policy capture一致；
   - selected index、score hash和selected canonical identity一致；
   - capture ordinal连续、final input唯一且最后；
   - capture composite identity含 Episode、ordinal、clock和两个 blob hash。
7. 未被 bank引用但位于 P79 manifest `artifacts` 或 P79 manifest `input.files` 的条目仍需
   完整 hash/size校验，禁止只验证被选中的 candidate。
8. consumer module不得导入 generator、score、selector、MuJoCo、policy或classifier。
9. 本轮新增正式 app读取 P79/P50 evidence只能调用 resolver；不得自行 `read_text()`、
   `read_bytes()`、`open()` 或按文件存在性 fallback。

## 预注册负控

所有负控必须由 focused pytest执行并在正式 report中留下名称与通过状态：

1. unknown bank schema；
2. unknown inner candidate schema；
3. outer/inner schema mismatch；
4. inner+outer共同重标并重算 candidate、bank与manifest自带 hash，但外部 Git anchor不变；
5. v2 candidate使用 legacy root；
6. legacy candidate或capture使用 v2 root；
7. 两个 root中存在同名文件时，禁止 existence fallback；
8. absolute path；
9. `..` traversal；
10. symlink file escape；
11. symlink directory/root escape；
12. blob bytes drift；
13. blob size/hash identity drift；
14. duplicate Episode identity；
15. capture ordinal gap/duplicate；
16. non-final或多个 final capture；
17. candidate count mismatch；
18. selected index越界；
19. selected canonical identity mismatch；
20. acquisition input hash mismatch；
21. producer commit drift；
22. producer source blob drift；
23. artifact commit/tree/blob drift；
24. trust anchor drift；
25. app绕过 resolver的 architecture mutation。

所有 mutation只在 `tmp_path` 或内存 fixture中执行，不修改正式 artifact。

## 正控与接受门

### 正控

- 正式 24/24 Episode解析成功；
- 384/384 capture复合 identity唯一；
- 24 个 v2 candidate、24 个 legacy candidate、768 个 capture blob均按显式 root
  解析；
- P79 manifest中28个 artifact全部验证；
- P79 manifest中795个 input file全部验证；
- 重复两次解析后的 canonical receipt bit-identical；
- default generator仍为 v1；
- history tree `0001`～`0014`与 R0015起点一致；
- 无 MuJoCo、generator、score、selector、policy、classifier调用。

### 判定

全部正控、负控、provenance、architecture与资源门通过：

`accepted as version-sealed candidate artifact consumer contract`

任一重标、错根、未知版本、路径逃逸、producer/artifact drift、consumer bypass仍通过：

`rejected`

正式 artifact缺少无法从冻结 Git receipt恢复的必要 lineage：

`inconclusive_artifact_contract_insufficient`

输入被修改、历史 tree漂移、正式运行非干净提交、输出覆盖或验收后调门：

`invalid`

## 测试与命令

Focused：

```text
.venv/bin/python -m pytest \
  tests/test_candidate_artifact_contract.py \
  tests/test_candidate_artifact_contract_app.py \
  tests/test_candidate_mask_ownership_app.py \
  tests/test_initial_candidate_association.py
```

Repository gates：

```text
.venv/bin/python scripts/check_python_size.py
.venv/bin/python scripts/check_architecture.py
.venv/bin/python -m compileall -q src tests
git diff --check
```

正式 evaluator：

```text
.venv/bin/python -m hwr.apps.evaluate_candidate_artifact_contract \
  --bank runs/research-loop/0014/r0014-p79-candidate-bank-s20267901 \
  --output runs/research-loop/0015/r0015-p80-artifact-contract-s20268001
```

完整 pytest在正式运行前执行。只允许记录可在冻结实验提交的 detached clean worktree中
复现的既有失败，不得把新失败列为既有问题。

## 资源预算

- CPU-only；
- wall time：5min；
- peak RSS：2GiB；
- artifact：5MiB；
- 最低磁盘余量：20GiB；
- 不使用 GPU、MuJoCo、tmux、后台任务、休眠或 `traex-host-exec`。

## 提交与停止规则

1. 本文件先单独提交，形成冻结实验提交。
2. 实施 Agent只能修改四个允许文件并创建一个原子提交，提交信息引用
   `R0001-P80`。
3. 正式 evaluator只从干净、已提交且通过门禁的 source commit启动。
4. 正式 output必须原子写入；已有 output或 `.tmp` 时 fail closed，不覆盖。
5. 结果提交后更新 `04-results.md`、`05-summary.md`，提交并 push。
6. 即使 P80 accepted，本轮也不追加 P68、P74、P76、P77、默认迁移或训练。
