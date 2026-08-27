# R0016 冻结实验

## 实验身份

- 提案：`R0001-P83`
- 名称：consumer-local v2 selection-lineage oracle
- 类型：具体 consumer 的 dependency/measurement ablation
- 状态：实验设计冻结，尚未实施
- 起始提交：`7c0a226bf9fbac621ce056cff891d9ea8608a5a4`
- 上下文提交：`e74c82003a1d94ab6a46cd53d3229158fed79ea8`
- 提案提交：`315d68482b6cd97630cce007091f2336db0bdba6`
- 筛选提交：`84665c9f1b6419134e41343c053dcfb7c57d482f`
- 实施分支：`feat/research-loop`
- 负责人：唯一实施 Agent；主 Agent 只做审查、集成、门禁与归因

本轮只实施 `R0001-P83`。不得顺带实施 P84、P85、P68-E3、P76-E3、P76-E4、
P86、P77、默认 v2 migration、selector 修改或训练。

## 冻结问题

P79 v2 bank 缺少 exact score bytes 与原子的 score→selection receipt，但当前 P68/P76
consumer 可访问由 P79 manifest 绑定的 P50 policy-visible captures。

本实验只检验：

> 在当前 checkout 的 P50 冻结 bytes 可用、且 blind worker 在运行时未接触 P79
> score/selection metadata 的条件下，一个 source-disjoint oracle 是否能精确重建
> P79 的 full-precision v2 candidate、score bytes hash、selected index 与 selected
> canonical identity。

### Treatment

blind worker 只收到：

- 净化后的 Episode plan；
- P50 input root；
- blind output path；
- worker 自身固定 source。

plan 只含：

- Episode/task/cell/replicate identity；
- acquisition base pose；
- capture ordinal、final-input flag、observation identity；
- policy-input 与 candidate-visible input 的 path/size/SHA-256。

plan 不得含：

- P79 bank/root/path；
- v2 expected candidate bytes/hash/count；
- expected score hash或 score bytes；
- expected selected index/identity；
- legacy P50 candidate、score或 selected metadata；
- task/entity/interaction labels；
- segmentation、geom/body/site、contact、force、reward、success或专家动作。

### Control 与揭盲

blind worker 正常退出并原子封存两次重建 receipt 后，父 comparer 才读取 P79
commitments并比较。P79 metadata 是冻结 control，不作为 worker 输入。

本实验不把该隔离称为恶意代码安全沙箱；只验证冻结 source、参数、cwd、环境与实际
read ledger 所描述的执行路径没有读取 P79 metadata。

## 冻结输入

### P50 source bytes

- root：
  `runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`
- `capsules.json` SHA-256：
  `223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf`
- `plan.json` SHA-256：
  `5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab`
- `report.json` SHA-256：
  `b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0`
- `manifest.json` SHA-256：
  `cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86`
- P79 manifest 绑定的 P50 input file：796 个；
- capture：384 个；
- policy/candidate-visible blob：768 个；
- capture ledger SHA-256：
  `ff8c5cf53942e89e5ebc04dd8e9020313e5a120dc62ad6ca8764d93a6eda6145`。

这些 bytes 当前存在但受 `.gitignore` 排除。输入缺失或 hash 漂移必须判 `invalid`；
不得把 Git 中的 commitment 误称为输入 bytes 自包含或跨 checkout 可恢复。

### P79 commitments

- root：
  `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`
- Git tree：
  `9a78c75e1f26b2c80399626042252b4e87404169`
- `bank.json` SHA-256：
  `888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e`
- `manifest.json` SHA-256：
  `162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9`
- producer source commit：
  `9eef9953f8a8558228a5e8870d7d2d8f7499ee1e`
- artifact commit：
  `93ea4e7afad8c52d83abd54f41a2d08d40a3cab4`
- v2 producer blob：
  `3d3839605eb290f9f2e0b77ec7db22ac7de15a31`
- selector blob：
  `d7e588ba76ce18882255e3e22b1f86459ab235dd`
- cohort：24 Episode、384 capture、36 candidate、22 nonempty、2 empty。

### 历史文档

`docs/research-loop/0001/`～`0015/` 使用 `00-context.md` 已记录的 Git tree，
实施、运行和结果提交均必须保持不变。冻结设计只锁定本文件 blob，不锁定整个
`docs/research-loop/0016/` tree，以允许后续新增结果和总结。

## 独立 oracle 边界

### 允许依赖

- Python 标准库；
- NumPy；
- 输入 plan 与 P50 blob bytes。

### 禁止依赖

blind worker source及运行时不得导入或调用：

- `hwr` 任意模块；
- `candidate_mask_ownership`；
- `target_selection`；
- production candidate/merge/score/selection helper；
- P68/P76 evaluator；
- MuJoCo、Torch 或任何模型 runtime。

worker必须独立实现并测试：

1. policy-input binary解析与输入 hash；
2. acquisition/base frame变换；
3. camera back-projection；
4. robot self-mask与 arm chain；
5. v2 local mask-copy anchor scan；
6. raw candidate几何；
7. connected-component merge；
8. top-64 ranking与 canonical order；
9. v2 canonical document；
10. score公式与 `<f8` bytes；
11. `max(score, -index)` tie-break；
12. selected canonical identity。

共享 NumPy线性代数与 Python/JSON/IEEE 编码属于明确的共享可信边界。source-disjoint
表示不复用 production HWR 实现，不表示数学、解释器或第三方库完全独立。

## 两阶段执行

### Phase A：blind worker

1. 父进程先验证固定路径、Git identity、工作区干净和 P50/P79 输入完整性。
2. 父进程从 P50 `capsules.json` 生成净化 plan；不得拷入 candidate/score/selection
   metadata。
3. 父进程创建 output sibling staging；正式 output 或 staging 已存在时 fail closed。
4. 使用当前 `.venv/bin/python` 以绝对 worker script路径启动独立 subprocess：
   - cwd为 staging下独立目录；
   - 清空 `PYTHONPATH`；
   - argv不含 P79 root；
   - environment不含 P79 path、expected hash/index/identity。
5. worker逐个读取 plan列出的 P50 blob，并记录实际 read ledger；不得扫描 repo。
6. worker对全部24 Episode执行两次完整重建，输出两个 canonical blind receipt 与
   boundary-control report。
7. worker输出后退出；父进程验证 exit、schema、source hash、read ledger、资源与
   两次 receipt bit identity。

### Phase B：reveal/comparer

1. 只有 Phase A 成功后，父进程读取 P79 bank与24个 candidate blob。
2. 按 Episode identity严格一一对应，不按路径存在性猜 root。
3. 比较 blind receipt与 P79：
   - candidate canonical bytes/hash/count；
   - `<f8` score bytes SHA-256；
   - selected index；
   - selected canonical identity。
4. 单列 empty、singleton与 multi-candidate Episode；candidate 不是独立样本。
5. 生成 comparison、report、manifest，并在全部门通过后原子 rename staging。

## 冻结负控

正式 runner 或其受测 boundary suite 至少覆盖：

1. P50/P79 schema互换；
2. Episode duplicate/missing/order漂移；
3. capture ordinal gap/duplicate；
4. final input缺失或多个；
5. observation identity漂移；
6. policy/candidate-visible path、size、hash漂移；
7. absolute path、`..` traversal、symlink escape；
8. candidate order mutation；
9. final base pose mutation；
10. score weight mutation；
11. tie-break mutation；
12. selected metadata mutation；
13. canonical-only candidate冒充 full-precision score input；
14. worker import `hwr` 或 production helper；
15. worker argv/env/read ledger出现 P79 root或 expected metadata；
16. blind receipt在退出前非原子或不完整；
17. source、frozen design、历史 tree或正式 artifact漂移。

不能只靠修改顶层 hash使所有 mutation在同一类别失败；语义 mutation必须命中对应
category。测试不得嵌入正式 cohort 的 expected score hash、selected index或 selected
identity。

## 主要指标

- `candidate_exact_match_count / 24`
- `score_hash_exact_match_count / 24`
- `selected_index_exact_match_count / 24`
- `selected_identity_exact_match_count / 22`
- `empty_selection_exact_match_count / 2`
- `blind_rebuild_bit_identical`
- `blind_p79_path_or_metadata_read_count`
- `legacy_v1_generator_call_count`，由禁止导入和 source/runtime审计表示
- `input_file_match_count / 796`
- `mutation_control_pass_count / mutation_control_count`

描述性报告：

- candidate count总数与任务分账；
- singleton/multi-candidate Episode数；
- full-precision top-2 score margin，仅对8个 multi-candidate Episode；
- canonical-only score hash mismatch数；
- worker与 comparer wall/RSS分账。

## 接受与停止门

### `accepted as consumer-local v2 selection-lineage evidence`

必须同时满足：

1. 796/796 P50 input file与 P79 manifest commitment一致；
2. 24/24 Episode、384/384 capture、36 candidate和22/2 nonempty/empty分账一致；
3. candidate bytes/hash/count exact match `24/24`；
4. score hash exact match `24/24`；
5. selected index exact match `24/24`；
6. nonempty selected canonical identity exact match `22/22`；
7. empty selected index/identity exact match `2/2`；
8. 两次 blind receipt bit-identical；
9. worker source/import/argv/env/read ledger不含 P79 metadata读取；
10. 全部预注册 mutation/control通过；
11. source scope、历史 tree、P79 artifact与 P50 input bytes保持不变；
12. 正式 evaluator wall、RSS和artifact预算通过。

允许声明：

> 在当前 checkout 可用、由 P79 manifest 绑定的 P50 capture bytes 和冻结的 P68/P76
> 数据需求上，不读取 P79 score/selection metadata 的 source-disjoint oracle可以
> 重建相同 v2 candidate、score 与 selection lineage；新增 producer receipt不是这
> 两个具体 consumer 的硬前置。

### `rejected`

- 输入、provenance、oracle isolation与预算均有效，但任一 Episode 的 candidate或
  selected index/identity 与 P79不一致；或
- mutation表明 oracle未对 candidate/score/tie-break/selection关系敏感。

### `inconclusive_score_bytes`

- candidate `24/24`、selected index `24/24`、selected identity `22/22`一致；
- 但 exact score hash至少一个不一致；
- 不能把不一致后验解释为平台差异，也不能接受 lineage 假设。

### `invalid`

- P50 source bytes缺失或任一 size/hash漂移；
- P79 path/tree/blob/schema/commitment漂移；
- blind worker接触 P79 metadata；
- oracle复用 production HWR helper或导入 `hwr`；
- path escape、重复/缺失 Episode/capture、非原子输出；
- source/frozen design/history tree漂移；
- 预算超限、正式 output预先存在或留下 partial final output；
- 结果后修改输入、阈值、判定或 mutation集合。

## 守护与声明边界

- `training_executed=false`
- `policy_inference_executed=false`
- `physical_acquisition_executed=false`
- `capability_evaluation_executed=false`
- `candidate_quality_claim_allowed=false`
- `selector_improvement_claim_allowed=false`
- `association_claim_allowed=false`
- `reachability_claim_allowed=false`
- `generalization_claim_allowed=false`
- `hardware_safety_claim_allowed=false`
- `task_success_claim_allowed=false`
- `artifact_self_contained_claim_allowed=false`
- `whole_program_completeness_claim_allowed=false`

P83无论结果如何，都不得授权默认 v2 migration、selector、P68/P76正式物理运行、
Replay、Actor、世界模型训练或 capability evaluation；这些必须在后续轮次重新创新
与筛选。

## 允许文件与提交

唯一实施 Agent只能新增：

- `scripts/evaluate_v2_selection_lineage_oracle.py`
- `src/hwr/apps/evaluate_v2_selection_lineage.py`
- `tests/test_v2_selection_lineage_oracle.py`
- `tests/test_v2_selection_lineage_app.py`

不得修改：

- 任何现有 source/test/config/artifact；
- `docs/research-loop/0001/`～`0015/`；
- P50/P79 input与 artifact；
- `.gitignore`、`pyproject.toml` 或共享 helper。

实施必须压缩为一个原子提交，提交信息引用 `R0001-P83`。正式运行前，主 Agent另行提交
本冻结文档；正式 artifact和结果文档不属于实施提交。

## 测试与门禁

Focused：

```text
.venv/bin/python -m pytest \
  tests/test_v2_selection_lineage_oracle.py \
  tests/test_v2_selection_lineage_app.py \
  tests/test_candidate_mask_ownership.py \
  tests/test_candidate_mask_ownership_app.py
```

Repository gates：

```text
.venv/bin/python scripts/check_python_size.py
.venv/bin/python scripts/check_architecture.py
.venv/bin/python -m compileall -q src tests scripts
git diff --check
```

Broad validation：

```text
.venv/bin/python -m pytest
```

- 冻结基线为 `1186 passed, 1 failed, 11 skipped`。
- 唯一允许的既有 failure：
  `tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete`。
- 实施不得新增 failure；新增测试必须全部通过。
- full pytest与正式 evaluator资源分账，不纳入 P83 evaluator 的 `3min/<2GiB`。

## 正式命令与 artifact

```text
.venv/bin/python -m hwr.apps.evaluate_v2_selection_lineage \
  --p50 runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --p79 runs/research-loop/0014/r0014-p79-candidate-bank-s20267901 \
  --output runs/research-loop/0016/r0016-p83-selection-lineage-s20268301
```

正式 output至少包含：

- `blind-plan.json`
- `blind-receipt-a.json`
- `blind-receipt-b.json`
- `boundary-controls.json`
- `comparison.json`
- `report.json`
- `manifest.json`

正式 output与 `.tmp` 已存在时 fail closed；先写 sibling staging，全部验证与预算通过后
原子 rename。失败不得留下 partial final artifact。

## 资源预算

正式 evaluator：

- CPU-only；
- wall `<=180s`；
- process-tree peak RSS `<=2GiB`；
- artifact `<=16MiB`；
- 最低磁盘余量 `20GiB`；
- 不使用 MuJoCo、Torch、GPU、tmux、后台任务、休眠或 `traex-host-exec`。

验证：

- focused与repository gates前台执行；
- full pytest独立 wall上限 `360s`、process-tree RSS描述性记录，不与 evaluator预算
  混算；
- 该实验不是训练，不得为其休眠或启动看门狗。

## 冻结后顺序

1. 单独提交本文件，形成冻结实验提交。
2. 唯一实施 Agent在独立工作区只修改四个允许文件并形成一个原子提交。
3. 主 Agent审查实现范围与独立性，运行 focused、repository gates和 broad validation。
4. 只有干净、已提交、通过门禁的 source commit才可启动正式 evaluator。
5. 正式 artifact成功后强制加入 Git，单独提交。
6. 更新本轮 `04-results.md`、`05-summary.md`，提交并 push。
7. 本轮不追加其他候选；下一轮重新创新与筛选。
