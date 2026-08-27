# R0014 结果

## 结果总表

| ID | 判定 | 结论边界 |
|---|---|---|
| `R0001-P79-E1` | `accepted as deterministic isolated v2 candidate-generator correction` | 修复后的独立 v2 generator 对原始 validity mask 无副作用，并在三种遍历下生成相同 raw multiset 与 candidate bytes；不代表默认 runtime 已切换到 v2 |
| `R0001-P73` | `rejected as standalone remedy` | capture-only segmentation 正确但探索加速不足 |
| `R0001-P74` | `deferred` | 仅在 v2 bank 上重新冻结 association 执行设计后考虑 |
| `R0001-P75` | `rejected` | 扩大旧-bank预算不解决 generator 根因 |
| `R0001-P76` | `deferred` | 应在 v2 candidate/continuation lineage 上重新设计 |
| `R0001-P77` | `rejected in current form` | 搜索自由度与 negative 解释不充分 |
| `R0001-P78` | `deferred` | P61 完整性修复真实但当前无直接 consumer |

本轮没有训练、参数更新、checkpoint、policy inference、MuJoCo physical acquisition、
B0–B7 action、contact phase、capability evaluation 或新家务任务成功。

## `R0001-P79-E1`

### 最终实现边界

最终 source commit：

`9eef9953f8a8558228a5e8870d7d2d8f7499ee1e`

关键边界：

1. 共享 `src/hwr/eval/target_selection.py` 恢复并保持冻结 blob：
   `d7e588ba76ce18882255e3e22b1f86459ab235dd`。
2. legacy 默认入口仍为 `hwr.p41-target-candidates/v1`，旧 P50/P51/P57/P60/P68
   consumer 继续使用 v1。
3. 修复后的 generator 隔离在
   `src/hwr/eval/candidate_mask_ownership.py`：
   - `generate_candidate_set_v2()`；
   - `_frame_candidates_v2()`；
   - schema：`hwr.p79-target-candidates/v2`。
4. v2 相对冻结 v1 的生成算法只有两个结果前冻结的差异：
   - `patch_valid` 持有独立 copy，不再原地修改 frame-level `valid`；
   - candidate schema 版本化为 v2。
5. threshold、anchor grid、geometry、self mask、merge、top-64、canonical sort、
   score 与 selector 均保持冻结语义。
6. v2 当前只进入 P79 evaluator 和离线 bank，不代表默认 production/runtime pipeline
   已迁移。

### 正式命令

```text
.venv/bin/python -m hwr.apps.evaluate_candidate_mask_ownership \
  --input runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --output runs/research-loop/0014/r0014-p79-candidate-bank-s20267901
```

### 确认性结果

全部冻结确认性门通过：

- legacy source AST：
  - `patch_valid = valid[...]` slice alias 存在；
  - 后续 `patch_valid &= ...` 原地写入存在；
  - assignment 先于 mutation。
- overlap fixture：
  - legacy parent mutation count `>0`；
  - legacy traversal support 不一致；
  - corrected parent mutation count `=0`；
  - corrected 三种 traversal 一致。
- 正式 cohort：
  - 24/24 Episode；
  - 384/384 capture frame；
  - 1,152 个 frame-traversal；
  - 2,806,272 个 probe 的 before/after hash commitment；
  - parent mask mutation count 全部为 0；
  - production v2 row-major 与独立 immutable oracle raw multiset 全部一致；
  - row-major、reverse-row-major、column-major raw multiset 全部一致；
  - 24/24 Episode 的三种 traversal final candidate bytes 全部一致；
  - 两次完整 bank build 的 `bank.json`、`regression.json` 和 24 个 candidate blob
    bit-identical；
  - 24 个 Episode 与旧 bank 一一对应，无替换、无新增。
- provenance：
  - 795 个旧 P50 manifest artifact 全部验证 size/SHA-256；
  - 384 个 capture identity、768 个 policy/candidate-visible blob 全部绑定；
  - capture ledger SHA-256：
    `ff8c5cf53942e89e5ebc04dd8e9020313e5a120dc62ad6ca8764d93a6eda6145`；
  - P51、P57、P60、P66 tracked artifact tree 保持冻结；
  - 历史 `docs/research-loop/0001/`～`0013/` 未修改；
  - 正式运行前工作区干净，source commit 等于 `HEAD`。

因此：

`accepted as deterministic isolated v2 candidate-generator correction`

允许声明：

> 对冻结 P50 policy-visible captures，独立 v2 generator 消除了 local patch 对 parent
> validity mask 的扫描顺序副作用；修复后的 raw support 与 final candidate bytes 在
> 三种预注册遍历下确定一致，且版本化 v2 bank 已生成。

不得声明：

- 默认 runtime 或 production candidate pipeline 已切换到 v2；
- candidate 更正确、更相关或更可交互；
- selector、路径规划、reachability、安全、闭环成功或泛化改善；
- 24 个旧 Episode 是修复后新鲜未见分布 cohort。

### 描述性 paired 结果

以下只描述旧 v1 与隔离 v2 对同一 24 Episode 的差异，不参与接受门：

| 指标 | 结果 |
|---|---:|
| candidate-set hash 改变 | 24/24 |
| selected canonical identity 改变 | 22/24 |
| candidate count 增加 | 7/24 |
| candidate count 减少 | 9/24 |
| candidate count 不变 | 8/24 |
| empty → nonempty | 3/24 |
| nonempty → empty | 0/24 |
| 总 candidate count | 39 → 36 |

任务分账：

| Task | hash 改变 | selected 改变 | count 增 | count 减 | count 不变 | empty→nonempty | 总数差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| living | 8/8 | 6/8 | 2 | 4 | 2 | 2 | -4 |
| dining | 8/8 | 8/8 | 0 | 3 | 5 | 0 | -3 |
| kitchen | 8/8 | 8/8 | 5 | 2 | 1 | 1 | +4 |

这些结果证明旧 bank 的 candidate identity 对扫描副作用高度敏感，因此旧 P68/P76
不能授权 v2 pipeline 的 selector 或 routing 决策。

## 正式运行时序

1. source `3003095` 的串行双重构在第二次 build 触发冻结 `10min` hard gate：
   - 非零退出；
   - 无 output 或 `.tmp`；
   - 判定为执行预算失败，不是科学结果。
2. source `62cec9c` 加入 Episode 级进程并行，算法输出稳定，但独立审计发现：
   - 初版 v2 schema 改动共享 `target_selection.py`，破坏旧 P60 v1 reader；
   - RSS 只统计 parent；
   - artifact 被删除，未进入 Git。
3. source `d4d49dc` 补了 v1 reader、pytest receipt 与进程内存证据，但独立审计进一步
   指出共享 frozen-source guard 被改动，不能把 P51/P60 报警列为既有失败：
   - preliminary bank/regression 内容 hash 与最终相同；
   - preliminary report：
     `d0c2cf202d524a6b299014da9f2c6f2344523255928cf3070318a4fa395ba069`；
   - preliminary manifest：
     `2f4a77521ad9e1221b3169d1c8da0004e240a37f8b3a87edae9229f2b3c34f0a`；
   - artifact 被删除，未进入 Git。
4. source `9eef995` 将 v2 完整隔离到 P79 模块，恢复共享 frozen blob，并重新正式运行：
   - P51/P60 frozen-source guard 通过；
   - pytest 只保留唯一在真正 `61d85cd` detached cwd 可复现的既有失败；
   - 最终 artifact 原子写入并提交。

## 验证

### 完整 pytest

正式 runner 保存原始输出 `pytest-output.txt`，结果：

- `1115 passed`；
- `11 skipped`；
- `1 failed`；
- 18 个既有 deprecation warnings。

唯一失败：

`tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete`

该失败在干净 detached `61d85cd` worktree 的真实 cwd 下复现。P79 新增失败为 0。

原始 pytest 输出 SHA-256：

`9ea09269023dfab495a4dffe7287ae70115fb83c4e6a7872136b334abe9a03cf`

### 其他门禁

- P79 focused、P68 history、P60 selected-record、P51/P60 provenance guard：通过。
- Python size：454 files 通过，file `<=800` lines、function `<=200` lines。
- architecture、compileall、`git diff --check`：通过。
- `pytest-output.txt` 保留 pytest 原始尾随空格，故 artifact 提交时仅该原始证据文件
  排除 `git diff --check`；其 SHA 与 manifest/receipt 一致。
- 独立最终审计：0 blocker；0 未解决 major；minor 仅为文档和 Git 收尾，现已处理。

## 资源

- 正式 wall time：`297.7296477500022s`；
- report analysis wall time：`295.0390392090012s`；
- bank build 1 child RSS sum：`1,393,704,960 bytes`；
- bank build 2 child RSS sum：`1,352,171,520 bytes`；
- parent peak RSS：`60,948,480 bytes`；
- bank process-tree 保守上界：`1,454,653,440 bytes`，低于 4GiB 门；
- pytest child peak：`1,612,333,056 bytes`，只作 validation 描述，不计入 bank-only
  实验 RSS 门；
- manifest artifact：28 项，合计 `1,946,793 bytes`；
- 磁盘目录：约 `2.7MiB`；
- 正式运行没有 GPU、MuJoCo rollout、训练、policy inference 或后台进程。

## Artifact

目录：

`runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`

关键 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `bank.json` | `888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e` |
| `regression.json` | `1f2f94c39c6f9b8799ec1f3c1def9ce049571fe1edf7bc4dd5a4567e0dbd3582` |
| `pytest-output.txt` | `9ea09269023dfab495a4dffe7287ae70115fb83c4e6a7872136b334abe9a03cf` |
| `report.json` | `13913e80070ff415c895f78a78a5210e27611f682a64cd9904756600aa62db6e` |
| `manifest.json` | `162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9` |

artifact 固化提交：

`93ea4e7afad8c52d83abd54f41a2d08d40a3cab4`
