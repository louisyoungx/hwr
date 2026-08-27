# R0014 冻结实验

## 冻结状态

- 提案冻结提交：`5966e3f4b99cfd96d8dc403938f0fb4837a3eb65`。
- 评审提交：`41eae6575407263fdcbe1b96667b33bdc2392fd6`。
- 唯一入选：`R0001-P79-E1`，candidate mask ownership correction。
- 本轮不训练、不运行物理 acquisition 或 association classification、不执行 B0–B7。
- 本文件提交后不得修改；runner 必须绑定本文件 blob。

## 假设与单变量

`R0001-P79-H1`：

> `_frame_candidates()` 中 `patch_valid` 共享 frame-level `valid` 的内存，原地 `&=`
> 使后续 anchor 读取被先前 anchor 改写的 mask。仅将 patch 改为独立 ownership，
> 在其他算法不变时，应使 raw support 与 final candidate bytes 对预注册遍历顺序不变。

唯一生产变量：

```text
patch_valid = valid[row - 10 : row + 11, column - 10 : column + 11]
```

变为独立局部 mask。禁止同时修改任何阈值、网格、geometry helper、self mask、merge、
top-64、canonical sort、score 或 selector。

生成语义改变后，candidate schema 从 legacy
`hwr.p41-target-candidates/v1` 版本化为
`hwr.p79-target-candidates/v2`。`target_selection` 必须保留显式 legacy-v1 生成入口供
旧 P68 只读重建；默认生产入口使用 v2。旧 artifact 不覆盖、不重命名、不补写。

## 实施所有权

- 唯一负责人：P79 实施 Agent。
- 分支：`feat/r0014-p79-mask-ownership`。
- worktree：`/private/tmp/hwr-r0014-p79`。
- 允许修改：
  - `src/hwr/eval/target_selection.py`；
  - `src/hwr/eval/candidate_mask_ownership.py`（新）；
  - `src/hwr/apps/evaluate_candidate_mask_ownership.py`（新）；
  - `src/hwr/eval/initial_candidate_association.py`，仅允许把旧 P68 重建显式绑定到
    legacy-v1 generator；
  - `tests/test_candidate_mask_ownership.py`（新）；
  - `tests/test_candidate_mask_ownership_app.py`（新）；
  - `tests/test_candidate_association.py`，仅允许增加 v1/v2 schema 隔离回归。
- 禁止修改：
  - `docs/research-loop/0001/`～`0013/`；
  - 历史 P50/P51/P57/P60/P66/P68 producer、artifact 与合同；
  - ranking、selector、primitive、backend、safety、task config。

实施 Agent 提交一个引用 `R0001-P79` 的原子提交；主 Agent 复核后 cherry-pick。

## 冻结输入

输入：

`runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001`

| 文件 | SHA-256 |
|---|---|
| `capsules.json` | `223cb403def742b82f6c8cfe1916b204b76bb29304e2fc94d64318b58ee74cbf` |
| `plan.json` | `5eec1d65527a837667d2e53d015c8cb38672a42c3856155ae279921c6b6413ab` |
| `report.json` | `b9649a5198bd9a755dd030ea0ef39422c3a5eb09e0a6633e69aa57f5cc87f4d0` |
| `manifest.json` | `cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86` |

- 24 个固定 Episode、384 个 capture identity、768 个 policy/candidate-visible blob。
- 768 项 `(path, sha256, bytes)` canonical ledger SHA-256：
  `ff8c5cf53942e89e5ebc04dd8e9020313e5a120dc62ad6ca8764d93a6eda6145`。
- runner 逐项验证旧 manifest 全部 795 个 artifact 的 size/SHA-256。
- 旧 source commit `d67791a53491ce37cddaef4bd7d6b71ad3e66ac2` 必须为当前
  `HEAD` ancestor。
- legacy defect source 固定为评审提交下的
  `src/hwr/eval/target_selection.py`。runner 以 AST 验证原实现存在
  `patch_valid = valid[...]` 后对该变量执行 `BitAnd` augmented assignment，
  不接受当前源码中的硬编码 claim。

## 预注册遍历与独立 Oracle

anchor 固定为 `rows=range(12,180,4)`、`columns=range(12,244,4)`：

1. `row_major`：row 外层升序，column 内层升序；
2. `reverse_row_major`：完整 row-major anchor sequence 逆序；
3. `column_major`：column 外层升序，row 内层升序。

独立 immutable-mask oracle：

1. 位于新 evaluator 模块，不调用 production `_frame_candidates()`。
2. 可复用纯 geometry helper，但必须以非原地表达式构造
   `valid_slice & local_depth_condition`。
3. 每个 anchor probe 前后记录 parent `valid` SHA-256；mutation count 必须为 0。
4. 每个 frame 的 production row-major raw canonical multiset必须等于 oracle row-major。
5. oracle 三种 traversal 的 raw canonical multiset必须相同。
6. 每个 Episode 对三种 raw sequence分别执行冻结 merge/top-64/canonical reorder，
   final v2 candidate bytes必须相同。
7. 若 raw multiset相同但 downstream reduction 导致 final bytes不同，判
   `inconclusive_secondary_order_dependence`；不得现场修改 merge。

## 缺陷 Fixture 与边界控制

确认性接受门不使用已观察的 `22/24` drift。

预注册 overlap fixture：

- 至少两个重叠 21×21 patch；
- 第一 patch 的 local condition 会清除第二 anchor 所需 valid pixel；
- legacy alias + in-place `&=` 必须使 parent mutation count `>0`，并使正向/逆向
  probe support 或 raw candidate不同；
- corrected independent mask必须使 parent mutation count `=0` 且三 traversal一致。

边界控制：

- 全无效 frame；
- 全平坦 frame；
- 单个可接受 anchor；
- 重复 observation identity；
- nonfinite depth；
- input `head_depth_valid` 在 production调用前后 byte-identical。

fixture不能复现 legacy顺序差异时实验 `invalid`，不得只凭 AST 接受。

## 版本化 Candidate Bank

正式输出：

`runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`

只读旧 P50 policy-visible capture bytes，不运行 MuJoCo、不采样新 Episode：

1. 非 final capture作为 keyframes，最后一个作为 final input；
2. corrected generator生成 v2 candidate-set；
3. 未修改的 score/selector生成 selected index；
4. 输出：
   - `bank.json`：24 Episode、新 identity/count/index/score hash与旧 bank identity；
   - `regression.json`：旧/新 paired描述账本；
   - `blobs/<episode>/candidate-set.json`：v2 canonical bytes；
   - `report.json`、`manifest.json`。

不复制旧 policy/candidate-visible blob；manifest 用 repo-relative path、size与 SHA-256
绑定输入。bank schema 为 `hwr.p79-candidate-bank/v1`，candidate document 必须为
`hwr.p79-target-candidates/v2`。至少一个负测试证明旧 P68 v1 consumer 对新 schema
fail closed；同时旧 P68 对旧 v1 bank 的历史重建继续通过。

## 指标与判定

确认性主要指标：

- legacy AST 与 overlap fixture：通过；
- corrected parent mask mutation count：`0`；
- 384/384 frame production row-major 等于 oracle；
- 384/384 frame 三 traversal raw multiset相同；
- 24/24 Episode 三 traversal final bytes相同；
- 两次 bank build canonical bytes bit-identical；
- 24 个 Episode与旧 bank一一对应，无替换、无新增。

守护指标：

- 所有输入 hash通过；
- 384 capture identity 与 768 input blob全部绑定；
- score/selector source未修改；
- 历史 artifact byte diff为 0，历史文档 tree不变；
- workspace干净、source commit等于 `HEAD`、冻结文档 blob匹配；
- generator不新增语义或 simulator-private输入；
- training、policy inference、physical acquisition、capability evaluation均为 false；
- Python size、architecture、focused tests与完整 pytest通过；仅允许记录在起始提交可复现
  的无关既有失败。

描述性报告但不参与接受门：

- old/new hash changed Episode count；
- candidate count增减/不变；
- selected canonical identity changed count；
- empty→nonempty、nonempty→empty；
- 每任务 paired分账；
- 总 candidate count差值。

全部确认性指标与守护通过：

`accepted as deterministic candidate-generator correction`

corrected production或 oracle raw support仍受 traversal影响：

`rejected`

raw multiset一致但 downstream final bytes受顺序影响：

`inconclusive_secondary_order_dependence`

legacy fixture失败、parent仍被修改、production与 oracle不一致、provenance/history漂移、
越界修改、覆盖旧 artifact、新 bank误标 v1或旧 P68静默接受 v2：

`invalid`

## 命令与预算

```text
.venv/bin/python -m pytest \
  tests/test_candidate_mask_ownership.py \
  tests/test_candidate_mask_ownership_app.py \
  tests/test_target_selection.py \
  tests/test_candidate_acquisition.py \
  tests/test_candidate_funnel.py
```

```text
.venv/bin/python -m hwr.apps.evaluate_candidate_mask_ownership \
  --input runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --output runs/research-loop/0014/r0014-p79-candidate-bank-s20267901
```

- wall time：10min；
- peak RSS：4GiB；
- artifact：25MiB；
- 最低磁盘余量：20GiB；
- CPU-only，无 MuJoCo rollout、训练或 GPU 独占要求；
- 小型正式评测，前台等待，不休眠。

## 收尾停止规则

1. 本轮只实施、验证和运行 P79-E1。
2. 即使 accepted，也不追加 P74、P68、P76、P77、selector 或训练。
3. 结果提交后更新 `04-results.md`、`05-summary.md`，提交并 push。
4. 下一轮基于 v2 bank重新创新和筛选，不自动继承候选。
