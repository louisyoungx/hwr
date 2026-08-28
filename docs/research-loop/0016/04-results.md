# R0016 结果

## 结果总表

| ID | 判定 | 结论边界 |
|---|---|---|
| `R0001-P83-E1` | `accepted as consumer-local v2 selection-lineage evidence` | 当前 checkout 的 P50 冻结 bytes 可用时，source-disjoint blind oracle 在不读取 P79 score/selection metadata 的路径上精确恢复 24/24 v2 candidate、score hash与selected index；不证明 artifact自包含、任意consumer完备或candidate质量 |
| `R0001-P84` | `deferred` | 原子 producer receipt仍有 provenance价值，但不再是当前 P68/P76 的共同硬前置 |
| `R0001-P85` | `rejected in current form` | 6-sentinel设计遗漏 action latency，capture count也不是 eager per-observation segmentation 的wall-time上界 |
| `R0001-P68-E3` | `deferred` | unique-coordinate association科学价值高，但需重新设计执行预算门并在下一轮独立筛选 |
| `R0001-P76-E3` | `deferred` | authoritative-prefix bridge仍需独立轮次实施与物理coverage评测 |
| `R0001-P76-E4` | `deferred` | 严格依赖 P76-E3 accepted，不能用 acquisition-base描述性几何替代 |
| `R0001-P86` | `rejected in current form` | 三步hold不足以证明主动搜索所需的full-runtime continuation完整性 |
| `R0001-P77-E3` | `not eligible` | 具体association、prefix、geometry、restore与bounded-witness门尚未全部满足 |

本轮没有训练、参数更新、checkpoint、policy inference、MuJoCo physical acquisition、
B0–B7 action、contact phase、capability evaluation 或新家务任务成功。

## `R0001-P83-E1`

### 冻结目标

P83 检验：

> 在当前 checkout 可用、由 P79 manifest 绑定的 P50 capture bytes 和冻结的 P68/P76
> 数据需求上，不读取 P79 score/selection metadata 的 source-disjoint oracle是否能
> 重建相同 v2 candidate、score 与 selection lineage。

若全部门通过，只允许把新的 producer receipt 从这两个具体 consumer 的共同硬前置降为
可选 provenance增强；不得声称任意未来 consumer完备、artifact自包含、candidate相关、
可达、安全或具有家务能力。

### 最终实现

最终单一实现提交：

`b39b9fb07085b0067512a27ad98ba95c64459f06`

只新增冻结允许的四个文件：

- `scripts/evaluate_v2_selection_lineage_oracle.py`
- `src/hwr/apps/evaluate_v2_selection_lineage.py`
- `tests/test_v2_selection_lineage_oracle.py`
- `tests/test_v2_selection_lineage_app.py`

实现包含：

1. 独立 binary policy-input parser、candidate-visible serializer、v2 anchor scan、
   self-mask、surface fit、union-find merge、ranking、canonical encoding、score 与
   tie-break；
2. worker source与 production source 的结构差异门：
   - whole token ratio `0.35069902400422054`；
   - whole AST ratio `0.32482346550787616`；
   - 最大主要函数 ratio `0.7448275862068966`；
3. 固定 worker SHA-256：
   `714e8ffb8eeb1c28dee83dc5c687694e23e50d36c414c2d604f4a9eca41387a6`；
4. 父进程先验证 P50/P79 provenance，再物化只含净化 plan 与 768 个允许 blob 的
   closed-world blind root；
5. 由父进程验证 worker bytes，写入只读 staging副本，并用 `-I -S` 启动；隔离环境中
   `hwr` 不可导入；
6. worker只消费 staging blind root，不接触原 P50 root、P79 root、candidate-set、
   capsules、manifest、score hash或selected metadata；
7. app与worker都使用目录 FD、逐级 `dir_fd`、`O_NOFOLLOW`、device/inode/size和读前后
   identity完成稳定读取；
8. 两次完整 blind rebuild；worker退出并原子封存 receipt后，父进程才揭盲 P79；
9. 15 类 plan/path/symlink真实 fail-closed mutation、5 类算法 mutation、9 类隔离/
   source/selection control；
10. worker、app和 staging副本在运行前后重新核验 source identity；
11. 最终 wall/RSS使用保守上界，rename后再次测量并校验；失败会删除 final output；
12. claim flags全部为 false。

### 独立红队过程

实现没有一次性通过。主 Agent在正式运行前组织独立只读审计，并两次退回同一唯一实施
Agent修订；正式 evaluator在最终审计放行前从未启动。

#### 第一版 blocker

初版 focused为 `63 passed`，完整pytest为 `1144 passed, 1 failed, 11 skipped`，但独立
审计发现：

1. oracle与production函数结构高度同源，不能支持source-disjoint声明；
2. worker仍可见整个P50 root，`PYTHONPATH=""`不足以隐藏editable-installed `hwr`；
3. read ledger与“零禁止读取”主要由worker自报；
4.多数正式mutation只是检查有效输入布尔值，没有对实际流水线执行mutation；
5. RSS/wall在尾部serialization/rename前采样，存在低报；
6. symlink/path读取没有绑定打开后的FD；
7. focused fixture由production helper生成expected，且未覆盖正式24-job路径。

正式运行因此被阻止。

#### 第二版残余

第二版 focused为 `66 passed`，已解决source同源与真实mutation问题，但独立复审仍发现：

1. read audit仍是worker自报，不能作为主要信任基础；
2. 实际执行的worker路径未与已审计bytes原子绑定；
3. source和workspace在worker完成后没有完整二次核验；
4. finalized budget未重采样父进程尾部RSS；
5.路径祖先没有逐级FD锁定；
6. 24-job测试只有72 capture/144 blob，未覆盖正式384/768规模。

正式运行再次被阻止。

#### 最终版

最终版将read audit降为辅助证据，把主要信任基础改为：

- fixed worker blob；
- 父进程验证并执行只读 staging副本；
- closed-world blind root；
- `-I -S` 且 `hwr`不可导入；
- source AST与相似度门；
- source前后稳定性；
- 逐级目录FD与file FD identity；
- parent-observed resource上界；
- 24 Episode、384 capture、768 blob的临时规模测试。

最终独立复审结论：

- 原三个 blocker全部resolved；
- 原三个 major全部resolved；
- 新 blocker `0`；
- 新 major `0`；
- 新 minor `0`；
- 可放行正式 evaluator。

### 正式命令

source commit：

`b39b9fb07085b0067512a27ad98ba95c64459f06`

命令：

```text
.venv/bin/python -m hwr.apps.evaluate_v2_selection_lineage \
  --p50 runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001 \
  --p79 runs/research-loop/0014/r0014-p79-candidate-bank-s20267901 \
  --output runs/research-loop/0016/r0016-p83-selection-lineage-s20268301
```

运行前：

- 工作区干净；
- source commit等于 `HEAD`；
- output与 `.tmp` 均不存在；
- 796/796 P50输入与P79 manifest匹配；
- P79 tree、producer blob、selector blob、P68/P76 consumer blob和历史tree均匹配；
- focused、repository gates、完整pytest和最终独立审计已完成。

### 正式结果

全部冻结主要门通过：

| 指标 | 结果 |
|---|---:|
| Episode | 24/24 |
| capture | 384/384 |
| blind input blob | 768/768 |
| P50 manifest-bound input | 796/796 |
| final candidate | 36 |
| candidate bytes/hash/count exact | 24/24 |
| score hash exact | 24/24 |
| selected index exact | 24/24 |
| nonempty selected identity exact | 22/22 |
| empty selection exact | 2/2 |
| blind rebuild bit-identical | true |
| mutation/control | 29/29 |
| legacy v1 generator call | 0 |
| private truth read | 0 |
| early P79 metadata read | 0 |

候选分账：

- empty：2 Episode；
- singleton：14 Episode；
- multi-candidate：8 Episode；
- living：8 Episode、7 candidate；
- dining：8 Episode、9 candidate；
- kitchen：8 Episode、20 candidate。

8 个 multi-candidate Episode的 full-precision top-2 score margin：

```text
0.04506789295459246
0.16988405231644155
0.07752429052170773
0.06661504556516984
0.0337253156855174
0.1430379701657185
0.03652572409903099
0.012365174520925004
```

最小 margin为 `0.012365174520925004`。

### Canonical-only反例

只从量化后的 candidate canonical record重算 score：

- 两个 empty Episode的 empty score hash自然一致；
- 22/22 nonempty Episode的 exact score hash均不一致；
- 正式指标 `canonical_only_score_hash_mismatch_count=22`。

因此 P80/R0015指出的事实仍成立：

> P79 candidate canonical bytes不足以恢复原始 full-precision score bytes。

P83新增的证据是：

> 当冻结的 P50 policy-visible capture bytes仍可用时，source-disjoint oracle可以从这些
> bytes重新生成 full-precision candidate，并精确恢复24/24 score hash与selection。

两者不矛盾。P83不能把P79 bank改写成self-contained artifact。

### 判定

`accepted as consumer-local v2 selection-lineage evidence`

允许声明：

> 在当前checkout可用、由P79 manifest绑定的P50 capture bytes和冻结的P68/P76数据需求
> 上，不读取P79 score/selection metadata的source-disjoint blind oracle精确重建了
> 相同v2 candidate、score与selection lineage；因此新增producer receipt不是这两个
> 具体consumer的硬前置。

不得声明：

- P79 artifact自包含或可从Git独立恢复；
- 任意未来consumer都不会绕过合同；
- Python执行路径是恶意代码安全沙箱；
- candidate与任务实体相关；
- candidate可达、安全、可被控制或能完成家务；
- score/selector质量改善；
- 默认runtime已迁移到v2；
- P68/P76已获准在本轮继续运行。

## 验证

### Focused

最终：

`68 passed in 34.31s`

### Repository gates

- Python size：458 files通过，file `<=800` lines、function `<=200` lines；
- architecture：通过；
- compileall：通过；
- `git diff --check`：通过；
- 实现提交数：1；
- changed file集合精确等于冻结四文件；
- `docs/research-loop/0001/`～`0015/` tree保持不变。

### 完整 pytest 与冻结记录误差

冻结 `03-experiment.md` 引用了 R0015 P80实现仍存在时的：

`1186 passed, 1 failed, 11 skipped`

但 R0015 结束前已用 `788a61c` 回退 P80四文件。主 Agent在正式运行前使用真正 detached
`2e9eb1d` worktree，并补入同一 ignored `runs/` bytes，得到实际起点：

`1115 passed, 1 failed, 11 skipped`

最终 `b39b9fb`：

`1149 passed, 1 failed, 11 skipped`

差分：

- 新增通过节点：34；
- 新增失败：0；
- skipped变化：0；
- failure ID不变：
  `tests/test_entity_candidate_mapping.py::test_frozen_document_history_and_input_provenance_are_complete`。

最终完整pytest：

- pytest wall：`309.51s`；
- 外部 real：`311.86s`；
- maximum resident set size：`1,410,433,024 bytes`；
- 18个既有deprecation warnings。

冻结文档没有后验修改。`1186` 被记录为起点inventory误差，不改变P83正式接受门、输入、
threshold、failure-set guard或结果。

## 资源

正式 evaluator：

- report wall upper bound：`24.925580708004418s`；
- command finalized wall：`24.127305166999577s`；
- comparer wall：`2.950475000005099s`；
- process-tree peak RSS upper bound：`1,365,606,400 bytes`；
- wall门：`180s`；
- RSS门：`2,147,483,648 bytes`；
- artifact门：`16MiB`；
- 最低磁盘门：`20GiB`；
- output实际约 `1.4MiB`，7个文件；
- 无MuJoCo、Torch、GPU、tmux、后台任务、休眠或host-exec。

## Artifact

目录：

`runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`

固化提交：

`4c172cd64dad7b8878e77dffe1163eebc524359a`

| 文件 | SHA-256 |
|---|---|
| `blind-plan.json` | `7763bcb52fb00a71e70797adede68ed67edf95816b4005872f68e4351981e25d` |
| `blind-receipt-a.json` | `e7853a3c6b99e3bb47c41b9bed8d28ce8eb7a01a352aaab4e6e66289e1263ca8` |
| `blind-receipt-b.json` | `e7853a3c6b99e3bb47c41b9bed8d28ce8eb7a01a352aaab4e6e66289e1263ca8` |
| `boundary-controls.json` | `100d7963a5abf39b8556339725239b7d083c874ff3b36374313e903ff5be6971` |
| `comparison.json` | `f5e2716267af11023fc8d9526ffdc8ca63290cdb94a3a49ff11c5f42c52605b9` |
| `report.json` | `0f09944aadb052d8b92f0b0c54cd40fb31f4835fdab1c44d2f24ef48fffd2513` |
| `manifest.json` | `f149a586e30cb8d29156c685e084507b6a9adf490786993ffbaba589a04bd565` |

manifest中6个pre-manifest artifact的size/hash全部通过独立复核；实际文件集合精确等于
这6项加 `manifest.json`，没有 `.tmp` 或partial output。
