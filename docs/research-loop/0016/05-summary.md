# R0016 轮次总结

## 结论

| ID | 判定 |
|---|---|
| `R0001-P83-E1` | `accepted as consumer-local v2 selection-lineage evidence` |
| `R0001-P84` | `deferred` |
| `R0001-P85` | `rejected in current form` |
| `R0001-P68-E3` | `deferred` |
| `R0001-P76-E3` | `deferred` |
| `R0001-P76-E4` | `deferred` |
| `R0001-P86` | `rejected in current form` |
| `R0001-P77-E3` | `not eligible` |

本轮没有训练、policy inference、MuJoCo physical acquisition、B0–B7 action、contact
phase、capability evaluation 或新家务任务成功，能力基线不变。

P83 接受为**当前冻结 P68/P76 consumer 的 v2 selection-lineage 证据**：

- 在看不到 P79 score hash、selected index 与 selected identity 的 Phase A 中，
  source-disjoint blind oracle 从 P50 policy-visible capture bytes 独立重建；
- candidate bytes/hash/count `24/24`；
- full-precision score hash `24/24`；
- selected index `24/24`；
- nonempty selected identity `22/22`；
- empty selection `2/2`；
- 两次完整 blind rebuild bit-identical；
- 29/29 真实 mutation/control 通过；
- 796/796 P50 manifest-bound input、384 capture与768 blind blob完整。

因此新增 producer receipt不再是这两个具体 consumer的共同硬前置。P84仍可作为未来
producer provenance增强，但不能继续阻塞 P68/P76。

## 当前基线

- 能力基线不变：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`
  - 24 Episode；
  - 1,600 update；
  - 0 success；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor未解锁。
- measurement/design baseline新增：
  - `R0001-P83-E1`：consumer-local v2 selection-lineage evidence；
  - artifact：
    `runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`。
- 已有 v2 generator baseline保持：
  - `R0001-P79-E1`：isolated v2 mask-ownership correction；
  - bank：
    `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`。
- P83没有修改或迁移默认 runtime，不改变candidate、score、selector、动作、安全或训练。
- legacy v1与isolated v2仍必须显式区分。
- 不宣称机器人已学会任何家务任务。

## 关键发现

1. P79 canonical candidate document不是完整score receipt：
   - 22/22 nonempty Episode从量化canonical record重算的exact score hash均不同；
   - 两个empty Episode的empty score hash自然一致。
2. 但P50 capture bytes仍可用时，独立oracle能精确恢复24/24 full-precision score hash和
   selection；因此“exact score不在P79 bank”不等于“具体P68/P76无法重建selection”。
3. 8个multi-candidate Episode的最小top-2 margin为
   `0.012365174520925004`；当前selection不是由canonical-only score receipt证明，
   而是由full-precision capture reconstruction证明。
4. P50 source bytes当前全部存在并受P79 manifest绑定，但受`.gitignore`排除：
   - P83不使P79 artifact自包含；
   - 输入缺失时后续consumer仍必须fail closed；
   - Git commitment不能替代原始bytes。
5. source-disjoint声明经过三轮实现审计才成立：
   - 初版因逐字同源、worker可见整个P50 root、自报read ledger和假mutation被拒；
   - 第二版因实际worker bytes未绑定、尾部RSS和formal-scale测试不足继续被拒；
   - 最终版固定worker blob、执行staging副本、构建closed-world blind root、
     使用`-I -S`、逐级目录FD、source前后核验与正式规模测试后获准运行。
6. 最终worker相对production source：
   - whole token ratio `0.35069902400422054`；
   - whole AST ratio `0.32482346550787616`；
   - 最大主要函数ratio `0.7448275862068966`。
7. P85当前设计被拒：
   - 6 sentinel遗漏action latency；
   - capture count不是eager per-observation segmentation的wall-time上界；
   - 下一轮若重提，应覆盖12个task/observation/action cell或建立可验证的组成上界。
8. P86当前设计被拒：
   - 三步hold只能覆盖latency queue flush；
   - 不足以证明非零动作、predictive rejection、contact/task counter和multi-clone隔离。
9. P68-E3与P76-E3都不自动继承：
   - P68路线需重新冻结有效执行预算门与unique-coordinate association；
   - P76路线需重新冻结authoritative-prefix bridge与safe-prefix coverage；
   - 两者必须在下一轮重新创新和独立筛选。
10. fixed-base outer-envelope仍只是瞬时必要条件；不得外推为free-base动态不可达。

## 声明边界

P83允许声明：

> 在当前checkout可用、由P79 manifest绑定的P50 capture bytes和冻结的P68/P76数据需求
> 上，不读取P79 score/selection metadata的source-disjoint blind oracle精确重建了
> 相同v2 candidate、score与selection lineage；新增producer receipt不是这两个具体
> consumer的硬前置。

P83不允许声明：

- P79 artifact自包含或可从Git独立恢复；
- 任意未来consumer不会绕过合同；
- Python执行路径是恶意代码安全沙箱；
- candidate与任务实体相关；
- candidate可达、安全、可被控制或能完成家务；
- score/selector质量改善；
- 默认runtime已迁移到v2；
- P68/P76、selector、Replay、Actor或训练已获授权。

## 提交与产物

### 文档

- R0016初始化：
  `e74c82003a1d94ab6a46cd53d3229158fed79ea8`
- 提案冻结：
  `315d68482b6cd97630cce007091f2336db0bdba6`
- 独立筛选：
  `84665c9f1b6419134e41343c053dcfb7c57d482f`
- 实验冻结：
  `2e9eb1d0426749b1cd6239a5982ffd19e5c422fd`

### 实现

- 单一实现提交：
  `b39b9fb07085b0067512a27ad98ba95c64459f06`
- 只新增冻结四文件，没有修改legacy v1、P79、shared selector、runtime或安全层。

### Artifact

- 路径：
  `runs/research-loop/0016/r0016-p83-selection-lineage-s20268301`
- 固化提交：
  `4c172cd64dad7b8878e77dffe1163eebc524359a`
- report：
  `0f09944aadb052d8b92f0b0c54cd40fb31f4835fdab1c44d2f24ef48fffd2513`
- manifest：
  `f149a586e30cb8d29156c685e084507b6a9adf490786993ffbaba589a04bd565`
- 两次blind receipt：
  `e7853a3c6b99e3bb47c41b9bed8d28ce8eb7a01a352aaab4e6e66289e1263ca8`

## 验证

- focused：`68 passed in 34.31s`。
- Python size：458 files通过。
- architecture、compileall、`git diff --check`：通过。
- 单实现提交、四文件scope、frozen document ancestry、history tree：通过。
- 最终独立审计：0 blocker、0 major、0 minor，允许正式运行。
- 正式结果：
  `accepted as consumer-local v2 selection-lineage evidence`。
- 正式wall upper bound：`24.925580708004418s`。
- 正式process-tree RSS upper bound：`1,365,606,400 bytes`。
- 正式artifact约`1.4MiB`。
- 完整pytest：
  - 冻结commit真实起点：
    `1115 passed, 1 failed, 11 skipped`；
  - 最终：
    `1149 passed, 1 failed, 11 skipped`；
  - 新增通过节点34，新增失败0；
  - 唯一failure ID不变。
- `03-experiment.md`中的`1186 passed`来自已回退P80实现仍存在时的R0015快照；
  本轮没有修改冻结文档或后验改变P83接受门。

## 下一轮问题

下一轮必须重新创新和筛选，不自动继承本轮候选。优先问题：

1. 在association与feasibility两条路线中重新选择一个主要瓶颈，不得同轮捆绑。
2. 若重提P68：
   - 先重做P85预算设计，覆盖12个task/observation/action latency cell，或建立可验证
     wall上界；
   - Phase A/B保持进程级label隔离；
   - 以unique
     `(Episode, capture ordinal, row, column)`为candidate内票据单位；
   - route `22/24`与selected relevance `22`分母分账；
   - 不复用legacy未发布P68结果。
3. 若重提P76：
   - 建立P50 acquisition到v2 selected candidate再到authoritative B2-entry的完整bridge；
   - 固定22个nonempty Episode分母；
   - safety-stopped、terminal、input-invalid与safe-entry完整分账；
   - coverage不足判`inconclusive_prefix_coverage`；
   - 禁止生成或执行B2 action。
4. P76-E4只有在P76-E3 accepted后才可重提：
   - compiled-model导出保守链长；
   - safety-stopped不进入geometry denominator；
   - fixed-base exclusion不外推free-base动态不可达。
5. P84仅作为可选producer provenance增强，不再是P68/P76共同硬前置。
6. P77仍为no-go；必须先有明确association、prefix、geometry、restore和bounded
   positive witness门。
7. selector、default v2 migration、Replay、Actor、世界模型训练与capability
   evaluation仍不授权。

## 清理

- 清理前可用空间：`92,630,876 KiB`。
- 删除：
  - `/private/tmp/hwr-r0016-baseline`，`228,556 KiB`；
  - detached worktree `/private/tmp/hwr-r0016-baseline-wt`，
    HEAD `2e9eb1d0426749b1cd6239a5982ffd19e5c422fd`，`12,064,672 KiB`。
- 清理后可用空间：`92,926,840 KiB`。
- 实际释放：`295,964 KiB`，约`289.0MiB`；APFS clone的逻辑大小不等于物理释放量。
- 两个临时路径均已不存在，detached worktree已从`git worktree list`移除。
- 未删除当前基线、P50/P79输入、P83正式artifact、唯一原始数据、checkpoint、manifest、
  日志或其他worktree。
