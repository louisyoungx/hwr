# R0015 轮次总结

## 结论

| ID | 判定 |
|---|---|
| `R0001-P80-E1` | `rejected` |
| `R0001-P81` | `rejected as standalone proposal` |
| `R0001-P82` | `deferred` |
| `R0001-P68-E2` | `deferred` |
| `R0001-P74-E1` | `deferred` |
| `R0001-P76-E2` | `deferred` |
| `R0001-P77-E2` | `rejected in current form` |

本轮没有训练、policy inference、MuJoCo physical acquisition、B0–B7 action、contact
phase、capability evaluation 或新家务任务成功，能力基线不变。

P80 作为工程/evidence-hygiene前置项实施并经过三轮独立红队，但最终 rejected：

- 24/384/24/24/768/28/795/796完整账本、Git receipt、typed双根和大量路径/版本
  mutation均可验证；
- 最终 focused `104 passed`，repository gates全部通过；
- 但 default-version audit仍可被动态拼接 v2 schema绕过；
- architecture fingerprint与实现同源，不能支持通用 consumer completeness声明；
- P79 artifact缺少独立 v2 score/selection ledger；
- full validation process-tree RSS实测约 `2.443GiB`，超过冻结 `2GiB`门。

因此正式 runner没有启动，正式 output与 `.tmp` 都不存在。未把 exploratory、独立审计
或内存 mutation结果冒充正式 artifact。

## 当前基线

- 能力基线不变：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`
  - 24 Episode；
  - 1,600 update；
  - 0 success；
  - action causality aggregate ratio `1.0012791539504238`；
  - Actor未解锁。
- measurement/design baseline保持：
  - `R0001-P79-E1`：isolated v2 mask-ownership correction；
  - v2 bank：
    `runs/research-loop/0014/r0014-p79-candidate-bank-s20267901`。
- P80没有进入新基线。
- legacy v1与隔离 v2仍必须显式区分；不宣称默认 runtime已迁移。
- 不宣称机器人已学会任何家务任务。

## 关键发现

1. P79 v2 bank的双根歧义是真实的：
   - v2 candidate相对P79 root；
   - legacy candidate与capture相对P50 root；
   - 24/24 Episode中的 candidate path同名。
2. 仅靠 candidate document的schema与自带 hash不能恢复producer lineage；typed
   envelope和外部 Git receipt有实际价值。
3. 25类结果前注册的版本、路径、symlink、capture、candidate、producer和Git mutation
   可以在独立fixture中命中具体 fail-closed category，而不是都被顶层hash提前拦截。
4. 但“未来 consumer只能经 resolver”不是当前同源 AST fingerprint能证明的通用性质：
   - fingerprint与被审计app位于同一允许修改文件；
   - 同步改变代码与fingerprint仍是同源自证；
   - 应冻结具体consumer source identity，而不是声称Python whole-program completeness。
5. default generator的静态审计同样存在数据流缺口：
   - dead branch可保留 v1引用；
   - live branch可动态拼接 v2 schema；
   - 当前审计仍可误判通过。
6. P79 bank的selected metadata不是完整selection receipt：
   - bank有v2 score hash；
   - regression没有独立v2 score hash或score bytes；
   - canonical candidate又不足以精确恢复full-precision score；
   - consumer不能独立验证score→selected index关系。
7. 完整pytest当前为：
   - `1186 passed`；
   - `1 failed`；
   - `11 skipped`；
   - 约 `246.77s`；
   - 唯一失败在冻结 `f224149` detached worktree复现。
8. 完整验证的保守process-tree RSS为：
   - self：`36,569,088 bytes`；
   - children：`2,406,711,296 bytes`；
   - upper bound：`2,443,280,384 bytes`；
   - 比冻结2GiB门高 `295,796,736 bytes`。
9. 完整pytest包含本机可用的MPS路径；不能把validation suite整体称为CPU-only。P80
   resolver本身没有执行MuJoCo、训练或科学计算，这与回归验证使用何种设备必须分账。
10. 静态symlink与读前后可见替换可拒绝，但严格路径来源若要抵御并发
    replace/restore，需要基于文件描述符的 `openat/O_NOFOLLOW`，不能只靠lstat快照。

## Git与产物

### 本轮文档提交

- R0015初始化：`59015cc`
- 提案冻结：`0f2a235`
- 独立筛选：`ade8b8e`
- P80实验冻结：`f224149`

### Rejected实现

- 压缩后的单一实现提交：
  `3fff8389645f9cf24a0de2b934eb689de627b9f3`
- 从当前基线回退：
  `788a61c`

实现提交只包含冻结四文件，可由提交引用恢复；当前 branch tree不包含 rejected实现。

### 正式 artifact

无。

以下路径均不存在：

- `runs/research-loop/0015/r0015-p80-artifact-contract-s20268001`
- `runs/research-loop/0015/r0015-p80-artifact-contract-s20268001.tmp`

## 验证

- 最终 focused：`104 passed in 29.67s`。
- Python size：458 files通过。
- architecture：通过。
- compileall：通过。
- `git diff --check`：通过。
- 单实现提交、四文件scope、frozen document ancestry与history tree：通过。
- 确认性default-schema mutation：错误地 `passed=True`，直接触发rejected。
- full pytest：`1186 passed, 1 failed, 11 skipped`。
- 唯一full-pytest失败在冻结worktree复现。
- 独立红队最终意见：因default/architecture completeness与RSS门，不准入正式
  evaluator。

## 下一轮问题

下一轮必须重新创新与筛选，不自动继承本轮候选。优先问题：

1. 不再重提“证明所有未来consumer不会绕过”的通用P80合同。
2. 若继续v2路线，优先考虑producer-side版本化selection receipt：
   - 保存或可独立重建v2 score bytes；
   - 绑定score hash、selected index、full-precision candidate与producer source；
   - 将新的producer artifact与P79历史bank明确版本化，不覆盖旧bank。
3. 对具体P68或P76 consumer，在实验冻结时直接锁定该consumer source blob、imports与
   入口；不要依赖同文件内可同步修改的通用AST fingerprint。
4. 重提P68时必须：
   - 拆分24 Episode route availability与22 nonempty selected-support association；
   - 为unique support coordinates建立独立oracle与ledger；
   - 先做capture-only segmentation的label-blind timing preflight；
   - 不继承旧v1未发布结果。
5. 重提P76时必须：
   - 建立显式v2 prefix bridge；
   - 分开safe-prefix coverage与eligible geometry denominator；
   - coverage不足判`inconclusive_prefix_coverage`；
   - safety-stopped不计outer-envelope negative。
6. 默认production migration、P77 search、selector、Replay、Actor、世界模型训练和
   capability evaluation仍不授权。

## 清理

- 起始可用空间约 `109GiB`；结束前约 `103GiB`。
- 空间变化主要受项目外Android/Gradle并发影响，不能归因于本轮。
- 本轮没有正式artifact、checkpoint、训练输出或partial staging需要清理。
- rejected实现已通过Git revert从当前基线移除，原提交保留可追溯。
- 没有删除当前基线、历史artifact、唯一原始数据、manifest或日志。
- 未唤起清理Agent：没有可获得显著空间且不影响证据的本轮资源。
