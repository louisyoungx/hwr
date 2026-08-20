# R0006 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P39-E1` | `accepted as evaluation leakage fix evidence` |
| `R0001-P36-E2` | `accepted as balanced benchmark contract evidence` |
| `R0001-P32-E1` | `approved, next priority; not executed` |
| `R0001-P40` | `changes_required` |
| `R0001-P41` | `rejected` |
| `R0001-P42` | `deferred` |

本轮没有启动正式训练，没有运行 policy 的完整闭环能力 Episode，没有新任务成功，也没有
世界模型、Actor、闭环成功率、安全或泛化能力改善。接受项均为未来评测可信度所需的合同
证据。

## 关键发现

1. 当前通用双臂 evaluator 会把同一个 raw seed 同时传给 environment 与 policy。
   environment seed 同时决定物理随机化、语言变体、latency 和传感器噪声，因此这是标准
   policy reset 接口的真实评测泄露通道。
2. P39 用 opaque planned Episode identity 和独立 environment/policy seed domain 关闭了
   raw seed 直通，同时保留旧 evaluator 的 environment-seed 行为兼容。
3. P39 的结论只适用于标准接口，不能声称隔离同进程恶意 policy 对 evaluator 内存或
   文件系统的读取。
4. 当前 evaluation profile 的 latency 是连续区间采样后取整，不能自然形成平衡 3×3
   cell；现有 evaluator 也没有 latency-cell estimand。
5. P36-E2 建立了结果前冻结的：
   - 3 task × observation latency 1/2/3 × action latency 1/2/3；
   - 27-cell 等权完整挑战账本；
   - latency 1/2 的 18-cell supported conditional 账本；
   - baseline/candidate paired seed；
   - training-seed 外层统计；
   - fail-closed planned/terminal ledger；
   - 联合 latency-only reset provenance。
6. 合成功效完整执行 72 个 strata，`n=12` 是第一个达到全部冻结门的 replicate 数：
   - 最坏 null FPR 双侧 95% Clopper-Pearson 上界
     `0.023181388069658915`；
   - 最坏 planted 10pp power 双侧 95% Clopper-Pearson 下界
     `0.839503609048341`。
7. `n=12` 对应 972 个 paired identity 和未来每个 baseline/candidate 角色合计 1,944 个
   execution plan slot；这些不是本轮实际运行的 Episode，也不自动授权未来预算。
8. 三任务共 27 次 reset-only smoke 证明联合 override 只改变两项 latency，其他
   randomization、instruction、physical state 和 camera calibration 保持一致。
9. 联合 diagnostic 被强制为 reset-only，不能 `apply()` action，因此不会让 latency 3
   的 stale observation 绕过 100ms safety validity。
10. 普通 Replay 的 P32-E1 修订已通过两名筛选 Agent 评审：
    - 24 个 source Episode 才是独立单位；
    - action 与 target 必须双重 OOF residual；
    - 必须有 controller-history、configuration-target、safety rewrite 和 shard-boundary
      守护；
    - 结论只能是 salience-retained Replay conditional information。
11. P40 的 allowed-contact 力—冲量盲区证据成立，但当前把 floor、support、容器、目标物
    与 articulation 合并会产生错误解释；需先冻结 physics-substep impulse 和分类合同。
12. P41 同时改变 RGB-D 条件化、目标选择、速度、轨迹 primitive、双臂耦合和撤回过程，
    触发多主变量强制拒绝。
13. P42 能更严格检验 language grounding，但需相同 task ID、物理可完成且互斥的 swapped
    mapping，并等待 qualified deployment 与 P36-E2。

## 当前能力基线

- 无 causality-qualified deployment；
- 最新完整 3D 世界模型负基线仍为：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`；
- 24 Episode、1,600 update、0 成功；
- action causality aggregate ratio `1.0012791539504238`；
- Actor 未解锁，task/exploration Actor update 均为 0；
- 三任务没有 bilateral contact 或 controlled motion；
- P17 plant causality evidence 保持；
- P29 runtime contract diagnostic 保持；
- P36-E1 support-domain evidence 保持；
- 新增 P39 evaluation seed isolation evidence；
- 新增 P36-E2 balanced benchmark contract evidence；
- 不宣称机器人已经学会任何家务任务。

## 提交与产物

- R0006 冻结文档：`81b4d9c`
- P39 实现：`23cbcac53f86122df08bce114f75c945bf8ee2d7`
- P39 结果文档：`9fdadf1`
- P39 run：
  `runs/research-loop/0006/r0006-p39-seed-isolation-e1-s20263901`
  - report SHA-256：
    `89dc8aaf5c7288fe03f71eb64d1d908b61a6bb3272f9d7b4486d15b7671cbda0`
  - manifest SHA-256：
    `224a1549a6d5905813bac0dc3132c92179d47448d8c8f790d2b8ad860972e539`
- P36-E2 实现：`fadaf59b79d16ddebc5f0666183fdb21c33a2c2b`
- P36-E2 run：
  `runs/research-loop/0006/r0006-p36-factorial-e2-s20263602`
  - report SHA-256：
    `7f0174bc1250627d2f0d76ed8f47698b5ce86693429fc86f63f173706d60c419`
  - planned ledger SHA-256：
    `70f44328058afd38d0a7c194e0655b442022a63026b51d220628ccc8e47fd26c`
  - reset smoke SHA-256：
    `0c0e45034120bbcac92ce4116f8d574f337e4a5f1021170fd030f6cf2289f6bf`
  - manifest SHA-256：
    `e46a7d469dd9ae71db8adb3cce505d8db925ce5c70dec57409c67bfdf77c9729`

最终门禁：

- P39 正式 run 与 P36-E2 正式 run 逐字段验证通过；
- 全量 pytest 在 P39 和 P36-E2 后各通过一次；
- 最终全量 pytest：
  - 11 项既有 skip；
  - 18 条 warning 均来自 `torch.jit.script` deprecation；
  - 无失败；
- training semantics、physics integrity、Python size、architecture、pycompile 与 diff check
  通过；
- 历史 `docs/research-loop/0001/`～`0005/` 零差异。

## 下一轮问题

下一轮应重新执行完整创新和独立筛选，不直接把下列草案当成已批准实施。优先问题：

1. 重新确认并实施 `R0001-P32-E1`：
   - 冻结 16-D rate 与 17-D configuration target 索引；
   - nested source-Episode cross-fit；
   - action/target 双重 OOF residual；
   - controller-history、safety rewrite、shard boundary 和 configuration guard；
   - exact-pipeline null/planted power；
   - 只允许 retained-Replay conditional information 结论。
2. 若 P32-E1 在全部守护下通过，再修订 P31：
   - fresh probe 与 production prior 使用可比普通 Replay 训练分布；
   - paired bank 只作评测；
   - 只有出现预注册 objective-routing gap 才重审 P33。
3. 修订 P40：
   - floor/support、manipulated object、target container、articulation 分账；
   - 按 physics substep 积分 impulse；
   - 冻结 contact pair 去重和 timestep/solver 稳定性；
   - 首轮只作描述性内部一致性报告，不冒充硬件阈值。
4. P41 若重提，必须拆成单变量：
   - 固定速度、轨迹 primitive、双臂耦合与动作幅值；
   - 只比较 RGB-D target selection 与 geometry-matched blind control；
   - 依赖修订后的 P40；
   - 无训练 smoke 与后续训练必须分开。
5. P42 等待未来 qualified deployment；不得用 P01 v4 或 scripted policy 制造语言能力
   数值。
6. 未来正式 P36-E2 capability run 仍需：
   - 三个 baseline 与三个 candidate hash-bound qualified deployment；
   - deployment hash 固定后生成私有 salt；
   - 完整 1,944 execution 预算和既有消融预算的独立资源审批；
   - 不复用本轮公开诊断 salt；
   - 不绕过 action-causality/deployment gate。

## 清理

- 启动时可用空间：`124,888,104 KiB`，约 119 GiB。
- 收尾记录时可用空间：`120,849,492 KiB`，约 115 GiB。
- 共享数据卷在本轮期间减少 `4,038,612 KiB`；不能把该共享变化归因于本轮。
- P39 run 占 32 KiB；P36-E2 run 占 772 KiB。
- 删除内容：无。
- 未启动清理 Agent：
  - 本轮新增正式 run 合计不足 1 MiB；
  - 历史训练、checkpoint、report、manifest、日志和失败证据均需保留；
  - 没有主 Agent 确认可重建且无引用的大体量资源。
