# R0007 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P32-E1` | `inconclusive` |
| `R0001-P40-E1` | `accepted as safety measurement contract evidence` |
| `R0001-P41-E1` | `changes_required` |
| `R0001-P42-E1` | `deferred` |
| `R0001-P43` | `deferred` |
| `R0001-P44` | `deferred` |
| `R0001-P45` | `changes_required` |
| `R0001-P46` | `approved, not selected` |

本轮没有启动正式训练，没有运行 policy 闭环能力 Episode，没有新任务成功，也没有世界
模型、Actor、闭环成功率、泛化或安全能力改善。P40 的接受项只代表测量合同成立；P32
因功效不足不能接受或拒绝主要假设。

## 关键发现

1. P01 v4 普通 Replay 的输入谱系完整可重建：
   - 24 个 source Episode；
   - 168 个 shard；
   - 2,688 transition；
   - task/source 为 6/6/12；
   - manifest 与全部 shard hash 通过。
2. state-only nuisance 下，executed-action residual 对 rate target 的 aggregate ratio 为
   `1.1114809121271965`，三任务均大于 1，source bootstrap p05 为
   `0.08909454222505013`。
3. configuration target 同向，aggregate ratio 为 `1.0937914362756587`，三任务均大于
   1，说明主点估计不只存在于即时 rate target。
4. no-rewrite 与 shard-interior 点估计仍为正，表面上排除了信号只来自 safety rewrite 或
   shard 前四步。
5. 加入 current proposal 和过去 4 步 proposal/executed action 后，aggregate ratio 降至
   `0.9069818836808229`，三任务均小于 1，bootstrap p05 为
   `-0.11092589471031235`。观察结果更符合 controller-history/FIFO 可解释主信号，而不是
   独立 executed-action residual。
6. 但 exact-pipeline 10% planted power 仅 1/200，Clopper-Pearson 95% lower 为
   `0.0002564335872234741`，远低于 0.80；按预注册顺序必须 `inconclusive`，不能用
   controller guard 的负结果直接拒绝 P32。
7. 两个 null 都是 0/200 pass，FPR upper 均为
   `0.014867039231272056`，说明当前问题是低功效，不是明显的假阳性膨胀。
8. safety-rewrite 只有 6 个 source，多个 fold-task cell 为 0 或 1 个 source，0 个有限
   bootstrap replicate；这再次否定本轮直接执行 P43 的可行性。
9. P40 建立了 robot–environment contact 的五类 report-only 总账：
   floor/support、manipulated object、target container、articulation、forbidden。
10. P40 的 measurement enabled/disabled 三任务 trace bit-identical；新测量没有进入
    safety、reward、termination 或 success。
11. timestep 0.002/0.001 的 fixture impulse 相对差约 `1.81e-16`，通过 10% 稳定性门。
12. 三任务 fixed hold trace 均出现 floor/support category peak 约 1.74–1.80kN，而旧
    severe collision 与 forbidden force 均为 0，证明旧总账确实跳过 allowed-contact
    负载。
13. 上述 floor/support 数值主要是静态支撑接触；220N 不是硬件或地面支撑安全阈值，
    不能据此声称危险或修改安全层。

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
- P39 evaluation seed isolation evidence 保持；
- P36-E2 balanced benchmark contract evidence 保持；
- 新增 P40-E1 safety measurement contract evidence；
- P32-E1 为 inconclusive，不改变数据、模型或能力基线；
- 不宣称机器人已经学会任何家务任务。

## 提交与产物

- R0007 冻结文档：`4ef5f3b`
- P32 实现：`63032f54b5e926511cfd4530d6da483611b3573d`
- P40 实现：`7bd995e9720ff7196b72a1111f26cef1377e1c75`
- P32 run：
  `runs/research-loop/0007/r0007-p32-replay-conditional-e1-s20263201`
  - report：
    `f2e2af4fa2db982c970a3882b45459e6afb22f103caf55865bfd84280eec6174`
  - folds：
    `df7a135f0c4fee4a948a05079c3657a9fda243d594c5a41dc2a57592e88d1729`
  - manifest：
    `0ebb4ce5bb9efbef7a248f50091f9130732a3949efb2aafb5b1401803b4db4af`
- P40 run：
  `runs/research-loop/0007/r0007-p40-contact-ledger-e1-s20264001`
  - report：
    `73a1a6e0a65f55b5d6d170219d847e84723c8ae01c2e876233efae44dd5a864c`
  - manifest：
    `7da768a3edbdea7707534d87edc888c7935f0eabfad7fe0812630ef8d1d32294`

## 下一轮问题

下一轮必须重新执行创新与独立筛选，不直接继承批准状态。优先问题：

1. 不要在当前 24-source Replay 上继续 P31、P43 或新的 action objective 训练：
   - P32 10% planted power 严重不足；
   - controller history 已解释并反转主点估计；
   - safety-positive source 尤其厨房不足。
2. 若重提普通 Replay 条件信息诊断，应先解决独立单位和设计方差：
   - 采集更多 task-balanced source Episode；
   - 结果前保证每 task/fold 有足够 source；
   - 保持 controller-history guard；
   - 不降低 1.05/1.02 门槛，不删除 configuration target；
   - 重新冻结 exact-pipeline MDE 和 power。
3. 更高价值的路线可能是改善普通数据的交互支持，而不是继续挖同一 Replay：
   - 先修订 P41-E1 为唯一 target-index 变量；
   - candidate/control 必须从相同 policy-visible RGB-D candidate set 选择；
   - 固定 primitive、动作幅值、双臂耦合和提前终止处理；
   - 先冻结 paired power 与 MDE。
4. P40 已满足 P41 的测量前置，但未来接触探索仍需：
   - 按类别发布 force/impulse；
   - 不把 220N 当硬件阈值；
   - 不用 floor/support 接触冒充物体交互；
   - severe collision、stale action 和 safety burden 不得劣化。
5. P43 若重提，必须先新增普通采集的独立 safety-positive source：
   - 每任务均能形成 source-disjoint train/selection/test；
   - 不能把正式 action-execution holdout 加入训练；
   - 需要真正 detached head-only optimizer；
   - 三优化 seed 不能替代独立 source。
6. P42、P44 继续等待 qualified deployment；不得用 scripted policy、P01 v4 或放宽
   action-causality gate制造泛化数值。
7. P46 可作为后续可靠性侧车重新筛选，但不应挤占交互数据瓶颈实验。

## 清理

- 启动时数据卷可用空间：`120,783,000 KiB`，约 115 GiB。
- 收尾记录时数据卷可用空间：`90,892,188 KiB`，约 86.7 GiB。
- 共享数据卷在本轮期间减少约 28.5 GiB；本轮新增正式 run 合计约 1.0 MiB，不能把共享
  空间变化归因于本轮。
- P32 run：440 KiB。
- P40 run：592 KiB。
- 删除内容：无。
- 未启动清理 Agent：
  - 本轮新增产物很小；
  - 历史 checkpoint、Replay、report、manifest、日志和失败证据均需保留；
  - 没有主 Agent 确认可重建且无引用的大体量资源。
