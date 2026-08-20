# R0005 轮次总结

## 结论

| ID | 结论 |
|---|---|
| `R0001-P31` | `changes_required` |
| `R0001-P32` | `changes_required` |
| `R0001-P33` | `deferred` |
| `R0001-P34` | `rejected` |
| `R0001-P35` | `deferred` |
| `R0001-P36-E1` | `accepted as evaluation contract evidence` |
| `R0001-P37` | `rejected` |
| `R0001-P38` | `changes_required, shadow/report-only` |

本轮没有启动正式训练，没有新任务成功，也没有能力改善。唯一接受项是 P36-E1 的评测
合同证据。

## 关键发现

1. latency3 不是 latest-bundle 选择 bug，而是 150ms visible observation age 与当前
   100ms source-age/action-validity 合同不相容。
2. P34 的“双时钟租约”虽然保留 100ms command lease，却让 150ms old observation
   获得新的可执行命令，实质把 source-age safety 边界从 100ms 放宽到 150ms。
3. 单调 sequence 和可信 capture timestamp 只能证明“这确实是一帧旧图”，不能恢复过去
   150ms 内丢失的动态环境状态；固定滞后故障流与正式 latency3 也不可区分。
4. 因此 P34 触发安全强制拒绝，P35 缺少安全 runtime 前置而延后。
5. P37 与当前 `BimanualTaskTracker` 已有的 left/right reach、worst-side reach、
   bilateral occupancy、joint grasp readiness 与 contact shaping 实质重复。
6. P37 提议的 `min(left,right)` 比现有双臂约束更弱，容易制造单臂接近捷径，因此拒绝。
7. P31/P32 避开了 action shuffle/derangement，但仍需修订：
   - P32 必须 nested source-Episode cross-fit，并把结论限制为普通 Replay 条件预测信息；
   - P31 必须消除 paired probe 与 production prior 的训练分布混淆；
   - P33 只有 P31 出现预注册缺口后才能重新冻结。
8. P38 确认了 P16 低功效结论后的 evidence-role 漂移，但 P17 只能证明 plant
   controllability，不能替代普通 Replay identifiability；当前只保留 report-only 草案。
9. P36-E1 成功把历史 144 Episode 唯一分为：
   - supported：108；
   - challenge：36；
   - 完整首要总账：144。
10. 36 个 challenge Episode 保留在完整分母，包含全部 2,196 次安全拒绝；没有删除、
    重标或拆分 latency3。
11. P11 的 observation-latency 分布为 18/90/36，27 个 cell 为 2/10/4，不是 balanced
    factorial capability benchmark，不能据此报告成功率。
12. P36-E1 报告明确禁止能力结论，只有固定条件声明：

> 支持 visible observation age <=100ms 的 evaluation 子域；完整 evaluation profile
> 尚未支持。

## 当前能力基线

- 无 causality-qualified deployment；
- 最新完整 3D 世界模型基线仍为 24 Episode、1,600 update、0 成功；
- `R0001-P17` 物理因果证据保持；
- `R0001-P29` runtime 合同诊断保持；
- 新增 `R0001-P36-E1` evaluation contract evidence：
  - 完整挑战总账优先；
  - supported conditional 只能作限定声明；
  - latency3 继续记为完整 profile 未支持；
- 不宣称世界模型、Actor、闭环成功率或安全能力改善。

## 提交与产物

- R0005 冻结文档：`376c172`
- 共享规则更新：`52433e8`
  - 只新增“给子 Agent 充足排队和任务处理时间”；
  - 本轮后续调度遵守该规则。
- P36 实现：`998abb9ca63439ce0b09045c91f30f7d72fcca07`
- P36 结果文档：`1e8bb73`
- P36 run：
  `runs/research-loop/0005/r0005-p36-support-domain-e1-s20263601`
- report SHA-256：
  `b9648d9c82b081433bd7040b06c537526345a1bfa6ba5bb3813e1278c7d7db6d`
- manifest SHA-256：
  `74823721bb6a1aca08b9b1b93eb4ee4fd53f6cfa3e5e42ed41cd2ef17e1e8ceb`
- run 磁盘占用：88 KiB。

最终门禁：

- 全量 pytest 在沙箱外通过；
- 11 项既有 skip；
- 18 条 warning 来自 `torch.jit.script` deprecation；
- training semantics 通过；
- physics integrity 通过；
- Python size、architecture、py_compile 通过；
- 历史 `docs/research-loop/0001/`～`0004/` 零差异。

沙箱内全量 pytest 的 65 项失败均为 MuJoCo CGL 创建时的
`invalid CoreGraphics connection`；沙箱外同一源码全部通过，故不归因为代码回归。

## 下一轮问题

下一轮应重新执行完整创新和独立筛选，不直接把下列草案当成已批准实验。优先问题：

1. P36-E2：如何结果前冻结 balanced factorial capability benchmark：
   - task × observation latency 1/2/3 × action latency 1/2/3；
   - seed、Episode 数、权重、缺失处理、Wilson/bootstrap 和 acceptance；
   - supported 与 complete ledger 同屏；
   - 先建立无能力行为变化的新评测基线。
2. P32 修订：普通训练 Replay 中，给定冻结 pre-action state 后的 executed-action residual
   是否含稳定 successor physical information：
   - nested source-Episode cross-fit；
   - nuisance predictor 完全 out-of-fold；
   - safety rewrite 独立分层；
   - null/planted power；
   - 不把条件预测信息夸大为 plant causality 或 production utilization。
3. 若 P32 提供足够信息，再修订 P31：
   - fresh probe 与 production path 使用可比训练分布；
   - paired holdout 只作评测；
   - support、target、sampling、统计和 response matrix 唯一可重建。
4. 只有 P31 出现“fresh conditioned probe 通过、blind control 不通过、production prior
   不通过”的预注册模式，才重新筛选 P33。
5. 若未来重新研究 latency3 可执行性，必须使用新稳定 ID，并先提供独立实时安全感知或
   delay-aware reachable-set 证据；不得复活当前 P34 或用 lease 重命名绕过 100ms
   source-age 边界。

## 清理

- 启动时可用空间：`131,205,392 KiB`。
- 收尾时可用空间：`122,523,648 KiB`。
- 共享数据卷在本轮期间减少 `8,681,744 KiB`；P36 正式 run 只占 88 KiB，不能把该共享
  变化归因于本实验。
- 删除内容：无。
- 未启动清理 Agent。
- 历史训练、checkpoint、report、manifest、日志和失败证据全部保留。
