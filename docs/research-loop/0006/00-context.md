# R0006 研究上下文

## 轮次身份

- 轮次：`R0006`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`c1bf06f83400aabea9e63bfe57eceec1f8e85516`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-20`
- 前一轮：`R0005`
- 本轮能力基线：不变

本轮文档只写入 `docs/research-loop/0006/`。只有经独立筛选和结果前冻结后，才允许修改
明确归属的新实现、测试和实验配置。下列历史目录冻结只读：

| 目录 | 起始提交下的 Git tree |
|---|---|
| `docs/research-loop/0001/` | `416912b7dc1c19611bcfc4375028180014a1989b` |
| `docs/research-loop/0002/` | `6fb603dbd52451fe1749157daf05aa482ca7222f` |
| `docs/research-loop/0003/` | `f56011eda321ea803bc24051db001e632c1549fb` |
| `docs/research-loop/0004/` | `611c420e539a53a8c7578cd66aa8bdfe46fe82b7` |
| `docs/research-loop/0005/` | `0352d379d5754adb03e9158c0fa72393ab322d58` |

## 可信基线

- 当前没有通过因果与未见分布闭环门禁的 deployment，不宣称机器人已学会家务。
- 最新完整 3D 世界模型负基线仍为
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`：
  - 24 Episode；
  - 1,600 update；
  - 0 成功；
  - action causality aggregate ratio `1.0012791539504238`，低于冻结门槛 `1.05`；
  - Actor 从未解锁，task/exploration Actor update 均为 0；
  - 三任务 interaction coverage 均没有 bilateral contact 或 controlled motion。
- 该 run 保留 update 1,200/1,400/1,600 的训练 checkpoint、replay、causality holdout 和
  recovery state，可以支持只读诊断；它不是合格 deployment。
- `R0001-P17` 已证明实际 plant action 在三个正式任务和 `1/4/8/16` 步 horizon 上具有
  稳定物理因果效应，因此不能把世界模型失败归因于环境完全不响应动作。
- R0003 已排除 batch source、原始 posterior overshooting、free-nats、纯 action scale、
  普遍 GRU/prior shortcut、argmax 离散化和 decoder 系统低 gain 作为充分解释。
- `R0001-P29` 已接受为 runtime 合同诊断：
  - 20Hz 控制周期为 50ms；
  - action validity 从 visible observation timestamp 起为 100ms；
  - observation latency 0/1/2 的最大 visible age 为 0/50/100ms；
  - latency 3 的最大 age 为 150ms，超出当前安全支持域；
  - 不允许通过延长 validity、删除 latency 3 或读取 future/latest observation 绕过安全层。
- `R0001-P36-E1` 已接受为 evaluation contract evidence：
  - 首要 `complete_challenge` 总账保留全部 144 个 P11 Episode；
  - `supported_conditional` 为 latency 1/2 的 108 个 Episode；
  - challenge 为 latency 3 的 36 个 Episode，包含全部 2,196 次安全拒绝；
  - P11 的 27 个 cell 为每任务每 action latency 下 `2/10/4`，不是 balanced factorial
    capability benchmark；
  - 报告明确禁止能力结论。

## 上一轮结论与未决依赖

- `R0001-P34` 已因实质放宽 100ms source-age safety 边界而拒绝，不得换名复活。
- `R0001-P35` 缺少独立实时安全感知或 delay-aware reachable-set 前置，继续延后。
- `R0001-P37` 与现有更强双臂 shaping 重复且会引入单臂捷径，已拒绝。
- `R0001-P31` 需要消除 paired probe 与 production prior 的训练分布混淆。
- `R0001-P32` 需要 nested source-Episode cross-fit、完全 out-of-fold nuisance predictor、
  safety rewrite 分层和 null/planted power；其结论只能是普通 Replay 条件预测信息。
- `R0001-P33` 只有 P31 出现预注册的 objective-routing 缺口后才可重新筛选。
- `R0001-P38` 只能保留 report-only shadow，不能改变 Actor/deployment unlock。

## 本轮主要瓶颈

`R0006-C01`：

> 项目已经能把当前时效安全支持域与完整挑战域唯一分账，但还没有结果前冻结、cell
> 平衡、可重建且能直接承载未来 closed-loop success 对比的物理能力基线。现有 P11
> 分布不平衡且没有任务成功标签；现有 foundation evaluator 又要求旧 action-shuffle
> causality 通过，而 P01 v4 没有已解锁 Actor。若直接开始新的能力训练，最终成功率可能
> 因评测 cell 权重、seed、缺失处理或准入门漂移而不可归因。

本轮优先回答：

1. 能否在不改变 policy、runtime、task 或 safety 的前提下，结果前冻结
   `task × observation latency 1/2/3 × action latency 1/2/3` 的 balanced factorial
   capability benchmark 合同、seed、权重、缺失处理、统计和双账本发布规则；
2. 若当前没有合格 deployment 可执行该 benchmark，能否建立明确标记为
   `unqualified/null capability baseline` 的物理运行或验证证据，而不把它误称为模型能力；
3. 普通训练 Replay 中，给定冻结 pre-action state 后的 executed-action residual 是否含
   稳定 successor physical information，从而决定后续是否值得采集 paired probe 或训练。

评测合同建立与 Replay 可识别性诊断不得和世界模型训练捆绑。若筛选结果认为评测基线
当前无法在不改变行为的前提下建立，则本轮应先完成最低成本且可证伪的 P32 修订诊断，
而不是为了“有结果”启动昂贵训练。

## 当前入口与证据缺口

- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 世界模型训练：
  `.venv/bin/python -m hwr.apps.train_foundation_world_model --run-id <RUN_ID>`
- 世界模型评测：
  `.venv/bin/python -m hwr.apps.evaluate_foundation_world_model <RUN_PATH>`
- 双臂通用闭环评测：
  `.venv/bin/python -m hwr.apps.evaluate_bimanual_rl <RUN_PATH>`
- support-domain 聚合：
  `.venv/bin/python -m hwr.apps.evaluate_support_domain`

当前 foundation evaluator：

- 默认每任务 40 个未见 seed；
- 会排除训练与 causality-holdout seed；
- 只运行正式任务的 evaluation profile；
- 仍要求 passed action-shuffle causality；
- P01 v4 没有已解锁、已导出的合格 Actor，因此不能直接把该入口的拒绝绕过为能力评测。

任何 benchmark integration 必须把“评测 runner 能运行”和“被评策略合格”分开报告；不得
通过删除准入门、伪装 scripted policy、复用正式 holdout 或读取成功标签来制造基线。

## 启动资源状态

- 主机：Apple M5 Pro，18 核 CPU、20 核 GPU、48GB 统一内存。
- `.venv`：Python 3.11，PyTorch 2.13.0。
- 当前沙箱：
  - `torch.backends.mps.is_built() == True`；
  - `torch.backends.mps.is_available() == False`；
  - `torch.cuda.is_available() == False`。
- 正式 MPS 运行必须在 host-exec 环境重新核验加速器与进程独占状态。
- 可用工具：`tmux`、`traex`、`traecli`。
- 启动时数据卷可用空间：`124,888,104 KiB`，约 119 GiB。
- `runs/research-loop/0003/` 约 4.1 GiB；旧完整世界模型 run 仍为大体量不可重建证据，
  本轮不清理。

## 本轮组织

- 创新 Agent A：评测合同、平衡设计与统计功效。
- 创新 Agent B：普通 Replay 可识别性与世界模型因果路由。
- 创新 Agent C：闭环学习、泛化与安全的独立反例方向。
- 两名筛选 Agent 在同一冻结提案集上独立评分，完成前互不查看结果。
- 每个入选候选由唯一实施 Agent 负责，文件所有权在 `03-experiment.md` 中冻结。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 所有新观点使用稳定 ID，并在提案、实现、提交、运行和结果间保持引用。
3. 先执行不更新 production 参数的最低成本判别验证；只有预注册门槛通过才允许训练。
4. 评测/runtime 修复与能力训练分离；一个候选只改变一个主变量。
5. 不延长 100ms safety validity，不移除 latency 3，不用 latest/future observation 绕过
   延迟，不降低碰撞或安全门槛。
6. latency 3 必须保留在完整挑战总账；supported conditional 不能取代完整结果。
7. 不使用 action shuffle/derangement family 作为训练监督、候选选择或能力验收依据。
8. 不按结果挑 seed、任务、Episode、cell、权重、缺失处理、统计方法或门槛。
9. 普通 Replay 诊断必须以 source Episode 为最外层切分单位，所有 nuisance 拟合和
   标准化只能使用训练折。
10. 所有行为变化必须有测试；训练只从干净、已提交且通过项目门禁的提交启动。
11. 能力结论只能来自冻结未见分布上的闭环物理结果；诊断、loss、probe、训练回报或视频
    不能替代。
12. 正式长时训练必须经 `traex-host-exec` 与 tmux 启动，并设置带“阅读 AGENTS.md”的
    完成唤起和定时看门狗；小型验证不得休眠。
