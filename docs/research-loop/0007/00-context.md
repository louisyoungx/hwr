# R0007 研究上下文

## 轮次身份

- 轮次：`R0007`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`a722f3522cdb8f12c1a78c56ce8c1d7c873e9190`
- 起始远端：`origin/feat/research-loop`，与本地 `+0/-0`
- 起始工作区：干净
- 起始日期：`2026-08-20`
- 前一轮：`R0006`
- 本轮能力基线：不变

本轮文档只写入 `docs/research-loop/0007/`。只有经本轮独立筛选和结果前冻结后，才允许
修改明确归属的新实现、测试和实验配置。下列历史目录冻结只读：

| 目录 | 起始提交下的 Git tree |
|---|---|
| `docs/research-loop/0001/` | `416912b7dc1c19611bcfc4375028180014a1989b` |
| `docs/research-loop/0002/` | `6fb603dbd52451fe1749157daf05aa482ca7222f` |
| `docs/research-loop/0003/` | `f56011eda321ea803bc24051db001e632c1549fb` |
| `docs/research-loop/0004/` | `611c420e539a53a8c7578cd66aa8bdfe46fe82b7` |
| `docs/research-loop/0005/` | `0352d379d5754adb03e9158c0fa72393ab322d58` |
| `docs/research-loop/0006/` | `ee3a6f5b25887f67f812750d2a75424df12823d4` |

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
- `R0001-P17` 已证明实际 plant action 在三个正式任务和 `1/4/8/16` 步 horizon 上具有
  稳定物理因果效应；环境并非完全不响应动作。
- R0003 已排除 batch source、原始 posterior overshooting、free-nats、纯 action scale、
  普遍 GRU/prior shortcut、argmax 离散化和 decoder 系统低 gain 作为充分解释。
- `R0001-P29` 已接受为 50ms 控制周期、100ms visible-source-age validity 和
  observation latency 1/2 支持域、latency 3 挑战域的 runtime 合同证据。
- `R0001-P36-E1` 已接受为完整挑战与支持条件唯一分账的历史评测合同证据。
- `R0001-P39-E1` 已接受为标准 policy reset 接口的环境 seed / policy RNG seed 隔离证据。
- `R0001-P36-E2` 已接受为未来能力评测的 balanced factorial benchmark 合同证据：
  - 3 task × observation latency 1/2/3 × action latency 1/2/3；
  - 27-cell 等权完整挑战账本和 18-cell supported conditional 账本；
  - `n=12` 是第一个通过冻结合成功效门的每 cell replicate 数；
  - 只运行 reset-only smoke，没有 policy inference、action 或完整 Episode；
  - 不授权正式 1,944-slot capability run。

## 上一轮结论与未决依赖

- R0006 接受的两项均为评测可信度合同，没有能力改善、训练或新任务成功。
- `R0001-P32-E1` 在 R0006 中曾获两名筛选 Agent 支持，但未实施；本轮必须重新生成提案
  并独立筛选，不能直接继承批准状态。
- `R0001-P31` 仍需消除 paired probe 与 production prior 的训练分布混淆；只有普通
  Replay 条件信息门支持继续采集时才值得修订。
- `R0001-P33` 只有后续 P31 出现预注册的 objective-routing 缺口后才可重新筛选。
- `R0001-P40` 需先拆分 floor/support、manipulated object、target container 和
  articulation，冻结 physics-substep impulse 与 contact-pair 去重合同。
- `R0001-P41` 因多主变量被拒绝；不得换名捆绑 RGB-D、目标选择、速度、轨迹和双臂耦合。
- `R0001-P42` 依赖未来 qualified deployment，不得用 P01 v4 或 scripted policy 生成
  language-grounding 能力数值。

## 本轮主要瓶颈

`R0007-C01`：

> plant 已被独立物理干预证明可控，但当前 24 个 source Episode 的普通 salience-retained
> Replay 是否在给定 pre-action visible state 后仍含 executed-action 对 successor physical
> observation 的稳定增量条件信息，尚未被无 source 泄露且具有足够功效的设计回答。若
> Replay 本身没有可识别信号，继续修改 production 世界模型目标或启动昂贵训练没有依据；
> 若信号存在，也只能授权下一步区分“数据有信息”与“production 模型是否使用信息”，不能
> 直接宣称因果、能力或闭环成功。

本轮优先回答：

1. 能否在 source-Episode 完全 out-of-fold、action 与 successor target 双重残差化、
   control/candidate 共享 state baseline 的条件下，建立普通 Replay 条件信息诊断；
2. 该信号是否在 controller history、安全改写、shard 边界、任务和 configuration target
   守护下仍存在，而不是 FIFO、控制器记忆、retention selector 或即时 rate target 伪影；
3. exact-pipeline null/planted power、有效秩和 source-level bootstrap 是否足以支持明确
   的 `accepted`、`rejected` 或 `inconclusive` 判定；
4. 若本轮独立筛选发现更优先且不依赖 qualified deployment 的单变量候选，是否应改走
   该候选；不得仅因上一轮已写草案而机械选择 P32-E1。

## 当前数据、入口与证据缺口

- 冻结输入候选：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/replay/autonomous`。
- R0006 记录的 replay manifest SHA-256：
  `c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985`。
- 数据包含 24 个 source Episode、168 个 shard、2,688 transition；独立统计单位只能是
  source Episode，不能把 shard 或 transition 当独立样本。
- source 分布为餐桌 6、厨房 6、客厅 12；每个 source 恰有 7×16 transition。
- Replay 保存 pre/successor visible proprio、actor proposal、executed action 和 safety
  intervention；不保存无延迟 physical snapshot，也不保存逐 Episode plant latency/scale。
- retained-window selector 使用 successor motion、action innovation、安全和交互结果；
  任何正结果只能外推到当前 salience-retained distribution。
- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output <PATH>`
- 当前不存在 P32-E1 专用 evaluator；实现前必须冻结数据列、fold、nuisance、统计、功效、
  守护和结论边界。

## 启动资源状态

- 主机：Apple M5 Pro，18 核 CPU、20 核 GPU、48GB 统一内存。
- `.venv`：Python 3.11.0，PyTorch 2.13.0。
- 当前沙箱：
  - `torch.backends.mps.is_built() == True`；
  - `torch.backends.mps.is_available() == False`；
  - `torch.cuda.is_available() == False`。
- 正式 MPS 运行必须在 host-exec 环境重新核验加速器与进程独占状态。
- 启动时数据卷可用空间：`120,783,000 KiB`，约 115 GiB。
- 本轮预期首项是 CPU 离线诊断，不应休眠或占用正式训练资源。

## 本轮组织

- 创新 Agent A：学习信号、动作因果路由和双臂控制。
- 创新 Agent B：评测有效性、泛化和安全。
- 创新 Agent C：数据效率、计算效率和训练系统。
- 两名筛选 Agent 将在同一冻结提案集上独立评分，完成前互不查看结果。
- 每个入选候选由唯一实施 Agent 负责，文件所有权在 `03-experiment.md` 中冻结。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 所有新观点使用稳定 ID，并在提案、实现、提交、运行和结果间保持引用。
3. 先执行不更新 production 参数的最低成本判别验证；只有预注册门槛通过才允许训练。
4. 一个候选只改变一个主变量；评测/runtime 修复、离线诊断与能力训练分离。
5. 不延长 100ms safety validity，不移除 latency 3，不读取 future/latest observation，
   不降低碰撞、安全或 deployment/action-causality 门槛。
6. 不使用 action shuffle/derangement family 作为训练监督、候选选择或能力验收依据。
7. 不按结果挑 seed、source、fold、task、target、mask、权重、统计方法或门槛。
8. 普通 Replay 诊断必须以 source Episode 为最外层切分单位；nuisance、标准化、rank、
   超参和 target scale 只能使用训练折。
9. 普通 Replay 观测研究不得称为 plant causality、production utilization、模型能力、
   安全改善或闭环成功。
10. 所有行为变化必须有测试；正式运行只从干净、已提交且通过项目门禁的提交启动。
11. 能力结论只能来自冻结未见分布上的闭环物理结果；诊断、loss、probe、训练回报或视频
    不能替代。
12. 正式长时训练必须经 `traex-host-exec` 与 tmux 启动，并设置带“阅读 AGENTS.md”的
    完成唤起和定时看门狗；小型验证不得休眠。
