# R0004 研究上下文

## 轮次身份

- 轮次：`R0004`
- 状态：已完成
- 起始分支：`feat/research-loop`
- 起始提交：`bb73138ea7f84ceeebcf1424a7cf9b60bc3860f7`
- 起始工作区：干净
- 起始日期：`2026-08-20`
- 前一轮：`R0003`
- 本轮能力基线：不变
- 本轮接受证据：`R0001-P29` runtime contract diagnostic

本轮只修改 `docs/research-loop/0004/` 及经筛选批准的新实现、测试和实验配置。
下列历史目录冻结只读：

| 目录 | 起始提交下的 Git tree |
|---|---|
| `docs/research-loop/0001/` | `416912b7dc1c19611bcfc4375028180014a1989b` |
| `docs/research-loop/0002/` | `6fb603dbd52451fe1749157daf05aa482ca7222f` |
| `docs/research-loop/0003/` | `f56011eda321ea803bc24051db001e632c1549fb` |

## 可信基线

- 能力基线不变：当前没有通过因果与闭环门禁的 deployment，不宣称任务能力改善。
- 数据因果证据：`R0001-P17` 已证明实际 plant action 在三个正式任务和
  `1/4/8/16` 步 horizon 上具有稳定物理因果效应。
- 测量结论：现有 7×16 Action Probe 在高相关动作下功效不足；其失败只能标记
  `inconclusive`，不能作为能力否定。
- R0003 已排除 batch source、原始 posterior overshooting、free-nats、纯 action scale、
  普遍 GRU/prior shortcut、argmax 离散化和 decoder 系统低 gain 作为充分解释。
- `R0001-P25a` 因 shift=1 的 per-shift output guard 未过而在计算前拒绝；
  `R0001-P25b` 因依赖失败未运行。

## 当前主要瓶颈

> 为什么冻结模型的 action-conditioned feature effect 能稳定到达 decoder 输入，
> 但少数 shift=1/任务 Episode 的 decoded output effect 低于门槛；现有训练目标是否没有
> 稳定奖励 action-discriminative 方向？

证据边界：

- P24-R2 的 visual/proprio decoder input 与 aggregate output effect 均在 24/24 Episode
  上存活，没有系统性的相邻层低 retention。
- 更严格的 per-shift 守护显示 shift=1 不是全覆盖：
  visual 为 23/24，proprioception 为 21/24。
- 失败集中在 store kitchen items，proprioception 另有一个 tidy living room Episode。
- 不能选择 shift=5/9、任务子集或 Episode 子集规避失败。
- 不得把评测修复、stale-frame 修复、安全正例分层和能力训练捆绑为一个因果对比。

## 训练与评测入口

- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output artifacts/development-ready.json`
- 世界模型训练：
  `.venv/bin/python -m hwr.apps.train_foundation_world_model`
- 世界模型评测：
  `.venv/bin/python -m hwr.apps.evaluate_foundation_world_model`
- R0003 冻结诊断入口包括：
  `evaluate_layerwise_action_effect`、`evaluate_prior_argmax_effect` 和
  `evaluate_decoder_gain`。
- 最新 `artifacts/development-ready.json` 通过，但其 source commit 早于本轮起点；
  任何正式训练前必须在候选干净提交上重新执行，不沿用旧门禁。

## 资源状态

- 主机：Apple M5 Pro，20 核 GPU，48 GiB unified memory。
- Python：`.venv` 中 Python 3.11.0。
- 启动时没有本项目训练或评测进程，没有 tmux session。
- 数据卷可用空间约 104 GiB；旧 readiness 估计单次正式 run 约 28.41 GiB。
- 加速器后端：MPS；正式训练默认独占本机可用加速器。

## 本轮约束

1. 先由三个创新 Agent 独立提案，再由两个筛选 Agent 独立评分。
2. 先执行不更新参数的低成本判别验证；只有预注册门槛通过才允许训练。
3. 一个训练候选只改变一个主变量，不降低既有门槛，不挑 seed、shift、任务或 Episode。
4. 训练前冻结源码、输入、checkpoint、命令、seed、预算、主要指标和守护指标。
5. 所有行为变化必须有测试；训练只从干净、已提交且通过项目门禁的提交启动。
6. 能力结论必须来自未见分布上的闭环物理结果，诊断或 loss 改善不能替代能力证据。
7. 正式长时训练必须通过 `traex-host-exec` 与 tmux 启动，并设置带“阅读 AGENTS.md”
   的完成唤起和定时看门狗；小型验证不得休眠。
