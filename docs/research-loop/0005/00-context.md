# R0005 研究上下文

## 轮次身份

- 轮次：`R0005`
- 状态：进行中
- 起始分支：`feat/research-loop`
- 起始提交：`cae86e48eebc45e170f91c633525f00e965cec98`
- 起始工作区：干净
- 起始日期：`2026-08-20`
- 前一轮：`R0004`
- 本轮能力基线：不变

本轮文档只写入 `docs/research-loop/0005/`。只有经独立筛选和冻结批准后，才允许修改
明确归属的新实现、测试和实验配置。下列历史目录冻结只读：

| 目录 | 起始提交下的 Git tree |
|---|---|
| `docs/research-loop/0001/` | `416912b7dc1c19611bcfc4375028180014a1989b` |
| `docs/research-loop/0002/` | `6fb603dbd52451fe1749157daf05aa482ca7222f` |
| `docs/research-loop/0003/` | `f56011eda321ea803bc24051db001e632c1549fb` |
| `docs/research-loop/0004/` | `611c420e539a53a8c7578cd66aa8bdfe46fe82b7` |

## 可信基线

- 当前没有通过因果与未见分布闭环门禁的 deployment，不宣称家务任务能力改善。
- 最新完整 3D 世界模型负基线仍为
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`：
  24 Episode、1,600 update、0 成功。
- 该基线 action causality aggregate ratio 为 `1.0012791539504238`，低于冻结门槛
  `1.05`；action execution 三任务 recall 均为 `0.0`。
- `R0001-P17` 已证明实际 plant action 在三个正式任务和 `1/4/8/16` 步 horizon 上具有
  稳定物理因果效应，因此不能把世界模型失败归因于环境完全不响应动作。
- R0003 已排除 batch source、原始 posterior overshooting、free-nats、纯 action scale、
  普遍 GRU/prior shortcut、argmax 离散化和 decoder 系统低 gain 作为充分解释。
- `R0001-P28` 因 successor-posterior oracle、负例指纹、采样不一致和不可重建统计合同等
  结构性风险，在实现前被独立复审拒绝；本轮不得直接复活该草案。
- `R0001-P29` 已接受为 runtime contract diagnostic：
  - 20Hz 控制周期为 50ms；
  - action validity 从 visible observation timestamp 起为 100ms；
  - observation latency 0/1/2 的最大 age 为 0/50/100ms，均可执行；
  - latency 3 的 age 为 150ms，64-step Episode 从 step 3 起产生 61 次
    `outside_validity_window`；
  - P11 的 36 个 latency=3 Episode 每个均有 61 次拒绝，总计 2,196 次；
  - 不允许通过延长窗口、删除 latency=3 或替换 latest observation 绕过独立安全层。

## 本轮主要瓶颈

`R0005-C01`：

> 正式未见分布包含 observation latency=3，但当前 100ms 独立安全动作时效合同使该域在
> 20Hz 下结构性不可执行。只要该矛盾未解决或未被明确冻结为不支持域，任何闭环成功率
> 对比都可能混入“动作被合同拒绝”而非策略能力差异，无法形成可信能力验收。

本轮优先回答一个互斥决策：

1. 是否存在不延长安全窗口、不使用 future/latest frame、且不让真正 stale 输入通过的
   单变量 latency-aware action scheduling 机制；
2. 若不存在，则是否应把 latency=3 明确冻结为当前硬件/控制栈不支持域，并建立不削弱
   其余任务、布局、语言和动力学难度的闭环域支持声明。

action-discriminative 目标仍是能力主线，但只有不依赖 action shuffle/derangement 形状、
具有信息匹配 control、且不复活 P28 研究自由度的新候选才可进入筛选。本轮不得把 runtime
合同变更与世界模型训练捆绑为同一因果对比。

## 训练与评测入口

- 开发门：
  `.venv/bin/python scripts/verify_development_ready.py --foundation-device cpu --output artifacts/development-ready.json`
- 世界模型训练：
  `.venv/bin/python -m hwr.apps.train_foundation_world_model --run-id <RUN_ID>`
- 世界模型评测：
  `.venv/bin/python -m hwr.apps.evaluate_foundation_world_model <RUN_PATH>`
- runtime 合同诊断参考入口：
  `.venv/bin/python -m hwr.apps.evaluate_stale_observation_validity`

现有 `artifacts/development-ready.json` 的 source commit 为
`d6d9a43b187591c43262efe1938130b358d63799`，早于本轮起点，不可授权本轮正式训练。
任何训练必须在候选干净提交上重新运行完整门禁。

## 启动资源状态

- 主机显示为 Apple M5 Pro、20 核 GPU。
- `.venv` 为 Python 3.11.0，PyTorch 2.13.0。
- 当前沙箱探测 `torch.backends.mps.is_built() == True`，
  `torch.backends.mps.is_available() == False`；正式 MPS 候选必须在 host-exec 环境重新核验，
  不能据此启动或否定训练。
- 启动时没有 tmux session。
- 沙箱禁止 `ps` 进程枚举，因此正式运行前必须在 host-exec 环境确认加速器独占状态。
- 数据卷可用空间为 `131,205,392 KiB`，约 125 GiB。
- 旧 readiness 估计单次正式世界模型 run 约 28.41 GiB。

## 本轮组织

- 创新 Agent A：信息匹配、非 shuffle 的 action-discriminative 诊断或目标。
- 创新 Agent B：latency-aware scheduling 与独立安全层边界。
- 创新 Agent C：闭环域支持声明及更接近成功率的反例方向。
- 两名筛选 Agent 将在提案冻结后独立评分，完成前互不查看结果。
- 实施 Agent 只负责最终入选且冻结的单一候选。

## 本轮硬约束

1. 三个创新 Agent 独立提案；两个筛选 Agent 独立评分。
2. 所有新观点使用稳定 ID，并在提案、实现、提交、运行和结果间保持引用。
3. 先执行不更新参数的最低成本判别验证；只有预注册门槛通过才允许训练。
4. runtime/评测修复与能力训练必须分离；一个候选只改变一个主变量。
5. 不延长 100ms safety validity，不移除 latency=3，不用 latest/future observation 绕过
   延迟，不降低碰撞或安全门槛。
6. 不使用 action-shuffle/derangement family 作为训练监督或事后选择依据。
7. 不按结果挑 seed、任务、Episode、latency、shift、负例、权重或门槛。
8. 所有行为变化必须有测试；训练只从干净、已提交且通过项目门禁的提交启动。
9. 能力结论只能来自冻结未见分布上的闭环物理结果；诊断、loss、probe 或视频不能替代。
10. 正式长时训练必须经 `traex-host-exec` 与 tmux 启动，并设置带“阅读 AGENTS.md”的
    完成唤起和定时看门狗；小型验证不得休眠。
