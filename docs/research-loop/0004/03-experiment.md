# R0004 冻结实验

## `R0001-P29` stale observation validity 合同诊断

### 问题与结论边界

> 当前 20Hz runtime 把 action validity 绑定到 visible observation timestamp，并固定为
> 100ms；evaluation observation latency=3 时 visible timestamp 落后真实控制时钟 150ms。
> 这是错误的 bundle/timestamp 实现，还是评测域与安全动作合同的结构性不相容？

P29 只回答上述时间语义问题：

- 不修改模型、策略、动作、validity window、安全阈值、observation latency 或评测任务；
- 不运行能力训练或正式闭环能力评测；
- 不把“减少 safety rejection”记为能力改善；
- 不使用 latest observation 替代正式延迟 observation；
- 不覆盖 P11 的 `rejected` 结论。

### 唯一变量

新增一个只读时间线诊断与报告：

- 输入为 observation timestamp、runtime now、control period、固定 validity duration 和
  既有 P11 Episode 元数据；
- 输出为每个 latency 的预期 validity 判定、与 P11 实际 safety count 的一致性、合同分类；
- 正式 action chain 与所有既有 artifact 字节不变。

### 固定语义

- control frequency：`20Hz`；
- control period：`50,000,000ns`；
- `dual_arm_action_frame`：
  - `created_at = observation.timestamp_ns`；
  - `valid_from = observation.timestamp_ns`；
  - `valid_until = observation.timestamp_ns + 100,000,000ns`；
- safety 判定：
  `now < valid_from or now > valid_until` 时拒绝；
- observation latency `L` 的合成 runtime age：
  `age_ns = L * 50,000,000`；
- latency 0/1/2 应在窗口内；latency 3 应过期；
- 边界 `now == valid_until` 必须合法，`now == valid_until + 1` 必须拒绝。

不得把 created/valid-from 改为 inference completion time，不得延长 100ms，也不得降低
evaluation latency。这些都属于新的系统设计候选，不是 P29 诊断。

### 固定真实证据

- P11 run：
  `runs/research-loop/0003/r0003-p11-causal-plant-s20261101`
- source commit：
  `ef86971ecd9528c00022e3f944e7878f66665f4a`
- report SHA-256：
  `79195b94f48e8b59ce04bb7a9a3a680f8717f6484da69b2f28d54b3a9b842cda`
- manifest SHA-256：
  `509f1b7f11caa2aa0dced3324bcc6ea9af9b045071c85b1f6a2bc66a7160da3a`
- artifact：145 个，全部 SHA-256 和 bytes 必须复核通过；
- Episode：144 个，任务×rho×action latency 完整；
- observation latency 实际分层：
  - latency 1：18 Episode，safety intervention 全部 0；
  - latency 2：90 Episode，safety intervention 全部 0；
  - latency 3：36 Episode，每个 61 次，总计 2,196 次；
- 所有 Episode 均无提前终止和严重碰撞。

诊断不得读取 P11 artifact 之外的未见能力结果。

### 实现与测试范围

唯一负责人：主 Agent。

允许新增：

- `src/hwr/eval/stale_observation_validity.py`
- `src/hwr/apps/evaluate_stale_observation_validity.py`
- `tests/test_stale_observation_validity.py`

不得修改：

- `src/hwr/train/bimanual_runtime.py`
- `src/hwr/safety/dual_arm.py`
- `src/hwr/adapters/mujoco/**`
- 配置、训练代码、P11 artifact 和历史轮次文档。

测试必须覆盖：

1. latency 0/1/2 合法、latency 3 过期；
2. exact boundary inclusive 与 `+1ns` rejection；
3. 真正 stale frame 仍输出 `outside_validity_window`；
4. report 中 144 Episode、18/90/36 分层和 0/0/2196 safety count；
5. 任一 P11 hash、artifact bytes、Episode coverage 或字段漂移即失败；
6. 不存在 action validity 延长、latest bundle 替换或 future timestamp。

### 执行元数据

- proposal：`R0001-P29`
- 分支：`feat/research-loop`
- run：
  `runs/research-loop/0004/r0004-p29-stale-validity-s20262901`
- 命令：
  `.venv/bin/python -m hwr.apps.evaluate_stale_observation_validity`
- device：CPU；
- 资源预算：只读 JSON/manifest/hash 与常数时间线计算，预计小于 2 分钟；
- 不允许 tmux、后台、休眠或 `traex-host-exec`；
- 执行前要求实现、测试与本文已提交，工作区干净，run 路径不存在。

### 判定

`contract_incompatible` 需要全部满足：

1. 合成 latency 0/1/2 均合法，latency 3 均过期；
2. exact boundary 语义与 safety 实现逐项一致；
3. P11 145 个 artifact 全部 hash/bytes 通过；
4. P11 latency 1/2/3 的 Episode 数为 `18/90/36`；
5. safety intervention total 为 `0/0/2196`；
6. latency 3 每个 Episode 都为 61 次，latency 1/2 每个 Episode 都为 0；
7. 无 future timestamp、覆盖缺失或非有限值。

若全部满足：

- 接受“evaluation observation latency=3 与当前 observation-timestamp-based 100ms action
  envelope 结构性不相容”；
- P29 作为合同诊断 `accepted`，但不是能力或平台修复；
- 禁止延长窗口、移除 latency 或把 P11 改判；
- 下一步必须另行提出单变量、保持安全威胁模型的 latency-aware action scheduling 设计，
  或明确把 latency=3 定义为当前系统安全不可执行域。

`implementation_bug` 只在以下情况成立：

- 合成语义预测与真实 safety 判定不一致；或
- P11 observation latency 元数据与 safety count 不按上述分层一致；或
- runtime 在相同时间线上出现非确定判定。

若 `implementation_bug`：

- 只允许另立纯运行时/评测修复；
- 修复后先重建无能力改动的 baseline；
- 不与 P26/P28 首次训练捆绑。

任一输入 hash、coverage 或 provenance 不完整则为 `inconclusive`，不得改阈值、补 Episode、
选择 seed 或重跑 P11。

## 后续候选冻结状态

- `R0001-P30`：与 R0001-P09 重复，拒绝，不实施。
- `R0001-P27`：筛选拒绝，不实施。
- `R0001-P28`：保留但未冻结；必须先完成 P29，再重新筛选修订合同。
- `R0001-P26`：保留但未冻结；排在 P28 后。

## 通用执行门

- 候选源码必须是干净、已提交状态；
- 行为变化必须有聚焦测试；
- `scripts/check_python_size.py` 必须通过；
- 相关测试和项目门禁必须通过；
- 正式训练与基线使用相同评测和可比预算；
- 不得看到结果后修改阈值、筛选 seed 或排除失败；
- 评测修复与能力改进必须分开归因。
