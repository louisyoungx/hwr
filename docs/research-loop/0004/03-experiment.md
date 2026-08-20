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
- `R0001-P28`：两名复审 Agent 均要求结构性修改，拒绝执行。
- `R0001-P26`：保留但未冻结；排在 P28 后。

## `R0001-P28` prior-feature successor head-only 可学习性门

### 状态与问题

状态：`rejected before implementation after independent re-review`。

以下内容永久保留为未执行草案与反例证据，不再具有执行权限。

> 在冻结 RSSM 与 visual student 后，只用 true actual action 生成的一步 prior feature，
> fresh physical heads 能否跨 source Episode 学会 successor visual/proprio target，并在
> 从未进入训练的两类经验负动作上稳定表现更差？

该实验只决定是否允许正式 P28 世界模型训练，不宣称能力改善。

### 对原提案的修订

- 纠正比值方向：统一使用 `negative_error / true_error`，门槛为 `>=1.05`。
- 不再把 posterior-trained frozen head 直接用于 prior feature；该用法是 OOD，不能作为
  否决门。
- 使用 fresh heads 的两臂：
  - A `posterior_oracle`：从 frozen successor posterior feature 预测 successor target；
  - B `prior_candidate`：从 frozen `(h_t,z_t,actual_action_t)` 一步 prior feature预测同一
    successor target。
- 两臂使用完全相同的 source split、target、head 架构、初始化 seed、优化器、更新数和
  batch schedule；唯一输入变量是 posterior feature 或 prior feature。
- 训练只使用 true actual action；负动作仅在验证时出现，不向 head 提供 action、任务 ID、
  source ID、slot、时间索引或负例标签。

### 固定输入

- frozen Replay：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/replay/autonomous`
- Replay manifest SHA-256：
  `c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985`
- frozen checkpoint：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/checkpoints/update-000001600`
- checkpoint manifest/artifact SHA-256：
  `72f9361762d7ff5086f086b9ae1db05396caa3cf91822ece20686095df4ad75b` /
  `ef24bdfcca3cc46274bdfebc1d8b1a4afc81c73abff3aa4128e393e6da2109c6`
- vision-language/dense-vision/language index SHA-256：
  - `496f48e714f18714a9da0cde3bcfd9157b17f9f4a46f06122a6b6ab8df3cc39a`
  - `5652d92b03c6cb26c6e30e5b0408f812163ee10e551790c9e97586f2dcd03ac4`
  - `bb4d2828f4ef18452751175d04ac131d49f7eb16d210f769ad3eb7c4f25ce460`
- 24 个 source Episode、168 个 16-transition window 全部使用；
- 有序 window identity SHA-256：
  `d1e484bbb1f2c6357274044a3efbde8cbdf82d56aca4c0002e028136b37bf5a5`
- split + window full identity SHA-256：
  `edffd3784c43a182a7805fa2d96255453ebdb7d552ab766fb7204865789f54bc`
- split seed：`20262801`。

不得读取 causality holdout、P24 失败标签、P29 latency 分层或任何最终闭环结果。

### 三折 source-level split

每折 validation 固定 8 个 source Episode，任务配额 `2/2/4`；其余 16 个训练。
同一 source 的 7 个 window 不能跨折。

fold identity SHA-256：

- fold 0：
  `1041124e1b8bc63f30a31eea7b7b1d0f6e213363afb6d6dbbee6c27e728f0790`
- fold 1：
  `2790a27cb93eb4bf1af3f009ef12ab262358f94613b191621b37feb7df2f596a`
- fold 2：
  `20ae62b2126336a3fa3734af0d292dfb43072c796433d61ad067fe80d1050f27`

每个 source 恰好 validation 一次；结果按 24 个 source Episode 聚合，不把 window、
transition、fold 或初始化 seed 当独立统计单位。

### Frozen feature 与 target

- 用 checkpoint frozen visual student 生成 17 个 observation 的 256-D visual target。
- 用 frozen world model `observe` 获得 posterior `(h_t,z_t)` 与 successor posterior feature。
- prior feature：
  - 起点为每个 transition 的 frozen posterior `(h_t,z_t)`；
  - 只输入 Replay 记录的 true actual executed action；
  - `sample=False`；
  - feature 为 `[h_{t+1},z_{t+1}]`。
- visual target：successor visual student pooled state `t+1`。
- proprio target：Replay successor 37-D proprioception `t+1`。
- current-state target 只用于描述性 innovation；不能进入 head 输入。

### Fresh head 与优化

每个 arm、fold、initialization seed 独立创建 visual/proprio head：

`Linear(1536,512) -> LayerNorm(512) -> SiLU -> Linear(512,target_dim)`。

固定 initialization seeds：

- `20262801`
- `20367530`
- `20472259`

固定优化：

- optimizer：AdamW；
- learning rate：`1e-3`；
- weight decay：`1e-4`；
- updates：`400`；
- transition batch size：`256`；
- maximum gradient norm：`100`；
- batch schedule 由
  `SHA256("<init_seed>:<fold>:<update>:<training_source>:<slot>:<transition>")`
  固定排序后循环取样；
- visual loss：与 production 一致的 `MSE + (1-cosine)`；
- proprio loss：与 production 一致的 MSE；
- 两个 head 无共享参数；总 loss 只作一次 optimizer step；
- 不做 early stopping、checkpoint 选择、learning-rate scan、seed 选择或 loss reweight。

posterior/prior 两臂对应 head 必须由同一初始 state dict 克隆，且使用相同 target 与 batch
schedule。frozen visual student、world model 和输入 tensor 均不得获得梯度。

### 验证专用负动作

负动作不参与训练、调参或 checkpoint 选择。

family A `within_window_reverse`：

- 对每个 16-transition validation window，把 executed action 时间轴完全反转；
- 16 为偶数，因此没有位置固定点；
- 保留该 window 的精确经验 action multiset。

family B `source_slot_rotate`：

- 每个 validation source 的 7 个 window 按 reservoir slot 排序；
- slot `j` 使用 slot `(j+1) mod 7` 的完整 action 序列；
- transition index 保持；
- 保留 source Episode 的精确 112-action multiset；
- 没有 window 固定点。

正式 action-shuffle audit 使用全 batch/time 的 deterministic global derangement；P28
不得使用该算法、seed 或 pairing。两类验证负动作与正式 audit 算法不同。

### 误差与 source 统计

每个 head、arm、fold、seed 在 validation transition 上计算：

- visual per-transition error：`MSE + (1-cosine)`；
- proprio per-transition error：MSE；
- source error：该 source 全 7 window、112 transition 的算术平均；
- prior negative ratio：
  `family_error / true_action_error`；
- constant-mean baseline：
  只用该 fold 训练 source target 均值预测 validation target；
- posterior oracle relative fit：
  `posterior_true_error / constant_mean_error`；
- prior absolute fit：
  `prior_true_error / constant_mean_error`；
- prior/oracle gap：
  `prior_true_error / posterior_true_error`。

proprio 另报告：

- prior prediction innovation：
  `prediction_true - current_visible_proprioception`；
- observed innovation：
  `successor - current_visible_proprioception`；
- 在可控 16 维上计算 source-level cosine；零 norm 不使用 floor，source 技术失效。

每个 source 的三个 initialization seed 先分别报告，再对 log ratio/error ratio 求均值；不能
选择 seed。

### Posterior oracle 守护

每个 head 的 oracle 必须同时满足：

- 每个 initialization seed 的 aggregate
  `posterior_true / constant_mean <=0.80`；
- source 通过数至少 20/24，任务配额 `5/6、5/6、10/12`；
- 所有 loss 与 gradient 有限；
- 三折 source coverage 与 hash 完整。

oracle 失败表示 head/data/预算不足，P28 为 `inconclusive`；不能增加 updates 或改学习率
重跑。

### Prior candidate 通过条件

visual 与 proprio 分头判定，但 P28 整体要求两头都通过：

1. 每个 initialization seed：
   - aggregate `prior_true / constant_mean <=0.80`；
   - aggregate `prior_true / posterior_true <=1.25`；
   - family A 与 B 的 aggregate `negative/true >=1.05`。
2. 三 seed 平均后的 source 联合通过数至少 20/24，任务配额
   `5/6、5/6、10/12`；单 source 需要：
   - `prior_true/constant_mean <=0.80`；
   - `prior_true/posterior_true <=1.25`；
   - 两个 family 的 `negative/true >=1.05`。
3. 每个 head×family 对 24 source 做 10,000 次 Episode-block bootstrap，seed
   `20262801 + head_index*104729 + family_index*1009`，
   mean log ratio 的 95% percentile CI lower `>=log(1.01)`。
4. proprio 三任务的 source-mean innovation cosine 都 `>0`，aggregate至少 `0.20`。
5. frozen model/visual student hash 前后逐元素一致；head 之外无参数更新。

### 拒绝与失效

- oracle 通过但任一 prior 条件失败：P28 `rejected without world-model training`。
- oracle 不通过、输入/hash/coverage 漂移、零 norm、非有限值或实现越界：P28
  `inconclusive`；只允许修测量实现，不改预算/门槛/seed。
- 不得从单头、单 family、单 fold、单 seed 或 source 子集选取成功。
- 不得因 P29 的 latency=3 结果删除或重分训练 source。

### 执行草案

- 允许新增：
  - `src/hwr/eval/prior_feature_successor.py`
  - `src/hwr/apps/evaluate_prior_feature_successor.py`
  - `tests/test_prior_feature_successor.py`
- 不修改 production world model、objective、trainer、Replay、配置或历史文档。
- run：
  `runs/research-loop/0004/r0004-p28-prior-successor-s20262801`
- device：MPS；
- 资源预算：frozen feature materialization + 3 folds × 3 seeds × 2 arms 的 fresh head
  训练，单次前台短实验，预计小于 30 分钟；
- 在筛选 Agent 复审批准、实现提交、聚焦测试、size/architecture gate 和全量项目门禁通过前
  不得运行。

### 复审后终止

两个独立复审均为 `changes_required`，共同阻断包括：

- successor posterior 结果可见性使 oracle 对 prior 不公平；
- `sample=False` 与拟议训练采样路径不一致；
- 两个时间重配负例族仍可能复刻正式 audit 并携带位置/幅值/时距指纹；
- split manifest、batch schedule、metric、可控维映射和 seed/source 聚合不够唯一；
- 缺少 action-blind/current-state 信息匹配 control；
- head-only 通过不能直接授权正式 world-model objective 改动；
- MPS 时间、显存和 CPU parity 未经 smoke 证明。

因此：

- 不新增 P28 实现文件；
- 不创建 P28 run 目录；
- 不运行 head-only 或 world-model 训练；
- 不根据复审意见继续扫描负例、oracle、采样方式、预算或门槛；
- P28 状态为 `rejected before implementation after independent re-review`。

## 通用执行门

- 候选源码必须是干净、已提交状态；
- 行为变化必须有聚焦测试；
- `scripts/check_python_size.py` 必须通过；
- 相关测试和项目门禁必须通过；
- 正式训练与基线使用相同评测和可比预算；
- 不得看到结果后修改阈值、筛选 seed 或排除失败；
- 评测修复与能力改进必须分开归因。
