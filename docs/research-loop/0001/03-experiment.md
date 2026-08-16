# R0001 冻结实验

## 冻结状态

- 冻结日期：2026-08-15
- 主 Agent：当前 TRAE 主会话
- 分支：`feat/research-loop`
- 训练候选：`R0001-P01`
- 类型：平台修复后的当前谱系校准基线，`baseline-only`
- 测量合同：action probe 不变；动作执行验证使用实际 plant action；Actor readiness
  门槛不变
- 后续评测修复：`R0001-P04`，必须在 `P01` 旧统计基线完成后进入独立分支
- 行为候选：本文件冻结时不实施；只按 `02-review.md` 的失败指纹路由

第一次 run `r0001-p01-baseline-s20260812` 已因 `R0001-F01`～`R0001-F05`
平台缺陷记为 `inconclusive`，详见 `04-results.md`。v2 只用于建立修复后平台的当前
基线，不将 v1 与 v2 的差异解释为能力提升。

第二次 run `r0001-p01-baseline-v2-s20260812` 验证了有界系统辨识采集，但因
`R0001-F06` 将系统辨识专用的 `ρ=0.0` 激励误用于动作执行正例槽而记为
`inconclusive`。v3 将四档相关系数严格限制在系统辨识 phase；动作执行与碰撞留出使用
冻结正式随机探索参数 `ρ=0.96`。v1、v2 均无 checkpoint，不恢复。

第三次 run `r0001-p01-baseline-v3-s20260812` 证明 phase 隔离正确，但资源审计显示
启动留出逐步渲染四相机仅使用约一个 CPU 核和 27%～29% GPU，剩余启动阶段预计仍需
两小时以上，因此记为 `inconclusive`。v4 使用确定性两遍采集：搜索 pass 不渲染未保存
中间帧，命中后同 seed 重放并只渲染正式尾窗。v1～v3 均无 checkpoint，不恢复。

训练源码提交为本目录 `00-context.md`～`03-experiment.md` 完成原子提交后的
`feat/research-loop` HEAD。Git 提交身份由随后生成的
`artifacts/development-ready.json`、run 内副本和 `run-manifest.json` 共同不可变记录；
门禁后若 HEAD、受保护源码或 foundation 配置变化，训练入口必须拒绝启动。

## `R0001-P01` 目标

不改变现有采集、模型、优化、门槛或安全行为，回答：

> 当前谱系是否能在固定 24 Episode 校准判定点前，形成跨三个正式任务、跨
> `1/4/8/16` 步的动作可辨识性，通过单步物理动作利用和动作执行验证，连续两周期
> 解锁并稳定 warm-up 独立探索 Actor？

该实验不能证明家务能力提升。若同一 run 通过 24 Episode 校准判定，则继续执行冻结的
120 Episode 正式预算，以便后续判断任务 Actor、deployment 和闭环评测是否可达；不得
在通过后人为把它改成 24 Episode 特制 run。

## 负责人和所有权

| 范围 | 负责人 | 分支或运行 | 文件所有权 |
|---|---|---|---|
| 基线冻结、门禁、运行、结果归因 | 主 Agent | `feat/research-loop` / `r0001-p01-baseline-v4-s20260812` | `docs/research-loop/0001/`、运行与日志索引 |
| `R0001-P04` 统计修复 | 后续唯一实施 Agent | `eval/R0001-P04-bootstrap` | `src/hwr/train/foundation_action_probe.py` 及对应测试 |
| 条件行为候选 | 尚未分配 | 独立 `exp/R0001-*` worktree | 只在读取 `P01` 失败指纹后分配 |

在 `P01` 完成前，不得创建 `P02/P03/P05/P06/P08` 的正式实现，不得启动候选训练。

## 起始门禁

### 工作区要求

- 当前分支严格为 `feat/research-loop`。
- `git status --porcelain` 为空。
- 研究冻结文档已经提交。
- 不存在其他 foundation 训练或开发总门禁进程。
- 可用磁盘不低于 35 GiB，且静态估算通过 30 GiB 配置上限。

### 门禁命令

```bash
.venv/bin/python scripts/verify_development_ready.py \
  --foundation-device cpu \
  --model-root models/foundation \
  --output artifacts/development-ready.json
```

门禁必须产生：

- schema：`hwr.foundation-development-ready/v3`；
- `training_unlocked=true`；
- 精确 11 项必需检查全部 `passed=true`；
- `source_commit` 等于门禁执行时 HEAD；
- committed-snapshot 的架构、Python 尺寸、全量测试和训练语义检查绑定同一提交；
- 三个冻结基础模型的本地权重校验和与真实 CPU 推理通过。

门禁失败时，`P01` 记为 `inconclusive`。只允许修复可复现的平台问题；任何评测修复或
能力改动必须重新冻结，不得直接继续该对照。

## 冻结配置

### 训练与模型

所有 foundation 配置保持仓库当前值，关键项如下：

| 项目 | 冻结值 |
|---|---:|
| 正式总 Episode | 120 |
| 校准判定 Episode | 24 |
| 每周期 Episode | 3 |
| 每周期 update | 200 |
| batch size | 2 |
| 序列 transition | 16 |
| Replay transition 容量 | 18,000 |
| 每 Episode Replay 窗口 | 7 |
| 每 Episode 视觉监督窗口 | 2 |
| action probe horizon | 1、4、8、16 |
| 动作相关系数 | 0.96 |
| 夹爪翻转概率 | 0.05 |
| 探索准入连续通过 | 2 个周期 |
| 最少 Replay source Episode | 12 |
| probe 点估计门 | 1.05 |
| probe Episode bootstrap `p05` 门 | 1.01 |
| 物理动作因果比 | 1.05 |
| 物理因果恶化 horizon 比例 | 0.60 |
| 活跃动作维度比例 | 0.75 |
| 动作有效秩 | 6.0 |
| 探索 Actor warm-up | 最少 200、最多 1,000 update |

关键配置的原始文件 SHA-256 由开发门禁记录。冻结时额外核对：

- `configs/foundation/online-training-v1.json`：
  `a25aa0e0cda96afc6321fcbf23c7f8d11a0a248a15c5e17a221f578a63b594ad`
- `configs/foundation/unified-trainer-v1.json`：
  `1028a3cf88afc62eb89a5c9170ba737cadd31593d2d8325942b33ba641a9f6b4`
- `configs/foundation/world-model-v1.json`：
  `50b6724e7d105436744c165a642f7761984674c5673c6fce562d4e8eadb0b7f1`
- `configs/foundation/world-objective-v1.json`：
  `0c77b3d95726b81f2662c04a760c6a3d00bad6e2a87ee7dd29f9fbb31cc18597`

### 任务

三个任务共享同一模型、Replay 和优化器：

1. `clear_dining_table_3d/v1`
2. `store_kitchen_items_3d/v1`
3. `tidy_living_room_3d/v1`

不得加入专家、演示、行为克隆、脚本动作、任务阶段、对象 token、目标 token 或旧
checkpoint。

### 训练 seed

- 本次基线训练 seed：`20260812`
- 第 `i` 个训练 Episode 的环境和动作 seed：
  `20260812 + i * 104729`，`i ∈ [0, 119]`
- 前 24 个预注册 seed：

```text
20260812, 20365541, 20470270, 20574999, 20679728, 20784457,
20889186, 20993915, 21098644, 21203373, 21308102, 21412831,
21517560, 21622289, 21727018, 21831747, 21936476, 22041205,
22145934, 22250663, 22355392, 22460121, 22564850, 22669579
```

系统辨识、动作执行和后续碰撞留出 seed 由
`src/hwr/train/foundation_holdout.py::_holdout_seed` 的冻结公式生成，与训练 Episode
seed 分离。不得根据结果换 seed 或重跑挑选。

单 seed 足以建立当前校准基线，但不能形成跨 seed 稳定性或正式能力结论。正式未见分布
验收仍要求至少三个不同训练 seed 的独立通过结果。

## 设备与资源预算

- 主机：Apple M5 Pro，18 CPU 核、20 GPU 核、48 GB 统一内存。
- Python：3.11.0。
- PyTorch：2.13.0。
- MuJoCo：3.10.0。
- 训练设备：`mps`。
- 冻结教师特征设备：`mps`。
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65`。
- `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.50`。
- 进程 nice：`0`。不主动降低训练优先级；训练期间独占可用加速器。
- 正式训练期间独占本机加速器，不并行启动其他训练 run。
- 静态存储估算：`28.409005165100098 GiB`。
- 配置存储上限：`30 GiB`。
- 最低空闲空间：`35 GiB`。
- 冻结前实测空闲空间：约 `190 GiB`。

若资源门禁失败、MPS OOM、磁盘低于下界或出现不可恢复的非有限值，停止并保留证据，
结果记 `inconclusive`；不得通过降低模型、图像、任务、序列、留出或安全标准继续。

## 正式训练命令

run ID：

```text
r0001-p01-baseline-v4-s20260812
```

命令：

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.50 \
.venv/bin/python -m hwr.apps.train_foundation_world_model \
  --run-id r0001-p01-baseline-v4-s20260812 \
  --output-root runs/foundation-world-model \
  --device mps \
  --foundation-device mps \
  --seed 20260812 \
  --development-ready artifacts/development-ready.json \
  --model-root models/foundation
```

训练使用 `traex-host-exec` 的 Host-owned 后台续接流程。监督器在门禁成功后立即串行启动
训练，并在门禁或训练退出时即时唤起当前线程。独立看门狗从训练进程真实启动后计时：
15 分钟执行第一次检查，之后每 60 分钟唤起一次。每次只检查同一 run 的进程、CPU/GPU/
内存/磁盘、`metrics/latest.json`、周期指标、日志和 checkpoint；正常则继续等待，不重复
启动。若指标、checkpoint 和日志长时间无进展，或效果已满足预注册停止条件，则主 Agent
停止训练并进入归因。只有已发布 `latest.json` 且恢复合同通过时才可使用同一命令加
`--resume`；看门狗不得盲目重启训练进程。

## 24 Episode 校准判定

### 通过

在不晚于第 24 Episode 的周期指标中同时满足：

1. 三个任务的每个 `1/4/8/16` 步 probe 点估计均不低于 `1.05`；
2. 每任务跨 horizon 保守 Episode bootstrap `ratio_p05` 不低于 `1.01`；
3. 单步视觉潜变量与本体的动作 shuffle 因果评估通过；
4. 活跃动作维度比例不低于 `0.75`，有效秩不低于 `6.0`；
5. 动作执行模型独立留出验证通过；
6. 上述探索检查连续两个周期通过；
7. 探索 Actor 已解锁，warm-up 通过且 update count 至少为 200。

通过后不宣称能力改善，同一 run 按冻结配置继续到正式终点。

### 未通过

若 runner 在 24 Episode 后以
`foundation calibration stopped early after missing evidence` 停止，且运行产物完整，则
这是有效否定结果，不是基础设施异常。记录全部失败检查，不启动完整 120 Episode。

### 实验失效

以下情况记 `inconclusive`：

- 门禁、留出采集、特征物化、checkpoint 或恢复合同错误；
- 非有限损失、MPS OOM、磁盘或进程异常导致未到判定点；
- 源码、配置、模型权重、seed、任务或安全层发生冻结外变化；
- 结果文件缺失或无法将数据、checkpoint、因果报告和源码提交连成哈希链。

## 120 Episode 与闭环评测

只有 `P01` 通过校准并产生 causality-qualified deployment，才进入当前轮后续评测。

单 run 未见 seed 评测命令：

```bash
.venv/bin/python -m hwr.apps.evaluate_foundation_world_model \
  runs/foundation-world-model/r0001-p01-baseline-v4-s20260812 \
  --output-root runs/foundation-world-model-eval \
  --evaluation-id r0001-p01-baseline-v4-s20260812 \
  --seed-count 40 \
  --device mps \
  --video-seed-count 1
```

最终能力接受仍要求三个不同训练 seed、相同不可变配置、互斥训练与留出 seed、同一组未见
评测 seed，并通过：

```bash
.venv/bin/python -m hwr.apps.aggregate_foundation_evaluations \
  <evaluation-1> <evaluation-2> <evaluation-3> \
  --output runs/foundation-world-model-eval/r0001-p01-aggregate
```

本轮首个单 seed 结果即使闭环通过，也只能标记为单 seed 证据；没有三 seed 聚合前，不得
宣称正式家务能力通过。

## `R0001-P04` 冻结入口

`P04` 只能在 `P01` 产生冻结 Replay 与系统辨识留出后开始：

- 从 `P01` 测量基线提交建立 `eval/R0001-P04-bootstrap` 独立 worktree；
- 只修改 probe bootstrap 与对应测试，不修改训练、数据或门槛；
- 同一份冻结数据发布旧、新双报告；
- 每任务四个 horizon 共用同一 Episode multiplicity；
- 增加零效应、单 horizon 失败、相关 horizon、异方差和动作打乱负对照；
- 若统计修复被接受，必须用新测量合同重新建立后续基线；
- 旧新门禁变化不得计入能力候选收益。

## 条件候选路由

- 数据 probe 点估计失败，且无任务失衡或执行链漂移解释：短验证 `R0001-P02`。
- 数据 probe 通过、世界模型动作利用失败：冻结 Replay 验证 `R0001-P05`。
- 最弱任务等于样本最少任务：先做 `R0001-P03` 离线功效诊断。
- proposal/execution 漂移主导：先补执行链 provenance，再重审 `R0001-P08`。
- `R0001-P06` 只有 `P05` 被否定后才允许重新冻结。
- `R0001-P07` 不进入本轮正式实验。

任何条件候选必须另建分支或 worktree、指定唯一实施负责人并补充自己的冻结合同；不得在
`P01` run 内切换条件。

## `R0001-P04` 实施后冻结证据

- 实现提交：`b665c9d96049d80e1951c6a8e941af4695d23d2a`
- 类型：隔离评测修复，不是能力候选。
- action probe schema：`hwr.foundation-data-action-probe/v4`
- bootstrap 合同：
  `shared-holdout-episode-multiplicity-across-horizons/v1`
- 同一任务四个 horizon 的每个 replicate 共用相同 holdout Episode multiplicity，再在
  replicate 内取最弱 horizon。
- v4 冻结数据旧/新双报告路径：
  - `diagnostics/action-probe-p04-v3.json`
  - `diagnostics/action-probe-p04-v4.json`
  - `diagnostics/action-probe-p04-comparison.json`
- 双报告必须满足点估计、绝对 MSE、训练/留出数量和 Episode 列表零差异。
