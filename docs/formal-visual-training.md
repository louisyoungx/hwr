# 正式三维视觉训练运行说明

> 状态：历史 V1/V2 基线；不再代表当前训练架构
>
> 当前方案：[端到端视觉—语言—动作训练范式](./end-to-end-training-paradigm.md)
>
> 配置：[formal_visual_v1.json](../configs/training/formal_visual_v1.json)
> 训练结果：[formal_visual_v1_results.json](../configs/training/formal_visual_v1_results.json)

首轮普通行为克隆在餐厅未见种子 30000 上能导航到餐桌附近，但混淆机械臂操作与后续导航阶段，6000 步超时且没有夹持接触。该结果保留为 V1 基线。V2 曾通过训练标签 `phase` 增加阶段分类和分阶段动作头；这一做法只保留作历史对照，后续端到端 Actor 不读取、预测或监督人工任务阶段。

V2 数据与训练配置见 [formal_visual_v2.json](../configs/training/formal_visual_v2.json)。三个 V2 数据集仍使用相同的 9 个成功专家 Episode，但按 `hwr.visual-behavior-dataset/v2` 重采并锁定新的 shard 哈希。

## 数据边界

三个任务分别使用 3 条接触有效的专家 Episode。正式策略输入固定为头部 RGB、头部深度、腕部 RGB、24 维本体状态、任务指令 ID 和 8 步动作历史；引擎真值位姿、目标位置、专家阶段和成功状态只作为标签或审计信息，不进入训练张量。

| 任务 | 训练种子 | 样本数 | 数据目录 |
|---|---:|---:|---|
| 餐桌清理 | 1000–1002 | 2485 | `datasets/formal-v1-r4/clear_dining_table_3d_v1-expert-s1000` |
| 起居室收纳 | 2000–2002 | 2198 | `datasets/formal-v1-r5/tidy_living_room_3d_v1-expert-s2000` |
| 厨房入柜 | 3000–3002 | 4059 | `datasets/formal-v1-r6/store_kitchen_items_3d_v1-expert-s3000` |

数据目录由 Git 忽略；受版本管理的训练配置保存每个 shard 的 SHA-256。训练前加载器会再次检查 manifest、字段白名单和 shard 哈希。

## 本机训练命令

每个任务独立训练、登记和重载，避免不同任务的动作分布互相掩盖：

```bash
.venv/bin/python -m hwr.apps.train_formal_visual \
  --dataset datasets/formal-v1-r4/clear_dining_table_3d_v1-expert-s1000 \
  --run-id formal-v1-dining-s0 --epochs 30 --batch-size 128 --seed 0

.venv/bin/python -m hwr.apps.train_formal_visual \
  --dataset datasets/formal-v1-r5/tidy_living_room_3d_v1-expert-s2000 \
  --run-id formal-v1-living-s0 --epochs 30 --batch-size 128 --seed 0

.venv/bin/python -m hwr.apps.train_formal_visual \
  --dataset datasets/formal-v1-r6/store_kitchen_items_3d_v1-expert-s3000 \
  --run-id formal-v1-kitchen-s0 --epochs 30 --batch-size 128 --seed 0
```

训练器在本机自动选择 MPS、CUDA 或 CPU，保存最佳验证 checkpoint，并立即从磁盘重载一次。`models/` 和 `runs/` 中的大文件由 Git 忽略；完成训练后需把模型哈希、设备、损失和评测种子写入受版本管理的运行清单。

首轮三套策略已在本机 MPS 上各训练 30 epoch 并完成磁盘重载。受版本管理的结果清单记录训练代码提交、训练种子、损失和 checkpoint SHA-256；模型文件仍保存在 `models/formal-v1/`。这些结果只证明本机训练链路跑通，不代表闭环门槛已经达成。

## 闭环评测门槛

- 每个任务使用与训练集合不相交的 20 个种子；
- 推理动作来源必须以 `learned:` 开头；
- 成功率至少 70%；
- 严重碰撞总数为 0；
- 每个成功 Episode 的目标状态连续稳定至少 2 秒；
- 视频必须来自同一个 checkpoint、种子和未剪辑 Episode。

离线损失下降只证明训练器工作，不代表家务任务完成。只有 checkpoint 重载后的闭环报告满足上述门槛，训练阶段才算通过。
