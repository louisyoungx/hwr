# 家务场景训练基准

## 准入标准

- 至少三个不同家务场景；
- 每个场景至少 20 个隔离种子闭环评测；
- 成功率不低于 70%；
- 平均碰撞次数为 0；
- 报告包含数据集校验和、完整训练配置和模型路径。

运行检查：

```bash
python3 scripts/verify_benchmarks.py
```

## 当前结果

| 场景 | 训练 Episode | 样本 | 闭环成功率 | 平均步数 | 平均碰撞 |
|---|---:|---:|---:|---:|---:|
| 桌面整理 | 100 | 36,374 | 100%（20/20） | 317.55 | 0 |
| 衣物分类 | 155 | 55,413 | 100%（20/20） | 323.95 | 0 |
| 餐具收纳 | 155 | 60,267 | 100%（20/20） | 339.95 | 0 |

每个模型均独立生成数据、训练和登记，没有把规则专家用于最终评测动作。规则专家只用于初始示范和策略访问状态的纠正标签。

## 可复现训练命令

```bash
PYTHONPATH=src python3 -m hwr.apps.train_scenario tidy_table/v1 \
  --run-id tidy-table-v3-mps --episodes 60 --eval-episodes 20 \
  --epochs 40 --batch-size 512 --device mps --seed 300 \
  --aggregation-rounds 2 --aggregation-episodes 20 \
  --expert-action-probability 0.3 \
  --report-path benchmarks/results/tidy-table-v1.json

PYTHONPATH=src python3 -m hwr.apps.train_scenario sort_laundry/v1 \
  --run-id sort-laundry-v2-mps --episodes 80 --eval-episodes 20 \
  --epochs 50 --batch-size 512 --device mps --seed 500 \
  --aggregation-rounds 3 --aggregation-episodes 25 \
  --expert-action-probability 0.35 \
  --report-path benchmarks/results/sort-laundry-v1.json

PYTHONPATH=src python3 -m hwr.apps.train_scenario clear_dishes/v1 \
  --run-id clear-dishes-v1-mps --episodes 80 --eval-episodes 20 \
  --epochs 50 --batch-size 512 --device mps --seed 600 \
  --aggregation-rounds 3 --aggregation-episodes 25 \
  --expert-action-probability 0.35 \
  --report-path benchmarks/results/clear-dishes-v1.json
```

数据、模型和运行时产物分别写入 `datasets/`、`models/` 和 `runs/`，这些大文件不提交 Git；可审计的小型结果报告保存在 `benchmarks/results/`。

