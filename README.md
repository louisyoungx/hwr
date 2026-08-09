# Housework Robot Training Platform

自有抽象、厂商无关的家务具身智能训练平台。目前已完成连续二维移动操作拟真环境、Episode 数据、行为策略训练、模型注册、闭环评测和三个家务场景的实际训练。

## 已实现

- 版本化 Observation、Action、Event 和 Episode；
- 仿真与未来真机共用的 `RuntimeBackend`；
- 独立安全动作过滤和有效期检查；
- 差速底盘、二维机械臂、夹爪、物体和障碍拟真；
- 固定种子确定性重放；
- Parquet 行为数据集和按 Episode 切分；
- 本机 MPS/CPU 行为克隆训练；
- 策略访问状态的数据聚合和专家纠正；
- 带校验和的本地模型注册表；
- 三个独立家务场景的闭环训练基准。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"

python3 scripts/check_python_size.py
python3 -m pytest
python3 scripts/verify_benchmarks.py
```

训练一个场景：

```bash
hwr-train-scenario tidy_table/v1 \
  --run-id local-tidy-table \
  --episodes 60 \
  --epochs 40 \
  --aggregation-rounds 2 \
  --aggregation-episodes 20
```

## 文档

- [平台架构与模块边界](docs/architecture.md)
- [训练与拟真环境方案](docs/training-and-simulation-plan.md)
- [万元内平台方案](docs/low-cost-platform-proposal.md)
- [训练基准与复现命令](benchmarks/README.md)

## 当前边界

当前拟真后端是连续二维运动与接触近似，目的是先验证平台抽象、数据闭环和多场景训练，而不是替代三维刚体物理。具体机械臂和硬件尚未选型。下一阶段应在保持核心协议不变的前提下，增加三维 `SimBackend` 适配器、视觉观测和真实硬件系统辨识。

