# Housework Robot Training Platform

自有抽象、厂商无关的家务具身智能训练平台。现有连续二维环境只作为软件链路 smoke test；正式工作已转入带刚体接触、视觉观察和真实三维家庭资产的 V1 平台，二维结果不计入三维拟真验收。

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
python3 -m pip install -e ".[dev,video]"

python3 scripts/check_python_size.py
python3 scripts/check_architecture.py
python3 -m pytest
python3 scripts/verify_benchmarks.py
```

三维开发环境与机器人模型验证：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,video,sim3d]"
.venv/bin/python scripts/verify_robot_model.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m pytest tests/test_mujoco_backend.py
.venv/bin/python -m hwr.apps.render_3d_smoke \
  --output-path artifacts/3d-smoke.png
.venv/bin/python -m hwr.apps.verify_contact_grasp \
  --output-path artifacts/contact-grasp-smoke.mp4
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

复现三个已训练基准的闭环动作，并输出并排视频（需要系统安装 `ffmpeg`）：

```bash
hwr-render-benchmarks --output-path artifacts/benchmark-rollouts.mp4
```

## 文档

- [平台架构与模块边界](docs/architecture.md)
- [训练与拟真环境方案](docs/training-and-simulation-plan.md)
- [万元内平台方案](docs/low-cost-platform-proposal.md)
- [训练基准与复现命令](benchmarks/README.md)
- [三维拟真 V1 实施与验收合同](docs/three-dimensional-v1-acceptance.md)
- [三维引擎架构决策](docs/adr/0001-mujoco-3d-backend.md)

## 当前边界

二维后端不代表家务仿真能力。三维 V1 尚在实施中；在三个带纹理家庭场景、六轴移动机械臂、非特权视觉策略、真实接触操作和隔离种子评测全部通过前，项目不会宣称完成拟真家务训练平台。
