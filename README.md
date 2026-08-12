# Housework Robot Training Platform

自有抽象、厂商无关的家务具身智能训练平台。现有连续二维环境只作为软件链路 smoke test；正式工作已转入带刚体接触、视觉观察和真实三维家庭资产的 V1 平台，二维结果不计入三维拟真验收。

## 已实现

- 版本化 Observation、Action、Event 和 Episode；
- 仿真与未来真机共用的 `RuntimeBackend`；
- 独立安全动作过滤和有效期检查；
- 差速底盘、二维机械臂、夹爪、物体和障碍拟真；
- 固定种子确定性重放；
- Parquet 行为数据集和按 Episode 切分；
- 16 维底盘—双臂—双夹爪动作契约；
- 可部署 VLA Actor、训练期特权双 Critic 和经验回放基础组件；
- 1.60 m 四轮中央箱体、左右各 6-DOF 机械臂、双钳形夹爪和头部/双腕相机的 MuJoCo 模型；
- 客厅收纳篮、餐厅托盘和厨房回弹抽屉三个必须双臂并发的物理任务；
- 无专家在线采样、仅保存自主 transition 的分层回放、自动课程、断点续训和 Actor 导出；
- 重载 Actor 的未见种子评测、左右单臂锁定消融及四视角同进程视频录制；
- 带校验和的本地模型注册表；
- 三个独立家务场景的历史二维闭环基准；
- 三个正式三维家庭场景的 12 个 CC0 纹理网格、许可/哈希锁和可重复转换工具；
- 客厅、餐厅、厨房 MJCF 装配，含独立视觉/碰撞几何与无执行器物理抽屉。

历史行为克隆、数据聚合和规则专家实现不再属于正式训练路线。P076～P080 的小型视觉前端、字符哈希语言输入和直接无模型 Actor-Critic 已确认为失败基线。当前主线重建为“基础模型连续表征、动作条件世界模型和想象空间强化学习”；所有开发与总门禁完成前不启动正式训练。

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
.venv/bin/python -m pip install -e ".[dev,video,sim3d,assets3d]"
.venv/bin/python scripts/fetch_3d_assets.py
.venv/bin/python scripts/verify_3d_assets.py
.venv/bin/python scripts/verify_robot_model.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m pytest tests/test_mujoco_backend.py
.venv/bin/python -m hwr.apps.render_3d_smoke \
  --output-path artifacts/3d-smoke.png
.venv/bin/python -m hwr.apps.verify_contact_grasp \
  --output-path artifacts/contact-grasp-smoke.mp4
.venv/bin/python -m hwr.apps.render_formal_scenes \
  --output artifacts/formal-scenes.png
```

复现三个历史二维基准的闭环动作，并输出并排视频（只用于软件链路回归，需要系统安装 `ffmpeg`）：

```bash
hwr-render-benchmarks --output-path artifacts/benchmark-rollouts.mp4
```

## 文档

- [平台架构与模块边界](docs/architecture.md)
- [基础模型感知、世界模型与想象强化学习范式](docs/foundation-world-model-training-paradigm.md)
- [历史端到端训练范式](docs/end-to-end-training-paradigm.md)
- [本机训练进度与阶段诊断](docs/training-progress.md)
- [训练与拟真环境方案](docs/training-and-simulation-plan.md)
- [万元内平台方案](docs/low-cost-platform-proposal.md)
- [训练基准与复现命令](benchmarks/README.md)
- [三维拟真 V1 实施与验收合同](docs/three-dimensional-v1-acceptance.md)
- [三维引擎架构决策](docs/adr/0001-mujoco-3d-backend.md)
- [正式三维场景与复现命令](docs/formal-scenes.md)

## 当前边界

二维后端不代表家务仿真能力。三维 V1 尚在实施中；在三个带纹理家庭场景、四轮双六轴机器人、无专家的非特权视觉策略、真实双臂接触操作和隔离种子评测全部通过前，项目不会宣称完成拟真家务训练平台。
