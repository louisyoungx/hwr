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
- 客厅双物体收纳、餐桌杯盘归位和厨房双瓶入抽屉三个正式多物体任务；
- 无专家在线采样、仅保存自主 transition 的分层回放、自动课程、断点续训和 Actor 导出；
- 锁定 SigLIP2、DINOv3 ViT-S/16 与 Qwen3-Embedding 的高分辨率连续感知缓存；
- 24.4M 参数视觉学生、动作条件 categorical RSSM 与想象空间强化学习；
- 动态腕部相机标定、自主序列 replay、统一三任务在线闭环及剥离式部署导出；
- 每任务训练/评测指令改写隔离，以及相机、深度、执行器和延迟的分布外评测；
- 13,440 transition 的显著交互优先 Replay，以及只依据实际 retained transition 的准入证据；
- 部署视觉融合梯度、正式任务入口、Replay 规模和动作边界的可执行开发硬门；
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

基础模型—世界模型主线必须先通过一次不可跳过的总开发门禁，再启动正式训练：

```bash
.venv/bin/python -m pip install -e ".[dev,video,sim3d,foundation]"
.venv/bin/python scripts/verify_development_ready.py \
  --output artifacts/development-ready.json
hwr-train-foundation-world-model --run-id foundation-wm-001 --device cpu
```

在门禁产生与当前提交、配置和受保护源码哈希一致的报告前，训练命令会直接拒绝运行。可先
单独运行不访问基础模型权重的训练语义检查：

```bash
.venv/bin/python scripts/verify_training_semantics.py
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
- [从玩具闭环到严肃研究平台的判定合同](docs/serious-platform-vnext.md)
- [历史端到端训练范式](docs/end-to-end-training-paradigm.md)
- [本机训练进度与阶段诊断](docs/training-progress.md)
- [训练与拟真环境方案](docs/training-and-simulation-plan.md)
- [万元内平台方案](docs/low-cost-platform-proposal.md)
- [训练基准与复现命令](benchmarks/README.md)
- [三维拟真 V1 实施与验收合同](docs/three-dimensional-v1-acceptance.md)
- [三维引擎架构决策](docs/adr/0001-mujoco-3d-backend.md)
- [正式三维场景与复现命令](docs/formal-scenes.md)

## 当前边界

二维后端不代表家务仿真能力。当前代码已经具备正式三维家庭环境、基础模型感知、世界模型、
想象 RL、语言/OOD 留出和可执行训练门禁，但尚未产生新谱系的训练成功证据。在三个带纹理
家庭场景、无专家非特权视觉策略、真实双臂接触、三个训练 seed 和隔离种子评测全部通过前，
项目只声明“严肃实验平台已实现”，不宣称机器人已经学会家务，也不外推为开放世界通用能力。
