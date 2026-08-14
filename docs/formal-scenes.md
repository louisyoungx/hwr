# 正式三维家务场景

本文件记录 `household_v1` 的场景装配事实。图库只是资产与相机检查，不是训练成绩；只有后续从已加载学习策略执行的完整 Episode 视频才算任务复现。

## 场景清单

| 场景 | 实际尺度与主要家具 | 物理操作物 | 目标/可动家具 |
|---|---|---|---|
| `living_room_3d/v1` | 6.0 m × 5.0 m；皮质沙发、实木茶几、藤编收纳篮、地毯 | 橡皮鸭、足球 | 有底和四壁碰撞体的收纳篮 |
| `dining_room_3d/v1` | 6.4 m × 5.4 m；实木圆桌、三把餐椅、餐边柜 | 陶瓷杯、木盘 | 分离的杯托与盘托目标体积 |
| `kitchen_3d/v1` | 6.8 m × 5.6 m；三组木质柜架、岛台、操作台 | 两种不同网格/质量的清洁剂瓶 | 左右分仓抽屉；0.42 m 无执行器滑轨 |

所有外部网格均经过 Y-up 到 Z-up 转换、米制缩放和 XY 居中，详见
`assets/manifests/household_v1_sources.json` 与锁文件。正式 MJCF 对每件操作物同时声明：

- 带 UV/纹理、关闭碰撞的高细节可见 mesh；
- 独立的简化碰撞几何、质量和摩擦参数；
- 由 MuJoCo 管理的自由关节，Episode 内不允许任务代码写位姿。

厨房抽屉只有带限位、阻尼和摩擦的 slide joint，没有 actuator、weld 或 equality constraint。当前阶段只证明其物理结构；后续训练/评测还必须提供夹爪接触拉开抽屉的接触日志和原始视频。

## 正式训练与评测运行时

`hwr.adapters.mujoco.formal_household_backend.MujocoFormalHouseholdDualArmBackend`
把以上三套资产接入统一 16 维双臂运行时。训练和最终评测入口均直接加载
`configs/tasks/formal_3d_v1.json`，不再以简化篮子、托盘或抽屉代理任务替代正式场景。

每个任务包含 4 条训练指令和 3 条不重叠的评测改写。训练随机化覆盖物体质量/摩擦、光照、
材质、RGB/深度噪声、相机安装位姿、焦距、执行器比例以及动作/观测延迟；最终评测自动使用
更宽的分布外区间。随机化只改变环境，不为策略生成动作或任务阶段。

成功需要所有操作物在各自目标体积内稳定 2 秒、严重碰撞为零、左右夹爪分别存在真实双指
接触，并至少有 0.5 秒左右臂同时接触。厨房还要求无执行器抽屉被物理拉开到 0.30 m。所有
字段只进入环境奖励、终止和只读审计；Actor 仍只看到四相机、本体、动作历史和原始指令。

## 复现资产图库

```bash
python -m pip install -e '.[sim3d,assets3d,video]'
python scripts/verify_3d_assets.py
python -m hwr.apps.render_formal_scenes --output artifacts/formal-scenes.png
```

图库每行左侧为独立证据相机，右侧为同一模型、同一物理时刻的机器人头部 RGB。图上明确标记 `not a trained rollout`，不能拿它替代策略闭环视频。
