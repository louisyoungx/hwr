# R0001 研究上下文

## 轮次身份

- 轮次：`R0001`
- 启动日期：2026-08-15
- 分支：`feat/research-loop`
- 起始提交：`fc10cca515a541b0fcb94c4284b13f800246f896`
- 起始工作区：干净
- 对照主线：`main` / `ca4e50e2156a3777cb6050c10fc242f60e89770c`
- 起始差异：仅 `AGENTS.md` 研究循环规则更新，无训练行为差异

## 模块边界与依赖方向

本轮保持既有单向依赖：

1. `hwr.core` 只定义项目自有 schema 与运行时合同；
2. MuJoCo、基础模型和未来设备 SDK 只位于适配器；
3. 仿真与未来真机共同实现项目自有 `RuntimeBackend`；
4. 世界模型、策略和训练代码不导入具体引擎；
5. 独立安全层过滤 Actor 提议动作，学习模型只能预测干预，不能取代硬安全约束；
6. 评测修复与能力改进不进入同一个因果对比。

本轮若新增模块，必须先在 `03-experiment.md` 冻结其职责、允许依赖和禁止依赖。

## 当前可信证据

### 平台基线

`artifacts/development-ready.json` 证明 `ca4e50e` 上的平台门禁通过，包括：

- 全量测试、架构检查与 Python 尺寸检查；
- DINOv3、SigLIP2 与 Qwen3-Embedding 锁定权重的真实推理；
- 正式三任务、16 维动作、部署剥离与无专家谱系检查；
- 部署视觉融合梯度和参数更新；
- Replay retained transition、动作边界及训练语义检查。

该报告绑定 `ca4e50e`，不是当前起始提交 `fc10cca`，因此不能直接解锁新训练。它只证明严肃实验平台可执行，不是能力成功基线。

### 历史失败基线

`foundation-wm-007` 只能作为负证据：

- 来源提交：`bb8f7430922f42fda387b0d00167997662dfd278`；
- 最后耐久状态：111 Episode、7,400 update；中断前观测到 7,490 update；
- 三个旧代理任务共 111 个 Episode，成功数均为 0；
- 所有 Episode 的动作来源仍为 `random_rl_exploration`；
- 探索 Actor 与任务 Actor 均未解锁；
- 动作覆盖有效秩约 15.78，说明“动作数值范围不足”不是主要解释；
- 最后阶段动作 probe 约为 1.002，多步动作因果约为 0.995，未通过门槛；
- 没有 deployment，且 Replay、checkpoint 和留出缓存已按失败运行清理；
- 任务、Replay 保留策略、动作 probe、正式留出和训练门禁均早于当前实现，不能恢复或充当当前对照。

`foundation-wm-006` 同样是旧谱系负证据：9 Episode、600 update，聚合动作打乱比约 1.000，未通过动作因果门。

## 当前能力结论

当前没有与 `fc10cca` 行为等价、在三个正式家庭任务上完成训练和未见种子闭环评测的可信基线。

因此本轮不得宣称任何能力提升。第一项正式实验必须先建立最小可复现的当前谱系基线；若基线在冻结的校准预算内无法解锁探索 Actor，则结果应记为 `rejected` 或 `inconclusive`，而不是继续消耗完整训练预算。

## 主要瓶颈

主要瓶颈定义为：

> 当前实现能否在三个正式家庭任务的自主数据中形成跨任务、跨 Episode、跨 1/4/8/16 步均可复核的动作可辨识性，并在固定校准预算内解锁独立探索 Actor，尚无实证。

选择依据：

1. 旧运行已有充分动作数值覆盖，却长期无法通过动作 probe 和动作因果门；
2. 当前实现已改变 Replay retained transition、正式任务、动作 probe 目标、Episode bootstrap 和留出设计，但从未形成新谱系结果；
3. 在动作可辨识性成立前调整任务 Actor、奖励或最终成功优化，无法归因且成本过高；
4. 该瓶颈可用校准阶段的低成本证据先判别，不需要立即运行完整 120 Episode 训练。

## 当前入口

开发总门禁：

```bash
.venv/bin/python scripts/verify_development_ready.py \
  --foundation-device cpu \
  --output artifacts/development-ready.json
```

正式训练：

```bash
.venv/bin/python -m hwr.apps.train_foundation_world_model \
  --run-id <run-id> \
  --device mps \
  --foundation-device mps \
  --development-ready artifacts/development-ready.json
```

正式评测：

```bash
.venv/bin/python -m hwr.apps.evaluate_foundation_world_model --help
.venv/bin/python -m hwr.apps.aggregate_foundation_evaluations --help
```

训练只能基于干净、已提交、通过当前提交门禁的源码启动。

## 资源边界

- 主机：Apple M5 Pro，18 核，48 GB 统一内存；
- 加速器：PyTorch MPS 可用；
- MuJoCo：3.10.0；
- 当前可用磁盘：约 223 GiB；
- 当前配置估算单 run：约 28.41 GiB；
- 配置硬门：估算不超过 30 GiB，启动时至少保留 35 GiB 空闲空间；
- 正式训练默认独占本机加速器，不并行启动多个训练 run。

## 本轮决策约束

- 创新 Agent 只提出少量、单变量、可证伪假设；
- 筛选 Agent 独立评分，不互看结论；
- 在 `03-experiment.md` 冻结前不修改训练行为、不启动正式训练；
- 若没有提案能在当前基线缺失的前提下保持因果可归因性，先运行无行为改动的校准基线；
- 所有行为变化必须有测试，并满足 Python 文件与函数尺寸门；
- 长时门禁、训练和评测使用 `traex-host-exec`，超时后检查同一 run，不重复启动。
