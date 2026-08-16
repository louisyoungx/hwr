# R0002 冻结实验

## `R0001-P09` 冻结合同

- 分支：`exp/R0001-P09-observation-lag`
- 实现提交：`3d199c5bfe9ff17634ac423113e85beedc5b2be5`
- run ID：`r0001-p09-observation-lag-s20260901`
- 运行目录保持历史路径：
  `runs/research-loop/0001/r0001-p09-observation-lag-s20260901`

路径中的 `0001` 是迁移前产生的历史 artifact 路径，不表示文档轮次；不得重命名已有哈希证据。

### 数据

- 任务：
  - `clear_dining_table_3d/v1`
  - `store_kitchen_items_3d/v1`
  - `tidy_living_room_3d/v1`
- training seeds：
  `20260901, 20365630, 20470359, 20575088, 20679817, 20784546, 20889275, 20994004`
- holdout seeds：
  `520260901, 520365630, 520470359, 520575088, 520679817, 520784546, 520889275, 520994004`
- 每任务、每 rho：8 training + 8 holdout。
- rho：0.96 主诊断；0.50 确认。
- observation lag：0/1 交替。
- 每条 Episode：1 个前置动作、128 个评分 transition。

### 判定

同时要求：

1. 全任务全 horizon ratio `>=1.05`；
2. 同步 Episode bootstrap `p05>=1.01`；
3. lag=0 新旧一致；
4. lag=1 不丢 Episode；
5. lag=1 每个 horizon state-action MSE 至少下降 10%；
6. provenance、hash 和样本数完整。

### 禁止项

- 不修改模型、训练、Replay sampler、动作幅值、安全层或门槛。
- 不使用任务语义、奖励、对象 token 或评测数据。

## `R0001-P12` 冻结合同

- 实现提交：`34a7bb0`
- evaluation-only 物化 9 条未见 instruction。
- artifact manifest 直接哈希语言 manifest 和 embedding 文件。
- 训练 run 全部只读。

## `R0001-P13` 冻结合同

- 实现提交：`aeaedd7`
- acceptance schema：`hwr.bimanual-acceptance/v3`
- 每任务 normal mode：
  - p95 `<=0.01`
  - bootstrap upper `<=0.02`
  - max `<=0.05`
- 与原全部能力、安全和消融门联合。

## 资源

- P01：CPU/MuJoCo，无训练、无教师特征。
- P04/P05：测试与离线评测修复。
- 不启动正式 MPS 训练。
