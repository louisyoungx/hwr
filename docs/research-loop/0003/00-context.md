# R0003 研究上下文

## 轮次身份

- 轮次：`R0003`
- 起始提交：`4050dcf`
- 状态：进行中
- 当前证据提交：`ed0f90feadf2a82c589e0355fe3dee6368c526a6`
- 前一轮结论：observation lag 单变量假设被否定
- 本轮主要证据：P09 完整 128-transition 轨迹与 R0001 7×16 Replay 的 Probe 结论不一致

`R0001-P14`～`R0001-P16` 及延续候选继续沿用原 ID。

## 当前瓶颈

> 现有 Action Probe 的 7×16 证据设计和线性 ridge 判定是否有足够功效支持训练准入？

关键反例：

- P09 完整连续 128-transition 数据全部任务/horizon 聚合过线；
- 同一数据固定切成 7×16 后，horizon=16 ratio 在 rho=0.96 降到 0.294～0.400，
  在 rho=0.50 降到 0.160～0.224；
- R0001 正式 Replay 的每个 source Episode 同样只有 7 个 16-transition shard。

## 轮次约束

- 先测功效，再决定是否进入连续性物理确认。
- 不降低 1.05/1.01 门槛。
- 不把重叠 transition 当独立 bootstrap 单位。
- P09 数据只用于机制发现；新物理确认必须用新 seed。
- 不在本轮修改世界模型训练或正式 Replay 分布。
