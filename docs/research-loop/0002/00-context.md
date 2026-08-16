# R0002 研究上下文

## 轮次身份

- 轮次：`R0002`
- 起始提交：`b665c9d96049d80e1951c6a8e941af4695d23d2a`
- 前一轮可信基线：`R0001-P01` v4 有效负基线
- 前一轮测量合同：`hwr.foundation-data-action-probe/v4`
- 本轮结束提交：`4050dcf`

`R0001-P09`～`R0001-P13` 已进入实现提交、运行和结果，继续沿用原 ID。

## 起始证据

R0001 v4 在 24 Episode、1,600 update 形成完整负基线：

- 厨房 1-step data probe ratio `1.002280`；
- 客厅 16-step ratio `1.023931`；
- 世界模型 aggregate action-shuffle ratio `1.001279`；
- 三任务动作执行 recall 均为 0；
- 动作有效秩约 15.63，简单边际动作覆盖不足不是充分解释。

R0001-P04 接受后，跨 horizon bootstrap 使用同步 Episode multiplicity；统计更保守，但不改变
点估计和失败指纹。

## 主要瓶颈

> P01 的数据动作可辨识失败是否主要由 observation latency 下的动作—可见状态错位造成？

该问题优先于改变随机激励、Replay sampler 或世界模型。若时序测量合同本身无效，任何能力
候选都缺乏可归因基线。

## 正交评测阻断

本轮同时发现两个不改变训练能力、但会阻断未来正式闭环的问题：

1. 9 条未见 evaluation instruction 没有预物化 embedding；
2. 闭环接受合同没有限制依赖硬安全层持续兜底的干预负担。

两项必须作为独立平台/评测修复实施，不能计入能力收益。

## 资源与边界

- P01 v4 冻结 Replay、checkpoint、留出和日志保持只读。
- `R0001-P09` 只做 CPU/MuJoCo 短物理诊断，不训练模型。
- P02/P03 不与 P01 首次捆绑。
- P04/P05 只修改评测可执行性或接受合同。
- 正式训练仍需干净提交、项目门禁和可信动作因果证据。
