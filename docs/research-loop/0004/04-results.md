# R0004 结果

## `R0001-P29`：`accepted as contract diagnostic`

### 完整性

- 冻结文档提交：`ce63f95235cae7df4265db3101edc6e9c821c3db`
- 实现提交：`b34716d32cccfe5e619b18aaecaaeda8954a765a`
- run：
  `runs/research-loop/0004/r0004-p29-stale-validity-s20262901`
- 命令：
  `.venv/bin/python -m hwr.apps.evaluate_stale_observation_validity`
- report SHA-256：
  `7729eed53e034bef5a9d5c50bd1d87d6025370b290226f3055c480f3014274fb`
- manifest SHA-256：
  `41495b1a1efa9f24b65e8c5359bc98cfa28ecee4dbdbc20a5c20d37aa7900cef`
- 输入 P11 report/manifest hash 与冻结值一致；
- P11 145 个 artifact 的 SHA-256 和 bytes 全部通过；
- 14 项聚焦及相邻 safety/collector 测试通过；
- Python size、architecture 和 py_compile 通过；
- 没有训练、后台进程、tmux、休眠或重复 run。

### 合成时间线

固定语义：

- control frequency：20Hz；
- control period：50ms；
- action validity：visible observation timestamp 起 100ms；
- safety exact boundary：`now == valid_until` 合法，`+1ns` 拒绝。

64-step observation queue 的拒绝数：

| observation latency | 最大 observation age | 首次拒绝 step | 拒绝数 |
|---:|---:|---:|---:|
| 0 | 0ms | 无 | 0 |
| 1 | 50ms | 无 | 0 |
| 2 | 100ms | 无 | 0 |
| 3 | 150ms | 3 | 61 |

没有 future timestamp；validity duration 仍为 100ms；没有替换 latest bundle。

### P11 真实证据复核

| observation latency | Episode 数 | 每 Episode safety intervention | 总数 |
|---:|---:|---:|---:|
| 1 | 18 | 0 | 0 |
| 2 | 90 | 0 | 0 |
| 3 | 36 | 61 | 2,196 |

- 144 Episode 的 task×rho×action-latency coverage 完整；
- 所有 Episode 均无提前终止；
- maximum severe collision count 为 0；
- 合成预测与 P11 真实分层逐项一致。

### 判定

所有 13 个冻结检查通过，正式 decision 为 `contract_incompatible`：

> evaluation observation latency=3 与当前 observation-timestamp-based 100ms action
> envelope 结构性不相容。

该结论接受为 `contract diagnostic`，不是能力或平台修复：

- P11 仍为 `rejected`；
- 不延长 validity window；
- 不移除或降低 evaluation latency；
- 不用 latest observation 绕过延迟；
- 不修改 safety 威胁模型；
- 不宣称世界模型、Actor 或家务成功率改善。

### 路由

- 若未来保留 latency=3 为必测域，需要另立单变量的 latency-aware action scheduling
  系统设计，并保持真正 stale 输入仍被安全拒绝；
- 本轮不立即修改 runtime，继续回到主瓶颈，只冻结并独立复审 P28 head-only 草案；
- P29 结果不得与 P28 训练首次捆绑。

### 最终仓库门禁

- 全量 `.venv/bin/python -m pytest -q` 通过；
- 既有 11 项 skip；
- 18 条 warning 均来自 `torch.jit.script` deprecation；
- `scripts/check_python_size.py` 通过；
- `scripts/check_architecture.py` 通过；
- `scripts/verify_training_semantics.py` 通过；
- `scripts/verify_physics_integrity.py` 通过；
- 历史 `docs/research-loop/0001`～`0003` 零差异。

## `R0001-P30`：`rejected without run`

- 筛选后历史去重发现与 R0001-P09 实质重复；
- P09 已用 96 个新 Episode 正式检验显式 observation lag 与 actual plant action 对齐，
  并拒绝“单一对齐跨 horizon 稳定改善”；
- P01 v4 数据源的 collector/writer/window/loader 血缘也支持 runtime applied action 与
  同次 outcome transition 成对存储；
- 未实现 P30、未搜索 offset、未重标 Replay、未重跑 P09。

## `R0001-P28`：`rejected without implementation`

- 主 Agent 按首次筛选意见冻结了三折 source-level、fresh-head、true-action-only 草案；
- 两名新的独立复审 Agent 均给出 `changes_required`；
- 共同阻断为：
  - successor posterior oracle 结果可见且对 prior 不公平；
  - `sample=False` 与拟议训练采样路径不同；
  - reverse/slot-rotate 负例可能携带位置、幅值、时距和阶段指纹，并与正式 audit 同质；
  - split/batch/metric/statistic 尚不能由文档唯一重建；
  - head-only 可解码性不能直接路由到正式 world-model objective 与闭环能力；
  - MPS 预算和数值 parity 未验证。
- 未新增 P28 实现文件；
- 未创建
  `runs/research-loop/0004/r0004-p28-prior-successor-s20262801`；
- 未运行 head-only 或 world-model 训练；
- 未扫描负例、采样、预算、学习率、seed 或门槛。

P28 正式标记
`rejected before implementation after independent re-review`。

## `R0001-P27`：`rejected without implementation`

- successor posterior 已看到结果 observation，inverse dynamics probe 容易读取结果瞬态；
- 新 paired 数据、辅助头和 loss 难以形成单变量对比；
- 到 action-conditioned prior 与闭环控制的路由不足；
- 未采集新数据、未实现、未训练。

## `R0001-P26`：`deferred`

- 两名首轮筛选均要求解决负例位置/幅值/时距指纹与正式 audit 同质问题；
- P28 复审进一步确认该类风险仍未消除；
- 本轮不实现、不训练，不把 P28 负例修改事后迁移到 P26。

## 当前状态

R0004 没有启动任何完整训练，也没有新的能力候选。唯一接受结论为 P29 runtime 合同诊断；
能力基线保持不变。
