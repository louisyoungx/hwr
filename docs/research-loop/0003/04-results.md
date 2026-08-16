# R0003 实验结果

## `R0001-P16`：`rejected`

### 完整性

- 源码提交：`03de3fcfaa151efc6dd92ae2081ca3ad08732798`
- report SHA-256：
  `b81615e80fe10bfb7314d6519d182e3da88449c8948c3adaca192e870e662821`
- 每种条件 500 trial；
- 每 trial 200 次同步 Episode bootstrap；
- 墙钟 65.02 秒；
- 500 个 trial 全部保留；
- source commit、P09 输入 hash、report 和 manifest 一致；
- 无重复启动、无残留进程。

### 三臂结果

| 设计 | null FPR | permutation FPR | planted power |
|---|---:|---:|---:|
| fragmented 7×16 | 0.0% | 0.0% | 0.0% |
| continuous same 7 starts | 0.0% | 0.0% | 0.0% |
| continuous all starts | 0.0% | 0.0% | 5.8% |

没有设计达到 80% 功效，因此 `qualified_arms=[]`，P01 路由 blocked。

### 失败指纹

- rho=0.50 连续全起点几乎满功效：三任务 16-step 为 99.8%、99.8%、100%。
- rho=0.96 长时功效只有约 39.8%～43.0%。
- 连续全起点 h16 有 776 行、54 列、数值秩 53，condition number 约 `4.7e6～5.35e6`。
- 7-start 设计只有 56 行、54 列，整体功效为 0。

### 结论

现有 Action Probe 设计对零效应保守，但对冻结动作效应检出功效不足。当前 data probe
失败只能标记为测量 `inconclusive`；不得降低 1.05/1.01 门槛，也不得启动 P01/P02。

下一轮应使用不依赖 state nuisance 拟合的训练前物理因果证据。该未解决问题进入 `R0004`，
不属于本轮文档。
