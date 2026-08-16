# R0003 冻结实验

## `R0001-P16` 冻结合同

- 实现提交：`03de3fcfaa151efc6dd92ae2081ca3ad08732798`
- run ID：`r0001-p16-probe-power-s20260916`
- 历史产物路径：
  `runs/research-loop/0001/r0001-p16-probe-power-s20260916`

路径中的 `0001` 是迁移前的历史 artifact 路径，不表示文档轮次。

### 固定设计

- P09 96 Episode 作为设计矩阵；
- 每 Episode 前 112 transition；
- 三臂：
  - `fragmented_7x16`
  - `continuous_same_7_starts`
  - `continuous_all_starts`
- rho：0.50、0.96；
- 三任务；
- horizon：1/4/8/16；
- ridge：`1e-3`；
- bootstrap：每 trial 200 samples，同步 Episode 合同；
- 每种条件 500 trial。

### 合成条件

- null：state signal + AR(1) noise；
- planted：state signal + 0.5×action signal + noise；
- permutation：使用置换 action signal；
- state signal RMS 1.0；
- noise RMS 0.5，AR rho 0.8；
- 系数、噪声、bootstrap seed 按冻结公式生成；
- 不读取真实 target、reward 或安全标签。

### 接受标准

每个可用设计臂：

- null FPR `<=0.05`；
- permutation FPR `<=0.05`；
- planted power `>=0.80`。

连续全起点自身必须合格，才能路由到 `R0001-P14`。

## 资源与命令

```bash
.venv/bin/python -m hwr.apps.evaluate_action_probe_power
```

- CPU；
- 无模型训练；
- Host 一次性 runner，回调前移除自身服务；
- 保留全部 trial 和 manifest hash。
