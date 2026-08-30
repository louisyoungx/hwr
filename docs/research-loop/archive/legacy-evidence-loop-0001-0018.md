# 旧 evidence-first 研究循环归档：R0001–R0018

## 归档决定

- 归档日期：2026-08-31。
- 范围：`docs/research-loop/0001/`～`0018/` 及其对应历史 run、提交和实验分支。
- 方式：原位只读归档，不移动目录，不改写 R0001–R0017 内容。
- 原因：多个历史 evaluator、测试和 artifact 把原路径、提交祖先和 Git tree hash 作为
  provenance；物理移动会破坏复现，而不会增加研究价值。
- R0018 是机制重置时尚未完成的轮次，只更新其结果与总结以记录主动终止。

## 总体结论

旧循环建立了大量评测、测量、provenance 和局部正确性证据，但没有推进能力基线：

- 最新完整三维世界模型基线仍为 24 Episode、1,600 update、0 success；
- Actor 未解锁；
- 三个正式任务没有通过新的闭环能力结果；
- R0004–R0017 没有正式能力训练，R0018 也没有正式 physics/capability run；
- 历史 `accepted as ...` 主要代表 evaluator、measurement、contract 或局部修复成立，
  不能重解释为 `accepted_capability`。

因此旧循环的最终能力判定为：`L0 未通过`。

## 阶段索引

| 轮次 | 主要内容 | 归档解释 |
|---|---|---|
| R0001–R0003 | 世界模型负基线、动作因果和模型诊断 | 保留负基线与 plant action 因果证据；没有能力成功 |
| R0004–R0008 | 延迟、安全、评测泄露、Replay 信息和接触账本 | 多数是测量/评测合同，不进入能力基线 |
| R0009–R0013 | 坐标、FK、candidate acquisition、可达性、interaction contract、安全 witness | 提供控制链和任务表达缺陷证据；没有训练成功 |
| R0014–R0016 | v2 candidate 修复、artifact/selection lineage | 局部正确性成立，但默认 runtime 未迁移、能力不变 |
| R0017–R0018 | contract oracle 与 coordinate oracle 资格链 | 已进入递归前置证明；主动终止并切换研究机制 |

## 值得保留的开发证据

以下证据可用于新路线的设计，但必须遵守原结论边界：

- `R0001-P17`：正式任务数据中的 plant action 具有物理因果效应。
- `R0001-P40-E1/E2`：接触和实体接触图测量可以作为开发诊断。
- `R0001-P51/P52`：Cartesian frame 修复和 policy FK/MuJoCo site agreement。
- `R0001-P57`：固定 cohort 的双臂 pre-contact command support 明显不足；当前 B2/B3/B4
  时域不能作为可行控制上界。
- `R0001-P61/P72`：generic candidate-centered primitive 缺少表达完整长时任务的信息。
- `R0001-P66-E1`：一次 legacy 路径上的预测安全拒绝 witness，可用于设计安全保持的
  teacher/controller，但不能外推成整体安全能力。
- `R0001-P79/P83`：isolated v2 candidate 与 selection lineage 的局部证据，只供开发和
  历史重建，不自动授权 selector、训练或能力结论。

## 降级为 development-only 的材料

- R0001–R0018 反复使用的 24-Episode bank及其派生 candidate、selection 和 prefix outcome；
- outcome 已暴露的 sentinel、公开 salt、固定 threshold 和历史 cohort；
- 为旧 evaluator 构造的 source hash、AST、加密 fixture、contract oracle 和 lineage 工具；
- superseded 与 failed artifact。

这些材料仍可做回归、debug 和反例库，但不能再称为未见确认性证据。

## R0018 与 P88 保存状态

R0018 在正式运行前因研究机制重置而主动终止：

- 主分支冻结提交：`39f95f606659bf31eaccbc7b235060503a4ad5ad`；
- 实验分支：`exp/R0001-P88-E1-coordinate-oracle`；
- 已提交实现：`f9f417ea346755fcc60d0bc8332b42452b175b49`；
- worktree：`/Users/louis/Developer/AIWorkspace/50-housework-robot-r0018-p88`；
- 归档检查时 worktree 仍有 4 个已修改文件，约 `+47/-21`，未提交；
- formal evaluator 未运行，formal artifact 不存在，P88 没有科学结论。

上述分支和 dirty worktree 暂不删除、不合并、不清理，等待用户单独决定。它们不构成新基线，
也不应阻塞 R0019 的 L0 capability reset。

## 新循环继承边界

新循环只继承：

1. 当前真实能力基线为 0 success；
2. 正常物理、成功状态机和独立安全层不得在最终能力评测中削弱；
3. P57/P61 指向的控制可行性和任务表达缺口值得优先验证；
4. 历史失败和反例不得重复消耗资源。

新循环不继承旧候选批准、六文档模板、固定多 Agent 编制、自动下一轮、无专家/无课程信条，
也不继承“在动作实验前继续完成 P88/P76/P68 前置链”的要求。
