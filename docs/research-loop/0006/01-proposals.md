# R0006 提案

## 生成过程

- 三个创新 Agent 分别从评测统计、普通 Replay 可识别性、闭环学习与安全方向独立只读
  检查 `AGENTS.md`、R0001～R0005、当前源码、测试和 run 产物。
- 三者未修改正式实现、未启动正式训练，完成前未查看其他创新 Agent 的结果。
- 主 Agent 只做稳定编号、重复项合并、证据复核和文字归并，未按建议排序提前筛选。
- R0005 已存在且未完成的观点继续沿用原 ID；新观点从 `R0001-P39` 递增。

## 提案总表

| ID | 名称 | 类型 | 来源 | 初始状态 |
|---|---|---|---|---|
| `R0001-P36-E2` | 平衡、配对、双账本能力基准合同 | 评测合同 | A | 待筛选 |
| `R0001-P39` | 环境 seed 与策略 RNG seed 隔离 | 评测泄露修复 | A | 待筛选 |
| `R0001-P32-E1` | 双重正交普通 Replay 条件信息门 | 离线数据诊断 | B | 待筛选 |
| `R0001-P40` | 白名单接触力—冲量安全总账 | 安全评测诊断 | C | 待筛选 |
| `R0001-P41` | 无任务语义的 RGB-D 安全接触探索 | 无训练行为候选 | C | 待筛选 |
| `R0001-P42` | 同场景反事实目标—语言绑定门 | 泛化评测合同 | C | 待筛选 |

## 共同证据与禁止边界

- 当前完整 3D 负基线是 24 Episode、1,600 update、0 成功；Actor 从未解锁。
- P17 只证明 plant action 可控，不证明普通 Replay 条件可识别，也不证明 production RSSM
  使用动作。
- P36-E1 只接受为评测合同证据，不含成功标签，不是 capability benchmark。
- P11 的 27 cell 为每任务每 action latency 下 `2/10/4`，不是平衡设计。
- 当前 evaluation profile 从 latency `[1,3]` 连续区间采样后取整，不保证 3×3 cell 平衡。
- 当前 evaluator 把同一个 seed 传给 environment reset 和 policy reset；环境 seed 同时决定
  物理随机化、语言变体、latency 和传感器噪声，构成潜在评测泄露通道。
- 普通 Replay 为 168 个 shard、2,688 transition，但只来自 24 个 source Episode；
  source 分布为餐桌 6、厨房 6、客厅 12，每个 source 恰有 7×16 transition。
- Replay 保存 pre/successor visible proprio、actor proposal、executed action 和 safety
  intervention；不保存无延迟 physical snapshot，也不保存逐 Episode plant latency/scale。
- retained-window selector 使用 successor motion、action innovation、安全和交互结果，因此
  普通 Replay 诊断最多外推到当前 salience-retained distribution。
- 任何提案不得延长 100ms validity、删除 latency 3、读取 future/latest observation、
  放宽碰撞或安全门槛、将 scripted action 当能力、读取正式成功标签训练，或按结果选择
  seed/cell/权重/门槛。

## `R0001-P36-E2`：平衡、配对、双账本能力基准合同

### 来源、证据与假设

- 来源：A；延续 R0005 已完成的 P36-E1，不新建竞争性能力观点。
- P36-E1 证明 supported/challenge 可以唯一分账，但明确
  `balanced_factorial_benchmark=false`、`closed_loop_success_available=false`。
- 当前 `BimanualEvaluationReport` 只按 `task × ablation` 聚合，没有 latency cell estimand；
  foundation evaluator 的默认 seed 又由各训练 run 的 training seed 派生。
- 当前 observation-latency diagnostic override 只允许 0/1，action-latency override 禁止
  与 observation override 同时启用，不能直接构造联合 3×3 cell。
- 假设：在任何能力结果出现前，可以冻结一个 cell 平衡、baseline/candidate 完全配对、
  缺失 fail-closed 且同时发布完整挑战与支持条件账本的 benchmark 合同，从而消除 seed、
  latency 频率、缺失和聚合权重漂移。

### 唯一主变量与影响范围

- 只改变评测计划、联合 latency override、持久化总账、报告 schema 和统计聚合。
- 不修改 policy、checkpoint、任务成功条件、环境除两项 latency 外的随机化、runtime、
  safety、100ms validity 或训练。
- 本 E2 不修复 P39 seed 泄露；正式 seed bank 只有在 P39 独立通过后才能生成。
- 当前没有 qualified deployment，E2 最多形成 benchmark contract/runner integrity
  evidence；不得为获得物理成功率而绕过 foundation deployment 准入。

### 结果前冻结的设计

- cell：
  `3 task × observation latency {1,2,3} × action latency {1,2,3}`。
- 每个 cell 使用相同、结果前冻结的 `n` 个环境 replicate；baseline/candidate deployment
  对每个 planned Episode 一一配对。
- 完整首要账本：
  - 27-cell 等权 `complete_challenge` macro；
  - latency 3 永远占完整 observation-latency 权重的三分之一。
- 条件账本：
  - observation latency 1/2 的 18-cell 等权 `supported_conditional` macro；
  - 不能替代完整首要账本。
- task、observation latency、action latency、训练 seed 和 deployment hash 全部分层。
- 原始 Episode、失败、异常、安全拒绝和缺失全部发布；成功条件化视频只能作附录。

### 最小验证

1. manifest/property test：
   - 27 个 cell 各恰有 `n` 个唯一 planned Episode；
   - baseline/candidate identity、环境 seed、cell 和统计权重一一对应；
   - 无重复、越界或 replacement seed。
2. reset-only MuJoCo smoke：
   - 联合 override 后只允许 observation/action latency 两字段改变；
   - 其余 sampled/effective randomization canonical hash 完全一致；
   - latency 1/2/3 的 source age 与 P29 合同一致。
3. 合成 null/planted power：
   - cell 固定；
   - cell 内按 paired Episode 同步 bootstrap；
   - 多训练 seed 时再以训练 seed 为外层统计单位；
   - 预注册最小相关效应为 supported macro 10 个百分点；
   - null FPR `<=0.05`，10pp planted power `>=0.80`。
4. 故障注入：
   - 第 k Episode 退出、损坏、重复、错误 cell/seed；
   - planned manifest 在结果前原子写入；
   - terminal record append-only 且 hash-bound；
   - `planned = valid_terminal + unresolved` 恒成立；
   - coverage 非 100% 时只能 `inconclusive`，不能 complete-case 聚合；
   - candidate timeout/safety/collision 属于有效失败，不得伪装 infrastructure missing。

### 主要指标、守护与失效

- 未来正式能力比较的共同主要效应：
  - paired `Δcomplete_macro` bootstrap 95% 下界 `>0`；
  - paired `Δsupported_macro >=10pp` 且 95% 下界 `>0`。
- 每任务 supported success 不得发生预注册不可接受回归。
- severe collision 为 0；过期动作实际应用率为 0；safety intervention burden 不劣化；
  双臂并发、稳定保持和左右臂消融门保持。
- latency 3 若仍结构性不可执行，必须报告 `full_profile_supported=false`，不能复用旧
  70% 成功门宣称全域能力。
- 以下任一发生即拒绝或判为无效：
  - 联合 override 改变非 latency 随机化；
  - cell 缺失、结果后改权重、替换 seed；
  - 功效不足却把阴性解释为无效；
  - 只展示 supported ledger；
  - incomplete run 仍给 acceptance；
  - 为运行 benchmark 同时删除 deployment/action-causality 准入。

### 成本、风险、依赖与去重

- 合同、合成统计、故障注入和 reset smoke 成本低至中，CPU，无训练。
- 正式 normal-policy 预算为每 deployment `27n` Episode；三 baseline seed 加三
  candidate seed 为 `162n` Episode，尚未计消融，必须先通过功效—资源门。
- 风险：正式成本高；macro 可能掩盖单任务失败；联合 override 是新评测实现。
- 依赖：P36-E1；P39；未来 hash-bound qualified deployment。
- 不重复 P36-E1：E1 只重放历史不平衡诊断证据；E2 建立未来能力基准。

## `R0001-P39`：环境 seed 与策略 RNG seed 隔离

### 来源、证据与假设

- 来源：A。
- 当前 `_evaluate_episode()` 将同一个 raw seed 同时传给 environment 和 policy。
- 环境 seed 决定物理随机化、latency、instruction variant 和逐帧传感器噪声。
- 即使当前 production policy 未显式利用 seed，正式 benchmark 也不应暴露能唯一预测环境
  隐变量的标识。
- 假设：commit-before-reveal 的环境 seed bank 加独立 domain-separated policy RNG seed
  能关闭标准 policy interface 的直接 seed lookup 通道，同时保持候选—基线完全配对。

### 唯一主变量与最小验证

- 只修改 evaluator seed contract、policy reset seed 语义和 manifest 字段。
- 不改变环境 seed、本体/视觉 observation、动作、安全、任务或随机化结果。
- 在 deployment hash 冻结前只提交 seed-bank commitment 与固定推导算法；deployment
  冻结后 reveal salt 和实际环境 seed。
- `policy_rng_seed = H(domain, environment_seed, deployment-independent salt)`，但 policy
  interface 不得收到 raw environment seed；baseline/candidate 对应 Episode 使用完全相同
  policy RNG seed。
- canary policy 只能看到 opaque policy RNG seed；对 environment seed、latency cell、
  instruction variant 和随机化参数的预测不得高于预注册 chance/null。

### 指标、守护与失效

- raw environment seed 进入 policy interface 次数为 0。
- commitment 验证率与 baseline/candidate seed-pair coverage 均为 100%。
- 相同 manifest 重放得到相同环境与 policy RNG 序列。
- 旧新 evaluator 的 environment randomization hash、instruction、latency 和传感器噪声
  逐项一致；确定性 canary action 序列不得改变。
- salt/seed bank 在结果后更换、候选使用不同 policy RNG、policy 仍能直接读取 raw seed、
  或 commitment 无法 reveal/replay，均拒绝。
- 成本低，无训练；风险是 salt 丢失和只修复标准接口、不能隔离恶意代码读取文件系统。

### 依赖、泄露边界与去重

- 必须在 P36-E2 正式 seed bank 生成前完成。
- 不宣称提升能力，只接受为 evaluation leakage fix。
- 不重复旧 P07：P07 处理采集时环境 seed 与随机动作 seed；P39 处理正式 evaluator 向被评
  policy 暴露环境 seed。

## `R0001-P32-E1`：双重正交普通 Replay 条件信息门

### 来源、证据与假设

- 来源：B；修订 R0005 的 P32。
- P16 已证明旧 state/state-action ridge 的设计功效不足，不能据其失败宣称 Replay 无动作
  信息。
- P17 证明 plant 可控，但普通 Replay 没有同状态随机动作分配。
- P32 原稿只残差化 action，并比较两个独立 forward head；这可能混入 state-head 容量、
  优化和 target nuisance。
- 假设：在 source-Episode 完全 out-of-fold、action 和 successor target 双重残差化后，
  `u_t = a_t - E[a_t|S_t]` 仍能预测
  `v_t = y_t - E[y_t|S_t]`；这只表示 retained Replay 含有给定 pre-action visible state
  后的增量条件预测信息。

### 结论边界

- 不称为 plant causality：普通 Replay 无随机动作分配，residual 可能编码 controller
  history、FIFO、actuator scale、安全改写或遗漏随机化。
- 不称为 production utilization：fresh residual map 通过不证明 RSSM 使用该信息。
- target 是 recorded successor visible-proprio information，不等于 P17 的无延迟 current
  physical state。
- 只外推到当前 salience-retained Replay，不外推到完整自主采集分布。

### 冻结数据与唯一主变量

- 输入固定为 P01 v4：
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812/replay/autonomous`。
- manifest SHA-256：
  `c7f7a50925b581307dc95787078c1fc2ee520f8b210e61fd91e1007db21a1985`。
- 24 个 source Episode 是最外层独立单位；168 个 shard 不能跨 fold。
- 主判定 `S_t` 固定为原始 37-D `proprioception[t]`；不使用已在全部 Replay 上训练的
  production posterior 作为主输入。
- `a_t` 固定为 `executed_action[t]`。
- 主 target 固定为一步 16-D rate-like controllable visible-proprio innovation。
- control 为 outer-test `m_y(S)`；candidate 唯一新增
  `B(a-m_a(S))`，共享同一个 state baseline。

### Nested cross-fit

- 按 task 分层 outer 3-fold，每折固定留出餐桌 2、厨房 2、客厅 4 个 source。
- 每个 outer-train 内再按 source 做 inner 2-fold。
- inner OOF 分别拟合固定 ridge `m_a(S)` 与 `m_y(S)`，只用 inner-train 的均值、scale 和
  ridge 解。
- residual map `B` 只拟合 inner OOF 的 action/target residual。
- outer-test nuisance 由全部 outer-train 重拟合；outer-test 不参与标准化、超参、rank、
  target scale 或门槛。
- fold 由冻结 source identity、task 分层和固定 hash 规则唯一生成；不得扫描分折。
- ridge、bootstrap seed/次数、planted scale 和缺失 mask 均在实现前冻结，不调参。

### 预注册机制守护

下列守护与主判定同一次正式报告全部运行，不按主结果决定是否隐藏：

1. controller-context guard：
   - nuisance 额外输入只允许 current actor proposal、过去 4 步 actor proposal/executed
     action 和 availability mask；
   - 禁止真实 latency/scale、source position、seed、task ID、reward、successor 和 safety
     label；
   - shard 前缺失 history 必须 mask，不得删除。
2. configuration-target guard：
   - target 改为一步 17-D configuration delta：12 joint position、2 gripper position、
     3 base pose；yaw 使用冻结 wrap；
   - joint/base/gripper 分族报告，禁止 pooled 维度掩盖。
3. strata：
   - safety rewrite / no rewrite；
   - shard 首四步 / interior；
   - 三任务；
   - 全部 planned source。

### 指标、功效与判定

- 主指标：
  - outer-test source-Episode 等权 `MSE_control/MSE_candidate >=1.05`；
  - Episode-block mean log-ratio bootstrap 95% 下界 `>0`；
  - 三任务 point estimate 分别为正；
  - candidate 绝对 target MSE 改善。
- 数据功效：
  - outer-OOF action residual 有效秩 `>=6`；
  - exact-pipeline empirical-scale planted power `>=0.80`；
  - zero residual 和 random target FPR 均 `<=0.05`。
- `accepted as retained-Replay conditional information evidence`：
  - 主指标和功效全部通过；
  - 信号在 no-rewrite 与 shard interior 不消失；
  - controller-context 后仍至少 `1.02` 且下界 `>0`；
  - configuration target 至少三任务同向且整体下界 `>0`。
- `rejected`：
  - 功效合格但主指标失败；
  - 收益只来自 safety rewrite/边界；
  - controller context 消除信号；
  - 只存在即时 rate target而 configuration 完全不支持；
  - 任一 outer source 泄露或绝对误差恶化。
- `inconclusive`：rank/power 不足、observation lag 使 configuration target 无测量功效、
  artifact 缺失或 lineage 无法重建。

### 成本、风险、依赖与去重

- CPU 离线 evaluator，成本低，不更新 production 参数，不启动物理 Episode。
- 风险：24 个 source 统计功效有限；visible proprio 有 observation lag；retention selector
  已看 outcome；shard 前 controller history 不完整。
- 不依赖 P31/P33；通过后最多提高修订 P31 的优先级。
- 不重复 P16：共享 state baseline、双重 OOF residual、nested source folds。
- 不重复 P17：无随机物理干预，不作因果声明。
- 不复活 P28/P30：不使用 successor posterior oracle，不扫描 offset，不构造错误动作。

## `R0001-P40`：白名单接触力—冲量安全总账

### 来源、证据与假设

- 来源：C。
- manipulated object、篮筐、托架和抽屉等位于 `allowed_robot_contact_geoms`。
- 正式 safety 扫描对白名单 geometry 直接跳过，之后才计算 forbidden contact force 和
  severe collision。
- `GraspContactMonitor` 已计算左右夹爪法向力，但后端主要消费其 bilateral 布尔信号。
- 当前闭环验收主要看 severe collision count 与 safety intervention burden，可能遗漏
  “允许接触但力/冲量过大”的硬件迁移安全风险。
- 假设：用结果前冻结的接触类别、峰值法向力与时间积分冲量总账，可以发现当前
  `severe_collision_count=0` 轨迹中的安全假阴性，并阻止虚假安全结论。

### 唯一主变量与最小验证

- 第一阶段只增加 evaluator/report-only 测量，不改变 policy、reward、动作、接触白名单、
  task success、100ms validity、predictive stop 或终止。
- 对三个正式场景执行固定、可重建的短接触测量轨迹；它只验证测量合同，不作能力证据。
- 同时报告 forbidden/allowed contact 的：
  - geometry pair 与类别；
  - 左右夹爪来源；
  - peak normal force；
  - 每控制周期 impulse；
  - Episode p95 与 maximum。
- 首轮只检验当前内部 severe threshold 一致性；真实硬件/脆弱物体阈值必须另立实验。

### 指标、守护与失效

- 主要：`allowed_force >= current severe threshold` 且
  `severe_collision_count == 0` 的 Episode/step 数。
- 原 forbidden contact、action、termination、success、P13 intervention burden 必须逐项
  不变；缺失 force 必须标为 missing，不得按 0。
- 固定预算内无假阴性，或指标无法稳定重建、对 timestep/solver 极端敏感，则拒绝升级为
  硬门，仅允许保留描述性报告。
- 成本低，短 CPU MuJoCo 验证，无训练。
- 不重复 P13：P13 统计 intervention，不覆盖被白名单跳过的力/冲量。
- 不向 policy 提供 force、geometry ID 或安全标签，不制造任务捷径。

## `R0001-P41`：无任务语义的 RGB-D 安全接触探索

### 来源、证据与假设

- 来源：C。
- 当前预解锁随机源明确 `observation_conditioned=false`、
  `task_conditioned=false`，且动作过程高度相关。
- 24 Episode、约 13k transition 和全维动作覆盖下，三任务仍无 bilateral contact 或
  controlled motion；runner 在 Actor 未准入前持续回退到该随机源。
- 假设：零交互瓶颈主要来自 observation-blind 高维随机游走难以命中可操作表面；只把
  预解锁 action source 改为部署可见 RGB-D/本体条件化的低速、对称、可逆接触探测，可
  提高跨任务安全接触和受控运动密度。

### 唯一主变量与最小验证

- 唯一行为变量是 Actor 解锁前的 action source。
- 候选只能使用 RGB-D、相机标定、本体状态和已执行动作。
- 禁止语言、task/object/target ID、reward、success、仿真真值或专家轨迹。
- 三任务使用相同未见环境 seed、相同物理 step，对比现有随机源与候选，不训练。
- 候选只能做通用近表面低速双臂探测；不得包含对象或任务专属动作序列。

### 指标、守护与失效

- 主要：每万物理 step 的：
  - bilateral-contact Episode；
  - `>=0.01m` controlled object/articulation motion Episode；
  - 三任务最弱分区。
- severe collision 为 0；P13 burden 不劣；动作有效秩、活跃维度、执行 RMS 不塌缩；
  P40 力—冲量守护不劣。
- 收益只在单场景、主要接触固定家具/地面、只增加接触不增加受控运动，或通过几乎不动来
  降低风险，均拒绝。
- 成本低至中；只允许先做配对物理 smoke。通过后若要训练，必须另行冻结等 Episode/update
  正式候选。
- 依赖 P40 的测量合同；不重复 P02/P08 的 observation-blind 随机过程调参，也不使用 P37
  privileged shaping。

## `R0001-P42`：同场景反事实目标—语言绑定门

### 来源、证据与假设

- 来源：C。
- 当前每个 task ID 唯一绑定 scene、物体集合与 target mapping；evaluation language 只是
  同一目标的未见改写。
- policy 使用 instruction embedding，但当前 benchmark 无法排除“按场景外观识别固定
  任务、忽略语言”的捷径。
- 假设：在相同场景、初态、物体、目标容器和动力学下，冻结两个互换的 object-target
  mapping，只通过自然语言指定目标，可以区分真正 language grounding 与 scene-ID 捷径。

### 唯一主变量与最小验证

- 只增加 evaluator-only 同场景目标变体，不修改训练、policy、原 canonical benchmark、
  reward shaping 或 safety。
- 先在餐桌与厨房构造 canonical/swapped assignment；scene、seed 和非目标随机化完全
  一致。
- policy 唯一可见差异只能是 instruction embedding；不得提供 variant task ID、mapping
  token、文件路径、seed 或视觉颜色改动。
- 先做合同测试，证明两个 mapping 均物理可完成；没有 qualified deployment 时不运行能力
  比较。

### 指标、守护与失效

- 主要：
  - 每任务 `min(canonical success, swapped success)`；
  - paired-seed 双条件成功率；
  - matched instruction 相对 mismatched instruction 的成功差。
- canonical 成功率不回归；完整挑战与 supported ledger 同屏；P13、零严重碰撞、双臂
  消融、稳定保持和预算门不变。
- 只会 canonical、换语言动作近似不变、mismatched instruction 仍高成功、或 variant 身份
  从非语言通道泄露，均拒绝。
- 合同验证成本低，未来闭环比较中等；依赖 qualified deployment 与 P36-E2。
- 不重复 P12：P12 只验证未见改写物化；P42 检验语言是否因果决定目标。

## 明确不复活与依赖路由

- P34 不得以 command lease、sequence 或 provenance 换名复活。
- P35 在没有独立实时安全感知或 reachable-set 证据前继续延后。
- P37 已拒绝，不再提出更弱的单臂可利用 shaping。
- P38 本轮不改变 unlock；最多保留 report-only shadow。
- P32-E1 通过最多授权修订 P31；不直接解锁 P33。
- P31 只有在普通 Replay 可比训练分布上出现“fresh conditioned probe 通过、blind
  control 不通过、production prior 不通过”的预注册缺口，才可重新筛选 P33。
- P36-E2、P39、P40、P42 都是评测/安全合同，不得与 P41 或未来世界模型训练放进同一首次
  因果比较。
