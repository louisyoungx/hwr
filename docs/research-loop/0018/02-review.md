# R0018 独立筛选

## 筛选过程

- `01-proposals.md` 已在提交 `168d4b9` 冻结。
- 筛选 Agent D、E 均先读取 `AGENTS.md`、R0018 上下文与冻结提案，再按需只读核查
  R0017 结果、相关代码与 artifact。
- 两名 Agent 完成前互不查看结果，没有修改文件、启动训练或运行正式物理 cohort。
- 两者独立得出同一唯一条件首选：`R0001-P88-E1`。
- 主 Agent不按总分机械选择；最终裁决同时考虑依赖闭合、证据增益、可归因性、成本与
  blocker 是否能在实现前关闭。

## 评分维度

每项按 1–5 分分别评分：

- 目标价值；
- 证据强度；
- 可检验性；
- 因果可归因性；
- 通用性；
- 实施成本；
- 回归风险。

主 Agent 不仅按总分选择，还必须审查依赖、判定可达性、评测泄露、结果暴露、
样本单位、资源预算和停止门。

实施成本、回归风险均为分数越高越好，即成本越低、风险越低。

## 筛选 Agent D

| 提案 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P87-E2` | 3 | 5 | 5 | 5 | 3 | 5 | 5 | 保留，非首选 |
| `R0001-P88-E1` | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 唯一首选，修订后冻结 |
| `R0001-P88-E2` | 5 | 4 | 3 | 3 | 4 | 4 | 4 | hard defer |
| `R0001-P76-E7` | 4 | 5 | 4 | 3 | 4 | 5 | 3 | defer |
| `R0001-P83-E2` | 4 | 4 | 2 | 2 | 3 | 4 | 3 | reject current form |
| `R0001-P76-E8` | 5 | 2 | 2 | 3 | 3 | 2 | 2 | reject current form / hard defer |

### D 的关键反驳

1. `P83-E2` 会自证：
   - 当前 P83 receipt只保存量化 canonical candidate、score hash、index和identity；
   - canonical center量化到毫米、normal量化到 `1e-4`；
   - 父进程若只对worker自己给出的full-precision bytes重算hash，不能独立证明动作将
     消费的精确candidate；
   - 该路线需要独立full-precision reconstruction，不能只加self-signed capsule。
2. `P76-E8` 依赖不闭合：
   - P83-E2提案只在旧24-Episode bank发放授权；
   - P76-E8却要求历史不相交的新鲜seed；
   - 旧bank authorization不能授权fresh holdout，临时扩展issuer是未冻结变量。
3. `P76-E8` 同时要求request精确 `0..399` 并允许predictive rejection提前终止，
   两者矛盾；有效Episode应要求连续前缀 `0..k, k<=399`，只有safe-entry才是完整
   `0..399`。
4. P88-E2当前proof verifier只从申报坐标复算，能检查soundness但不能排除漏掉整个
   raw anchor或dropped component；必须独立扫描完整anchor域。
5. P88-E1的30像素base fixture可复算为 `6 masked / 24 retained`，但手臂、component
   和重复坐标fixture尚未展开完整数值与裕量。

## 筛选 Agent E

| 提案 | 目标价值 | 证据强度 | 可检验性 | 因果可归因性 | 通用性 | 实施成本 | 回归风险 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R0001-P87-E2` | 2 | 5 | 5 | 5 | 2 | 5 | 5 | major，defer |
| `R0001-P88-E1` | 4 | 4 | 4 | 4 | 3 | 5 | 4 | 唯一条件首选 |
| `R0001-P88-E2` | 5 | 3 | 4 | 3 | 4 | 4 | 3 | blocker，hard defer |
| `R0001-P76-E7` | 4 | 4 | 3 | 2 | 3 | 3 | 2 | blocker，reject current form |
| `R0001-P83-E2` | 3 | 5 | 5 | 4 | 3 | 4 | 3 | blocker/major，defer |
| `R0001-P76-E8` | 5 | 3 | 4 | 2 | 3 | 2 | 2 | blocker，hard defer |

### E 的关键反驳

1. P88-E1的公开truth会形成“看过答案后的自测”。如果实施者能看到全部人工truth，
   可以按fixture名称、尺寸、case ID或已知坐标硬编码，之后的comparer mutation仍可能
   通过。必须加入在worker blob冻结前hash-committed、实施者不可见、冻结后才揭盲的
   hidden challenge。
2. P76-E7不是单一变量：
   - 从in-process generic callable改为外部service，同时引入serialization、IPC、
     supervisor、session、重连与错误处理；
   - `400/400` synthetic bytes相同不能证明live payload、时序、latency和physical
     trace同构；
   - 当前形式最多是least-authority protocol qualification，不能直接成为P76-E8
     production-isomorphic证书。
3. P83-E2与fresh P76-E8存在同一cohort身份矛盾；issuer若要在fresh Episode重新发证，
   必须另行冻结source、session、单次消费、失效与揭盲时点。
4. P87-E2的80个状态是有效完整枚举，但只证明有限合成fixture的普通合同kernel；
   不能缩小当前coordinate/association主瓶颈，研究价值低于P88-E1。

## 主 Agent 裁决

### 唯一入选

`R0001-P88-E1`

选择理由：

1. 两名筛选 Agent 独立一致选择；
2. 无科学前置，不改production，成本最低且能直接处理当前association路线缺失的外部
   coordinate/component真值；
3. 若失败可在秒级阻断P88-E2与P68，避免支付真实bank重建和segmentation replay成本；
4. 若通过也只建立oracle资格，不提前声明真实bank、association、prefix或能力改善；
5. 筛选发现的truth可见性blocker可通过“密封challenge commitment早于worker commit、
   worker冻结后才揭盲”在实现前关闭。

### 必须收窄

1. 只验证coordinate/self-mask/component/merge ledger资格；不读取P50/P79/P83，不运行
   真实bank，不生成selection、action、MuJoCo状态或能力结果。
2. truth author、worker implementer、final red-team auditor角色分离。
3. 先提交public truth与加密hidden package/hash commitment，再创建worker实施worktree；
   worker prompt禁止访问密钥、明文challenge及项目外路径。
4. worker固定提交后才允许主 Agent揭盲hidden challenge；任何worker改动使资格失效。
5. worker必须走同一generic schema和code path；禁止按case名称、fixture大小、ID、
   固定坐标或known count分支。
6. E1同时资格化未来E2 worker与proof-verifier所需的基础算法；accepted不自动运行E2。
7. truth/comparer、worker和独立红队不得共享科学helper；AST/token similarity仅是辅助
   门，不能替代隐藏challenge与人工复算。

### 冻结前必须展开

- public base fixture的30个完整
  `(capture,row,column,depth,x,y,z,masked,margin)`；
- base pose、camera transform、intrinsics、dtype、坐标系、左右6关节与lateral offset；
- 左右arm capsule各自inside/outside点、segment端点、最近距离、threshold与至少
  `0.01m` margin，且不得同时命中base box；
- A-B-C全部center、normal、width、两两distance/cosine/width delta与margin；边集必须
  精确为 `{A-B,B-C}`，component为 `{A,B,C}`；
- single-view dropped component、raw multiset `6` 与unique identity `5` 的完整账本；
- hidden package的ciphertext/hash、plaintext hash commitment、case/inventory数量，\n+  但不在worker冻结前公开plaintext或key。

### Exact mutation inventory

| ID | 唯一变异 | 预期 |
|---|---|---|
| `M01` | masked/retained等量交换 | semantic reject |
| `M02` | row/column交换 | semantic reject |
| `M03` | duplicate-one/drop-another | semantic reject |
| `M04` | 删除capture ordinal造成跨帧碰撞 | semantic reject |
| `M05` | 等量跨component成员交换 | semantic reject |
| `M06` | 连通分量误写为clique | semantic reject |
| `M07` | 保留应丢弃的single-view component | semantic reject |
| `M08` | pre-self-mask冒充post-self-mask | semantic reject |
| `M09` | 删除一个完整dropped component | completeness reject |
| `M10` | 加入伪raw anchor | completeness reject |
| `M11` | 混淆同capture与跨capture重复 | identity reject |
| `C01` | 输入anchor顺序置换 | 结果规范化不变 |
| `C02` | opaque case ID双射重映射 | 结果不变 |

inventory必须精确；缺失、重复、额外、未进入真实comparer路径或仅触发schema/hash错误
均为invalid。

### 接受与停止原则

`accepted`必须同时满足：

- public和hidden全部fixture的coordinate set/multiset、component partition、edge set、
  retained/dropped状态及原因、raw与unique ledger均100% exact；
- base fixture精确 `6 masked / 24 retained`；
- A-B-C边集与传递component exact；
- raw `6`、unique `5` exact；
- hidden challenge在worker冻结前未向实施者暴露；
- worker blob在两次执行与揭盲前后不变；
- worker环境中`hwr`不可导入、truth read count为0、项目外读取为0；
- exact mutation/control inventory全通过；
- 两次输出bit-identical；
- 外部观测wall `<30s`、process-tree RSS `<512MiB`、artifact `<4MiB`；
- 独立红队0 blocker、0 major。

任一truth/fixture在worker实现后变化、数值落在未冻结容差、worker读取truth或production、
mutation survivor、receipt未原子封存、资源越界，或试图同轮运行P88-E2/P68/P76时，
立即判`invalid/stop`。

## 其他提案裁决

| ID | 裁决 | 理由 |
|---|---|---|
| `R0001-P87-E2` | deferred | 数学设计有效，但只属合成合同kernel资格测试，不能直接缩小主瓶颈 |
| `R0001-P88-E2` | hard defer | 严格依赖P88-E1 accepted，且proof completeness需重设计 |
| `R0001-P76-E7` | rejected in current form | 新IPC协议不是单变量，synthetic equality不足以证明production同构 |
| `R0001-P83-E2` | rejected in current form | full-precision capsule自证，且不能授权fresh holdout |
| `R0001-P76-E8` | hard defer | 依赖不闭合，request门与提前停止矛盾，fresh issuance未冻结 |

## 本轮禁止项

- 不实施或运行P87-E2、P88-E2、P76-E7、P83-E2、P76-E8；
- 不读取或修改P50/P79/P83 artifact；
- 不修改production candidate、selector、runtime、安全或训练代码；
- 不启动MuJoCo物理cohort、selector、Replay、Actor、世界模型训练或capability
  evaluation；
- 不修改`docs/research-loop/0001/`～`0017/`。
