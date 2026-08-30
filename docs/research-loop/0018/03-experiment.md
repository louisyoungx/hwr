# R0018 冻结实验

## 实验身份

- 提案：`R0001-P88-E1`
- 实验：`R0001-P88-E1`
- 名称：manual exact coordinate-oracle qualification
- 类型：合成 fixture 上的 measurement-oracle 资格验证
- 状态：已冻结，等待实现
- 基线：当前不存在通过人工 exact truth 与 hidden challenge 的 coordinate oracle
- 候选：独立、通用、无 `hwr` import 的 coordinate/self-mask/component/ledger worker
- truth commitment提交：
  `c8b715b65374eaa4f1ce68222cfaeceff136e0fa`
- 实施分支：`exp/R0001-P88-E1-coordinate-oracle`
- 实施 worktree：
  `/Users/louis/Developer/AIWorkspace/50-housework-robot-r0018-p88`
- 唯一实施 Agent：一名，不得参与 truth 编写或 hidden challenge解密
- 最终审计 Agent：一名独立只读 Agent，不得参与 truth或worker实现

本轮只实施 `R0001-P88-E1`。不得顺带实施或运行 P87-E2、P88-E2、P76-E7、
P83-E2、P76-E8、P68、P77、selector、默认 v2 migration、Replay、Actor、世界模型
训练或 capability evaluation。

## 唯一主假设

> 一个不依赖 production helper、在闭合输入上执行同一通用算法路径的 worker，能够在
> 实现前冻结的公开人工真值和实施者不可见的密封 hidden challenge 上，精确恢复
> camera back-projection、base/arm self-mask、raw-anchor graph、connected-component
> retention 与 coordinate identity multiset/set，并拒绝全部预注册语义 mutation。

本实验不检验：

- 真实 P50/P79/P83 bank 的 coordinate lineage；
- candidate quality、selection、association、reachability、prefix、安全能力；
- 训练、泛化、闭环成功或 deployment。

`accepted` 只意味着该固定 worker有资格进入未来重新筛选和冻结的 P88-E2；不自动
授权真实 bank运行或 P68。

## 因果变量

- 唯一主变量：是否使用候选 worker从冻结 fixture输入计算 ledger。
- 正确性对照：实现前冻结且与 worker代码来源分离的 exact truth。
- 防硬编码对照：worker冻结前不可见的 hidden challenge。
- 不改变 production generator、threshold、candidate、selector、association或历史
  artifact。
- public/hidden都使用相同 fixture schema、truth schema、case kind与通用worker路径；
  禁止按case name、ID、输入尺寸、known count或固定坐标分支。

## 冻结输入

根目录：

`docs/research-loop/0018/fixtures/r0001-p88-e1`

### 文件身份

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `README.md` | 4,415 | `192f2433688623509050b7ea3d6a96df015cd5455bd718087714ccefb77376c2` |
| `public-key.pem` | 625 | `12a7a0d7504b42b56ca36171e759cd7c09974fc7546db41fce773aecd9fc5da2` |
| `public-fixtures.json` | 7,655 | `a2778edeac61a77a4f13a099ce0942b009f1536bfe658a1e09bef9e7c89bc63d` |
| `public-truth.json` | 21,157 | `c0bddb58773a6ff5f70464f64f5e858123c5aa6b49029bfa91aa9458587e2c85` |
| `mutation-inventory.json` | 4,230 | `ae860cd67758c01e0c12e0645373ceffb2625975021313b8ec62e5087ffb7af0` |
| `commitment.json` | 1,341 | `4dcc49c34bdae3f3c74d3cc9595e743829e6b1225aa942f454ef9e0e43de7e17` |
| `hidden-challenge.aes-256-ctr.bin` | 16,963 | `16eb680df74224e6b7464be6822d5aaae785d506e3af6c317762d10d68b1dd89` |
| `hidden-key.rsa-oaep-sha256.bin` | 384 | `31696a8b7c4e764aad14a40c2b21ef47ae2be00cacb6fec3992dea29eaace6f8` |
| `manifest.json` | 3,703 | `5efa8aef564fb279e7705f0a0fded6d8a4819c88c6e20385a736f6baffbf6e5b` |

Hidden plaintext package commitment：

`8e7864059f4d1fccef9c3be7edee29c88524371ed05475c35e95de2470b9e501`

### Public truth

公开包冻结 3 个 case：

1. `public-base-grid-30`
   - intrinsics：`fx=20, fy=80, cx=189.5, cy=96`；
   - identity camera/base transform；
   - rows `94..99`、columns `198..202`、depth `0.8`；
   - 30 个像素全部展开；
   - expected：column 198 的 6 点 masked，其余 24 点 retained；
   - 最小绝对 margin `0.01m`。
2. `public-arm-capsules`
   - base/acquisition pose、identity transform、左右6关节全零；
   - lateral `+0.31/-0.31`；
   - 左右 link-2各一个inside与outside query；
   - 每项展开segment endpoints、nearest distance、`radius+0.06` threshold、
     signed margin；
   - 四点均不与base box重叠，最小capsule绝对margin `0.021m`。
3. `public-merge-ledger`
   - raw anchors A–E完整展开；
   - edge set精确为 `{A-B, B-C, D-E}`；
   - A-B-C形成传递component并retained；
   - D-E只有一个distinct capture，必须dropped；
   - raw identity multiplicity `6`、unique identity `5`；
   - identity为 `(episode,capture,row,column)`。

主 Agent已在不 import `hwr` 的独立脚本中复算：

- manifest `8/8`、commitment `3/3`；
- base pixels `30/30`；
- arm chain/endpoints/distance/margin `4/4`；
- pair metrics `10/10`、edge与component partition一致；
- raw/unique identity `6/5`。

### Hidden challenge

- case count：3；
- distinct schema count：3；
- 使用与public相同的fixture/truth schema与case kind，但数值和排列不同；
- plaintext、AES key与IV未进入Git；
- content encryption：AES-256-CTR；
- key wrapping：RSA-OAEP-SHA256；
- 私钥由主 Agent保存在项目外，worker实施者不得访问；
- worker提交与hash冻结前禁止解密；
- 揭盲后若worker任何字节变化，实验立即invalid，必须重新密封challenge后再开始。

## 样本单位

- 这是软件/测量合同资格验证，没有科学Episode或seed样本。
- public case为3，hidden case为3。
- pixel、arm point、anchor、edge、component、coordinate、mutation或worker run都不是
  独立科学样本，不进行显著性检验或置信区间。
- 结果只报告 exact agreement与mutation判别。

## Worker 输出合同

worker只从stdin读取一个fixture JSON，只向stdout写一个canonical JSON receipt：

- schema与proposal ID；
- 每个case的opaque case ID与kind；
- back-projected point ledger及masked/retained；
- arm chain、query nearest segment/distance/threshold/mask；
- raw pair metrics、edge/non-edge set；
- component partition、view set、retained/dropped状态与reason；
- retained component merged candidate；
- coordinate raw multiset、unique set、duplicate identities；
- input SHA-256与receipt payload SHA-256。

要求：

1. 纯标准库 Python；使用 `python -I -S`，空 `PYTHONPATH`；
2. 禁止 import `hwr`、NumPy、production或项目模块；
3. 禁止文件读取、网络、subprocess、dynamic import、`eval/exec/compile`；
4. 同一generic路径处理public与hidden；
5. canonical排序只能基于语义字段，不能依赖输入anchor顺序；
6. worker异常时不得输出partial valid receipt。

## Parent comparer 与隔离

- 父进程负责：
  - fixture/truth/manifest/hash验证；
  - staging只含固定worker与当前fixture，truth不进入blind root；
  - 通过stdin/stdout执行worker；
  - worker source静态审计与hash稳定性；
  - public/hidden exact comparison；
  - mutation/control；
  - atomic artifact；
  - wall、process-tree RSS与artifact bytes外部观测。
- comparer不得import worker或production helper，不得用worker生成expected。
- exact comparison必须覆盖set、multiset、edge、partition、retained/dropped reason和
  merged ledger；count/hash不能替代内容。
- worker source与 `target_selection.py`、`candidate_mask_ownership.py` 的whole-source
  token/AST similarity必须均 `<=0.45`；这只是辅助门。
- blind root和output路径逐级拒绝symlink；现有output或staging存在时fail closed。

## Exact mutation/control inventory

| ID | 唯一变异 | 预期类别 |
|---|---|---|
| `M01` | masked/retained等量交换 | `semantic_reject` |
| `M02` | row/column交换 | `semantic_reject` |
| `M03` | duplicate-one/drop-another | `semantic_reject` |
| `M04` | 删除capture ordinal造成跨帧碰撞 | `semantic_reject` |
| `M05` | 等量跨component成员交换 | `semantic_reject` |
| `M06` | 连通分量误写为clique | `semantic_reject` |
| `M07` | 保留应丢弃的single-view component | `semantic_reject` |
| `M08` | pre-self-mask冒充post-self-mask | `semantic_reject` |
| `M09` | 删除完整dropped component | `completeness_reject` |
| `M10` | 加入伪raw anchor | `completeness_reject` |
| `M11` | 混淆同capture与跨capture重复 | `identity_reject` |
| `C01` | 输入raw-anchor顺序置换 | `canonical_result_unchanged` |
| `C02` | opaque case ID双射重映射 | `semantic_result_unchanged` |

inventory必须精确为13项。缺失、重复、额外、未进入真实comparer路径、只触发schema/hash
错误或修改未执行代码均为invalid。M01–M11必须被对应语义类别拒绝；C01–C02必须保持
语义结果不变。

## 主要指标

- public exact case count `3/3`；
- hidden exact case count `3/3`；
- public/hidden coordinate set与multiset exact；
- public/hidden arm chain、nearest distance、threshold、mask exact；
- public/hidden edge set、component partition、retained/dropped reason exact；
- public/hidden raw/unique identity ledger exact；
- base public fixture `6 masked / 24 retained`；
- public A-B-C edge与传递component exact；
- public raw/unique `6/5` exact；
- mutation/control `13/13`；
- worker重复执行 bit-identical。

## 守护指标

- truth commit严格早于worker实现提交；
- public与hidden input/truth hash全部匹配；
- hidden在worker冻结前未揭盲；
- worker blob在揭盲前后与两次运行间不变；
- worker `hwr` import count、production helper count、user-file read count均0；
- truth进入blind root count 0；
- 项目外读取count 0；
- source similarity通过；
- history tree `0001`～`0017` 不变；
- changed files精确等于允许集合；
- focused tests、Python size、architecture、compileall、`git diff --check`通过；
- 独立红队 0 blocker、0 major；
- wall `<30s`；
- parent-observed process-tree peak RSS `<512MiB`；
- final artifact `<4MiB`；
- 数据卷剩余空间 `>=20GiB`。

## 判定

### `accepted as fixture-qualified coordinate-oracle`

所有主要与守护指标通过。只允许声明固定worker通过public与hidden人工exact truth资格门。

### `rejected`

fixture、truth、commitment、揭盲、隔离与独立comparer均有效，但固定worker在任一public
或hidden语义上稳定不一致。

### `invalid`

包括但不限于：

- truth/fixture在worker实现后变化；
- hidden在worker提交冻结前泄露；
- worker揭盲后变化；
- 输入或commitment hash漂移；
- 任一点/边落入未冻结容差或独立复算不一致；
- worker读取truth、项目外路径、`hwr`或production helper；
- worker按case ID/size/known count/固定坐标分支；
- comparer只验证count/hash或与worker共享科学helper；
- mutation inventory不精确、mutation未进入真实路径或任一survivor；
- receipt非原子、partial artifact、资源越界；
- 同一实验运行真实P88-E2、P68、P76或任何物理/训练工作。

### 提前停止

任一invalid条件出现立即停止，不修补冻结truth或worker，不启动真实bank重建。

## 资源预算

- CPU-only；
- formal wall `<30s`；
- process-tree peak RSS `<512MiB`；
- artifact `<4MiB`；
- 数据卷剩余空间 `>=20GiB`；
- 无MuJoCo、Torch、GPU、tmux、后台任务、休眠或host-exec；
- focused tests、repository gates、full pytest与formal evaluator分开计量。

## 允许写集合

实施 Agent只允许新增：

- `scripts/evaluate_manual_coordinate_oracle_worker.py`
- `src/hwr/eval/manual_coordinate_oracle.py`
- `src/hwr/apps/evaluate_manual_coordinate_oracle.py`
- `tests/test_manual_coordinate_oracle.py`
- `tests/test_manual_coordinate_oracle_app.py`

主 Agent另外只允许修改：

- `docs/research-loop/0018/03-experiment.md`
- `docs/research-loop/0018/04-results.md`
- `docs/research-loop/0018/05-summary.md`

正式输出只允许新建：

`runs/research-loop/0018/r0018-p88-e1-coordinate-oracle-s20268801`

禁止修改：

- `docs/research-loop/0018/fixtures/r0001-p88-e1/`
- `src/hwr/eval/target_selection.py`
- `src/hwr/eval/candidate_mask_ownership.py`
- P50/P79/P83代码与artifact；
- `docs/research-loop/0001/`～`0017/`；
- 其他production、训练、安全与评测代码。

## 实施与验证顺序

1. 提交本冻结文档；记录该提交为worker parent。
2. 从该提交创建独立worktree与`exp/R0001-P88-E1-coordinate-oracle`。
3. 唯一实施 Agent只能读取仓库内公开内容，不得访问项目外路径、私钥或hidden明文。
4. 实施 Agent完成上述五文件与一个原子提交；不得运行hidden challenge。
5. 主 Agent核查changed-file scope、测试、size、architecture、compileall和source audit。
6. worker source hash与commit冻结；从此禁止修改。
7. 独立红队在不揭盲条件下审计public路径、隔离、mutation、硬编码与atomic output。
8. 只有0 blocker、0 major时，主 Agent才在项目外临时目录解密hidden package并校验
   plaintext commitment。
9. hidden fixture只进入blind input；hidden truth只由parent comparer在worker退出后读取。
10. 运行public与hidden formal evaluator；两次worker output必须bit-identical。
11. 独立红队复核artifact、hash、资源与worker不变性。
12. 按本文件预注册门写入`04-results.md`；不得后验调整。
