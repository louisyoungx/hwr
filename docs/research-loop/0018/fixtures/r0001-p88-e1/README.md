# R0001-P88-E1 人工精确真值包

本目录是 `R0001-P88-E1` 在 worker 实现前冻结的 public fixture/truth 与 sealed
hidden challenge。它只资格化 coordinate back-projection、self-mask、raw identity、
component 与 merge ledger；不读取 P50/P79/P83，不改 production，不构成真实 bank、
association、selection、action、物理闭环或能力改善证据。

## 公开文件

- `manifest.json`：文件角色、schema、字节数与 SHA-256 索引。
- `public-fixtures.json`：公开输入、固定 pose/transform/intrinsics/dtype/坐标系、
  双臂关节与 lateral、30 像素输入、arm capsule query、raw anchor 和 identity
  multiset。
- `public-truth.json`：不调用 `hwr` 或 production helper 独立推导的完整公开真值。
- `mutation-inventory.json`：精确的 `M01`–`M11` 与 `C01`–`C02` inventory。
- `hidden-challenge.aes-256-ctr.bin`：hidden input 与 hidden truth 的密文。
- `hidden-key.rsa-oaep-sha256.bin`：RSA-OAEP-SHA256 封装的 32-byte AES key 与
  16-byte IV，明文布局为 `key || IV`。
- `commitment.json`：明文包、密文、封装 key、public key 的 SHA-256 commitment，
  以及不会泄露答案的 case/schema/inventory 数量。
- `public-key.pem`：只用于封装 challenge key material 的冻结 RSA 公钥。

所有 JSON 都使用 UTF-8、2-space indentation、单个末尾 LF。hidden plaintext 是
递归 key 排序、compact separators、单个末尾 LF 的 deterministic JSON package；
给定同一 plaintext bytes 时其 package SHA-256 唯一确定。AES key 与 IV 每次封装均由
加密安全随机源产生，因此密文不要求跨重新封装一致。

## 加密格式

1. content encryption：`AES-256-CTR`；
2. key material：随机 32-byte AES key 后紧接随机 16-byte IV；
3. key encryption：RSA-OAEP，OAEP digest 与 MGF1 digest 均为 `SHA-256`；
4. `commitment.json` 绑定 plaintext package、ciphertext、encrypted key 与 public key。

授权揭盲者可按以下占位命令操作；占位符不包含私钥路径、私钥值、AES key 或 IV：

```sh
openssl pkeyutl -decrypt \
  -inkey <AUTHORIZED_PRIVATE_KEY_PEM> \
  -pkeyopt rsa_padding_mode:oaep \
  -pkeyopt rsa_oaep_md:sha256 \
  -pkeyopt rsa_mgf1_md:sha256 \
  -in hidden-key.rsa-oaep-sha256.bin \
  -out <KEY_AND_IV_BIN>

openssl enc -d -aes-256-ctr \
  -K <64_HEX_AES_KEY_FROM_FIRST_32_BYTES> \
  -iv <32_HEX_IV_FROM_FINAL_16_BYTES> \
  -in hidden-challenge.aes-256-ctr.bin \
  -out <AUTHORIZED_HIDDEN_PACKAGE_JSON>
```

解密后必须先校验 `commitment.json` 中的 plaintext package SHA-256，再解析 JSON。
本目录不保存私钥、hidden plaintext、AES key、IV 或其十六进制表示。

## 角色分离与揭盲

- truth author：在 worker 实现前冻结 public truth、hidden ciphertext 和 commitment；
  不实现待测 worker，不提交 Git。
- worker implementer：只能看到公开输入、公开真值与密封文件；不得获得私钥、hidden
  plaintext、key/IV，不得按 case ID、fixture 尺寸、固定坐标或 known count 分支。
- main Agent：只有在 worker blob 固定并记录不可变 hash/commit 后才可授权揭盲。
- final red-team auditor：从 worker 独立比较 public 与 hidden exact semantics，并检查
  mutation/control inventory、隔离、原子封存和 worker blob 不变性。

worker 固定后才揭盲。若 worker 在揭盲后发生任何变化，本资格结果失效，必须重新密封
challenge 并重新执行完整流程。

## 公开真值边界

- base grid 明确展开 30 个 `(capture,row,column,depth,x,y,z,masked,retained,
  base_signed_margin)`；精确为 6 masked、24 retained。
- 左右 arm 各有一个 inside 和 outside query；truth 展开 chain、segment endpoints、
  nearest distance、`threshold = radius + 0.06`、signed margin 与 base-box
  non-overlap。
- merge ledger 展开全部 raw anchor、全部 pair metrics、edge/non-edge set、component
  partition、view set、retained/dropped reason 与 merged candidate。
- identity ledger 展开 raw multiplicity 6 与 unique identity 5；identity 固定为
  `(episode,capture,row,column)`。
- 所有公开判定 margin 非零；arm query 不与 base box 重叠。

文件完整性以 `manifest.json` 和 `commitment.json` 中的 SHA-256 为准。验证流程不得
尝试解密，除非 main Agent 已确认 worker 固定并正式授权揭盲。
