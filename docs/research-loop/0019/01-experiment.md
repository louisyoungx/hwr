# R0019 实验

## 主假设

`carry_living_room_basket/v1` 在不改变正常 MuJoCo 物理、任务成功状态机和独立安全层的前提下，
可由读取 simulator-private state 的闭环 teacher 稳定完成。teacher 使用显式任务状态机和
当前几何计算动作，不依赖 generic candidate 或 B0–B7 的有限时域。

## 对照

- baseline：现有 generic B0–B7 primitive，以任务 payload 几何构造唯一 candidate，
  走正式 16 维动作、MuJoCo physics、任务 tracker 和安全层。
- teacher：任务专用 privileged feedback controller，复用同一动作/物理/安全/成功路径。
- paired 条件：同一 seed、task config、randomization、physics、安全阈值和 Episode horizon。

## 开发阶段

1. 先运行 baseline Episode，记录最早失败阶段、动作、接触、安全干预和终止原因。
2. 优先复用 backend 的 Cartesian twist + Jacobian DLS；teacher 直接从当前 handle、payload、
   target 和 tool 位姿计算闭环目标。
3. 先通过单 seed physics smoke；若失败，只修复最早稳定阻塞，不创建额外资格门。
4. development seed domain：`0 <= seed < 1_000_000`；不得进入 confirmation。

## 冻结确认设计

以下设计在任何 confirmation 结果可见前冻结：

- controller：`GenericBasketPrimitiveBaseline` 与
  `PrivilegedBasketTeacher`；
- seed domain：`91_900_001 + 104_729 * index`，`index=0..99`；
- Episode：每个 controller 100 个，按相同 seed paired；
- teacher success `>=80/100`；
- actual severe collision `=0`；
- 只允许从干净、已提交的 worktree 启动；runner 在执行首个 Episode 前拒绝脏树；
- 安全守护：`DualArmSafetySupervisor`、两步 predictive collision、`220N` severe
  threshold 均保持默认值；
- 每个 Episode 最多 1,200 step，整次运行 wall time 最多 1,800 秒；
- 单个失败或安全拒绝只记为该 Episode 结果，不中断 cohort；
- 完整报告 baseline/teacher 每个 Episode、全部失败、安全干预、双臂接触和终止原因；
- 判定器严格验证冻结 seed domain 以及每个 seed 恰有一条 baseline 和一条 teacher 结果；
- 唯一命令：

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_bimanual_teacher \
  --mode confirmation \
  --controller paired \
  --output runs/research-loop/0019/confirmation/paired-100.json
```

若 development 资源实测表明 100 paired Episode 明显不合理，只能在首次 confirmation 运行前
调整并记录理由，不能查看 confirmation 结果后改门。

本轮 development 结果为 teacher `0/6` success，未满足“至少先出现一个端到端开发成功”的
扩大条件。因此本轮不启动上述 confirmation 命令；这不是调整 seed、Episode 数或成功门。

## 结论

- 达门：`validated_development`。允许声明任务与控制链存在可行 teacher ceiling；不允许声明
  可部署状态/视觉策略、泛化或硬件能力。
- 未达门：按证据使用 `inconclusive_capability`、`invalid` 或 `abandoned`，并报告最早稳定
  阻塞；不自动启动下一轮。
