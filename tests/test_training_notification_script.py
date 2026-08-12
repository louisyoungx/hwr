from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_training_with_lark_notify.sh"
MESSAGE_SCRIPT = ROOT / "scripts" / "send_lark_agent_message.sh"
FOUNDATION_LAUNCHER = ROOT / "scripts" / "start_foundation_training_tmux.sh"


def _fake_lark_cli(tmp_path: Path) -> tuple[Path, Path]:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    arguments_path = tmp_path / "lark-arguments.txt"
    binary = binary_dir / "lark-cli"
    binary.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$LARK_ARGUMENTS"\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary_dir, arguments_path


def _run_wrapper(
    tmp_path: Path,
    training_exit: int,
    *,
    run_root: str = "runs/bimanual-rl",
    versioned_checkpoint: bool = False,
) -> subprocess.CompletedProcess[str]:
    binary_dir, arguments_path = _fake_lark_cli(tmp_path)
    run_id = "pilot-test"
    run_path = f"{run_root}/{run_id}"
    if versioned_checkpoint:
        command = (
            f"mkdir -p {run_path}/checkpoints/update-000000200; "
            f"printf '{{}}\\n' > {run_path}/episodes.jsonl; "
            f"printf checkpoint > {run_path}/checkpoints/update-000000200/training-state.pt; "
            f"printf '{{\"training_checkpoint\":\"checkpoints/update-000000200\"}}\\n' "
            f"> {run_path}/latest.json; exit {training_exit}"
        )
    else:
        command = (
            f"mkdir -p {run_path}; "
            f"printf '{{}}\\n' > {run_path}/episodes.jsonl; "
            f"printf checkpoint > {run_path}/training-checkpoint.pt; "
            f"exit {training_exit}"
        )
    environment = os.environ.copy()
    environment.update(
        PATH=f"{binary_dir}:{environment['PATH']}",
        LARK_ARGUMENTS=str(arguments_path),
        HWR_TRAINING_RUN_ROOT=run_root,
    )
    result = subprocess.run(
        (
            str(SCRIPT),
            run_id,
            "logs/training.log",
            "bash",
            "-c",
            command,
        ),
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    result.lark_arguments = arguments_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    return result


def test_message_script_encapsulates_agent_identity_and_recipient(tmp_path: Path) -> None:
    binary_dir, arguments_path = _fake_lark_cli(tmp_path)
    environment = os.environ.copy()
    environment.update(
        PATH=f"{binary_dir}:{environment['PATH']}",
        LARK_ARGUMENTS=str(arguments_path),
    )

    result = subprocess.run(
        (str(MESSAGE_SCRIPT), "阶段训练已完成"),
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    arguments = arguments_path.read_text(encoding="utf-8")
    assert "--as\nbot\n" in arguments
    assert "--user-id\nou_663a48636b9cd51d4a4aec323de37703\n" in arguments
    assert "--text\n阶段训练已完成\n" in arguments


def test_message_script_retries_and_returns_send_failure(tmp_path: Path) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    calls_path = tmp_path / "lark-calls.txt"
    binary = binary_dir / "lark-cli"
    binary.write_text(
        '#!/usr/bin/env bash\nprintf "call\\n" >> "$LARK_CALLS"\nexit 9\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        PATH=f"{binary_dir}:{environment['PATH']}",
        LARK_CALLS=str(calls_path),
        HWR_LARK_RETRY_BASE_SECONDS="0",
    )

    result = subprocess.run(
        (str(MESSAGE_SCRIPT), "发送失败测试"),
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 9
    assert calls_path.read_text(encoding="utf-8").splitlines() == ["call"] * 3


def test_training_wrapper_notifies_as_bot_after_success(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, training_exit=0)

    assert result.returncode == 0
    arguments = result.lark_arguments  # type: ignore[attr-defined]
    assert "--as\nbot\n" in arguments
    assert "--user-id\nou_663a48636b9cd51d4a4aec323de37703\n" in arguments
    assert "训练已完成" in arguments
    assert "Episode 记录数: 1" in arguments


def test_training_wrapper_preserves_training_failure_status(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, training_exit=7)

    assert result.returncode == 7
    arguments = result.lark_arguments  # type: ignore[attr-defined]
    assert "训练异常退出" in arguments
    assert "状态码: 7" in arguments


def test_training_wrapper_resolves_versioned_foundation_checkpoint(tmp_path: Path) -> None:
    result = _run_wrapper(
        tmp_path,
        training_exit=0,
        run_root="runs/foundation-world-model",
        versioned_checkpoint=True,
    )

    assert result.returncode == 0
    arguments = result.lark_arguments  # type: ignore[attr-defined]
    assert "runs/foundation-world-model/pilot-test" in arguments
    assert "checkpoints/update-000000200/training-state.pt" in arguments
    assert "Checkpoint SHA-256: missing" not in arguments


def test_foundation_launcher_builds_one_detached_gated_notifying_command(
    tmp_path: Path,
) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    tmux_arguments = tmp_path / "tmux-arguments.txt"
    tmux = binary_dir / "tmux"
    tmux.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == "has-session" ]]; then exit 1; fi\n'
        'printf "%s\\n" "$@" > "$TMUX_ARGUMENTS"\n',
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    lark = binary_dir / "lark-cli"
    lark.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    lark.chmod(0o755)
    readiness = tmp_path / "development-ready.json"
    readiness.write_text("{}")
    model_root = tmp_path / "models"
    model_root.mkdir()
    output_root = tmp_path / "runs"
    log_root = tmp_path / "logs"
    environment = os.environ.copy()
    environment.update(
        PATH=f"{binary_dir}:{environment['PATH']}",
        TMUX_ARGUMENTS=str(tmux_arguments),
        HWR_FOUNDATION_DEVELOPMENT_READY=str(readiness),
        HWR_FOUNDATION_MODEL_ROOT=str(model_root),
        HWR_FOUNDATION_OUTPUT_ROOT=str(output_root),
        HWR_FOUNDATION_LOG_ROOT=str(log_root),
        HWR_FOUNDATION_DEVICE="mps",
        HWR_FOUNDATION_TEACHER_DEVICE="cpu",
    )

    result = subprocess.run(
        (str(FOUNDATION_LAUNCHER), "foundation-wm-test", "--resume"),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    arguments = tmux_arguments.read_text().splitlines()
    assert arguments[:6] == [
        "new-session",
        "-d",
        "-s",
        "hwr-foundation-foundation-wm-test",
        "-c",
        str(ROOT),
    ]
    assert f"HWR_TRAINING_RUN_ROOT={output_root}" in arguments
    assert str(ROOT / "scripts/run_training_with_lark_notify.sh") in arguments
    assert "hwr.apps.train_foundation_world_model" in arguments
    assert "--development-ready" in arguments
    assert str(readiness) in arguments
    assert "--model-root" in arguments
    assert str(model_root) in arguments
    assert "--resume" in arguments


def test_foundation_launcher_rejects_unsafe_run_id_before_tmux() -> None:
    result = subprocess.run(
        (str(FOUNDATION_LAUNCHER), "unsafe/run"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unsupported characters" in result.stderr
