from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_training_with_lark_notify.sh"
MESSAGE_SCRIPT = ROOT / "scripts" / "send_lark_agent_message.sh"


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


def _run_wrapper(tmp_path: Path, training_exit: int) -> subprocess.CompletedProcess[str]:
    binary_dir, arguments_path = _fake_lark_cli(tmp_path)
    run_id = "pilot-test"
    command = (
        "mkdir -p runs/bimanual-rl/pilot-test; "
        "printf '{}\\n' > runs/bimanual-rl/pilot-test/episodes.jsonl; "
        "printf checkpoint > runs/bimanual-rl/pilot-test/training-checkpoint.pt; "
        f"exit {training_exit}"
    )
    environment = os.environ.copy()
    environment.update(
        PATH=f"{binary_dir}:{environment['PATH']}",
        LARK_ARGUMENTS=str(arguments_path),
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
