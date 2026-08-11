from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_training_with_lark_notify.sh"


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
            "ou_recipient",
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


def test_training_wrapper_notifies_as_bot_after_success(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, training_exit=0)

    assert result.returncode == 0
    arguments = result.lark_arguments  # type: ignore[attr-defined]
    assert "--as\nbot\n" in arguments
    assert "--user-id\nou_recipient\n" in arguments
    assert "训练已完成" in arguments
    assert "Episode 记录数: 1" in arguments


def test_training_wrapper_preserves_training_failure_status(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, training_exit=7)

    assert result.returncode == 7
    arguments = result.lark_arguments  # type: ignore[attr-defined]
    assert "训练异常退出" in arguments
    assert "状态码: 7" in arguments
