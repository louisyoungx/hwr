from __future__ import annotations

from hwr.apps.train_scenario import build_parser


def test_training_cli_parses_scenario_configuration(tmp_path) -> None:
    arguments = build_parser().parse_args(
        [
            "tidy_table/v1",
            "--run-id",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--episodes",
            "2",
            "--epochs",
            "1",
        ]
    )

    assert arguments.task_id == "tidy_table/v1"
    assert arguments.run_id == "smoke"
    assert arguments.episodes == 2

