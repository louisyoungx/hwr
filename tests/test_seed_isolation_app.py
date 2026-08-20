import json
import hashlib

import pytest

from hwr.apps.evaluate_seed_isolation import build_parser, run


SALT = "R0001-P39-E1-s20263901"


def test_seed_isolation_cli_requires_explicit_output_and_salt() -> None:
    arguments = build_parser().parse_args(
        ["--output", "runs/diagnostic", "--salt", SALT]
    )

    assert arguments.output.as_posix() == "runs/diagnostic"
    assert arguments.salt == SALT
    assert arguments.episode_count == 8


def test_seed_isolation_diagnostic_is_replayable_and_forbids_claims(
    tmp_path,
) -> None:
    output = tmp_path / "seed-isolation"
    arguments = build_parser().parse_args(
        [
            "--output",
            str(output),
            "--salt",
            SALT,
            "--episode-count",
            "4",
        ]
    )

    result = run(arguments)
    report = json.loads((output / "report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())

    assert result["passed"] is True
    assert result["raw_environment_seed_pass_through_count"] == 0
    assert len(report["source_commit"]) == 40
    assert manifest["source_commit"] == report["source_commit"]
    assert report["invocation"]["module"] == (
        "hwr.apps.evaluate_seed_isolation"
    )
    assert manifest["command"] == report["invocation"]["command"]
    assert report["formal_seed_bank"] is False
    assert report["capability_claim_allowed"] is False
    assert report["threat_model"] == "standard_policy_reset_interface_only"
    assert report["all_episode_domains_separated"] is True
    assert report["baseline_candidate_seed_pair_coverage"] == 1.0
    assert report["bit_identical_replay"] is True
    assert manifest["seed_lineage"]["commitment"]["salt_sha256"] == (
        "a94db502b86fd2c83a9096eb856b110de"
        "53158f588cc7496a60e4264fc190237"
    )
    assert manifest["seed_lineage"]["reveal"] == {
        "commitment_verified": True,
        "salt": SALT,
    }
    assert len(manifest["seed_lineage"]["episodes"]) == 4
    report_bytes = (output / "report.json").read_bytes()
    manifest_bytes = (output / "manifest.json").read_bytes()
    assert manifest["artifacts"]["report.json"] == {
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
        "bytes": len(report_bytes),
    }
    assert result["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert result["report_bytes"] == len(report_bytes)
    assert result["manifest_sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    assert result["manifest_bytes"] == len(manifest_bytes)

    with pytest.raises(FileExistsError):
        run(arguments)
