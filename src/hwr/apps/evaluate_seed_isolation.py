"""Diagnose standard policy-reset seed isolation without capability claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hwr.core.embodied import (
    ActionChunk,
    DualArmAction,
    DualArmActionFrame,
    DualArmObservation,
    DualArmProprioception,
    NaturalLanguageInstruction,
)
from hwr.core.runtime import PolicySpec, RuntimeStepOutcome
from hwr.core.types import EpisodeResult
from hwr.eval import (
    PlannedEpisodeSeed,
    evaluate_bimanual_policy,
    plan_episode_seeds,
    seed_lineage_manifest,
)


TASK_ID = "seed-isolation-diagnostic/v1"
PLAN_ID = "R0001-P39-E1-diagnostic"
THREAT_MODEL = "standard_policy_reset_interface_only"
MODULE_NAME = "hwr.apps.evaluate_seed_isolation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--episode-count", type=int, default=8)
    return parser


def _observation() -> DualArmObservation:
    return DualArmObservation(
        timestamp_ns=0,
        sequence_id=0,
        task_id=TASK_ID,
        instruction=NaturalLanguageInstruction(
            "Run the seed-isolation interface diagnostic"
        ),
        proprioception=DualArmProprioception(
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            (0.0,) * 6,
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0),
        ),
        cameras=(),
    )


class _DiagnosticEnvironment:
    def __init__(self, received_seeds: list[int]) -> None:
        self.received_seeds = received_seeds
        self.observation: DualArmObservation | None = None
        self._result: EpisodeResult | None = None

    def reset(self, *, seed: int, task_id: str) -> DualArmObservation:
        if task_id != TASK_ID:
            raise ValueError("diagnostic task identity differs")
        self.received_seeds.append(seed)
        self.observation = _observation()
        self._result = None
        return self.observation

    def observe(self) -> DualArmObservation:
        if self.observation is None:
            raise RuntimeError("diagnostic environment was not reset")
        return self.observation

    def apply(self, frame: DualArmActionFrame) -> RuntimeStepOutcome:
        self._result = EpisodeResult(True, "diagnostic_complete", 1, {})
        return RuntimeStepOutcome(
            self.observe(),
            terminated=True,
            info={"applied_action": frame, "safety_intervened": False},
        )

    def result(self) -> EpisodeResult | None:
        return self._result

    def task_audit(self) -> dict[str, object]:
        return {
            "stable_steps": 1,
            "maximum_concurrent_steps": 1,
            "left_contact_steps": 0,
            "right_contact_steps": 0,
            "simultaneous_contact_steps": 0,
            "severe_collision_count": 0,
            "maximum_forbidden_force": 0.0,
        }

    def close(self) -> None:
        pass


@dataclass
class _SeedCanaryPolicy:
    received_seeds: list[int]
    action_values: list[float]
    generator: random.Random | None = None

    def spec(self) -> PolicySpec:
        return PolicySpec("R0001-P39-seed-canary", 1, 1, 20.0, 12)

    def reset(self, *, task_id: str, seed: int) -> None:
        if task_id != TASK_ID:
            raise ValueError("diagnostic policy task identity differs")
        self.received_seeds.append(seed)
        self.generator = random.Random(seed)

    def infer(self, observations) -> ActionChunk:
        del observations
        if self.generator is None:
            raise RuntimeError("diagnostic policy was not reset")
        value = self.generator.uniform(-0.5, 0.5)
        self.action_values.append(value)
        return ActionChunk(
            (
                DualArmAction(
                    value,
                    0.0,
                    (0.0,) * 6,
                    (0.0,) * 6,
                    0.0,
                    0.0,
                ),
            ),
            1,
        )

    def record_applied_action(self, action: DualArmAction) -> None:
        del action

    def close(self) -> None:
        pass


def _execute(
    episode_seeds: Sequence[PlannedEpisodeSeed],
) -> tuple[dict[str, Any], list[int], list[int], list[float]]:
    environment_seeds: list[int] = []
    policy_seeds: list[int] = []
    action_values: list[float] = []
    policy = _SeedCanaryPolicy(policy_seeds, action_values)
    report = evaluate_bimanual_policy(
        TASK_ID,
        1,
        lambda: _DiagnosticEnvironment(environment_seeds),
        policy,
        episode_seeds,
    )
    return report.to_dict(), environment_seeds, policy_seeds, action_values


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    output = (
        arguments.output.resolve()
        if arguments.output.is_absolute()
        else (root / arguments.output).resolve()
    )
    if output.exists():
        raise FileExistsError(output)
    source_commit = _source_commit(root)
    episodes = plan_episode_seeds(
        PLAN_ID,
        TASK_ID,
        "none",
        arguments.episode_count,
        arguments.salt,
    )
    baseline = _execute(episodes)
    candidate = _execute(episodes)
    environment_expected = [episode.environment_seed for episode in episodes]
    policy_expected = [episode.policy_rng_seed for episode in episodes]
    raw_pass_through_count = sum(
        environment_seed == policy_seed
        for environment_seed, policy_seed in zip(
            baseline[1] + candidate[1],
            baseline[2] + candidate[2],
            strict=True,
        )
    )
    paired = baseline[1:3] == candidate[1:3]
    replay_identical = baseline == candidate
    passed = (
        baseline[1] == environment_expected
        and baseline[2] == policy_expected
        and paired
        and replay_identical
        and raw_pass_through_count == 0
    )
    lineage = seed_lineage_manifest(PLAN_ID, arguments.salt, episodes)
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--output",
        str(output),
        "--salt",
        arguments.salt,
        "--episode-count",
        str(arguments.episode_count),
    ]
    report = {
        "schema_version": "hwr.seed-isolation-diagnostic/v1",
        "proposal_id": "R0001-P39",
        "source_commit": source_commit,
        "invocation": {
            "module": MODULE_NAME,
            "command": command,
            "output": str(output),
        },
        "passed": passed,
        "formal_seed_bank": False,
        "capability_claim_allowed": False,
        "threat_model": THREAT_MODEL,
        "claims_excluded": [
            "closed_loop_capability_improvement",
            "malicious_same_process_policy_isolation",
            "observation_information_hiding",
        ],
        "episode_count": len(episodes),
        "raw_environment_seed_pass_through_count": raw_pass_through_count,
        "all_episode_domains_separated": all(
            episode.environment_seed != episode.policy_rng_seed
            for episode in episodes
        ),
        "baseline_candidate_seed_pair_coverage": (
            1.0 if paired else 0.0
        ),
        "bit_identical_replay": replay_identical,
        "baseline_report": baseline[0],
        "candidate_report": candidate[0],
    }
    report_bytes = _json_bytes(report)
    manifest = {
        "schema_version": "hwr.seed-isolation-run/v1",
        "proposal_id": "R0001-P39",
        "source_commit": source_commit,
        "command": command,
        "formal_seed_bank": False,
        "capability_claim_allowed": False,
        "threat_model": THREAT_MODEL,
        "seed_lineage": lineage,
        "artifacts": {
            "report.json": {
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "bytes": len(report_bytes),
            }
        },
    }
    manifest_bytes = _json_bytes(manifest)
    _create_output(output, report_bytes, manifest_bytes)
    return {
        "output": str(output),
        "passed": passed,
        "episode_count": len(episodes),
        "raw_environment_seed_pass_through_count": raw_pass_through_count,
        "report_sha256": manifest["artifacts"]["report.json"]["sha256"],
        "report_bytes": manifest["artifacts"]["report.json"]["bytes"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_bytes": len(manifest_bytes),
    }


def _create_output(
    output: Path, report_bytes: bytes, manifest_bytes: bytes
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        _atomic_write(output / "report.json", report_bytes)
        _atomic_write(output / "manifest.json", manifest_bytes)
    except BaseException:
        for name in (
            "report.json.tmp",
            "manifest.json.tmp",
            "report.json",
            "manifest.json",
        ):
            path = output / name
            if path.exists():
                path.unlink()
        output.rmdir()
        raise


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_commit(root: Path) -> str:
    git_path = root / ".git"
    if git_path.is_file():
        marker = git_path.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise RuntimeError("P39 source Git metadata is invalid")
        git_path = (root / marker.removeprefix("gitdir: ")).resolve()
    head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        loose = git_path / reference
        head = (
            loose.read_text(encoding="utf-8").strip()
            if loose.is_file()
            else _packed_reference(git_path, reference)
        )
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise RuntimeError("P39 requires a full Git source commit")
    return head


def _packed_reference(git_path: Path, reference: str) -> str:
    for line in (git_path / "packed-refs").read_text(
        encoding="utf-8"
    ).splitlines():
        fields = line.split(" ", 1)
        if len(fields) == 2 and fields[1] == reference:
            return fields[0]
    raise RuntimeError("P39 source Git reference is unresolved")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
