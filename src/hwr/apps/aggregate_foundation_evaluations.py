"""Aggregate independent foundation evaluations into one formal multi-seed gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


AGGREGATE_ACCEPTANCE_SCHEMA = "hwr.foundation-multiseed-acceptance/v1"
AGGREGATE_MANIFEST_SCHEMA = "hwr.foundation-multiseed-evaluation/v1"
EVALUATION_SCHEMA = "hwr.foundation-evaluation-run/v3"
PER_SEED_ACCEPTANCE_SCHEMA = "hwr.foundation-per-seed-acceptance/v1"


@dataclass(frozen=True)
class EvaluationIdentity:
    path: Path
    training_run: Path
    training_seed: int
    source_commit: str
    run_manifest_sha256: str
    deployment_manifest_sha256: str
    configuration: str
    evaluation_seeds: tuple[int, ...]
    training_data_seeds: frozenset[int]
    per_seed_passed: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def aggregate_foundation_evaluations(
    evaluation_paths: Sequence[Path], output_path: Path
) -> dict[str, object]:
    identities = tuple(_load_evaluation(path.resolve()) for path in evaluation_paths)
    if len({identity.path for identity in identities}) != len(identities):
        raise ValueError("foundation aggregate contains duplicate evaluation paths")
    checks = _aggregate_checks(identities)
    acceptance = {
        "schema_version": AGGREGATE_ACCEPTANCE_SCHEMA,
        "passed": all(checks.values()),
        "formal_passed": all(checks.values()),
        "required_training_seed_count": 3,
        "training_seeds": [value.training_seed for value in identities],
        "checks": checks,
        "evaluations": [
            {
                "path": str(value.path),
                "training_run": str(value.training_run),
                "training_seed": value.training_seed,
                "per_seed_passed": value.per_seed_passed,
            }
            for value in identities
        ],
    }
    output_path.mkdir(parents=True, exist_ok=False)
    _write_json(output_path / "acceptance.json", acceptance)
    manifest = _aggregate_manifest(output_path, identities, acceptance)
    _write_json(output_path / "manifest.json", manifest)
    return acceptance


def _load_evaluation(path: Path) -> EvaluationIdentity:
    manifest_path = path / "manifest.json"
    acceptance_path = path / "acceptance.json"
    report_path = path / "report.json"
    manifest = _read_json(manifest_path)
    acceptance = _read_json(acceptance_path)
    if manifest.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError("foundation evaluation schema differs")
    if acceptance.get("schema_version") != PER_SEED_ACCEPTANCE_SCHEMA:
        raise ValueError("foundation per-seed acceptance schema differs")
    if manifest.get("formal_passed") is not False:
        raise ValueError("single-seed evaluation claimed formal passage")
    _verify_evaluation_artifact(manifest, "evaluation/report.json", report_path)
    _verify_evaluation_artifact(
        manifest, "evaluation/acceptance.json", acceptance_path
    )
    training_run = Path(str(manifest["training_run"])).resolve()
    run_manifest_path = training_run / "run-manifest.json"
    run_manifest = _read_json(run_manifest_path)
    run_digest = _sha256(run_manifest_path)
    if run_digest != manifest.get("training_run_manifest_sha256"):
        raise ValueError("aggregate training run manifest hash differs")
    if manifest.get("source_commit") != run_manifest.get("source_commit"):
        raise ValueError("aggregate source commit differs")
    latest = _read_json(training_run / "latest.json")
    deployment_manifest = (
        training_run / str(latest["deployment"]) / "manifest.json"
    )
    if _sha256(deployment_manifest) != manifest.get("deployment_manifest_sha256"):
        raise ValueError("aggregate deployment manifest hash differs")
    training_seed = int(run_manifest["training_config"]["seed"])
    if training_seed != int(manifest.get("training_seed", -1)):
        raise ValueError("aggregate training seed identity differs")
    training_data_seeds = _training_data_seeds(training_run, training_seed)
    return EvaluationIdentity(
        path,
        training_run,
        training_seed,
        str(run_manifest["source_commit"]),
        run_digest,
        str(manifest["deployment_manifest_sha256"]),
        _canonical_configuration(run_manifest),
        tuple(int(value) for value in manifest["unseen_seeds"]),
        frozenset(training_data_seeds),
        acceptance.get("per_seed_passed") is True
        and manifest.get("per_seed_passed") is True,
    )


def _aggregate_checks(
    identities: Sequence[EvaluationIdentity],
) -> dict[str, bool]:
    training_sets = [value.training_data_seeds for value in identities]
    evaluation_seeds = set(
        seed for value in identities for seed in value.evaluation_seeds
    )
    return {
        "minimum_three_training_seeds": len(identities) >= 3,
        "distinct_training_seeds": len({value.training_seed for value in identities})
        == len(identities),
        "distinct_run_manifests": len(
            {value.run_manifest_sha256 for value in identities}
        )
        == len(identities),
        "distinct_deployments": len(
            {value.deployment_manifest_sha256 for value in identities}
        )
        == len(identities),
        "same_source_commit": len({value.source_commit for value in identities}) == 1,
        "same_immutable_configuration": len(
            {value.configuration for value in identities}
        )
        == 1,
        "same_evaluation_seeds": len(
            {value.evaluation_seeds for value in identities}
        )
        == 1,
        "training_seed_sets_disjoint": _sets_are_disjoint(training_sets),
        "evaluation_seeds_unseen_by_all_runs": not any(
            evaluation_seeds & seeds for seeds in training_sets
        ),
        "all_per_seed_evaluations_passed": all(
            value.per_seed_passed for value in identities
        ),
    }


def _canonical_configuration(run_manifest: Mapping[str, object]) -> str:
    value = json.loads(json.dumps(run_manifest, sort_keys=True))
    value["training_config"]["seed"] = "<training-seed>"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _training_data_seeds(run_path: Path, training_seed: int) -> set[int]:
    episodes = {
        int(json.loads(line)["seed"])
        for line in (run_path / "episodes.jsonl").read_text().splitlines()
        if line
    }
    holdout = _read_json(run_path / "causality-holdout/autonomous/manifest.json")
    return {training_seed, *episodes, *(int(shard["seed"]) for shard in holdout["shards"])}


def _sets_are_disjoint(values: Sequence[frozenset[int]]) -> bool:
    observed: set[int] = set()
    for value in values:
        if observed & value:
            return False
        observed.update(value)
    return True


def _verify_evaluation_artifact(
    manifest: Mapping[str, object], name: str, path: Path
) -> None:
    artifacts = manifest.get("artifacts")
    identity = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(identity, Mapping):
        raise ValueError(f"foundation evaluation artifact is missing: {name}")
    if identity.get("sha256") != _sha256(path) or int(identity.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"foundation evaluation artifact hash differs: {name}")


def _aggregate_manifest(
    output_path: Path,
    identities: Sequence[EvaluationIdentity],
    acceptance: Mapping[str, object],
) -> dict[str, object]:
    inputs = {}
    for index, identity in enumerate(identities):
        for name in ("manifest.json", "acceptance.json", "report.json"):
            path = identity.path / name
            inputs[f"evaluation-{index + 1}/{name}"] = {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    acceptance_path = output_path / "acceptance.json"
    return {
        "schema_version": AGGREGATE_MANIFEST_SCHEMA,
        "formal_passed": acceptance["formal_passed"],
        "acceptance_sha256": _sha256(acceptance_path),
        "inputs": inputs,
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = aggregate_foundation_evaluations(
        arguments.evaluation_paths, arguments.output.resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
