"""Task-independent diagnostics published by formal foundation training."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import torch

from hwr.train.foundation_batch import FoundationTrainingBatch
from hwr.train.foundation_registry import ACTION_CAUSALITY_SCHEMA
from hwr.train.foundation_trainer import FoundationWorldModelTrainer
from hwr.world_model import (
    ActionCausalityCriteria,
    CounterfactualCausalityReport,
    aggregate_action_causality_reports,
    assess_action_causality,
    evaluate_action_causality,
)


def evaluate_foundation_action_causality(
    trainer: FoundationWorldModelTrainer,
    batch: FoundationTrainingBatch,
    criteria: ActionCausalityCriteria,
    *,
    shuffle_seed: int = 0,
) -> dict[str, object]:
    """Evaluate every sequence in a generic batch using executed actions."""
    report = _evaluate_batch_report(trainer, batch, shuffle_seed=shuffle_seed)
    return {
        "schema_version": ACTION_CAUSALITY_SCHEMA,
        "action_source": "actual_executed_action",
        "safety_action_source": "actor_proposal",
        "counterfactual_pairing": "proposal-executed-pair/v1",
        "counterfactual_transform": "deterministic-global-derangement/v1",
        "window_count": report.sample_count,
        "report": report.to_dict(),
        "assessment": assess_action_causality(report, criteria),
    }


def _evaluate_batch_report(
    trainer: FoundationWorldModelTrainer,
    batch: FoundationTrainingBatch,
    *,
    shuffle_seed: int,
) -> CounterfactualCausalityReport:
    observations = batch.observation_count
    was_training = trainer.visual_student.training
    trainer.visual_student.eval()
    try:
        with torch.inference_mode():
            visual = trainer.visual_student(batch.student_inputs).pooled_state.reshape(
                batch.sequence_batch_size,
                observations,
                trainer.world_model.config.visual_dimension,
            )
    finally:
        trainer.visual_student.train(was_training)
    report = evaluate_action_causality(
        trainer.world_model,
        visual,
        batch.language_features,
        batch.proprioception,
        batch.actor_proposals,
        batch.executed_actions,
        batch.rewards,
        batch.continues,
        batch.safety_interventions,
        shuffle_seed=shuffle_seed,
    )
    return report


def evaluate_foundation_action_causality_audit(
    trainer: FoundationWorldModelTrainer,
    batches_by_partition: Mapping[str, Iterable[FoundationTrainingBatch]],
    criteria: ActionCausalityCriteria,
    *,
    shuffle_seed: int,
) -> dict[str, object]:
    """Require aggregate and every generic data partition to pass."""
    if shuffle_seed < 0 or not batches_by_partition:
        raise ValueError("foundation action causality audit configuration is invalid")
    partition_values: dict[str, object] = {}
    all_reports: list[CounterfactualCausalityReport] = []
    batch_index = 0
    for partition in sorted(batches_by_partition):
        batches = batches_by_partition[partition]
        if not partition:
            raise ValueError("action causality audit partitions must be non-empty")
        reports = []
        partition_batch_count = 0
        for batch in batches:
            reports.append(
                _evaluate_batch_report(
                    trainer,
                    batch,
                    shuffle_seed=shuffle_seed + batch_index,
                )
            )
            batch_index += 1
            partition_batch_count += 1
        if not reports:
            raise ValueError("action causality audit partitions must be non-empty")
        aggregate = aggregate_action_causality_reports(tuple(reports))
        all_reports.extend(reports)
        partition_values[partition] = {
            "window_count": aggregate.sample_count,
            "batch_count": partition_batch_count,
            "report": aggregate.to_dict(),
            "assessment": assess_action_causality(aggregate, criteria),
        }
    aggregate = aggregate_action_causality_reports(tuple(all_reports))
    aggregate_assessment = assess_action_causality(aggregate, criteria)
    all_partitions_passed = all(
        value["assessment"]["passed"] is True
        for value in partition_values.values()
    )
    assessment = {
        **aggregate_assessment,
        "aggregate_passed": aggregate_assessment["passed"],
        "all_partitions_passed": all_partitions_passed,
        "partition_count": len(partition_values),
        "passed": aggregate_assessment["passed"] and all_partitions_passed,
    }
    return {
        "schema_version": ACTION_CAUSALITY_SCHEMA,
        "action_source": "actual_executed_action",
        "safety_action_source": "actor_proposal",
        "counterfactual_pairing": "proposal-executed-pair/v1",
        "counterfactual_transform": "deterministic-global-derangement/v1",
        "partition_key": "task_id",
        "window_count": aggregate.sample_count,
        "batch_count": batch_index,
        "partitions": partition_values,
        "report": aggregate.to_dict(),
        "assessment": assessment,
    }


def publish_action_causality_report(
    path: Path,
    diagnostic: Mapping[str, object],
    *,
    source_commit: str,
    update_count: int,
    training_data_manifest_sha256: str,
    audit_data_manifest_sha256: str,
) -> Path:
    if diagnostic.get("schema_version") != ACTION_CAUSALITY_SCHEMA:
        raise ValueError("action causality diagnostic schema differs")
    digests = (training_data_manifest_sha256, audit_data_manifest_sha256)
    if not source_commit or any(len(value) != 64 for value in digests) or update_count <= 0:
        raise ValueError("action causality provenance is incomplete")
    value = {
        **dict(diagnostic),
        "source_commit": source_commit,
        "update_count": update_count,
        "training_data_manifest_sha256": training_data_manifest_sha256,
        "audit_data_manifest_sha256": audit_data_manifest_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return path / "report.json"
