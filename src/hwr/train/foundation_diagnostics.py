"""Task-independent diagnostics published by formal foundation training."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import torch
import numpy as np

from hwr.train.foundation_batch import FoundationTrainingBatch
from hwr.train.foundation_registry import ACTION_CAUSALITY_SCHEMA
from hwr.train.foundation_trainer import FoundationWorldModelTrainer
from hwr.world_model import (
    ActionCausalityCriteria,
    CounterfactualCausalityReport,
    aggregate_action_causality_reports,
    assess_action_causality,
    evaluate_action_causality,
    evaluate_one_step_action_utilization,
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
    reports, _ = _evaluate_batch_reports(
        trainer, batch, shuffle_seeds=(shuffle_seed,)
    )
    return reports[0]


def _evaluate_batch_reports(
    trainer: FoundationWorldModelTrainer,
    batch: FoundationTrainingBatch,
    *,
    shuffle_seeds: tuple[int, ...],
) -> tuple[
    tuple[CounterfactualCausalityReport, ...],
    tuple[CounterfactualCausalityReport, ...],
]:
    if not shuffle_seeds or min(shuffle_seeds) < 0:
        raise ValueError("foundation action shuffle seeds are invalid")
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
    open_loop = tuple(
        evaluate_action_causality(
            trainer.world_model,
            visual,
            batch.language_features,
            batch.proprioception,
            batch.actor_proposals,
            batch.executed_actions,
            batch.rewards,
            batch.continues,
            batch.safety_interventions,
            shuffle_seed=seed,
        )
        for seed in shuffle_seeds
    )
    one_step = tuple(
        evaluate_one_step_action_utilization(
            trainer.world_model,
            visual,
            batch.language_features,
            batch.proprioception,
            batch.actor_proposals,
            batch.executed_actions,
            shuffle_seed=seed,
        )
        for seed in shuffle_seeds
    )
    return open_loop, one_step


def evaluate_foundation_action_causality_audit(
    trainer: FoundationWorldModelTrainer,
    batches_by_partition: Mapping[str, Iterable[FoundationTrainingBatch]],
    criteria: ActionCausalityCriteria,
    *,
    shuffle_seed: int,
    shuffle_repeats: int = 1,
) -> dict[str, object]:
    """Require aggregate and every generic data partition to pass."""
    if shuffle_seed < 0 or shuffle_repeats <= 0 or not batches_by_partition:
        raise ValueError("foundation action causality audit configuration is invalid")
    materialized = {
        partition: tuple(batches)
        for partition, batches in batches_by_partition.items()
    }
    partition_values: dict[str, object] = {}
    all_reports: list[CounterfactualCausalityReport] = []
    all_one_step_reports: list[CounterfactualCausalityReport] = []
    batch_index = 0
    for partition in sorted(batches_by_partition):
        batches = materialized[partition]
        if not partition:
            raise ValueError("action causality audit partitions must be non-empty")
        reports_by_shuffle = [[] for _ in range(shuffle_repeats)]
        one_step_by_shuffle = [[] for _ in range(shuffle_repeats)]
        partition_batch_count = 0
        for batch in batches:
            seeds = tuple(
                shuffle_seed + repeat * 1_000_003 + batch_index
                for repeat in range(shuffle_repeats)
            )
            reports, one_steps = _evaluate_batch_reports(
                trainer,
                batch,
                shuffle_seeds=seeds,
            )
            for repeat, report in enumerate(reports):
                reports_by_shuffle[repeat].append(report)
                one_step_by_shuffle[repeat].append(one_steps[repeat])
            batch_index += 1
            partition_batch_count += 1
        if not reports_by_shuffle[0]:
            raise ValueError("action causality audit partitions must be non-empty")
        shuffled_reports = tuple(
            aggregate_action_causality_reports(tuple(values))
            for values in reports_by_shuffle
        )
        shuffled_one_step = tuple(
            aggregate_action_causality_reports(tuple(values))
            for values in one_step_by_shuffle
        )
        aggregate = aggregate_action_causality_reports(shuffled_reports)
        one_step_aggregate = aggregate_action_causality_reports(shuffled_one_step)
        all_reports.extend(shuffled_reports)
        all_one_step_reports.extend(shuffled_one_step)
        partition_values[partition] = {
            "window_count": aggregate.sample_count // shuffle_repeats,
            "batch_count": partition_batch_count,
            "report": aggregate.to_dict(),
            "assessment": assess_action_causality(aggregate, criteria),
            "shuffle_statistics": _shuffle_statistics(shuffled_reports, criteria),
            "one_step_action_utilization": {
                "report": one_step_aggregate.to_dict(),
                "assessment": assess_action_causality(
                    one_step_aggregate, criteria
                ),
                "shuffle_statistics": _shuffle_statistics(
                    shuffled_one_step, criteria
                ),
            },
        }
    aggregate = aggregate_action_causality_reports(tuple(all_reports))
    one_step_aggregate = aggregate_action_causality_reports(
        tuple(all_one_step_reports)
    )
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
        "window_count": aggregate.sample_count // shuffle_repeats,
        "batch_count": batch_index,
        "shuffle_repeats": shuffle_repeats,
        "partitions": partition_values,
        "report": aggregate.to_dict(),
        "assessment": assessment,
        "shuffle_statistics": _shuffle_statistics(
            tuple(all_reports), criteria
        ),
        "one_step_action_utilization": {
            "conditioning": "teacher-forced-posterior-state/v1",
            "physical_components": ["visual_latent", "proprioception"],
            "report": one_step_aggregate.to_dict(),
            "assessment": assess_action_causality(one_step_aggregate, criteria),
            "shuffle_statistics": _shuffle_statistics(
                tuple(all_one_step_reports), criteria
            ),
        },
    }


def _shuffle_statistics(
    reports: tuple[CounterfactualCausalityReport, ...],
    criteria: ActionCausalityCriteria,
) -> dict[str, object]:
    ratios = np.asarray(
        [
            float(assess_action_causality(report, criteria)["shuffled_to_true_ratio"])
            for report in reports
        ],
        np.float64,
    )
    p05 = float(np.quantile(ratios, 0.05))
    report_passes = [
        assess_action_causality(report, criteria)["passed"] is True
        for report in reports
    ]
    lower_bound_passed = p05 >= criteria.minimum_shuffled_to_true_ratio
    return {
        "count": len(reports),
        "reports": [report.to_dict() for report in reports],
        "shuffled_to_true_ratios": ratios.tolist(),
        "ratio_p05": p05,
        "ratio_median": float(np.median(ratios)),
        "ratio_p95": float(np.quantile(ratios, 0.95)),
        "lower_bound_passed": lower_bound_passed,
        "passed_fraction": sum(report_passes) / len(report_passes),
        "all_reports_passed": all(report_passes),
        "robust_passed": lower_bound_passed and all(report_passes),
    }


def foundation_action_causality_qualified(
    diagnostic: Mapping[str, object]
) -> bool:
    assessment = diagnostic.get("assessment")
    statistics = diagnostic.get("shuffle_statistics")
    partitions = diagnostic.get("partitions")
    if (
        not isinstance(assessment, Mapping)
        or assessment.get("passed") is not True
        or not isinstance(statistics, Mapping)
        or statistics.get("robust_passed") is not True
        or not isinstance(partitions, Mapping)
    ):
        return False
    return all(
        isinstance(value, Mapping)
        and value.get("assessment", {}).get("passed") is True
        and value.get("shuffle_statistics", {}).get("robust_passed") is True
        for value in partitions.values()
    )


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
