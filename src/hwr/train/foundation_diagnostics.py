"""Task-independent diagnostics published by formal foundation training."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import torch

from hwr.train.foundation_batch import FoundationTrainingBatch
from hwr.train.foundation_trainer import FoundationWorldModelTrainer
from hwr.world_model import (
    ActionCausalityCriteria,
    assess_action_causality,
    evaluate_action_causality,
)


ACTION_CAUSALITY_SCHEMA = "hwr.foundation-action-causality/v1"


def evaluate_foundation_action_causality(
    trainer: FoundationWorldModelTrainer,
    batch: FoundationTrainingBatch,
    criteria: ActionCausalityCriteria,
) -> dict[str, object]:
    """Evaluate every sequence in a generic batch using executed actions."""
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
        batch.executed_actions,
        batch.rewards,
        batch.continues,
        batch.safety,
    )
    return {
        "schema_version": ACTION_CAUSALITY_SCHEMA,
        "action_source": "actual_executed_action",
        "report": report.to_dict(),
        "assessment": assess_action_causality(report, criteria),
    }


def publish_action_causality_report(
    path: Path,
    diagnostic: Mapping[str, object],
    *,
    source_commit: str,
    update_count: int,
    data_manifest_sha256: str,
) -> Path:
    if diagnostic.get("schema_version") != ACTION_CAUSALITY_SCHEMA:
        raise ValueError("action causality diagnostic schema differs")
    if not source_commit or len(data_manifest_sha256) != 64 or update_count <= 0:
        raise ValueError("action causality provenance is incomplete")
    value = {
        **dict(diagnostic),
        "source_commit": source_commit,
        "update_count": update_count,
        "data_manifest_sha256": data_manifest_sha256,
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
