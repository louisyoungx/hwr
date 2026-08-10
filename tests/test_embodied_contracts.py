from __future__ import annotations

import pytest

from hwr.core.embodied import (
    DUAL_ARM_ACTION_DIM,
    ActionChunk,
    DualArmAction,
    FrozenLanguageEmbedding,
    NaturalLanguageInstruction,
)


def _action(value: float = 0.1) -> DualArmAction:
    return DualArmAction(value, -value, (value,) * 6, (-value,) * 6, 0.0, 1.0)


def test_natural_instruction_is_normalized_without_symbolic_parsing() -> None:
    instruction = NaturalLanguageInstruction("  把杯子\n放到水槽旁边  ")

    assert instruction.text == "把杯子 放到水槽旁边"
    assert not hasattr(instruction, "object_token")
    assert not hasattr(instruction, "skill_plan")


def test_frozen_language_embedding_requires_weight_identity() -> None:
    embedding = FrozenLanguageEmbedding("local-text/v1", "a" * 64, (0.1, -0.2))

    assert embedding.weights_sha256 == "a" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        FrozenLanguageEmbedding("local-text/v1", "unknown", (0.1,))


def test_dual_arm_action_chunk_round_trips_vectors() -> None:
    action = _action()
    chunk = ActionChunk((action, _action(0.2)), valid_steps=1)

    assert len(action.vector()) == DUAL_ARM_ACTION_DIM
    assert DualArmAction.from_vector(action.vector()) == action
    assert chunk.vectors()[0] == action.vector()


def test_dual_arm_action_rejects_incomplete_arm() -> None:
    with pytest.raises(ValueError, match="six commands"):
        DualArmAction(0.0, 0.0, (0.0,) * 5, (0.0,) * 6, 0.0, 0.0)
