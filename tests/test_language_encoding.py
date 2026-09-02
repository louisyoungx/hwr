from __future__ import annotations

import numpy as np

from hwr.core.embodied import NaturalLanguageInstruction
from hwr.perception import FrozenNgramLanguageConfig, FrozenNgramLanguageEncoder


def test_raw_language_encoding_is_frozen_deterministic_and_not_tokenized_plan() -> None:
    encoder = FrozenNgramLanguageEncoder(FrozenNgramLanguageConfig(dimension=32))
    instruction = NaturalLanguageInstruction("  Hold the tray with both hands and place it steadily on the sideboard. ")

    first = encoder.encode(instruction)
    second = encoder.encode(instruction)
    changed = encoder.encode(NaturalLanguageInstruction("Pull the drawer with the left hand and place the cleaner with the right hand."))

    assert first == second
    assert len(first.values) == 32
    assert np.linalg.norm(first.values) == np.float64(1.0)
    assert first.weights_sha256 == encoder.weights_sha256
    assert first.values != changed.values
    assert not hasattr(first, "skill_plan")
    assert not hasattr(first, "object_token")
