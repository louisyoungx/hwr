"""Local frozen raw-language encoding without symbolic instruction parsing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from hwr.core.embodied import FrozenLanguageEmbedding, NaturalLanguageInstruction


@dataclass(frozen=True)
class FrozenNgramLanguageConfig:
    dimension: int = 64
    minimum_ngram: int = 1
    maximum_ngram: int = 3
    salt: str = "hwr-language-v1"

    def __post_init__(self) -> None:
        if self.dimension <= 0 or not 1 <= self.minimum_ngram <= self.maximum_ngram:
            raise ValueError("frozen language encoder dimensions are invalid")
        if not self.salt:
            raise ValueError("frozen language encoder salt is required")


class FrozenNgramLanguageEncoder:
    """Hash raw Unicode n-grams into a fixed, versioned continuous embedding."""

    def __init__(self, config: FrozenNgramLanguageConfig) -> None:
        self.config = config
        payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        self.weights_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        self.encoder_id = "hwr-frozen-unicode-ngram/v1"

    def encode(self, instruction: NaturalLanguageInstruction) -> FrozenLanguageEmbedding:
        text = f"{instruction.locale}\0{instruction.text}"
        values = np.zeros(self.config.dimension, dtype=np.float64)
        for size in range(self.config.minimum_ngram, self.config.maximum_ngram + 1):
            for start in range(max(0, len(text) - size + 1)):
                ngram = text[start : start + size]
                digest = hashlib.blake2b(
                    ngram.encode(),
                    digest_size=16,
                    key=self.config.salt.encode()[:64],
                ).digest()
                index = int.from_bytes(digest[:8], "little") % self.config.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                values[index] += sign / math.sqrt(size)
        norm = float(np.linalg.norm(values))
        if norm > 0:
            values /= norm
        return FrozenLanguageEmbedding(
            self.encoder_id,
            self.weights_sha256,
            tuple(float(value) for value in values),
        )
