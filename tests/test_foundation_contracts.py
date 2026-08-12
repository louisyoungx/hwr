from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hwr.perception import (
    DenseVisualFeatures,
    FoundationModelLock,
    SemanticLanguageFeatures,
    WeightArtifact,
)


DIGEST = "1" * 64


def test_foundation_lock_is_immutable_and_verifies_local_file(tmp_path) -> None:
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"frozen-test-weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    lock = FoundationModelLock(
        model_id="fixture/dense-vision",
        revision="0123456789abcdef",
        role="dense_vision",
        license_id="Apache-2.0",
        output_dimension=8,
        artifacts=(WeightArtifact("weights.safetensors", digest, weight.stat().st_size),),
    )

    assert len(lock.lock_sha256) == 64
    assert lock.verify_local_files(tmp_path) == ()
    weight.write_bytes(b"corrupted-test-weight")
    assert lock.verify_local_files(tmp_path) == ("size:weights.safetensors",)


def test_foundation_lock_rejects_moving_revision_and_escaping_path() -> None:
    artifact = WeightArtifact("weights.safetensors", DIGEST, 12)
    with pytest.raises(ValueError, match="immutable"):
        FoundationModelLock("fixture/model", "main", "language", "MIT", 4, (artifact,))
    with pytest.raises(ValueError, match="contained"):
        WeightArtifact("../weights", DIGEST, 12)


def test_foundation_outputs_preserve_dense_axes_and_have_no_symbolic_fields() -> None:
    visual = DenseVisualFeatures(
        np.ones((3, 4, 5, 8), dtype=np.float32),
        np.ones((3, 4, 5), dtype=np.bool_),
        DIGEST,
        "2" * 64,
    )
    language = SemanticLanguageFeatures(
        np.asarray([0.2, -0.4, 0.8], dtype=np.float32), DIGEST, "3" * 64
    )

    assert visual.values.shape == (3, 4, 5, 8)
    assert not visual.values.flags.writeable
    assert not language.values.flags.writeable
    forbidden = {"action", "waypoint", "skill", "stage", "object_id", "target_id"}
    assert forbidden.isdisjoint(visual.__dataclass_fields__)
    assert forbidden.isdisjoint(language.__dataclass_fields__)


def test_dense_features_reject_global_pooling_and_invalid_mask() -> None:
    with pytest.raises(ValueError, match="camera, row, column, channel"):
        DenseVisualFeatures(np.ones((3, 8)), np.ones((3,), bool), DIGEST, DIGEST)
    with pytest.raises(ValueError, match="validity"):
        DenseVisualFeatures(np.ones((3, 2, 2, 8)), np.ones((3, 2), bool), DIGEST, DIGEST)
