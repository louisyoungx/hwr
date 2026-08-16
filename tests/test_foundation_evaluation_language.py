from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.data.foundation_features import (
    LANGUAGE_PREPROCESS_SHA256,
    FoundationFeatureIndex,
)
from hwr.eval.foundation_language import (
    evaluation_language_artifacts,
    materialize_evaluation_language,
)
from hwr.perception.foundation import (
    FoundationModelLock,
    SemanticLanguageFeatures,
    WeightArtifact,
    language_source_sha256,
)
from hwr.perception.language_cache import StaticLanguageFeatureResolver
from hwr.policy.latent_actor import LatentActor, LatentActorConfig
from hwr.scenarios.formal3d import load_formal_3d_tasks
from hwr.world_model import DeployableWorldModelStateFilter, WorldModelConfig


ROOT = Path(__file__).resolve().parents[1]


class _LanguageProvider:
    def __init__(self, marker: str = "c") -> None:
        self._lock = FoundationModelLock(
            "fixture/qwen3",
            marker * 40,
            "language",
            "Apache-2.0",
            6,
            "last-token-pool-l2/v1",
            (WeightArtifact("fixture.bin", marker * 64, 1),),
        )

    @property
    def model_lock(self):
        return self._lock

    def encode_language(self, text, locale):
        source = language_source_sha256(text, locale)
        values = np.frombuffer(bytes.fromhex(source[:12]), dtype=np.uint8)
        return SemanticLanguageFeatures(
            values.astype(np.float32) + 1.0,
            self._lock.lock_sha256,
            source,
        )


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _language_run(tmp_path, provider):
    tasks = load_formal_3d_tasks(ROOT / "configs/tasks/formal_3d_v1.json")
    run = tmp_path / "run"
    shards = [
        {
            "task_id": task_id,
            "instruction": task.training_instructions[0],
            "locale": "zh-CN",
        }
        for task_id, task in sorted(tasks.items())
    ]
    replay_path = run / "replay/autonomous/manifest.json"
    _write_json(replay_path, {"shards": shards})
    index = FoundationFeatureIndex(
        "language",
        "language",
        _digest(replay_path),
        provider.model_lock.lock_sha256,
        LANGUAGE_PREPROCESS_SHA256,
        provider.model_lock.output_dimension,
        len(shards),
    )
    _write_json(run / "features/language.json", index.to_dict())
    cache = FoundationFeatureCache(run / "feature-cache")
    values = {}
    for shard in shards:
        text, locale = shard["instruction"], shard["locale"]
        feature = provider.encode_language(text, locale)
        key = FoundationCacheKey(
            "language",
            feature.source_sha256,
            index.encoder_lock_sha256,
            index.preprocess_sha256,
        )
        cache.store_language(key, feature)
        values[(locale, text)] = feature.values.copy()
    return run, tasks, values


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _fixture_action(resolver, text):
    torch.manual_seed(7)
    config = WorldModelConfig(
        visual_dimension=8,
        language_dimension=6,
        proprioception_dimension=5,
        observation_embedding_dimension=12,
        deterministic_dimension=10,
        stochastic_variables=3,
        stochastic_classes=4,
        hidden_dimension=16,
        prior_ensemble=2,
        reward_bins=11,
        formal=False,
    )
    world = DeployableWorldModelStateFilter(config).eval()
    actor = LatentActor(
        LatentActorConfig(
            config.feature_dimension,
            hidden_dimension=16,
            hidden_layers=1,
            formal=False,
        )
    ).eval()
    language = torch.from_numpy(
        resolver.resolve(text, "zh-CN").values.copy()
    )[None]
    with torch.inference_mode():
        state = world.posterior_step(
            torch.ones(1, 8),
            language,
            torch.ones(1, 5),
            previous=None,
            executed_action=None,
            sample=False,
        )
        return actor.deterministic(world.features(state)).numpy()


def test_evaluation_language_materializes_nine_isolated_embeddings(tmp_path) -> None:
    provider = _LanguageProvider()
    run, tasks, training = _language_run(tmp_path, provider)
    before = _snapshot(run)
    output = tmp_path / "evaluation"
    output.mkdir()

    bundle = materialize_evaluation_language(
        run, output, tasks, provider
    )

    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["instruction_count"] == 9
    assert manifest["task_count"] == 3
    assert manifest["encoder"]["lock_sha256"] == provider.model_lock.lock_sha256
    assert manifest["preprocess_sha256"] == LANGUAGE_PREPROCESS_SHA256
    assert len(bundle.embedding_paths) == 9
    assert len(evaluation_language_artifacts(output)) == 10
    assert _snapshot(run) == before
    for task in tasks.values():
        for text in task.evaluation_instructions:
            assert bundle.resolver.resolve(text, "zh-CN").values.shape == (6,)
    for (locale, text), expected in training.items():
        np.testing.assert_array_equal(
            bundle.resolver.resolve(text, locale).values, expected
        )


def test_merged_resolver_preserves_seen_deterministic_action(tmp_path) -> None:
    provider = _LanguageProvider()
    run, tasks, training = _language_run(tmp_path, provider)
    output = tmp_path / "evaluation"
    output.mkdir()
    old = StaticLanguageFeatureResolver(
        training,
        encoder_lock_sha256=provider.model_lock.lock_sha256,
        output_dimension=6,
    )
    merged = materialize_evaluation_language(
        run, output, tasks, provider
    ).resolver
    text = next(iter(training))[1]

    np.testing.assert_array_equal(
        _fixture_action(old, text),
        _fixture_action(merged, text),
    )


def test_evaluation_language_fails_on_encoder_text_and_isolation_drift(
    tmp_path,
) -> None:
    provider = _LanguageProvider()
    run, tasks, _ = _language_run(tmp_path, provider)
    output = tmp_path / "wrong-lock"
    output.mkdir()
    with pytest.raises(ValueError, match="encoder differs"):
        materialize_evaluation_language(
            run, output, tasks, _LanguageProvider("d")
        )

    missing = dict(tasks)
    first_id = sorted(missing)[0]
    missing[first_id] = SimpleNamespace(
        training_instructions=tasks[first_id].training_instructions,
        evaluation_instructions=("",) * 3,
    )
    output = tmp_path / "missing-text"
    output.mkdir()
    with pytest.raises(ValueError, match="evaluation instructions differ"):
        materialize_evaluation_language(run, output, missing, provider)

    with pytest.raises(ValueError, match="isolated"):
        materialize_evaluation_language(
            run, run / "evaluation", tasks, provider
        )


def test_evaluation_language_artifact_hash_is_mandatory(tmp_path) -> None:
    provider = _LanguageProvider()
    run, tasks, _ = _language_run(tmp_path, provider)
    output = tmp_path / "evaluation"
    output.mkdir()
    bundle = materialize_evaluation_language(
        run, output, tasks, provider
    )
    bundle.embedding_paths[0].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="embedding hash differs"):
        evaluation_language_artifacts(output)
