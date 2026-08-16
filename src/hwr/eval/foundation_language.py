"""Evaluation-only language materialization for formal foundation rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from hwr.data.foundation_cache import FoundationCacheKey, FoundationFeatureCache
from hwr.data.foundation_features import (
    LANGUAGE_PREPROCESS_SHA256,
    FoundationFeatureIndex,
    load_feature_index,
)
from hwr.perception.foundation import (
    FrozenLanguageFeatureProvider,
    language_source_sha256,
)
from hwr.perception.language_cache import StaticLanguageFeatureResolver
from hwr.scenarios.formal3d import Formal3DTaskSpec


EVALUATION_LANGUAGE_SCHEMA = "hwr.foundation-evaluation-language/v1"
EVALUATION_INSTRUCTIONS_PER_TASK = 3
FORMAL_LANGUAGE_LOCALE = "zh-CN"


@dataclass(frozen=True)
class EvaluationLanguageBundle:
    resolver: StaticLanguageFeatureResolver
    manifest_path: Path
    embedding_paths: tuple[Path, ...]


def materialize_evaluation_language(
    run_path: Path,
    output_path: Path,
    tasks: Mapping[str, Formal3DTaskSpec],
    provider: FrozenLanguageFeatureProvider,
) -> EvaluationLanguageBundle:
    """Build one isolated evaluation index before any physical Episode starts."""
    run_path = run_path.resolve()
    output_path = output_path.resolve()
    _require_isolated_output(run_path, output_path)
    index_path = run_path / "features/language.json"
    replay_path = run_path / "replay/autonomous/manifest.json"
    index = load_feature_index(index_path)
    replay = _read_json(replay_path)
    _require_encoder_identity(index, replay_path, provider)
    instructions = _evaluation_instructions(tasks)
    training_values, training_entries = _training_language_values(
        run_path, replay, tasks, index
    )
    training_evidence_before = _training_evidence(
        run_path, index_path, replay_path, training_entries
    )
    language_root = output_path / "evaluation-language"
    if language_root.exists():
        raise FileExistsError(language_root)
    cache = FoundationFeatureCache(language_root / "cache")
    evaluation_values, evaluation_entries = _materialize_instructions(
        cache, instructions, provider, index, language_root
    )
    resolver = _merged_resolver(training_values, evaluation_values, index)
    _require_training_equivalence(training_values, resolver, index)
    training_evidence_after = _training_evidence(
        run_path, index_path, replay_path, training_entries
    )
    if training_evidence_after != training_evidence_before:
        raise RuntimeError("evaluation language preparation modified training artifacts")
    manifest = _language_manifest(
        run_path,
        output_path,
        index,
        provider,
        training_values,
        training_evidence_before,
        evaluation_entries,
    )
    manifest_path = language_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    artifacts = evaluation_language_artifacts(output_path)
    return EvaluationLanguageBundle(
        resolver,
        manifest_path,
        tuple(
            path
            for name, path in artifacts.items()
            if name != "evaluation-language/manifest.json"
        ),
    )


def evaluation_language_artifacts(output_path: Path) -> dict[str, Path]:
    """Verify and return every language artifact bound by the evaluation run."""
    output_path = output_path.resolve()
    language_root = output_path / "evaluation-language"
    manifest_path = language_root / "manifest.json"
    manifest = _read_json(manifest_path)
    entries = manifest.get("instructions")
    isolation = manifest.get("isolation")
    encoder = manifest.get("encoder")
    if (
        manifest.get("schema_version") != EVALUATION_LANGUAGE_SCHEMA
        or manifest.get("instruction_count") != 9
        or manifest.get("task_count") != 3
        or not isinstance(entries, list)
        or len(entries) != 9
        or not isinstance(encoder, Mapping)
        or not _is_sha256(encoder.get("lock_sha256"))
        or int(encoder.get("output_dimension", 0)) <= 0
        or not _is_sha256(manifest.get("preprocess_sha256"))
        or not isinstance(isolation, Mapping)
        or isolation.get("evaluation_only") is not True
        or isolation.get("training_artifacts_unchanged") is not True
    ):
        raise ValueError("evaluation language manifest is incomplete")
    artifacts = {"evaluation-language/manifest.json": manifest_path}
    sources: set[str] = set()
    cache_keys: set[str] = set()
    task_counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("evaluation language instruction entry is invalid")
        _require_instruction_entry(entry)
        if (
            entry["encoder_lock_sha256"] != encoder["lock_sha256"]
            or entry["preprocess_sha256"] != manifest["preprocess_sha256"]
            or int(entry["output_dimension"]) != int(encoder["output_dimension"])
        ):
            raise ValueError("evaluation language instruction identity differs")
        relative = Path(str(entry["path"]))
        path = _contained_member(language_root, relative)
        expected_relative = (
            Path("cache/language")
            / str(entry["cache_key"])[:2]
            / f"{entry['cache_key']}.npz"
        )
        if relative != expected_relative:
            raise ValueError("evaluation language embedding escaped its cache")
        if (
            not path.is_file()
            or _sha256(path) != entry["file_sha256"]
            or path.stat().st_size != int(entry["bytes"])
        ):
            raise ValueError("evaluation language embedding hash differs")
        source = str(entry["source_sha256"])
        cache_key = str(entry["cache_key"])
        if source in sources or cache_key in cache_keys:
            raise ValueError("evaluation language embeddings are not unique")
        sources.add(source)
        cache_keys.add(cache_key)
        task_id = str(entry["task_id"])
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
        artifacts[f"evaluation-language/{relative.as_posix()}"] = path
    if len(task_counts) != 3 or set(task_counts.values()) != {3}:
        raise ValueError("evaluation language task coverage differs")
    _require_training_evidence(manifest)
    return artifacts


def _require_isolated_output(run_path: Path, output_path: Path) -> None:
    if output_path.is_relative_to(run_path) or run_path.is_relative_to(output_path):
        raise ValueError("evaluation output must be isolated from the training run")


def _require_encoder_identity(
    index: FoundationFeatureIndex,
    replay_path: Path,
    provider: FrozenLanguageFeatureProvider,
) -> None:
    lock = provider.model_lock
    if index.kind != "language" or index.role != "language":
        raise ValueError("training language index role differs")
    if index.dataset_sha256 != _sha256(replay_path):
        raise ValueError("training language index dataset hash differs")
    if (
        index.encoder_lock_sha256 != lock.lock_sha256
        or index.preprocess_sha256 != LANGUAGE_PREPROCESS_SHA256
        or index.output_dimension != lock.output_dimension
    ):
        raise ValueError("evaluation language encoder differs from training index")


def _evaluation_instructions(
    tasks: Mapping[str, Formal3DTaskSpec],
) -> tuple[tuple[str, str, str], ...]:
    if len(tasks) != 3:
        raise ValueError("formal evaluation requires exactly three tasks")
    entries: list[tuple[str, str, str]] = []
    training_sources: set[str] = set()
    for task_id in sorted(tasks):
        task = tasks[task_id]
        if not task.training_instructions:
            raise ValueError(f"training instructions are missing for {task_id}")
        if (
            len(task.evaluation_instructions) != EVALUATION_INSTRUCTIONS_PER_TASK
            or any(not " ".join(text.split()) for text in task.evaluation_instructions)
        ):
            raise ValueError(f"evaluation instructions differ for {task_id}")
        training_sources.update(
            language_source_sha256(text, FORMAL_LANGUAGE_LOCALE)
            for text in task.training_instructions
        )
        entries.extend(
            (task_id, FORMAL_LANGUAGE_LOCALE, text)
            for text in task.evaluation_instructions
        )
    sources = [
        language_source_sha256(text, locale) for _, locale, text in entries
    ]
    if len(set(sources)) != 9 or set(sources) & training_sources:
        raise ValueError("evaluation instructions must be unique and unseen")
    return tuple(entries)


def _training_language_values(
    run_path: Path,
    replay: Mapping[str, object],
    tasks: Mapping[str, Formal3DTaskSpec],
    index: FoundationFeatureIndex,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    tuple[tuple[FoundationCacheKey, Path], ...],
]:
    shards = replay.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("training Replay has no language instructions")
    cache = FoundationFeatureCache(run_path / "feature-cache")
    values: dict[tuple[str, str], np.ndarray] = {}
    entries: dict[str, tuple[FoundationCacheKey, Path]] = {}
    seen_tasks: set[str] = set()
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("training Replay shard is invalid")
        task_id = str(shard.get("task_id", ""))
        text = str(shard.get("instruction", ""))
        locale = str(shard.get("locale", ""))
        task = tasks.get(task_id)
        if (
            task is None
            or locale != FORMAL_LANGUAGE_LOCALE
            or text not in task.training_instructions
        ):
            raise ValueError("training Replay language split differs")
        source = language_source_sha256(text, locale)
        key = FoundationCacheKey(
            "language",
            source,
            index.encoder_lock_sha256,
            index.preprocess_sha256,
        )
        feature = cache.load_language(key)
        if feature.values.shape != (index.output_dimension,):
            raise ValueError("training language embedding dimension differs")
        previous = values.get((locale, text))
        if previous is not None and not np.array_equal(previous, feature.values):
            raise ValueError("training language embedding changed within Replay")
        values[(locale, text)] = feature.values.copy()
        entries[key.digest] = (key, cache.path_for(key))
        seen_tasks.add(task_id)
    if seen_tasks != set(tasks) or len(values) != index.entry_count:
        raise ValueError("training language index is missing instructions")
    return values, tuple(entries[key] for key in sorted(entries))


def _materialize_instructions(
    cache: FoundationFeatureCache,
    instructions: Iterable[tuple[str, str, str]],
    provider: FrozenLanguageFeatureProvider,
    index: FoundationFeatureIndex,
    language_root: Path,
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict[str, object]]]:
    values: dict[tuple[str, str], np.ndarray] = {}
    entries: list[dict[str, object]] = []
    for task_id, locale, text in instructions:
        normalized = " ".join(text.split())
        source = language_source_sha256(normalized, locale)
        key = FoundationCacheKey(
            "language",
            source,
            index.encoder_lock_sha256,
            index.preprocess_sha256,
        )
        if cache.contains(key):
            raise ValueError("evaluation language cache was not empty")
        encoded = provider.encode_language(normalized, locale)
        if (
            encoded.encoder_lock_sha256 != index.encoder_lock_sha256
            or encoded.source_sha256 != source
            or encoded.values.shape != (index.output_dimension,)
        ):
            raise ValueError("evaluation language provider output differs")
        path = cache.store_language(key, encoded)
        stored = cache.load_language(key)
        if not np.array_equal(stored.values, encoded.values):
            raise ValueError("evaluation language cache changed an embedding")
        relative = path.relative_to(language_root)
        values[(locale, normalized)] = stored.values.copy()
        entries.append(
            {
                "task_id": task_id,
                "locale": locale,
                "text": normalized,
                "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                "source_sha256": source,
                "cache_key": key.digest,
                "encoder_lock_sha256": index.encoder_lock_sha256,
                "preprocess_sha256": index.preprocess_sha256,
                "output_dimension": index.output_dimension,
                "path": relative.as_posix(),
                "file_sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return values, entries


def _merged_resolver(
    training: Mapping[tuple[str, str], np.ndarray],
    evaluation: Mapping[tuple[str, str], np.ndarray],
    index: FoundationFeatureIndex,
) -> StaticLanguageFeatureResolver:
    overlap = set(training) & set(evaluation)
    if overlap:
        raise ValueError("evaluation language entered the training instruction map")
    return StaticLanguageFeatureResolver(
        {**training, **evaluation},
        encoder_lock_sha256=index.encoder_lock_sha256,
        output_dimension=index.output_dimension,
    )


def _require_training_equivalence(
    training: Mapping[tuple[str, str], np.ndarray],
    resolver: StaticLanguageFeatureResolver,
    index: FoundationFeatureIndex,
) -> None:
    old = StaticLanguageFeatureResolver(
        training,
        encoder_lock_sha256=index.encoder_lock_sha256,
        output_dimension=index.output_dimension,
    )
    for (locale, text), expected in training.items():
        if (
            not np.array_equal(old.resolve(text, locale).values, expected)
            or not np.array_equal(resolver.resolve(text, locale).values, expected)
        ):
            raise RuntimeError("merged resolver changed a training embedding")


def _training_evidence(
    run_path: Path,
    index_path: Path,
    replay_path: Path,
    entries: Iterable[tuple[FoundationCacheKey, Path]],
) -> dict[str, object]:
    return {
        "run_path": str(run_path),
        "language_index": _identity(run_path, index_path),
        "replay_manifest": _identity(run_path, replay_path),
        "embedding_files": [
            {
                **_identity(run_path, path),
                "cache_key": key.digest,
            }
            for key, path in entries
        ],
    }


def _language_manifest(
    run_path: Path,
    output_path: Path,
    index: FoundationFeatureIndex,
    provider: FrozenLanguageFeatureProvider,
    training: Mapping[tuple[str, str], np.ndarray],
    training_evidence: Mapping[str, object],
    evaluation_entries: list[dict[str, object]],
) -> dict[str, object]:
    lock = provider.model_lock
    return {
        "schema_version": EVALUATION_LANGUAGE_SCHEMA,
        "instruction_count": len(evaluation_entries),
        "task_count": len({str(entry["task_id"]) for entry in evaluation_entries}),
        "locale": FORMAL_LANGUAGE_LOCALE,
        "encoder": {
            "model_id": lock.model_id,
            "revision": lock.revision,
            "representation_id": lock.representation_id,
            "lock_sha256": lock.lock_sha256,
            "output_dimension": lock.output_dimension,
        },
        "preprocess_sha256": index.preprocess_sha256,
        "training_embedding_count": len(training),
        "training_inputs": dict(training_evidence),
        "instructions": evaluation_entries,
        "isolation": {
            "evaluation_only": True,
            "training_artifacts_unchanged": True,
            "training_run": str(run_path),
            "evaluation_output": str(output_path),
            "training_cache": str(run_path / "feature-cache"),
            "evaluation_cache": str(output_path / "evaluation-language/cache"),
            "forbidden_training_targets": [
                "feature-cache",
                "features/language.json",
                "replay",
                "checkpoints",
                "deployments",
            ],
        },
    }


def _require_instruction_entry(entry: Mapping[str, object]) -> None:
    required_strings = (
        "task_id",
        "locale",
        "text",
        "text_sha256",
        "source_sha256",
        "cache_key",
        "encoder_lock_sha256",
        "preprocess_sha256",
        "path",
        "file_sha256",
    )
    if (
        any(not isinstance(entry.get(name), str) or not entry[name] for name in required_strings)
        or any(
            not _is_sha256(entry[name])
            for name in (
                "text_sha256",
                "source_sha256",
                "cache_key",
                "encoder_lock_sha256",
                "preprocess_sha256",
                "file_sha256",
            )
        )
        or int(entry.get("output_dimension", 0)) <= 0
        or int(entry.get("bytes", 0)) <= 0
    ):
        raise ValueError("evaluation language instruction evidence is incomplete")
    normalized = " ".join(str(entry["text"]).split())
    if hashlib.sha256(normalized.encode()).hexdigest() != entry["text_sha256"]:
        raise ValueError("evaluation language text hash differs")
    if language_source_sha256(normalized, str(entry["locale"])) != entry["source_sha256"]:
        raise ValueError("evaluation language source hash differs")
    key = FoundationCacheKey(
        "language",
        str(entry["source_sha256"]),
        str(entry["encoder_lock_sha256"]),
        str(entry["preprocess_sha256"]),
    )
    if key.digest != entry["cache_key"]:
        raise ValueError("evaluation language cache key differs")


def _require_training_evidence(manifest: Mapping[str, object]) -> None:
    isolation = manifest["isolation"]
    evidence = manifest.get("training_inputs")
    if not isinstance(isolation, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("evaluation language isolation evidence is missing")
    run_path = Path(str(isolation.get("training_run", ""))).resolve()
    if str(run_path) != evidence.get("run_path"):
        raise ValueError("evaluation language training run identity differs")
    identities = [
        evidence.get("language_index"),
        evidence.get("replay_manifest"),
        *(evidence.get("embedding_files", ()) or ()),
    ]
    if len(identities) < 3:
        raise ValueError("evaluation language training evidence is incomplete")
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise ValueError("evaluation language training evidence is invalid")
        path = _contained_member(run_path, Path(str(identity.get("path", ""))))
        if (
            not path.is_file()
            or not _is_sha256(identity.get("sha256"))
            or _sha256(path) != identity["sha256"]
            or path.stat().st_size != int(identity.get("bytes", -1))
        ):
            raise ValueError("evaluation language training evidence hash differs")


def _identity(root: Path, path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("training language input escaped its run")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _contained_member(root: Path, relative: Path) -> Path:
    if relative.is_absolute():
        raise ValueError("evaluation language artifact path must be relative")
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("evaluation language artifact escaped its directory")
    return result


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)
