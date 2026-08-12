#!/usr/bin/env python3
"""Run one pinned foundation provider in an isolated inference-only process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hwr.adapters.foundation import (
    Dinov3ViTDenseVisionProvider,
    Qwen3LanguageProvider,
    Siglip2VisionLanguageProvider,
    load_foundation_model_locks,
)


REPORT_SCHEMA = "hwr.foundation-model-verification/v1"


def _fixture_images(size: int = 224) -> np.ndarray:
    rows, columns = np.indices((size, size), dtype=np.float32)
    base = np.stack(
        (columns / (size - 1), rows / (size - 1), (rows + columns) / (2 * size - 2)),
        axis=-1,
    )
    return np.stack((base, np.flip(base, axis=1), np.flip(base, axis=0)))


def _vision_report(provider: Any) -> dict[str, Any]:
    valid = np.asarray((True, True, False), dtype=np.bool_)
    features = provider.encode_vision(_fixture_images(), valid, "a" * 64)
    norms = np.linalg.norm(features.values[features.valid], axis=-1)
    return {
        "feature_shape": list(features.values.shape),
        "valid_patch_count": int(features.valid.sum()),
        "minimum_valid_norm": float(norms.min()),
        "maximum_valid_norm": float(norms.max()),
        "invalid_features_zero": bool(np.all(features.values[~features.valid] == 0.0)),
        "mean_patch_variation": float(
            np.std(features.values[features.valid], axis=0).mean()
        ),
        "mean_valid_image_difference": float(
            np.abs(features.values[0] - features.values[1]).mean()
        ),
    }


def _language_report(provider: Any) -> dict[str, Any]:
    first = provider.encode_language("双手稳定搬运托盘", "zh-CN")
    paraphrase = provider.encode_language("请用两只手平稳地移动餐盘", "zh-CN")
    different = provider.encode_language("打开抽屉并放入物品", "zh-CN")
    return {
        "feature_shape": list(first.values.shape),
        "feature_norm": float(np.linalg.norm(first.values)),
        "paraphrase_cosine": float(first.values @ paraphrase.values),
        "different_intent_cosine": float(first.values @ different.values),
    }


def _verify(name: str, lock: Any, device: str) -> dict[str, Any]:
    common = {
        "model_id": lock.model_lock.model_id,
        "revision": lock.model_lock.revision,
        "lock_sha256": lock.model_lock.lock_sha256,
        "device": device,
    }
    if lock.adapter == "dinov3_vit":
        provider = Dinov3ViTDenseVisionProvider(lock, device=device)
        vision = _vision_report(provider)
        _require_vision_quality(vision, lock.model_lock.output_dimension)
        return {**common, "vision": vision}
    if lock.adapter == "siglip2":
        provider = Siglip2VisionLanguageProvider(lock, device=device)
        vision = _vision_report(provider)
        language = _language_report(provider)
        _require_vision_quality(vision, lock.model_lock.output_dimension)
        _require_language_quality(language, lock.model_lock.output_dimension)
        return {
            **common,
            "vision": vision,
            "language": language,
        }
    if lock.adapter == "qwen3_embedding":
        provider = Qwen3LanguageProvider(lock, device=device)
        language = _language_report(provider)
        _require_language_quality(language, lock.model_lock.output_dimension)
        return {**common, "language": language}
    raise ValueError(f"unsupported foundation adapter for {name}: {lock.adapter}")


def _require_vision_quality(report: Mapping[str, Any], dimension: int) -> None:
    if (
        report.get("feature_shape") != [3, 14, 14, dimension]
        or int(report.get("valid_patch_count", -1)) != 2 * 14 * 14
        or not 0.999 <= float(report.get("minimum_valid_norm", 0.0)) <= 1.001
        or not 0.999 <= float(report.get("maximum_valid_norm", 0.0)) <= 1.001
        or report.get("invalid_features_zero") is not True
        or float(report.get("mean_patch_variation", 0.0)) <= 1.0e-5
        or float(report.get("mean_valid_image_difference", 0.0)) <= 1.0e-5
    ):
        raise RuntimeError("foundation vision provider failed the non-degeneracy gate")


def _require_language_quality(report: Mapping[str, Any], dimension: int) -> None:
    norm = float(report.get("feature_norm", 0.0))
    paraphrase = float(report.get("paraphrase_cosine", -1.0))
    different = float(report.get("different_intent_cosine", 1.0))
    if (
        report.get("feature_shape") != [dimension]
        or not 0.999 <= norm <= 1.001
        or paraphrase <= different + 0.01
    ):
        raise RuntimeError("foundation language provider failed the semantic gate")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--locks", type=Path, default=root / "configs/foundation/model-locks.json"
    )
    parser.add_argument("--model-root", type=Path, default=root / "models/foundation")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    locks = load_foundation_model_locks(arguments.locks, arguments.model_root)
    if arguments.model not in locks:
        raise ValueError(f"unknown foundation model: {arguments.model}")
    report = {
        "schema_version": REPORT_SCHEMA,
        "model": arguments.model,
        **_verify(arguments.model, locks[arguments.model], arguments.device),
    }
    if arguments.output is not None:
        _write_atomic(arguments.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
