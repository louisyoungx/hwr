#!/usr/bin/env python3
"""Verify provenance, dimensions, texture data, and hashes of formal 3D assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "assets/manifests/household_v1_sources.json"
DEFAULT_LOCK = ROOT / "assets/manifests/household_v1.lock.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(spec_path: Path, lock_path: Path) -> dict[str, Any]:
    spec = _read(spec_path)
    lock = _read(lock_path)
    errors: list[str] = []
    if lock.get("spec_sha256") != _sha256(spec_path):
        errors.append("source spec hash does not match lock")
    provider = spec.get("provider", {})
    if provider.get("license") != "CC0-1.0" or not provider.get("license_url"):
        errors.append("provider must declare the CC0 license and URL")
    specs = {item["id"]: item for item in spec.get("assets", [])}
    products = {item["id"]: item for item in lock.get("products", [])}
    if specs.keys() != products.keys():
        errors.append("source and product asset IDs differ")
    for asset_id, asset in specs.items():
        if not asset.get("source_page") or not asset.get("authors"):
            errors.append(f"{asset_id}: provenance is incomplete")
        product = products.get(asset_id, {})
        if product.get("vertices", 0) < 100 or product.get("faces", 0) < 100:
            errors.append(f"{asset_id}: mesh is too trivial")
        if product.get("uv_coordinates", 0) < 100:
            errors.append(f"{asset_id}: no substantive UV map")
        extents = product.get("extents_m", [0, 0, 0])
        if min(extents, default=0) <= 0 or max(extents, default=0) > 3.0:
            errors.append(f"{asset_id}: implausible metric extents {extents}")
        file_types: set[str] = set()
        for item in product.get("files", []):
            path = ROOT / item["path"]
            file_types.add(path.suffix.lower())
            if not path.is_file():
                errors.append(f"{asset_id}: missing {path}")
            elif path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
                errors.append(f"{asset_id}: hash or size mismatch for {path.name}")
        if ".obj" not in file_types or ".mtl" not in file_types or ".png" not in file_types:
            errors.append(f"{asset_id}: OBJ, MTL, and baked texture are required")
    upstream_ids = {item["source_id"] for item in lock.get("upstreams", [])}
    if upstream_ids != {item["source_id"] for item in specs.values()}:
        errors.append("upstream source set is incomplete")
    result = {
        "valid": not errors,
        "asset_count": len(products),
        "textured_meshes": sum(
            product.get("uv_coordinates", 0) >= 100 for product in products.values()
        ),
        "total_faces": sum(product.get("faces", 0) for product in products.values()),
        "license": provider.get("license"),
        "errors": errors,
    }
    if errors:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    print(json.dumps(verify(args.spec.resolve(), args.lock.resolve()), indent=2))


if __name__ == "__main__":
    main()
