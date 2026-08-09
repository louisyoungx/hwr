#!/usr/bin/env python3
"""Fetch, normalize, and lock the textured household mesh assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import ssl
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "assets/manifests/household_v1_sources.json"
DEFAULT_LOCK = ROOT / "assets/manifests/household_v1.lock.json"
DEFAULT_OUTPUT = ROOT / "assets/household_v1"
DEFAULT_CACHE = ROOT / ".cache/hwr/polyhaven"


@dataclass(frozen=True)
class Download:
    relative_path: str
    url: str
    md5: str
    size: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_bytes(url: str) -> bytes:
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "hwr-assets/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, context=context, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def _download(path: Path, record: Download) -> None:
    if path.exists() and _digest(path, "md5") == record.md5:
        return
    payload = _fetch_bytes(record.url)
    if hashlib.md5(payload).hexdigest() != record.md5:
        raise ValueError(f"upstream MD5 mismatch for {record.url}")
    if len(payload) != record.size:
        raise ValueError(f"upstream size mismatch for {record.url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _upstream_from_api(provider: dict[str, Any], source_id: str) -> dict[str, Any]:
    url = provider["api_url"].format(source_id=source_id)
    response = json.loads(_fetch_bytes(url))
    gltf = response["gltf"][provider["resolution"]]["gltf"]
    primary_name = Path(gltf["url"]).name
    downloads = [
        Download(primary_name, gltf["url"], gltf["md5"], int(gltf["size"]))
    ]
    downloads.extend(
        Download(path, value["url"], value["md5"], int(value["size"]))
        for path, value in sorted(gltf["include"].items())
    )
    return {
        "source_id": source_id,
        "api_url": url,
        "primary": primary_name,
        "downloads": [record.__dict__ for record in downloads],
    }


def _locked_upstreams(lock: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not lock:
        return {}
    return {item["source_id"]: item for item in lock.get("upstreams", [])}


def _resolve_upstreams(
    spec: dict[str, Any], lock: dict[str, Any] | None, refresh: bool
) -> dict[str, dict[str, Any]]:
    locked = _locked_upstreams(lock)
    resolved: dict[str, dict[str, Any]] = {}
    for asset in spec["assets"]:
        source_id = asset["source_id"]
        if source_id in resolved:
            continue
        if not refresh and source_id in locked:
            resolved[source_id] = locked[source_id]
        else:
            resolved[source_id] = _upstream_from_api(spec["provider"], source_id)
    return resolved


def _cache_source(cache: Path, upstream: dict[str, Any]) -> Path:
    source_dir = cache / upstream["source_id"]
    for value in upstream["downloads"]:
        record = Download(**value)
        _download(source_dir / record.relative_path, record)
    return source_dir / upstream["primary"]


def _prepare_scene(source: Path, asset: dict[str, Any]) -> trimesh.Scene:
    scene = trimesh.load(source, force="scene")
    selected = set(asset.get("include_geometries", []))
    if selected:
        missing = selected.difference(scene.geometry)
        if missing:
            raise ValueError(f"missing geometries for {asset['id']}: {sorted(missing)}")
        scene.delete_geometry(set(scene.geometry).difference(selected))
    rotate = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    scene.apply_transform(rotate)
    axis = asset["scale_axis"]
    index = {"x": 0, "y": 1, "z": 2}.get(axis)
    denominator = float(max(scene.extents) if index is None else scene.extents[index])
    scene.apply_scale(float(asset["target_m"]) / denominator)
    bounds = scene.bounds
    scene.apply_translation(
        [-float((bounds[0, 0] + bounds[1, 0]) / 2),
         -float((bounds[0, 1] + bounds[1, 1]) / 2),
         -float(bounds[0, 2])]
    )
    return scene


def _export(scene: trimesh.Scene, asset_id: str, output: Path) -> dict[str, Any]:
    asset_dir = output / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    mesh = scene.to_geometry()
    mesh.remove_unreferenced_vertices()
    obj, resources = export_obj(
        mesh,
        include_normals=False,
        include_texture=True,
        return_texture=True,
        mtl_name=f"{asset_id}.mtl",
        header="HWR household_v1 locked asset",
    )
    obj_path = asset_dir / f"{asset_id}.obj"
    obj_path.write_text(obj, encoding="utf-8")
    for name, payload in resources.items():
        path = asset_dir / name
        path.write_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    files = sorted(path for path in asset_dir.iterdir() if path.is_file())
    uv_count = int(len(mesh.visual.uv)) if hasattr(mesh.visual, "uv") else 0
    return {
        "id": asset_id,
        "directory": str(asset_dir.relative_to(ROOT)),
        "obj": str(obj_path.relative_to(ROOT)),
        "bounds_m": np.round(mesh.bounds, 8).tolist(),
        "extents_m": np.round(mesh.extents, 8).tolist(),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "uv_coordinates": uv_count,
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _digest(path, "sha256"),
            }
            for path in files
        ],
    }


def build_assets(
    spec_path: Path,
    lock_path: Path,
    output: Path,
    cache: Path,
    refresh_lock: bool,
) -> dict[str, Any]:
    spec = _read_json(spec_path)
    old_lock = _read_json(lock_path) if lock_path.exists() else None
    upstreams = _resolve_upstreams(spec, old_lock, refresh_lock)
    products = []
    for asset in spec["assets"]:
        upstream = upstreams[asset["source_id"]]
        source = _cache_source(cache, upstream)
        products.append(_export(_prepare_scene(source, asset), asset["id"], output))
        print(f"built {asset['id']}: {products[-1]['faces']} faces")
    lock = {
        "schema_version": "hwr.household-assets-lock/v1",
        "spec_sha256": _digest(spec_path, "sha256"),
        "provider": spec["provider"],
        "upstreams": sorted(upstreams.values(), key=lambda item: item["source_id"]),
        "products": products,
    }
    if not refresh_lock and old_lock and lock != old_lock:
        raise ValueError("generated assets differ from lock; use --refresh-lock intentionally")
    _write_json(lock_path, lock)
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh-lock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_assets(
        args.spec.resolve(),
        args.lock.resolve(),
        args.output.resolve(),
        args.cache.resolve(),
        args.refresh_lock,
    )
    print(json.dumps({"assets": len(result["products"]), "lock": str(args.lock)}, indent=2))


if __name__ == "__main__":
    main()
