# Household V1 textured meshes

These OBJ/MTL/PNG files are deterministic, metric, Z-up derivatives of the
CC0 Poly Haven sources listed in `../manifests/household_v1_sources.json`.
`../manifests/household_v1.lock.json` freezes every upstream MD5, processed
SHA-256, dimension, vertex count, face count, and UV count.

Rebuild and verify from the repository root:

```bash
python -m pip install -e '.[assets3d]'
python scripts/fetch_3d_assets.py
python scripts/verify_3d_assets.py
```

`--refresh-lock` is an intentional provenance update, not a normal install
step. The visible meshes are rendering geometry. Formal MuJoCo scenes declare
separate, simpler collision geometry so that visual detail cannot silently
change contact behavior.
