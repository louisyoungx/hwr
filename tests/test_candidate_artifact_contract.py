from __future__ import annotations
from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib, json
from pathlib import Path
import shutil
import subprocess
import pytest
from hwr.eval import candidate_artifact_contract as contract
from hwr.apps import evaluate_candidate_artifact_contract as contract_app
ROOT = Path(__file__).resolve().parents[1]
EPISODE_IDENTITY = {
    "planned_episode_id": "episode", "task_id": "task/v1",
    "cell_id": "cell-00", "cell_ordinal": 0, "replicate_ordinal": 0,
    "candidate_ordinal": 0, "environment_seed": 1, "policy_rng_seed": 2,
    "replacement": False, "acquisition_base_pose": [0.0, 0.0, 0.0],
}
NEGATIVE_CATEGORIES = dict(line.split(":") for line in """
unknown_bank_schema:outer_schema unknown_inner_candidate_schema:candidate_schema
outer_inner_schema_mismatch:candidate_schema joint_relabel_with_self_hashes:git_anchor
v2_candidate_uses_legacy_root:root_binding legacy_or_capture_uses_v2_root:root_binding
same_name_existence_fallback:path_safety absolute_path:path_safety
parent_traversal:path_safety symlink_file_escape:path_safety
symlink_root_escape:path_safety blob_bytes_drift:blob_identity
blob_identity_drift:blob_identity duplicate_episode_identity:episode_identity
capture_ordinal_gap_or_duplicate:capture_order
nonfinal_or_multiple_final_capture:capture_order candidate_count_mismatch:candidate_content
selected_index_out_of_bounds:selection_metadata
selected_canonical_identity_mismatch:selection_metadata
acquisition_input_hash_mismatch:acquisition_binding producer_commit_drift:producer_commit
producer_source_blob_drift:producer_source artifact_commit_tree_or_blob_drift:git_anchor
trust_anchor_drift:anchor_fingerprint app_bypasses_resolver:architecture
""".split())

def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                      sort_keys=True).encode("ascii")

def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def _identity(path: str, content: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(content), "sha256": _sha256(content)}

def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()

def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

@dataclass(frozen=True)
class ArtifactFixture:
    repository: Path; bank_root: Path; legacy_root: Path
    anchor: contract.CandidateArtifactTrustAnchor
    anchor_fingerprint: str; candidate_path: str

def _build_fixture(repository: Path) -> ArtifactFixture:
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "p80@example.invalid")
    _git(repository, "config", "user.name", "P80 Test")
    source_contents = {
        "src/hwr/eval/candidate_mask_ownership.py": b"producer = 'v2'\n",
        "src/hwr/eval/target_selection.py": b"CANDIDATE_SCHEMA = 'v1'\n",
        "src/hwr/apps/evaluate_candidate_mask_ownership.py": b"runner = 'v2'\n",
    }
    for name, content in source_contents.items():
        _write(repository / name, content)
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "producer")
    producer_commit = _git(repository, "rev-parse", "HEAD")
    return _build_fixture_artifacts(repository, producer_commit, source_contents)

def _build_fixture_artifacts(repository: Path,
                             producer_commit: str,
                             source_contents: dict[str, bytes]) -> ArtifactFixture:
    bank_root = Path("runs/p79-bank"); legacy_root = Path("runs/p50-source")
    candidate_path = "blobs/episode/candidate-set.json"; policy_path = "blobs/episode/capture-00-policy.bin"
    visible_path = "blobs/episode/capture-00-candidate-visible.bin"
    policy = b"policy-visible-input"; visible = b"candidate-visible-input"
    candidate_record = [100, 200, 300, 400, 500, 600, 700, 0, 10, 20, 30, 40, 2]
    selected_identity = _sha256(_canonical(candidate_record))
    input_hashes = [_sha256(policy)]
    legacy_candidate = _canonical({
        "schema_version": contract.LEGACY_CANDIDATE_SCHEMA,
        "acquisition_input_sha256": input_hashes,
        "candidate_count": 1,
        "candidates": [candidate_record],
    })
    v2_candidate = _canonical({
        "schema_version": contract.V2_CANDIDATE_SCHEMA,
        "acquisition_input_sha256": input_hashes,
        "candidate_count": 1,
        "candidates": [candidate_record],
    })
    _write(repository / legacy_root / candidate_path, legacy_candidate)
    _write(repository / legacy_root / policy_path, policy)
    _write(repository / legacy_root / visible_path, visible)
    score_hash = _sha256(b"score")
    legacy_artifacts, legacy_manifest = _build_legacy_fixture(
        repository, legacy_root, candidate_path, policy_path, visible_path,
        legacy_candidate, policy, visible, score_hash,
    )
    bank_bytes, p79_manifest = _build_p79_fixture(
        repository, bank_root, legacy_root, candidate_path, v2_candidate,
        legacy_artifacts, legacy_manifest, producer_commit, source_contents,
        score_hash, selected_identity,
    )
    return _commit_fixture(
        repository=repository, producer_commit=producer_commit,
        bank_root=bank_root, legacy_root=legacy_root,
        candidate_path=candidate_path, source_contents=source_contents,
        legacy_manifest=legacy_manifest, bank_bytes=bank_bytes,
        p79_manifest=p79_manifest,
    )

def _build_legacy_fixture(
    repository, legacy_root, candidate_path, policy_path, visible_path,
    legacy_candidate, policy, visible, score_hash,
):
    capture = {
        "schema_version": contract.P50_CAPTURE_SCHEMA,
        "capture_ordinal": 0, "acquisition_phase": "A4_seal",
        "final_input": True, "observation_timestamp_ns": 50_000_000,
        "sequence_id": 1,
        "policy_input": _identity(policy_path, policy),
        "candidate_visible_input": _identity(visible_path, visible),
    }
    capsule_candidate = {
        **_identity(candidate_path, legacy_candidate),
        "schema_version": contract.LEGACY_CANDIDATE_SCHEMA,
        "candidate_count": 1,
        "selected_index": 0,
        "score_bytes_sha256": score_hash,
        "generated_online": True,
    }
    capsules = _canonical({
        "capsule_count": 1,
        "episodes": [{
            "schema_version": contract.P50_CAPSULE_SCHEMA,
            **EPISODE_IDENTITY,
            "captures": [capture],
            "candidate_set": capsule_candidate,
        }],
    })
    _write(repository / legacy_root / "capsules.json", capsules)
    legacy_artifacts = {
        candidate_path: _identity(candidate_path, legacy_candidate),
        policy_path: _identity(policy_path, policy),
        visible_path: _identity(visible_path, visible),
        "capsules.json": _identity("capsules.json", capsules),
    }
    legacy_manifest = _canonical({
        "schema_version": contract.P50_MANIFEST_SCHEMA,
        "proposal_id": "R0001-P50-E1",
        "status": "complete",
        "source_commit": contract.P50_SOURCE_COMMIT,
        "schemas": {"candidate": contract.LEGACY_CANDIDATE_SCHEMA},
        "artifacts": {
            name: {"bytes": value["bytes"], "sha256": value["sha256"]}
            for name, value in sorted(legacy_artifacts.items())
        },
    })
    _write(repository / legacy_root / "manifest.json", legacy_manifest)
    return legacy_artifacts, legacy_manifest

def _build_p79_fixture(
    repository, bank_root, legacy_root, candidate_path, v2_candidate,
    legacy_artifacts, legacy_manifest, producer_commit, source_contents,
    score_hash, selected_identity,
):
    policy_path = "blobs/episode/capture-00-policy.bin"
    visible_path = "blobs/episode/capture-00-candidate-visible.bin"
    _write(repository / bank_root / candidate_path, v2_candidate)
    bank = {
        "schema_version": contract.P79_BANK_SCHEMA,
        "proposal_id": contract.P79_PROPOSAL_ID,
        "source_acquisition": legacy_root.as_posix(),
        "episode_count": 1,
        "capture_count": 1,
        "episodes": [{
            **EPISODE_IDENTITY,
            "captures": [{
                "capture_ordinal": 0, "final_input": True,
                "observation_identity": [50_000_000, 1],
                "policy_input": legacy_artifacts[policy_path],
                "candidate_visible_input": legacy_artifacts[visible_path],
            }],
            "old_candidate_set": {
                **legacy_artifacts[candidate_path],
                "schema_version": contract.LEGACY_CANDIDATE_SCHEMA,
                "candidate_count": 1,
                "selected_index": 0,
                "score_bytes_sha256": score_hash,
                "selected_canonical_identity": selected_identity,
                "generated_online": True,
            },
            "candidate_set": {
                **_identity(candidate_path, v2_candidate),
                "schema_version": contract.V2_CANDIDATE_SCHEMA,
                "candidate_count": 1,
                "selected_index": 0,
                "score_bytes_sha256": score_hash,
                "selected_canonical_identity": selected_identity,
            },
        }],
    }
    bank_bytes = _canonical(bank)
    _write(repository / bank_root / "bank.json", bank_bytes)
    report = b"{}\n"
    _write(repository / bank_root / "report.json", report)
    regression = _canonical({
        "schema_version": contract.P79_REGRESSION_SCHEMA,
        "proposal_id": contract.P79_PROPOSAL_ID,
        "episode_count": 1,
        "records": [{
            "planned_episode_id": "episode", "task_id": "task/v1",
            "old": {
                "schema_version": contract.LEGACY_CANDIDATE_SCHEMA,
                "candidate_count": 1,
                "candidate_set_sha256":
                    legacy_artifacts[candidate_path]["sha256"],
                "selected_index": 0,
                "selected_canonical_identity": selected_identity,
            },
            "new": {
                "schema_version": contract.V2_CANDIDATE_SCHEMA,
                "candidate_count": 1,
                "candidate_set_sha256": _sha256(v2_candidate),
                "selected_index": 0,
                "selected_canonical_identity": selected_identity,
            },
            "audit": {
                "schema_version": contract.P79_AUDIT_SCHEMA,
                "proposal_id": contract.P79_PROPOSAL_ID,
                "decision":
                    "accepted as deterministic candidate-generator correction",
                "frame_count": 1,
                "frames": [{
                    "frame_ordinal": 0,
                    "observation_identity": [50_000_000, 1],
                }],
                "checks": {
                    "input_head_depth_valid_byte_identical": True,
                    "oracle_parent_mask_mutation_count_zero": True,
                    "oracle_traversal_raw_multisets_equal": True,
                    "passed": True,
                    "production_final_equals_oracle_row_major": True,
                    "production_parent_valid_byte_identical": True,
                    "production_row_major_equals_oracle": True,
                    "traversal_final_candidate_bytes_equal": True,
                },
                "traversals": {
                    name: {
                        "candidate_set_sha256": _sha256(v2_candidate),
                        "candidate_count": 1,
                    }
                    for name in (
                        "row_major", "reverse_row_major", "column_major"
                    )
                },
            },
        }],
    })
    _write(repository / bank_root / "regression.json", regression)
    p79_artifacts = {
        "bank.json": _identity("bank.json", bank_bytes),
        candidate_path: _identity(candidate_path, v2_candidate),
        "regression.json": _identity("regression.json", regression),
        "report.json": _identity("report.json", report),
    }
    input_files = [
        {
            **value,
            "path": (legacy_root / name).as_posix(),
        }
        for name, value in sorted(legacy_artifacts.items())
    ]
    input_files.append(
        _identity(
            (legacy_root / "manifest.json").as_posix(),
            legacy_manifest,
        )
    )
    sources = {
        name: {
            "path": name,
            "bytes": len(content),
            "sha256": _sha256(content),
        }
        for name, content in source_contents.items()
    }
    p79_manifest = _canonical({
        "schema_version": contract.P79_MANIFEST_SCHEMA,
        "proposal_id": contract.P79_PROPOSAL_ID,
        "status": "complete",
        "source_commit": producer_commit,
        "input": {
            "path": legacy_root.as_posix(),
            "files": input_files,
        },
        "provenance": {"sources": sources},
        "artifacts": {
            name: {"bytes": value["bytes"], "sha256": value["sha256"]}
            for name, value in sorted(p79_artifacts.items())
        },
    })
    return bank_bytes, p79_manifest

def _commit_fixture(
    *,
    repository: Path,
    producer_commit: str,
    bank_root: Path,
    legacy_root: Path,
    candidate_path: str,
    source_contents: dict[str, bytes],
    legacy_manifest: bytes,
    bank_bytes: bytes,
    p79_manifest: bytes,
) -> ArtifactFixture:
    _write(repository / bank_root / "manifest.json", p79_manifest)
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "artifact")
    artifact_commit = _git(repository, "rev-parse", "HEAD")
    artifact_tree = _git(repository, "rev-parse", f"HEAD:{bank_root}")
    bank_blob = _git(repository, "rev-parse", f"HEAD:{bank_root}/bank.json")
    manifest_blob = _git(
        repository, "rev-parse", f"HEAD:{bank_root}/manifest.json"
    )
    source_anchors = tuple(
        contract.GitBlobAnchor(
            path=name,
            git_blob=_git(repository, "rev-parse", f"{producer_commit}:{name}"),
            sha256=_sha256(content),
            byte_count=len(content),
        )
        for name, content in sorted(source_contents.items())
    )
    anchor = contract.CandidateArtifactTrustAnchor(
        artifact_commit=artifact_commit,
        artifact_root=bank_root.as_posix(),
        artifact_tree=artifact_tree,
        bank=contract.GitBlobAnchor(
            "bank.json", bank_blob, _sha256(bank_bytes), len(bank_bytes)
        ),
        manifest=contract.GitBlobAnchor(
            "manifest.json", manifest_blob, _sha256(p79_manifest),
            len(p79_manifest)
        ),
        bank_schema=contract.P79_BANK_SCHEMA,
        manifest_schema=contract.P79_MANIFEST_SCHEMA,
        candidate_schema=contract.V2_CANDIDATE_SCHEMA,
        proposal_id=contract.P79_PROPOSAL_ID,
        expected_episodes=1,
        expected_captures=1,
        expected_p79_artifacts=4,
        producer_commit=producer_commit,
        producer_sources=source_anchors,
        legacy_root=legacy_root.as_posix(),
        legacy_manifest_bytes=len(legacy_manifest),
        legacy_manifest_sha256=_sha256(legacy_manifest),
        legacy_source_commit=contract.P50_SOURCE_COMMIT,
        legacy_manifest_schema=contract.P50_MANIFEST_SCHEMA,
        legacy_candidate_schema=contract.LEGACY_CANDIDATE_SCHEMA,
        expected_p50_artifacts=4,
        frozen_document_commit=artifact_commit,
        frozen_document_path="docs/research-loop/0015/03-experiment.md",
        frozen_document_blob="0" * 40,
        frozen_document_sha256="0" * 64,
    )
    anchor = _freeze_anchor(repository, anchor)
    return ArtifactFixture(
        repository=repository,
        bank_root=bank_root,
        legacy_root=legacy_root,
        anchor=anchor,
        anchor_fingerprint=contract.candidate_artifact_anchor_fingerprint(anchor),
        candidate_path=candidate_path,
    )
def _freeze_anchor(repository: Path, anchor):
    values = (
        anchor.artifact_commit, anchor.artifact_root, anchor.artifact_tree,
        anchor.bank.git_blob, anchor.bank.sha256, anchor.manifest.git_blob,
        anchor.manifest.sha256, anchor.bank_schema, anchor.manifest_schema,
        anchor.candidate_schema, anchor.proposal_id, anchor.producer_commit,
        anchor.legacy_root, str(anchor.legacy_manifest_bytes),
        anchor.legacy_manifest_sha256, anchor.legacy_source_commit,
        anchor.legacy_manifest_schema, anchor.legacy_candidate_schema,
        *(item for source in anchor.producer_sources
          for item in (source.path, source.git_blob, source.sha256)),
    )
    path = Path(anchor.frozen_document_path)
    content = ("\n".join(values) + "\n").encode()
    _write(repository / path, content)
    _git(repository, "add", path.as_posix())
    _git(repository, "commit", "-qm", "refreeze receipt")
    return replace(
        anchor, frozen_document_commit=_git(repository, "rev-parse", "HEAD"),
        frozen_document_blob=_git(repository, "rev-parse", f"HEAD:{path}"),
        frozen_document_sha256=_sha256(content),
    )

@pytest.fixture
def artifact_fixture(tmp_path: Path) -> ArtifactFixture:
    return _build_fixture(tmp_path / "repository")

def _resolve(fixture: ArtifactFixture, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        contract, "FORMAL_TRUST_ANCHOR_FINGERPRINT", fixture.anchor_fingerprint
    )
    return contract.resolve_candidate_artifact(
        fixture.repository,
        p79_v2_bank=fixture.bank_root,
        p50_legacy_source=fixture.legacy_root,
        trust_anchor=fixture.anchor,
    )

def test_typed_envelope_is_frozen_and_roots_are_explicit(
    artifact_fixture: ArtifactFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _resolve(artifact_fixture, monkeypatch)
    assert envelope.bank_schema == contract.P79_BANK_SCHEMA
    assert envelope.producer_commit == artifact_fixture.anchor.producer_commit
    assert envelope.validation_counts == contract.ValidationCounts(
        episode_count=1,
        capture_count=1,
        v2_candidate_count=1,
        legacy_candidate_count=1,
        capture_blob_count=2,
        p79_artifact_count=4,
        p50_artifact_count=4,
        p50_input_file_count=5,
    )
    episode = envelope.episodes[0]
    assert episode.candidate.root_role is contract.RootRole.P79_V2_BANK
    assert episode.legacy_candidate.root_role is (
        contract.RootRole.P50_LEGACY_SOURCE
    )
    assert episode.captures[0].policy_input.root_role is (
        contract.RootRole.P50_LEGACY_SOURCE
    )
    assert episode.captures[0].schema == contract.P50_CAPTURE_SCHEMA
    assert episode.captures[0].acquisition_phase == "A4_seal"
    assert episode.candidate.selection_relation_validated is False
    assert episode.candidate.blob.content != episode.legacy_candidate.blob.content
    assert len(envelope.canonical_receipt_bytes()) > 0
    with pytest.raises(FrozenInstanceError):
        episode.candidate.bound_selected_index = -1  # type: ignore[misc]

def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value

def _store(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))

def _candidate_file(fixture: ArtifactFixture, role: contract.RootRole) -> Path:
    root = (
        fixture.bank_root
        if role is contract.RootRole.P79_V2_BANK
        else fixture.legacy_root
    )
    return fixture.repository / root / fixture.candidate_path

def _mutate(
    name: str,
    fixture: ArtifactFixture,
) -> tuple[Path, Path, contract.CandidateArtifactTrustAnchor]:
    bank_path = fixture.repository / fixture.bank_root / "bank.json"
    manifest_path = fixture.repository / fixture.bank_root / "manifest.json"
    bank = _load(bank_path)
    episode = bank["episodes"][0]
    assert isinstance(episode, dict)
    candidate = episode["candidate_set"]
    legacy = episode["old_candidate_set"]
    captures = episode["captures"]
    assert isinstance(candidate, dict)
    assert isinstance(legacy, dict)
    assert isinstance(captures, list)
    bank_root = fixture.bank_root
    legacy_root = fixture.legacy_root
    anchor = fixture.anchor
    if name == "unknown_bank_schema":
        bank["schema_version"] = "unknown/bank"
        _store(bank_path, bank)
    elif name == "unknown_inner_candidate_schema":
        document = _load(_candidate_file(fixture, contract.RootRole.P79_V2_BANK))
        document["schema_version"] = "unknown/candidate"
        _store(_candidate_file(fixture, contract.RootRole.P79_V2_BANK), document)
    elif name == "outer_inner_schema_mismatch":
        candidate["schema_version"] = contract.LEGACY_CANDIDATE_SCHEMA
        _store(bank_path, bank)
    elif name == "joint_relabel_with_self_hashes":
        candidate_path = _candidate_file(fixture, contract.RootRole.P79_V2_BANK)
        document = _load(candidate_path)
        document["schema_version"] = contract.LEGACY_CANDIDATE_SCHEMA
        content = _canonical(document)
        candidate_path.write_bytes(content)
        candidate.update({
            "schema_version": contract.LEGACY_CANDIDATE_SCHEMA,
            "bytes": len(content),
            "sha256": _sha256(content),
        })
        bank_bytes = _canonical(bank)
        bank_path.write_bytes(bank_bytes)
        manifest = _load(manifest_path)
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts[fixture.candidate_path] = {
            "bytes": len(content), "sha256": _sha256(content)
        }
        artifacts["bank.json"] = {
            "bytes": len(bank_bytes), "sha256": _sha256(bank_bytes)
        }
        regression_path = fixture.repository / fixture.bank_root / "regression.json"
        regression = _load(regression_path)
        regression["records"][0]["new"].update({
            "schema_version": contract.LEGACY_CANDIDATE_SCHEMA,
            "candidate_set_sha256": _sha256(content),
        })
        for traversal in regression["records"][0]["audit"]["traversals"].values():
            traversal["candidate_set_sha256"] = _sha256(content)
        _store(regression_path, regression)
        regression_content = regression_path.read_bytes()
        artifacts["regression.json"] = {
            "bytes": len(regression_content),
            "sha256": _sha256(regression_content),
        }
        _store(manifest_path, manifest)
    elif name == "v2_candidate_uses_legacy_root":
        bank_root = fixture.legacy_root
    elif name == "legacy_or_capture_uses_v2_root":
        legacy_root = fixture.bank_root
    elif name == "same_name_existence_fallback":
        _candidate_file(fixture, contract.RootRole.P79_V2_BANK).unlink()
    elif name == "absolute_path":
        candidate["path"] = _candidate_file(
            fixture, contract.RootRole.P79_V2_BANK
        ).as_posix()
        _store(bank_path, bank)
    elif name == "parent_traversal":
        candidate["path"] = "../candidate-set.json"
        _store(bank_path, bank)
    elif name == "symlink_file_escape":
        candidate_path = _candidate_file(fixture, contract.RootRole.P79_V2_BANK)
        outside = fixture.repository / "outside-candidate.json"
        outside.write_bytes(candidate_path.read_bytes())
        candidate_path.unlink()
        candidate_path.symlink_to(outside)
    elif name == "symlink_root_escape":
        source = fixture.repository / fixture.bank_root
        outside = fixture.repository / "outside-bank"
        shutil.copytree(source, outside)
        shutil.rmtree(source)
        source.symlink_to(outside, target_is_directory=True)
    elif name == "blob_bytes_drift":
        path = _candidate_file(fixture, contract.RootRole.P79_V2_BANK)
        path.write_bytes(path.read_bytes() + b"x")
    elif name == "blob_identity_drift":
        manifest = _load(manifest_path)
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, dict)
        identity = artifacts[fixture.candidate_path]
        assert isinstance(identity, dict)
        identity["bytes"] = int(identity["bytes"]) + 1
        _store(manifest_path, manifest)
    elif name == "duplicate_episode_identity":
        bank["episodes"].append(dict(episode))
        _store(bank_path, bank)
    elif name == "capture_ordinal_gap_or_duplicate":
        duplicate = dict(captures[0])
        captures.append(duplicate)
        _store(bank_path, bank)
        capsules_path = fixture.repository / fixture.legacy_root / "capsules.json"
        capsules = _load(capsules_path)
        capsules["episodes"][0]["captures"].append(
            dict(capsules["episodes"][0]["captures"][0])
        )
        _store(capsules_path, capsules)
    elif name == "nonfinal_or_multiple_final_capture":
        second = dict(captures[0])
        second["capture_ordinal"] = 1
        captures.append(second)
        _store(bank_path, bank)
        capsules_path = fixture.repository / fixture.legacy_root / "capsules.json"
        capsules = _load(capsules_path)
        second = dict(capsules["episodes"][0]["captures"][0])
        second["capture_ordinal"] = 1
        capsules["episodes"][0]["captures"].append(second)
        _store(capsules_path, capsules)
    elif name == "candidate_count_mismatch":
        candidate["candidate_count"] = 2
        _store(bank_path, bank)
    elif name == "selected_index_out_of_bounds":
        candidate["selected_index"] = 3
        _store(bank_path, bank)
    elif name == "selected_canonical_identity_mismatch":
        candidate["selected_canonical_identity"] = "0" * 64
        _store(bank_path, bank)
    elif name == "acquisition_input_hash_mismatch":
        path = _candidate_file(fixture, contract.RootRole.P79_V2_BANK)
        document = _load(path)
        document["acquisition_input_sha256"] = ["0" * 64]
        _store(path, document)
    elif name == "producer_commit_drift":
        manifest = _load(manifest_path)
        manifest["source_commit"] = "0" * 40
        _store(manifest_path, manifest)
    elif name == "producer_source_blob_drift":
        source = anchor.producer_sources[0]
        anchor = replace(
            anchor,
            producer_sources=(
                replace(source, git_blob="0" * 40),
                *anchor.producer_sources[1:],
            ),
        )
    elif name == "artifact_commit_tree_or_blob_drift":
        pass
    elif name == "trust_anchor_drift":
        anchor = replace(anchor, expected_captures=2)
    else:
        raise AssertionError(name)
    return bank_root, legacy_root, anchor

def _copy_fixture(
    source: ArtifactFixture,
    destination: Path,
) -> ArtifactFixture:
    shutil.copytree(source.repository, destination, symlinks=True)
    return replace(source, repository=destination)

def _refresh_internal_ledgers(fixture: ArtifactFixture) -> None:
    repository = fixture.repository
    bank_path = repository / fixture.bank_root / "bank.json"
    bank = _load(bank_path)
    for episode in bank["episodes"]:
        for name, root in (
            ("candidate_set", fixture.bank_root),
            ("old_candidate_set", fixture.legacy_root),
        ):
            value = episode[name]
            path = repository / root / value["path"]
            if path.exists():
                content = path.read_bytes()
                value.update(bytes=len(content), sha256=_sha256(content))
    _store(bank_path, bank)
    p50_manifest_path = repository / fixture.legacy_root / "manifest.json"
    p50_manifest = _load(p50_manifest_path)
    for name, identity in p50_manifest["artifacts"].items():
        path = repository / fixture.legacy_root / name
        if path.exists():
            content = path.read_bytes()
            identity.update(bytes=len(content), sha256=_sha256(content))
    _store(p50_manifest_path, p50_manifest)
    p79_manifest_path = repository / fixture.bank_root / "manifest.json"
    p79_manifest = _load(p79_manifest_path)
    for name, identity in p79_manifest["artifacts"].items():
        path = repository / fixture.bank_root / name
        if path.exists():
            content = path.read_bytes()
            identity.update(bytes=len(content), sha256=_sha256(content))
    for identity in p79_manifest["input"]["files"]:
        path = repository / identity["path"]
        if path.exists():
            content = path.read_bytes()
            identity.update(bytes=len(content), sha256=_sha256(content))
    _store(p79_manifest_path, p79_manifest)

def _refreshed_anchor(
    fixture: ArtifactFixture,
    *,
    seal_ledgers: bool = True,
    freeze_external_anchor: bool = True,
) -> ArtifactFixture:
    if seal_ledgers:
        _refresh_internal_ledgers(fixture)
    _git(fixture.repository, "add", "-A")
    _git(fixture.repository, "commit", "-qm", "mutated artifact")
    if not freeze_external_anchor:
        return fixture
    commit = _git(fixture.repository, "rev-parse", "HEAD")
    bank_path = fixture.repository / fixture.bank_root / "bank.json"
    manifest_path = fixture.repository / fixture.bank_root / "manifest.json"
    bank = bank_path.read_bytes()
    manifest = manifest_path.read_bytes()
    legacy_manifest = (
        fixture.repository / fixture.legacy_root / "manifest.json"
    ).read_bytes()
    anchor = replace(
        fixture.anchor,
        artifact_commit=commit,
        artifact_tree=_git(
            fixture.repository, "rev-parse", f"HEAD:{fixture.bank_root}"
        ),
        bank=contract.GitBlobAnchor(
            "bank.json",
            _git(fixture.repository, "rev-parse", f"HEAD:{fixture.bank_root}/bank.json"),
            _sha256(bank),
            len(bank),
        ),
        manifest=contract.GitBlobAnchor(
            "manifest.json",
            _git(
                fixture.repository,
                "rev-parse",
                f"HEAD:{fixture.bank_root}/manifest.json",
            ),
            _sha256(manifest),
            len(manifest),
        ),
        legacy_manifest_bytes=len(legacy_manifest),
        legacy_manifest_sha256=_sha256(legacy_manifest),
    )
    anchor = _freeze_anchor(fixture.repository, anchor)
    return replace(
        fixture,
        anchor=anchor,
        anchor_fingerprint=contract.candidate_artifact_anchor_fingerprint(anchor),
    )

@pytest.mark.parametrize("name", contract.PREREGISTERED_NEGATIVE_CONTROLS)
def test_preregistered_negative_control_fails_closed(
    name: str, artifact_fixture: ArtifactFixture, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(NEGATIVE_CATEGORIES) == contract.PREREGISTERED_NEGATIVE_CONTROLS
    if name == "app_bypasses_resolver":
        audit = contract_app.audit_consumer_architecture(
            "from hwr.apps import read_bound_blob\n",
            (
                "from pathlib import Path\n"
                "from hwr.eval.candidate_artifact_contract import "
                "resolve_candidate_artifact\n"
                "def run():\n"
                "    return Path('artifact').read_bytes()\n"
            ),
        )
        assert audit["passed"] is False
        assert audit["category"] == "architecture"
        return
    fixture = _copy_fixture(artifact_fixture, tmp_path / "mutation")
    bank_root, legacy_root, anchor = _mutate(name, fixture)
    fixture = replace(fixture, bank_root=bank_root, legacy_root=legacy_root,
                      anchor=anchor)
    if name == "joint_relabel_with_self_hashes":
        fixture = _refreshed_anchor(fixture, freeze_external_anchor=False)
    elif name not in {"symlink_root_escape", "artifact_commit_tree_or_blob_drift",
                      "trust_anchor_drift", "producer_source_blob_drift",
                      "v2_candidate_uses_legacy_root",
                      "legacy_or_capture_uses_v2_root"}:
        fixture = _refreshed_anchor(
            fixture,
            seal_ledgers=name not in {
                "blob_bytes_drift", "blob_identity_drift",
                "same_name_existence_fallback",
            },
        )
    expected_fingerprint = fixture.anchor_fingerprint
    if name == "producer_source_blob_drift":
        expected_fingerprint = contract.candidate_artifact_anchor_fingerprint(
            fixture.anchor
        )
    if name == "artifact_commit_tree_or_blob_drift":
        for field in ("artifact_tree", "bank"):
            altered = (
                replace(fixture.anchor, artifact_tree="0" * 40)
                if field == "artifact_tree"
                else replace(fixture.anchor, bank=replace(
                    fixture.anchor.bank, git_blob="0" * 40
                ))
            )
            monkeypatch.setattr(
                contract,
                "FORMAL_TRUST_ANCHOR_FINGERPRINT",
                contract.candidate_artifact_anchor_fingerprint(altered),
            )
            with pytest.raises(contract.CandidateArtifactContractError) as drift:
                contract.resolve_candidate_artifact(
                    fixture.repository,
                    p79_v2_bank=fixture.bank_root,
                    p50_legacy_source=fixture.legacy_root,
                    trust_anchor=altered,
                )
            assert drift.value.category == "git_anchor"
        return
    with pytest.raises(contract.CandidateArtifactContractError) as raised:
        monkeypatch.setattr(
            contract, "FORMAL_TRUST_ANCHOR_FINGERPRINT", expected_fingerprint
        )
        contract.resolve_candidate_artifact(
            fixture.repository,
            p79_v2_bank=fixture.bank_root,
            p50_legacy_source=fixture.legacy_root,
            trust_anchor=fixture.anchor,
        )
    assert raised.value.category == NEGATIVE_CATEGORIES[name]
