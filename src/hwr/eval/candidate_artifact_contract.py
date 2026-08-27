from __future__ import annotations
import hashlib, json, subprocess
from dataclasses import asdict, dataclass; from enum import Enum
from pathlib import Path, PurePosixPath; from typing import Mapping, Sequence
from hwr.apps import read_bound_blob; P79_PROPOSAL_ID = "R0001-P79-E1"; P79_BANK_SCHEMA = "hwr.p79-candidate-bank/v1"
P79_MANIFEST_SCHEMA = "hwr.p79-candidate-mask-ownership-artifacts/v1"; P79_REGRESSION_SCHEMA = "hwr.p79-candidate-mask-regression/v1"
V2_CANDIDATE_SCHEMA = "hwr.p79-target-candidates/v2"; P50_MANIFEST_SCHEMA = "hwr.p50-acquisition-artifacts/v1"
LEGACY_CANDIDATE_SCHEMA = "hwr.p41-target-candidates/v1"; P50_CAPTURE_SCHEMA = "hwr.p50-acquisition-capture/v1"
P79_AUDIT_SCHEMA = "hwr.p79-candidate-mask-ownership-audit/v1"; P50_SOURCE_COMMIT = "d67791a53491ce37cddaef4bd7d6b71ad3e66ac2"
P50_CAPSULE_SCHEMA = "hwr.p50-acquisition-capsule/v1"; ACCEPTED_DECISION = "accepted as version-sealed candidate artifact consumer contract"
PREREGISTERED_NEGATIVE_CONTROLS = tuple("""
unknown_bank_schema unknown_inner_candidate_schema outer_inner_schema_mismatch
joint_relabel_with_self_hashes v2_candidate_uses_legacy_root
legacy_or_capture_uses_v2_root same_name_existence_fallback absolute_path
parent_traversal symlink_file_escape symlink_root_escape blob_bytes_drift
blob_identity_drift duplicate_episode_identity capture_ordinal_gap_or_duplicate
nonfinal_or_multiple_final_capture candidate_count_mismatch
selected_index_out_of_bounds selected_canonical_identity_mismatch
acquisition_input_hash_mismatch producer_commit_drift producer_source_blob_drift
artifact_commit_tree_or_blob_drift trust_anchor_drift app_bypasses_resolver
""".split())
class CandidateArtifactContractError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}"); self.category = category
def _require(condition: bool, message: str, category: str = "contract") -> None:
    if not condition: raise CandidateArtifactContractError(category, message)
class RootRole(str, Enum):
    P79_V2_BANK = "P79_V2_BANK"
    P50_LEGACY_SOURCE = "P50_LEGACY_SOURCE"
@dataclass(frozen=True)
class GitBlobAnchor:
    path: str; git_blob: str; sha256: str; byte_count: int
@dataclass(frozen=True)
class CandidateArtifactTrustAnchor:
    artifact_commit: str; artifact_root: str; artifact_tree: str
    bank: GitBlobAnchor; manifest: GitBlobAnchor
    bank_schema: str; manifest_schema: str; candidate_schema: str
    proposal_id: str; expected_episodes: int; expected_captures: int
    expected_p79_artifacts: int; producer_commit: str
    producer_sources: tuple[GitBlobAnchor, ...]
    legacy_root: str; legacy_manifest_bytes: int; legacy_manifest_sha256: str
    legacy_source_commit: str; legacy_manifest_schema: str; legacy_candidate_schema: str
    expected_p50_artifacts: int; frozen_document_commit: str
    frozen_document_path: str; frozen_document_blob: str; frozen_document_sha256: str
@dataclass(frozen=True)
class BoundBlob:
    root_role: RootRole; path: str; byte_count: int; sha256: str; content: bytes
@dataclass(frozen=True)
class CandidateEnvelope:
    root_role: RootRole; schema: str; producer_commit: str; path: str
    blob: BoundBlob; canonical_bytes: bytes; candidate_count: int
    bound_selected_index: int; bound_score_bytes_sha256: str
    bound_selected_canonical_identity: str | None; acquisition_input_sha256: tuple[str, ...]
    selection_relation_validated: bool = False
@dataclass(frozen=True)
class CaptureEnvelope:
    root_role: RootRole; planned_episode_id: str; capture_ordinal: int
    final_input: bool; observation_timestamp_ns: int; sequence_id: int
    schema: str; acquisition_phase: str
    policy_input: BoundBlob; candidate_visible_input: BoundBlob
    @property
    def composite_identity(self) -> tuple[object, ...]:
        return (
            self.planned_episode_id,
            self.capture_ordinal,
            self.observation_timestamp_ns,
            self.sequence_id,
            self.policy_input.sha256,
            self.candidate_visible_input.sha256,
        )
@dataclass(frozen=True)
class EpisodeEnvelope:
    planned_episode_id: str; task_id: str; cell_id: str; cell_ordinal: int
    replicate_ordinal: int; candidate_ordinal: int; environment_seed: int; policy_rng_seed: int
    acquisition_base_pose: tuple[float, float, float]
    candidate: CandidateEnvelope; legacy_candidate: CandidateEnvelope
    captures: tuple[CaptureEnvelope, ...]
@dataclass(frozen=True)
class ValidationCounts:
    episode_count: int; capture_count: int; v2_candidate_count: int
    legacy_candidate_count: int; capture_blob_count: int
    p79_artifact_count: int; p50_artifact_count: int; p50_input_file_count: int
    def as_dict(self) -> dict[str, int]:
        return asdict(self)
@dataclass(frozen=True)
class CandidateArtifactEnvelope:
    bank_schema: str; proposal_id: str; producer_commit: str; artifact_commit: str
    bank_root_role: RootRole
    legacy_root_role: RootRole; episodes: tuple[EpisodeEnvelope, ...]
    validation_counts: ValidationCounts; artifact_tree: str
    bank_git_blob: str; manifest_git_blob: str
    def canonical_receipt_bytes(self) -> bytes:
        value = {
            "schema_version": "hwr.p80-candidate-artifact-receipt/v1",
            "bank_schema": self.bank_schema, "proposal_id": self.proposal_id,
            "producer_commit": self.producer_commit,
            "artifact_commit": self.artifact_commit,
            "bank_root_role": self.bank_root_role.value,
            "legacy_root_role": self.legacy_root_role.value,
            "artifact_tree": self.artifact_tree, "bank_git_blob": self.bank_git_blob,
            "manifest_git_blob": self.manifest_git_blob,
            "validation_counts": self.validation_counts.as_dict(),
            "episodes": [_episode_receipt(value) for value in self.episodes],
        }
        return _canonical(value)
EXPECTED_PRODUCER_SOURCE_PATHS = (
    "src/hwr/apps/evaluate_candidate_mask_ownership.py",
    "src/hwr/eval/candidate_mask_ownership.py",
    "src/hwr/eval/target_selection.py",
)
FORMAL_TRUST_ANCHOR = CandidateArtifactTrustAnchor(
    artifact_commit="93ea4e7afad8c52d83abd54f41a2d08d40a3cab4",
    artifact_root="runs/research-loop/0014/r0014-p79-candidate-bank-s20267901",
    artifact_tree="9a78c75e1f26b2c80399626042252b4e87404169",
    bank=GitBlobAnchor("bank.json", "471d7fbc526ac1c73b1efdafd03c9f073bcf3e5c",
        "888bf3ee42854a726bd86cc6c703c0c33a74f72d9029dc2cb6a9824ae9becd8e", 328423),
    manifest=GitBlobAnchor("manifest.json",
        "5d99ddc39e475c98e8eb3a64132f52192ed84061",
        "162e4bb6d06daf8e53e7192385274f059adde1a254bf81f2f4f8750ccce8d9c9", 816791),
    bank_schema=P79_BANK_SCHEMA, manifest_schema=P79_MANIFEST_SCHEMA,
    candidate_schema=V2_CANDIDATE_SCHEMA, proposal_id=P79_PROPOSAL_ID,
    expected_episodes=24, expected_captures=384, expected_p79_artifacts=28,
    producer_commit="9eef9953f8a8558228a5e8870d7d2d8f7499ee1e",
    producer_sources=(
        GitBlobAnchor("src/hwr/apps/evaluate_candidate_mask_ownership.py",
            "30759570f978eb73e612515e4e0c256f3f374dcf",
            "70cc24d20ba00a79f1694882005602b1cca8807c7e4f8fc5f082aa7e59e455a8",
            35340),
        GitBlobAnchor("src/hwr/eval/candidate_mask_ownership.py",
            "3d3839605eb290f9f2e0b77ec7db22ac7de15a31",
            "9bcf9eaa45238f3053022010158188b642478185c39ab24976130e4cd4fd6c9a",
            33530),
        GitBlobAnchor("src/hwr/eval/target_selection.py",
            "d7e588ba76ce18882255e3e22b1f86459ab235dd",
            "54961b5e84f29d58efe01ccfe24d04ffcf04b76a36697aac8d24a431ec4c9b4a",
            28661)),
    legacy_root="runs/research-loop/0010/r0010-p50-e1-acquisition-s20265001",
    legacy_manifest_bytes=186310,
    legacy_manifest_sha256="cefb4855305103fe6b205446295372d0e1060ba3e0f0e90d740264f715350d86",
    legacy_source_commit=P50_SOURCE_COMMIT,
    legacy_manifest_schema=P50_MANIFEST_SCHEMA,
    legacy_candidate_schema=LEGACY_CANDIDATE_SCHEMA,
    expected_p50_artifacts=795,
    frozen_document_commit="f224149e0a5ab0ae3cea981e669dd661d7d64ffe",
    frozen_document_path="docs/research-loop/0015/03-experiment.md",
    frozen_document_blob="768c1fb309ea662f72319cc9688b1f50ce7eeada",
    frozen_document_sha256="1672848c3c0907eef56af4ca56d98adcc540ce242b2b7b55ea4eda2d41b0a153")
def candidate_artifact_anchor_fingerprint(anchor: CandidateArtifactTrustAnchor) -> str:
    return _sha256(_canonical(asdict(anchor)))
FORMAL_TRUST_ANCHOR_FINGERPRINT = "fa2c707573e4067f2bfb37848669263db644effc8fc8c6a031654c670253182e"
def resolve_candidate_artifact(
    repository: Path, *, p79_v2_bank: Path, p50_legacy_source: Path,
    trust_anchor: CandidateArtifactTrustAnchor,
) -> CandidateArtifactEnvelope:
    root = repository.resolve()
    bank_root = _explicit_root(root, p79_v2_bank, RootRole.P79_V2_BANK)
    legacy_root = _explicit_root(root, p50_legacy_source,
                                 RootRole.P50_LEGACY_SOURCE)
    _validate_anchor_shape(trust_anchor, FORMAL_TRUST_ANCHOR_FINGERPRINT)
    _validate_root_binding(root, bank_root, legacy_root, trust_anchor)
    _validate_git_anchor(root, trust_anchor)
    bank = _object(_read_top_level(
        bank_root, trust_anchor.bank, RootRole.P79_V2_BANK), "P79 bank")
    manifest = _object(_read_top_level(
        bank_root, trust_anchor.manifest, RootRole.P79_V2_BANK), "P79 manifest")
    _validate_outer_documents(bank, manifest, trust_anchor)
    p79_blobs = _validate_p79_artifacts(bank_root, manifest, trust_anchor)
    input_blobs = _validate_p50_inputs(legacy_root, manifest, trust_anchor)
    legacy_manifest = _legacy_manifest(legacy_root, input_blobs, trust_anchor)
    capsules = _object(input_blobs[
        _legacy_path(trust_anchor, "capsules.json")].content, "P50 capsules")
    regression = _object(p79_blobs["regression.json"].content, "P79 regression")
    episodes = _resolve_episodes(bank, capsules, regression, p79_blobs,
                                 input_blobs, trust_anchor)
    captures = tuple(capture for episode in episodes for capture in episode.captures)
    counts = ValidationCounts(
        len(episodes), len(captures), len(episodes), len(episodes),
        2 * len(captures), len(p79_blobs),
        len(legacy_manifest["artifacts"]), len(input_blobs))
    _validate_counts(counts, trust_anchor)
    _require(len({value.composite_identity for value in captures}) == len(captures),
             "capture composite identities are not unique")
    return CandidateArtifactEnvelope(
        trust_anchor.bank_schema, trust_anchor.proposal_id,
        trust_anchor.producer_commit, trust_anchor.artifact_commit,
        RootRole.P79_V2_BANK, RootRole.P50_LEGACY_SOURCE, episodes, counts,
        trust_anchor.artifact_tree, trust_anchor.bank.git_blob,
        trust_anchor.manifest.git_blob)
def _validate_anchor_shape(anchor: CandidateArtifactTrustAnchor,
                           expected_fingerprint: str) -> None:
    expected = {
        "bank_schema": P79_BANK_SCHEMA,
        "manifest_schema": P79_MANIFEST_SCHEMA,
        "candidate_schema": V2_CANDIDATE_SCHEMA,
        "proposal_id": P79_PROPOSAL_ID,
        "legacy_manifest_schema": P50_MANIFEST_SCHEMA,
        "legacy_candidate_schema": LEGACY_CANDIDATE_SCHEMA,
    }
    failed = [name for name, value in expected.items()
              if getattr(anchor, name) != value]
    _require(not failed, f"external trust anchor drifted: {', '.join(failed)}",
             "anchor_shape")
    _require(tuple(value.path for value in anchor.producer_sources)
             == EXPECTED_PRODUCER_SOURCE_PATHS,
             "producer source anchors must equal the frozen three-source set",
             "producer_source_set")
    _require(candidate_artifact_anchor_fingerprint(anchor)
             == expected_fingerprint,
             "external trust anchor fingerprint differs", "anchor_fingerprint")
def _validate_root_binding(root: Path, bank_root: Path, legacy_root: Path,
                           anchor: CandidateArtifactTrustAnchor) -> None:
    expected_bank = _explicit_root(root, Path(anchor.artifact_root),
                                   RootRole.P79_V2_BANK)
    expected_legacy = _explicit_root(root, Path(anchor.legacy_root),
                                     RootRole.P50_LEGACY_SOURCE)
    _require(bank_root == expected_bank and legacy_root == expected_legacy,
             "explicit artifact root role drifted", "root_binding")
    _require(bank_root != legacy_root
             and not bank_root.is_relative_to(legacy_root)
             and not legacy_root.is_relative_to(bank_root),
             "artifact root roles overlap", "root_overlap")
def _validate_git_anchor(root: Path,
                         anchor: CandidateArtifactTrustAnchor) -> None:
    for commit in (anchor.artifact_commit, anchor.producer_commit,
                   anchor.frozen_document_commit):
        _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    _require(_is_ancestor(root, anchor.artifact_commit, "HEAD"),
             "HEAD does not contain artifact commit", "git_anchor")
    _require(_is_ancestor(root, anchor.producer_commit, anchor.artifact_commit),
             "artifact commit does not contain producer commit", "git_anchor")
    _require(_is_ancestor(root, anchor.frozen_document_commit, "HEAD"),
             "HEAD does not contain frozen document commit", "git_anchor")
    tree = _git(root, "rev-parse",
                f"{anchor.artifact_commit}:{anchor.artifact_root}")
    _require(tree == anchor.artifact_tree, "artifact Git tree drifted",
             "git_anchor")
    for identity in (anchor.bank, anchor.manifest):
        blob = _git(root, "rev-parse", f"{anchor.artifact_commit}:"
                    f"{anchor.artifact_root}/{identity.path}")
        _require(blob == identity.git_blob, "artifact Git blob drifted",
                 "git_anchor")
    for source in anchor.producer_sources:
        blob = _git(root, "rev-parse",
                    f"{anchor.producer_commit}:{source.path}")
        content = _git_bytes(root, "show",
                             f"{anchor.producer_commit}:{source.path}")
        _require(blob == source.git_blob
                 and _git(root, "rev-parse", f"HEAD:{source.path}") == source.git_blob
                 and _git(root, "hash-object", "--", source.path) == source.git_blob
                 and len(content) == source.byte_count
                 and _sha256(content) == source.sha256,
                 "producer source blob drifted", "producer_source")
    specification = (f"{anchor.frozen_document_commit}:"
                     f"{anchor.frozen_document_path}")
    frozen_blob = _git(root, "rev-parse", specification)
    frozen = _git_bytes(root, "show", specification)
    _require(frozen_blob == anchor.frozen_document_blob
             and _sha256(frozen) == anchor.frozen_document_sha256,
             "frozen document anchor drifted", "frozen_document")
    _validate_anchor_is_documented(frozen, anchor)
def _validate_anchor_is_documented(document: bytes,
                                   anchor: CandidateArtifactTrustAnchor) -> None:
    required = (
        anchor.artifact_commit, anchor.artifact_root, anchor.artifact_tree,
        anchor.bank.git_blob, anchor.bank.sha256, anchor.manifest.git_blob,
        anchor.manifest.sha256, anchor.bank_schema, anchor.manifest_schema,
        anchor.candidate_schema, anchor.proposal_id, anchor.producer_commit,
        anchor.legacy_root, str(anchor.legacy_manifest_bytes),
        anchor.legacy_manifest_sha256, anchor.legacy_source_commit,
        anchor.legacy_manifest_schema, anchor.legacy_candidate_schema,
    )
    source_values = tuple(item for source in anchor.producer_sources
                          for item in (source.path, source.git_blob,
                                       source.sha256))
    _require(all(value.encode() in document for value in
                 (*required, *source_values)),
             "trust anchor is not frozen in the experiment document")
def _validate_outer_documents(bank: Mapping[str, object],
                              manifest: Mapping[str, object],
                              anchor: CandidateArtifactTrustAnchor) -> None:
    checks = (
        (bank.get("schema_version"), anchor.bank_schema, "bank schema",
         "outer_schema"),
        (bank.get("proposal_id"), anchor.proposal_id, "bank proposal",
         "outer_schema"),
        (manifest.get("schema_version"), anchor.manifest_schema,
         "manifest schema", "outer_schema"),
        (manifest.get("proposal_id"), anchor.proposal_id,
         "manifest proposal", "outer_schema"),
        (manifest.get("source_commit"), anchor.producer_commit,
         "producer commit", "producer_commit"),
        (manifest.get("status"), "complete", "manifest status",
         "outer_schema"),
        (bank.get("source_acquisition"), anchor.legacy_root, "legacy root",
         "root_binding"),
    )
    for actual, expected, name, category in checks:
        _require(actual == expected, f"{name} differs", category)
    sources = manifest.get("provenance", {})
    sources = sources.get("sources") if isinstance(sources, Mapping) else None
    _require(isinstance(sources, Mapping),
             "producer source identities missing")
    for expected in anchor.producer_sources:
        value = sources.get(expected.path)
        _require(isinstance(value, Mapping)
                 and value.get("path") == expected.path
                 and value.get("bytes") == expected.byte_count
                 and value.get("sha256") == expected.sha256,
                 "producer source manifest identity drifted",
                 "producer_source")
def _validate_p79_artifacts(bank_root: Path, manifest: Mapping[str, object],
                            anchor: CandidateArtifactTrustAnchor
                            ) -> dict[str, BoundBlob]:
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, Mapping), "P79 artifacts are missing",
             "artifact_ledger")
    _require(len(artifacts) == anchor.expected_p79_artifacts,
             "P79 artifact count differs", "artifact_ledger")
    return {
        name: _bound_blob(bank_root, _identity(identity, path=name),
                          RootRole.P79_V2_BANK)
        for name, identity in sorted(artifacts.items())
    }
def _validate_p50_inputs(legacy_root: Path, manifest: Mapping[str, object],
                         anchor: CandidateArtifactTrustAnchor
                         ) -> dict[str, BoundBlob]:
    value = manifest.get("input")
    _require(isinstance(value, Mapping)
             and value.get("path") == anchor.legacy_root
             and isinstance(value.get("files"), list),
             "P50 input root or ledger differs", "root_binding")
    files = value["files"]
    result = {}
    for item in files:
        identity = _identity(item); path = str(identity["path"])
        expected_prefix = anchor.legacy_root + "/"
        _require(path.startswith(expected_prefix),
                 "P50 input root role differs", "root_binding")
        relative = path[len(expected_prefix):]
        result[path] = _bound_blob(legacy_root,
            {**identity, "path": relative}, RootRole.P50_LEGACY_SOURCE)
    _require(len(result) == len(files), "P50 input paths duplicate",
             "artifact_ledger")
    return result
def _legacy_manifest(
    legacy_root: Path, input_blobs: Mapping[str, BoundBlob],
    anchor: CandidateArtifactTrustAnchor,
) -> Mapping[str, object]:
    del legacy_root
    path = _legacy_path(anchor, "manifest.json")
    blob = input_blobs.get(path)
    _require(blob is not None
             and blob.byte_count == anchor.legacy_manifest_bytes
             and blob.sha256 == anchor.legacy_manifest_sha256,
             "P50 manifest anchor differs", "legacy_manifest")
    manifest = _object(blob.content, "P50 manifest")
    _require(manifest.get("schema_version") == anchor.legacy_manifest_schema
             and manifest.get("source_commit") == anchor.legacy_source_commit
             and manifest.get("status") == "complete",
             "P50 manifest lineage differs", "legacy_manifest")
    schemas = manifest.get("schemas")
    _require(isinstance(schemas, Mapping)
             and schemas.get("candidate") == anchor.legacy_candidate_schema,
             "legacy candidate schema differs", "candidate_schema")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, Mapping)
             and len(artifacts) == anchor.expected_p50_artifacts,
             "P50 artifact count differs", "artifact_ledger")
    for name, identity in artifacts.items():
        path = _legacy_path(anchor, str(name))
        blob = input_blobs.get(path); expected = _identity(identity, path=str(name))
        _require(blob is not None and blob.byte_count == expected["bytes"]
                 and blob.sha256 == expected["sha256"],
                 "P50 manifest artifact identity differs", "blob_identity")
    return manifest
def _resolve_episodes(
    bank: Mapping[str, object], capsules: Mapping[str, object],
    regression: Mapping[str, object],
    p79_blobs: Mapping[str, BoundBlob],
    input_blobs: Mapping[str, BoundBlob],
    anchor: CandidateArtifactTrustAnchor,
) -> tuple[EpisodeEnvelope, ...]:
    rows = bank.get("episodes"); capsule_rows = capsules.get("episodes")
    regression_rows = regression.get("records")
    _require(all(isinstance(value, list)
                 for value in (rows, capsule_rows, regression_rows)),
             "Episode ledgers are missing", "episode_ledger")
    _require(bank.get("episode_count") == anchor.expected_episodes
             and bank.get("capture_count") == anchor.expected_captures
             and capsules.get("capsule_count") == anchor.expected_episodes
             and regression.get("schema_version") == P79_REGRESSION_SCHEMA
             and regression.get("proposal_id") == anchor.proposal_id
             and regression.get("episode_count") == anchor.expected_episodes,
             "Episode cohort metadata differs", "episode_ledger")
    capsule_by_id = _unique_records(capsule_rows, "P50 capsule")
    regression_by_id = _unique_records(regression_rows, "P79 regression")
    identities = [str(value.get("planned_episode_id")) for value in rows]
    _require(len(identities) == len(set(identities)),
             "bank Episode identity duplicates", "episode_identity")
    _require(set(identities) == set(capsule_by_id) == set(regression_by_id),
             "Episode ledgers are not bijective", "episode_ledger")
    episodes = tuple(
        _resolve_episode(value,
            capsule_by_id[str(value["planned_episode_id"])],
            regression_by_id[str(value["planned_episode_id"])],
            p79_blobs, input_blobs, anchor) for value in rows)
    _require(sum(len(value.captures) for value in episodes)
             == anchor.expected_captures, "capture count differs")
    return episodes
def _resolve_episode(
    row: Mapping[str, object], capsule: Mapping[str, object],
    regression: Mapping[str, object], p79_blobs: Mapping[str, BoundBlob],
    input_blobs: Mapping[str, BoundBlob],
    anchor: CandidateArtifactTrustAnchor,
) -> EpisodeEnvelope:
    identity_fields = (
        "planned_episode_id", "task_id", "cell_id", "cell_ordinal",
        "replicate_ordinal", "candidate_ordinal", "environment_seed",
        "policy_rng_seed", "replacement", "acquisition_base_pose",
    )
    _require(not any(row.get(name) != capsule.get(name)
                     for name in identity_fields),
             "Episode identity metadata differs", "episode_identity")
    _require(capsule.get("schema_version") == P50_CAPSULE_SCHEMA,
             "P50 capsule schema differs", "episode_identity")
    _require(row.get("replacement") is False,
             "replacement Episode is forbidden", "episode_identity")
    _require(row.get("old_candidate_set", {}).get("generated_online") is True,
             "bank legacy generated_online differs", "selection_metadata")
    captures = _resolve_captures(row, capsule, input_blobs, anchor)
    acquisition_hashes = tuple(value.policy_input.sha256 for value in captures)
    candidate = _resolve_candidate(row.get("candidate_set"), p79_blobs,
        RootRole.P79_V2_BANK, anchor.candidate_schema,
        anchor.producer_commit, acquisition_hashes)
    legacy = _resolve_candidate(row.get("old_candidate_set"), input_blobs,
        RootRole.P50_LEGACY_SOURCE, anchor.legacy_candidate_schema,
        anchor.legacy_source_commit, acquisition_hashes,
        path_prefix=anchor.legacy_root + "/")
    _validate_capsule_candidate(capsule.get("candidate_set"), legacy)
    _validate_regression(regression, row, candidate, legacy, len(captures))
    return EpisodeEnvelope(
        planned_episode_id=str(row["planned_episode_id"]),
        task_id=str(row["task_id"]), cell_id=str(row["cell_id"]),
        cell_ordinal=int(row["cell_ordinal"]),
        replicate_ordinal=int(row["replicate_ordinal"]),
        candidate_ordinal=int(row["candidate_ordinal"]),
        environment_seed=int(row["environment_seed"]),
        policy_rng_seed=int(row["policy_rng_seed"]),
        acquisition_base_pose=tuple(float(value)
                                    for value in row["acquisition_base_pose"]),
        candidate=candidate, legacy_candidate=legacy, captures=captures,
    )
def _resolve_captures(
    row: Mapping[str, object], capsule: Mapping[str, object],
    input_blobs: Mapping[str, BoundBlob],
    anchor: CandidateArtifactTrustAnchor,
) -> tuple[CaptureEnvelope, ...]:
    captures = row.get("captures"); legacy_captures = capsule.get("captures")
    _require(isinstance(captures, list)
             and isinstance(legacy_captures, list),
             "ordered capture ledger is missing", "capture_ledger")
    _require(len(captures) == len(legacy_captures),
             "ordered capture ledger differs", "capture_ledger")
    ordinals = [int(value.get("capture_ordinal", -1)) for value in captures]
    _require(bool(captures) and ordinals == list(range(len(captures)))
             and sum(value.get("final_input") is True for value in captures) == 1
             and captures[-1].get("final_input") is True,
             "capture ordinal or final input differs", "capture_order")
    result = []
    for capture, legacy_capture in zip(captures, legacy_captures, strict=True):
        _require(isinstance(capture, Mapping)
                 and isinstance(legacy_capture, Mapping),
                 "capture row is invalid", "capture_ledger")
        comparable = {name: capture.get(name) for name in (
            "capture_ordinal", "final_input", "policy_input",
            "candidate_visible_input")}
        legacy_comparable = {name: legacy_capture.get(name) for name in comparable}
        observation = capture.get("observation_identity")
        legacy_observation = [legacy_capture.get("observation_timestamp_ns"),
                              legacy_capture.get("sequence_id")]
        _require(comparable == legacy_comparable
                 and observation == legacy_observation,
                 "ordered capture ledger differs", "capture_ledger")
        phase = legacy_capture.get("acquisition_phase")
        _require(legacy_capture.get("schema_version") == P50_CAPTURE_SCHEMA
                 and phase in ("A1_panorama", "A3_panorama", "A4_seal")
                 and bool(capture["final_input"]) == (phase == "A4_seal"),
                 "capture schema or acquisition phase differs",
                 "capture_contract")
        policy = _input_blob(input_blobs, capture.get("policy_input"), anchor)
        visible = _input_blob(input_blobs, capture.get("candidate_visible_input"),
                              anchor)
        _require(isinstance(observation, list) and len(observation) == 2,
                 "capture clock identity differs", "capture_contract")
        result.append(CaptureEnvelope(
            RootRole.P50_LEGACY_SOURCE, str(row["planned_episode_id"]),
            int(capture["capture_ordinal"]), bool(capture["final_input"]),
            int(observation[0]), int(observation[1]), P50_CAPTURE_SCHEMA,
            str(phase), policy, visible))
    return tuple(result)
def _resolve_candidate(
    value: object, blobs: Mapping[str, BoundBlob], role: RootRole,
    schema: str, producer_commit: str,
    acquisition_hashes: tuple[str, ...],
    *,
    path_prefix: str = "",
) -> CandidateEnvelope:
    _require(isinstance(value, Mapping), "candidate metadata is missing",
             "candidate_metadata")
    _require(value.get("schema_version") == schema,
             "outer candidate schema differs", "candidate_schema")
    relative = _safe_relative(str(value.get("path", "")))
    blob = blobs.get(path_prefix + relative)
    _require(blob is not None and blob.root_role is role,
             "candidate root role differs", "candidate_root")
    expected = _identity(value)
    _require(blob.byte_count == expected["bytes"]
             and blob.sha256 == expected["sha256"],
             "candidate blob identity differs", "blob_identity")
    document = _object(blob.content, "candidate document")
    candidates = document.get("candidates")
    _require(document.get("schema_version") == schema,
             "inner candidate schema differs", "candidate_schema")
    _require(_canonical(document) == blob.content,
             "candidate bytes are not canonical", "candidate_content")
    _require(isinstance(candidates, list)
             and document.get("candidate_count") == len(candidates)
             and value.get("candidate_count") == len(candidates),
             "candidate count differs", "candidate_content")
    _require(document.get("acquisition_input_sha256")
             == list(acquisition_hashes),
             "candidate acquisition input hash differs",
             "acquisition_binding")
    selected = int(value.get("selected_index", -2))
    _require(selected >= -1 and selected < len(candidates)
             and not (selected == -1 and candidates),
             "selected index is invalid", "selection_metadata")
    selected_identity = None if selected < 0 else _sha256(
        _canonical(candidates[selected]))
    _require(selected_identity == value.get("selected_canonical_identity"),
             "selected canonical identity differs", "selection_metadata")
    score_hash = str(value.get("score_bytes_sha256", ""))
    _require(_is_hex(score_hash, 64), "score hash is invalid",
             "selection_metadata")
    return CandidateEnvelope(
        role, schema, producer_commit, relative, blob, blob.content,
        len(candidates), selected, score_hash, selected_identity,
        acquisition_hashes)
def _validate_capsule_candidate(value: object,
                                candidate: CandidateEnvelope) -> None:
    _require(isinstance(value, Mapping),
             "P50 candidate metadata is missing", "selection_metadata")
    fields = {
        "schema_version": candidate.schema,
        "path": candidate.path,
        "sha256": candidate.blob.sha256,
        "bytes": candidate.blob.byte_count,
        "candidate_count": candidate.candidate_count,
        "selected_index": candidate.bound_selected_index,
        "score_bytes_sha256": candidate.bound_score_bytes_sha256,
        "generated_online": True,
    }
    _require(not any(value.get(name) != expected
                     for name, expected in fields.items()),
             "P50 candidate ledger differs", "selection_metadata")
def _validate_regression(value: Mapping[str, object], bank: Mapping[str, object],
                         candidate: CandidateEnvelope, legacy: CandidateEnvelope,
                         capture_count: int) -> None:
    _require(value.get("task_id") == bank.get("task_id"),
             "P79 regression task identity differs", "regression_audit")
    for name, envelope in (("new", candidate), ("old", legacy)):
        row = value.get(name); expected = {
            "schema_version": envelope.schema,
            "candidate_count": envelope.candidate_count,
            "candidate_set_sha256": envelope.blob.sha256,
            "selected_index": envelope.bound_selected_index,
            "selected_canonical_identity": envelope.bound_selected_canonical_identity,
        }
        _require(isinstance(row, Mapping) and not any(
            row.get(field) != item for field, item in expected.items()),
            f"P79 regression {name} candidate ledger differs",
            "selection_metadata")
    audit = value.get("audit"); valid_audit = isinstance(audit, Mapping)
    traversals = audit.get("traversals") if valid_audit else None
    checks = audit.get("checks") if valid_audit else None
    required_checks = (
        "input_head_depth_valid_byte_identical", "oracle_parent_mask_mutation_count_zero",
        "oracle_traversal_raw_multisets_equal", "production_final_equals_oracle_row_major",
        "production_parent_valid_byte_identical", "production_row_major_equals_oracle",
        "traversal_final_candidate_bytes_equal",
    )
    _require(valid_audit
             and audit.get("schema_version") == P79_AUDIT_SCHEMA
             and audit.get("proposal_id") == P79_PROPOSAL_ID
             and audit.get("decision")
                 == "accepted as deterministic candidate-generator correction"
             and audit.get("frame_count") == capture_count
             and isinstance(audit.get("frames"), list)
             and len(audit["frames"]) == capture_count
             and [item.get("frame_ordinal") for item in audit["frames"]]
                 == list(range(capture_count))
             and [item.get("observation_identity") for item in audit["frames"]]
                 == [item.get("observation_identity") for item in bank["captures"]]
             and isinstance(checks, Mapping) and checks.get("passed") is True
             and all(checks.get(name) is True for name in required_checks)
             and isinstance(traversals, Mapping)
             and set(traversals)
                 == {"row_major", "reverse_row_major", "column_major"}
             and all(item.get("candidate_set_sha256")
                     == candidate.blob.sha256
                     and item.get("candidate_count") == candidate.candidate_count
                     for item in traversals.values()),
             "P79 regression audit binding differs", "regression_audit")
def _validate_counts(counts: ValidationCounts, anchor: CandidateArtifactTrustAnchor) -> None:
    expected = ValidationCounts(
        anchor.expected_episodes, anchor.expected_captures,
        anchor.expected_episodes, anchor.expected_episodes,
        2 * anchor.expected_captures, anchor.expected_p79_artifacts,
        anchor.expected_p50_artifacts, anchor.expected_p50_artifacts + 1)
    _require(counts == expected, "complete validation counts differ")
def _bound_blob(
    root: Path, identity: Mapping[str, object], role: RootRole
) -> BoundBlob:
    path = _safe_relative(str(identity["path"]))
    before = _path_snapshot(root, path)
    try:
        content = read_bound_blob(root, identity)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise CandidateArtifactContractError(
            "blob_identity", f"{role.value} bound blob validation failed") from error
    _require(_path_snapshot(root, path) == before,
             "artifact path changed while reading", "path_safety")
    return BoundBlob(role, path, int(identity["bytes"]),
                     str(identity["sha256"]), content)
def _read_top_level(
    root: Path, anchor: GitBlobAnchor, role: RootRole
) -> bytes:
    path = _safe_relative(anchor.path)
    before = _path_snapshot(root, path)
    try:
        content = read_bound_blob(root, {"path": path,
            "bytes": anchor.byte_count, "sha256": anchor.sha256})
    except (OSError, ValueError, RuntimeError) as error:
        raise CandidateArtifactContractError(
            "git_anchor", f"{role.value} top-level anchor differs") from error
    _require(_path_snapshot(root, path) == before,
             "top-level artifact changed while reading", "path_safety")
    return content
def _input_blob(blobs: Mapping[str, BoundBlob], value: object,
                anchor: CandidateArtifactTrustAnchor) -> BoundBlob:
    identity = _identity(value)
    relative = _safe_relative(str(identity["path"]))
    blob = blobs.get(_legacy_path(anchor, relative))
    _require(blob is not None and blob.root_role is RootRole.P50_LEGACY_SOURCE
             and blob.byte_count == identity["bytes"]
             and blob.sha256 == identity["sha256"],
             "capture input identity differs", "capture_root")
    return blob
def _identity(value: object, *, path: str | None = None) -> dict[str, object]:
    _require(isinstance(value, Mapping), "blob identity is missing",
             "blob_identity")
    result = {
        "path": str(value.get("path") if path is None else path),
        "bytes": value.get("bytes"),
        "sha256": value.get("sha256"),
    }
    _require(isinstance(result["bytes"], int) and result["bytes"] >= 0
             and isinstance(result["sha256"], str)
             and _is_hex(result["sha256"], 64),
             "blob identity is invalid", "blob_identity")
    _safe_relative(result["path"])
    return result
def _explicit_root(root: Path, value: Path, role: RootRole) -> Path:
    _require(not value.is_absolute(),
             f"{role.value} root must be repository-relative", "path_safety")
    relative = _safe_relative(value.as_posix())
    path = root / relative
    _reject_symlinks(root, relative, include_leaf=True)
    resolved = path.resolve()
    _require(resolved.is_relative_to(root),
             f"{role.value} root escaped repository", "path_safety")
    _require(resolved.is_dir(), f"{role.value} root is not a directory",
             "root_binding")
    return resolved
def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    _require(bool(value) and not path.is_absolute()
             and ".." not in path.parts and "." not in path.parts
             and value == path.as_posix(),
             "artifact path is not canonical relative", "path_safety")
    return value
def _reject_symlinks(root: Path, relative: str,
                     *, include_leaf: bool = True) -> None:
    path = root
    parts = PurePosixPath(relative).parts
    limit = len(parts) if include_leaf else max(0, len(parts) - 1)
    for part in parts[:limit]:
        path = path / part
        _require(not path.is_symlink(), "artifact symlink is forbidden",
                 "path_safety")
def _path_snapshot(root: Path, relative: str) -> tuple[tuple[int, int, int], ...]:
    _reject_symlinks(root, relative)
    path = root; result = []
    try:
        for part in PurePosixPath(relative).parts:
            path = path / part; value = path.lstat()
            result.append((value.st_dev, value.st_ino, value.st_mode))
    except OSError as error:
        raise CandidateArtifactContractError(
            "path_safety", "artifact path lstat failed") from error
    return tuple(result)
def _object(content: bytes, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateArtifactContractError(
            "document", f"{name} is not valid JSON") from error
    _require(isinstance(value, dict), f"{name} must contain an object")
    return value
def _unique_records(values: Sequence[object],
                    name: str) -> dict[str, Mapping[str, object]]:
    result = {}
    for value in values:
        _require(isinstance(value, Mapping), f"{name} row is invalid")
        identity = str(value.get("planned_episode_id", ""))
        _require(bool(identity) and identity not in result,
                 f"{name} identity duplicates")
        result[identity] = value
    return result
def _episode_receipt(episode: EpisodeEnvelope) -> dict[str, object]:
    return {
        "planned_episode_id": episode.planned_episode_id, "task_id": episode.task_id,
        "cell_id": episode.cell_id, "cell_ordinal": episode.cell_ordinal,
        "replicate_ordinal": episode.replicate_ordinal,
        "candidate_ordinal": episode.candidate_ordinal,
        "environment_seed": episode.environment_seed,
        "policy_rng_seed": episode.policy_rng_seed,
        "acquisition_base_pose": list(episode.acquisition_base_pose),
        "candidate": _candidate_receipt(episode.candidate),
        "legacy_candidate": _candidate_receipt(episode.legacy_candidate),
        "captures": [
            {
                "root_role": value.root_role.value,
                "capture_ordinal": value.capture_ordinal, "final_input": value.final_input,
                "schema": value.schema, "acquisition_phase": value.acquisition_phase,
                "observation_timestamp_ns": value.observation_timestamp_ns,
                "sequence_id": value.sequence_id,
                "policy_input_sha256": value.policy_input.sha256,
                "candidate_visible_input_sha256": value.candidate_visible_input.sha256,
            }
            for value in episode.captures
        ],
    }
def _candidate_receipt(value: CandidateEnvelope) -> dict[str, object]:
    return {
        "root_role": value.root_role.value, "schema": value.schema,
        "producer_commit": value.producer_commit, "path": value.path,
        "bytes": value.blob.byte_count, "sha256": value.blob.sha256,
        "candidate_count": value.candidate_count,
        "bound_selected_index": value.bound_selected_index,
        "bound_score_bytes_sha256": value.bound_score_bytes_sha256,
        "bound_selected_canonical_identity": value.bound_selected_canonical_identity,
        "selection_relation_validated": value.selection_relation_validated,
        "acquisition_input_sha256": list(value.acquisition_input_sha256),
    }
def _legacy_path(anchor: CandidateArtifactTrustAnchor, path: str) -> str:
    return f"{anchor.legacy_root}/{_safe_relative(path)}"
def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", ancestor,
        descendant), cwd=root, check=False, capture_output=True).returncode == 0
def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return _is_ancestor(root, ancestor, descendant)
def git_text(root: Path, *arguments: str) -> str: return _git(root, *arguments)
def git_bytes(root: Path, *arguments: str) -> bytes:
    return _git_bytes(root, *arguments)
def repository_root(module_file: str) -> Path:
    return Path(module_file).resolve().parents[3]
def repository_path(root: Path, value: object) -> Path:
    path = Path(value); return path.resolve() if path.is_absolute() else (root / path).resolve()
def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(("git", *arguments), cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise CandidateArtifactContractError(
            "git_anchor", "Git trust anchor lookup failed") from error
def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(("git", *arguments), cwd=root, check=True,
            capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise CandidateArtifactContractError(
            "git_anchor", "Git trust anchor read failed") from error
def _canonical(value: object) -> bytes: return json.dumps(
    value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()
def _is_hex(value: str, length: int) -> bool: return len(value) == length and all(character in "0123456789abcdef" for character in value)
