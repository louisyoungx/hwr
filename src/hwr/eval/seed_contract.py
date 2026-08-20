"""Opaque Episode identities and domain-separated evaluation seeds."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence


SEED_SCHEMA = "hwr.opaque-episode-seeds/v1"
INT63_LIMIT = 1 << 63
EnvironmentSeedMode = Literal["derived", "compatibility"]


@dataclass(frozen=True)
class PlannedEpisodeSeed:
    plan_id: str
    task_id: str
    ablation: str
    episode_ordinal: int
    planned_episode_id: str
    environment_seed: int
    policy_rng_seed: int
    seed_commitment: str
    environment_seed_mode: EnvironmentSeedMode
    seed_schema: str = SEED_SCHEMA

    def __post_init__(self) -> None:
        if self.seed_schema != SEED_SCHEMA:
            raise ValueError("planned Episode seed schema differs")
        if not self.plan_id or not self.task_id or not self.ablation:
            raise ValueError("planned Episode identity fields cannot be empty")
        if self.episode_ordinal < 0:
            raise ValueError("planned Episode ordinal cannot be negative")
        expected = planned_episode_id(
            self.plan_id,
            self.task_id,
            self.ablation,
            self.episode_ordinal,
            schema=self.seed_schema,
        )
        if not hmac.compare_digest(self.planned_episode_id, expected):
            raise ValueError("planned Episode identity differs")
        if not _is_sha256(self.seed_commitment):
            raise ValueError("seed commitment must be a SHA-256 identity")
        if self.environment_seed_mode not in ("derived", "compatibility"):
            raise ValueError("unknown environment seed mode")
        if not _is_int63(self.environment_seed) or not _is_int63(
            self.policy_rng_seed
        ):
            raise ValueError("Episode seeds must be int63 values")
        if self.environment_seed == self.policy_rng_seed:
            raise ValueError("environment and policy seed domains collided")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def planned_episode_id(
    plan_id: str,
    task_id: str,
    ablation: str,
    episode_ordinal: int,
    *,
    schema: str = SEED_SCHEMA,
) -> str:
    if not schema or not plan_id or not task_id or not ablation:
        raise ValueError("planned Episode identity fields cannot be empty")
    if episode_ordinal < 0:
        raise ValueError("planned Episode ordinal cannot be negative")
    payload = (
        schema
        + plan_id
        + task_id
        + ablation
        + str(episode_ordinal)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_commitment(salt: str) -> str:
    return hashlib.sha256(_salt_bytes(salt)).hexdigest()


def verify_seed_reveal(commitment: str, salt: str) -> bool:
    if not _is_sha256(commitment):
        raise ValueError("seed commitment must be a SHA-256 identity")
    return hmac.compare_digest(commitment, seed_commitment(salt))


def require_seed_reveal(commitment: str, salt: str) -> None:
    if not verify_seed_reveal(commitment, salt):
        raise ValueError("seed reveal does not match its commitment")


def derive_domain_seed(
    salt: str, domain: Literal["environment", "policy"], episode_id: str
) -> int:
    if domain not in ("environment", "policy"):
        raise ValueError("unknown seed derivation domain")
    if not _is_sha256(episode_id):
        raise ValueError("planned Episode identity must be SHA-256")
    digest = hashlib.sha256(
        _salt_bytes(salt) + domain.encode("ascii") + episode_id.encode("ascii")
    ).digest()
    return int.from_bytes(digest, "big") & (INT63_LIMIT - 1)


def plan_episode_seeds(
    plan_id: str,
    task_id: str,
    ablation: str,
    episode_count: int,
    salt: str,
    *,
    environment_seeds: Sequence[int] | None = None,
) -> tuple[PlannedEpisodeSeed, ...]:
    if episode_count <= 0:
        raise ValueError("seed plan requires at least one Episode")
    if environment_seeds is not None and len(environment_seeds) != episode_count:
        raise ValueError("compatibility environment seed count differs")
    commitment = seed_commitment(salt)
    planned: list[PlannedEpisodeSeed] = []
    for ordinal in range(episode_count):
        identity = planned_episode_id(plan_id, task_id, ablation, ordinal)
        policy_seed = derive_domain_seed(salt, "policy", identity)
        environment_seed = (
            derive_domain_seed(salt, "environment", identity)
            if environment_seeds is None
            else int(environment_seeds[ordinal])
        )
        planned.append(
            PlannedEpisodeSeed(
                plan_id=plan_id,
                task_id=task_id,
                ablation=ablation,
                episode_ordinal=ordinal,
                planned_episode_id=identity,
                environment_seed=environment_seed,
                policy_rng_seed=policy_seed,
                seed_commitment=commitment,
                environment_seed_mode=(
                    "derived" if environment_seeds is None else "compatibility"
                ),
            )
        )
    result = tuple(planned)
    validate_episode_seed_plan(result, salt=salt)
    return result


def validate_episode_seed_plan(
    episodes: Sequence[PlannedEpisodeSeed],
    *,
    salt: str | None = None,
    plan_id: str | None = None,
    task_id: str | None = None,
    ablation: str | None = None,
) -> None:
    if not episodes:
        raise ValueError("Episode seed plan cannot be empty")
    if any(not isinstance(episode, PlannedEpisodeSeed) for episode in episodes):
        raise TypeError("Episode seed plan requires planned seed records")
    identities = [episode.planned_episode_id for episode in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError("planned Episode identities must be unique")
    policy_seeds = [episode.policy_rng_seed for episode in episodes]
    if len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("policy RNG seeds collided across the plan")
    derived_environment_seeds = [
        episode.environment_seed
        for episode in episodes
        if episode.environment_seed_mode == "derived"
    ]
    if len(set(derived_environment_seeds)) != len(derived_environment_seeds):
        raise ValueError("derived environment seeds collided across the plan")
    if {
        episode.environment_seed for episode in episodes
    } & set(policy_seeds):
        raise ValueError("environment and policy seed domains collided")
    commitments = {episode.seed_commitment for episode in episodes}
    if len(commitments) != 1:
        raise ValueError("Episode seed plan commitments differ")
    groups: dict[tuple[str, str, str], set[int]] = {}
    for episode in episodes:
        group = (episode.plan_id, episode.task_id, episode.ablation)
        ordinals = groups.setdefault(group, set())
        if episode.episode_ordinal in ordinals:
            raise ValueError("Episode seed plan repeats an ordinal")
        ordinals.add(episode.episode_ordinal)
    if any(ordinals != set(range(len(ordinals))) for ordinals in groups.values()):
        raise ValueError("Episode seed plan ordinals are incomplete")
    for group in groups:
        compatibility_seeds = [
            episode.environment_seed
            for episode in episodes
            if (
                episode.plan_id,
                episode.task_id,
                episode.ablation,
            )
            == group
            and episode.environment_seed_mode == "compatibility"
        ]
        if len(set(compatibility_seeds)) != len(compatibility_seeds):
            raise ValueError("compatibility environment seed sequence collided")
    for episode in episodes:
        if plan_id is not None and episode.plan_id != plan_id:
            raise ValueError("Episode seed plan identity differs")
        if task_id is not None and episode.task_id != task_id:
            raise ValueError("Episode seed plan task differs")
        if ablation is not None and episode.ablation != ablation:
            raise ValueError("Episode seed plan ablation differs")
        if salt is not None:
            require_seed_reveal(episode.seed_commitment, salt)
            expected_policy = derive_domain_seed(
                salt, "policy", episode.planned_episode_id
            )
            if episode.policy_rng_seed != expected_policy:
                raise ValueError("policy RNG seed derivation differs")
            if episode.environment_seed_mode == "derived":
                expected_environment = derive_domain_seed(
                    salt, "environment", episode.planned_episode_id
                )
                if episode.environment_seed != expected_environment:
                    raise ValueError("environment seed derivation differs")


def seed_lineage_manifest(
    plan_id: str,
    salt: str,
    episodes: Sequence[PlannedEpisodeSeed],
) -> dict[str, object]:
    validate_episode_seed_plan(episodes, salt=salt, plan_id=plan_id)
    modes = {episode.environment_seed_mode for episode in episodes}
    if len(modes) != 1:
        raise ValueError("seed lineage mixes environment seed modes")
    commitment = seed_commitment(salt)
    return {
        "schema_version": SEED_SCHEMA,
        "plan_id": plan_id,
        "commitment": {
            "algorithm": "SHA-256",
            "salt_sha256": commitment,
        },
        "derivation": {
            "planned_episode_id": (
                "SHA256(schema || plan_id || task_id || ablation || "
                "episode_ordinal)"
            ),
            "environment_seed": (
                "int63(SHA256(salt || environment || "
                "planned_episode_id))"
            ),
            "policy_rng_seed": (
                "int63(SHA256(salt || policy || "
                "planned_episode_id))"
            ),
        },
        "environment_seed_mode": next(iter(modes)),
        "reveal": {
            "salt": salt,
            "commitment_verified": verify_seed_reveal(commitment, salt),
        },
        "episodes": [episode.to_dict() for episode in episodes],
    }


def random_seed_salt() -> str:
    return secrets.token_hex(32)


def read_seed_salt(path: Path) -> str:
    salt = path.read_text(encoding="utf-8").strip()
    _salt_bytes(salt)
    if not _is_sha256(salt):
        raise ValueError("formal seed salt file must contain 256-bit lowercase hex")
    return salt


def _salt_bytes(salt: str) -> bytes:
    if not isinstance(salt, str) or not salt or salt != salt.strip():
        raise ValueError("seed salt must be a non-empty trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in salt):
        raise ValueError("seed salt cannot contain control characters")
    encoded = salt.encode("utf-8")
    if len(encoded) > 4_096:
        raise ValueError("seed salt is unreasonably large")
    return encoded


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _is_int63(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < INT63_LIMIT
