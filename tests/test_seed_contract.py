from dataclasses import replace
from pathlib import Path

import pytest

from hwr.eval import (
    derive_domain_seed,
    plan_episode_seeds,
    planned_episode_id,
    random_seed_salt,
    require_seed_reveal,
    read_seed_salt,
    seed_commitment,
    seed_lineage_manifest,
    validate_episode_seed_plan,
    verify_seed_reveal,
)


SALT = "R0001-P39-E1-s20263901"


def test_frozen_diagnostic_commitment_and_domain_derivation_replay() -> None:
    identity = planned_episode_id("plan-a", "task-a/v1", "none", 0)
    first = plan_episode_seeds("plan-a", "task-a/v1", "none", 3, SALT)
    replay = plan_episode_seeds("plan-a", "task-a/v1", "none", 3, SALT)

    assert seed_commitment(SALT) == (
        "a94db502b86fd2c83a9096eb856b110de"
        "53158f588cc7496a60e4264fc190237"
    )
    assert first == replay
    assert first[0].planned_episode_id == identity
    assert first[0].environment_seed == derive_domain_seed(
        SALT, "environment", identity
    )
    assert first[0].policy_rng_seed == derive_domain_seed(
        SALT, "policy", identity
    )
    assert all(
        0 <= episode.environment_seed < 2**63
        and 0 <= episode.policy_rng_seed < 2**63
        and episode.environment_seed != episode.policy_rng_seed
        for episode in first
    )


def test_commitment_reveal_fails_closed() -> None:
    commitment = seed_commitment(SALT)

    assert verify_seed_reveal(commitment, SALT)
    assert not verify_seed_reveal(commitment, "different")
    with pytest.raises(ValueError, match="does not match"):
        require_seed_reveal(commitment, "different")
    with pytest.raises(ValueError, match="non-empty trimmed"):
        seed_commitment("")
    with pytest.raises(ValueError, match="control characters"):
        seed_commitment("bad\nsalt")


def test_compatibility_mode_preserves_environment_seed_sequence() -> None:
    legacy = (500, 105229, 209958)

    planned = plan_episode_seeds(
        "legacy-plan",
        "task-a/v1",
        "lock_left",
        len(legacy),
        SALT,
        environment_seeds=legacy,
    )

    assert tuple(episode.environment_seed for episode in planned) == legacy
    assert {episode.environment_seed_mode for episode in planned} == {
        "compatibility"
    }
    assert all(
        episode.policy_rng_seed != episode.environment_seed
        for episode in planned
    )


def test_invalid_duplicate_identity_and_seed_collision_fail_closed() -> None:
    planned = plan_episode_seeds("plan-a", "task-a/v1", "none", 2, SALT)

    with pytest.raises(ValueError, match="identities must be unique"):
        validate_episode_seed_plan((planned[0], planned[0]))
    with pytest.raises(ValueError, match="domains collided"):
        replace(planned[0], policy_rng_seed=planned[0].environment_seed)
    with pytest.raises(ValueError, match="identity differs"):
        replace(planned[0], planned_episode_id="0" * 64)
    with pytest.raises(ValueError, match="derived environment seeds collided"):
        validate_episode_seed_plan(
            (
                planned[0],
                replace(
                    planned[1],
                    environment_seed=planned[0].environment_seed,
                ),
            )
        )
    with pytest.raises(ValueError, match="seed domains collided"):
        validate_episode_seed_plan(
            (
                planned[0],
                replace(
                    planned[1],
                    policy_rng_seed=planned[0].environment_seed,
                ),
            )
        )
    with pytest.raises(TypeError, match="planned seed records"):
        validate_episode_seed_plan((1,))  # type: ignore[arg-type]


def test_compatibility_seed_values_can_repeat_across_task_sequences() -> None:
    first = plan_episode_seeds(
        "legacy-plan",
        "task-a/v1",
        "none",
        2,
        SALT,
        environment_seeds=(31, 32),
    )
    second = plan_episode_seeds(
        "legacy-plan",
        "task-b/v1",
        "none",
        2,
        SALT,
        environment_seeds=(31, 32),
    )

    validate_episode_seed_plan(first + second, salt=SALT)
    assert seed_lineage_manifest(
        "legacy-plan", SALT, first + second
    )["environment_seed_mode"] == "compatibility"


def test_tampered_derived_seed_and_reveal_are_rejected() -> None:
    planned = plan_episode_seeds("plan-a", "task-a/v1", "none", 1, SALT)

    with pytest.raises(ValueError, match="environment seed derivation differs"):
        validate_episode_seed_plan(
            (replace(planned[0], environment_seed=1),),
            salt=SALT,
        )
    with pytest.raises(ValueError, match="reveal does not match"):
        seed_lineage_manifest("plan-a", "different", planned)


def test_baseline_candidate_plans_pair_without_role_in_seed_identity() -> None:
    baseline = plan_episode_seeds(
        "paired-plan", "task-a/v1", "none", 4, SALT
    )
    candidate = plan_episode_seeds(
        "paired-plan", "task-a/v1", "none", 4, SALT
    )

    assert [
        (
            episode.planned_episode_id,
            episode.environment_seed,
            episode.policy_rng_seed,
        )
        for episode in baseline
    ] == [
        (
            episode.planned_episode_id,
            episode.environment_seed,
            episode.policy_rng_seed,
        )
        for episode in candidate
    ]


def test_formal_salt_file_requires_256_bit_lowercase_hex(tmp_path: Path) -> None:
    valid = tmp_path / "valid.salt"
    invalid = tmp_path / "invalid.salt"
    valid.write_text("a" * 64 + "\n", encoding="utf-8")
    invalid.write_text(SALT, encoding="utf-8")

    assert read_seed_salt(valid) == "a" * 64
    with pytest.raises(ValueError, match="256-bit lowercase hex"):
        read_seed_salt(invalid)


def test_random_formal_salt_has_256_bits_of_hex_entropy() -> None:
    first = random_seed_salt()
    second = random_seed_salt()

    assert len(first) == 64
    assert int(first, 16) >= 0
    assert first == first.lower()
    assert first != second
