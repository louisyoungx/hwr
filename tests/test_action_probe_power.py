from __future__ import annotations

import hashlib

import numpy as np
import pytest

import hwr.eval.action_probe_power as power


TASKS = ("task-a/v1", "task-b/v1", "task-c/v1")
TRAINING_SEEDS = tuple(range(8))
HOLDOUT_SEEDS = tuple(range(100, 108))


def _episodes() -> tuple[power.PowerEpisode, ...]:
    result = []
    for rho_index, rho in enumerate((0.5, 0.96)):
        for task_index, task_id in enumerate(TASKS):
            for split, seeds in (
                ("training", TRAINING_SEEDS),
                ("holdout", HOLDOUT_SEEDS),
            ):
                for seed in seeds:
                    rng = np.random.default_rng(
                        seed + task_index * 10_009 + rho_index * 100_003
                    )
                    action = rng.standard_normal((power.POWER_TRANSITIONS, 16))
                    state = rng.standard_normal(
                        (power.POWER_TRANSITIONS + 1, 17)
                    )
                    digest = hashlib.sha256(
                        state.tobytes() + action.tobytes()
                    ).hexdigest()
                    result.append(
                        power.PowerEpisode(
                            f"rho-{rho}.{task_id}.{split}.seed-{seed}",
                            task_id,
                            split,
                            rho,
                            state,
                            action,
                            digest,
                        )
                    )
    return tuple(result)


def test_power_arms_keep_one_budget_and_distinguish_legal_rows() -> None:
    assert len(power.arm_indices("fragmented_7x16", 16)) == 7
    assert len(power.arm_indices("continuous_same_7_starts", 16)) == 7
    assert len(power.arm_indices("continuous_all_starts", 16)) == 97
    assert np.array_equal(
        power.arm_indices("fragmented_7x16", 16),
        power.arm_indices("continuous_same_7_starts", 16),
    )
    assert len(power.arm_indices("fragmented_7x16", 1)) == 112
    assert len(power.arm_indices("continuous_same_7_starts", 1)) == 7
    assert len(power.arm_indices("continuous_all_starts", 1)) == 112


def test_power_requires_complete_episode_plan() -> None:
    episodes = _episodes()

    with pytest.raises(ValueError, match="96"):
        power.run_action_probe_power(
            episodes[:-1], trials=1, bootstrap_samples=2
        )


def test_power_trial_targets_are_deterministic_and_zero_action_is_null() -> None:
    episodes = _episodes()

    first = power._trial_targets(episodes, 3, power.POWER_BASE_SEED)
    second = power._trial_targets(episodes, 3, power.POWER_BASE_SEED)

    assert set(first) == {"null", "zero_action", "planted", "permutation"}
    for key in first["null"]:
        np.testing.assert_array_equal(first["null"][key], second["null"][key])
        np.testing.assert_array_equal(
            first["zero_action"][key], first["null"][key]
        )
        assert not np.array_equal(
            first["planted"][key], first["permutation"][key]
        )


def test_power_report_is_reproducible_and_episode_clustered() -> None:
    episodes = _episodes()

    first = power.run_action_probe_power(
        episodes, trials=1, bootstrap_samples=3, base_seed=19
    )
    second = power.run_action_probe_power(
        episodes, trials=1, bootstrap_samples=3, base_seed=19
    )

    assert first == second
    assert first["mode"] == "smoke"
    assert first["decision"] == "smoke_only"
    assert first["bootstrap_contract"] == (
        power.ACTION_PROBE_BOOTSTRAP_CONTRACT
    )
    assert first["content_check"][
        "same_112_transition_content_across_arms"
    ]
    assert set(first["arms"]) == set(power.POWER_ARMS)
    assert len(first["trials"]) == 1
    for arm in first["arms"].values():
        assert arm["null_false_positive_rate"] == 0.0
        assert arm["permutation_false_positive_rate"] == 0.0


def test_power_design_uses_episode_lengths_not_transition_bootstrap() -> None:
    designs = power._build_designs(_episodes())
    design = designs[
        ("continuous_all_starts", 0.96, "task-a/v1", 16)
    ]

    assert design.holdout_lengths == (97,) * 8
    assert len(design.holdout_refs) == 8
    assert design.state_model.rows == 8 * 97
    assert design.action_model.rows == 8 * 97
    assert design.action_model.columns == design.state_model.columns + 16


def test_power_episode_rejects_corrupt_content_identity() -> None:
    episode = _episodes()[0]

    with pytest.raises(ValueError, match="invalid"):
        power.PowerEpisode(
            episode.episode_id,
            episode.task_id,
            episode.split,
            episode.correlation,
            episode.state,
            episode.action[:-1],
            episode.content_sha256,
        )
    with pytest.raises(ValueError, match="invalid"):
        power.PowerEpisode(
            episode.episode_id,
            episode.task_id,
            episode.split,
            episode.correlation,
            episode.state,
            episode.action,
            "0" * 64,
        )
