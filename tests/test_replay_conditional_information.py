from __future__ import annotations

import copy

import numpy as np
import pytest

from hwr.eval.replay_conditional_information import (
    BootstrapPlan,
    EvaluationErrors,
    ReplayDataset,
    build_fold_manifest,
    evaluate_replay_conditional_information,
    summarize_errors,
    target_deltas,
    validate_fold_manifest,
)


TASKS = ("task-a/v1", "task-b/v1", "task-c/v1")


def _dataset(*, rows_per_source: int = 8, planted: bool = True) -> ReplayDataset:
    rng = np.random.default_rng(17)
    task_ids = []
    source_ids = []
    shard_ids = []
    shard_offsets = []
    states = []
    actions = []
    proposals = []
    rates = []
    configurations = []
    rewrites = []
    state_weight = rng.standard_normal((37, 17)) * 0.05
    action_weight = rng.standard_normal((16, 17)) * 0.4
    for task_index, task_id in enumerate(TASKS):
        for source_index in range(6):
            source = f"{task_id}-source-{source_index}"
            state = rng.standard_normal((rows_per_source, 37))
            action = rng.standard_normal((rows_per_source, 16))
            nuisance = state @ state_weight
            signal = action @ action_weight if planted else 0.0
            target = nuisance + signal + rng.standard_normal(
                (rows_per_source, 17)
            ) * 0.02
            task_ids.append(np.full(rows_per_source, task_id))
            source_ids.append(np.full(rows_per_source, source))
            shard_ids.append(
                np.asarray(
                    [
                        f"{source}-shard-{offset // 4}"
                        for offset in range(rows_per_source)
                    ]
                )
            )
            shard_offsets.append(
                np.asarray([offset % 4 for offset in range(rows_per_source)])
            )
            states.append(state)
            actions.append(action)
            proposals.append(np.zeros_like(action))
            rates.append(target[:, :16])
            configurations.append(target)
            rewrite = np.zeros(rows_per_source, bool)
            if (task_index + source_index) % 2 == 0:
                rewrite[-1] = True
            rewrites.append(rewrite)
    return ReplayDataset(
        task_ids=np.concatenate(task_ids),
        source_ids=np.concatenate(source_ids),
        shard_ids=np.concatenate(shard_ids),
        shard_offsets=np.concatenate(shard_offsets),
        state=np.concatenate(states),
        action=np.concatenate(actions),
        actor_proposal=np.concatenate(proposals),
        rate_target=np.concatenate(rates),
        configuration_target=np.concatenate(configurations),
        safety_rewrite=np.concatenate(rewrites),
    )


def test_target_indices_and_yaw_wrap_are_frozen() -> None:
    proprioception = np.zeros((2, 37), np.float64)
    proprioception[0] = np.arange(37)
    proprioception[1] = np.arange(37) + 10.0
    proprioception[0, 28] = np.pi - 0.1
    proprioception[1, 28] = -np.pi + 0.2

    rate, configuration = target_deltas(proprioception)

    assert rate.shape == (1, 16)
    assert configuration.shape == (1, 17)
    np.testing.assert_allclose(rate, 10.0)
    np.testing.assert_allclose(configuration[0, :-1], 10.0)
    assert configuration[0, -1] == pytest.approx(0.3)


def test_source_task_stratified_nested_folds_are_deterministic_and_disjoint() -> None:
    dataset = _dataset()

    first = build_fold_manifest(dataset)
    second = build_fold_manifest(dataset)

    assert first == second
    assert [fold["outer_fold"] for fold in first["outer_folds"]] == [0, 1, 2]
    for fold in first["outer_folds"]:
        assert {
            task: len(sources)
            for task, sources in fold["outer_test_by_task"].items()
        } == {task: 2 for task in TASKS}
        assert {
            task: len(sources)
            for task, sources in fold["outer_train_by_task"].items()
        } == {task: 4 for task in TASKS}
        for inner in fold["inner_folds"]:
            assert {
                task: len(sources)
                for task, sources in inner["validation_by_task"].items()
            } == {task: 2 for task in TASKS}

    leaked = copy.deepcopy(first)
    fold = leaked["outer_folds"][0]
    source = fold["outer_test_by_task"][TASKS[0]][0]
    fold["outer_train_by_task"][TASKS[0]].append(source)
    with pytest.raises(ValueError, match="outer train/test"):
        validate_fold_manifest(dataset, leaked)


def test_controller_history_is_zero_masked_and_never_crosses_shards() -> None:
    dataset = _dataset(rows_per_source=8)
    proposal = np.arange(dataset.actor_proposal.size, dtype=np.float64).reshape(
        dataset.actor_proposal.shape
    )
    action = proposal + 0.5
    dataset = ReplayDataset(
        task_ids=dataset.task_ids,
        source_ids=dataset.source_ids,
        shard_ids=dataset.shard_ids,
        shard_offsets=dataset.shard_offsets,
        state=dataset.state,
        action=action,
        actor_proposal=proposal,
        rate_target=dataset.rate_target,
        configuration_target=dataset.configuration_target,
        safety_rewrite=dataset.safety_rewrite,
    )

    features = dataset.controller_features()
    first = 0
    second = 1
    next_shard = 4

    assert features.shape[1] == 185
    np.testing.assert_array_equal(features[first, 53:181], 0.0)
    np.testing.assert_array_equal(features[first, 181:185], 0.0)
    np.testing.assert_array_equal(features[second, 53:69], proposal[first])
    np.testing.assert_array_equal(features[second, 69:85], action[first])
    np.testing.assert_array_equal(features[second, 181:185], (1, 0, 0, 0))
    np.testing.assert_array_equal(features[next_shard, 53:181], 0.0)
    np.testing.assert_array_equal(features[next_shard, 181:185], 0.0)


def test_double_oof_report_publishes_all_targets_strata_sources_and_folds() -> None:
    dataset = _dataset(rows_per_source=12)
    folds = build_fold_manifest(dataset)

    report, design = evaluate_replay_conditional_information(
        dataset, folds, bootstrap_samples=40, bootstrap_seed=13
    )

    assert report["comparison"]["shared_target_nuisance_baseline"] is True
    assert report["comparison"]["double_oof_residual"] is True
    assert set(report["target_families"]) == {"rate", "configuration"}
    assert set(report["target_families"]["rate"]["strata"]) == {
        "all",
        "safety_rewrite",
        "no_rewrite",
        "shard_prefix",
        "shard_interior",
    }
    rate = report["target_families"]["rate"]["strata"]["all"]
    assert rate["aggregate"]["ratio"] > 1.05
    assert rate["aggregate"]["bootstrap"]["mean_log_ratio_p05"] > 0.0
    assert set(rate["tasks"]) == set(TASKS)
    assert len(rate["sources"]) == 18
    assert set(rate["outer_folds"]) == {"0", "1", "2"}
    assert all(
        set(value["tasks"]) == set(TASKS)
        for value in rate["outer_folds"].values()
    )
    assert report["bootstrap"]["multiplicity_sha256"] == (
        report["controller_context_guard"]["strata"]["all"]["aggregate"][
            "bootstrap"
        ]["multiplicity_sha256"]
    )
    assert min(fold.effective_rank for fold in design.folds) > 6.0


def test_source_and_task_equal_weighting_and_synchronized_bootstrap() -> None:
    dataset = _dataset(rows_per_source=2)
    source_order = dataset.source_order
    source_values = {
        source: (1.0 + index, 1.0)
        for index, source in enumerate(source_order)
    }
    control = np.asarray(
        [source_values[source][0] for source in dataset.source_ids]
    )
    candidate = np.ones_like(control)
    fold_ids = np.asarray(
        [
            next(
                fold["outer_fold"]
                for fold in build_fold_manifest(dataset)["outer_folds"]
                if source in {
                    value
                    for values in fold["outer_test_by_task"].values()
                    for value in values
                }
            )
            for source in dataset.source_ids
        ]
    )
    errors = EvaluationErrors(control, candidate, fold_ids)
    plan = BootstrapPlan.build(dataset, errors, samples=25, seed=7)

    summary = summarize_errors(
        dataset, errors, plan, np.ones(len(dataset.state), bool)
    )
    expected_tasks = []
    for task in TASKS:
        values = [
            source_values[source][0]
            for source in source_order
            if source.startswith(task)
        ]
        expected_tasks.append(np.mean(values))

    assert summary["aggregate"]["control_mse"] == pytest.approx(
        np.mean(expected_tasks)
    )
    assert summary["aggregate"]["candidate_mse"] == 1.0
    assert summary["aggregate"]["bootstrap"]["finite_replicates"] == 25
    assert plan.identity == BootstrapPlan.build(
        dataset, errors, samples=25, seed=7
    ).identity
