"""Double-orthogonal conditional-information evaluation for retained Replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PROPOSAL_ID = "R0001-P32-E1"
REPORT_SCHEMA = "hwr.replay-conditional-information/v1"
FOLD_SCHEMA = "hwr.replay-conditional-information-folds/v1"
RIDGE = 1.0e-3
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_263_201
POWER_TRIALS = 200
POWER_BOOTSTRAP_SAMPLES = 1_000
POWER_BASE_SEED = 20_263_211
RATE_INDICES = np.asarray(
    (6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30),
    np.int64,
)
CONFIGURATION_INDICES = np.asarray(
    (0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28),
    np.int64,
)
FORMAL_TASK_SOURCE_COUNTS = {
    "clear_dining_table_3d/v1": 6,
    "store_kitchen_items_3d/v1": 6,
    "tidy_living_room_3d/v1": 12,
}
STRATA = (
    "all",
    "safety_rewrite",
    "no_rewrite",
    "shard_prefix",
    "shard_interior",
)


@dataclass(frozen=True)
class ReplayDataset:
    task_ids: np.ndarray
    source_ids: np.ndarray
    shard_ids: np.ndarray
    shard_offsets: np.ndarray
    state: np.ndarray
    action: np.ndarray
    actor_proposal: np.ndarray
    rate_target: np.ndarray
    configuration_target: np.ndarray
    safety_rewrite: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "task_ids": np.asarray(self.task_ids).astype(str),
            "source_ids": np.asarray(self.source_ids).astype(str),
            "shard_ids": np.asarray(self.shard_ids).astype(str),
            "shard_offsets": np.asarray(self.shard_offsets, np.int64),
            "state": np.asarray(self.state, np.float64),
            "action": np.asarray(self.action, np.float64),
            "actor_proposal": np.asarray(self.actor_proposal, np.float64),
            "rate_target": np.asarray(self.rate_target, np.float64),
            "configuration_target": np.asarray(
                self.configuration_target, np.float64
            ),
            "safety_rewrite": np.asarray(self.safety_rewrite, bool),
        }
        count = len(arrays["task_ids"])
        expected = {
            "source_ids": (count,),
            "shard_ids": (count,),
            "shard_offsets": (count,),
            "state": (count, 37),
            "action": (count, 16),
            "actor_proposal": (count, 16),
            "rate_target": (count, 16),
            "configuration_target": (count, 17),
            "safety_rewrite": (count,),
        }
        if count == 0 or any(arrays[name].shape != shape for name, shape in expected.items()):
            raise ValueError("conditional-information Replay shapes are invalid")
        numeric = (
            "state",
            "action",
            "actor_proposal",
            "rate_target",
            "configuration_target",
        )
        if any(not np.isfinite(arrays[name]).all() for name in numeric):
            raise ValueError("conditional-information Replay contains non-finite values")
        if any(not value for name in ("task_ids", "source_ids", "shard_ids") for value in arrays[name]):
            raise ValueError("conditional-information Replay identity is missing")
        source_tasks: dict[str, set[str]] = {}
        for task_id, source_id in zip(arrays["task_ids"], arrays["source_ids"]):
            source_tasks.setdefault(source_id, set()).add(task_id)
        if any(len(tasks) != 1 for tasks in source_tasks.values()):
            raise ValueError("one source Episode spans multiple tasks")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def source_order(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.source_ids)))

    def masks(self) -> dict[str, np.ndarray]:
        return {
            "all": np.ones(len(self.state), bool),
            "safety_rewrite": self.safety_rewrite,
            "no_rewrite": ~self.safety_rewrite,
            "shard_prefix": self.shard_offsets < 4,
            "shard_interior": self.shard_offsets >= 4,
        }

    def controller_features(self) -> np.ndarray:
        history = np.zeros((len(self.state), 4, 32), np.float64)
        available = np.zeros((len(self.state), 4), np.float64)
        row_by_position = {
            (shard, int(offset)): index
            for index, (shard, offset) in enumerate(
                zip(self.shard_ids, self.shard_offsets)
            )
        }
        for row, (shard, offset) in enumerate(
            zip(self.shard_ids, self.shard_offsets)
        ):
            for lag in range(1, 5):
                previous = row_by_position.get((shard, int(offset) - lag))
                if previous is None:
                    continue
                history[row, lag - 1, :16] = self.actor_proposal[previous]
                history[row, lag - 1, 16:] = self.action[previous]
                available[row, lag - 1] = 1.0
        return np.concatenate(
            (
                self.state,
                self.actor_proposal,
                history.reshape(len(self.state), 128),
                available,
            ),
            axis=1,
        )


def target_deltas(proprioception: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(proprioception, np.float64)
    if values.ndim != 2 or values.shape[1] != 37 or len(values) < 2:
        raise ValueError("visible proprioception shape is invalid")
    rate = values[1:, RATE_INDICES] - values[:-1, RATE_INDICES]
    configuration = (
        values[1:, CONFIGURATION_INDICES]
        - values[:-1, CONFIGURATION_INDICES]
    )
    yaw = values[1:, 28] - values[:-1, 28]
    configuration[:, -1] = np.arctan2(np.sin(yaw), np.cos(yaw))
    return rate, configuration


def build_fold_manifest(
    dataset: ReplayDataset, *, formal: bool = False
) -> dict[str, object]:
    sources_by_task = {
        task_id: sorted(set(dataset.source_ids[dataset.task_ids == task_id]))
        for task_id in sorted(set(dataset.task_ids))
    }
    counts = {task_id: len(values) for task_id, values in sources_by_task.items()}
    if formal and counts != FORMAL_TASK_SOURCE_COUNTS:
        raise ValueError("formal Replay task/source coverage differs")
    if any(count < 6 or count % 3 for count in counts.values()):
        raise ValueError("outer source folds cannot satisfy the frozen split")
    outer_assignment: dict[str, int] = {}
    for task_id, sources in sources_by_task.items():
        ordered = sorted(
            sources,
            key=lambda source: _digest(
                f"{PROPOSAL_ID}|outer-v1|{task_id}|{source}"
            ),
        )
        outer_assignment.update(
            {source: index % 3 for index, source in enumerate(ordered)}
        )
    folds = []
    for outer_fold in range(3):
        test = {
            task: [source for source in sources if outer_assignment[source] == outer_fold]
            for task, sources in sources_by_task.items()
        }
        train = {
            task: [source for source in sources if outer_assignment[source] != outer_fold]
            for task, sources in sources_by_task.items()
        }
        inner_assignment: dict[str, int] = {}
        for task_id, sources in train.items():
            if len(sources) < 2 or len(sources) % 2:
                raise ValueError("inner source folds cannot satisfy the frozen split")
            ordered = sorted(
                sources,
                key=lambda source: _digest(
                    f"{PROPOSAL_ID}|inner-v1|{outer_fold}|{task_id}|{source}"
                ),
            )
            inner_assignment.update(
                {source: index % 2 for index, source in enumerate(ordered)}
            )
        folds.append(
            {
                "outer_fold": outer_fold,
                "outer_train_by_task": train,
                "outer_test_by_task": test,
                "inner_folds": [
                    {
                        "inner_fold": inner_fold,
                        "train_by_task": {
                            task: [
                                source
                                for source in sources
                                if inner_assignment[source] != inner_fold
                            ]
                            for task, sources in train.items()
                        },
                        "validation_by_task": {
                            task: [
                                source
                                for source in sources
                                if inner_assignment[source] == inner_fold
                            ]
                            for task, sources in train.items()
                        },
                    }
                    for inner_fold in range(2)
                ],
            }
        )
    manifest = {
        "schema_version": FOLD_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "outer_hash_contract": (
            'SHA256("R0001-P32-E1|outer-v1|" + task_id + "|" + source_id)'
        ),
        "inner_hash_contract": (
            'SHA256("R0001-P32-E1|inner-v1|" + outer_fold + "|" '
            '+ task_id + "|" + source_id)'
        ),
        "source_counts_by_task": counts,
        "outer_folds": folds,
    }
    validate_fold_manifest(dataset, manifest)
    return manifest


def validate_fold_manifest(
    dataset: ReplayDataset, manifest: Mapping[str, object]
) -> None:
    if manifest.get("schema_version") != FOLD_SCHEMA:
        raise ValueError("conditional-information fold schema differs")
    all_sources = set(dataset.source_order)
    outer_seen: set[str] = set()
    folds = manifest.get("outer_folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("conditional-information outer fold count differs")
    for expected_fold, fold in enumerate(folds):
        if int(fold["outer_fold"]) != expected_fold:
            raise ValueError("conditional-information outer fold identity differs")
        train = _flatten_by_task(fold["outer_train_by_task"])
        test = _flatten_by_task(fold["outer_test_by_task"])
        if train & test or train | test != all_sources or outer_seen & test:
            raise ValueError("source crossed an outer train/test boundary")
        outer_seen |= test
        inner_seen: set[str] = set()
        for inner in fold["inner_folds"]:
            inner_train = _flatten_by_task(inner["train_by_task"])
            validation = _flatten_by_task(inner["validation_by_task"])
            if inner_train & validation or inner_train | validation != train:
                raise ValueError("source crossed an inner train/validation boundary")
            if inner_seen & validation:
                raise ValueError("source entered multiple inner validation folds")
            inner_seen |= validation
        if inner_seen != train:
            raise ValueError("inner OOF coverage is incomplete")
    if outer_seen != all_sources:
        raise ValueError("outer OOF coverage is incomplete")


@dataclass(frozen=True)
class _RidgeDesign:
    operator: np.ndarray
    evaluation: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def build(
        cls, training: np.ndarray, evaluation: np.ndarray
    ) -> "_RidgeDesign":
        training = np.asarray(training, np.float64)
        evaluation = np.asarray(evaluation, np.float64)
        if training.ndim != 2 or evaluation.ndim != 2 or not len(training):
            raise ValueError("ridge design is empty or malformed")
        mean = training.mean(axis=0)
        scale = training.std(axis=0)
        scale[scale < 1.0e-8] = 1.0
        fitted = np.column_stack(((training - mean) / scale, np.ones(len(training))))
        evaluated = np.column_stack(
            ((evaluation - mean) / scale, np.ones(len(evaluation)))
        )
        penalty = np.eye(fitted.shape[1], dtype=np.float64) * RIDGE
        penalty[-1, -1] = 0.0
        operator = np.linalg.solve(fitted.T @ fitted + penalty, fitted.T)
        return cls(operator, evaluated, mean, scale)

    def predict(self, training_target: np.ndarray) -> np.ndarray:
        value = self.evaluation @ (self.operator @ training_target)
        if not np.isfinite(value).all():
            raise ValueError("ridge prediction is non-finite")
        return value


@dataclass(frozen=True)
class _FoldDesign:
    outer_fold: int
    train: np.ndarray
    test: np.ndarray
    inner: tuple[tuple[np.ndarray, np.ndarray, _RidgeDesign], ...]
    outer: _RidgeDesign
    residual: _RidgeDesign
    action_oof: np.ndarray
    action_test: np.ndarray
    effective_rank: float


@dataclass(frozen=True)
class NestedDesign:
    feature_family: str
    dataset: ReplayDataset
    folds: tuple[_FoldDesign, ...]

    @classmethod
    def build(
        cls,
        dataset: ReplayDataset,
        fold_manifest: Mapping[str, object],
        *,
        controller_context: bool = False,
    ) -> "NestedDesign":
        validate_fold_manifest(dataset, fold_manifest)
        features = dataset.controller_features() if controller_context else dataset.state
        fold_designs = []
        for fold in fold_manifest["outer_folds"]:
            train = _rows_for_sources(dataset, _flatten_by_task(fold["outer_train_by_task"]))
            test = _rows_for_sources(dataset, _flatten_by_task(fold["outer_test_by_task"]))
            action_oof = np.full((len(dataset.state), 16), np.nan)
            inner_designs = []
            for inner in fold["inner_folds"]:
                inner_train = _rows_for_sources(
                    dataset, _flatten_by_task(inner["train_by_task"])
                )
                validation = _rows_for_sources(
                    dataset, _flatten_by_task(inner["validation_by_task"])
                )
                design = _RidgeDesign.build(
                    features[inner_train], features[validation]
                )
                action_oof[validation] = design.predict(dataset.action[inner_train])
                inner_designs.append((inner_train, validation, design))
            if not np.isfinite(action_oof[train]).all():
                raise ValueError("action nuisance OOF coverage is incomplete")
            outer = _RidgeDesign.build(features[train], features[test])
            action_test_prediction = outer.predict(dataset.action[train])
            u_train = dataset.action[train] - action_oof[train]
            u_test = dataset.action[test] - action_test_prediction
            residual = _RidgeDesign.build(u_train, u_test)
            rank = _effective_rank((u_train - residual.mean) / residual.scale)
            fold_designs.append(
                _FoldDesign(
                    int(fold["outer_fold"]),
                    train,
                    test,
                    tuple(inner_designs),
                    outer,
                    residual,
                    u_train,
                    u_test,
                    rank,
                )
            )
        return cls(
            "controller_history" if controller_context else "visible_state",
            dataset,
            tuple(fold_designs),
        )

    def apply(
        self, target: np.ndarray | Mapping[int, np.ndarray]
    ) -> "EvaluationErrors":
        control = np.full(len(self.dataset.state), np.nan)
        candidate = np.full(len(self.dataset.state), np.nan)
        fold_ids = np.full(len(self.dataset.state), -1, np.int64)
        for fold in self.folds:
            values = (
                np.asarray(target[fold.outer_fold], np.float64)
                if isinstance(target, Mapping)
                else np.asarray(target, np.float64)
            )
            if values.ndim != 2 or len(values) != len(self.dataset.state):
                raise ValueError("conditional-information target shape differs")
            target_oof = np.full_like(values, np.nan)
            for inner_train, validation, design in fold.inner:
                target_oof[validation] = design.predict(values[inner_train])
            if not np.isfinite(target_oof[fold.train]).all():
                raise ValueError("target nuisance OOF coverage is incomplete")
            baseline = fold.outer.predict(values[fold.train])
            residual_prediction = fold.residual.predict(
                values[fold.train] - target_oof[fold.train]
            )
            scale = values[fold.train].std(axis=0)
            scale[scale < 1.0e-8] = 1.0
            control[fold.test] = np.square(
                (values[fold.test] - baseline) / scale
            ).mean(axis=1)
            candidate[fold.test] = np.square(
                (values[fold.test] - baseline - residual_prediction) / scale
            ).mean(axis=1)
            fold_ids[fold.test] = fold.outer_fold
        if (
            not np.isfinite(control).all()
            or not np.isfinite(candidate).all()
            or np.any(fold_ids < 0)
        ):
            raise ValueError("outer OOF prediction coverage is incomplete")
        return EvaluationErrors(control, candidate, fold_ids)


@dataclass(frozen=True)
class EvaluationErrors:
    control: np.ndarray
    candidate: np.ndarray
    fold_ids: np.ndarray


@dataclass(frozen=True)
class BootstrapPlan:
    source_order: tuple[str, ...]
    counts: np.ndarray
    seed: int

    @classmethod
    def build(
        cls, dataset: ReplayDataset, errors: EvaluationErrors, samples: int, seed: int
    ) -> "BootstrapPlan":
        if samples <= 0 or seed < 0:
            raise ValueError("bootstrap configuration is invalid")
        sources = dataset.source_order
        source_index = {source: index for index, source in enumerate(sources)}
        source_fold = {
            source: int(np.unique(errors.fold_ids[dataset.source_ids == source]).item())
            for source in sources
        }
        counts = np.zeros((samples, len(sources)), np.int16)
        rng = np.random.default_rng(seed)
        for outer_fold in range(3):
            for task_id in sorted(set(dataset.task_ids)):
                selected = [
                    source
                    for source in sources
                    if source_fold[source] == outer_fold
                    and np.all(dataset.task_ids[dataset.source_ids == source] == task_id)
                ]
                if not selected:
                    raise ValueError("bootstrap fold/task source cell is empty")
                draws = rng.integers(0, len(selected), size=(samples, len(selected)))
                for local, source in enumerate(selected):
                    counts[:, source_index[source]] = np.sum(draws == local, axis=1)
        return cls(sources, counts, seed)

    @property
    def identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.source_order, separators=(",", ":")).encode())
        digest.update(self.counts.tobytes())
        return digest.hexdigest()


def summarize_errors(
    dataset: ReplayDataset,
    errors: EvaluationErrors,
    plan: BootstrapPlan,
    mask: np.ndarray,
) -> dict[str, object]:
    mask = np.asarray(mask, bool)
    if mask.shape != (len(dataset.state),):
        raise ValueError("stratum mask shape differs")
    source_rows = []
    control = np.full(len(plan.source_order), np.nan)
    candidate = np.full(len(plan.source_order), np.nan)
    source_tasks = []
    source_folds = []
    for index, source in enumerate(plan.source_order):
        rows = (dataset.source_ids == source) & mask
        task = str(np.unique(dataset.task_ids[dataset.source_ids == source]).item())
        fold = int(np.unique(errors.fold_ids[dataset.source_ids == source]).item())
        source_tasks.append(task)
        source_folds.append(fold)
        if np.any(rows):
            control[index] = errors.control[rows].mean()
            candidate[index] = errors.candidate[rows].mean()
        source_rows.append(
            _metric_record(
                source_id=source,
                task_id=task,
                outer_fold=fold,
                transition_count=int(rows.sum()),
                control=control[index],
                candidate=candidate[index],
            )
        )
    task_ids = sorted(set(source_tasks))
    task_reports = {
        task: _point_metrics(
            control[np.asarray(source_tasks) == task],
            candidate[np.asarray(source_tasks) == task],
        )
        for task in task_ids
    }
    fold_reports = {}
    for fold in range(3):
        tasks = {
            task: _point_metrics(
                control[(np.asarray(source_folds) == fold) & (np.asarray(source_tasks) == task)],
                candidate[(np.asarray(source_folds) == fold) & (np.asarray(source_tasks) == task)],
            )
            for task in task_ids
        }
        fold_reports[str(fold)] = {"aggregate": _equal_task_metrics(tasks), "tasks": tasks}
    aggregate = _equal_task_metrics(task_reports)
    logs = np.log(
        np.maximum(control, 1.0e-12) / np.maximum(candidate, 1.0e-12)
    )
    supported = np.isfinite(logs)
    task_values = []
    for task in task_ids:
        fold_values = []
        for fold in range(3):
            cell = (
                (np.asarray(source_tasks) == task)
                & (np.asarray(source_folds) == fold)
                & supported
            )
            numerator = plan.counts[:, cell] @ logs[cell]
            denominator = plan.counts[:, cell].sum(axis=1)
            fold_values.append(
                np.divide(
                    numerator,
                    denominator,
                    out=np.full(len(numerator), np.nan),
                    where=denominator > 0,
                )
            )
        stacked = np.stack(fold_values)
        task_values.append(
            np.where(np.isfinite(stacked).all(axis=0), stacked.mean(axis=0), np.nan)
        )
    bootstrap = np.stack(task_values)
    bootstrap = np.where(
        np.isfinite(bootstrap).all(axis=0), bootstrap.mean(axis=0), np.nan
    )
    finite = bootstrap[np.isfinite(bootstrap)]
    cell_support = {
        f"{fold}|{task}": int(
            np.sum(
                (np.asarray(source_folds) == fold)
                & (np.asarray(source_tasks) == task)
                & supported
            )
        )
        for fold in range(3)
        for task in task_ids
    }
    aggregate["mean_source_log_ratio"] = (
        float(np.mean([value["mean_source_log_ratio"] for value in task_reports.values()]))
        if all(value["measured"] for value in task_reports.values())
        else None
    )
    aggregate["bootstrap"] = {
        "samples": len(plan.counts),
        "seed": plan.seed,
        "multiplicity_sha256": plan.identity,
        "finite_replicates": len(finite),
        "mean_log_ratio_p05": float(np.quantile(finite, 0.05)) if len(finite) else None,
    }
    aggregate["source_level_measurable"] = all(
        value >= 2 for value in cell_support.values()
    )
    aggregate["complete_source_coverage"] = bool(supported.all())
    return {
        "aggregate": aggregate,
        "tasks": task_reports,
        "outer_folds": fold_reports,
        "sources": source_rows,
        "source_support_by_fold_task": cell_support,
    }


def evaluate_replay_conditional_information(
    dataset: ReplayDataset,
    fold_manifest: Mapping[str, object],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, object], NestedDesign]:
    main_design = NestedDesign.build(dataset, fold_manifest)
    controller_design = NestedDesign.build(
        dataset, fold_manifest, controller_context=True
    )
    main_rate = main_design.apply(dataset.rate_target)
    configuration = main_design.apply(dataset.configuration_target)
    controller_rate = controller_design.apply(dataset.rate_target)
    plan = BootstrapPlan.build(
        dataset, main_rate, bootstrap_samples, bootstrap_seed
    )
    masks = dataset.masks()

    def family(errors: EvaluationErrors) -> dict[str, object]:
        return {
            "strata": {
                name: summarize_errors(dataset, errors, plan, masks[name])
                for name in STRATA
            }
        }

    report = {
        "schema_version": REPORT_SCHEMA,
        "proposal_id": PROPOSAL_ID,
        "comparison": {
            "control": "m_y(S)",
            "candidate": "m_y(S) + B(a - m_a(S))",
            "shared_target_nuisance_baseline": True,
            "double_oof_residual": True,
            "ridge": RIDGE,
        },
        "bootstrap": {
            "unit": "source_episode_within_outer_fold_and_task",
            "task_weighting": "equal",
            "source_weighting": "equal",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "multiplicity_sha256": plan.identity,
            "synchronized_across_targets_guards_and_strata": True,
        },
        "effective_rank": {
            "outer_folds": [
                {"outer_fold": fold.outer_fold, "value": fold.effective_rank}
                for fold in main_design.folds
            ],
            "minimum": min(fold.effective_rank for fold in main_design.folds),
            "definition": "covariance_entropy_rank_of_scaled_inner_oof_action_residual",
        },
        "target_families": {
            "rate": family(main_rate),
            "configuration": family(configuration),
        },
        "controller_context_guard": {
            "features": {
                "state": 37,
                "current_actor_proposal": 16,
                "past_actor_proposal_and_executed_action": 128,
                "availability_mask": 4,
                "history_steps": 4,
                "cross_shard_history": False,
            },
            "effective_rank": {
                "outer_folds": [
                    {"outer_fold": fold.outer_fold, "value": fold.effective_rank}
                    for fold in controller_design.folds
                ],
                "minimum": min(
                    fold.effective_rank for fold in controller_design.folds
                ),
            },
            **family(controller_rate),
        },
    }
    return report, main_design


def _metric_record(**values: object) -> dict[str, object]:
    control = float(values.pop("control"))
    candidate = float(values.pop("candidate"))
    measured = math.isfinite(control) and math.isfinite(candidate)
    return {
        **values,
        "measured": measured,
        "control_mse": control if measured else None,
        "candidate_mse": candidate if measured else None,
        "ratio": control / max(candidate, 1.0e-12) if measured else None,
        "log_ratio": (
            math.log(max(control, 1.0e-12) / max(candidate, 1.0e-12))
            if measured
            else None
        ),
    }


def _point_metrics(control: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    supported = np.isfinite(control) & np.isfinite(candidate)
    if not np.any(supported):
        return {
            "measured": False,
            "source_count": 0,
            "control_mse": None,
            "candidate_mse": None,
            "ratio": None,
            "mean_source_log_ratio": None,
        }
    control_mse = float(control[supported].mean())
    candidate_mse = float(candidate[supported].mean())
    return {
        "measured": True,
        "source_count": int(supported.sum()),
        "control_mse": control_mse,
        "candidate_mse": candidate_mse,
        "ratio": control_mse / max(candidate_mse, 1.0e-12),
        "mean_source_log_ratio": float(
            np.log(
                np.maximum(control[supported], 1.0e-12)
                / np.maximum(candidate[supported], 1.0e-12)
            ).mean()
        ),
    }


def _equal_task_metrics(tasks: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    measured = all(value["measured"] for value in tasks.values())
    if not measured:
        return {
            "measured": False,
            "control_mse": None,
            "candidate_mse": None,
            "ratio": None,
        }
    control = float(np.mean([value["control_mse"] for value in tasks.values()]))
    candidate = float(np.mean([value["candidate_mse"] for value in tasks.values()]))
    return {
        "measured": True,
        "control_mse": control,
        "candidate_mse": candidate,
        "ratio": control / max(candidate, 1.0e-12),
    }


def _effective_rank(values: np.ndarray) -> float:
    covariance = np.cov(values, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total <= 1.0e-12:
        return 0.0
    probabilities = eigenvalues[eigenvalues > total * 1.0e-12] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _rows_for_sources(dataset: ReplayDataset, sources: set[str]) -> np.ndarray:
    rows = np.flatnonzero(np.isin(dataset.source_ids, tuple(sources)))
    if not len(rows):
        raise ValueError("source fold has no transition rows")
    return rows


def _flatten_by_task(value: Mapping[str, Sequence[str]]) -> set[str]:
    return {source for sources in value.values() for source in sources}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _seed(base_seed: int, *parts: object) -> int:
    payload = "|".join((str(base_seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
