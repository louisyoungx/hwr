"""Engine-independent closed-loop evaluation for deployable bimanual policies."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from hwr.core.embodied import DualArmAction, DualArmActionFrame, DualArmObservation
from hwr.core.runtime import Policy, RuntimeBackend


AblationMode = Literal["none", "lock_left", "lock_right"]
_SAFETY_INTERVENTION_BOOTSTRAP_SAMPLES = 2_000
_SAFETY_INTERVENTION_BOOTSTRAP_SEED = 20_260_913
_SAFETY_INTERVENTION_BOOTSTRAP_SEED_STRIDE = 104_729


class AuditedDualArmBackend(RuntimeBackend, Protocol):
    """Runtime plus read-only task metrics unavailable to the Actor."""

    def task_audit(self) -> dict[str, object]: ...


class AppliedActionAwarePolicy(Policy, Protocol):
    """Deployment policy whose action history follows runtime execution."""

    def record_applied_action(self, action: DualArmAction) -> None: ...


class BimanualEvaluationObserver(Protocol):
    """Optional evidence sink; it may observe but cannot replace actions."""

    def episode_started(
        self,
        backend: AuditedDualArmBackend,
        observation: DualArmObservation,
        *,
        seed: int,
        ablation: AblationMode,
    ) -> None: ...

    def step_recorded(
        self,
        backend: AuditedDualArmBackend,
        observation: DualArmObservation,
        *,
        step: int,
    ) -> None: ...

    def episode_finished(
        self,
        backend: AuditedDualArmBackend,
        record: "BimanualEpisodeEvaluation",
    ) -> None: ...


@dataclass(frozen=True)
class BimanualEpisodeEvaluation:
    task_id: str
    seed: int
    ablation: AblationMode
    success: bool
    reason: str
    steps: int
    stable_steps: int
    maximum_concurrent_steps: int
    left_contact_steps: int
    right_contact_steps: int
    simultaneous_contact_steps: int
    severe_collisions: int
    maximum_forbidden_force: float
    safety_interventions: int
    action_sources: tuple[str, ...]
    audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BimanualEvaluationReport:
    policy_id: str
    episodes: tuple[BimanualEpisodeEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.policy_id or not self.episodes:
            raise ValueError("bimanual evaluation report cannot be empty")

    def group(
        self, task_id: str, ablation: AblationMode
    ) -> tuple[BimanualEpisodeEvaluation, ...]:
        return tuple(
            episode
            for episode in self.episodes
            if episode.task_id == task_id and episode.ablation == ablation
        )

    def to_dict(self) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        identities = sorted(
            {(episode.task_id, episode.ablation) for episode in self.episodes}
        )
        for task_id, ablation in identities:
            episodes = self.group(task_id, ablation)
            groups.append(
                {
                    "task_id": task_id,
                    "ablation": ablation,
                    "episode_count": len(episodes),
                    "success_count": sum(item.success for item in episodes),
                    "success_rate": sum(item.success for item in episodes)
                    / len(episodes),
                    "severe_collision_count": sum(
                        item.severe_collisions for item in episodes
                    ),
                }
            )
        return {
            "schema_version": "hwr.bimanual-evaluation/v1",
            "policy_id": self.policy_id,
            "episode_count": len(self.episodes),
            "groups": groups,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


@dataclass(frozen=True)
class BimanualAcceptanceCriteria:
    minimum_unseen_episodes: int = 40
    minimum_success_rate: float = 0.70
    maximum_ablation_success_rate: float = 0.10
    confidence_level: float = 0.95
    minimum_stable_seconds: float = 2.0
    minimum_concurrent_seconds: float = 0.5
    maximum_safety_intervention_rate_p95: float = 0.01
    maximum_safety_intervention_rate_bootstrap_upper: float = 0.02
    maximum_safety_intervention_rate: float = 0.05

    def __post_init__(self) -> None:
        if (
            self.minimum_unseen_episodes <= 0
            or self.minimum_stable_seconds <= 0
            or self.minimum_concurrent_seconds <= 0
        ):
            raise ValueError("acceptance episode count and stability must be positive")
        rates = (
            self.minimum_success_rate,
            self.maximum_ablation_success_rate,
            self.confidence_level,
            self.maximum_safety_intervention_rate_p95,
            self.maximum_safety_intervention_rate_bootstrap_upper,
            self.maximum_safety_intervention_rate,
        )
        if not all(0.0 <= value <= 1.0 for value in rates):
            raise ValueError("acceptance rates must be in [0, 1]")
        if self.confidence_level <= 0.5 or self.confidence_level >= 1.0:
            raise ValueError("acceptance confidence level must be in (0.5, 1)")


def evaluate_bimanual_policy(
    task_id: str,
    max_steps: int,
    environment_factory: Callable[[], AuditedDualArmBackend],
    policy: AppliedActionAwarePolicy,
    seeds: Sequence[int],
    *,
    ablation: AblationMode = "none",
    observer: BimanualEvaluationObserver | None = None,
) -> BimanualEvaluationReport:
    """Evaluate one task and condition without exploration or privileged inputs."""
    if max_steps <= 0 or not seeds:
        raise ValueError("bimanual evaluation requires steps and unseen seeds")
    if ablation not in ("none", "lock_left", "lock_right"):
        raise ValueError("unknown bimanual ablation")
    episodes = tuple(
        _evaluate_episode(
            task_id,
            max_steps,
            environment_factory,
            policy,
            int(seed),
            ablation,
            observer,
        )
        for seed in seeds
    )
    return BimanualEvaluationReport(policy.spec().policy_id, episodes)


def combine_bimanual_reports(
    reports: Sequence[BimanualEvaluationReport],
) -> BimanualEvaluationReport:
    if not reports or len({report.policy_id for report in reports}) != 1:
        raise ValueError("combined evaluation reports require one policy")
    return BimanualEvaluationReport(
        reports[0].policy_id,
        tuple(episode for report in reports for episode in report.episodes),
    )


def assess_bimanual_acceptance(
    report: BimanualEvaluationReport,
    control_hz_by_task: Mapping[str, float],
    criteria: BimanualAcceptanceCriteria | None = None,
) -> dict[str, Any]:
    """Apply the fixed multi-scene, safety, stability, and ablation gates."""
    settings = criteria or BimanualAcceptanceCriteria()
    details: list[dict[str, Any]] = []
    passed = True
    for task_index, (task_id, control_hz) in enumerate(
        sorted(control_hz_by_task.items())
    ):
        if control_hz <= 0:
            raise ValueError("evaluation control frequency must be positive")
        normal = report.group(task_id, "none")
        success_rate = _success_rate(normal)
        success_interval = _wilson_interval(
            sum(item.success for item in normal),
            len(normal),
            settings.confidence_level,
        )
        stable_steps = math.ceil(settings.minimum_stable_seconds * control_hz)
        concurrent_steps = math.ceil(
            settings.minimum_concurrent_seconds * control_hz
        )
        safety_burden = _safety_intervention_burden(
            normal,
            _SAFETY_INTERVENTION_BOOTSTRAP_SEED
            + task_index * _SAFETY_INTERVENTION_BOOTSTRAP_SEED_STRIDE,
        )
        safety_checks = {
            "empirical_p95": (
                safety_burden["empirical_p95"]
                <= settings.maximum_safety_intervention_rate_p95
            ),
            "bootstrap_upper": (
                safety_burden["bootstrap"]["p95_upper"]
                <= settings.maximum_safety_intervention_rate_bootstrap_upper
            ),
            "maximum": (
                safety_burden["maximum"]
                <= settings.maximum_safety_intervention_rate
            ),
        }
        safety_burden["checks"] = safety_checks
        safety_burden["passed"] = bool(normal) and all(safety_checks.values())
        normal_passed = (
            len(normal) >= settings.minimum_unseen_episodes
            and success_rate >= settings.minimum_success_rate
            and success_interval[0] >= settings.minimum_success_rate
            and sum(item.severe_collisions for item in normal) == 0
            and all(item.stable_steps >= stable_steps for item in normal if item.success)
            and all(
                item.maximum_concurrent_steps >= concurrent_steps
                for item in normal
                if item.success
            )
            and safety_burden["passed"]
        )
        ablations: dict[str, float] = {}
        ablation_intervals: dict[str, dict[str, float]] = {}
        ablation_passed = True
        for mode in ("lock_left", "lock_right"):
            episodes = report.group(task_id, mode)
            rate = _success_rate(episodes)
            interval = _wilson_interval(
                sum(item.success for item in episodes),
                len(episodes),
                settings.confidence_level,
            )
            ablations[mode] = rate
            ablation_intervals[mode] = {
                "lower": interval[0],
                "upper": interval[1],
            }
            ablation_passed &= (
                len(episodes) >= settings.minimum_unseen_episodes
                and rate < settings.maximum_ablation_success_rate
                and interval[1] < settings.maximum_ablation_success_rate
                and sum(item.severe_collisions for item in episodes) == 0
            )
        task_passed = normal_passed and ablation_passed
        passed &= task_passed
        details.append(
            {
                "task_id": task_id,
                "passed": task_passed,
                "normal_episode_count": len(normal),
                "normal_success_rate": success_rate,
                "normal_success_interval": {
                    "lower": success_interval[0],
                    "upper": success_interval[1],
                },
                "required_stable_steps": stable_steps,
                "required_concurrent_steps": concurrent_steps,
                "safety_intervention_burden": safety_burden,
                "ablation_success_rates": ablations,
                "ablation_success_intervals": ablation_intervals,
            }
        )
    return {
        "schema_version": "hwr.bimanual-acceptance/v3",
        "passed": passed,
        "criteria": asdict(settings),
        "tasks": details,
    }


def _safety_intervention_burden(
    episodes: Sequence[BimanualEpisodeEvaluation], seed: int
) -> dict[str, Any]:
    rates = tuple(
        sorted(
            episode.safety_interventions / max(episode.steps, 1)
            for episode in episodes
        )
    )
    bootstrap_p95 = _bootstrap_p95_distribution(
        rates,
        samples=_SAFETY_INTERVENTION_BOOTSTRAP_SAMPLES,
        seed=seed,
    )
    return {
        "episode_rate_definition": "safety_interventions/max(steps,1)",
        "empirical_p95": _quantile(rates, 0.95),
        "bootstrap": {
            "unit": "episode_cluster",
            "samples": _SAFETY_INTERVENTION_BOOTSTRAP_SAMPLES,
            "seed": seed,
            "statistic": "episode_rate_p95",
            "p95_distribution": list(bootstrap_p95),
            "p95_upper": _quantile(bootstrap_p95, 0.95),
        },
        "maximum": max(rates, default=0.0),
    }


def _bootstrap_p95_distribution(
    rates: Sequence[float], *, samples: int, seed: int
) -> tuple[float, ...]:
    if not rates:
        return ()
    generator = random.Random(seed)
    count = len(rates)
    return tuple(
        _quantile(
            tuple(rates[generator.randrange(count)] for _ in range(count)),
            0.95,
        )
        for _ in range(samples)
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    return lower + (upper - lower) * (position - lower_index)


def _wilson_interval(
    successes: int, count: int, confidence: float
) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 1.0
    if successes < 0 or successes > count or not 0.5 < confidence < 1.0:
        raise ValueError("Wilson interval inputs are invalid")
    probability = successes / count
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / count
        + z * z / (4.0 * count * count)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _evaluate_episode(
    task_id: str,
    max_steps: int,
    environment_factory: Callable[[], AuditedDualArmBackend],
    policy: AppliedActionAwarePolicy,
    seed: int,
    ablation: AblationMode,
    observer: BimanualEvaluationObserver | None,
) -> BimanualEpisodeEvaluation:
    environment = environment_factory()
    record: BimanualEpisodeEvaluation | None = None
    try:
        observation = environment.reset(seed=seed, task_id=task_id)
        policy.reset(task_id=task_id, seed=seed)
        if observer is not None:
            observer.episode_started(
                environment, observation, seed=seed, ablation=ablation
            )
        source = f"learned:{policy.spec().policy_id}"
        interventions = 0
        steps = 0
        for steps in range(1, max_steps + 1):
            chunk = policy.infer((observation,))
            action = _apply_ablation(
                chunk.actions[0], observation, ablation
            )
            frame = _action_frame(observation.timestamp_ns, action, source)
            outcome = environment.apply(frame)
            applied = outcome.info.get("applied_action")
            if not isinstance(applied, DualArmActionFrame):
                raise TypeError("runtime did not report its applied dual-arm action")
            policy.record_applied_action(applied.action)
            interventions += int(outcome.info.get("safety_intervened", False))
            observation = outcome.observation
            if observer is not None:
                observer.step_recorded(environment, observation, step=steps)
            if outcome.terminated or outcome.truncated:
                break
        result = environment.result()
        if result is None:
            raise RuntimeError("bimanual runtime produced no terminal result")
        audit = environment.task_audit()
        record = BimanualEpisodeEvaluation(
            task_id=task_id,
            seed=seed,
            ablation=ablation,
            success=result.success,
            reason=result.reason,
            steps=steps,
            stable_steps=int(audit["stable_steps"]),
            maximum_concurrent_steps=int(audit["maximum_concurrent_steps"]),
            left_contact_steps=int(audit["left_contact_steps"]),
            right_contact_steps=int(audit["right_contact_steps"]),
            simultaneous_contact_steps=int(audit["simultaneous_contact_steps"]),
            severe_collisions=int(audit["severe_collision_count"]),
            maximum_forbidden_force=float(audit["maximum_forbidden_force"]),
            safety_interventions=interventions,
            action_sources=(source,),
            audit=audit,
        )
        if observer is not None:
            observer.episode_finished(environment, record)
        return record
    finally:
        environment.close()


def _apply_ablation(
    action: DualArmAction,
    observation: DualArmObservation,
    ablation: AblationMode,
) -> DualArmAction:
    if ablation == "none":
        return action
    zero = (0.0,) * 6
    proprioception = observation.proprioception
    return DualArmAction(
        action.base_linear,
        action.base_angular,
        zero if ablation == "lock_left" else action.left_arm,
        zero if ablation == "lock_right" else action.right_arm,
        (
            proprioception.left_gripper_position
            if ablation == "lock_left"
            else action.left_gripper
        ),
        (
            proprioception.right_gripper_position
            if ablation == "lock_right"
            else action.right_gripper
        ),
    )


def _action_frame(
    timestamp_ns: int, action: DualArmAction, source: str
) -> DualArmActionFrame:
    period_ns = round(1_000_000_000 / 20.0)
    return DualArmActionFrame(
        timestamp_ns,
        timestamp_ns,
        timestamp_ns + 2 * period_ns,
        source,
        action,
    )


def _success_rate(episodes: Sequence[BimanualEpisodeEvaluation]) -> float:
    return sum(episode.success for episode in episodes) / len(episodes) if episodes else 0.0
