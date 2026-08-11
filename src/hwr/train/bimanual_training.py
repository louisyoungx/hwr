"""Local no-demonstration training loop for the three bimanual household tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import numpy as np
import torch

from hwr.core.embodied import DualArmAction, DualArmActionFrame
from hwr.core.runtime import SnapshotRuntimeBackend
from hwr.core.state_snapshot import PhysicalStateSnapshot
from hwr.policy.bimanual_input import (
    BimanualActorInputPipeline,
    BimanualInputConfig,
    actor_input_tensors,
    stack_actor_inputs,
)
from hwr.policy.privileged_critic import PrivilegedCriticConfig
from hwr.policy.vla_input import VLAActorInput
from hwr.policy.vla_model import VLAActorConfig, VLAActorModel
from hwr.perception import FrozenNgramLanguageConfig, FrozenNgramLanguageEncoder
from hwr.tasks import BimanualTaskSpec, PrivilegedTaskState
from hwr.train.asymmetric_rl import (
    AsymmetricActorCriticTrainer,
    AsymmetricRLBatch,
    AsymmetricRLConfig,
)
from hwr.train.bimanual_config import BimanualRLTrainingConfig
from hwr.train.bimanual_metrics import (
    bilateral_near_statistics,
    physical_progress_record_fields,
    transition_safety_cost,
)
from hwr.train.bimanual_records import TrainingEpisodeRecord
from hwr.train.bimanual_runtime import dual_arm_action_frame
from hwr.train.action_exploration import (
    TemporalActionExplorer,
    TemporalExplorationConfig,
)
from hwr.train.curriculum import AutomaticCurriculum, CurriculumConfig
from hwr.train.learning_frontier import (
    LearningFrontierCandidate,
    LearningFrontierConfig,
    LearningSignal,
    TaskAgnosticLearningFrontier,
    prepare_learning_frontier_reset,
)
from hwr.train.learning_signals import (
    failure_boundary_step,
    reward_improvement_speeds,
)
from hwr.train.autonomous_replay import AutonomousEpisode
from hwr.train.n_step import build_n_step_targets
from hwr.train.task_replay import TaskPartitionedAutonomousReplayBuffer
from hwr.train.task_sampling import (
    OutcomeAdaptiveTaskSampler,
    OutcomeAdaptiveTaskSamplingConfig,
    TaskOutcome,
)


@dataclass
class BimanualTrainingResult:
    config: BimanualRLTrainingConfig
    actor_config: VLAActorConfig
    rl_config: AsymmetricRLConfig
    trainer: AsymmetricActorCriticTrainer
    replay: TaskPartitionedAutonomousReplayBuffer
    curriculum: AutomaticCurriculum
    frontier: TaskAgnosticLearningFrontier
    task_sampler: OutcomeAdaptiveTaskSampler
    records: list[TrainingEpisodeRecord]
    language_encoder: FrozenNgramLanguageEncoder
    preprocess_fingerprint: str
    exploration_audit: Mapping[str, object]
    environment_steps: int
    task_rng_state: Mapping[str, object]
    frontier_rng_state: Mapping[str, object]
    exploration_rng_state: Mapping[str, object]
    torch_rng_state: torch.Tensor


@dataclass
class _EpisodeBuffers:
    actor_inputs: list[VLAActorInput]
    next_actor_inputs: list[VLAActorInput]
    states: list[tuple[float, ...]]
    next_states: list[tuple[float, ...]]
    actions: list[tuple[float, ...]]
    proposed_actions: list[tuple[float, ...]]
    safety_costs: list[float]
    rewards: list[float]
    done: list[float]
    snapshots: list[PhysicalStateSnapshot]

    @classmethod
    def empty(cls) -> "_EpisodeBuffers":
        return cls([], [], [], [], [], [], [], [], [], [])


class BimanualTrainingBackend(SnapshotRuntimeBackend, Protocol):
    def set_curriculum_level(self, level: float) -> None: ...

    def privileged_training_state(self) -> PrivilegedTaskState: ...

    def task_audit(self) -> dict[str, object]: ...


BimanualEnvironmentFactory = Callable[
    [BimanualTaskSpec, int, int], BimanualTrainingBackend
]


class BimanualTrainingRunner:
    """Collect random/Actor experience and update without an action-label source."""

    def __init__(
        self,
        tasks: Mapping[str, BimanualTaskSpec],
        environment_factory: BimanualEnvironmentFactory,
        config: BimanualRLTrainingConfig,
    ) -> None:
        if len(tasks) != 3:
            raise ValueError("training requires exactly three bimanual tasks")
        self.tasks = dict(tasks)
        self.environment_factory = environment_factory
        self.config = config
        self.task_ids = tuple(sorted(tasks))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        task_seed, frontier_seed, exploration_seed = np.random.SeedSequence(
            config.seed
        ).spawn(3)
        self.task_rng = np.random.default_rng(task_seed)
        self.frontier_rng = np.random.default_rng(frontier_seed)
        self.exploration_rng = np.random.default_rng(exploration_seed)
        self.rl_config = AsymmetricRLConfig(
            actor_learning_rate=config.actor_learning_rate,
            final_actor_learning_rate=config.final_actor_learning_rate,
            actor_learning_rate_decay_updates=(
                config.actor_learning_rate_decay_updates
            ),
            behavior_regularization=0.0,
            visual_temporal_contrastive_weight=config.visual_temporal_contrastive_weight,
            augmentation_consistency_weight=(
                config.augmentation_consistency_weight
            ),
        )
        self.explorer = TemporalActionExplorer(
            TemporalExplorationConfig(
                noise_standard_deviation=config.exploration_noise,
                noise_correlation=config.exploration_correlation,
                action_smoothing=config.action_smoothing,
                gripper_epsilon=config.gripper_exploration_probability,
                gripper_hold_steps=config.gripper_exploration_hold_steps,
                policy_gripper_hold_steps=config.policy_gripper_hold_steps,
                reflection_coupled_probability=(
                    config.reflection_coupled_exploration_probability
                ),
                paired_gripper_probability=(
                    config.paired_gripper_exploration_probability
                ),
                global_random_burst_probability=(
                    config.global_random_burst_probability
                ),
                global_random_burst_steps=config.global_random_burst_steps,
                actuator_dwell_probability=config.actuator_dwell_probability,
                actuator_dwell_steps=config.actuator_dwell_steps,
                actuator_initial_dwell_probability=(
                    config.actuator_initial_dwell_probability
                ),
                actuator_dwell_closed_probability=(
                    config.actuator_dwell_closed_probability
                ),
                base_linear_scale=self.rl_config.base_linear_scale,
                base_angular_scale=self.rl_config.base_angular_scale,
                arm_twist_scale=self.rl_config.arm_velocity_scale,
            ),
            self.exploration_rng,
        )
        self.language = FrozenNgramLanguageEncoder(
            FrozenNgramLanguageConfig(dimension=config.language_dim)
        )
        self.input_config = BimanualInputConfig(
            config.raw_image_width,
            config.raw_image_height,
            image_width=config.image_width,
            image_height=config.image_height,
            point_count=config.point_count,
        )
        self.pipeline = BimanualActorInputPipeline(self.input_config, self.language)
        self.actor_config = VLAActorConfig(
            visual_history=1,
            action_history=1,
            proprioception_dim=37,
            language_dim=config.language_dim,
            point_count=config.point_count,
            action_chunk_size=1,
            hidden_dim=config.hidden_dim,
            attention_heads=config.attention_heads,
            transformer_layers=config.transformer_layers,
        )
        actor = VLAActorModel(self.actor_config)
        self.trainer = AsymmetricActorCriticTrainer(
            actor,
            PrivilegedCriticConfig(62, 1, hidden_dim=max(128, config.hidden_dim)),
            self.rl_config,
            device=config.device,
        )
        self.replay = TaskPartitionedAutonomousReplayBuffer(
            config.replay_capacity, self.task_ids, seed=config.seed
        )
        self.curriculum = AutomaticCurriculum(
            self.task_ids, CurriculumConfig(initial_level=0.1)
        )
        self.frontier = TaskAgnosticLearningFrontier(
            self.task_ids,
            LearningFrontierConfig(
                capacity_per_task=config.frontier_capacity_per_task,
                reset_probability=config.frontier_reset_probability,
                signature_uniform_fraction=(
                    config.frontier_signature_uniform_fraction
                ),
                maximum_entries_per_source_signature=(
                    config.frontier_max_entries_per_source_signature
                ),
            ),
        )
        self.task_sampler = OutcomeAdaptiveTaskSampler(
            self.task_ids,
            OutcomeAdaptiveTaskSamplingConfig(
                temperature=config.task_sampling_temperature,
                maximum_probability=config.task_sampling_maximum_probability,
            ),
        )
        self.records: list[TrainingEpisodeRecord] = []
        self._environment_steps = 0

    def train(
        self,
        on_episode: Callable[[BimanualTrainingResult], None] | None = None,
    ) -> BimanualTrainingResult:
        environments = {
            task_id: self.environment_factory(
                self.tasks[task_id],
                self.config.raw_image_width,
                self.config.raw_image_height,
            )
            for task_id in self.task_ids
        }
        try:
            for episode_index in range(len(self.records), self.config.episodes):
                task_id, sampling_probability = self.task_sampler.sample(
                    self.task_rng
                )
                record = self._run_episode(
                    episode_index,
                    task_id,
                    environments[task_id],
                    sampling_probability=sampling_probability,
                )
                self.records.append(record)
                if on_episode is not None:
                    on_episode(self.result())
        finally:
            for environment in environments.values():
                environment.close()
        return self.result()

    def result(self) -> BimanualTrainingResult:
        return BimanualTrainingResult(
            self.config,
            self.actor_config,
            self.rl_config,
            self.trainer,
            self.replay,
            self.curriculum,
            self.frontier,
            self.task_sampler,
            self.records,
            self.language,
            self.pipeline.preprocessor.fingerprint,
            self.explorer.audit(),
            self._environment_steps,
            self.task_rng.bit_generator.state,
            self.frontier_rng.bit_generator.state,
            self.exploration_rng.bit_generator.state,
            torch.get_rng_state(),
        )

    def load_training_state(self, value: Mapping[str, object]) -> None:
        """Restore all learning state before continuing at the next episode."""
        self.trainer.load_state_dict(value["trainer"])
        self.replay.load_state_dict(value["replay"])
        self.curriculum.load_state_dict(value["curriculum"])
        if "frontier" in value:
            self.frontier.load_state_dict(value["frontier"])
        self.task_sampler.load_state_dict(value["task_sampler"])
        records = []
        for record in value["records"]:
            fields = dict(record)
            fields.setdefault(
                "minimum_worst_side_reach_distance",
                max(
                    fields["minimum_left_reach_distance"],
                    fields["minimum_right_reach_distance"],
                ),
            )
            fields.setdefault("environment_reset_seed", fields["seed"])
            entry = self.frontier.find(
                fields["task_id"],
                int(fields.get("frontier_source_episode", -1)),
                int(fields.get("frontier_source_step", -1)),
            )
            fields.setdefault(
                "frontier_source_signature", entry.signature if entry else -1
            )
            fields.setdefault("frontier_reset_contact_steps", 0)
            fields.setdefault("frontier_reset_validated", False)
            fields.setdefault("frontier_reset_reproduced", False)
            fields.setdefault(
                "frontier_reset_applied", bool(fields.get("frontier_reset", False))
            )
            records.append(TrainingEpisodeRecord(**fields))
        self.records = records
        self._environment_steps = int(value["environment_steps"])
        self.task_rng.bit_generator.state = value["task_rng_state"]
        self.frontier_rng.bit_generator.state = value["frontier_rng_state"]
        self.exploration_rng.bit_generator.state = value[
            "exploration_rng_state"
        ]
        torch.set_rng_state(value["torch_rng_state"])
        if len(self.records) > self.config.episodes:
            raise ValueError("resume checkpoint exceeds configured total episodes")

    def _run_episode(
        self,
        episode_index: int,
        task_id: str,
        environment: BimanualTrainingBackend,
        *,
        sampling_probability: float,
    ) -> TrainingEpisodeRecord:
        seed = self.config.seed + episode_index * 104729
        level = self.curriculum.level(task_id)
        environment.set_curriculum_level(level)
        frontier_entry = None
        if episode_index >= self.config.initial_random_episodes:
            frontier_entry = self.frontier.select(task_id, self.frontier_rng)
        source_seed = self.config.seed + (frontier_entry.source_episode if frontier_entry else episode_index) * 104729
        prepared = prepare_learning_frontier_reset(
            environment,
            frontier_entry,
            task_id=task_id,
            episode_seed=seed,
            source_seed=source_seed,
        )
        observation = prepared.observation
        self.explorer.reset()
        self.pipeline.reset()
        actor_input = self.pipeline.build(observation)
        state = environment.privileged_training_state()
        buffers = _EpisodeBuffers.empty()
        previous_action: DualArmAction | None = None
        total_reward, success = 0.0, False
        safety_interventions = 0
        environment_terminated = environment_truncated = False
        step_limit = self.config.episode_step_limit or self.tasks[task_id].max_steps
        for step in range(step_limit):
            random_phase = episode_index < self.config.initial_random_episodes
            action = self._select_action(
                actor_input,
                previous_action,
                random_phase=random_phase,
                refresh_random=step % self.config.random_action_hold_steps == 0,
            )
            frame = dual_arm_action_frame(
                observation.timestamp_ns,
                action,
                source=(
                    "random_actor" if random_phase else "learned:asymmetric_rl"
                ),
            )
            outcome = environment.apply(frame)
            applied_frame = outcome.info["applied_action"]
            if not isinstance(applied_frame, DualArmActionFrame):
                raise TypeError("runtime did not report its applied dual-arm action")
            applied_action = applied_frame.action
            safety_interventions += int(outcome.info["safety_intervened"])
            self.pipeline.record_action(applied_action)
            next_input = self.pipeline.build(outcome.observation)
            next_state = environment.privileged_training_state()
            terminal = outcome.terminated or outcome.truncated
            limit = step + 1 >= step_limit
            safety_cost = transition_safety_cost(outcome.info, next_state.metrics)
            self._append_transition(
                buffers,
                actor_input,
                next_input,
                state,
                next_state,
                action,
                applied_action,
                safety_cost,
                outcome.reward,
                terminal or limit,
                environment.capture_state_snapshot(),
            )
            total_reward += outcome.reward
            self._environment_steps += 1
            actor_input, state = next_input, next_state
            observation = outcome.observation
            previous_action = applied_action
            success = bool(environment.result() and environment.result().success)
            if terminal or limit:
                environment_terminated = outcome.terminated
                environment_truncated = bool(outcome.truncated or limit and not terminal)
                break
        audit = environment.task_audit()
        episode = self._autonomous_episode(
            buffers,
            success,
            tuple(
                item.transform_id
                for item in environment.legal_environment_transforms()
            ),
        )
        td_errors = self.trainer.estimate_td_error(episode.batch)
        reward_improvement = self.task_sampler.reward_improvement(
            task_id, total_reward
        )
        terminated_failure = environment_terminated and not success
        candidates = self._learning_frontier_candidates(
            task_id,
            episode_index,
            buffers,
            td_errors,
            terminated_failure,
        )
        self.frontier.consider_episode(task_id, candidates)
        self.replay.add_episode(task_id, episode)
        update_summary = self._update_after_episode(len(buffers.rewards))
        self.curriculum.record(
            task_id,
            success=success,
            severe_collision=int(audit["severe_collision_count"]) > 0,
        )
        self.task_sampler.record(
            task_id,
            TaskOutcome(
                total_reward,
                _mean_signal(candidates, "state_novelty"),
                float(td_errors.mean()),
                reward_improvement,
                float(terminated_failure),
                success,
                sum(buffers.safety_costs) / len(buffers.safety_costs),
            ),
        )
        bilateral_near_steps, maximum_bilateral_near_steps = bilateral_near_statistics(buffers.states)
        return TrainingEpisodeRecord(
            episode=episode_index,
            task_id=task_id,
            seed=seed,
            steps=len(buffers.rewards),
            reward=total_reward,
            success=success,
            severe_collisions=int(audit["severe_collision_count"]),
            maximum_concurrent_steps=int(audit["maximum_concurrent_steps"]),
            left_contact_steps=int(audit["left_contact_steps"]),
            right_contact_steps=int(audit["right_contact_steps"]),
            simultaneous_contact_steps=int(audit["simultaneous_contact_steps"]),
            stable_steps=int(audit["stable_steps"]),
            minimum_left_reach_distance=min(state[24] for state in buffers.states),
            minimum_right_reach_distance=min(state[25] for state in buffers.states),
            minimum_worst_side_reach_distance=min(
                max(state[24], state[25]) for state in buffers.states
            ),
            curriculum_level=level,
            replay_size=self.replay.size,
            updates=self.trainer.update_count,
            bilateral_near_steps=bilateral_near_steps,
            maximum_bilateral_near_steps=maximum_bilateral_near_steps,
            safety_interventions=safety_interventions,
            mean_state_novelty=_mean_signal(candidates, "state_novelty"),
            mean_td_error=float(td_errors.mean()),
            reward_improvement=reward_improvement,
            failure_boundary=float(terminated_failure),
            sampling_probability=sampling_probability,
            actor_updates=int(update_summary["actor_updates"]),
            mean_critic_loss=update_summary["mean_critic_loss"],
            mean_safety_loss=update_summary["mean_safety_loss"],
            mean_actor_loss=update_summary["mean_actor_loss"],
            mean_actor_reward_value=update_summary["mean_actor_reward_value"],
            mean_actor_safety_risk=update_summary["mean_actor_safety_risk"],
            mean_reward_critic_disagreement=update_summary[
                "mean_reward_critic_disagreement"
            ],
            mean_safety_critic_disagreement=update_summary[
                "mean_safety_critic_disagreement"
            ],
            mean_actor_motion_ratio=update_summary["mean_actor_motion_ratio"],
            maximum_actor_motion_ratio=update_summary[
                "maximum_actor_motion_ratio"
            ],
            mean_actor_entropy=update_summary["mean_actor_entropy"],
            mean_actor_motion_log_standard_deviation=update_summary[
                "mean_actor_motion_log_standard_deviation"
            ],
            mean_actor_gripper_log_standard_deviation=update_summary[
                "mean_actor_gripper_log_standard_deviation"
            ],
            mean_actor_augmentation_consistency_loss=update_summary[
                "mean_actor_augmentation_consistency_loss"
            ],
            actor_learning_rate=update_summary["actor_learning_rate"],
            frontier_reset=frontier_entry is not None,
            frontier_source_episode=(
                frontier_entry.source_episode if frontier_entry else -1
            ),
            frontier_source_step=(
                frontier_entry.source_step if frontier_entry else -1
            ),
            environment_reset_seed=prepared.reset_seed,
            frontier_source_signature=(
                frontier_entry.signature if frontier_entry else -1
            ),
            frontier_reset_contact_steps=0,
            frontier_reset_validated=prepared.validated,
            frontier_reset_reproduced=prepared.reproduced,
            frontier_reset_applied=prepared.applied,
            environment_terminated=environment_terminated,
            environment_truncated=environment_truncated,
            **physical_progress_record_fields(buffers.next_states),
        )

    def _select_action(
        self,
        actor_input: VLAActorInput,
        previous: DualArmAction | None,
        *,
        random_phase: bool,
        refresh_random: bool,
    ) -> DualArmAction:
        if random_phase and previous is not None and not refresh_random:
            return previous
        if random_phase:
            vector = self.explorer.sample_random()
        else:
            tensors = actor_input_tensors(
                actor_input, device=self.trainer.device
            )
            with torch.inference_mode():
                vector = self.trainer.sample_actor_action(tensors)[
                    0, 0
                ].cpu().numpy()
            vector = self.explorer.perturb(vector)
        return DualArmAction.from_vector(vector)

    def _append_transition(
        self,
        buffers: _EpisodeBuffers,
        actor_input: VLAActorInput,
        next_actor_input: VLAActorInput,
        state,
        next_state,
        proposed_action: DualArmAction,
        applied_action: DualArmAction,
        safety_intervened: bool,
        reward: float,
        done: bool,
        snapshot: PhysicalStateSnapshot,
    ) -> None:
        buffers.actor_inputs.append(actor_input)
        buffers.next_actor_inputs.append(next_actor_input)
        buffers.states.append(state.critic_state)
        buffers.next_states.append(next_state.critic_state)
        buffers.actions.append(applied_action.vector())
        buffers.proposed_actions.append(proposed_action.vector())
        buffers.safety_costs.append(float(safety_intervened))
        buffers.rewards.append(float(reward))
        buffers.done.append(float(done))
        buffers.snapshots.append(snapshot)

    def _learning_frontier_candidates(
        self,
        task_id: str,
        episode_index: int,
        buffers: _EpisodeBuffers,
        td_errors: torch.Tensor,
        terminated_failure: bool,
    ) -> list[LearningFrontierCandidate]:
        boundary_step = failure_boundary_step(
            buffers.safety_costs, terminated_failure=terminated_failure
        )
        improvement_speeds = reward_improvement_speeds(buffers.rewards)
        candidates = []
        for step, (snapshot, state, safety_cost) in enumerate(
            zip(
                buffers.snapshots,
                buffers.next_states,
                buffers.safety_costs,
                strict=True,
            )
        ):
            candidates.append(
                LearningFrontierCandidate(
                    snapshot,
                    tuple(float(item) for item in state),
                    LearningSignal(
                        self.frontier.state_novelty(task_id, state),
                        float(td_errors[step]),
                        improvement_speeds[step],
                        float(step == boundary_step),
                        safe=safety_cost <= 0.0,
                    ),
                    episode_index,
                    step,
                )
            )
        return candidates

    def _autonomous_episode(
        self,
        buffers: _EpisodeBuffers,
        success: bool,
        legal_transforms: tuple[str, ...],
    ) -> AutonomousEpisode:
        targets = build_n_step_targets(
            buffers.rewards,
            buffers.done,
            horizon=self.config.n_step_horizon,
            discount=self.rl_config.discount,
        )
        next_inputs = [
            buffers.next_actor_inputs[index] for index in targets.next_indices
        ]
        next_states = [buffers.next_states[index] for index in targets.next_indices]
        batch = AsymmetricRLBatch(
            actor_inputs=stack_actor_inputs(buffers.actor_inputs),
            next_actor_inputs=stack_actor_inputs(next_inputs),
            privileged_state=torch.tensor(buffers.states, dtype=torch.float32),
            next_privileged_state=torch.tensor(next_states, dtype=torch.float32),
            action_chunks=torch.tensor(buffers.actions, dtype=torch.float32)[:, None],
            stop_decisions=torch.zeros(len(buffers.actions), 1),
            rewards=torch.tensor(targets.rewards, dtype=torch.float32),
            done=torch.tensor(targets.done, dtype=torch.float32),
            augmentation_transform_indices=torch.zeros(
                len(buffers.rewards), dtype=torch.int64
            ),
            proposed_action_chunks=torch.tensor(
                buffers.proposed_actions, dtype=torch.float32
            )[:, None],
            safety_costs=torch.tensor(buffers.safety_costs, dtype=torch.float32),
            bootstrap_discounts=torch.tensor(
                targets.bootstrap_discounts, dtype=torch.float32
            ),
        )
        improvements = torch.tensor(
            reward_improvement_speeds(buffers.rewards), dtype=torch.float32
        )
        return AutonomousEpisode(
            batch,
            success,
            legal_transforms,
            reward_improvements=improvements,
        )

    def _update_after_episode(self, episode_steps: int) -> dict[str, float]:
        if self.replay.size < max(self.config.learning_starts, self.config.batch_size):
            return _summarize_updates([])
        count = max(1, round(episode_steps * self.config.updates_per_environment_step))
        metrics = []
        for _ in range(count):
            batch = self.replay.sample(
                self.config.batch_size,
                failure_fraction=self.config.failure_replay_fraction,
                discovery_fraction=self.config.discovery_replay_fraction,
                progress_fraction=self.config.progress_replay_fraction,
                safety_fraction=self.config.safety_replay_fraction,
            )
            metrics.append(self.trainer.update(batch))
        return _summarize_updates(metrics)


def _summarize_updates(metrics: list[Mapping[str, float]]) -> dict[str, float]:
    actor = [item for item in metrics if item["actor_updated"] > 0.5]

    def mean(items: list[Mapping[str, float]], name: str) -> float:
        return sum(item[name] for item in items) / len(items) if items else 0.0

    return {
        "actor_updates": float(len(actor)),
        "mean_critic_loss": mean(metrics, "critic_loss"),
        "mean_safety_loss": mean(metrics, "safety_loss"),
        "mean_actor_loss": mean(actor, "actor_loss"),
        "mean_actor_reward_value": mean(actor, "actor_reward_value"),
        "mean_actor_safety_risk": mean(actor, "actor_safety_risk"),
        "mean_reward_critic_disagreement": mean(
            actor, "reward_critic_disagreement"
        ),
        "mean_safety_critic_disagreement": mean(
            actor, "safety_critic_disagreement"
        ),
        "mean_actor_motion_ratio": mean(actor, "actor_motion_mean_ratio"),
        "maximum_actor_motion_ratio": max(
            (item["actor_motion_max_ratio"] for item in actor), default=0.0
        ),
        "mean_actor_entropy": mean(actor, "actor_entropy"),
        "mean_actor_motion_log_standard_deviation": mean(
            actor, "actor_motion_log_standard_deviation"
        ),
        "mean_actor_gripper_log_standard_deviation": mean(
            actor, "actor_gripper_log_standard_deviation"
        ),
        "mean_actor_augmentation_consistency_loss": mean(
            actor, "augmentation_consistency_loss"
        ),
        "actor_learning_rate": mean(actor, "actor_learning_rate"),
    }


def _mean_signal(
    candidates: list[LearningFrontierCandidate], name: str
) -> float:
    values = [
        float(getattr(candidate.signal, name))
        for candidate in candidates
        if candidate.signal.safe
    ]
    return sum(values) / len(values) if values else 0.0
