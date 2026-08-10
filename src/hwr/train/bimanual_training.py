"""Local no-demonstration training loop for the three bimanual household tasks."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Protocol

import numpy as np
import torch

from hwr.core.embodied import DualArmAction, DualArmActionFrame
from hwr.core.runtime import SnapshotRuntimeBackend
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
from hwr.train.action_exploration import (
    TemporalActionExplorer,
    TemporalExplorationConfig,
)
from hwr.train.curriculum import AutomaticCurriculum, CurriculumConfig
from hwr.train.frontier_curriculum import (
    FrontierCurriculumConfig,
    OutcomeFrontierCurriculum,
)
from hwr.train.goal_replay import GoalEpisode
from hwr.train.n_step import build_n_step_targets
from hwr.train.task_replay import TaskPartitionedGoalReplayBuffer
from hwr.train.task_sampling import OutcomeAdaptiveTaskSampler, TaskOutcome


@dataclass(frozen=True)
class BimanualRLTrainingConfig:
    episodes: int = 120
    episode_step_limit: int | None = None
    replay_capacity: int = 80_000
    batch_size: int = 64
    learning_starts: int = 512
    updates_per_environment_step: float = 0.25
    initial_random_episodes: int = 9
    random_action_hold_steps: int = 8
    exploration_noise: float = 0.18
    exploration_correlation: float = 0.85
    action_smoothing: float = 0.65
    gripper_exploration_probability: float = 0.35
    gripper_exploration_hold_steps: int = 16
    policy_gripper_hold_steps: int = 12
    reflection_coupled_exploration_probability: float = 0.60
    paired_gripper_exploration_probability: float = 0.60
    global_random_burst_probability: float = 0.01
    global_random_burst_steps: int = 8
    frontier_reset_probability: float = 0.50
    frontier_capacity_per_task: int = 16
    failure_replay_fraction: float = 0.5
    discovery_replay_fraction: float = 0.35
    safety_replay_fraction: float = 0.15
    n_step_horizon: int = 8
    seed: int = 20260810
    device: str = "cpu"
    raw_image_width: int = 64
    raw_image_height: int = 48
    image_width: int = 32
    image_height: int = 24
    point_count: int = 32
    language_dim: int = 64
    hidden_dim: int = 64
    attention_heads: int = 4
    transformer_layers: int = 1

    def __post_init__(self) -> None:
        positive = (
            self.episodes,
            self.replay_capacity,
            self.batch_size,
            self.learning_starts,
            self.random_action_hold_steps,
            self.gripper_exploration_hold_steps,
            self.policy_gripper_hold_steps,
            self.global_random_burst_steps,
            self.frontier_capacity_per_task,
            self.raw_image_width,
            self.raw_image_height,
            self.image_width,
            self.image_height,
            self.point_count,
            self.language_dim,
            self.hidden_dim,
            self.attention_heads,
            self.transformer_layers,
            self.n_step_horizon,
        )
        if min(positive) <= 0 or self.initial_random_episodes < 0:
            raise ValueError("bimanual training dimensions must be positive")
        if self.episode_step_limit is not None and self.episode_step_limit <= 0:
            raise ValueError("bimanual episode step limit must be positive when set")
        fractions = (
            self.updates_per_environment_step,
            self.exploration_noise,
            self.exploration_correlation,
            self.action_smoothing,
            self.gripper_exploration_probability,
            self.reflection_coupled_exploration_probability,
            self.paired_gripper_exploration_probability,
            self.global_random_burst_probability,
            self.frontier_reset_probability,
            self.failure_replay_fraction,
            self.discovery_replay_fraction,
            self.safety_replay_fraction,
        )
        if min(fractions) < 0 or any(value > 1 for value in fractions[1:]):
            raise ValueError("bimanual training fractions are invalid")
        replay_fraction = (
            self.failure_replay_fraction
            + self.discovery_replay_fraction
            + self.safety_replay_fraction
        )
        if replay_fraction > 1.0 + 1e-9:
            raise ValueError("bimanual replay fractions exceed one batch")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingEpisodeRecord:
    episode: int
    task_id: str
    seed: int
    steps: int
    reward: float
    success: bool
    severe_collisions: int
    maximum_concurrent_steps: int
    left_contact_steps: int
    right_contact_steps: int
    simultaneous_contact_steps: int
    stable_steps: int
    minimum_left_reach_distance: float
    minimum_right_reach_distance: float
    minimum_worst_side_reach_distance: float
    curriculum_level: float
    replay_size: int
    updates: int
    bilateral_near_steps: int = 0
    maximum_bilateral_near_steps: int = 0
    safety_interventions: int = 0
    sampling_probability: float = 1.0 / 3.0
    actor_updates: int = 0
    mean_critic_loss: float = 0.0
    mean_safety_loss: float = 0.0
    mean_actor_loss: float = 0.0
    mean_actor_reward_value: float = 0.0
    mean_actor_safety_risk: float = 0.0
    mean_reward_critic_disagreement: float = 0.0
    mean_safety_critic_disagreement: float = 0.0
    mean_actor_motion_ratio: float = 0.0
    maximum_actor_motion_ratio: float = 0.0
    mean_actor_entropy: float = 0.0
    mean_actor_motion_log_standard_deviation: float = 0.0
    mean_actor_gripper_log_standard_deviation: float = 0.0
    frontier_reset: bool = False
    frontier_source_episode: int = -1
    frontier_source_step: int = -1


@dataclass
class BimanualTrainingResult:
    config: BimanualRLTrainingConfig
    actor_config: VLAActorConfig
    rl_config: AsymmetricRLConfig
    trainer: AsymmetricActorCriticTrainer
    replay: TaskPartitionedGoalReplayBuffer
    curriculum: AutomaticCurriculum
    frontier: OutcomeFrontierCurriculum
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
    achieved: list[tuple[float, ...]]
    next_achieved: list[tuple[float, ...]]
    desired: list[tuple[float, ...]]
    actions: list[tuple[float, ...]]
    proposed_actions: list[tuple[float, ...]]
    safety_costs: list[float]
    rewards: list[float]
    done: list[float]

    @classmethod
    def empty(cls) -> "_EpisodeBuffers":
        return cls([], [], [], [], [], [], [], [], [], [], [], [])


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
        self.rl_config = AsymmetricRLConfig(behavior_regularization=0.0)
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
            separate_gripper_head=True,
        )
        actor = VLAActorModel(self.actor_config)
        self.trainer = AsymmetricActorCriticTrainer(
            actor,
            PrivilegedCriticConfig(60, 1, hidden_dim=max(128, config.hidden_dim)),
            self.rl_config,
            device=config.device,
        )
        self.replay = TaskPartitionedGoalReplayBuffer(
            config.replay_capacity, self.task_ids, seed=config.seed
        )
        self.curriculum = AutomaticCurriculum(
            self.task_ids, CurriculumConfig(initial_level=0.1)
        )
        self.frontier = OutcomeFrontierCurriculum(
            self.task_ids,
            FrontierCurriculumConfig(
                capacity_per_task=config.frontier_capacity_per_task,
                reset_probability=config.frontier_reset_probability,
            ),
        )
        self.task_sampler = OutcomeAdaptiveTaskSampler(self.task_ids)
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
        observation = environment.reset(
            seed=seed,
            task_id=task_id,
            initial_state=(frontier_entry.snapshot if frontier_entry else None),
        )
        self.explorer.reset()
        self.pipeline.reset()
        actor_input = self.pipeline.build(observation)
        state = environment.privileged_training_state()
        buffers = _EpisodeBuffers.empty()
        previous_action: DualArmAction | None = None
        total_reward = 0.0
        success = False
        safety_interventions = 0
        step_limit = self.config.episode_step_limit or self.tasks[task_id].max_steps
        for step in range(step_limit):
            random_phase = episode_index < self.config.initial_random_episodes
            action = self._select_action(
                actor_input,
                previous_action,
                random_phase=random_phase,
                refresh_random=step % self.config.random_action_hold_steps == 0,
            )
            frame = self._action_frame(
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
            if not bool(outcome.info["safety_intervened"]):
                frontier_outcome = self.frontier.outcome_from_metrics(
                    next_state.metrics
                )
                if self.frontier.qualifies(frontier_outcome):
                    self.frontier.consider(
                        task_id,
                        environment.capture_state_snapshot(),
                        frontier_outcome,
                        source_episode=episode_index,
                        source_step=step,
                    )
            terminal = outcome.terminated or outcome.truncated
            limit = step + 1 >= step_limit
            self._append_transition(
                buffers,
                actor_input,
                next_input,
                state,
                next_state,
                action,
                applied_action,
                bool(outcome.info["safety_intervened"]),
                outcome.reward,
                terminal or limit,
            )
            total_reward += outcome.reward
            self._environment_steps += 1
            actor_input, state = next_input, next_state
            observation = outcome.observation
            previous_action = applied_action
            success = bool(environment.result() and environment.result().success)
            if terminal or limit:
                break
        audit = environment.task_audit()
        self.replay.add_episode(
            task_id,
            self._goal_episode(
                buffers,
                success,
                self.tasks[task_id].objective == "carry_payload",
            ),
        )
        update_summary = self._update_after_episode(len(buffers.rewards))
        self.curriculum.record(
            task_id,
            success=success,
            severe_collision=int(audit["severe_collision_count"]) > 0,
        )
        self.task_sampler.record(
            task_id,
            TaskOutcome(
                int(audit["left_contact_steps"]),
                int(audit["right_contact_steps"]),
                int(audit["simultaneous_contact_steps"]),
                min(state[24] for state in buffers.states),
                min(state[25] for state in buffers.states),
                min(max(state[24], state[25]) for state in buffers.states),
            ),
        )
        bilateral_near_steps, maximum_bilateral_near_steps = (
            _bilateral_near_statistics(buffers.states)
        )
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
            frontier_reset=frontier_entry is not None,
            frontier_source_episode=(
                frontier_entry.source_episode if frontier_entry else -1
            ),
            frontier_source_step=(
                frontier_entry.source_step if frontier_entry else -1
            ),
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

    def _action_frame(
        self, timestamp_ns: int, action: DualArmAction, *, source: str
    ) -> DualArmActionFrame:
        period_ns = round(1_000_000_000 / 20.0)
        return DualArmActionFrame(
            timestamp_ns,
            timestamp_ns,
            timestamp_ns + 2 * period_ns,
            source,
            action,
        )

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
    ) -> None:
        buffers.actor_inputs.append(actor_input)
        buffers.next_actor_inputs.append(next_actor_input)
        buffers.states.append(state.critic_state)
        buffers.next_states.append(next_state.critic_state)
        buffers.achieved.append(state.achieved_goal)
        buffers.next_achieved.append(next_state.achieved_goal)
        buffers.desired.append(state.desired_goal)
        buffers.actions.append(applied_action.vector())
        buffers.proposed_actions.append(proposed_action.vector())
        buffers.safety_costs.append(float(safety_intervened))
        buffers.rewards.append(float(reward))
        buffers.done.append(float(done))

    def _goal_episode(
        self, buffers: _EpisodeBuffers, success: bool, mirrorable: bool
    ) -> GoalEpisode:
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
        next_achieved = [
            buffers.next_achieved[index] for index in targets.next_indices
        ]
        batch = AsymmetricRLBatch(
            actor_inputs=stack_actor_inputs(buffers.actor_inputs),
            next_actor_inputs=stack_actor_inputs(next_inputs),
            privileged_state=torch.tensor(buffers.states, dtype=torch.float32),
            next_privileged_state=torch.tensor(next_states, dtype=torch.float32),
            action_chunks=torch.tensor(buffers.actions, dtype=torch.float32)[:, None],
            stop_decisions=torch.zeros(len(buffers.actions), 1),
            rewards=torch.tensor(targets.rewards, dtype=torch.float32),
            done=torch.tensor(targets.done, dtype=torch.float32),
            proposed_action_chunks=torch.tensor(
                buffers.proposed_actions, dtype=torch.float32
            )[:, None],
            safety_costs=torch.tensor(buffers.safety_costs, dtype=torch.float32),
            bootstrap_discounts=torch.tensor(
                targets.bootstrap_discounts, dtype=torch.float32
            ),
        )
        return GoalEpisode(
            batch,
            torch.tensor(buffers.achieved, dtype=torch.float32),
            torch.tensor(next_achieved, dtype=torch.float32),
            torch.tensor(buffers.desired, dtype=torch.float32),
            success,
            mirrorable,
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
                safety_fraction=self.config.safety_replay_fraction,
            )
            metrics.append(self.trainer.update(batch))
        return _summarize_updates(metrics)


def _bilateral_near_statistics(
    states: list[tuple[float, ...]], threshold: float = 0.10
) -> tuple[int, int]:
    total = 0
    current = 0
    longest = 0
    for state in states:
        near = max(state[24], state[25]) <= threshold
        total += int(near)
        current = current + 1 if near else 0
        longest = max(longest, current)
    return total, longest


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
    }
