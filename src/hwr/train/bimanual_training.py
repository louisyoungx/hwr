"""Local no-demonstration training loop for the three bimanual household tasks."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from hwr.adapters.mujoco import (
    BimanualMujocoBinding,
    MujocoBimanualTaskBackend,
    load_bimanual_mujoco_bindings,
)
from hwr.core.embodied import DualArmAction, DualArmActionFrame
from hwr.policy.bimanual_input import (
    BimanualActorInputPipeline,
    BimanualInputConfig,
    actor_input_tensors,
    stack_actor_inputs,
)
from hwr.policy.privileged_critic import PrivilegedCriticConfig
from hwr.policy.vla_actions import bounded_vla_actions
from hwr.policy.vla_input import VLAActorInput
from hwr.policy.vla_model import VLAActorConfig, VLAActorModel
from hwr.perception import FrozenNgramLanguageConfig, FrozenNgramLanguageEncoder
from hwr.tasks import BimanualTaskSpec, load_bimanual_task_specs
from hwr.train.asymmetric_rl import (
    AsymmetricActorCriticTrainer,
    AsymmetricRLBatch,
    AsymmetricRLConfig,
)
from hwr.train.curriculum import AutomaticCurriculum, CurriculumConfig
from hwr.train.goal_replay import GoalConditionedReplayBuffer, GoalEpisode


@dataclass(frozen=True)
class BimanualRLTrainingConfig:
    episodes: int = 120
    episode_step_limit: int = 240
    replay_capacity: int = 80_000
    batch_size: int = 64
    learning_starts: int = 512
    updates_per_environment_step: float = 0.25
    initial_random_episodes: int = 9
    random_action_hold_steps: int = 4
    exploration_noise: float = 0.18
    failure_replay_fraction: float = 0.5
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
            self.episode_step_limit,
            self.replay_capacity,
            self.batch_size,
            self.learning_starts,
            self.random_action_hold_steps,
            self.raw_image_width,
            self.raw_image_height,
            self.image_width,
            self.image_height,
            self.point_count,
            self.language_dim,
            self.hidden_dim,
            self.attention_heads,
            self.transformer_layers,
        )
        if min(positive) <= 0 or self.initial_random_episodes < 0:
            raise ValueError("bimanual training dimensions must be positive")
        fractions = (
            self.updates_per_environment_step,
            self.exploration_noise,
            self.failure_replay_fraction,
        )
        if min(fractions) < 0 or self.failure_replay_fraction > 1:
            raise ValueError("bimanual training fractions are invalid")

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
    stable_steps: int
    minimum_left_reach_distance: float
    minimum_right_reach_distance: float
    curriculum_level: float
    replay_size: int
    updates: int


@dataclass
class BimanualTrainingResult:
    config: BimanualRLTrainingConfig
    actor_config: VLAActorConfig
    rl_config: AsymmetricRLConfig
    trainer: AsymmetricActorCriticTrainer
    replay: GoalConditionedReplayBuffer
    curriculum: AutomaticCurriculum
    records: list[TrainingEpisodeRecord]
    language_encoder: FrozenNgramLanguageEncoder
    preprocess_fingerprint: str
    environment_steps: int
    numpy_rng_state: Mapping[str, object]
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
    rewards: list[float]
    done: list[float]

    @classmethod
    def empty(cls) -> "_EpisodeBuffers":
        return cls([], [], [], [], [], [], [], [], [], [])


class BimanualTrainingRunner:
    """Collect random/Actor experience and update without an action-label source."""

    def __init__(
        self,
        tasks: Mapping[str, BimanualTaskSpec],
        bindings: Mapping[str, BimanualMujocoBinding],
        config: BimanualRLTrainingConfig,
    ) -> None:
        if set(tasks) != set(bindings) or len(tasks) != 3:
            raise ValueError("training requires exactly three matching bimanual tasks")
        self.tasks = dict(tasks)
        self.bindings = dict(bindings)
        self.config = config
        self.task_ids = tuple(sorted(tasks))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
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
        self.rl_config = AsymmetricRLConfig(behavior_regularization=0.0)
        actor = VLAActorModel(self.actor_config)
        self.trainer = AsymmetricActorCriticTrainer(
            actor,
            PrivilegedCriticConfig(60, 1, hidden_dim=max(128, config.hidden_dim)),
            self.rl_config,
            device=config.device,
        )
        self.replay = GoalConditionedReplayBuffer(
            config.replay_capacity, seed=config.seed
        )
        self.curriculum = AutomaticCurriculum(
            self.task_ids, CurriculumConfig(initial_level=0.1)
        )
        self.records: list[TrainingEpisodeRecord] = []
        self._environment_steps = 0

    def train(
        self,
        on_episode: Callable[[BimanualTrainingResult], None] | None = None,
    ) -> BimanualTrainingResult:
        environments = {
            task_id: MujocoBimanualTaskBackend(
                self.tasks[task_id],
                self.bindings[task_id],
                camera_width=self.config.raw_image_width,
                camera_height=self.config.raw_image_height,
            )
            for task_id in self.task_ids
        }
        try:
            for episode_index in range(len(self.records), self.config.episodes):
                task_id = self.task_ids[episode_index % len(self.task_ids)]
                record = self._run_episode(
                    episode_index, task_id, environments[task_id]
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
            self.records,
            self.language,
            self.pipeline.preprocessor.fingerprint,
            self._environment_steps,
            self.rng.bit_generator.state,
            torch.get_rng_state(),
        )

    def load_training_state(self, value: Mapping[str, object]) -> None:
        """Restore all learning state before continuing at the next episode."""
        self.trainer.load_state_dict(value["trainer"])
        self.replay.load_state_dict(value["replay"])
        self.curriculum.load_state_dict(value["curriculum"])
        self.records = [
            TrainingEpisodeRecord(**record) for record in value["records"]
        ]
        self._environment_steps = int(value["environment_steps"])
        self.rng.bit_generator.state = value["numpy_rng_state"]
        torch.set_rng_state(value["torch_rng_state"])
        if len(self.records) > self.config.episodes:
            raise ValueError("resume checkpoint exceeds configured total episodes")

    def _run_episode(
        self,
        episode_index: int,
        task_id: str,
        environment: MujocoBimanualTaskBackend,
    ) -> TrainingEpisodeRecord:
        seed = self.config.seed + episode_index * 104729
        level = self.curriculum.level(task_id)
        environment.set_curriculum_level(level)
        observation = environment.reset(seed=seed, task_id=task_id)
        self.pipeline.reset()
        actor_input = self.pipeline.build(observation)
        state = environment.privileged_training_state()
        buffers = _EpisodeBuffers.empty()
        previous_action: DualArmAction | None = None
        total_reward = 0.0
        success = False
        for step in range(self.config.episode_step_limit):
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
            self.pipeline.record_action(action)
            outcome = environment.apply(frame)
            next_input = self.pipeline.build(outcome.observation)
            next_state = environment.privileged_training_state()
            terminal = outcome.terminated or outcome.truncated
            limit = step + 1 >= self.config.episode_step_limit
            self._append_transition(
                buffers,
                actor_input,
                next_input,
                state,
                next_state,
                action,
                outcome.reward,
                terminal or limit,
            )
            total_reward += outcome.reward
            self._environment_steps += 1
            actor_input, state = next_input, next_state
            observation = outcome.observation
            previous_action = action
            success = bool(environment.result() and environment.result().success)
            if terminal or limit:
                break
        audit = environment.task_audit()
        self.replay.add_episode(
            self._goal_episode(buffers, success, self.tasks[task_id].objective == "carry_payload")
        )
        updates = self._update_after_episode(len(buffers.rewards))
        self.curriculum.record(
            task_id,
            success=success,
            severe_collision=int(audit["severe_collision_count"]) > 0,
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
            stable_steps=int(audit["stable_steps"]),
            minimum_left_reach_distance=min(state[24] for state in buffers.states),
            minimum_right_reach_distance=min(state[25] for state in buffers.states),
            curriculum_level=level,
            replay_size=self.replay.size,
            updates=self.trainer.update_count,
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
            vector = np.concatenate(
                (
                    self.rng.uniform((-0.20, -0.6), (0.20, 0.6)),
                    self.rng.uniform(-0.8, 0.8, 12),
                    self.rng.uniform(0.0, 1.0, 2),
                )
            )
        else:
            tensors = actor_input_tensors(
                actor_input, device=self.trainer.device
            )
            with torch.inference_mode():
                output = self.trainer.actor(tensors)
                vector = bounded_vla_actions(
                    output, self.rl_config.action_scaling()
                )[0, 0].cpu().numpy()
            noise = self.rng.normal(0.0, self.config.exploration_noise, 14)
            scales = np.asarray((0.45, 1.0, *(1.2,) * 12))
            vector[:14] = np.clip(vector[:14] + noise * scales, -scales, scales)
            vector[14:] = np.clip(
                vector[14:] + self.rng.normal(0.0, 0.08, 2), 0.0, 1.0
            )
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
        action: DualArmAction,
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
        buffers.actions.append(action.vector())
        buffers.rewards.append(float(reward))
        buffers.done.append(float(done))

    def _goal_episode(
        self, buffers: _EpisodeBuffers, success: bool, mirrorable: bool
    ) -> GoalEpisode:
        batch = AsymmetricRLBatch(
            actor_inputs=stack_actor_inputs(buffers.actor_inputs),
            next_actor_inputs=stack_actor_inputs(buffers.next_actor_inputs),
            privileged_state=torch.tensor(buffers.states, dtype=torch.float32),
            next_privileged_state=torch.tensor(
                buffers.next_states, dtype=torch.float32
            ),
            action_chunks=torch.tensor(buffers.actions, dtype=torch.float32)[:, None],
            stop_decisions=torch.zeros(len(buffers.actions), 1),
            rewards=torch.tensor(buffers.rewards, dtype=torch.float32),
            done=torch.tensor(buffers.done, dtype=torch.float32),
        )
        return GoalEpisode(
            batch,
            torch.tensor(buffers.achieved, dtype=torch.float32),
            torch.tensor(buffers.next_achieved, dtype=torch.float32),
            torch.tensor(buffers.desired, dtype=torch.float32),
            success,
            mirrorable,
        )

    def _update_after_episode(self, episode_steps: int) -> int:
        if self.replay.size < max(self.config.learning_starts, self.config.batch_size):
            return 0
        count = max(1, round(episode_steps * self.config.updates_per_environment_step))
        for _ in range(count):
            batch = self.replay.sample(
                self.config.batch_size,
                failure_fraction=self.config.failure_replay_fraction,
            )
            self.trainer.update(batch)
        return count


def load_default_bimanual_training_catalogs(
    root: Path,
) -> tuple[dict[str, BimanualTaskSpec], dict[str, BimanualMujocoBinding]]:
    tasks = load_bimanual_task_specs(
        root / "configs/tasks/bimanual_household_v1.json"
    )
    bindings = load_bimanual_mujoco_bindings(
        root / "configs/adapters/mujoco/bimanual_household_v1.json",
        root=root,
    )
    return tasks, bindings
