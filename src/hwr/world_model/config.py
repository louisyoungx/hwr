"""Configuration for the project-owned recurrent state-space model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hwr.core.embodied import DUAL_ARM_ACTION_DIM


@dataclass(frozen=True)
class WorldModelConfig:
    visual_dimension: int = 256
    language_dimension: int = 1024
    proprioception_dimension: int = 37
    action_dimension: int = DUAL_ARM_ACTION_DIM
    observation_embedding_dimension: int = 512
    deterministic_dimension: int = 512
    stochastic_variables: int = 32
    stochastic_classes: int = 32
    hidden_dimension: int = 512
    prior_ensemble: int = 5
    reward_bins: int = 255
    reward_symlog_limit: float = 20.0
    categorical_unimix: float = 0.01
    formal: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.visual_dimension,
            self.language_dimension,
            self.proprioception_dimension,
            self.action_dimension,
            self.observation_embedding_dimension,
            self.deterministic_dimension,
            self.stochastic_variables,
            self.stochastic_classes,
            self.hidden_dimension,
            self.prior_ensemble,
            self.reward_bins,
        )
        if min(dimensions) <= 0:
            raise ValueError("world model dimensions must be positive")
        if self.reward_bins < 3 or not self.reward_bins % 2:
            raise ValueError("world model reward bins must be odd and at least three")
        if self.reward_symlog_limit <= 0.0:
            raise ValueError("world model reward symlog limit must be positive")
        if not 0.0 <= self.categorical_unimix < 1.0:
            raise ValueError("world model categorical unimix must be in [0, 1)")
        if self.formal and self.action_dimension != DUAL_ARM_ACTION_DIM:
            raise ValueError("formal world model requires the canonical 16-D action")

    @property
    def stochastic_dimension(self) -> int:
        return self.stochastic_variables * self.stochastic_classes

    @property
    def feature_dimension(self) -> int:
        return self.deterministic_dimension + self.stochastic_dimension

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
