from pathlib import Path

import torch

from hwr.train.foundation_setup import build_foundation_learning_stack


ROOT = Path(__file__).resolve().parents[1]


def test_formal_foundation_stack_is_built_from_one_consistent_config_set() -> None:
    stack = build_foundation_learning_stack(ROOT / "configs/foundation", seed=17)
    trainer = stack.trainer

    assert trainer.visual_student.config.formal is True
    assert trainer.visual_student.config.image_size == 160
    assert trainer.visual_student.config.visual_history == 4
    assert trainer.world_model.config.action_dimension == 16
    assert trainer.actor.config.latent_dimension == trainer.world_model.config.feature_dimension
    assert trainer.visual_objective.config.vision_language_dimension == 768
    assert trainer.visual_objective.config.dense_vision_dimension == 384
    assert trainer.imagination.action_scaling == stack.action_scaling
    assert sum(parameter.numel() for parameter in trainer.visual_student.parameters()) >= 20_000_000


def test_formal_foundation_stack_initialization_is_seed_reproducible() -> None:
    first = build_foundation_learning_stack(ROOT / "configs/foundation", seed=23)
    torch.rand(31)
    second = build_foundation_learning_stack(ROOT / "configs/foundation", seed=23)

    first_parameters = first.trainer.actor.state_dict()
    second_parameters = second.trainer.actor.state_dict()
    assert first_parameters.keys() == second_parameters.keys()
    assert all(
        torch.equal(first_parameters[name], second_parameters[name])
        for name in first_parameters
    )
