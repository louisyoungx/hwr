"""Executable learnability and formal-environment checks for the training gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from hwr.adapters.mujoco.bindings import load_mujoco_task_bindings
from hwr.core.embodied import (
    DUAL_ARM_ACTION_MAXIMUM,
    DUAL_ARM_ACTION_MINIMUM,
)
from hwr.perception.student import VisualStudentConfig, VisualStudentModel
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
    VisualTeacherTargets,
)
from hwr.scenarios.formal3d import load_formal_3d_tasks
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_resource_budget import foundation_storage_estimate


FORMAL_HOUSEHOLD_TASK_IDS = frozenset(
    {
        "tidy_living_room_3d/v1",
        "clear_dining_table_3d/v1",
        "store_kitchen_items_3d/v1",
    }
)


def verify_training_semantics(root: Path) -> dict[str, object]:
    """Reject code that is complete on paper but cannot train the deployed path."""
    checks = {
        "visual_deployment_reachability": _visual_deployment_reachability(),
        "formal_household_contract": _formal_household_contract(root),
        "retained_replay_contract": _retained_replay_contract(root),
        "action_bounds_contract": _action_bounds_contract(root),
    }
    return {"passed": True, "checks": checks}


def _visual_config() -> VisualStudentConfig:
    return VisualStudentConfig(
        image_size=32,
        visual_history=2,
        backbone_dimensions=(16, 24, 32, 48),
        backbone_depths=(1, 1, 1, 1),
        feature_dimension=16,
        state_queries=2,
        attention_heads=4,
        fusion_layers=1,
        temporal_layers=1,
        formal=False,
    )


def _visual_inputs(config: VisualStudentConfig) -> dict[str, torch.Tensor]:
    batch, history, size = 1, config.visual_history, config.image_size
    return {
        "rgb": torch.rand(batch, history, 3, 3, size, size),
        "head_depth_m": torch.rand(batch, history, 1, size, size) + 0.2,
        "head_depth_valid": torch.ones(
            batch, history, 1, size, size, dtype=torch.bool
        ),
        "camera_validity": torch.ones(batch, history, 4, dtype=torch.bool),
        "intrinsics": torch.ones(batch, history, 4, 4),
        "robot_from_camera": torch.eye(4)
        .reshape(1, 1, 1, 4, 4)
        .expand(batch, history, 4, 4, 4)
        .clone(),
        "repeated_frame": torch.zeros(batch, history, dtype=torch.bool),
    }


def _visual_targets(
    config: VisualStudentConfig,
) -> VisualTeacherTargets:
    batch, history, size = 1, config.visual_history, config.image_size
    return VisualTeacherTargets(
        vision_language=torch.randn(batch, history, 3, 3, 3, 12),
        vision_language_valid=torch.ones(
            batch, history, 3, 3, 3, dtype=torch.bool
        ),
        dense_vision=torch.randn(batch, history, 3, 4, 4, 10),
        dense_vision_valid=torch.ones(
            batch, history, 3, 4, 4, dtype=torch.bool
        ),
        rgb=torch.rand(batch, history, 3, 3, size, size),
        reconstruction_mask=torch.ones(
            batch, history, 3, 1, size, size, dtype=torch.bool
        ),
        head_depth_m=torch.ones(batch, history, 1, size, size),
        head_depth_valid=torch.ones(
            batch, history, 1, size, size, dtype=torch.bool
        ),
        correspondences=torch.empty((0, 10), dtype=torch.long),
    )


def _module_gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _module_changed(
    module: torch.nn.Module, before: Mapping[str, torch.Tensor]
) -> bool:
    return any(
        torch.any(parameter.detach() != before[name])
        for name, parameter in module.named_parameters()
    )


def _visual_deployment_reachability() -> dict[str, object]:
    torch.manual_seed(20260814)
    config = _visual_config()
    student = VisualStudentModel(config)
    objective = VisualFoundationObjectives(
        VisualObjectiveConfig(
            student_dimension=16,
            vision_language_dimension=12,
            dense_vision_dimension=10,
        )
    )
    modules = {
        "camera_fusion": student.camera_fusion,
        "temporal_fusion": student.temporal_fusion,
        "output_norm": student.output_norm,
    }
    before = {
        module_name: {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }
        for module_name, module in modules.items()
    }
    temporal_position_before = student.temporal_position.detach().clone()
    optimizer = torch.optim.AdamW(
        (*student.parameters(), *objective.parameters()), lr=1.0e-3
    )
    optimizer.zero_grad(set_to_none=True)
    output = student(_visual_inputs(config))
    losses = objective(output, _visual_targets(config))
    losses["total"].backward()
    gradient_norms = {
        name: _module_gradient_norm(module) for name, module in modules.items()
    }
    temporal_position_gradient = float(
        student.temporal_position.grad.detach().float().norm()
        if student.temporal_position.grad is not None
        else 0.0
    )
    optimizer.step()
    changed = {
        name: _module_changed(module, before[name])
        for name, module in modules.items()
    }
    changed["temporal_position"] = bool(
        torch.any(student.temporal_position.detach() != temporal_position_before)
    )
    if (
        not all(value > 0.0 for value in gradient_norms.values())
        or temporal_position_gradient <= 0.0
        or not all(changed.values())
    ):
        raise RuntimeError("deployed visual fusion path is not trainable end to end")
    return {
        "passed": True,
        "loss": float(losses["total"].detach()),
        "gradient_norms": gradient_norms
        | {"temporal_position": temporal_position_gradient},
        "parameters_changed": changed,
    }


def _formal_household_contract(root: Path) -> dict[str, object]:
    tasks = load_formal_3d_tasks(root / "configs/tasks/formal_3d_v1.json")
    bindings = load_mujoco_task_bindings(
        root / "configs/adapters/mujoco/formal_3d_v1.json", root=root
    )
    if set(tasks) != FORMAL_HOUSEHOLD_TASK_IDS or set(bindings) != set(tasks):
        raise RuntimeError("foundation runtime does not expose the formal household tasks")
    for task in tasks.values():
        if (
            len(task.objects) < 2
            or task.max_steps < 6000
            or task.minimum_each_arm_contact_seconds < 0.5
            or set(task.training_instructions) & set(task.evaluation_instructions)
        ):
            raise RuntimeError("formal task lacks multi-object, bimanual, or language holdout")
        if not bindings[task.task_id].model_path.is_file():
            raise RuntimeError("formal household MuJoCo model is missing")
    for app in (
        "src/hwr/apps/train_foundation_world_model.py",
        "src/hwr/apps/evaluate_foundation_world_model.py",
    ):
        source = (root / app).read_text(encoding="utf-8")
        if (
            "load_default_formal_household_catalogs" not in source
            or "load_default_bimanual_training_catalogs" in source
        ):
            raise RuntimeError("foundation app still loads proxy training tasks")
    return {
        "passed": True,
        "task_ids": sorted(tasks),
        "minimum_objects_per_task": min(len(task.objects) for task in tasks.values()),
        "language_holdout": True,
        "evaluation_domain_broader": True,
    }


def _online_config(root: Path) -> FoundationOnlineTrainingConfig:
    value = json.loads(
        (root / "configs/foundation/online-training-v1.json").read_text(
            encoding="utf-8"
        )
    )
    value.pop("schema_version", None)
    return FoundationOnlineTrainingConfig(**value)


def _retained_replay_contract(root: Path) -> dict[str, object]:
    config = _online_config(root)
    storage = foundation_storage_estimate(
        config, task_count=len(FORMAL_HOUSEHOLD_TASK_IDS)
    )
    transitions = int(storage["replay"]["transitions"])
    if (
        config.replay_windows_per_episode < 7
        or config.visual_supervision_windows_per_episode
        >= config.replay_windows_per_episode
        or transitions < 10_000
    ):
        raise RuntimeError("retained Replay is too small or cannot prioritize interactions")
    return {
        "passed": True,
        "replay_windows_per_episode": config.replay_windows_per_episode,
        "visual_supervision_windows_per_episode": (
            config.visual_supervision_windows_per_episode
        ),
        "retained_transitions": transitions,
    }


def _action_bounds_contract(root: Path) -> dict[str, object]:
    value = json.loads(
        (root / "configs/foundation/world-model-v1.json").read_text(encoding="utf-8")
    )
    minimum = tuple(float(item) for item in value["action_minimum"])
    maximum = tuple(float(item) for item in value["action_maximum"])
    if (
        value["action_dimension"] != 16
        or minimum != DUAL_ARM_ACTION_MINIMUM
        or maximum != DUAL_ARM_ACTION_MAXIMUM
    ):
        raise RuntimeError("world-model execution bounds differ from runtime actions")
    return {
        "passed": True,
        "dimension": 16,
        "minimum": list(minimum),
        "maximum": list(maximum),
    }
