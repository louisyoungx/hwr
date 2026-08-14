"""Compose MuJoCo task catalogs and the training backend factory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_bindings import (
    BimanualMujocoBinding,
    load_bimanual_mujoco_bindings,
)
from hwr.adapters.mujoco.bindings import MujocoTaskBinding, load_mujoco_task_bindings
from hwr.adapters.mujoco.formal_household_backend import (
    MujocoFormalHouseholdDualArmBackend,
)
from hwr.scenarios.formal3d import Formal3DTaskSpec, load_formal_3d_tasks
from hwr.tasks import BimanualTaskSpec, load_bimanual_task_specs


@dataclass(frozen=True)
class MujocoBimanualBackendFactory:
    bindings: Mapping[str, BimanualMujocoBinding]

    def __call__(
        self,
        task: BimanualTaskSpec,
        camera_width: int,
        camera_height: int,
    ) -> MujocoBimanualTaskBackend:
        try:
            binding = self.bindings[task.task_id]
        except KeyError as exc:
            raise ValueError(f"missing MuJoCo binding for {task.task_id}") from exc
        return MujocoBimanualTaskBackend(
            task,
            binding,
            camera_width=camera_width,
            camera_height=camera_height,
        )


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


@dataclass(frozen=True)
class MujocoFormalHouseholdBackendFactory:
    bindings: Mapping[str, MujocoTaskBinding]
    evaluation_profile: bool = False

    def __call__(
        self,
        task: Formal3DTaskSpec,
        camera_width: int,
        camera_height: int,
    ) -> MujocoFormalHouseholdDualArmBackend:
        try:
            binding = self.bindings[task.task_id]
        except KeyError as exc:
            raise ValueError(f"missing formal MuJoCo binding for {task.task_id}") from exc
        return MujocoFormalHouseholdDualArmBackend(
            task,
            binding,
            camera_width=camera_width,
            camera_height=camera_height,
            evaluation_profile=self.evaluation_profile,
        )


def load_default_formal_household_catalogs(
    root: Path,
) -> tuple[dict[str, Formal3DTaskSpec], dict[str, MujocoTaskBinding]]:
    tasks = load_formal_3d_tasks(root / "configs/tasks/formal_3d_v1.json")
    bindings = load_mujoco_task_bindings(
        root / "configs/adapters/mujoco/formal_3d_v1.json",
        root=root,
    )
    if set(tasks) != set(bindings):
        raise ValueError("formal household task and MuJoCo binding catalogs differ")
    return tasks, bindings
