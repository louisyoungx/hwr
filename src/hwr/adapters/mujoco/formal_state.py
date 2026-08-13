"""State access helpers for the historical privileged formal-scene expert."""

from __future__ import annotations

import mujoco
import numpy as np

from hwr.core.types import ActionFrame, ObservationFrame


class FormalExpertStateMixin:
    """Keep engine-state plumbing separate from the expert state machine."""

    def _object_position(self, object_id: str | None) -> tuple[float, float, float]:
        if object_id is None:
            raise ValueError("object ID is required")
        body_id = self.backend.household_ids.object_bodies[object_id]
        return tuple(float(value) for value in self.backend.data.xpos[body_id])

    def _target_position(self, object_id: str | None) -> tuple[float, float, float]:
        if object_id is None:
            raise ValueError("object ID is required")
        site_id = self.backend.household_ids.target_sites[object_id]
        return tuple(float(value) for value in self.backend.data.site_xpos[site_id])

    def _drawer_handle_position(self) -> tuple[float, float, float]:
        binding = self.backend.binding.articulation
        if binding is None:
            raise RuntimeError("task has no drawer binding")
        geom_id = mujoco.mj_name2id(
            self.backend.model, mujoco.mjtObj.mjOBJ_GEOM, binding.handle_geom
        )
        return tuple(float(value) for value in self.backend.data.geom_xpos[geom_id])

    def _drawer_grasp_target(self) -> tuple[float, float, float]:
        handle = self._drawer_handle_position()
        return (handle[0], handle[1] - 0.024, handle[2] + 0.03)

    def _object_spec(self, object_id: str | None):
        return next(obj for obj in self.task.objects if obj.object_id == object_id)

    def _grip_fraction(self, object_id: str | None) -> float:
        return self._object_spec(object_id).grip_fraction

    def _gripper(self) -> float:
        if self.drawer_holding:
            return 1.0
        return self._grip_fraction(self.holding_object) if self.holding_object else 0.0

    def _hold_action(self, observation: ObservationFrame, gripper: float) -> ActionFrame:
        return self._action(
            observation,
            linear=0.0,
            angular=0.0,
            gripper=gripper,
            arm_command=self._hold_arm_command(observation),
        )

    def _hold_arm_command(self, observation: ObservationFrame) -> tuple[float, ...]:
        target_error = (
            np.asarray(observation.joint_position)
            - self.backend._arm_targets  # noqa: SLF001
        )
        return tuple(
            float(value) for value in np.clip(20.0 * target_error, -1.0, 1.0)
        )

    def _stop(
        self, observation: ObservationFrame, gripper: float | None = None
    ) -> ActionFrame:
        return self._hold_action(
            observation, self._gripper() if gripper is None else gripper
        )
