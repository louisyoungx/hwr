"""Read-only physical contact evidence from MuJoCo's solved contacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class GraspContactSample:
    left_contact: bool
    right_contact: bool
    left_normal_force: float
    right_normal_force: float

    @property
    def bilateral(self) -> bool:
        return self.left_contact and self.right_contact

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"bilateral": self.bilateral}


class GraspContactMonitor:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        object_geom: str,
        left_pad: str | tuple[str, ...] = "right_gripper_left_pad",
        right_pad: str | tuple[str, ...] = "right_gripper_right_pad",
    ) -> None:
        self.model = model
        self.object_geom_id = _geom_id(model, object_geom)
        self.left_pad_ids = _geom_ids(model, left_pad)
        self.right_pad_ids = _geom_ids(model, right_pad)

    def sample(self, data: mujoco.MjData) -> GraspContactSample:
        left_force = 0.0
        right_force = 0.0
        force = np.zeros(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.object_geom_id not in pair:
                continue
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            normal_force = abs(float(force[0]))
            if pair.intersection(self.left_pad_ids):
                left_force += normal_force
            if pair.intersection(self.right_pad_ids):
                right_force += normal_force
        return GraspContactSample(
            left_contact=left_force > 0,
            right_contact=right_force > 0,
            left_normal_force=left_force,
            right_normal_force=right_force,
        )


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"model is missing geom: {name}")
    return geom_id


def _geom_ids(
    model: mujoco.MjModel, names: str | tuple[str, ...]
) -> frozenset[int]:
    values = (names,) if isinstance(names, str) else names
    return frozenset(_geom_id(model, name) for name in values)
