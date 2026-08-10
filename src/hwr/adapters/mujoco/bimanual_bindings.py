"""MuJoCo name bindings kept outside project-owned bimanual task schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BimanualMujocoBinding:
    task_id: str
    model_path: Path
    payload_body: str
    payload_joint: str
    payload_geoms: tuple[str, ...]
    payload_reference_geom: str
    left_interaction_geom: str
    right_interaction_geom: str
    target_site: str
    target_support_geom: str
    allowed_robot_contact_geoms: frozenset[str]
    articulation_joint: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.task_id,
            self.payload_body,
            self.payload_joint,
            self.payload_reference_geom,
            self.left_interaction_geom,
            self.right_interaction_geom,
            self.target_site,
            self.target_support_geom,
        )
        if not all(required) or not self.payload_geoms:
            raise ValueError("bimanual MuJoCo binding names are required")


def load_bimanual_mujoco_bindings(
    path: Path, *, root: Path
) -> dict[str, BimanualMujocoBinding]:
    value = json.loads(path.read_text(encoding="utf-8"))
    bindings: dict[str, BimanualMujocoBinding] = {}
    for item in value["bindings"]:
        binding = BimanualMujocoBinding(
            task_id=item["task_id"],
            model_path=(root / item["model"]).resolve(),
            payload_body=item["payload_body"],
            payload_joint=item["payload_joint"],
            payload_geoms=tuple(item["payload_geoms"]),
            payload_reference_geom=item["payload_reference_geom"],
            left_interaction_geom=item["left_interaction_geom"],
            right_interaction_geom=item["right_interaction_geom"],
            target_site=item["target_site"],
            target_support_geom=item["target_support_geom"],
            allowed_robot_contact_geoms=frozenset(
                item["allowed_robot_contact_geoms"]
            ),
            articulation_joint=item.get("articulation_joint"),
        )
        if binding.task_id in bindings:
            raise ValueError(f"duplicate bimanual MuJoCo binding: {binding.task_id}")
        bindings[binding.task_id] = binding
    return bindings
