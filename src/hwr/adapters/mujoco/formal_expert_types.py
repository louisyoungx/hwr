"""Value objects emitted by the privileged formal-scene expert."""

from __future__ import annotations

from dataclasses import dataclass

from hwr.core.types import ActionFrame


@dataclass(frozen=True)
class ExpertStage:
    name: str
    kind: str
    object_id: str | None = None


@dataclass(frozen=True)
class FormalExpertOutput:
    action: ActionFrame
    stage: str
    stage_step: int
    privileged_label: bool = True
