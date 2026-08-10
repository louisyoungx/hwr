"""Deterministic compaction of expert micro-stages into learnable macro phases."""

from __future__ import annotations

from pathlib import Path

from hwr.data.visual import FormalPolicyInput, VisualBehaviorSample, VisualDatasetBuilder
from hwr.data.visual_loading import load_visual_dataset


DRAWER_PHASES = {
    "navigate_to_drawer": "approach_drawer",
    "unfold_arm_for_drawer": "approach_drawer",
    "approach_drawer_handle": "approach_drawer",
    "descend_to_drawer_handle": "approach_drawer",
    "close_on_drawer_handle": "open_drawer",
    "contact_pull_drawer": "open_drawer",
    "release_drawer_handle": "clear_drawer",
    "retract_from_drawer": "clear_drawer",
    "back_away_from_open_drawer": "clear_drawer",
}
OBJECT_PHASE_PREFIXES = (
    (("nav_object_", "unstow_arm_", "arm_object_above_"), "approach_object_"),
    (("arm_object_descend_", "grip_object_"), "grasp_object_"),
    (("arm_object_lift_", "transport_arm_"), "lift_object_"),
    (
        (
            "nav_target_",
            "arm_target_raise_",
            "arm_target_above_",
            "arm_target_lower_",
        ),
        "approach_target_",
    ),
    (("release_object_", "arm_target_retract_"), "release_object_"),
)


def compact_household_phase(stage: str) -> str:
    if stage in DRAWER_PHASES:
        return DRAWER_PHASES[stage]
    for prefixes, output_prefix in OBJECT_PHASE_PREFIXES:
        for prefix in prefixes:
            if stage.startswith(prefix):
                return output_prefix + stage.removeprefix(prefix)
    raise ValueError(f"formal expert stage has no macro phase mapping: {stage}")


def compact_visual_dataset(source: Path, output_root: Path, dataset_id: str) -> Path:
    dataset = load_visual_dataset(source)
    metadata = dict(dataset.manifest["metadata"])
    metadata["phase_compaction"] = "hwr.household-macro-phases/v1"
    metadata["source_dataset_id"] = dataset.manifest["dataset_id"]
    builder = VisualDatasetBuilder(
        output_root,
        dataset_id,
        task_id=str(dataset.manifest["task_id"]),
        instruction=str(dataset.manifest["instruction"]),
        image_size=tuple(dataset.manifest["image_size"]),
        action_history=int(dataset.manifest["action_history"]),
        metadata=metadata,
    )
    ordered_phases = tuple(
        dict.fromkeys(compact_household_phase(name) for name in dataset.phase_names)
    )
    builder.declare_phase_order(ordered_phases)
    offset = 0
    for shard in dataset.manifest["shards"]:
        count = int(shard["sample_count"])
        indices = range(offset, offset + count)
        samples = [
            VisualBehaviorSample(
                int(dataset.step_indices[index]),
                FormalPolicyInput(
                    **{name: dataset.inputs[name][index] for name in dataset.inputs}
                ),
                dataset.actions[index],
                phase=compact_household_phase(str(dataset.phases[index])),
            )
            for index in indices
        ]
        builder.write_episode(str(shard["episode_id"]), int(shard["seed"]), samples)
        offset += count
    return builder.seal()
