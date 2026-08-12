"""Episode and dataset utilities with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BehaviorDataset": ("hwr.data.dataset", "BehaviorDataset"),
    "EpisodeReader": ("hwr.data.episode", "EpisodeReader"),
    "EpisodeRecorder": ("hwr.data.episode", "EpisodeRecorder"),
    "FormalPolicyInput": ("hwr.data.visual", "FormalPolicyInput"),
    "LoadedVisualDataset": ("hwr.data.visual_loading", "LoadedVisualDataset"),
    "POLICY_INPUT_FIELDS": ("hwr.data.visual", "POLICY_INPUT_FIELDS"),
    "VisualBehaviorSample": ("hwr.data.visual", "VisualBehaviorSample"),
    "VisualDatasetBuilder": ("hwr.data.visual", "VisualDatasetBuilder"),
    "aggregate_policy_dataset": ("hwr.data.aggregation", "aggregate_policy_dataset"),
    "aggregate_visual_policy_dataset": (
        "hwr.data.visual_aggregation",
        "aggregate_visual_policy_dataset",
    ),
    "compact_household_phase": ("hwr.data.visual_phases", "compact_household_phase"),
    "compact_visual_dataset": ("hwr.data.visual_phases", "compact_visual_dataset"),
    "extract_formal_policy_input": ("hwr.data.visual", "extract_formal_policy_input"),
    "formal_action_vector": ("hwr.data.visual", "formal_action_vector"),
    "generate_expert_dataset": ("hwr.data.generation", "generate_expert_dataset"),
    "generate_visual_expert_dataset": (
        "hwr.data.visual_generation",
        "generate_visual_expert_dataset",
    ),
    "load_visual_dataset": ("hwr.data.visual_loading", "load_visual_dataset"),
    "verify_visual_dataset": ("hwr.data.visual", "verify_visual_dataset"),
    "LoadedVLADataset": ("hwr.data.vla_loading", "LoadedVLADataset"),
    "VLABehaviorDatasetBuilder": ("hwr.data.vla_dataset", "VLABehaviorDatasetBuilder"),
    "VLABehaviorSample": ("hwr.data.vla_dataset", "VLABehaviorSample"),
    "load_vla_dataset": ("hwr.data.vla_loading", "load_vla_dataset"),
    "verify_vla_dataset": ("hwr.data.vla_dataset", "verify_vla_dataset"),
    "FoundationCacheKey": ("hwr.data.foundation_cache", "FoundationCacheKey"),
    "FoundationFeatureCache": (
        "hwr.data.foundation_cache",
        "FoundationFeatureCache",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
