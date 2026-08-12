"""Policy APIs with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BehaviorMLP": ("hwr.policy.model", "BehaviorMLP"),
    "BimanualVLAActorPolicy": ("hwr.policy.bimanual_runtime", "BimanualVLAActorPolicy"),
    "HouseholdVisualPolicyModel": ("hwr.policy.visual_model", "HouseholdVisualPolicyModel"),
    "LearnedVisualPolicy": ("hwr.policy.visual_policy", "LearnedVisualPolicy"),
    "ModelConfig": ("hwr.policy.model", "ModelConfig"),
    "NeuralPolicy": ("hwr.policy.neural", "NeuralPolicy"),
    "Normalization": ("hwr.policy.neural", "Normalization"),
    "VisualModelConfig": ("hwr.policy.visual_model", "VisualModelConfig"),
    "VisualNormalization": ("hwr.policy.visual_policy", "VisualNormalization"),
    "VisualKnnConfig": ("hwr.policy.visual_knn", "VisualKnnConfig"),
    "VisualKnnPolicy": ("hwr.policy.visual_knn", "VisualKnnPolicy"),
    "VLAActorInput": ("hwr.policy.vla_input", "VLAActorInput"),
    "VLA_POLICY_INPUT_FIELDS": ("hwr.policy.vla_input", "VLA_POLICY_INPUT_FIELDS"),
    "build_vla_actor_input": ("hwr.policy.vla_input", "build_vla_actor_input"),
    "DeployableVLAActor": ("hwr.policy.vla_runtime", "DeployableVLAActor"),
    "VLAActorConfig": ("hwr.policy.vla_model", "VLAActorConfig"),
    "VLAActorModel": ("hwr.policy.vla_model", "VLAActorModel"),
    "VLAActorOutput": ("hwr.policy.vla_model", "VLAActorOutput"),
    "VLANormalization": ("hwr.policy.vla_runtime", "VLANormalization"),
    "PrivilegedCriticConfig": ("hwr.policy.privileged_critic", "PrivilegedCriticConfig"),
    "TwinPrivilegedCritic": ("hwr.policy.privileged_critic", "TwinPrivilegedCritic"),
    "LatentActor": ("hwr.policy.latent_actor", "LatentActor"),
    "LatentActorConfig": ("hwr.policy.latent_actor", "LatentActorConfig"),
    "LatentActorSample": ("hwr.policy.latent_actor", "LatentActorSample"),
    "LatentValueModel": ("hwr.policy.latent_value", "LatentValueModel"),
    "LatentActionScaling": ("hwr.policy.latent_actions", "LatentActionScaling"),
    "scale_latent_action": ("hwr.policy.latent_actions", "scale_latent_action"),
    "FoundationWorldModelPolicy": (
        "hwr.policy.foundation_runtime",
        "FoundationWorldModelPolicy",
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
