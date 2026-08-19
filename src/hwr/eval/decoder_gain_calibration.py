"""Calibration and path-JVP helpers for R0001-P24."""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from hwr.world_model.model import ActionConditionedWorldModel


HEAD_NAMES = ("visual", "proprioception")
ACTIVE_SCALE_MINIMUM = 1.0e-4
ACTIVE_FRACTION_MINIMUM = 0.25
EFFECT_DENOMINATOR_MINIMUM = 1.0e-6
PATH_SEGMENTS = 16


def build_decoder_calibration(
    model: ActionConditionedWorldModel,
    true_features: Sequence[torch.Tensor],
) -> dict[str, object]:
    if len(true_features) != 24:
        raise ValueError("decoder calibration requires 24 true features")
    combined = torch.cat(tuple(true_features), dim=1)
    if combined.shape[1] != 384:
        raise ValueError("decoder calibration requires 384 transitions")
    calibration = {
        "schema_version": "hwr.decoder-gain-calibration/v1",
        "transition_count": int(combined.shape[1]),
        "heads": {},
    }
    with torch.inference_mode():
        for head_name in HEAD_NAMES:
            head = _head(model, head_name)
            stages, layer_norm = _decoder_stages(head, combined)
            calibration["heads"][head_name] = {
                "stages": {
                    name: _calibration_stage(value)
                    for name, value in stages.items()
                },
                "layer_norm": layer_norm,
            }
    return calibration


def serialize_decoder_calibration(
    calibration: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": calibration["schema_version"],
        "transition_count": calibration["transition_count"],
        "heads": {
            head_name: {
                "stages": {
                    stage_name: {
                        key: (
                            value.tolist()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for key, value in stage.items()
                    }
                    for stage_name, stage in head["stages"].items()
                },
                "layer_norm": head["layer_norm"],
            }
            for head_name, head in calibration["heads"].items()
        },
    }


def deserialize_decoder_calibration(
    value: Mapping[str, object],
    *,
    device: str,
) -> dict[str, object]:
    if (
        value.get("schema_version") != "hwr.decoder-gain-calibration/v1"
        or int(value.get("transition_count", 0)) != 384
        or set(value.get("heads", {})) != set(HEAD_NAMES)
    ):
        raise ValueError("decoder calibration artifact identity differs")
    return {
        "schema_version": value["schema_version"],
        "transition_count": int(value["transition_count"]),
        "heads": {
            head_name: {
                "stages": {
                    stage_name: {
                        key: (
                            torch.tensor(
                                item,
                                device=device,
                                dtype=(
                                    torch.bool
                                    if key == "active_mask"
                                    else torch.float64
                                ),
                            )
                            if key in {"mean", "scale", "active_mask"}
                            and item is not None
                            else item
                        )
                        for key, item in stage.items()
                    }
                    for stage_name, stage in head["stages"].items()
                },
                "layer_norm": head["layer_norm"],
            }
            for head_name, head in value["heads"].items()
        },
    }


def _decoder_stages(
    head: torch.nn.Sequential,
    features: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    _require_head_structure(head, features.shape[-1])
    preactivation = head[0](features)
    layer_norm = head[1]
    mean = preactivation.mean(dim=-1, keepdim=True)
    variance = preactivation.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (preactivation - mean) / torch.sqrt(
        variance + layer_norm.eps
    )
    affine = normalized
    if layer_norm.elementwise_affine:
        affine = affine * layer_norm.weight + layer_norm.bias
    hidden = head[2](affine)
    output = head[3](hidden)
    return (
        {
            "feature": features,
            "linear_preactivation": preactivation,
            "layer_norm_normalized": normalized,
            "layer_norm_affine": affine,
            "hidden": hidden,
            "output": output,
        },
        {
            "eps": float(layer_norm.eps),
            "true_mean": _distribution(mean),
            "true_variance": _distribution(variance),
            "gamma": (
                _distribution(layer_norm.weight)
                if layer_norm.elementwise_affine
                else None
            ),
            "beta": (
                _distribution(layer_norm.bias)
                if layer_norm.elementwise_affine
                else None
            ),
        },
    )


def _calibration_stage(value: torch.Tensor) -> dict[str, object]:
    cpu = value.detach().cpu().double()
    finite = bool(torch.isfinite(cpu).all())
    if not finite:
        return {
            "finite": False,
            "mean": None,
            "scale": None,
            "active_mask": None,
            "active_dimension_count": 0,
            "active_fraction": 0.0,
        }
    mean = cpu.mean(dim=(0, 1))
    scale = (cpu - mean).square().mean(dim=(0, 1)).sqrt()
    active = scale >= ACTIVE_SCALE_MINIMUM
    active_count = int(active.sum())
    return {
        "finite": True,
        "mean": mean,
        "scale": scale,
        "active_mask": active,
        "active_dimension_count": active_count,
        "active_fraction": active_count / scale.numel(),
        "raw_rms": float(cpu.square().mean().sqrt()),
        "active_scale": _distribution(scale[active]) if active_count else None,
    }


def _calibrated_effect(
    true: torch.Tensor,
    shifted: torch.Tensor,
    calibration: Mapping[str, object],
) -> dict[str, object]:
    true_cpu = true.detach().cpu().double()
    shifted_cpu = shifted.detach().cpu().double()
    mask = calibration["active_mask"]
    scale = calibration["scale"]
    active_count = int(calibration["active_dimension_count"])
    valid = bool(
        calibration["finite"] is True
        and float(calibration["active_fraction"])
        >= ACTIVE_FRACTION_MINIMUM
        and torch.isfinite(true_cpu).all()
        and torch.isfinite(shifted_cpu).all()
        and mask is not None
        and scale is not None
        and active_count
    )
    if not valid:
        return {
            "valid": False,
            "raw_rms": None,
            "standardized_effect": None,
        }
    difference = shifted_cpu - true_cpu
    return {
        "valid": True,
        "raw_rms": float(difference.square().mean().sqrt()),
        "standardized_effect": float(
            (difference[0, :, mask] / scale[mask])
            .square()
            .mean()
            .sqrt()
        ),
    }


def _path_report(
    function: Callable[[torch.Tensor], torch.Tensor],
    true_input: torch.Tensor,
    shifted_input: torch.Tensor,
    true_output: torch.Tensor,
    shifted_output: torch.Tensor,
    input_calibration: Mapping[str, object],
    output_calibration: Mapping[str, object],
) -> dict[str, object]:
    delta = shifted_input - true_input
    tangents = []
    for index in range(PATH_SEGMENTS):
        alpha = (index + 0.5) / PATH_SEGMENTS
        midpoint = true_input + alpha * delta
        _, tangent = torch.func.jvp(function, (midpoint,), (delta,))
        tangents.append(tangent)
    reconstructed = torch.stack(tangents).mean(dim=0)
    actual_delta = shifted_output - true_output
    path_effect = _calibrated_effect(
        torch.zeros_like(reconstructed), reconstructed, output_calibration
    )
    input_effect = _calibrated_effect(
        true_input, shifted_input, input_calibration
    )
    path_retention = _ratio(
        path_effect["standardized_effect"],
        input_effect["standardized_effect"],
    )
    reconstruction = _reconstruction_report(reconstructed, actual_delta)
    valid = (
        path_effect["valid"] is True
        and input_effect["valid"] is True
        and path_retention is not None
        and reconstruction["valid"] is True
    )
    return {
        "valid": valid,
        "path_effect": path_effect,
        "input_effect": input_effect,
        "path_retention": path_retention,
        **reconstruction,
    }


def _reconstruction_report(
    reconstructed: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, object]:
    left = reconstructed.detach().cpu().double().reshape(-1)
    right = actual.detach().cpu().double().reshape(-1)
    actual_norm = float(right.norm())
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite or actual_norm < EFFECT_DENOMINATOR_MINIMUM:
        return {
            "valid": False,
            "reconstruction_cosine": None,
            "relative_error": None,
            "actual_delta_norm": actual_norm,
        }
    denominator = float(left.norm()) * actual_norm
    cosine = float(left @ right) / max(denominator, 1.0e-20)
    relative_error = float((left - right).norm()) / actual_norm
    return {
        "valid": math.isfinite(cosine) and math.isfinite(relative_error),
        "reconstruction_cosine": max(-1.0, min(1.0, cosine)),
        "relative_error": relative_error,
        "actual_delta_norm": actual_norm,
    }


def _head(
    model: ActionConditionedWorldModel,
    name: str,
) -> torch.nn.Sequential:
    head = (
        model.visual_head
        if name == "visual"
        else model.proprioception_head
    )
    expected_output = (
        model.config.visual_dimension
        if name == "visual"
        else model.config.proprioception_dimension
    )
    _require_head_structure(
        head,
        model.config.feature_dimension,
        model.config.hidden_dimension,
        expected_output,
    )
    return head


def _require_head_structure(
    head: torch.nn.Sequential,
    input_dimension: int,
    hidden_dimension: int | None = None,
    output_dimension: int | None = None,
) -> None:
    if (
        len(head) != 4
        or not isinstance(head[0], torch.nn.Linear)
        or head[0].in_features != input_dimension
        or (
            hidden_dimension is not None
            and head[0].out_features != hidden_dimension
        )
        or not isinstance(head[1], torch.nn.LayerNorm)
        or tuple(head[1].normalized_shape) != (head[0].out_features,)
        or not isinstance(head[2], torch.nn.SiLU)
        or not isinstance(head[3], torch.nn.Linear)
        or head[3].in_features != head[0].out_features
        or (
            output_dimension is not None
            and head[3].out_features != output_dimension
        )
    ):
        raise ValueError("P24 frozen decoder head structure differs")


def _edge_functions(
    head: torch.nn.Sequential,
) -> tuple[tuple[str, str, str, Callable[[torch.Tensor], torch.Tensor]], ...]:
    layer_norm = head[1]

    def normalized(value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=-1, keepdim=True)
        variance = value.var(dim=-1, unbiased=False, keepdim=True)
        return (value - mean) / torch.sqrt(variance + layer_norm.eps)

    def affine(value: torch.Tensor) -> torch.Tensor:
        if not layer_norm.elementwise_affine:
            return value
        return value * layer_norm.weight + layer_norm.bias

    return (
        ("feature_to_linear", "feature", "linear_preactivation", head[0]),
        (
            "linear_to_norm",
            "linear_preactivation",
            "layer_norm_normalized",
            normalized,
        ),
        (
            "norm_to_affine",
            "layer_norm_normalized",
            "layer_norm_affine",
            affine,
        ),
        ("affine_to_hidden", "layer_norm_affine", "hidden", head[2]),
        ("hidden_to_output", "hidden", "output", head[3]),
    )


def _distribution(value: torch.Tensor) -> dict[str, float]:
    array = value.detach().cpu().double().numpy().reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("decoder gain distribution is invalid")
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _ratio(numerator: object, denominator: object) -> float | None:
    if (
        numerator is None
        or denominator is None
        or float(denominator) < EFFECT_DENOMINATOR_MINIMUM
    ):
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None
