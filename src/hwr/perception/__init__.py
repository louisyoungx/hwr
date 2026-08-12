"""Engine-independent perception APIs with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CameraCalibration": ("hwr.perception.contracts", "CameraCalibration"),
    "DualArmProcessedVision": ("hwr.perception.contracts", "DualArmProcessedVision"),
    "PinholeIntrinsics": ("hwr.perception.contracts", "PinholeIntrinsics"),
    "ProcessedVision": ("hwr.perception.contracts", "ProcessedVision"),
    "VisionPreprocessConfig": ("hwr.perception.contracts", "VisionPreprocessConfig"),
    "DeterministicVisionPreprocessor": (
        "hwr.perception.preprocessing",
        "DeterministicVisionPreprocessor",
    ),
    "DualArmVisionPreprocessor": (
        "hwr.perception.preprocessing",
        "DualArmVisionPreprocessor",
    ),
    "FrozenNgramLanguageConfig": (
        "hwr.perception.language",
        "FrozenNgramLanguageConfig",
    ),
    "FrozenNgramLanguageEncoder": (
        "hwr.perception.language",
        "FrozenNgramLanguageEncoder",
    ),
    "DenseVisualFeatures": ("hwr.perception.foundation", "DenseVisualFeatures"),
    "FoundationModelLock": ("hwr.perception.foundation", "FoundationModelLock"),
    "FrozenLanguageFeatureProvider": (
        "hwr.perception.foundation",
        "FrozenLanguageFeatureProvider",
    ),
    "FrozenVisionFeatureProvider": (
        "hwr.perception.foundation",
        "FrozenVisionFeatureProvider",
    ),
    "FrozenVisionLanguageFeatureProvider": (
        "hwr.perception.foundation",
        "FrozenVisionLanguageFeatureProvider",
    ),
    "SemanticLanguageFeatures": (
        "hwr.perception.foundation",
        "SemanticLanguageFeatures",
    ),
    "WeightArtifact": ("hwr.perception.foundation", "WeightArtifact"),
    "language_source_sha256": (
        "hwr.perception.foundation",
        "language_source_sha256",
    ),
    "HighResolutionVision": (
        "hwr.perception.high_resolution",
        "HighResolutionVision",
    ),
    "HighResolutionVisionConfig": (
        "hwr.perception.high_resolution",
        "HighResolutionVisionConfig",
    ),
    "HighResolutionVisionPreprocessor": (
        "hwr.perception.high_resolution",
        "HighResolutionVisionPreprocessor",
    ),
    "VISUAL_STUDENT_INPUT_FIELDS": (
        "hwr.perception.student",
        "VISUAL_STUDENT_INPUT_FIELDS",
    ),
    "VisualStudentConfig": ("hwr.perception.student", "VisualStudentConfig"),
    "VisualStudentModel": ("hwr.perception.student", "VisualStudentModel"),
    "VisualStudentOutput": ("hwr.perception.student", "VisualStudentOutput"),
    "VisualFoundationObjectives": (
        "hwr.perception.student_objectives",
        "VisualFoundationObjectives",
    ),
    "VisualObjectiveConfig": (
        "hwr.perception.student_objectives",
        "VisualObjectiveConfig",
    ),
    "VisualTeacherTargets": (
        "hwr.perception.student_objectives",
        "VisualTeacherTargets",
    ),
    "VisualStudentInput": ("hwr.perception.student_input", "VisualStudentInput"),
    "VisualStudentInputAssembler": (
        "hwr.perception.student_input",
        "VisualStudentInputAssembler",
    ),
    "visual_student_tensors": (
        "hwr.perception.student_input",
        "visual_student_tensors",
    ),
    "batch_correspondence_indices": (
        "hwr.perception.geometric_correspondence",
        "batch_correspondence_indices",
    ),
    "build_cross_camera_patch_correspondences": (
        "hwr.perception.geometric_correspondence",
        "build_cross_camera_patch_correspondences",
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
