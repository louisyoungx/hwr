"""Engine-independent sensor preprocessing."""

from hwr.perception.contracts import (
    CameraCalibration,
    DualArmProcessedVision,
    PinholeIntrinsics,
    ProcessedVision,
    VisionPreprocessConfig,
)
from hwr.perception.preprocessing import (
    DeterministicVisionPreprocessor,
    DualArmVisionPreprocessor,
)
from hwr.perception.language import (
    FrozenNgramLanguageConfig,
    FrozenNgramLanguageEncoder,
)
from hwr.perception.foundation import (
    DenseVisualFeatures,
    FoundationModelLock,
    FrozenLanguageFeatureProvider,
    FrozenVisionFeatureProvider,
    FrozenVisionLanguageFeatureProvider,
    SemanticLanguageFeatures,
    WeightArtifact,
    language_source_sha256,
)
from hwr.perception.high_resolution import (
    HighResolutionVision,
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
)
from hwr.perception.student import (
    VISUAL_STUDENT_INPUT_FIELDS,
    VisualStudentConfig,
    VisualStudentModel,
    VisualStudentOutput,
)
from hwr.perception.student_objectives import (
    VisualFoundationObjectives,
    VisualObjectiveConfig,
    VisualTeacherTargets,
)
from hwr.perception.student_input import (
    VisualStudentInput,
    VisualStudentInputAssembler,
    visual_student_tensors,
)
from hwr.perception.geometric_correspondence import (
    batch_correspondence_indices,
    build_cross_camera_patch_correspondences,
)

__all__ = [
    "CameraCalibration",
    "DeterministicVisionPreprocessor",
    "DualArmProcessedVision",
    "DualArmVisionPreprocessor",
    "PinholeIntrinsics",
    "ProcessedVision",
    "VisionPreprocessConfig",
    "FrozenNgramLanguageConfig",
    "FrozenNgramLanguageEncoder",
    "DenseVisualFeatures",
    "FoundationModelLock",
    "FrozenLanguageFeatureProvider",
    "FrozenVisionFeatureProvider",
    "FrozenVisionLanguageFeatureProvider",
    "HighResolutionVision",
    "HighResolutionVisionConfig",
    "HighResolutionVisionPreprocessor",
    "SemanticLanguageFeatures",
    "WeightArtifact",
    "language_source_sha256",
    "VISUAL_STUDENT_INPUT_FIELDS",
    "VisualFoundationObjectives",
    "VisualObjectiveConfig",
    "VisualStudentConfig",
    "VisualStudentModel",
    "VisualStudentOutput",
    "VisualTeacherTargets",
    "VisualStudentInput",
    "VisualStudentInputAssembler",
    "batch_correspondence_indices",
    "build_cross_camera_patch_correspondences",
    "visual_student_tensors",
]
