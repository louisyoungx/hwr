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
    SemanticLanguageFeatures,
    WeightArtifact,
)
from hwr.perception.high_resolution import (
    HighResolutionVision,
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
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
    "HighResolutionVision",
    "HighResolutionVisionConfig",
    "HighResolutionVisionPreprocessor",
    "SemanticLanguageFeatures",
    "WeightArtifact",
]
