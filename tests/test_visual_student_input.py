from __future__ import annotations

import numpy as np
import pytest

from hwr.perception.high_resolution import HighResolutionVision
from hwr.perception.student_input import (
    VisualStudentInputAssembler,
    visual_student_tensors,
)


def _frame(source: str = "a", fingerprint: str = "b") -> HighResolutionVision:
    size = 160
    return HighResolutionVision(
        teacher_rgb=np.zeros((3, 224, 224, 3), np.float32),
        student_rgb=np.zeros((3, size, size, 3), np.float32),
        student_head_depth_m=np.ones((size, size), np.float32),
        student_head_depth_valid=np.ones((size, size), np.bool_),
        camera_validity=np.ones(4, np.bool_),
        frame_timestamps_ns=np.arange(4, dtype=np.int64),
        student_intrinsics=np.ones((4, 4), np.float32),
        robot_from_camera=np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0),
        preprocess_fingerprint=fingerprint * 64,
        source_sha256=source * 64,
    )


def test_visual_student_input_repeats_only_missing_history() -> None:
    assembler = VisualStudentInputAssembler(visual_history=4, image_size=160)

    first = assembler.build(_frame("a"))
    second = assembler.build(_frame("c"))

    assert first.rgb.shape == (4, 3, 3, 160, 160)
    assert first.repeated_frame.tolist() == [True, True, True, False]
    assert second.repeated_frame.tolist() == [True, True, False, False]
    assert second.source_sha256 == ("a" * 64, "a" * 64, "a" * 64, "c" * 64)
    tensors = visual_student_tensors(second)
    assert tensors["rgb"].shape == (1, 4, 3, 3, 160, 160)


def test_visual_student_input_rejects_preprocess_change_inside_history() -> None:
    assembler = VisualStudentInputAssembler(visual_history=4, image_size=160)
    assembler.build(_frame())

    with pytest.raises(ValueError, match="changed"):
        assembler.build(_frame(fingerprint="d"))
