from __future__ import annotations

from pathlib import Path

import numpy as np

from hwr.adapters.mujoco import (
    BIMANUAL_EVIDENCE_VIEWS,
    MujocoBimanualEvidenceSource,
    MujocoBimanualTaskBackend,
    load_default_bimanual_training_catalogs,
)
from hwr.render import BimanualVideoRecorder


ROOT = Path(__file__).resolve().parents[1]


def test_high_resolution_evidence_capture_does_not_change_physics(tmp_path) -> None:
    tasks, bindings = load_default_bimanual_training_catalogs(ROOT)
    task_id = "carry_dining_tray/v1"
    environment = MujocoBimanualTaskBackend(
        tasks[task_id], bindings[task_id], camera_width=16, camera_height=12
    )
    source = MujocoBimanualEvidenceSource(environment, width=32, height=24)
    try:
        observation = environment.reset(seed=77, task_id=task_id)
        before_qpos = environment.data.qpos.copy()
        before_qvel = environment.data.qvel.copy()

        frames = source.capture(observation)

        assert set(frames) == set(BIMANUAL_EVIDENCE_VIEWS)
        assert all(len(frame) == 32 * 24 * 3 for frame in frames.values())
        assert np.array_equal(environment.data.qpos, before_qpos)
        assert np.array_equal(environment.data.qvel, before_qvel)

        recorder = BimanualVideoRecorder(
            tmp_path, "episode", width=32, height=24
        )
        recorder.append(frames)
        recorder.append(frames)
        result = recorder.close()

        assert result.frame_count == 2
        assert result.duration_seconds == 0.1
        assert all(path.stat().st_size > 0 for path in result.paths.values())
    finally:
        source.close()
        environment.close()
