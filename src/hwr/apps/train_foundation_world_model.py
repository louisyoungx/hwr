"""Run the single gated foundation/world-model/RL training lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hwr.adapters.foundation import (
    Dinov3ViTDenseVisionProvider,
    Qwen3LanguageProvider,
    Siglip2VisionLanguageProvider,
    load_foundation_model_locks,
)
from hwr.adapters.mujoco import (
    MujocoBimanualTaskBackend,
    load_default_bimanual_training_catalogs,
)
from hwr.perception.high_resolution import (
    HighResolutionVisionConfig,
    HighResolutionVisionPreprocessor,
)
from hwr.policy.bimanual_input import (
    BimanualInputConfig,
    default_four_camera_calibrations,
)
from hwr.train.development_gate import require_development_ready
from hwr.train.foundation_online import (
    FoundationOnlineTrainingRunner,
    FoundationProviderFactories,
    FoundationTaskInterface,
)
from hwr.train.foundation_online_config import FoundationOnlineTrainingConfig
from hwr.train.foundation_setup import build_foundation_learning_stack


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/foundation-world-model")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--foundation-device", default="cpu")
    parser.add_argument(
        "--development-ready",
        type=Path,
        default=root / "artifacts/development-ready.json",
    )
    parser.add_argument(
        "--model-root", type=Path, default=root / "models/foundation"
    )
    return parser


def _config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.pop("schema_version", None) != "hwr.foundation-online-training/v1":
        raise ValueError("formal online training config schema differs")
    return value


def run(arguments: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    readiness = require_development_ready(
        root, arguments.development_ready.resolve()
    )
    tasks, bindings = load_default_bimanual_training_catalogs(root)
    config = FoundationOnlineTrainingConfig(
        **_config(root / "configs/foundation/online-training-v1.json")
    )
    interfaces = {
        task_id: FoundationTaskInterface(task_id, task.max_steps)
        for task_id, task in tasks.items()
    }
    raw = BimanualInputConfig(
        config.camera_width,
        config.camera_height,
        image_width=160,
        image_height=160,
    )
    preprocessor = HighResolutionVisionPreprocessor(
        HighResolutionVisionConfig(), default_four_camera_calibrations(raw)
    )
    locks = load_foundation_model_locks(
        root / "configs/foundation/model-locks.json", arguments.model_root.resolve()
    )
    providers = FoundationProviderFactories(
        lambda: Siglip2VisionLanguageProvider(
            locks["siglip2-base-patch16-224"], device=arguments.foundation_device
        ),
        lambda: Dinov3ViTDenseVisionProvider(
            locks["dinov3-vits16-pretrain-lvd1689m"],
            device=arguments.foundation_device,
        ),
        lambda: Qwen3LanguageProvider(
            locks["qwen3-embedding-0.6b"], device=arguments.foundation_device
        ),
    )
    output_root = (
        arguments.output_root
        if arguments.output_root.is_absolute()
        else root / arguments.output_root
    )
    run_path = output_root / arguments.run_id
    if run_path.exists() and not arguments.resume:
        raise FileExistsError(f"formal training run already exists: {run_path}")
    if arguments.resume and not run_path.exists():
        raise FileNotFoundError(run_path)

    def environment_factory(task_id: str, width: int, height: int):
        return MujocoBimanualTaskBackend(
            tasks[task_id], bindings[task_id], camera_width=width, camera_height=height
        )

    runner = FoundationOnlineTrainingRunner(
        interfaces,
        environment_factory,
        preprocessor,
        providers,
        build_foundation_learning_stack(
            root / "configs/foundation", device=arguments.device
        ),
        config,
        run_path,
        source_commit=readiness["source_commit"],
    )
    if arguments.resume:
        runner.resume_latest()
    result = runner.train()
    causality = json.loads(
        result.latest_action_causality_report.read_text(encoding="utf-8")
    )
    if causality.get("assessment", {}).get("passed") is not True:
        raise RuntimeError(
            "formal training finished without passing action-shuffle causality"
        )
    return {
        "run_path": str(run_path),
        "episodes": len(result.records),
        "updates": result.update_count,
        "successes": sum(item.success for item in result.records),
        "latest_checkpoint": str(result.latest_checkpoint),
        "latest_deployment": str(result.latest_deployment),
        "action_causality_report": str(result.latest_action_causality_report),
        "action_causality_passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
