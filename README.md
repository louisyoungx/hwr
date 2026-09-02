# Housework Robot Training Platform

An in-house, vendor-neutral training platform for embodied housework intelligence. The existing continuous 2D environment is retained only as a software-pipeline smoke test; formal work has moved to the V1 platform with rigid-body contact, visual observations, and real 3D household assets. 2D results do not count toward 3D realism acceptance.

## Implemented

- Versioned Observation, Action, Event, and Episode;
- A `RuntimeBackend` shared by simulation and future real hardware;
- Independent safety action filtering and validity-period checks;
- Realistic differential-drive base, 2D arms, grippers, objects, and obstacles;
- Deterministic replay with fixed seeds;
- Parquet behavior datasets split by Episode;
- A 16-dimensional base–dual-arm–dual-gripper action contract;
- A deployable VLA Actor, privileged dual Critics for training, and core experience-replay components;
- A MuJoCo model with a 1.60 m four-wheel central body, two 6-DOF arms, dual pinch grippers, and head/wrist cameras;
- Three formal multi-object tasks: living-room dual-object storage, dining-table cup-and-dish placement, and kitchen dual-bottle drawer placement;
- Expert-free online sampling, hierarchical replay retaining only autonomous transitions, automatic curricula, checkpoint resumption, and Actor export;
- High-resolution continuous perception caches locked to SigLIP2, DINOv3 ViT-S/16, and Qwen3-Embedding;
- A 24.4M-parameter visual student, action-conditioned categorical RSSM, and latent-space reinforcement learning;
- Dynamic wrist-camera calibration, autonomous sequence replay, unified online closed loops for all three tasks, and stripped deployment export;
- Per-task isolation of training/evaluation instruction rewrites, plus out-of-distribution evaluation for cameras, depth, actuators, and latency;
- Significant-interaction-prioritized Replay over 13,440 transitions, with admission evidence computed only from actual retained transitions;
- Executable development hard gates for deployment visual-fusion gradients, formal task entry, Replay size, and action bounds;
- Unseen-seed evaluation of reloaded Actors, left/right single-arm lockout ablations, and four-view same-process video recording;
- A local model registry with checksums;
- Historical 2D closed-loop benchmarks for three independent housework scenes;
- Twelve CC0 textured meshes for three formal 3D household scenes, with license/hash locks and reproducible conversion tools;
- Living-room, dining-room, and kitchen MJCF assemblies with separate visual/collision geometry and a physically modeled drawer without actuators.

Historical behavior cloning, data aggregation, and rule-based expert implementations are no longer part of the formal training route. The small visual front end, character-hash language input, and direct model-free Actor-Critic from P076–P080 have been confirmed as failed baselines. The current mainline is rebuilt as “foundation-model continuous representations, an action-conditioned world model, and latent-space reinforcement learning”; formal training does not start until all development work and the overall gate are complete.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev,video]"

python3 scripts/check_python_size.py
python3 scripts/check_architecture.py
python3 -m pytest
python3 scripts/verify_benchmarks.py
```

The foundation-model/world-model mainline must first pass one mandatory overall development gate before formal training starts:

```bash
.venv/bin/python -m pip install -e ".[dev,video,sim3d,foundation]"
.venv/bin/python scripts/verify_development_ready.py \
  --output artifacts/development-ready.json
hwr-train-foundation-world-model --run-id foundation-wm-001 --device cpu
```

The training command directly refuses to run until the gate produces a report consistent with the current commit, configuration, and protected-source hashes. You can first run the training-semantics check separately without accessing foundation-model weights:

```bash
.venv/bin/python scripts/verify_training_semantics.py
```

3D development environment and robot-model verification:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,video,sim3d,assets3d]"
.venv/bin/python scripts/fetch_3d_assets.py
.venv/bin/python scripts/verify_3d_assets.py
.venv/bin/python scripts/verify_robot_model.py
.venv/bin/python scripts/verify_physics_integrity.py
.venv/bin/python -m pytest tests/test_mujoco_backend.py
.venv/bin/python -m hwr.apps.render_3d_smoke \
  --output-path artifacts/3d-smoke.png
.venv/bin/python -m hwr.apps.verify_contact_grasp \
  --output-path artifacts/contact-grasp-smoke.mp4
.venv/bin/python -m hwr.apps.render_formal_scenes \
  --output artifacts/formal-scenes.png
```

Reproduce the closed-loop actions of the three historical 2D benchmarks and output a side-by-side video (for software-pipeline regression only; requires system-installed `ffmpeg`):

```bash
hwr-render-benchmarks --output-path artifacts/benchmark-rollouts.mp4
```

## Documentation

- [Platform Architecture and Module Boundaries](docs/architecture.md)
- [Foundation-Model Perception, World Model, and Latent Reinforcement-Learning Paradigm](docs/foundation-world-model-training-paradigm.md)
- [Assessment Contract for Moving from Toy Closed Loops to a Serious Research Platform](docs/serious-platform-vnext.md)
- [Historical End-to-End Training Paradigm](docs/end-to-end-training-paradigm.md)
- [Local Training Progress and Phase Diagnostics](docs/training-progress.md)
- [Training and Realistic-Environment Plan](docs/training-and-simulation-plan.md)
- [Low-Cost Platform Proposal](docs/low-cost-platform-proposal.md)
- [Training Benchmarks and Reproduction Commands](benchmarks/README.md)
- [3D-Realism V1 Implementation and Acceptance Contract](docs/three-dimensional-v1-acceptance.md)
- [3D Engine Architecture Decision](docs/adr/0001-mujoco-3d-backend.md)
- [Formal 3D Scenes and Reproduction Commands](docs/formal-scenes.md)

## Current Boundaries

The 2D backend does not represent housework-simulation capability. The current code has formal 3D household environments, foundation-model perception, a world model, latent RL, language/OOD holdouts, and executable training gates, but has not yet produced training-success evidence for a new lineage. Until all three textured household scenes, the expert-free non-privileged visual policy, real bimanual contact, three training seeds, and isolated-seed evaluations pass, the project claims only that “the serious experimental platform has been implemented”; it does not claim that the robot has learned housework or extrapolate to open-world general capability.
