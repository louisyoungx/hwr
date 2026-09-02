# Housework Robot

A research codebase for embodied household robotics, covering simulation, robot runtime interfaces, data collection, policy training, safety constraints, and closed-loop evaluation.

The project follows a capability-first approach: first demonstrate that the robot can complete a task in a physical closed loop, then progress toward learned state policies, visual policies, cross-domain generalization, and hardware transfer. World models, reinforcement learning, behavior cloning, and scripted controllers in this repository are experimental routes that can be compared or replaced; their presence does not imply that the corresponding capability has been achieved.

For the latest research status and experimental conclusions, see [Research Loop Status](docs/research-loop/README.md). This README intentionally contains only stable project-level information so that it does not become outdated after every experiment.

## Repository Layout

- `src/hwr/`: runtime contracts, simulation adapters, tasks, policies, training, and evaluation code;
- `configs/`: task, scene, training, and evaluation configurations;
- `assets/`: MuJoCo models, household-scene assets, and manifests;
- `scripts/`: environment checks, asset preparation, and experiment utilities;
- `tests/`: tests for core contracts, physics backends, training, and evaluation;
- `benchmarks/`: historical benchmarks and reproduction instructions;
- `docs/`: architecture notes, research paradigms, and experiment records;
- `runs/`, `models/`, and `datasets/`: local experiment artifacts, which do not by themselves constitute capability evidence.

## Quick Start

Python 3.11 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Install the optional dependencies for MuJoCo 3D simulation and asset tooling when needed:

```bash
python -m pip install -e ".[dev,sim3d,assets3d,video]"
python scripts/verify_robot_model.py
python scripts/verify_physics_integrity.py
python -m hwr.apps.render_3d_smoke --output-path artifacts/3d-smoke.png
```

Foundation-model experiments require the additional `foundation` dependency group. Use the relevant research-round record or topic document for experiment-specific commands rather than inferring a current mainline from this README.

## Research and Evidence

The project distinguishes three evidence tiers:

- `development`: used to establish solvability and locate bottlenecks; privileged state, teachers, and fixed development seeds are allowed;
- `capability`: closed-loop evaluation of a policy that uses only declared deployment observations on a frozen distribution;
- `claim`: generalization evidence from a sealed unseen distribution or real hardware.

Passing tests, reducing loss, improving trajectories, or completing one task stage does not automatically demonstrate improved household capability. See [AGENTS.md](AGENTS.md) for the capability ladder, evaluation isolation rules, stopping conditions, and conclusion vocabulary.

## Documentation

- [Current research status](docs/research-loop/README.md)
- [Platform architecture](docs/architecture.md)
- [Training and simulation plan](docs/training-and-simulation-plan.md)
- [3D scene documentation](docs/formal-scenes.md)
- [MuJoCo backend architecture decision](docs/adr/0001-mujoco-3d-backend.md)
- [Historical benchmarks and reproduction](benchmarks/README.md)

## Current Boundaries

This is a research platform, not a robot product with demonstrated general household capability. The presence of a model, controller, task, or evaluator means that an experimental route can be studied; capability claims require closed-loop behavioral evidence under frozen conditions.
