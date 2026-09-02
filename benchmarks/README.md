# Housework Scene Training Benchmarks

## Admission Criteria

- At least three different housework scenes;
- At least 20 isolated-seed closed-loop evaluations per scene;
- A success rate of at least 70%;
- An average collision count of 0;
- Reports containing dataset checksums, complete training configurations, and model paths.

Run the check:

```bash
python3 scripts/verify_benchmarks.py
```

## Current Results

| Scene | Training Episodes | Samples | Closed-loop success rate | Average steps | Average collisions |
|---|---:|---:|---:|---:|---:|
| Tidy Table | 100 | 36,374 | 100%（20/20） | 317.55 | 0 |
| Sort Laundry | 155 | 55,413 | 100%（20/20） | 323.95 | 0 |
| Clear Dishes | 155 | 60,267 | 100%（20/20） | 339.95 | 0 |

Each model independently generates data, trains, and is registered; no rule-based expert is used for final evaluation actions. The rule-based expert is used only for initial demonstrations and corrective labels for states accessed by the policy.

## Video Reproduction

The following command reads the three version-controlled benchmark reports, loads the actual model checkpoints registered in those reports, reruns closed-loop inference with the first isolated evaluation seed for each scene, and generates a synchronized side-by-side video:

```bash
PYTHONPATH=src python3 -m hwr.apps.render_benchmarks \
  --output-path artifacts/benchmark-rollouts.mp4
```

Video metadata is written to a same-named `.json` file and includes model versions, seeds, closed-loop results, and the video checksum. Rendering reads only immutable simulation snapshots and does not participate in policy inputs, action filtering, or success judgments. Python `Pillow` and system `ffmpeg` are required.

## Reproducible Training Commands

```bash
PYTHONPATH=src python3 -m hwr.apps.train_scenario tidy_table/v1 \
  --run-id tidy-table-v3-mps --episodes 60 --eval-episodes 20 \
  --epochs 40 --batch-size 512 --device mps --seed 300 \
  --aggregation-rounds 2 --aggregation-episodes 20 \
  --expert-action-probability 0.3 \
  --report-path benchmarks/results/tidy-table-v1.json

PYTHONPATH=src python3 -m hwr.apps.train_scenario sort_laundry/v1 \
  --run-id sort-laundry-v2-mps --episodes 80 --eval-episodes 20 \
  --epochs 50 --batch-size 512 --device mps --seed 500 \
  --aggregation-rounds 3 --aggregation-episodes 25 \
  --expert-action-probability 0.35 \
  --report-path benchmarks/results/sort-laundry-v1.json

PYTHONPATH=src python3 -m hwr.apps.train_scenario clear_dishes/v1 \
  --run-id clear-dishes-v1-mps --episodes 80 --eval-episodes 20 \
  --epochs 50 --batch-size 512 --device mps --seed 600 \
  --aggregation-rounds 3 --aggregation-episodes 25 \
  --expert-action-probability 0.35 \
  --report-path benchmarks/results/clear-dishes-v1.json
```

Data, models, and runtime artifacts are written to `datasets/`, `models/`, and `runs/`, respectively; these large files are not committed to Git. Small auditable result reports are stored in `benchmarks/results/`.
