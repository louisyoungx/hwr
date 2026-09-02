# Formal 3D Visual Training Runbook

> Status: Deprecated V1/V2 behavioral-cloning baseline; do not use for current formal training
>
> Current approach: [Foundation-Model Perception, World Models, and Imagination Reinforcement Learning Paradigm](./foundation-world-model-training-paradigm.md)
>
> The legacy visual-training commands in this document are only for reviewing historical runs. Do not use them to start new training before the unified development gates are complete.
>
> Configuration: [formal_visual_v1.json](../configs/training/formal_visual_v1.json)
> Training results: [formal_visual_v1_results.json](../configs/training/formal_visual_v1_results.json)

This page retains only traceable records of failed experiments; the listed data, checkpoints, and commands must not enter the current training lineage. The first ordinary behavioral-cloning run could navigate near the dining table on unseen dining-room seed 30000, but it confused arm manipulation with the subsequent navigation stage, timed out after 6000 steps, and made no grasping contact. V2 then added phase classification and phase-specific action heads through the training label `phase`; this introduced further dependence on manual structure without solving closed-loop generalization.

See [formal_visual_v2.json](../configs/training/formal_visual_v2.json) for the V2 data and training configuration. The three V2 datasets still use the same 9 successful expert Episodes, but were resampled under `hwr.visual-behavior-dataset/v2` and locked to new shard hashes.

## Data Boundary

At the time, each of the three tasks used 3 expert Episodes, and the action space covered only one arm. These data can reproduce historical conclusions only; the training loader must mark their source as `legacy` and reject them for current dual-arm Actor-Critic training.

| Task | Training seeds | Sample count | Data directory |
|---|---:|---:|---|
| Dining-table clearing | 1000–1002 | 2485 | `datasets/formal-v1-r4/clear_dining_table_3d_v1-expert-s1000` |
| Living-room tidying | 2000–2002 | 2198 | `datasets/formal-v1-r5/tidy_living_room_3d_v1-expert-s2000` |
| Storing kitchen items | 3000–3002 | 4059 | `datasets/formal-v1-r6/store_kitchen_items_3d_v1-expert-s3000` |

The data directories are ignored by Git; the version-controlled training configuration stores the SHA-256 for each shard. Before training, the loader checks the manifest, field allowlist, and shard hashes again.

## Historical Reproduction Commands

The following commands are only for auditing the old failed baseline and are not current training entry points:

```bash
.venv/bin/python -m hwr.apps.train_formal_visual \
  --dataset datasets/formal-v1-r4/clear_dining_table_3d_v1-expert-s1000 \
  --run-id formal-v1-dining-s0 --epochs 30 --batch-size 128 --seed 0

.venv/bin/python -m hwr.apps.train_formal_visual \
  --dataset datasets/formal-v1-r5/tidy_living_room_3d_v1-expert-s2000 \
  --run-id formal-v1-living-s0 --epochs 30 --batch-size 128 --seed 0

.venv/bin/python -m hwr.apps.train_formal_visual \
  --dataset datasets/formal-v1-r6/store_kitchen_items_3d_v1-expert-s3000 \
  --run-id formal-v1-kitchen-s0 --epochs 30 --batch-size 128 --seed 0
```

The trainer automatically selects MPS, CUDA, or CPU on the local machine, saves the best validation checkpoint, and immediately reloads it from disk once. Large files in `models/` and `runs/` are ignored by Git; after training, record the model hash, device, loss, and evaluation seeds in the version-controlled run manifest.

The first three policies were each trained for 30 epochs on local MPS and successfully reloaded from disk. The version-controlled results manifest records the training-code commit, training seed, loss, and checkpoint SHA-256; model files remain in `models/formal-v1/`. These results show only that the local training pipeline runs end to end; they do not demonstrate that the closed-loop gates have been met.

## Closed-Loop Evaluation Gates

- Use 20 seeds disjoint from the training set for each task;
- the inference action source must begin with `learned:`;
- success rate must be at least 70%;
- total severe collisions must be 0;
- the target state in every successful Episode must remain stable for at least 2 seconds;
- videos must come from the same checkpoint, seed, and unedited Episode.

A decrease in offline loss proves only that the trainer works; it does not mean that the household task was completed. The training stage passes only when the closed-loop report after checkpoint reload meets the gates above.
