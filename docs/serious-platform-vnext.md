# From Toy Closed Loops to a Serious Housework Research Platform

> Status: Development contract implemented; training capability still awaits validation on a new lineage
> Date: 2026-08-14

“More code and larger models” do not make a system non-toy. This project fixes the minimum definition of seriousness as follows: formal tasks and training entry points match, the deployment path is genuinely trainable, data evidence matches what the optimizer actually sees, the safety model cannot be exploited by the Actor, evaluation includes statistical repeats and out-of-distribution variation, and failure can trigger a clear stop instead of more piled-on updates.

## Assessment Matrix

| Toy characteristic | Current implementation | Required evidence |
|---|---|---|
| Training on a simplified proxy task while naming the result after a household scene | Training and evaluation directly run the three formal multi-object MJCF scenes: living room, dining room, and kitchen | The run manifest contains only three `formal_3d_v1` task IDs |
| The visual backbone is trained while the fusion layer consumed by the policy is random | Deployment representation-alignment loss runs through cross-camera fusion, temporal fusion, temporal positions, and output normalization | All four gradient groups are nonzero and parameters change in `training_semantics` |
| The Episode has contact, but Replay drops contact segments | Retain at most 7 significant-priority windows per Episode; recompute interaction evidence from retained transitions | A 13,440-transition limit and per-shard interaction trace |
| The safety layer changes actions, but the imagination model can output arbitrary actions | Per-dimension bounds for all 16 dimensions, plus an independent held-out set for safe-action execution | Recall, PR-AUC, Brier, two types of RMSE, and 0 out-of-bounds rate |
| A single fixed instruction merely memorizes the task ID | Four training rewrites and three non-overlapping evaluation rewrites per task | The instruction split and original text for each evaluation Episode |
| Only the random seed changes, leaving the same narrow simulation distribution | Evaluation uses broader physics, visual, calibration, actuator, and latency ranges | A complete randomization audit for every Episode |
| One accidental run is used to claim capability | 40 unseen seeds per task, Wilson intervals, left/right single-arm ablations, and 3 training seeds | An aggregate manifest for three runs and per-Episode reports |
| Lower loss is treated as learned control | Separate action probes, single-step use, multistep causality, safety/collision gates, and physical success | No deployment when a gate fails |

## Formal Run Objects

The three tasks are not placeholder circles or tabletop grasping:

- Living room: Move through the furniture and place a rubber duck and a mini soccer ball in a wicker basket;
- Dining room: Place a teacup and a wooden plate in separate target trays;
- Kitchen: Physically pull open a drawer without an actuator and place two cleaning bottles in the left and right compartments.

All tasks use a four-wheel mobile base, left and right 6-DOF arms, two-finger grippers, head RGB-D, and left/right wrist RGB. Success requires both manipulated objects to remain stable for 2 seconds, zero severe collisions, real two-finger contact by each arm, and at least 0.5 seconds of concurrent contact. The kitchen also requires drawer opening of at least 0.30 m. Success, rewards, and safety are environment interfaces, not action hints for the Actor.

## Training Order After Development Completion

Complete all development first, then allow any formal parameter updates. After that, control compute investment by evidence:

1. `development-ready/v3` passes the full test suite, real foundation-model inference, and executable training-semantics checks on the current committed snapshot;
2. Run a 24-Episode calibration. If per-task action probes, deployment visual gradients, held-out loss, or single-step physical causality do not improve, stop automatically and do not enter Actor collection;
3. After calibration passes, run one complete 120-Episode seed. Replication to the other two independent seeds is worthwhile only if it produces a qualified deployment and formal physical progress;
4. Each of the three seeds completes normal and single-arm-ablation evaluation over 40 seeds per task; the aggregation entry point then decides formal acceptance.

This is not trial and error during development; code, tasks, data, gates, evaluation, and resource policy are frozen before Step 1. The later tiers exist only to avoid wasting tens of hours of local compute on a falsified model.

## Capabilities Not Currently Claimable

The current commit shows only that the engineering loop has been upgraded from proxy benchmarks to an auditable three-scene research platform; it does not show that the model has learned housework. Even if all three tasks eventually pass, that proves only that these task families work under the declared language and physics perturbations; it does not imply open-world general housework, zero-shot manipulation of unknown objects, or real-robot sim-to-real success. To expand the capability boundary later, add acceptance tests on unseen layouts/object categories and real-robot replays rather than changing the wording of existing results.

See [Foundation-Model–World-Model Training Paradigm](foundation-world-model-training-paradigm.md) for formal technical details and [3D V1 Acceptance Contract](three-dimensional-v1-acceptance.md) for the definition of physical success.
