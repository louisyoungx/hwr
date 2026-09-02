# Local Training Progress

This document records only reviewable training facts and historical changes. It does not replace
the [current training paradigm](./foundation-world-model-training-paradigm.md) or the final
acceptance threshold. P076–P080 are frozen as the failed baseline; policy training will not
restart until the unified development gate is complete.

## 2026-08-14 “Toy Loop” Blockers Fixed; Formal Training Still Not Started

The latest external review identified two new P0 issues: the cross-camera/temporal fused
representation actually consumed by the policy had no gradient, and retaining only the first and
last windows of each Episode meant interaction segments never entered Replay. Both were fixed as
trainability defects, without avoiding them by lowering causality, safety, or success thresholds.

Vision objectives now include deployment-representation alignment; gradients must pass through
`camera_fusion`, `temporal_fusion`, `temporal_position`, and `output_norm`. The new
`development-ready/v3` no longer trusts the test suite alone: it performs a real optimization
step on an isolated commit and requires nonzero gradients and actual parameter changes in all
four components. Any regression locks training.

Ordinary Replay now retains at most 7 non-overlapping 16-transition windows per autonomous
Episode, prioritizing physically salient windows and filling the remainder with boundary and
uniform coverage. The interaction trace is recomputed from retained transitions when slicing,
and Actor admission no longer reads discarded Episode summary evidence. A 120-Episode run can
retain 13,440 dynamics transitions; only 2 windows per Episode generate DINOv3/SigLIP2 teacher
features, controlling visual-cache cost.

The independent holdout remains physically isolated from optimizers: at startup, collect 8
128-transition system-identification Episodes per task. Also collect 8+8 positive/negative
safety-action-intervention Episodes per task to validate intervention recall, PR-AUC, Brier,
executed-action error, non-intervention identity error, and out-of-bounds rate. After the
exploration Actor unlocks, collect an additional 8+8 positive/negative severe-collision
Episodes per task. None of the three holdout datasets generates teacher features. The current
uncompressed resource upper bound is approximately `28.41 GiB`; startup is rejected above the
`30 GiB` configuration limit or below `35 GiB` free space.

The action-identifiability probe now predicts changes in joint velocity, gripper position, and
base velocity 1/4/8/16 steps ahead, with bootstrap clustered by original Episode. The physical
action-causality gate constrains only visual latents and proprioceptive dynamics. Sparse reward,
termination, and safety heads remain diagnostic, but are no longer incorrectly required to use
the same `1.05` action-shuffling ratio. Collision-head validation now uses per-transition
endpoint recall, PR-AUC, Brier, false-positive rate, collision temporal alignment, and
action-shuffling sensitivity at fixed posterior states; a 16-step window maximum may not conceal
the wrong alert time.

The world model's generic “current latent state + Actor proposed action -> safety-layer executed
action” residual model is now constrained dimension by dimension by the formal action contract,
with supervision from real proposal/executed-action pairs in Replay. Imagined trajectories for
both the task Actor and intrinsic exploration Actor advance the RSSM with predicted executed
actions while penalizing safety-intervention probability, action-rewrite magnitude, and severe
collisions. The hard safety filter in deployment remains independent of the learned model. When
the safety-action holdout gate fails, neither Actor may train or collect data.

The training and final-evaluation entry points have migrated from the `bimanual_household_v1`
proxy task to three formal multi-object household scenes: storing a duck and a soccer ball in
the living room, placing tableware into separate dining-table locations, and opening a kitchen
drawer and storing two cleaning-product bottles. Each task has 4 training commands and 3
non-overlapping evaluation paraphrases. Evaluation uses broader and partly out-of-distribution
mass, friction, lighting, material, RGB/depth noise, camera mounting pose/focal length, actuator
scale, action latency, and observation latency. Success also requires separate left/right arm
contact and at least 0.5 seconds of concurrent contact. This prevents the next lineage from
presenting a proxy benchmark as a household task, but code completion still does not imply model
capability; until the full gate and subsequent training/evaluation pass, the project claims only
that “a serious experimental platform has been implemented; capability is not yet proven.”

## 2026-08-14 `foundation-wm-007` Stopped, Pruned, and Follow-Up Lineage Statistics Fixed

The last complete checkpoint of `foundation-wm-007` contained 111 Episodes and 7,400 updates;
the following cycle was interrupted at 7,490 updates. All three task success counts were 0, the
action probe was approximately `1.002`, the multi-step action-causality ratio approximately
`0.995`, and the exploration Actor never unlocked. It is bound to the old commit `bb8f743`,
cannot resume from the current HEAD, and does not enter subsequent checkpoint lineage.
`foundation-wm-006/007` have no deployment. Their rebuildable feature caches, replay,
non-deployable checkpoints, and holdout caches were deleted, retaining only manifests,
per-cycle metrics, causality diagnostics, and failure notes. `runs/foundation-world-model`
therefore fell from approximately 73GB to 3.1MB, releasing approximately 73GiB.

The following records the staged design before the current training-accessibility fixes; it has
been superseded by the new contract at the top of this document. At the time, the next run was
planned to fit the data-action probe independently per task, with every task required to pass;
the bootstrap unit changed from correlated transitions to Episode. The action-causality holdout
collector was upgraded to v3, with 16 independent Episodes per task balanced by severe-collision
outcome into 8 positives and 8 negatives; each task audited 64 windows. Collision batches
prioritized sampling across independent Episodes, and Episodes shorter than 16 transitions no
longer counted toward interaction coverage. The collision head also had to pass recall, PR-AUC,
and Brier score on an independent holdout, rather than merely proving that it had “seen one
positive example.”

Actor admission was split into two levels: action coverage, per-task probes, and one-step
physical causality unlock only the generic intrinsic exploration Actor; contact, controlled
motion, collision-head validation, and complete multi-step causality constrain only the task
Actor. The 24-Episode random-calibration stop also checks only first-level evidence and no longer
requires the random policy to complete contact discovery that the exploration Actor is responsible for.

The fixed one-time Actor warm-up was removed. A new Actor must update at least 200 times and
check gradients, motion/gripper entropy, and imagined-return stability over the latest three
50-update windows; terminate after at most 1,000 updates if it still fails. The single-run final
evaluation schema was raised to `hwr.foundation-evaluation-run/v3`, which may produce only
`per_seed_passed` and may not produce formal `passed=true`. A multi-run aggregation entry was
added; formal pass is written only when three different training seeds, identical immutable
configuration, disjoint training/holdout seeds, and three per-seed passing results all hold.

The run manifest was raised to `hwr.foundation-online-run/v4` and records the actual device,
nice value, and MPS watermarks. Formal acceptance defaults to 40 unseen seeds per task, uses the
95% Wilson lower bound for success rate and the Wilson upper bound for single-arm ablations, and
requires at least 3 independent training seeds per candidate configuration; a single-seed result
is for calibration only.

## 2026-08-13 `foundation-wm-006` Failed Baseline and Training Refactor

`foundation-wm-006` was stopped intentionally, will not resume, and cannot be the parent of a
later checkpoint. Its last complete point had only 9 training Episodes and 600 updates; all
three task success counts were 0, with no deployment or unseen-seed evaluation. World-model
total prediction error fell from `4.383` to `3.681`, but the shuffled/real action-error ratios
at 200, 400, and 600 updates were `0.99993`, `0.99989`, and `0.99999`, far below `1.05`. This
shows that the model mainly used state and video continuity and did not learn action causality
usable for policy imagination. The run is retained only as a negative control.

The original runner switched to the current Actor after 6 random Episodes based only on Episode
count, independently of causality evidence. This was a design defect and may no longer be
avoided by adding training time. The three current `carry/hold` tasks are explicitly downgraded
to a “bimanual pretraining and action-causality benchmark”; they are not formal acceptance of
`tidy_living_room_3d`, `clear_dining_table_3d`, or `store_kitchen_items_3d` household scenes.

The refactor follows “complete all development before training”: first persist complete losses,
gradients, action coverage, environment outcomes, and stage durations, and provide a local
read-only dashboard that does not read large files. Then unlock Actor training and collection
jointly through physical action causality, data identifiability, and actual action coverage, and
connect a task-agnostic curriculum using only state novelty, TD error, reward-improvement speed,
and failure boundaries. Do not start the next run until causality-statistics calibration,
intrinsic RL exploration, and replay I/O optimization are complete.

Observability and the first admission gate are complete. The runner no longer reads
`initial_random_episodes`: before admission, the task Actor and Value are not updated and the
Actor is not used for collection; admission state is restored atomically with the checkpoint.
The causality report was upgraded to `hwr.foundation-action-causality/v5`, adding one-step
visual/proprioceptive action-utilization diagnostics at fixed real posterior states. Each audit
performs 5 independent permutations and saves raw results; every permutation must pass and the
5th-percentile error ratio must clear the threshold. An independent ridge-regression probe using
state-only/state+executed-action also checks whether replay has action identifiability. The Actor
unlocks only after action coverage, effective rank, the probe bootstrap lower bound, and at least
12 replay Episodes pass twice consecutively. The recovery schema was raised in sync to
`hwr.foundation-runner-recovery/v4`. The v4 causality report and v3 recovery state from
`foundation-wm-006` cannot enter this new lineage.

Physical-causality admission and task-Actor admission were split further. After one-step
physical causality, the data probe, and action coverage pass consecutively, first train the
independent `intrinsic_rl_actor`. It maximizes only world-model ensemble uncertainty, latent
state novelty, and policy entropy, subtracting predicted safety intervention and using no
environment reward, task object, distance, or stage. Train and collect with the task `rl_actor`
only after complete multi-step five-head causality also passes consecutively. The exploration
Actor, exploration Value, slow Value, and optimizer exist only in training checkpoints and do
not enter deployment. The autonomous-trajectory schema was raised to
`hwr.autonomous-trajectory/v4`; no-expert lineage explicitly allows three action sources:
random exploration, intrinsic RL exploration, and task RL. None contains expert, demonstration,
teacher-action, or searcher data.

The task-agnostic physical-state frontier is now integrated into the unified runner. During
random cold start, save complete simulation-state candidates only from real autonomous
transitions and do not perform frontier reset; after the physical action-identifiability gate
passes, mix frontier starts at the default probability `0.20`. Candidates come only from each
Episode's world-model posterior states, one-step TD error, local reward-improvement speed, and
environment terminal-failure boundaries. States after termination/truncation and states with
safety intervention cannot enter the candidate pool. Before restoration, validate the backend
fingerprint, position, velocity, acceleration, actuator control, solver, and runtime state
together; failure returns to ordinary reset. The candidate pool, independent random streams,
selection count, and per-Episode source are atomically restored with the checkpoint, raising
the recovery schema to `hwr.foundation-runner-recovery/v5`. Frontier snapshots do not enter
replay, the vision student, the world model, or Actor input.

Local throughput and unified-memory optimizations are now in formal configuration. A training
batch still samples all replay windows with equal probability, but windows within a batch come
from one Episode shard to avoid repeatedly decompressing large files. The two frozen visual
feature sets are deduplicated by content hash first; each encoder reads a historical frame only
once, retaining at most 16 read-only LRU entries. The world model still updates every update,
while the vision student updates every fixed 4 updates. Other updates encode only gradient-free
microbatches of 8 observations, build no cross-camera correspondences, and do not read or move
DINOv3/SigLIP teacher targets. Vision loss is averaged only over actual vision updates, with
`trainer/visual_updated` recording the update ratio separately. This scheduling reads only the
global update count, not task, reward, object, contact, or stage.

## 2026-08-12 Foundation-Model/World-Model Mainline Rebuild

The old P081 number was not continued. `foundation-wm-001`–`003` were archived as invalid
development runs; they cannot be interpreted as evidence that the model acquired household
capability and cannot serve as parent models for the next formal lineage.

The single connected path is:

```text
Raw RGB-D from four cameras + dynamic extrinsics and intrinsics
  -> SigLIP2 / DINOv3 ViT-S/16 / Qwen3-Embedding offline continuous-feature cache
  -> 24.4M-parameter high-resolution visual student
  -> 13.0M-parameter action-conditioned categorical RSSM
  -> Imagination-space Actor-Critic
  -> The same 16-dimensional Actor
  -> Independent safety layer
  -> Three-task MuJoCo / future real-robot RuntimeBackend
```

Key implementation facts:

- Foundation-model revisions, continuous-representation specification, licenses, and expected-file SHA-256 values are locked; formal control loops do not load teachers. Official DINOv3 weights still require download by a Hugging Face account that has accepted Meta's terms; when absent, the development gate remains locked;
- Autonomous trajectories accept only `random_rl_exploration` and `rl_actor` as action sources, while recording both Actor proposals and actions actually executed by the safety layer;
- RSSM transitions and imagined rollouts use the same physical action units as real replay;
- All three tasks share one collector, cache, batch loader, Trainer, Actor, and checkpoint lineage;
- The action-causality gate at that time used an independent holdout of two fixed random-RL Episodes per task; this historical configuration was replaced by the 8-Episode system-identification set and delayed-collision calibration set at the top;
- Lateral reflection may be declared only by the environment; the generic augmenter transforms vision, dynamic calibration, proprioception, actions, and continuous teacher features;
- Deployment export contains only the vision student, RSSM posterior state filter, and Actor—not foundation teachers, Critic, reward/continue/safety prediction heads, or training optimizers;
- Evaluation is fixed at at least 40 unseen seeds per task, running normal, left-arm-locked, and right-arm-locked conditions. Normal Episodes synchronously record third-person, head, left-wrist, and right-wrist video; each task must retain at least one successful Episode, and frame count must equal control steps plus the initial frame exactly or the entire acceptance fails.

The development-gate command is:

```bash
.venv/bin/python scripts/verify_development_ready.py \
  --foundation-device cpu \
  --output artifacts/development-ready.json
```

It checks that protected source is committed and, in an isolated snapshot of the current commit,
checks whole-repository Python sizes, architecture boundaries, and the full test suite. It then
checks foundation-weight hashes, real fixture inference for the three frozen models, unified
three-task configuration, action-unit consistency, and deployment stripping. The formal training
entry point has no gate-bypass parameter and checks the source commit, configuration hash, and
protected-source tree hash again.

The development-completion review found that the old `hwr.foundation-development-ready/v1`
check required only that checks appearing in the report passed; it did not require all ten
checks. A manually incomplete report containing only tests and architecture checks could
therefore unlock training. The old AST audit also covered only some runner files and omitted the
vision student, trajectory data, Actor, world model, foundation adapters, and deployment
evaluation. The readiness schema is now v2: the check set must exactly equal the ten mandatory
pieces of evidence, and the isolated-snapshot commit must bind to architecture, size, and full
test checks. Algorithm auditing now discovers files automatically by the full foundation-module
pattern, currently covering 53 Python files. Formal configuration must also assemble exactly the
three target tasks and contain no lineage keys for experts, demonstrations, action labels,
waypoints, skill stages, object tokens, target tokens, or old checkpoints. Old v1 reports can no
longer start training.

Runtime-lineage review also found that old final evaluation read only `source_commit` from
checkpoint lineage and did not check expert, demonstration, behavior-cloning, teacher-action,
action-search, or old-checkpoint declarations; the run manifest itself also lacked the last
three. Saving, resuming, checkpoint loading, and final evaluation now share one exact no-expert
lineage. Any missing field, extra field, non-empty value, or source-commit mismatch fails.
The run-manifest schema was raised to `hwr.foundation-online-run/v2`; old v1 runs cannot silently
enter current formal acceptance.

The traceability audit between the development gate and training artifacts found that the
training entry read the readiness report but passed only its `source_commit` to the runner; the
report itself was not copied into the run, and no hash entered the run manifest. Final evaluation
therefore could not prove which DINOv3 weights, real inference, and full-test gate had preceded
startup. The current training entry atomically copies `development-ready.json` into the new run,
and the manifest records its schema, fixed path, and SHA-256. Resume requires the external report
and in-run copy to be identical; final evaluation rechecks all ten pieces of evidence, binds them
to the isolated commit, and includes the copy in the final artifact manifest. The run-manifest
schema for this stage was raised to `hwr.foundation-online-run/v3`; old v1/v2 cannot enter formal
acceptance for this stage, and the 2026-08-14 resource-trace fix raised it again to v4 described
at the top of this document.

After readiness traceability was integrated, the online runner briefly grew to 803 lines,
exceeding the project's hard limit of 800. Creation, atomic publication, and resume-consistency
validation for the run manifest were moved into `hwr.train.foundation_run_manifest`; subsequent
admission statistics were also split into a separate module, keeping the runner within 800
lines. This is a separation of responsibilities, not a way to evade the size check by merging
statements.

The final-evidence manifest was raised at the same time to
`hwr.foundation-evaluation-run/v2`. It no longer hashes only evaluation JSON, the action-causality
report, and videos; it directly binds readiness, run/latest, all training Episode records,
training replay manifest, causality-holdout manifest, training checkpoint manifest/weights,
deployment manifest/weights, per-Episode evaluation, acceptance results, and every unedited video
stream, so data, models, code, configuration, and results can be checked along one hash chain.

The old privileged-expert completion-rate test is no longer part of formal acceptance: the
expert in the current commit failed on all 11 fixed regression seeds, again validating the
decision to abandon the expert path. It does not enter collection, replay, Actor, or world model;
the gate continues to prohibit new training-source imports of the expert through AST auditing.

A pre-start capacity audit found that permanently retaining two three-view dense grids for 120
long Episodes would exceed 400 GB in theory, while the machine then had approximately 130 GB
available. Formal configuration therefore added task-fair rolling replay capped at 18,000
transitions and retained only the latest 3 checkpoint/deployment sets. Eviction uses only task
identity, time, and fixed capacity; it does not read scene semantics, distance thresholds, or
action content. After the capacity fix, the overall gate must be rerun, and previously generated
reports can no longer unlock training.

The first launch of `foundation-wm-001` was invalidated and stopped. It collected only three
random-exploration Episodes, one from each task, for 4,000 transitions total. The process
stopped during foundation-vision feature materialization, before the first gradient update, and
produced no `latest.json`, training checkpoint, or deployment model. The directory is retained
for audit and will not be resumed or merged into a formal run. The reason was not training
failure: a post-start audit found that action-shuffling diagnostics had computation functions
but were not persisted as a mandatory publication condition, and did not verify that shuffled
action error must exceed real-action error.

The corrected mandatory conditions are: counterfactual error covers future visual latents,
proprioception, reward, termination, and safety outcomes; actions come only from actions
actually executed by the safety layer in replay; the shuffled-action error ratio is at least
`1.05`, with at least `60%` of horizons degrading. Each update report binds source, data
manifest, and checkpoint hashes. Failed updates retain a training checkpoint only and cannot
export a deployment model; fixed-seed evaluation rechecks hash consistency among the training
checkpoint, deployment manifest, and report. Causality auditing later moved from the “last
training batch” to a task-balanced holdout physically isolated from optimizers, eliminating
sample leakage, two-sequence small-sample effects, and strong-task masking of weak tasks.
The formal batch was reduced from 4 to 2 to fit the 48 GB MPS visual-backpropagation peak;
model, total update count, three-task scope, and final 180-Episode acceptance were not reduced.

`foundation-wm-002` completed one random-exploration Episode per task on commit `e6edf67`, for
4,000 transitions total, and materialized all SigLIP2, DINOv2, and three-language-command
features. To avoid duplicate computation, 5,894 teacher features addressed by raw observation,
model lock, and preprocessing SHA-256 were reused from `foundation-wm-001` through hard links.
Before reuse, all 1,469 keys generated by the new run existed in the old cache; byte hashes for
12 sampled files were identical. Replay, Episodes, actions, models, and optimizers were not
reused. The operation is recorded in the run's `cache-reuse.json`.

During its first joint update, the run completed vision-student and world-model updates in
sequence, then exited abnormally on entering imagined Actor-Critic: the main Value was on MPS,
while the slow target Value created after initialization remained on CPU. Because the process
did not complete one `train_step`, `update_count` remained 0 and there was no `latest.json`,
checkpoint, deployment model, or action-causality report; `foundation-wm-002` therefore cannot
count as a training result. The fix deep-copies the slow target from the main Value so device,
dtype, and network structure match exactly, and uses `try/finally` to restore the world model's
original gradient state when imagination updates fail.

The fixed formal-scale local smoke test directly reads the run's real four-camera sequence and
teacher features, and fully executes four optimizers on MPS with batch 2 and 16 transitions.
One update takes approximately 4.98 seconds; total vision loss is approximately 2.894, total
world-model loss approximately 12.273, Actor loss approximately 2.020, and Value loss
approximately 5.727. PyTorch reports approximately 1.88 GB of MPS tensor usage and 30.46 GB of
total driver allocation, showing usable headroom on the 48 GB machine. After the full gate
passed again, `foundation-wm-003` was started without restoring any unpersisted parameters.

`foundation-wm-003` collected 6 Episodes and 8,000 transitions on commit `4ae381f` and
published training checkpoint `update-000000200`. Its shuffled-action error ratio was `0.9973`,
below `1.05`, and only `50%` of horizons degraded, so the causality gate correctly rejected
deployment. A subsequent static design review found that the formal implementation still used
DINOv2-Small, inconsistent with the approved DINOv3 perception plan, so the run was stopped
intentionally before the second update completed. Its replay, checkpoint, and failed-causality
report are retained for audit, but its feature caches, parameters, and optimizer must not enter
the new DINOv3 lineage.

The new dense teacher is fixed to the official `facebook/dinov3-vits16-pretrain-lvd1689m`
revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`. Core training modules also changed from
model brand names to the roles `vision_language` / `dense_vision`; the model lock added a
continuous-representation specification so any concrete output-layer change necessarily changes
the cache key. The next run may be created only after official weights, real CPU/MPS inference,
and the full development gate pass again.

Pre-training static Actor review also found that old-initialization random gripper actions
fell only around `0.42～0.59`; none of 20,000 samples approached fully open or fully closed,
so effective grasp exploration was impossible. Entropy reward also incorrectly used untransformed
`tanh/sigmoid` Gaussian entropy. It now uses the sampling entropy of the transformed action
distribution and adds entropy reward to imagined reward before computing λ-return, allowing it
to propagate across the imagined horizon. In fixed audits, the new generic random initialization
has approximately `9.3%` of gripper samples below `0.1` and `9.2%` above `0.9`; it reads no
observation, task, or scene and contains no grasp-action template.

The crash-recovery path was also refactored. Old `--resume` loaded only the model and root
runner state; if the process exited after replay append or capacity trimming but before
checkpoint publication, it could reuse seeds or reference deleted shards. Each checkpoint now
contains four hash snapshots for runner, Episodes, training replay, and causality holdout. Shards
pending trimming first enter a recovery staging area and are cleaned only after the new
`latest.json` is published atomically. Fixed regression simulated the crash point where old
shards were trimmed and new shards were not in the checkpoint; recovery restores old shards,
discards uncommitted shards, and restores exact record counts.

The foundation-model real-inference gate changed from “record metrics only” to “fail when
metrics do not qualify.” Existing official weights were revalidated on local MPS: Qwen3
Chinese paraphrase/different-intent cosine scores were `0.6280/0.3851`; SigLIP2 scores were
`0.8428/0.7977`; visual output was `3×14×14×768`, with mean patch change `0.0220` and mean
difference `0.0131` between two valid fixture images. Both passed the non-degeneracy gate;
DINOv3 still awaits official-account authorization for the same real gate.

Final-evaluation seed isolation was also completed: the generator now jointly excludes complete
training-Episode records and the action-causality holdout instead of relying only on a default
seed offset to “happen to be disjoint.” Explicit `--seed-start` also cannot reuse diagnostic data.

The pre-training task-interface audit found that the old success criterion required only historical
bilateral synchronized contact and final stable placement, leaving a terminal-reward loophole:
“touch with both arms first, release, then push into place.” Transport success now requires
continuous bimanual-contact motion evidence from the Episode's initial distance to the target
tolerance; drawer opening requires continuous left-arm contact motion evidence. Valid placement
retains evidence, while leaving the target tolerance revokes it. Fixed regressions cover both
uncontrolled pushing into place and moving an object away after controlled placement and pushing
it back; neither can succeed.

The autonomous-collection audit also found that the old random source independently redrew every
action every 50 ms, with grippers randomly opening and closing step by step, so physical
trajectories were mostly unusable high-frequency jitter. The new source is one global,
task-blind random process: motion correlation is `0.96`, and each gripper has an independent
per-step flip probability of `0.05`. In fixed 512-step regressions, adjacent-motion correlation
exceeded `0.90`, and actual gripper flip rate was `2%～8%`. The source still reads no
observation, task, or scene, and each Episode records the exact process configuration. The
action-causality holdout uses the same process with independent seeds and storage.

Static audit of the task-sampling curriculum found that the old implementation copied global
imagined uncertainty and global TD error from each training cycle to every Episode in that cycle,
so those fields could not express learning pressure across tasks. Each Episode now contributes at
most 4 uniform, non-overlapping windows from its own trajectory, computing normalized
world-model-posterior state change and one-step TD error based on real environment reward.
Values, along with reward improvement and failure boundaries, are written to Episode records and
recovery snapshots. The curriculum schema was raised to
`hwr.task-agnostic-learning-sampling/v4`; old pseudo-distinguishing history will not be restored
by the new lineage.

Multi-camera geometry audit found a 9 cm baseline between the robot head RGB and depth lenses in
the model, while old high-resolution preprocessing fused by identical pixels and used the depth
camera pose to interpret head-RGB patches for wrist correspondence. Each frame now performs
depth-camera back-projection, robot-coordinate transformation, head-RGB reprojection, and
nearest-depth z-buffering with dynamic intrinsics/extrinsics; correspondence generation also
uses aligned head-RGB geometry. The preprocessing schema was raised to
`hwr.high-resolution-vision/v2`, and all visual caches from old DINOv2 runs are unusable in the
new lineage. Fixed-frame smoke tests in three real MuJoCo scenes measured a `0.090 m` camera
baseline in each; aligned valid-depth coverage was `27.43%～39.83%`, retained valid depth within
the original range was `74.8%～80.6%`, and depth range was `0.150～3.590 m`.

Dependency audit of the DINOv3 adapter found that current Transformers provides only
`DINOv3ViTImageProcessorFast`, while old code explicitly requested `use_fast=False` and the
environment lacked the required torchvision; even with official weights, loading would fail.
Formal foundation dependencies now lock the matching Torch `2.13.x` / torchvision `0.28.x`
versions. The adapter explicitly uses Fast, and the development gate instantiates the processor
and records all three runtime-library versions before weight audit.

Randomness audit of the recovery path found that old runner snapshots saved only NumPy state;
the Torch random streams used by RSSM categorical sampling and imagined rollouts were absent
from checkpoints, and the autonomous-collection Actor ignored the Episode seed. The formal model
stack now initializes from the training-config seed before constructing parameters; the random
Actor uses an Episode-private device generator; and the recovery schema was raised to
`hwr.foundation-runner-recovery/v2`, atomically saving CPU and actual-training-device Torch RNG.
Fixed regression verified exact equality of the next random sequence after CPU snapshot restore,
and a real MPS smoke test verified bitwise restoration of the `cpu+mps` pair.

Completion audit of the action-causality gate found that although the v2 report declared five
components—visual latents, proprioception, reward, termination, and safety—it added errors and
checked only one total ratio; one strongly action-dependent head could mask heads that ignored
action completely. v3 now stores real/shuffled errors, ratios, and per-horizon sequences for
each component, and requires the total, every component, global aggregate, and all three task
partitions to pass. Deployment evaluation recomputes the assessment from raw sequences and
checks that total error equals the component sum; changing only `passed` or aggregate numbers
cannot pass. Old v2 causality reports do not enter the new lineage.

It was then found that safety labels and action conditioning were more fundamentally misaligned:
replay stored both Actor proposals and safety-layer-corrected executed actions, but the old
safety head saw only executed actions while trying to predict “whether the proposal was modified.”
The filtered action had already discarded the information causing intervention, making the target
unidentifiable in principle. The implementation now separates world dynamics and safety
prediction: RSSM receives only executed actions; the safety head receives current latent and Actor
proposal and predicts only the independent safety-layer intervention label, without mixing in
severe collision. Imagination RL penalizes the predicted intervention probability for Actor
proposals, while the actual safety filter remains independent. Counterfactual auditing treats each
proposal/executed-action pair as one unit for a global derangement, preserving safety-layer
pairing while breaking state-action causality. Trajectory, curriculum, action-causality, and
recovery schemas were raised to `hwr.autonomous-trajectory/v3`,
`hwr.task-agnostic-learning-sampling/v5`, `hwr.foundation-action-causality/v4`, and
`hwr.foundation-runner-recovery/v3`; old runs must not silently resume under this semantically
different lineage.

Execution is allowed only after the gate passes:

```bash
scripts/start_foundation_training_tmux.sh foundation-wm-004
```

The launcher runs formal training in an independent tmux session and uses the existing notification
wrapper to send the run, log, Episode count, checkpoint path, and SHA-256 with the Lark bot identity
after completion or abnormal exit; the current Codex session need not remain occupied during training.

The first launch of `foundation-wm-004` on 2026-08-13 exposed a local resource-gate gap:
frozen teacher features finished at 16:01, but the first 200-step update cycle still had not
written a checkpoint after more than thirty minutes. MPS/Metal unified-memory footprint reached
approximately 34 GiB, peaking at approximately 36 GiB, with system swap around 22 GiB and
noticeable desktop lag. The process was intentionally aborted after confirming no `latest.json`,
checkpoint, deployment, or per-Episode result; it has no recoverable learning parameters and
cannot serve as training or evaluation evidence. Small run manifest, feature index, replay
manifest, holdout manifest, and bounded logs were archived at
`artifacts/retired-foundation-runs/foundation-wm-004-memory-abort-audit-metadata.tar.gz`, with
SHA-256 `89ee8796784479b50d7c5ccdde5d83a0711d0d6d5ec9a96d597ee2a5a425d67e`; approximately 13 GiB
of rebuildable caches and trajectories without checkpoints were subsequently deleted.

Subsequent formal runs use the MPS resource policy in Section 9.2 of the training-paradigm
document: a 0.65 hard watermark and 0.50 soft watermark of the recommended working set, idle
accelerator-cache reclamation every 10 optimization steps, and `nice 10`. This change handles
task-agnostic resource use only and does not change no-expert lineage, policy action sources,
task sampling, or success conditions.

`foundation-wm-005` verified that the watermarks took effect but also exposed the active-tensor
peak from visual backpropagation itself: after the first frozen-feature pass, the first vision
student backward had allocated 23.71 GiB of MPS memory plus approximately 0.55 GiB of driver
allocation; a request for another 117.14 MiB was rejected by the 24.34 GiB hard limit. The run
therefore exited abnormally, and the Lark bot notification was sent successfully. Inspection
confirmed that it still had no `latest.json`, checkpoint, deployment, action-causality report,
or per-Episode training result; it cannot resume and has no model eligible for evaluation.
Small audit materials were archived at
`artifacts/retired-foundation-runs/foundation-wm-005-mps-oom-audit-metadata.tar.gz`, with SHA-256
`320d8c8dfdd0218954731bb6339c7f93b0e702386372746b2d4ab9855fbecb1e`; 13 GiB of rebuildable
feature cache and trajectories without checkpoints are no longer retained.

The root cause was not the effective batch or 16-step world-model window itself, but that the
vision loader sent the four-frame, three-camera history of 34 observations in a batch through
ConvNeXt at once and retained backward activations for approximately 408 images. The fix adds
task-agnostic visual gradient accumulation inside the unified trainer: effective batch=2 and
sequence length=16 remain unchanged; each visual microbatch processes at most 4 observations,
with one visual optimizer step after all microbatches finish. A real MPS smoke test from the
original run's replay covers complete vision, world-model, and imagination-RL updates with 9
visual microbatches; driver allocation at microbatch reclamation stayed at 3.29–3.31 GiB, and
the complete train step ended at 3.47 GiB with `trainer/update_count=1`, no longer touching the
hard watermark. This change does not recognize tasks or objects and does not alter no-expert
action lineage.

The fixed evaluation command after training completes is:

```bash
hwr-evaluate-foundation-world-model \
  runs/foundation-world-model/foundation-wm-004 \
  --seed-count 40 --video-seed-count 1
```

## `pilot-074` Check Conclusion

- Parent run: `pilot-073`; source commit: `cd57dbd77a396d6d6bd2090a905dc8240b8d3ac6`.
- Training reached 500 Episodes; `pilot-074` added 70; replay contained 12,000 entries, with 153,314 cumulative updates, and all three task success counts remained 0.
- New Episode allocation was tray 20, storage basket 35, and drawer 15. By the end, the best bilateral worst-reach distances for tray and basket improved to approximately 7.3 cm and 8.5 cm, but neither achieved simultaneous contact; the drawer scene regressed from bilateral contact in the previous stage to no contact in all 15 Episodes.
- Deterministic Actor action mean absolute inconsistency after visual/proprioceptive lateral reflection, normalized by action range, was 14.45% for the tray and 9.62% for the basket. Consistency training was effective but insufficient to eliminate single-arm collapse.
- Training artifacts, checkpoint, and model hashes were complete; the completion notification was sent by the Lark bot identity. This run proves only that the training loop completed, not task success.

## Root Cause

`pilot-074` exposed a generic curriculum-algorithm defect, not missing scene-specific action logic:

1. Every unsuccessful Episode, including trajectories that normally reached the time limit, was marked as a failure boundary;
2. The Episode's total-return improvement over history was copied to every state in that Episode;
3. Remote-from-task states at the ends of long trajectories had high novelty and TD error, inherited the improvement for the whole Episode and the timeout boundary, and consequently filled the frontier;
4. Stable task-ID sorting created artificial ranks on ties, and task-sampling probability temporarily concentrated at approximately 74.8% for the storage basket.

The highest-ranked tray, basket, and drawer frontier states in the checkpoint drifted to bilateral worst-reach distances of approximately 1.08 m, 1.46 m, and 1.67 m. This explains why return numbers improved while physical task progress degraded.

## Next-Stage Adjustments

- Frontier schema upgrade: discard the old 48 current candidates and historical migration count while retaining ordinary replay, Actor, Critic, and 500 Episodes of history; cumulative discarded-candidate count is 71.
- Change per-state reward improvement to local improvement speed relative to the within-Episode exponential baseline; no longer copy total Episode return.
- Accept only environment-declared failure termination as a failure boundary; timeouts and truncations produce no boundary.
- Use tied-percentile ranking for the current Episode and retain diverse candidates by four signal types; frontier capacity retains every signal signature that has appeared.
- Task-sampling schema upgrade: cumulative discard after old-history migration is 72; restart the three tasks at strict equal probability, with default temperature `0.75` and per-task probability cap `0.55`.
- Lower frontier-reset mixing probability in the next run, strengthen simulation-declared lateral-reflection consistency, and use task-agnostic coupled bimanual-reflection exploration. Policy actions still come only from random exploration or the current RL Actor.

## `pilot-075` Check Conclusion

- Parent run: `pilot-074`; source commit: `68a9889f7ee660ae18ccc7721bc12a6c07b13ec8`.
- Training reached 570 Episodes; `pilot-075` added 70; cumulative updates were 176,341, all three task success counts remained 0, and the new Episode allocation was tray 22, storage basket 20, and drawer 28.
- The tray had 7 left-contact Episodes, 6 right-contact Episodes, and 1 simultaneous-contact Episode (8 steps); the basket had left contact 5, right contact 14, and simultaneous contact 0; the drawer had left contact 2, right contact 6, and simultaneous contact 0. In the final 35 Episodes, the tray had only 2 left-contact and 2 right-contact Episodes, while the basket and drawer had no left contact, indicating a later collapse back toward a single-arm policy.
- Tray lateral-reflection inconsistency fell from 14.45% in `pilot-074` to 5.57%, but basket inconsistency rose from 9.62% to 11.11%. The old consistency loss used the EMA Actor as its target and could still lock in old-policy bias.
- None of 19 frontier resets produced left-arm contact; full reset produced left-contact Episodes of 7/16, 5/16, and 2/19 for tray, basket, and drawer. Median bilateral worst-reach distances inside the frontier remained approximately 0.79 m, 0.70 m, and 0.43 m, so frontier reset contributed negatively to exploration in this stage.
- P075 replay had only 4,000 primary entries per task. One 1,200-step Episode with legal lateral reflection wrote raw, hindsight, reflected-raw, and reflected-hindsight data in sequence, totaling 4,800 rows; a single Episode could overwrite earlier rare contacts. The final tray primary replay had no contact transitions, and novelty/reward-improvement strata did not preserve tray contact.
- Training artifacts and hashes were complete; the completion notification was sent by the Lark bot identity. This run proves only that the RL loop continued, not household-task acceptance.

## `pilot-076` Adjustments

- Replay storage semantics changed to “one physical autonomous transition occupies one primary capacity slot.” Hindsight goal relabeling no longer runs, and environment-transformation copies are not stored in advance; the environment declares legal transformations, and the algorithm applies generic augmentation at a fixed probability during sampling.
- Symmetry consistency changed to direct group equivariance of the same Actor, `Actor(T(o)) = T(Actor(o))`, with both original and transformed observations backpropagated; the EMA Actor is no longer used as an action target.
- The old P075 format could not reliably distinguish raw transitions from pre-reflected copies. P076 explicitly discarded old replay and inherited only the Actor, Critic, optimizer, 570 historical Episodes, and task-agnostic sampling history; lineage records per-task discard counts and does not present synthetic copies as autonomous experience.
- Primary replay capacity increased to 24,000, or 8,000 autonomous transitions per task; frontier-reset probability was set to `0`, so candidate states were no longer used as starting points until their value was proven.
- Task-sampling temperature changed from `0.75` to `1.0`, and the per-task probability cap tightened from `0.55` to `0.45`. Under the new parameters, P075 history corresponded to approximately 25.1% tray, 29.9% basket, and 45.0% drawer, preventing the drawer from occupying more than half of the collection budget long term.
- The environment-interface boundary was unchanged: a new scene may provide only observations, actions, rewards, termination, and legal environment transformations; training adds no scene branch and generates no action, goal, reward, or task stage.

## `pilot-076` Check Conclusion

- Source commit: `a330a90c713c596d119cad1788cae24bbff51332`. Training reached 640 Episodes, added 70, and added 21,726 updates; all three task success counts remained 0. Checkpoint SHA-256 was `1146c83633122cb8250dc955565072a0d772b27024fed10706a6985279040463`, and Actor SHA-256 was `21dd08c3942aea6c9c022e7e166aca778c1729431e26c1f18c8494b3308b2140`.
- New Episode allocation was tray 22, basket 25, and drawer 23. The basket produced 19 left-contact Episodes, 15 right-contact Episodes, and 11 simultaneous-contact Episodes, totaling 413 simultaneous-contact steps and producing the first 1.48 cm of controlled target-transport progress. P075 had zero basket simultaneous contact, so autonomous replay and full reset recovered part of the bimanual learning signal.
- The tray produced 7 left-contact and 8 right-contact Episodes but only 1 simultaneous-contact Episode lasting 1 step; in the latter 11 tray Episodes, 4 had left contact and none had right contact. The drawer produced 4 left-contact and 16 right-contact Episodes with no simultaneous contact, and controlled drawer opening remained near 0. Single-arm collapse did not disappear; only the basket scene achieved a local breakthrough.
- Of 70 Episodes, 61 normally reached the time limit and 9 terminated from severe collisions; 211 safety interventions were recorded. Four basket Episodes and 194 simultaneous-contact steps remained in the latter half, so progress was not confined to the beginning of training.
- Final ordinary replay had 8,000 entries per task, all Actor-autonomous transitions, with zero rows having `actor_weight=0`. Basket ordinary replay retained 96 simultaneous-contact transitions and the reward-improvement stratum retained 116; recent left-biased tray trajectories removed right contact from ordinary replay, but novelty/reward-improvement strata retained 130/116 right-contact transitions. Stratified capacity solved the P075 issue where rare experience was immediately overwritten by synthetic copies.
- In the same 10-initial-state diagnostic, P076 further reduced lateral-reflection action inconsistency relative to P075: tray from approximately 22.4% to 13.9%, basket from approximately 19.9% to 10.0%. This diagnostic convention differs from early single-Episode records and is only for comparison among checkpoints in the same group.
- The completion notification was successfully sent by the Lark bot identity, with message ID `om_x100b689be4373ca0c3673bcdea2e094`. P076 proves that the policy can autonomously discover longer bimanual contact, but has not converted stable contact into transport or drawer operation and cannot enter formal success-rate acceptance.

## `pilot-077` Adjustments

- Inherit P076's Actor, Critic, optimizer, and compatible autonomous replay without clearing verified bimanual-contact transitions.
- Increase primary replay capacity from 24,000 to 36,000, or 12,000 per task, extending the ordinary stratum from approximately 6 full Episodes to approximately 10; novelty, reward-improvement, and safety sub-strata also increase from 1,000 to 1,500 per task.
- All 70 P076 Episodes failed, so failed and ordinary replay currently share the same distribution; lower the failure quota from `0.25` to `0.10`. The safety stratum has only 220 entries but was repeatedly sampled during 21,726 new updates, so lower the safety quota from `0.20` to `0.10`.
- Give the released batch budget only to task-agnostic metrics: raise state novelty from `0.25` to `0.30` and environment reward improvement from `0.20` to `0.30`. Ordinary samples remain 20%, while failure, discovery, improvement, and safety total 80%. These ratios read no object, contact type, distance, or task stage.
- Keep frontier reset at `0`; keep task-sampling temperature `1.0`, per-task cap `0.45`, and same-Actor equivariance weight `0.5` unchanged to isolate replay time span and generic prioritized sampling.
- P077 targets 710 cumulative Episodes, adding 70; the key observation is whether basket controlled transport exceeds P076's 1.48 cm and whether bimanual concurrency returns for tray/drawer. These observations are for stage diagnostics only and do not affect action generation or training branches.

## `pilot-077` Check Conclusion

- Source commit: `603655b5617cad95ae52665a15b30939bc44adcc`. Training reached 710 Episodes, added 70, and reached 219,289 cumulative updates with 21,222 new updates; all three task success counts remained 0. Checkpoint SHA-256 was `06129f8f8e2068feb2910424c58331a5e0e7f634946ad0361780902d15045295`, and Actor SHA-256 was `7f7d8a40250b3a068bcb330f92971f82ed9e9eb7a015812b86de6686435f0874`.
- New Episode allocation was tray 25, basket 22, and drawer 23. One tray Episode formed 260 steps of continuous bimanual contact and produced 0.108 cm controlled-transport progress—the first long tray bimanual contact—but it mostly stayed in place. Basket simultaneous contact regressed from P076's 11 Episodes/413 steps to 6 Episodes/81 steps, with maximum controlled progress falling from 1.48 cm to 0.058 cm; only 1 Episode/7 simultaneous-contact steps remained in the latter half. Drawer left contact rose from 4/23 to 16/23 Episodes, but controlled opening was only approximately `3e-7 m`, not effective pulling.
- Across all new Episodes, 8 had simultaneous contact for 350 steps, below P076's 12 and 414 steps. Severe collisions remained at 9; safety interventions fell from 211 to 131. P077 therefore shows that larger replay can preserve long tray contact and reduce interventions, but did not stably preserve the basket breakthrough or convert contact into significant task progress.
- Primary replay reached 12,000 entries per task. Inspection of derived strata found that “reward improvement” in code and manifest was actually sorted by absolute environment reward: each Episode selected high-reward transitions without checking whether reward continued to rise relative to its own history. Of the final 1,500 entries per task in this stratum, tray and basket had only 5 and 35 positive controlled-target-displacement transitions. A high-reward contact plateau can be reused repeatedly, but it provides no generic signal for leaving the plateau.
- The completion notification was successfully sent by the Lark bot identity, with message ID `om_x100b688551d9c8a8dd26498f6887ea4`. P077 remains only an autonomous-training diagnostic and cannot enter formal success-rate acceptance.

## `pilot-078` Adjustments

- Reward-prioritized replay now uses true within-Episode local reward-improvement speed: positive stepwise environment-reward increases relative to the exponential historical baseline are ranked, while zero- or negative-improvement transitions do not enter the derived stratum. The algorithm reads only the environment-reward sequence, not task-ID semantics, objects, distance, contact, targets, or action answers.
- The derived-priority schema was raised to `hwr.task-agnostic-reward-improvement-speed/v4`. P077's 4,500 old absolute-reward indices were audited as obsolete and 4,500 local-improvement indices were rebuilt from 36,000 time-ordered autonomous primary-replay entries; primary replay, failure/novelty/safety strata, Actor, Critic, optimizer, 710 historical records, and random state were retained. Lineage records discarded, rebuilt, and retained counts per task.
- Replay capacity remains 36,000. All historical Episodes still failed and failed/ordinary strata share the same distribution, so failure quota falls from `0.10` to `0.05`; state novelty from `0.30` to `0.25`; local reward-improvement speed rises from `0.30` to `0.40`; safety remains `0.10`, and ordinary samples remain `0.20`. These ratios still depend only on task-agnostic signals.
- Use recovery probability `0.005` and a 6-step random burst across the full action space. The burst reads no observation, task, target, contact, or reward; grippers hold the Actor's current output and sample only short lateral-reflection-coupled motion within common action bounds to expand real autonomous experience around the Actor's local neighborhood. Other noise, lateral reflection, `frontier reset=0`, task sampling, and consistency weight remain unchanged.
- P078 targets 780 cumulative Episodes, adding 70. Stage judgment still observes success, simultaneous contact, controlled progress, and safety outcomes across the three tasks, but these physical quantities are offline diagnostics only and do not enter replay classification, curriculum, or action generation.

## `pilot-078` Check Conclusion

- Source commit: `52b30637ecfd6285a78aa29466119a4f77e9d14e`. Training reached 780 Episodes, added 70, and reached 241,068 cumulative updates with 21,779 new updates; all three task success counts remained 0. Checkpoint SHA-256 was `898ec4fca0fcfeafdf60984bcfbaf4518f0a01c08539bbaa278982cd49408bbe`, and Actor SHA-256 was `e6e9ef4ab3cf9d3c612afd357942699c738de0d2d78afb593fa3495310035d68`.
- New Episode allocation was tray 30, basket 17, and drawer 23. None of the three tasks produced simultaneous bimanual contact or controlled target transport; maximum controlled drawer opening remained approximately `3.85e-7 m`. P077 had 8 simultaneous-contact Episodes and 350 steps under the same convention, so P078 clearly regressed.
- Basket right-arm contact fell from P077's 13/22 Episodes to 1/17, lasting only 4 steps; all 9 basket Episodes in the latter half had no right-arm contact. Tray best bilateral worst-reach distance regressed from 2.09 cm to 5.81 cm. The drawer had left contact in 20/23 and right contact in 12/23, but never simultaneous contact and still formed no cooperative operation.
- The 70 Episodes still had 9 severe collisions, while safety interventions rose from P077's 131 to 229, including 158 for the tray. The new full-action random burst produced no verifiable progress and increased intervention burden, so it was disabled next stage.
- The reward-improvement stratum exposed two generic implementation defects. First, code ranked only within each Episode and still overwrote across Episodes by FIFO; the final basket improvement stratum had only 14 left-contact and 0 right-contact entries. Second, old primary replay stored 8-step return targets rather than raw stepwise rewards, but P078 used it to rebuild stepwise improvement speed, so signal units differed before and after migration. This stratum cannot remain a reliable training input.
- Artifact validation passed, with completion-notification message ID `om_x100b68800678aca0b2948da7cf80adc`. P078 is retained as a failed branch for audit, but its Actor, Critic, and optimizer do not enter the next stage.

## `pilot-079` Adjustments

- Fork from the physically best-performing `pilot-076` rather than inheriting P078's collapsed parameters. Inherit P076's Actor, Critic, optimizer, 24,000 autonomous primary-replay entries, state-novelty/safety strata, and 640 historical records; introduce no experts, demonstrations, action labels, or scene action logic.
- Raise the improvement-stratum schema to `hwr.task-agnostic-reward-improvement-speed/v5`. Each Episode still computes local improvement speed only from raw environment reward, but the stratum retains the globally highest bounded Top-K across all later Episodes and writes aligned scores to the checkpoint, so low-scoring new samples no longer FIFO-overwrite historical high-scoring samples.
- Audit-discard all 2,448 old P076 improvement-stratum entries without rebuilding them from primary replay, because primary replay's n-step targets cannot recover raw stepwise rewards. Retain the remaining autonomous replay; populate the new improvement stratum only from raw rewards actually executed by P079.
- Expand replay capacity to 36,000. Lower failure quota from P076's `0.25` to `0.05`, keep state novelty at `0.25`, use `0.25` for global improvement Top-K, lower safety from `0.20` to `0.10`, and raise ordinary autonomous replay to `0.35`. Restore task-agnostic random burst to `0`; keep frontier reset at `0`, with all other visual preprocessing, reflection augmentation, exploration noise, and task sampling at P076 settings.
- P079 targets 710 cumulative Episodes from 640, adding 70 independent training Episodes. Stage judgment focuses on preserving P076 basket bimanual concurrency and controlled transport without increasing tray safety interventions; these metrics are offline checks only and do not define training actions or priority.

## `pilot-079` Check Conclusion

- Source commit: `db3ef1510d03604a72665d1f621895a9db6c8a05`. Training reached 710 Episodes, added 70, and reached 218,869 cumulative updates with 20,802 new updates; all three task success counts remained 0. Checkpoint SHA-256 was `fd6e09887136d5780ae503b36b3074b7723c06124639dbce1227b1f7dc261f5e`, and Actor SHA-256 was `6dcff3f8b251054ab56b3ea531ce94d6ab3bdd0e76a7a4e2fea210c955e6d8d8`.
- New Episode allocation was tray 25, basket 19, and drawer 26. The basket produced 7 simultaneous-contact Episodes and 245 steps; among the latter 9 basket Episodes, 5 had simultaneous contact for 193 steps, showing that inherited P076 bimanual ability did not disappear later. Best bilateral worst-reach distance improved from P076's 3.53 cm to 1.86 cm.
- Yet none of the three tasks produced controlled target transport, the tray still had no simultaneous contact, and maximum controlled drawer opening was only approximately `2.41e-7 m`. Global reward-improvement Top-K correctly retained 1,715 real local-improvement transitions across Episodes but no basket simultaneous-contact transitions, showing that local reward increase and the key post-contact consequence are not the same signal.
- Severe collisions rose from P076's 9 to 14, and safety interventions rose from 211 to 229. The 182 latter-half basket interventions were concentrated in one Episode; other latter-half basket trajectories maintained long contact, so a single intervention peak is not interpreted as overall capability loss. Safety stratification and independent runtime filtering remain.
- The current basket ordinary replay has 244 simultaneous-contact transitions with mean TD error `1.256`, while uniform ordinary samples have only `0.319`. The existing algorithm computes TD error but uses it only for task sampling and the disabled frontier reset; it has no TD-error replay quota. Bimanual states reached but not learned by P079 therefore did not receive update frequency proportional to their Bellman error.
- Artifact and unsupervised-lineage validation passed, with completion-notification message ID `om_x100b6881c18540a8b21982c190d2940`. P079 proves that global reward-improvement retention works, but reward improvement alone cannot cross the post-contact learning bottleneck.

## `pilot-080` Adjustments

- Add an independent task-agnostic TD-error Top-K replay with schema `hwr.task-agnostic-td-error/v1`. The algorithm uses only Bellman error computed by the current Actor-Critic on autonomous transitions; it reads no task semantics, object type, contact, distance, stage, or action answer, and a new scene needs no training branch. Recompute retained-row TD error after each Episode update so nonstationary priorities do not remain permanently locked to old Critic error.
- Fork from P079, retaining its Actor, Critic, optimizer, 36,000 autonomous primary-replay entries, state-novelty/reward-improvement/safety strata, and 710 historical records. On first load, recompute TD error from autonomous primary replay and the state-novelty stratum, retain the global top 1,500 per task, and write scores to the checkpoint; this signal can be correctly recomputed from the current model and n-step transitions.
- Dry-run audit produced 4,500 TD-error transitions. The basket stratum contained 209 simultaneous-contact transitions and 7 historical controlled-transport transitions; these physical fields were used only to verify migration offline and were not read during selection.
- All historical Episodes failed, and failed replay exactly duplicates ordinary replay, so failure quota falls from `0.05` to `0`. State novelty changes from `0.25` to `0.20`, reward improvement from `0.25` to `0.15`, TD error uses `0.25`, safety remains `0.10`, and ordinary autonomous samples occupy `0.30`. Total replay capacity, visual preprocessing, reflection augmentation, exploration noise, `frontier reset=0`, and task sampling remain at P079 settings.
- P080 targets 780 cumulative Episodes, adding 70. Stage judgment first asks whether basket simultaneous contact converts into controlled transport, then whether tray simultaneous contact and controlled drawer opening appear; these metrics remain offline diagnostics only.
