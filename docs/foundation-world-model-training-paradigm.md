# Foundation Model Perception, World Model, and Imagination Reinforcement Learning Paradigm

> Status: Current and only formal training architecture decision
>
> Decision date: 2026-08-12
>
> Scope: 3D simulation, local training, and future real-robot data and deployment

## 1. Decision

Stop extending the “small convolutional vision encoder + character-hash language encoder +
direct model-free Actor-Critic” path. Retain P076–P080 as a failed baseline and do not continue
training from those checkpoints, replay buffers, or vision weights.

The new mainline uses:

```text
Frozen vision/language foundation models -> High-resolution spatial representation -> Action-conditioned world model
                                                                                      -> RL in imagined trajectories
                                                                                      -> 16-dimensional joint Actor
                                                                                      -> Independent safety layer
                                                                                      -> MuJoCo / real robot
```

Foundation models only produce continuous representations from deployable observations; they
cannot generate actions, skills, object lists, or task plans. The world model learns only the
physical consequences of executed actions. All policy actions for the base, left and right arms,
and left and right grippers must still come only from random exploration or the current RL Actor.

The platform does not directly adopt the runtime abstractions of external VLAs, robot-training
frameworks, or large models. Third-party models may enter only through
`hwr.adapters.foundation`, with their weights, preprocessors, and licenses locked. The project
continues to define the core representations, trajectories, world model, policy, training,
evaluation, and hardware interfaces.

## 2. Single Development Gate

This rebuild does not proceed by “developing part, training part, then filling in the remaining
development.” Git commits may be organized by verifiable module, but all development shares one
overall gate:

```text
Foundation-model adapters ─┐
Vision pipeline and caches ├─> development-ready overall acceptance ─> Sole formal training entry point
Visual student and fusion  ┤
World model                ┤
Imagined RL                ┤
Deployment and evaluation  ┘
```

Before the overall gate passes:

- Do not start policy training, formal world-model training, or formal vision-student training;
- Do not create P081 or similar runs for the purpose of showing training progress;
- Run only unit, integration, and smoke tests using synthetic tensors or tiny fixed fixtures;
- Smoke tests may verify only gradients, shapes, serialization, and closed-loop interfaces; their
  artifacts must not be registered as training checkpoints;
- Do not substitute local contact, loss reduction, or a short run for proof of development completion.

The overall gate must use one command to check code, dependencies, weight locks, data schema,
anti-cheating rules, deployment isolation, fixed fixtures, and Python size limits. If any item
fails, the formal training entry point must refuse to run. The unlock report must use the current
`hwr.foundation-development-ready/v3` schema and exactly include protected source, algorithmic
lineage, formal configuration, model selection, runtime dependencies, weights, architecture,
Python size, the full test suite, and real foundation-model inference evidence, while also
running training-semantics checks. Missing items, unknown items, an undeclared unlock, or a
mismatch between isolated-commit evidence and the current commit must all fail. The formal
training entry point copies the validated readiness report into the run unchanged and writes
its SHA-256 to the run manifest; resume and final evaluation revalidate the copy. The lineage
scan covers the entire new mainline—foundation adapters, perception, data, Actor, world model,
training, safety, deployment, and evaluation—not merely a few online-runner files.

## 3. Non-Negotiable Learning Boundaries

### 3.1 Allowed Inputs and Feedback

Deployment-side inputs to the Actor and world model include only:

- Head RGB-D, left- and right-wrist RGB, and synchronization, calibration, and validity information;
- Proprioceptive state and the most recently executed action;
- A frozen semantic vector for the raw natural-language command;
- Continuous visual, language, and temporal latents computed from the fields above.

The environment may define rewards, success, termination, safety outcomes, and legal environment
transformations. Training-time world-model prediction heads and the Critic may read these
outcomes. Simulation ground truth may be used for independent evaluation or a non-deployment
training Critic, but it must not enter the Actor, deployment world-model state, or foundation
model prompts.

Environment success outcomes must be tied to verifiable physical causality rather than merely
concatenated, unrelated terminal conditions. Each of the three current formal tasks requires
both manipulated objects to end inside their target volumes and remain stable for 2 seconds. A
successful Episode must accumulate real two-finger contact for both grippers and at least
0.5 seconds of simultaneous left/right arm contact; the kitchen task also requires the drawer to
be pulled to its minimum opening through physical contact. The robot may release objects after
placement, but leaving the target volume revokes the stability evidence. This state machine
checks only contacts, joint motion, and terminal state that have already occurred; it returns no
grasp point, path, action, skill, or task-stage information to the Actor.

Initial autonomous data may still come only from random RL, but random does not mean independent
jitter at every control cycle. Formal collection uses a stationary temporally correlated random
process that reads no observation, task, or object: the 14 motion dimensions follow shared
first-order correlated noise, while the two grippers independently flip at random and hold their
state between flips. The correlation coefficient and flip probability are global training
configuration and are written to each Episode's `action_process` lineage; they encode no
direction, pose, grasp timing, or scene step. This produces continuous, identifiable dynamics
and contact segments without any expert action answer.

### 3.2 Prohibited Items

Formal lineage prohibits:

- Expert, human, or teleoperation actions;
- Behavior cloning, DAgger, a teacher policy, or a teacher checkpoint;
- Grasp points, end-effector waypoints, base routes, or action searchers;
- Object tokens, target tokens, human-defined skill names, and task stages;
- Plans, actions, rewards, or pseudo-action labels generated by an LLM/VLM;
- Training branches added for trays, storage baskets, drawers, or other scenes;
- Actor, Critic, replay, or optimizer state inherited from P076–P080.

The historical privileged expert remains only as informal diagnostic code. Its “expert must
complete the scene” acceptance check is disabled and must not substitute for the development
gate, data collection, or training success. Auditing expert imports in training source remains a
hard failure condition.

Frozen vision/language models are not action teachers. They may return only continuous features,
and adapter interfaces must contain no `action`, `waypoint`, `skill`, `stage`, `object ID`, or
`target ID` fields in their types or tests.

## 4. Foundation Perception

### 4.1 Vision Teachers

By default, connect two complementary frozen vision teachers:

- Multilingual vision-language teacher: SigLIP2 Base, for instruction-related semantics and
  region-level vision-language alignment;
- Dense vision teacher: DINOv3 ViT-S/16, retaining the final-layer stride-16 patch grid for
  cross-view correspondence, local appearance, and spatial structure. It runs only during
  offline feature materialization and does not enter the control loop; it has fewer parameters
  than ConvNeXt-Tiny and is stronger on the official dense-correspondence benchmark.

DINOv3 uses a custom license updated by Meta on 2025-08-19, not Apache-2.0. The model lock
must include the SHA-256 of the official `LICENSE.md`; weights may be downloaded only by an
account that has accepted the terms on the official Hugging Face model page. Public mirrors
must not bypass the manual gate. If weights are missing, the account is unauthorized, or the
license-file hash differs, the overall development gate must remain locked.

The current Transformers DINOv3 ViT provides only a torchvision-based Fast image processor.
Formal dependencies lock the matching Torch `2.13.x` and torchvision `0.28.x` release family,
and the adapter must explicitly set `use_fast=True`. Before accessing weights, the development
gate instantiates the processor and records Torch, torchvision, and Transformers versions.
Missing torchvision, a mismatched release family, or fallback to a nonexistent Slow processor
must fail immediately.

The concrete model is selected through versioned configuration and not encoded in the core
interface. Adapters must run fully offline and record the model identifier, revision, file
SHA-256, license, input specification, output dimensions, and inference backend. Missing weights,
hash drift, or an unregistered license must fail the overall gate. The real-inference gate must
also reject constant or degenerate features: visual output must be the expected `14×14` dense
grid with unit-normalized valid patches, strict zeroing for invalid cameras, and response
variation across patches and inputs. The cosine similarity for language paraphrases must be at
least `0.01` higher than for different intents. The gate must fail on these conditions rather
than merely writing the numbers into a report.

Vision teachers must output at least a patch grid preserving 2D layout and a validity mask.
Keeping only a globally pooled vector is prohibited because bimanual contact requires local
spatial resolution. Teachers do not enter the 20 Hz control loop; before formal training they
produce caches offline or provide distillation supervision for the deployable vision student.

### 4.2 Language Teacher

The default language encoder is the frozen Qwen3-Embedding-0.6B. Each natural-language command
is computed once from normalized text and cached; runtime reads it by content hash. It outputs
only a continuous semantic vector and does not generate text, plans, or actions.

Each formal household task declares 4 training-command paraphrases and 3 non-overlapping
evaluation paraphrases. An Episode selects raw natural language from the corresponding set by
seed only; the algorithm does not read object names or task stages. Fixed unseen-seed
evaluation uses only the evaluation set, allowing robustness to instruction paraphrases within
the same task to be tested. This still does not prove generalization to new tasks, compositional
commands, or open vocabulary; those capabilities require separate acceptance on new task and
object distributions rather than an embedding-similarity comparison alone.

The old `FrozenNgramLanguageEncoder` is retained only for historical checkpoints and fast
interface regression. Formal `development-ready` checks must reject registering it as the
current deployment language encoder.

### 4.3 High-Resolution Vision Preprocessing

The unified pipeline performs:

1. Four-camera time synchronization, duplicate-frame and dropped-frame marking;
2. Intrinsic/extrinsic validation, undistortion, and RGB-D alignment;
3. Depth denoising, range clipping, validity masks, and coordinate transforms;
4. Deterministic conversion from teacher resolution to online-student resolution;
5. Independent normalization of RGB, depth, geometry, and quality information;
6. Versioned assembly of short-term history and cross-camera view-frustum information.

The teacher's default input resolution is no lower than `224 × 224`; the online student's
default input is no lower than `160 × 160`. Final values may be adjusted based on M5 Pro
latency measurements, but must not fall back to P080's toy `24 × 18` input.

RGB-D alignment must use per-frame dynamic calibration rather than assuming that the head RGB
and depth optical centers coincide. Depth pixels are first back-projected from the depth camera
into robot coordinates, then projected into head RGB, with the nearest-depth z-buffer handling
occlusion; pixels without a valid projection remain invalid. Only aligned depth may be fused
with head-RGB patches, and only head-RGB intrinsics and extrinsics may be used to establish
cross-camera wrist correspondence from that aligned image. This preprocessing semantic version
is `hwr.high-resolution-vision/v2`; old caches must be rebuilt.

### 4.4 Deployable Vision Student

The vision student is a project-owned model targeting 20M–40M parameters and includes:

- A multi-stage ConvNeXt or ViT image backbone;
- Independent depth and validity encoders;
- A multiscale feature pyramid retaining 2D position;
- Shared-weight left- and right-wrist encoders;
- Cross-camera attention based on calibration and camera identity;
- Short-term temporal fusion over at least four frames.

Training supervision may come only from action-label-free data: teacher-feature distillation,
cross-view geometric correspondence, temporal consistency, occlusion recovery, depth structure,
and valid-vision-augmentation consistency. Simulation segmentation may be used for independent
probe evaluation but must not enter deployment features or policy inputs.

The `pooled_state` actually consumed by deployment must not depend on a random, non-updated
fusion layer. Vision objectives therefore include a stop-gradient alignment loss from current
spatial features to the fused state; the backward path must pass through cross-camera fusion,
the temporal Transformer, temporal-position parameters, and output normalization.
`development-ready/v3` performs one real optimization step on an isolated commit and requires
nonzero gradients and actual parameter changes in all four parts; a finite overall loss alone
cannot unlock training.

## 5. Autonomous Trajectories and Feature Caches

The new sequence data schema atomically stores:

- Raw camera frames, calibration, timestamps, and validity masks;
- Proprioceptive state;
- The 16-dimensional Actor proposal and action actually executed by the safety layer; neither may
  overwrite or substitute for the other;
- Environment reward, termination, truncation, and the intervention label indicating whether the
  safety layer changed or rejected the proposal;
- The language content hash and a reference to the frozen-encoding cache;
- References to teacher-feature caches, model revision, and content hashes;
- Episode, task distribution, environment version, random seed, and code commit.

Caches are deletable, rebuildable derivatives and cannot replace raw observations. Training and
evaluation seeds, layouts, and language expressions must be isolated at the index layer. The
data loader reads temporally contiguous windows and must not randomly shuffle transitions and
present them as dynamics sequences.

## 6. Action-Conditioned World Model

The world model uses a project-owned recurrent state-space model. The recommended initial
structure is:

- deterministic recurrent state;
- categorical stochastic latent state;
- Encoders for multi-camera vision-student features, language features, and proprioceptive state;
- An action-conditioned transition that receives only actions actually executed by the safety layer;
- Prediction of visual latents, proprioception, rewards, and continue/terminal from future latent states;
- A separate training head that predicts whether the safety layer intervenes from the current latent and Actor proposal;
- A generic residual head that predicts the action actually executed by the safety layer from the current latent and Actor proposal;
- A separate head that predicts real severe collisions from the current latent and executed action.

Training objectives include latent reconstruction, free-bits regularization, future
proprioception prediction, reward distribution, termination, and safety-intervention
classification. The world model must not predict or infer a “correct action,” and must not
expose simulation object ground truth to the Actor.

The two causal questions must be kept distinct: physical dynamics are conditioned on the action
actually executed, while safety intervention is conditioned on the Actor proposal. The safety
head must not mix collision termination into its labels or receive an action already modified by
the safety layer. Severe collision remains defined by environment reward, termination, and the
final acceptance report. The safety filter itself is independent of the learned model; training
heads only estimate its intervention probability for a proposal and cannot replace the filter.
Imagined rollouts must advance the RSSM with the action actually executed as predicted by the
residual head. They must not assume that a proposal potentially rejected or clipped by the
safety layer changes the future unchanged. The Actor also bears the cost of action rewriting
magnitude.

The world model must pass a three-layer action-causality anti-cheating evaluation. Audit data
is generated by an independent fixed-seed random RL collector and stored separately from
training Replay; no sample may be given to the optimizers of the vision student, world model,
Actor, or Critic. At startup, collect 8 128-transition system-identification Episodes per task,
cycling action correlation values `0.0/0.5/0.9/0.96`. Deterministically select 64
non-overlapping sequence windows per task, keep the observation sequence unchanged, treat each
step's “Actor proposal, executed action” as an indivisible pair, and perform a global
derangement permutation over all tasks and time positions. Compute multi-step open-loop errors
for future visual latents, proprioception, rewards, continue/terminal, and safety intervention
separately. Pairwise permutation breaks state-action correspondence while preserving the real
pairing between the proposal and safety-layer output and each member's own multiset; the
algorithm does not read scene objects or task semantics. The report must declare both action
sources and the pairing transformation, and retain each of the five components' real-action
error, shuffled-action error, ratio, and per-horizon sequence rather than only an added total.

The three diagnostic layers answer different questions:

1. The data-identifiability probe fits two task-agnostic ridge regressions on ordinary replay. It
   compares only “proprioceptive state” with “proprioceptive state + executed action” when
   predicting changes in joint velocity, gripper position, and base velocity 1/4/8/16 steps
   ahead, and bootstraps the error ratio at the weakest horizon by original Episode;
   it checks only whether collected data contains action effects and does not enter policy or
   world-model training.
2. The one-step action-utilization diagnostic fixes each real posterior state, replaces only the
   next action, and evaluates visual-latent and proprioception prediction; this isolates
   open-loop drift and tests whether the world model actually reads actions.
3. The multi-step open-loop diagnostic continues to retain five heads for vision, proprioception,
   reward, termination, and safety. Only vision and proprioception enter the action-causality
   gate; sparse outcome heads use their own calibration metrics and must not reuse one action
   shuffling ratio.

Each audit uses five independent derangements. The report stores the raw counterfactual result
for each one. Deployment evaluation recomputes every assessment, the 5th percentile of error
ratios, and all aggregates; the lower error-ratio bound must reach `1.05`, and every permutation
must pass the component and horizon conditions rather than relying on one lucky permutation.

The formal physical gate requires the combined shuffled-action vision/proprioception error to
be at least `1.05` times the real-action error, with at least `60%` of prediction horizons
degrading individually. The conditions apply to vision, proprioception, the global aggregate,
and every partition split by generic `task_id`. Final evaluation recomputes every assessment
from raw per-horizon values and cannot trust a prefilled `passed` field in the report. Without
degradation, the model has learned only video continuity and cannot support policy imagination.

Actor admission is split into two levels and must not switch automatically after “some number
of Episodes” have been collected. The exploration Actor unlocks only after ordinary replay has
at least 12 Episodes and one-step visual/proprioceptive action utilization, per-task data
probes, actual 16-dimensional action coverage, and covariance effective rank pass twice
consecutively. External contact, controlled motion, and collisions do not participate in this
level; otherwise random policy would have to produce evidence that the exploration Actor is
supposed to acquire, creating a circular dependency. The task Actor must additionally pass
multi-step physical causality, per-task contact and controlled motion, severe-collision positive
and negative Episode coverage in replay, and collision-head validation on an independent holdout.
Collision-calibration data is collected only after the exploration Actor unlocks: 8 positive and
8 negative Episodes per task, retaining only the final 16 transitions of each. Validation also
requires endpoint recall, PR-AUC, per-transition Brier, false-positive rate, temporal alignment,
and action-shuffling sensitivity to pass. Any subsequent audit failure immediately revokes the
corresponding admission. Before exploration admission, update only the vision student and world
model; do not export deployment while the task Actor lacks a qualified update. Sparse reward,
termination, and collision heads remain gates for the task Actor and final deployment, but must
not block the generic exploration Actor.

The exploration Actor must not perform imagination optimization on an unvalidated safety-action
residual model. During the startup holdout phase, collect 8 safety-intervention positives and
8 non-intervention negatives per task. Clip residual-model outputs dimension by dimension under
the formal 16-dimensional action contract, then use independent data to check intervention
recall, PR-AUC, Brier, normalized RMSE on intervention samples, identity-mapping RMSE on
non-intervention samples, and out-of-bounds rate. The out-of-bounds rate must be exactly 0;
while this gate fails, both the exploration Actor and task Actor remain locked.

Every training checkpoint must generate an immutable action-causality report binding the source
commit, update count, training replay manifest SHA-256, independent audit-data manifest
SHA-256, and exact window list. The report SHA-256 is also written to the training checkpoint.
A failed update may save training state for diagnosis but must not export a deployment model.
Only the same update with a passing report may export a deployment model; the deployment
manifest, `latest.json`, and final evaluation cross-validate that report and hash. Final
evaluation must recompute hashes for training replay, the holdout set, training checkpoint, and
deployment artifacts, and verify that source commit, update number, task partitions, and windows
do not overlap. It cannot trust only the report's `passed` field. Any missing, failed, or
tampered item makes training/evaluation fail.

Each resumable checkpoint must also atomically save the NumPy runner RNG, the Torch RNG for
the training device, the task-agnostic sampler, per-Episode records, the training replay
manifest, and the causality-holdout manifest. Fresh training must initialize Torch with the
formal-config seed before initializing any project-owned model. The autonomous-collection
Actor uses a private device generator initialized from the Episode seed and must not consume
randomness from the RSSM/imagination-training streams. Resume must restore CPU and actual
training-device Torch state after loading models and optimizers. Before publishing a new
`latest.json`, rolling replay may move evicted shards only into a recovery staging area inside
the run and must not delete them directly. Crash recovery first rolls back to the manifest
bound to `latest.json`, restores shards still referenced by the old checkpoint, removes shards
added by an incomplete cycle, and records the recovery event before collection continues. This
prevents reuse of Episode numbers or training seeds and prevents checkpoints from pointing to
data already removed by capacity trimming.

The unified runner's initial-state curriculum reuses the same task-agnostic frontier. Random
cold start only accumulates physical snapshots autonomously visited by the policy and cannot
start from the frontier. Candidate-state restoration becomes allowed at a fixed probability
only after the physical action-identifiability gate passes. Candidate ranking uses only posterior
latent cosine novelty relative to the candidate pool, one-step TD error from real transitions,
per-Episode local reward-improvement speed, and environment-declared terminal failure boundaries.
States after termination or truncation and states with safety intervention do not enter the
candidate pool. State restoration must reproduce the backend fingerprint, generalized
position/velocity/acceleration, actuator control, solver state, and runtime state item by item;
any mismatch returns to the original reset. Snapshots exist only in the runner's curriculum
restoration state, do not enter replay or model inputs, and frontier reset is always disabled for
hidden evaluation.

## 7. Imagination-Space Reinforcement Learning

Optimize the Actor/Critic over latent trajectories generated by the world model:

- The Actor jointly outputs a 16-dimensional action distribution for the base, left and right arms, and left and right grippers;
- Compute the Actor's transformed sample entropy within the `tanh` motion and `sigmoid` gripper ranges; untransformed Gaussian entropy must not stand in for actual action entropy. Random initialization must let both grippers cover near-fully-open and near-fully-closed actions, but must not inject grasp poses, timing, or scene actions;
- The Critic estimates environment return, while the world-model safety head estimates the intervention probability for the current proposal; neither outputs actions;
- The default imagined rollout covers 15–30 latent time steps;
- Use λ-return, entropy regularization, a target value network, and gradient clipping;
- Exploration signals come only from world-model uncertainty, state novelty, TD error, and reward-improvement speed;
- Generic augmentation applies legal transformations declared by simulation; the algorithm does not know task objects.

The Actor may evaluate latent states and sample only its own actions. CEM, MPC, LLM, VLM, or any
searcher is prohibited from selecting actions on the Actor's behalf during deployment; otherwise
the actions are no longer learned by the RL policy.

State novelty and TD error used to allocate subsequent collection budget must be computed from
each Episode's own trajectory; global averages from one training cycle must not be copied to all
tasks in that cycle. The formal implementation selects at most 4 uniformly spaced, non-overlapping
windows from each Episode: state novelty is the scale-invariant cosine change between adjacent
world-model posterior states, and TD error is the difference between the current Value and the
one-step Bellman target formed from real environment reward plus the next posterior slow Value.
Both affect only semantic-free sampling order across tasks, do not enter the Actor input, and do
not produce actions.

If temporal hierarchy is needed, learn only an unnamed continuous latent intent. Train both high
and low levels with the same generic RL objective; do not hard-code the latent as human-defined
skills such as “approach, grasp, transport, or place.”

Online fine-tuning reuses the same algorithm and runtime contract: executed transitions update
world dynamics, proposals and their intervention labels update the safety head, the world model
generates imagined trajectories, and the updated Actor/Critic then interacts with the environment
again. Safety filtering remains independent; replay always records both the Actor proposal and
executed action, and only the executed action is used as physical causality.

## 8. Deployment and Local Budget

The current development machine is an Apple M5 Pro with 48 GB unified memory. The design follows:

- Run foundation teachers serially offline and release memory after materializing features to disk;
- Cache language embeddings by command instead of running the 0.6B model repeatedly in the control loop;
- Support MPS, mixed precision, and CPU fallback for the vision student, world model, and Actor;
- Load only the vision student, language-cache reader, world-model posterior, Actor, and safety layer in the control process;
- Record peak memory, throughput, disk budget, and device backend in the training manifest;
- Allow an external SSD to store raw frames and derived caches, but atomically write and verify checkpoints and manifests.

The formal sequence batch on the 48 GB unified-memory machine is fixed at `2`. A sequence has
17 time points, each with two history frames and three RGB teacher views, while teacher targets
and student backpropagation coexist in memory. `batch=4` expands one update to 816 RGB images
and corresponding dense grids, leaving no reliable margin. Lowering the batch does not change
the model, tasks, replay, update count, or acceptance scope; it only reduces the number of
parallel sequences in one update.

The deployment checkpoint must contain none of the teacher models, Critic, rewarder, simulation
ground-truth reader, augmentationer, or training application. It may restore only deployable
perception, the world-model state updater, the deterministic Actor, and versioned preprocessing
configuration.

## 9. Definition of Development Completion

Before the formal training entry point is unlocked, all of the following implementations must
exist and pass automated checks:

- Frozen foundation-model interfaces, offline weight locks, license and hash audits;
- High-resolution multi-camera preprocessing, language caches, and vision-feature caches;
- Vision student, cross-camera/temporal fusion, and all action-label-free losses;
- Sequence-trajectory schema, atomic storage, split isolation, and loader;
- RSSM, all prediction heads, sequence losses, checkpointing, and reload;
- Action-shuffling counterfactuals, multi-step rollouts, and uncertainty evaluation;
- Imagined environment, Actor/Critic, λ-return, optimizer, and online data loop;
- Generic legal transformations, task-agnostic exploration, and curriculum interfaces;
- Independent safety filtering, executed-action replay, and Safety Critic;
- Deployment export, privileged-field audit, unseen-seed evaluation, and four-view unedited video;
- One unified training entry that assembles the three formal tasks without task-specific branches;
- All tests, architecture checks, and Python file/function size checks in the immutable snapshot of the current commit; unrelated historical experiments in the workspace do not enter gate evidence.

The overall gate must also scan training source and manifests to prove there are no expert data,
action labels, teacher actions, task-name branches, object/target tokens, or inherited old
P-series checkpoints. Formal training commands may run only after the overall gate generates
`development-ready.json` with the code commit and configuration hash.

The formal run and training checkpoint must share the same exact no-expert lineage: randomly
initialized project-owned models; only the three action sources `random_rl_exploration`,
`intrinsic_rl_actor`, and `rl_actor`; an empty expert/demonstration set; behavior cloning,
teacher actions, and action search disabled; and no old P-series parent checkpoint. Save, resume,
and final evaluation must compare the complete structure rather than only `source_commit`, and
must not treat “field exists” as “field is empty.” The formal run manifest schema is
`hwr.foundation-online-run/v4`; old v1/v2/v3 do not enter the new lineage. v4 also freezes the
effective training device, foundation-teacher device, Python path, process nice value, and MPS
watermarks.

### 9.1 Local Storage Limit

Dense foundation-vision features must not grow without bound with the number of Episodes. The
formal runner uses task-agnostic bounded replay: total transition capacity is declared in
configuration and allocated fairly by task ID. Each partition evicts only its oldest complete
Episode without inspecting scene objects, distance, or action content. When a raw shard is
evicted, its rebuildable vision-feature cache is deleted as well.

Checkpoint and deployment exports also use a fixed retention count, deleting only old
format-valid `update-*` directories while always retaining the newest version and its hash
manifest. Each autonomous Episode retains at most 7 contiguous, non-overlapping
16-transition dynamics windows: first select windows with significant contact, controlled
motion, collision, and action changes from data actually retained, then fill boundary and
uniform coverage. Interaction admission recognizes only evidence recomputed from these
retained transitions and cannot read discarded Episode summaries. A formal 120-Episode run can
therefore retain 13,440 training transitions, with only 2 windows per Episode generating
expensive DINOv3/SigLIP2 teacher caches while other windows still train the world model.

The independent holdout includes 8×128 transitions of system-identification data per task,
16×16 transitions of collision-validation data, and 16×16 transitions of safety-action
execution validation data; none generates teacher caches. Formal configuration retains 18,000
transitions and the most recent 3 sets of training/deployment artifacts. The current uncompressed
static estimate is approximately `28.41 GiB`, with a configuration limit of `30 GiB`; startup
also requires at least `35 GiB` free space. The capacity policy manages storage only and produces
no actions or curriculum answers.

### 9.2 Local Unified-Memory Limit

Apple Silicon MPS memory is shared with system memory. PyTorch's default unified-memory low
watermark is 1.4 times the device recommended working set; on the 48 GiB machine, the
recommended MPS working set is approximately 37.44 GiB, so the default low watermark exceeds
physical memory and cannot guarantee memory remains available for the desktop. The formal tmux
launcher passes the following task-agnostic resource policy:

- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65`, giving an MPS allocation hard limit of approximately 24.34 GiB;
- `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.50`, enabling allocator reclamation and adaptive commit at approximately 18.72 GiB;
- Run the training process with `nice 10` so interactive programs receive CPU priority;
- After unloading each frozen foundation model and every 10 optimization steps, release MPS/CUDA allocator cache no longer referenced by active tensors; write the final effective values to the immutable run manifest rather than leaving them only in the launch shell;
- Keep the vision student's effective batch and full 16-step world-model window, but accumulate gradients over at most 4 observations; activations for four-frame, three-camera ConvNeXt inputs from adjacent observations no longer reside in unified memory simultaneously. After the vision update, pass concatenated, stop-gradient continuous latents in their original order to the world model and imagination RL;
- Keep the world model updating every step and update the vision student at a fixed global 4-step interval. Non-vision updates generate latents only through gradient-free microbatches of at most 8 observations, build no geometric correspondences, and load no DINOv3/SigLIP teacher target. The interval reads only global update count, not task or environment feedback;
- Keep the marginal distribution of ordinary replay windows uniform, while reserving 25% of the batch for severe-collision termination windows; use class-weighted BCE for the severe-collision head so rare positives are not overwhelmed by normal transitions. A batch still samples without replacement preferentially across collision Episodes, allowing replacement only when independent Episodes are insufficient; ordinary batches still sample only from one Episode shard to avoid repeatedly decompressing large shards. Deduplicate frozen visual features by source hash within the batch and use at most 16 read-only LRU entries; cache capacity must not grow with replay.

These constraints reclaim caches and adjust resource scheduling only. They do not change
observations, actions, rewards, termination, legal environment transformations, sampling rules,
or action labels, and do not provide scene action answers. Vision microbatch losses are weighted
by observation count for each block, while one optimizer step still covers the full configured
batch. The launcher defaults may be overridden by the same-named PyTorch environment variables
and `HWR_FOUNDATION_NICE_LEVEL`; override values must be retained with the run log.

### 9.3 Training Observability

The appearance of `loss` only in process stdout does not constitute training evidence. The
unified runner must atomically publish immutable small JSON files for the current stage and each
complete cycle in `run/metrics/`, containing at least:

- All mean losses and pre-clipping gradient norms for the vision student, world model, Actor, and Value; low-frequency vision loss averaged only over actual vision updates, with the `trainer/visual_updated` ratio recorded separately;
- Update count, Episode count, duration of each stage, and action-causality gate results;
- Mean, standard deviation, range, saturation rate, and effective rank of the 16-dimensional executed action after normalization to a common scale;
- Difference between Actor proposal and safety-layer executed action, gripper switch rate, and safety intervention rate;
- Per-task Episode count, success count, and numeric physical outcomes provided by the environment, without allowing the trainer to add task branches based on them.

On interrupted resume, delete uncommitted cycle metrics later than the restored checkpoint; do
not leave rolled-back updates in the curves. The local read-only dashboard may poll only these
JSON files and bounded recent-Episode records; it must not scan replay, feature caches, or model
weights. The dashboard listens only on `127.0.0.1` by default and is an out-of-band observation
tool that does not participate in sampling, reward, or optimization.

The action-identifiability probe must fit independently per task and per 1/4/8/16-step horizon;
it must not mix tasks and misclassify task-identity differences as action causality. Confidence
intervals use Episode-clustered bootstrap; every holdout Episode must contribute the same number
of non-overlapping windows, and short trajectories use deterministic substitute-seed
resampling. Interaction coverage counts only Episodes long enough to form a complete training
window; short collision trajectories cannot enter the admission denominator. Random calibration
over 24 Episodes checks only action identifiability, action coverage, and one-step physical
causality, not contact or controlled motion; the latter two enter the task-Actor gate only after
the exploration Actor receives a collection budget.

Intrinsic-exploration novelty is the kNN distance relative to current imagination history and
replay-batch states, not adjacent-state change; rapidly swinging an arm back to a seen state
does not keep earning high reward. The world model separately predicts safety-layer rewriting
and real severe collisions; both costs enter the imagined returns of the exploration and task
Actors, while the safety filter remains independent of the policy.

The observation system is a prerequisite for later admission gates, not a visualization wrapper
for failed training. If gradients are non-finite, effective action dimensions collapse, proposals
are continually overridden by the safety layer, or action causality remains near `1`, the runner
must use this to block advancement to the next collection stage instead of stacking more updates.

## 10. Unified Training and Final Acceptance

After development is complete, start only one formal training mainline and sample from all
three task distributions at the same time. Do not train a single-scene policy first or add an
algorithm branch for any scene. Training may resume from checkpoints and commit resumable
checkpoints, but intermediate metrics must not trigger a switch back to a hand-written
curriculum or expert data.

 A newly unlocked exploration Actor or task Actor cannot become a collection source after one
gradient update. The formal configuration must run a dedicated warm-up of at least 200 and at
most 1,000 updates without updating the world model, aggregating one window every 50 updates.
The latest three windows must simultaneously satisfy finite and bounded Actor/Value gradients,
non-collapsed motion and gripper policy entropy, and relative imagined-return variation no
greater than `0.25`. Record the Actor as collectible only after reaching the minimum update
count and passing the stability gate. If it still fails at the maximum update count, terminate
the run and write the complete failure check into Actor readiness state.

Formal background startup always uses
`scripts/start_foundation_training_tmux.sh RUN_ID [--resume] [--seed SEED]`. This entry point
always invokes the sole foundation training application, run root, readiness, model directory,
and Lark bot completion notification, and rejects duplicate tmux sessions or unsafe run IDs; it
provides no gate-bypass parameter.

The following must ultimately hold:

- Run at least 3 distinct training seeds for each candidate configuration; a single seed may be used only for calibration or failure analysis and cannot independently support a formal capability conclusion;
- Report five physical benchmarks before formal household-scene evaluation: reaching, unilateral contact, stable bilateral contact, controlled rigid-body/joint motion, and the complete task; these levels provide diagnostic coverage and expose no task stages or action answers to the Actor;
- Evaluate each of the three formal tasks—living-room two-object storage, dining-table plate-and-cup placement, and kitchen two-bottle drawer storage—on at least 40 unseen random seeds; “unseen” must exclude every seed used by training Episodes and the action-causality holdout, even when an evaluation start seed is specified manually;
- Observed success rate must be at least 70% for every task, and the lower bound of the 95% Wilson interval must also be at least 70%;
- Severe collisions must be 0;
- The successful state must remain stable for at least 2 seconds;
- After separately locking the left or right arm, the same-task success rate and the upper bound of its 95% Wilson interval must both be below 10%;
- Evaluation runs only the reloaded deterministic Actor, with exploration and training writes disabled;
- The same evaluation process directly records unedited third-person, head, left-wrist, and right-wrist video;
- Data, models, code commit, configuration, per-Episode results, videos, and anti-cheating reports must be mutually traceable by hash.

Evaluation for a single training seed may write only `per_seed_passed`; its `formal_passed` and
compatibility field `passed` must remain false. A formal conclusion may be written only by an
independent aggregation entry point after binding at least three different training seeds,
three different run manifests, and deployment, verifying that immutable configuration is
identical except for seed, training/holdout seeds do not overlap across runs, all three
evaluations use the same unseen-seed set, and every per-seed result passes.

Every `hwr.foundation-evaluation-run/v3` manifest must directly hash readiness, run/latest,
training Episodes, training replay, the causality holdout, the action-causality report,
training checkpoints, deployment artifacts, per-Episode evaluation, acceptance results, and
each video stream; it must not reference training data or models only indirectly through
directory paths.

Foundation-model semantic retrieval, world-model loss, and imagined return are not evidence of
household-task success. The final evidence remains the bimanual physical outcome actually
executed by the RL Actor on isolated seeds.

## 11. Current Implementation Mapping

As of 2026-08-14, the design above maps to the following project-owned modules:

| Responsibility | Implementation |
|---|---|
| Foundation-model boundary and locks | `hwr.perception.foundation`, `hwr.adapters.foundation`, `configs/foundation/model-locks.json` |
| High resolution and dynamic calibration | `hwr.perception.high_resolution`, `FrameCameraCalibration` |
| Vision student and action-label-free objectives | `hwr.perception.student`, `student_objectives`, `geometric_correspondence` |
| Autonomous sequences, evidence pool, and caches | `hwr.data.autonomous_trajectory`, `foundation_sequence_reservoir`, `foundation_cache`, `foundation_features`, `foundation_loading` |
| Action-conditioned world model | `hwr.world_model` |
| Data identifiability and Actor admission | `hwr.train.foundation_action_probe`, `foundation_actor_readiness` |
| Imagination RL | `hwr.train.imagination`, `imagination_rl` |
| Environment-reward-free intrinsic exploration RL | `hwr.train.intrinsic_exploration`, `intrinsic_rl_actor` |
| Task-agnostic physical-state curriculum | `hwr.train.foundation_frontier`, `learning_frontier`, `foundation_learning_signals` |
| Environment-declared generic augmentation | `hwr.train.foundation_augmentation` |
| Single online closed loop | `hwr.train.foundation_online`, `foundation_trainer`, `foundation_holdout_orchestration`, `foundation_run_manifest` |
| Formal household environment | `hwr.adapters.mujoco.formal_household_backend`, `hwr.scenarios.formal3d`, `configs/tasks/formal_3d_v1.json` |
| Local resource budget | `hwr.train.foundation_resource_budget` |
| Metrics and local dashboard | `hwr.train.foundation_metrics`, `foundation_dashboard`, `hwr.apps.serve_foundation_dashboard` |
| Training/deployment checkpoint | `hwr.train.foundation_registry` |
| Stripped deployment runtime | `hwr.world_model.deploy`, `hwr.policy.foundation_runtime` |
| Overall gate | `scripts/verify_development_ready.py`, `scripts/verify_training_semantics.py`, `hwr.train.development_gate` |
| Fixed acceptance and video | `hwr.apps.evaluate_foundation_world_model`, `hwr.eval.bimanual` |

`hwr-train-foundation-world-model` is the sole formal training entry point for the new mainline.
It cannot bypass `development-ready.json` and assembles all three tasks into one loop; task
differences enter the platform only through environment observations, rewards, termination, and
legal transformations. Development commits may be split by verifiable module, but the first
formal parameter update must occur only after all module development, the full test suite, real
foundation-model inference, and deployment audit have passed.
