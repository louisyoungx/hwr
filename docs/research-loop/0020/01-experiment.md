# R0020 Experiment

## Single Primary Hypothesis

Unlike R0019's independent per-arm CEM and fixed transport twist, a **joint bimanual keyframe
motion planner + payload-relative closed-loop tracker** can complete the following under the
standard interface and physical constraints:

`approach → acquire → secure → lift → target_transport → place → release → stabilize`.

The planner jointly solves keyframes for both arms and the base on a copied `MjData`, maintains
the geometric relationship between the two grasp sites and the basket handles, and checks the
interpolated path; the online tracker makes closed-loop corrections from the current
payload/handle/tool poses and does not execute long-duration fixed-constant actions.

## Controls and Invariants

- Historical control: R0019 seed 19001 teacher, `0 success`; although it achieved up to 83 steps
  of bimanual contact, it lost contact during transport.
- R0020 candidate: an independent new controller; the R0019 teacher is not modified.
- Keep the same `carry_living_room_basket/v1`, seed, task horizon, physics, randomization, safety
  thresholds, and formal 16-dimensional action interface.
- Primary metric: complete Episode `success=true`.
- Stage metrics: grasping, continuous bimanual contact, lifting, controlled target transport,
  target support, release, and 40-step stabilization.
- Guard metric: actual severe collision `=0`; report all safety interventions and failure stages.

## Minimal Development Experiment

The following "3 complete Episode" budget belongs to the initial execution contract and has been
superseded by the "2026-08-31 Reopening Revision" at the end; it is retained only to explain the
historical origin of attempts 1–3.

- Fixed development seed: `19001`.
- At most `1200` control steps per Episode.
- Candidate debugging budget: at most 3 complete physics Episodes; implementation-level unit tests
  or short smokes do not count as additional candidates, but may not replace complete Episodes.
- Raw artifact directory: `runs/research-loop/0020/development/`.
- Single command entry point:

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_joint_basket_teacher \
  --seed 19001 \
  --output runs/research-loop/0020/development/seed-19001.json
```

## Promotion and Stopping

- Expand to the fixed development cohort `19001, 19002, 19003, 19004` only if seed 19001
  completes successfully under the formal success state machine and actual severe collision is 0.
- The cohort is used only to check development stability; it is not confirmation.
- If there is no complete success within the fixed implementation budget or 3 complete Episodes,
  stop this candidate, conclude `abandoned`, and report the earliest failure stage and the stages
  covered.
- Do not switch seeds, delete failures, or automatically start another technical route, the next
  round, or confirmation.

## Decision

- Complete success on a single seed: point solvability evidence for `validated_development`, but
  still insufficient to advance L0.
- If the small cohort reaches `>=3/4` complete successes and all severe collisions are 0: R0020
  is `validated_development`, allowing L0 to be recorded as passed; deployable policy capability
  still may not be claimed.
- Otherwise, the result is `abandoned`; if caused by implementation deviation, a physics/safety
  bypass, or data contamination, the result is `invalid`.

## 2026-08-31 Reopening Revision

Under the new rules following commit `f7b27a3`, the user explicitly authorized reopening R0020.
The primary hypothesis remains unchanged; do not create R0021. Attempts 1–3 occurred after code
changes and are three implementation iterations of the same candidate; they did not reach the
differentiating mechanism relative to R0019 in this round, so they are neither independent
replication evidence nor a discriminating failure of the joint-planning route.

### Current Implementation Scope

- For now, address only the earliest failing `acquire` stage: convert the static joint-contact
  configuration into dynamic final approach, gripper closure, and sustained bimanual contact.
- Do not continue expanding the not-yet-reachable `lift/target_transport/place/release/stabilize`;
  existing downstream code is not evidence of current implementation readiness.
- Final approach and gripper closure must use online pad/handle geometry and real contact feedback;
  do not rely only on `<0.018rad` joint error, and do not perform unbounded constant actions or gain
  sweeps.

### Differentiating Mechanism and Behavior Entry

The differentiating mechanism remains "joint bimanual planner + payload-relative closed-loop
tracker". Its behavior entry condition before candidate discrimination is frozen as follows:

- seed `19001`;
- formal 16-dimensional action interface;
- normal MuJoCo physics;
- the original `DualArmSafetySupervisor` and two-step predictive collision filter;
- at least `10` consecutive control steps of real bimanual contact;
- the controller actually enters `secure`.

Class fields, a static joint solution, pad-distance thresholds, an unexecuted state machine, or
unilateral/non-continuous contact do not count as reaching entry.

### Implementation/Debug Budget

- Starting from this reopening, active implementation debugging is limited to at most `2` hours of
  wall time or at most `24` short physics smokes with raw records, whichever comes first.
- Each smoke uses seed `19001` and runs only until behavior entry, an unambiguous `acquire` failure,
  or the frozen short-run step limit; save the command, source version, configuration,
  stage/contact/pad geometry, safety result, and raw artifact.
- Retain at least a commit, patch, or minimal source hash for versions that affect the conclusion.
- Freeze the current implementation immediately upon reaching behavior entry; the debug budget ends
  and the candidate-discrimination budget begins.
- If the debug budget is exhausted without reaching entry, stop the current implementation; the
  conclusion applies only to that implementation, does not reject the joint-planning route, does
  not count as "three rounds without progress", and does not automatically start a new round or
  confirmation.

### Frozen Candidate-Discrimination Budget

1. After behavior entry is reached, first run one complete seed `19001` Episode.
2. Run the originally frozen development cohort `19001, 19002, 19003, 19004` only if that Episode
   succeeds end to end and actual severe collision is `0`.
3. Advance L0 as `validated_development` only if the cohort reaches `>=3/4` complete successes and
   all actual severe collisions are `0`.
4. Do not run confirmation or sealed final in this round.

## 2026-08-31 Second Reopening Revision

Under the new rules following commit `fc8938c`, the user explicitly authorized reopening R0020
again. The primary hypothesis remains "joint bimanual planner + payload-relative closed-loop
tracker"; do not create R0021.

Smoke 011 demonstrated that the `acquire` subgoal could form `11` consecutive steps of bimanual
contact and enter `secure`, but the behavior entry definition frozen in the first reopening was
too early: in `secure`, the controller immediately switched from online pad/handle feedback back
to a static joint target and `GRASP_GRIPPER`, after which the complete Episode lost contact; no
payload-relative lift action was executed through the formal interface. Therefore, smoke 011, the
old `behavior-entry-freeze.json`, and `reopened-candidate-seed-19001.json` are all reclassified
as pre-entry implementation evidence and do not constitute route failure or a countable
no-progress round.

### Current Implementation Scope

- Fix only the `acquire → secure` control continuity.
- `secure` must carry forward the successful online pad/handle geometry, contact feedback, target
  pose, and gripper preload; it must not switch back to a static joint target.
- Do not expand `transport/place/release/stabilize` or other downstream functionality; only the
  first action of the existing payload-relative lift tracker may be executed to validate the stage
  handoff.
- Do not perform unbounded constant actions or gain sweeps.

### Corrected Behavior Entry

All of the following conditions must be met in the same short physics smoke with seed `19001`:

- the formal 16-dimensional action interface, normal MuJoCo physics, the original
  `DualArmSafetySupervisor`, and the two-step predictive collision filter;
- the continuous contact required to complete `secure` while maintaining bimanual contact;
- the controller actually enters `lift`;
- at least one action generated by the payload-relative lift tracker has been executed through the
  formal interface;
- the observation after that action has been recorded;
- bimanual contact remains in the subsequent observation, with actual severe collision equal to `0`.

Entering `secure` alone, generating but not executing a lift action, or losing contact in the first
subsequent observation does not count as entry.

### Remaining Debug Budget

- Carry forward the original budget from the first reopening without resetting it: at most `13`
  additional short physics smokes with raw records, namely smokes 012–024; approximately `100`
  minutes of active implementation-debugging wall time remain, whichever comes first.
- Run each smoke until the corrected entry, an unambiguous `acquire/secure` failure, or the frozen
  short-run step limit; do not stop immediately after a stage transition.
- Continue saving the command, source version or minimal source hash, configuration, post-action
  observation, contact, and safety results.
- If the budget is exhausted without reaching entry, stop only the current implementation; do not
  reject the primary hypothesis or count a no-progress round.

### Corrected Candidate-Discrimination Budget

Immediately freeze the source and configuration after reaching entry, then run exactly one complete
seed `19001` Episode:

1. Run the originally frozen `19001`–`19004` development cohort only if the Episode succeeds end
   to end and actual severe collision is `0`;
2. if the differentiating payload-relative mechanism is actually executed but the complete Episode
   fails, reject only the precisely defined frozen candidate and do not extrapolate to reject the
   motion-planning/trajectory-optimization family;
3. do not run confirmation or sealed final, and do not automatically start another route or a new
   round.
