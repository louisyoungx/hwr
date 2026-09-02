# R0019 Experiment

## Main Hypothesis

`carry_living_room_basket/v1` can be completed reliably by a closed-loop teacher that reads
simulator-private state without changing normal MuJoCo physics, the task success state machine,
or the independent safety layer. The teacher uses an explicit task state machine and computes
actions from the current geometry; it does not depend on the generic candidate or the limited
horizon of B0–B7.

## Comparisons

- baseline: the existing generic B0–B7 primitive, which constructs a unique candidate from the
  task payload geometry and uses the formal 16-dimensional action interface, MuJoCo physics,
  the task tracker, and the safety layer.
- teacher: a task-specific privileged feedback controller that reuses the same
  action/physics/safety/success path.
- Paired conditions: the same seed, task config, randomization, physics, safety thresholds, and
  Episode horizon.
- The teacher must actually cover
  `approach/acquire/secure/lift/target_transport/place/release/stabilize`; omitting any major
  stage of the success state machine constitutes implementation deviation and disqualifies it
  from confirmation.

## Development Phase

1. Run a baseline Episode first, recording the earliest failure stage, actions, contacts, safety
   interventions, and termination reason.
2. Prefer reusing the backend's Cartesian twist + Jacobian DLS; the teacher directly computes
   closed-loop targets from the current handle, payload, target, and tool poses.
3. First pass a single-seed physics smoke; if it fails, fix only the earliest stable blocker and
   do not create an additional qualification gate.
4. Development seed domain: `0 <= seed < 1_000_000`; these seeds must not enter confirmation.

## Frozen Confirmation Design

The following design is frozen before any confirmation result becomes visible:

- controllers: `GenericBasketPrimitiveBaseline` and
  `PrivilegedBasketTeacher`；
- seed domain: `91_900_001 + 104_729 * index`, `index=0..99`;
- Episodes: 100 per controller, paired by the same seed;
- teacher success `>=80/100`;
- actual severe collision `=0`;
- Startup is allowed only from a clean, committed worktree; the runner rejects a dirty tree
  before executing the first Episode;
- A clean development qualification report under the same source commit and source-file hash
  must be provided; the report must finish completely, contain at least 1 successful teacher
  Episode, and have confirmation status `not_run`;
- The confirmation output path must not exist, preventing overwriting of viewed confirmation
  results;
- Safety guards: `DualArmSafetySupervisor`, two-step predictive collision, and the `220N`
  severe threshold all remain at their default values;
- Each Episode has at most 1,200 steps, and the entire run has a maximum wall time of 1,800
  seconds;
- An individual failure or safety rejection is recorded only as that Episode's result and does
  not interrupt the cohort;
- The complete report covers every baseline/teacher Episode, all failures, safety
  interventions, bimanual contacts, and termination reasons;
- The evaluator strictly verifies the frozen seed domain and exactly one baseline and one teacher
  result for each seed;
- Sole command:

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_bimanual_teacher \
  --mode confirmation \
  --controller paired \
  --qualification-report <clean-development-report.json> \
  --output runs/research-loop/0019/confirmation/paired-100.json
```

If measured development resources show that 100 paired Episodes are clearly unreasonable, the
design may be adjusted and the reason recorded only before the first confirmation run; the
gates cannot be changed after viewing confirmation results.

The actual candidate implements only `approach/acquire/secure/transport_probe` and lacks
`lift/target_transport/place/release/stabilize`; the development teacher also achieved `0/6`
success. Therefore, this round's candidate does not satisfy the implementation contract or the
minimum expansion conditions, and the confirmation command above is not started; this is not an
adjustment to the seed, Episode count, or success gate.

## Conclusion

- Gate result: `validated_development`. It is permissible to state that the task and control
  chain have a feasible teacher ceiling; it is not permissible to claim deployable state/vision
  policies, generalization, or hardware capability.
- Actual conclusion for this round: `invalid`, because the candidate did not implement the
  complete teacher state machine declared in advance. The physical observations from the
  implemented subgoals may be retained; the end-to-end `0/6` result must not be used to locate
  the bottleneck of the complete task, robot, or technical route.
