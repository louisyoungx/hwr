# R0020 Results

Status: concluded.

## Run Ledger

The fixed development seed `19001` ran 3 complete physics Episodes, reaching the preregistered
candidate budget. All three ran through the formal 16-dimensional action interface, normal MuJoCo
physics, `DualArmSafetySupervisor`, and the two-step predictive collision filter.

Under the new rules following `f7b27a3`, the preceding interpretation that the preregistered
candidate budget had been reached is withdrawn: attempts 1–3 occurred after implementation
changes and are now uniformly classified as implementation iterations 1–3. Their original
filenames, raw artifacts, hashes, and historical text are retained without overwriting or
renaming; none reached the behavior entry frozen after the 2026-08-31 reopening, so they are
neither three independent replication results nor a discriminating failure of the joint-planning
route.

Common command form:

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_joint_basket_teacher \
  --seed 19001 \
  --output runs/research-loop/0020/development/seed-19001-attempt-<N>.json
```

| attempt | success | stages reached | failure | bimanual contact | maximum lift | safety | severe |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 0 | `approach/acquire/failed_hold` | `acquire_timeout` | 0 step | `0.060549m` | 0 | 0 |
| 2 | 0 | `approach/acquire/failed_hold` | `acquire_timeout` | 0 step | `0m` | 0 | 0 |
| 3 | 0 | `approach/acquire/failed_hold` | `acquire_timeout` | 0 step | `0m` | 0 | 0 |

All three ran the full `1200` steps and ended with `bimanual_task_timeout`. The maximum forbidden
forces were `25.329396N`, `22.219264N`, and `0N`, respectively, all below the `220N` severe
threshold.

Attempt 1's planner solved after reset and before the basket had fully settled; during execution,
the basket was disturbed by a non-task-valid contact, producing a height change of `0.060549m`,
but the tracker recorded no complete dual-pad contact on either side, and
`maximum_controlled_target_progress=0`. This height change was not a successful grasp or lift.

Attempt 2 replanned after the base was in position and the basket had settled; the predicted static
terminal maximum pad signed distance was `0.000582m`, but dynamic execution still had approximately
`0.074rad` of joint error on entering the final waypoint and closed the grippers prematurely. A
short diagnostic used to localize this control timing was not retained as an independent artifact
and is recorded as an "unarchived observation"; it does not support the final conclusion. The final
conclusion uses only the three complete Episodes.

Attempt 3 entered `acquire` only after replanning converged and latched gripper closure only when
the final waypoint error was below `0.018rad`. The static terminal maximum pad signed distance was
`0.000747m`, but formal dynamic execution still recorded `0/0` contact steps on the left and
right, and did not enter `secure`. This shows that the current joint-space waypoint tracker did
not convert the static contact solution into a dynamically closable grasp; this round cannot
further attribute failure to `lift/transport/place/release/stabilize`.

## Promotion and Stopping

- Single-seed promotion gate not met: `0/3` complete successes.
- Small development cohort `19001`–`19004`: `not_run`.
- confirmation: `not_run`, `valid=null`.
- sealed final: `not_run`, `valid=null`.
- The preregistered 3-Episode candidate budget was reached, so this candidate was stopped; do not
  switch seeds, expand the cohort, or automatically start another technical route.

The stopping decision above is a historical record under the old rules. R0020 was reopened on
2026-08-31; the new debugging and candidate-discrimination budgets are governed by the reopening
revisions in `01-experiment.md`. Historical attempts 1–3 do not count toward the new `24`-smoke
limit.

## Raw Artifacts

- attempt 1:
  `runs/research-loop/0020/development/seed-19001-attempt-1.json`,
  SHA-256 `2bd0f644194c8c30a2c83a298dbd797e1cbdd0f8cf1ef84a29742c6b0d009839`,
  `5,221` bytes, Episode wall time `6.5328s`.
- attempt 2:
  `runs/research-loop/0020/development/seed-19001-attempt-2.json`,
  SHA-256 `1694886ec73874529bc5d3470bea5f11e85bb472fecc297f8763c90eabc034d3`,
  `5,178` bytes, Episode wall time `6.6917s`.
- attempt 3:
  `runs/research-loop/0020/development/seed-19001-attempt-3.json`,
  SHA-256 `7fcb863c1d88bfdec6c21cf607f9bc78b21924e138f34835eb1258e441898a30`,
  `5,177` bytes, Episode wall time `7.8084s`.
- Each JSON has a `.sha256` sidecar in the same directory; `runs/` is managed by `.gitignore`.

## Implementation and Verification

- The new implementation consists of the joint 13-dimensional grasp-configuration search and
  interpolated-path check in `joint_basket_planner.py`, together with keyframe tracking and the
  complete stage state machine in `joint_basket_teacher.py`.
- The planner runs on a copied `MjData`; a focused test verifies that it does not modify the
  authoritative `qpos/qvel/ctrl`.
- The following initial closeout checks are retained as historical records: focused tests
  `4 passed`, bimanual regression `50 passed`, the Python size check passed for `465` files, and
  the architecture check passed.

## 2026-08-31 Reopening Debug

After reopening, 11 short physics smokes with raw records were run on seed 19001, without reaching
the `24`-smoke limit; the total wall time recorded in the artifacts was `72.1399s`, and the local
time span from smoke 001 to smoke 011 was approximately 20 minutes, below the 2-hour active
debugging limit. All smokes used the formal 16-dimensional action interface, normal MuJoCo physics,
and the original safety layer.

| smoke | key implementation change | steps | maximum consecutive bimanual contact | final stage | entry | safety / severe |
|---|---|---:|---:|---|---|---|
| 001 | reopening baseline | 406 | 0 | `failed_hold` | no | 0 / 0 |
| 002 | online pad/handle target | 406 | 0 | `failed_hold` | no | 0 / 0 |
| 003 | proactively open when misaligned | 406 | 0 | `failed_hold` | no | 0 / 0 |
| 004 | early online closed-loop takeover | 406 | 0 | `failed_hold` | no | 0 / 0 |
| 005 | initial terminal collision constraint | 50 | 0 | `approach` | no | 0 / 0 |
| 006 | path-aware joint planner | 390 | 0 | `failed_hold` | no | 0 / 0 |
| 007 | continue closing after alignment | 390 | 0 | `failed_hold` | no | 0 / 0 |
| 008 | slightly negative signed-distance target | 390 | 0 | `failed_hold` | no | 0 / 0 |
| 009 | formal gripper limit and fixed acquire horizon | 550 | 2 | `failed_hold` | no | 0 / 0 |
| 010 | online dual-pad distance balancing | 550 | 2 | `failed_hold` | no | 0 / 0 |
| 011 | maintain gripper preload after dual-pad contact | 204 | 11 | `secure` | yes | 0 / 0 |

Key repair chain:

1. The old planner treated all basket geoms as allowing robot contact; the static four-pad solution
   included approximately `45mm` of palm/wall penetration at most. The reopened implementation
   allows only the four pad–corresponding handle contacts and adds four-point interpolated-path
   penetration to the elite selection in the joint search.
2. The joint path runs only to the near field without contact; thereafter, the real-time pad
   midpoint, handle pose, two-pad signed distance, and contact pair control the two-arm targets
   and gripper closure.
3. Online gripper closure actively opens when misaligned and closes progressively after alignment;
   when the two-pad distances are imbalanced, it applies a maximum `3mm` lateral correction; after
   dual-pad contact forms, it maintains gripper target `1.0`.

Smoke 011 reached the frozen behavior entry:

- seed `19001`;
- entered `secure`;
- `maximum_concurrent_steps=11`;
- all four pads produced real handle contact;
- `0` safety intervention;
- `0` actual severe collision;
- artifact:
  `runs/research-loop/0020/debug/smoke-011-contact-preload.json`;
- SHA-256:
  `a2cc719233ae95dd8137dfe41a785ddcc73bd1a151ce63358d1b977044a3a053`.

Frozen manifest:

- `runs/research-loop/0020/debug/behavior-entry-freeze.json`;
- SHA-256:
  `2a59912d914d35655a454448505a1f9fd12b0ec8f9c228d3fbe3c92e68d6dea7`;
- The hashes of the three frozen controller files were reverified after the complete Episode and
  matched.

## Reopened Candidate Discrimination

After behavior entry, exactly one complete seed 19001 Episode was run:

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_joint_basket_teacher \
  --seed 19001 \
  --output runs/research-loop/0020/development/reopened-candidate-seed-19001.json
```

Results:

- `0/1` success;
- ran `1200` steps and ended with `bimanual_task_timeout`;
- stages reached:
  `approach/acquire/secure/failed_hold`;
- `maximum_concurrent_steps=11`, confirming that the differentiating acquire mechanism was actually
  executed in the discrimination Episode;
- `left_contact_steps=11`, `right_contact_steps=17`, `simultaneous_contact_steps=11`;
- contact was lost after entering `secure`, with `teacher_failure_stage=secure_timeout`;
- `maximum_lift_m=0`, `maximum_controlled_target_progress=0`;
- `0` safety intervention, `0` actual severe collision, maximum forbidden force `0N`;
- artifact:
  `runs/research-loop/0020/development/reopened-candidate-seed-19001.json`;
- SHA-256:
  `870321f7c935b84dc8899fb8bec34b7ae2405e38a58bba3b77e65004d733c170`;
  `5,671` bytes.

The complete seed 19001 Episode did not succeed end to end, so the cohort promotion condition was
not met. The development cohort `19001`–`19004`, confirmation, and sealed final were all
`not_run`.

## Evidence Reclassification Before the Second Reopening

After commit `fc8938c`, smoke 011, the old freeze manifest, and
`reopened-candidate-seed-19001.json` retained their original files, names, and hashes, but the
conclusion boundaries were revised as follows:

- smoke 011 proves only that the `acquire` subgoal reached `11` steps of bimanual contact and
  entered `secure`;
- the old `behavior-entry-freeze.json` froze an entry definition that was too early and is now
  treated as a historical pre-entry manifest;
- the complete Episode's `secure` stage began at step 203, but immediately switched back to a
  static joint target and `GRASP_GRIPPER`; contact was then lost and `secure_timeout` occurred;
- that Episode had `maximum_lift_m=0` and `maximum_controlled_target_progress=0`, and executed no
  payload-relative lift action;
- it is therefore reclassified as a pre-entry implementation result, not a frozen candidate
  discrimination result, and does not count as route failure or a no-progress round.

R0020 was reopened for a second time; subsequent smokes start at `012` and carry forward the
remaining `13` smokes / approximately `100` minutes of debugging budget. The corrected behavior
entry and candidate-discrimination contract are in `01-experiment.md`.

## First Reopening Closeout Verification

- R0020 focused tests: `9 passed`.
- Bimanual-related regression: `55 passed in 48.70s`.
- Python size check: `466` files passed; no file exceeded 800 lines and no function exceeded 200
  lines.
- architecture check: passed.
- `git diff --check`: passed.

## Second Reopening Results

The second reopening changed only `acquire → secure` control continuity:

- `secure` carries forward the successful online pad/handle geometry, signed distance, contact
  feedback, and gripper target from `acquire` instead of switching back to a static joint target;
- the gripper target of the existing payload-relative lift tracker was changed from
  `GRASP_GRIPPER` to `1.0`, preserving the grasp preload established by `acquire/secure`;
- `transport`, `place`, `release`, and `stabilize` were not modified.

### Smoke 012

Command:

```bash
MUJOCO_GL=glfw .venv/bin/python - <<'PY'
# bounded R0020 secure-handoff smoke, seed=19001, max_steps=420
# Full step-by-step records are written to the artifact below
PY
```

Results:

- `219` control steps;
- after `acquire`, continuous contact was maintained in `secure`, with
  `maximum_concurrent_steps=26`;
- the controller actually entered `lift`;
- at step 218, one payload-relative lift action was generated and executed through the formal
  interface: normalized `vz=0.3166666667` for both arms, with both gripper targets equal to `1.0`;
- the action was not modified by the safety layer;
- both dual-pad contacts remained in the observation following the action;
- `0` safety intervention, `0` actual severe collision;
- corrected behavior entry: reached;
- artifact:
  `runs/research-loop/0020/debug/smoke-012-secure-continuity.json`;
- SHA-256:
  `1dbdf6d665c93ac34b632087f5a37da0d1543af7feedb86bcd8a2199d5afbbc3`.

The second reopening used only smoke 012 and did not exceed the remaining 13-smoke /
approximately 100-minute debugging budget.

### Freeze v2

- manifest:
  `runs/research-loop/0020/debug/behavior-entry-freeze-v2.json`;
- SHA-256:
  `87c54df07a71a2b6ed6ad4a666ad1670783834fd83526962d7fb8f47962ab8b8`;
- Frozen source:
  `joint_basket_acquire.py`,
  `joint_basket_planner.py`,
  `joint_basket_teacher.py`;
- All three source hashes were reverified after the complete Episode and matched the manifest.

### Frozen Candidate Discrimination v2

After reaching the corrected entry, exactly one complete seed 19001 Episode was run:

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_joint_basket_teacher \
  --seed 19001 \
  --output runs/research-loop/0020/development/reopened-v2-candidate-seed-19001.json
```

Results:

- `0/1` success;
- `1200` steps, `bimanual_task_timeout`;
- stages reached:
  `approach/acquire/secure/lift/failed_hold`;
- `maximum_concurrent_steps=26`;
- `left_contact_steps=26`, `right_contact_steps=32`,
  `simultaneous_contact_steps=26`;
- the payload-relative lift tracker executed `9` control steps from step 218–226, after which
  bimanual contact was lost, with `teacher_failure_stage=lift_contact_lost`;
- `maximum_lift_m=0`, `maximum_controlled_target_progress=0`;
- `0` safety intervention, `0` actual severe collision, maximum forbidden force `0N`;
- artifact:
  `runs/research-loop/0020/development/reopened-v2-candidate-seed-19001.json`;
- SHA-256:
  `f933c511ce56faec806c23005015caebc9d896ab9934e34567384b10c4cc1689`;
  `6,161` bytes.

The complete seed 19001 Episode did not succeed end to end, so the `19001`–`19004` development
cohort was not run. confirmation and sealed final remain `not_run`.
