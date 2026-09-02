# R0020 Context

## Starting Point

- The user explicitly authorized starting R0020 on 2026-08-31.
- Starting commit: `2ed7edd13b3edb003ba9fd53e04a1b693485eda8`.
- Branch: `feat/research-loop`; the worktree was clean at startup, and the local branch was 4
  commits ahead of the remote.
- Current capability level: `L0 not passed`.
- Sole objective: establish a complete privileged teacher/oracle ceiling for
  `carry_living_room_basket/v1`.

## R0019 Boundary

- R0019 is closed with the conclusion `invalid`; this round does not modify or continue
  `docs/research-loop/0019/`.
- R0019's independent per-arm CEM optimized only the grasp terminal state and then used a fixed
  transport twist; it formed bimanual contact only on development seed 19001, subsequently lost
  contact, and did not implement the complete success state machine.
- This round does not copy R0019 runner's source hash, clean-worktree, qualification report, or
  confirmation provenance gates.
- R0019's contact observations may serve as development clues, but not as R0020 success or
  capability evidence.

## Current Shortest Board and Evidence Boundary

- The shortest board remains that the simplest formal task has no complete L0 oracle ceiling.
- This round is `development` only: reading simulator-private robot, basket, handle, target, and
  contact states is allowed.
- The teacher must execute through the formal 16-dimensional action interface, original MuJoCo
  physics, `DualArmSafetySupervisor`, and the two-step predictive collision filter; it must not
  teleport, directly rewrite the authoritative `MjData`, bypass the safety layer, modify task
  success conditions, or change physics parameters.
- Both confirmation and sealed final are `not_run`; this round will not start them.

## Reopened on 2026-08-31

- After commit `f7b27a38b1b2ccbbeba8a8f3783c7feed646b203`, the user explicitly authorized
  reopening R0020; the main hypothesis is unchanged, and R0021 is not created.
- `f7b27a3` explicitly classifies attempts 1–3 as implementation iterations after code changes;
  they did not reach behavior entry, are not independent repeated evidence, and are insufficient
  to determine that the joint-planning route failed.
- Revoke the judgment that there were “three consecutive rounds without capability progress from
  R0018 to R0020”: R0018 is an archive of an old mechanism, R0019 is `invalid`, and the old R0020
  was not yet behavior-ready.
- The reopened phase addresses `acquire` only: the joint planner generates an executable
  near-field path without illegal robot–basket penetration, and the online tracker uses pad/handle
  geometry and actual contact feedback to complete alignment and close the grippers.
