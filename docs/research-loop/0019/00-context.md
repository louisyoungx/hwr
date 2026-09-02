# R0019 Context

## Starting Point

- The user explicitly authorized starting R0019 on 2026-08-31.
- Starting commit: `0df823abb602c528b7b138ab17fb6bc9097b3048`.
- Branch: `feat/research-loop`.
- Current capability level: `L0 not passed`.
- Latest complete 3D learning baseline: 24 Episode, 1,600 update, 0 success; Actor not unlocked.
- Sole objective: establish a privileged teacher/oracle ceiling for
  `carry_living_room_basket/v1` under normal MuJoCo physics and an independent safety layer.

## Historical Evidence

- `R0001-P57`: Both arms had negative command margins for all 36/36 pairs in the old B2
  preposition stage, with `ever_bilateral_ready=0/36`; the 100-step applied command budget
  was approximately `0.349–0.433m`, clearly smaller than the initial target distance. This
  evidence only shows that the old B2/B3/B4 horizon is not a feasible control ceiling.
- `R0001-P61`: generic `Candidate` and the B0–B7 primitives lack `entity_role`,
  `interaction_type`, and `destination_target`, and therefore cannot uniquely express the
  complete task transition.
- This round does not continue P88, P76, or P68, and does not add a new oracle, contract, or
  lineage qualification chain.

## Current System

- Action: 16-dimensional, consisting of chassis linear/angular velocity, a 6D base-frame tool
  twist for each arm, and target positions for both grippers.
- Control: the MuJoCo backend uses Jacobian damped least squares to convert tool twists into
  joint-velocity targets.
- Safety: every action first passes through `DualArmSafetySupervisor`, followed by a
  two-control-step predictive collision check; when forbidden robot contact reaches `220N`,
  the action is replaced with hold.
- Success: requires at least 10 consecutive steps of bimanual contact, completion of the
  target displacement under bimanual-contact control, and satisfaction of the pose/velocity
  gates on the target support for 40 consecutive steps; actual severe collision must be 0.

## Evidence Boundary

- This round is `development`; reading simulator-private state, object/target poses, and
  contact states is allowed.
- The teacher may enter the same MuJoCo backend only through the normal 16-dimensional action
  interface; it must not teleport, rewrite state, disable or weaken the safety layer, relax
  success conditions, or modify the physics.
- Development seeds may be inspected repeatedly; confirmation seeds are run only after the
  implementation, gates, and budget are frozen.
- The 24-Episode bank from R0001–R0018 is historical development evidence only and does not
  enter this round's confirmation set.
