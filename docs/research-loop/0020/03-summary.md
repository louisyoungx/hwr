# R0020 Summary

Status: second reopening work concluded.

- Current capability level: `L0 not passed`.
- Current conclusion: `abandoned`; only the precisely defined candidate frozen in the second
  reopening is rejected.
- confirmation: `not_run`.
- sealed final: `not_run`.

## Reopening Notes

Commit `f7b27a3` established that R0020 attempts 1–3 were implementation iterations after three
code changes; they did not reach the candidate's differentiating mechanism and cannot serve as
independent replication evidence or a discriminating route failure. The user explicitly
authorized reopening R0020 in the original directory; do not create R0021.

The current work addresses only the implementation-readiness gap in `acquire`. Behavior entry
requires seed 19001 to form at least 10 consecutive control steps of real bimanual contact and
enter `secure` under the formal 16-dimensional action interface, normal MuJoCo physics, and the
original safety layer. Do not expand or evaluate `lift`, `transport`, `place`, `release`, or
`stabilize` before entry is reached.

Smoke 011 verified the `acquire` subgoal: it entered `secure` with `11` steps of continuous
bimanual contact, `0` safety intervention, and `0` actual severe collision. However, commit
`fc8938c` established that this entry definition was too early: at least one payload-relative lift
action must be executed through the formal interface, and bimanual contact must remain in its
subsequent observation. This round had not previously satisfied that condition.

The second reopening reached the corrected behavior entry in smoke 012: `secure` continued the
online geometry/contact feedback and maintained continuous bimanual contact for `26` steps; the
first payload-relative lift action was executed formally, its subsequent observation still showed
contact on both arms, and there were `0` safety intervention and `0` severe collision.

## Historical Conclusion Boundaries

R0020 implemented a joint bimanual route different from R0019:

- jointly optimize the base approach position and 12 arm joints on a copied MuJoCo state;
- simultaneously optimize the signed distances from four finger pads to the handles on both sides,
  and check the joint interpolated path;
- use online keyframe joint tracking and post-grasp payload-relative Cartesian feedback;
- the controller explicitly implements
  `approach/acquire/secure/lift/target_transport/place/release/stabilize`,
  and no longer uses a fixed transport twist.

All 3 historical implementation iterations on fixed development seed 19001 were `0 success`. All
three reached only `approach/acquire/failed_hold`, formed no complete dual-pad contact, and had
`acquire_timeout` as the earliest failure stage. All Episodes had `0` safety intervention and `0`
actual severe collision.

The static planner can find joint terminal states where the four-pad signed distances approach or
enter contact, and the focused test confirms that the planner does not modify authoritative state;
however, the joint-space waypoint tracker did not convert that static solution into a real dynamic
grasp. These results only locate the behavior-readiness gap in the old implementation: the
payload-relative mechanism was not actually executed, so they cannot reject the joint-planning
primary hypothesis, evaluate downstream stages, or establish an L0 oracle ceiling.

## Reopening Results

The reopening phase fixed two root causes in the old implementation:

1. The old static planner allowed non-pad–handle basket contacts such as palm/wall contact,
   producing an infeasible penetrating terminal state; the new planner allows only the four
   pad–corresponding handle contacts and includes joint-path penetration in the search.
2. Final approach no longer triggers one-shot gripper closure through a joint-error gate; it
   tracks pad/handle relative geometry online, progressively closing based on signed distance,
   single/dual-pad contact, and a maximum 3mm lateral balancing correction.

The complete seed 19001 Episode from the previously purported frozen implementation reached
`approach/acquire/secure/failed_hold`, with a maximum of `11` consecutive bimanual-contact steps.
However, `secure` immediately switched back to a static joint target and lower gripper preload;
contact was lost before the first payload-relative lift action, ending in `secure_timeout`,
`0 success`, `0` lift, and `0` controlled target progress.

This complete Episode is therefore reclassified as a pre-entry implementation result: it verifies
the `acquire` subgoal and rejects the `acquire→secure` handoff implementation at that time, but it
did not execute the primary hypothesis's differentiating payload-relative mechanism. It cannot
reject the current frozen candidate or the joint-planning route, and does not count as a
no-progress round.

## Second Reopening Results

The second reopening fixed only `acquire→secure` control continuity:

- `secure` continues using `acquire`'s online pad/handle geometry, signed distance, contact
  feedback, and `1.0` gripper preload;
- after completing continuous contact in `secure`, it executes an action from the existing
  payload-relative lift tracker;
- `transport`, `place`, `release`, and `stabilize` were not modified.

Smoke 012 reached the corrected entry on seed 19001, after which three controller source files
were frozen. The frozen v2 complete Episode actually executed
`approach/acquire/secure/lift/failed_hold`:

- maximum consecutive bimanual contact: `26` steps;
- the payload-relative lift action actually executed for `9` control steps;
- contact was subsequently lost, `lift_contact_lost`;
- `maximum_lift_m=0`, `maximum_controlled_target_progress=0`;
- `0` safety intervention, `0` actual severe collision.

Because the complete seed 19001 Episode did not succeed, the development cohort was not run under
the frozen conditions. This result rejects the currently precisely defined frozen candidate,
  "collision-aware joint planner + online acquire/secure feedback + current payload-relative lift
  tracker"; it must not be extrapolated to reject the entire motion-planning/trajectory-optimization
family.

## Permitted Claims

- The current joint static planner can find candidate configurations satisfying the four-pad
  distance targets in a copied state.
- The historical joint-space waypoint tracker formed no complete dual-pad contact in any of the
  three implementation iterations on fixed development seed 19001.
- The three Episodes had no safety intervention or actual severe collision.
- The reopened implementation reached behavior entry through the formal action, physics, and safety
  layers, forming 11 steps of real bimanual contact and entering `secure`; under the new rules,
  this counts only as completion of the `acquire` subgoal.
- The old complete seed 19001 Episode shows that the `acquire→secure` handoff immediately loses
  contact and executes no lift action.
- Second-reopening smoke 012 satisfies the corrected entry: the observation after executing a
  payload-relative lift action still maintains bimanual contact.
- The frozen v2 candidate lost contact during `lift` and produced no measurable lift or target
  progress.

## Claims Not Permitted

- Do not claim that `carry_living_room_basket/v1` has a viable L0 teacher ceiling.
- Do not use historical attempts 1–3 to reject the joint motion-planning, trajectory-optimization,
  or payload-relative tracking route; they show only that the implementation at that time was not
  behavior-ready.
- Do not describe smoke 011 or the old complete Episode as having reached the corrected behavior
  entry.
- Do not use the old complete Episode to reject the frozen candidate or the joint-planning route;
  it rejects only the `acquire→secure` handoff implementation at that time.
- Do not extrapolate the v2 candidate's failure to failure of the motion-planning,
  trajectory-optimization, or payload-relative tracking family.
- Do not describe attempt 1's uncontrolled height change as a successful grasp or lift.
- Do not attribute failure to `lift`, `target transport`, `place`, `release`, or `stabilize`, which
  were not actually executed.
- Do not claim any deployable state-policy, visual-policy, generalization, or hardware capability.

## Retention and Rollback

- Retain the development-only planner, controller, runner, focused tests, and three raw artifacts
  to reproduce the gap between the static solution and dynamic contact.
- Retain the 11 reopening smokes, the historical behavior-entry freeze manifest, and the complete
  seed 19001 artifact; `runs/` is not written to Git, but the paths and hashes are recorded in the
  document.
- Retain smoke 012, `behavior-entry-freeze-v2.json`, and
  `reopened-v2-candidate-seed-19001.json` with their hashes.
- Do not modify `docs/research-loop/0019/` or `0001/`–`0018/`.
- Retain no confirmation gate; confirmation and sealed final for this round are both `not_run`.
- Do not automatically start another route or R0021.

## Counting Revision

Withdraw the judgment that "R0018–R0020 are three consecutive rounds without capability progress":

- R0018 is an archive of the old mechanism and does not count;
- R0019 is `invalid` and does not count;
- historical R0020 attempts 1–3 do not count; smoke 011 and the old complete Episode also do not
  count because they did not execute a payload-relative lift action.
- The second-reopening v2 candidate executed the differentiating payload-relative mechanism, but
  the conclusion explicitly does not permit rejecting the studied route family; under the
  `fc8938c` rule, this round still does not count as a route-level no-progress round.

There is currently no mandatory stop for "three rounds without progress." The second reopening of
R0020 has ended within budget; do not automatically create a new round or run confirmation.

## Resource Allocation

The first reopening ran 11 short smokes with raw records; the second reopening reached entry after
running only smoke 012 and did not exhaust the remaining 13-smoke / approximately 100-minute
debugging budget. It then ran only one frozen complete Episode.
