# Research Loop Status

## Current Status

- Current research mechanism: capability first; see the repository root `AGENTS.md` for the rules.
- Current capability level: `L0 not passed`.
- Latest capability baseline:
  `runs/foundation-world-model/r0001-p01-baseline-v4-s20260812`，24 Episode、
  1,600 update, 0 success, Actor not unlocked.
- Latest round: `0020`; the second reopen has ended, and the current capability level remains `L0 not passed`.
- R0020 used joint chassis/dual-arm grasp-configuration planning, a joint keyframe path, and payload-relative closed-loop tracking; it did not reuse R0019's independent per-arm CEM or fixed transport twist.
- All 3 historical physics runs on fixed development seed `19001` produced `0 success`, reaching only `approach/acquire/failed_hold`; no complete dual-pad contact was formed. All recorded `0` actual severe collision and `0` safety intervention.
- These three runs each followed a code change and are now classified as implementation iterations 1–3, not independent repeated evidence. Because they did not form 10 consecutive steps of dual-arm contact and enter `secure`, they did not reach the differentiated mechanism's behavior entry and do not constitute a discriminating failure of the joint-planning route.
- The static planner could find a joint terminal configuration approaching or entering contact with all four pads, but the dynamic joint waypoint tracker initially failed to turn the static solution into a real grasp. The reopened implementation fixed non-pad-to-handle basket penetration and used online pad/handle geometry, signed distance, and contact feedback for the final approach and gripper closure.
- Smoke 011 validated the acquire subgoal: on seed 19001 it entered `secure`, with a maximum of `11` consecutive dual-arm-contact steps, `0` safety intervention, and `0` actual severe collision; however, the behavior entry had previously been defined too early.
- The old complete seed 19001 Episode immediately switched back to the static joint target in `secure`; contact was lost before the first payload-relative lift action. It recorded `0 success`, `0` lift, and `0` controlled target progress.
- Smoke 011, the old freeze manifest, and that complete Episode are now all pre-entry implementation evidence; they do not constitute route failure or a countable no-progress round.
- The second reopen fixed only acquire-to-secure control continuity. Smoke 012 reached the corrected entry: after completing continuous contact in `secure`, it executed the first payload-relative lift action, and the subsequent observation still showed dual-arm contact, with no safety intervention or severe collision.
- The frozen v2 complete seed 19001 Episode executed 9 payload-relative lift control steps and then reached `lift_contact_lost`; `0 success`, `0` measurable lift, and `0` controlled target progress.
- The 4-seed development cohort, confirmation, and sealed final are all `not_run`.
- The judgment that “R0018–R0020 are three consecutive rounds without capability progress” has been withdrawn: R0018 is an archive of the old mechanism, R0019 is `invalid`, and R0020's historical implementation results do not count. The v2 result may not be extrapolated to reject the studied route family and does not accumulate as a route-level no-progress round.
- The 19001–19004 development cohort, confirmation, and sealed final are all `not_run`. Do not automatically create R0021, run confirmation, or start another route.

## Historical Archive

`0001`–`0018` belong to the old evidence-first research mechanism and were archived in place on 2026-08-31. Because multiple historical evaluations treat these paths and Git trees as frozen evidence, the directories are neither physically moved nor rewritten in bulk.

Detailed index: `archive/legacy-evidence-loop-0001-0018.md`.

These historical materials may be used to:

- reproduce old experiments and failures;
- extract validated physical, kinematic, safety, and measurement facts;
- avoid repeating hypotheses that have already been rejected.

They must not be used to:

- treat `accepted` for measurement/contract/oracle as capability progress;
- treat the repeatedly inspected, outcome-exposed 24-Episode bank as a sealed confirmation set;
- automatically inherit the approval status of old candidates;
- continue creating layered oracle, lineage, or contract prerequisite chains.

## New-Round Navigation

Each new round retains only four required documents:

- `00-context.md`
- `01-experiment.md`
- `02-results.md`
- `03-summary.md`

Diagnostics, fixes, and reruns for the same capability hypothesis remain in the same directory. Stop after each round; do not automatically create the next round.
