# R0019 Summary

Status: closed.

- Current capability level: `L0 not passed`.
- Conclusion: `invalid`.

## Conclusion

This round's candidate did not implement the complete teacher state machine declared in advance:
it contains only `approach/acquire/secure/transport_probe` and lacks
`lift/target_transport/place/release/stabilize`. Therefore, the end-to-end experiment constitutes
implementation deviation and cannot be used to determine whether the environment, robot, control
horizon, or complete teacher route is solvable.

In the final development paired cohort:

- generic B0–B7 baseline: `0/6` success, `0/6` bimanual contact;
- privileged teacher: `0/6` success, `1/6` bimanual contact;
- both groups had `0` actual severe collision and `0` safety intervention.

seed 19001 shows that this prototype can form actual bimanual contact through normal MuJoCo
physics, the formal 16-dimensional action interface, and the independent safety layer, for up to
`83` consecutive steps, after which it lost contact during the prototype transport action. The
other 5 development seeds did not form synchronized bimanual contact. No Episode completed
carrying, target placement, release, or the 40-step stabilization gate.

The archivable subgoal observations are:

1. The current acquire prototype is unstable across seeds: 5/6 seeds produced no synchronized
   bimanual contact;
2. The current transport prototype lost contact on one side for the only seed with synchronized
   contact, with maximum controlled target progress of only `0.0035074962m`;
3. There was no safety intervention or actual severe collision in these 12 Episodes.

These observations cover only the implemented subgoals and must not be claimed to be the
earliest or sole bottleneck of the complete task. The 100-seed confirmation status is `not_run`;
no frozen seed was viewed. Runner v2 rejected the current incomplete teacher and requires future
confirmation to provide a clean development qualification report from the same commit with at
least one end-to-end teacher success.

## Permitted Claims

- The existing generic B0–B7 did not reach bimanual contact on this formal task.
- The current privileged CEM + inverse-DLS prototype can form sustained actual bimanual contact
  on one development seed.
- The current prototype has the reproducible acquire and transport substage failures described
  above.

## Prohibited Claims

- Do not claim that `carry_living_room_basket/v1` has a feasible L0 teacher ceiling.
- Do not use this round's end-to-end `0/6` to conclude that the environment, robot, control
  horizon, or complete teacher technical route is unsolvable.
- Do not call the current substage failures the earliest or sole bottleneck of the complete task.
- Do not claim any deployable state policy, vision policy, generalization, or hardware capability.
- Do not treat local bimanual contact, lifting height, passing tests, or loss/distance
  improvement as capability-level progress.

## Retention and Rollback

- Retain R0019 runner v2, the teacher subgoal prototype, and tests as the entry point for failure
  reproduction and prevention of incorrect confirmation; the old v1 decision must not be reused.
- Do not modify `docs/research-loop/0001/`～`0018/`.
- Do not run the world model, Actor, or large-scale training.
- Do not automatically start L1 or the next round; wait for the user to decide whether to narrow
  the task, strengthen the grasp fixture/end effector, introduce a demonstration/motion-planning
  library, or stop this task route.

## Resource Allocation

The main work in this round was actual MuJoCo action/contact/Episode and controller
implementation debugging, exceeding 70%; static auditing and documentation were below 20%, and
evaluation runner/tests were below 10%. No new recursive oracle/lineage gate was created.
