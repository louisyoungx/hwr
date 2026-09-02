# Housework Robot Capability-First Research Rules

## 1. North Star Objective

The project aims to continuously improve the closed-loop physical capabilities of the housework robot, with ultimate focus on generalization across tasks, objects, layouts, language, dynamics, and hardware transfer, as well as success rate, safety, data efficiency, and compute efficiency.

The research loop first optimizes “distance to the next observable capability milestone,” not the number of documents, gates, tests, artifact completeness, or `accepted` entries. Infrastructure and measurement work is valuable only when it directly unlocks a behavior experiment, rules out a high-value route, or protects a final conclusion.

The existing world model, reinforcement learning, candidate generator, B0–B7 primitives, evaluation contract, and documents are all modifiable historical baselines, not architecture that must be preserved. The world-model route must be fairly compared with simpler behavior cloning, state policies, visual policies, or other reproducible baselines; prior investment cannot automatically keep it as the mainline.

## 2. Final Claims and Development-Evidence Tiers

Three evidence tiers must be distinguished. The requirements of the highest tier must not block low-cost development, and low-tier evidence must not be presented as capability:

1. `development`: Locate system solvability and bottlenecks.
   - `simulator-private state`, scripted or planned teachers, manual stages, fixed tasks, fixed layouts, seen seeds, dense diagnostic metrics, and short smokes are allowed.
   - Teachers may be used to generate training data.
   - Results may only be called development evidence or a feasibility upper bound and do not enter the capability baseline.
2. `capability`: Verify closed-loop behavioral improvement by a deployable policy on a frozen distribution.
   - The policy may use only declared deployment observations; evaluation retains real physics, the success state machine, and an independent safety layer.
   - Paired seeds, budget, primary metrics, and guard metrics are frozen before results are used.
   - Only this tier can produce `accepted_capability` and advance the capability ladder at `L2` and beyond; development milestones at `L0` and `L1` follow Section 3.
3. `claim`: Verify generalization to unseen distributions or hardware transfer.
   - Sealed evaluation is generated or opened only after the model, thresholds, and source commit are frozen.
   - Report all seeds, failures, safety events, compute cost, and confidence intervals.
   - Only this tier can produce an external generalization or hardware-capability claim.

Any scripted action, privileged state, or teacher entering `capability`/`claim` policy inputs, actions, or evaluation decisions makes the experiment invalid. Conversely, the fact that the final policy cannot use a teacher must not prohibit establishing an oracle ceiling, collecting demonstrations, or locating control defects during `development`.

## 3. Current Capability Ladder

Research must advance along the following ladder; in principle, it may cross at most one level at a time:

| Level | Goal | Minimum evidence |
|---|---|---|
| `L0` | Task and control chain are solvable | A privileged teacher succeeds consistently under normal physics and the safety layer |
| `L1` | A state policy is learnable | A learning policy using structured simulation state but no scripted actions succeeds in closed loop |
| `L2` | A visual policy is learnable | An RGB-D/language/proprioception policy succeeds on unseen seeds for a single task |
| `L3` | Single-task domain generalization | Generalization across position, appearance, mass, friction, or latency individually |
| `L4` | Multi-task capability | Multiple tasks each meet the frozen success gate, without hiding failures behind an average |
| `L5` | Compositional and unseen-distribution generalization | Sealed closed-loop evaluation on new objects, layouts, language, or dynamics |
| `L6` | Hardware transfer | Closed-loop success on the real robot with independent safety evidence |

`L0` and `L1` are development milestones, not deployable visual capability; they may advance with `validated_development`. From `L2` onward, only `accepted_capability` under the corresponding frozen evaluation can advance the ladder.

The project's current research ladder is `L0 not passed`: the latest complete 3D world-model baseline has only 24 Episodes, 1,600 updates, and 0 success; the Actor is not unlocked. In R0019's paired development cohort, both the generic baseline and privileged teacher achieved `0/6` success; the teacher formed real bimanual contact on only `1/6` seeds, lasting at most `83` steps, and did not establish a complete grasp, lift, transport, place, release, and stable closed loop; confirmation was not started. The three R0020 runs on seed 19001 occurred after three code changes, reached only `approach/acquire`, and did not execute the payload-relative lift/transport mechanism claimed as the distinction from R0019; they are implementation iterations, not three independent repetitions, and are insufficient to count the joint-planning route as a discriminating failure. R0018 is archived under the old mechanism, and R0019 is `invalid`, so R0018–R0020 must not trigger the “three rounds without progress” stop.
After R0020 was reopened, online pad/handle feedback reached 11-step bimanual contact and entered `secure`; the frozen implementation then lost contact in the complete Episode at `secure`, without executing any `lift` or payload-relative action. This result validates the acquire subgoal and eliminates the current acquire→secure handoff implementation, but the differentiated payload-relative mechanism of the main hypothesis was still not executed, so this cannot count as route failure or as a countable no-progress round.
`docs/research-loop/0001/`–`0018/` are archives of the old mechanism. Their measurement evidence may be reused, but no `accepted as ... contract/evidence` entry may be treated as capability progress.

By default, the next round should first establish the `L0` oracle ceiling for the simplest formal task. If `L0` has not passed, do not devote primary resources to the world model, Actor, open-world generalization, or deeper evaluator provenance.

## 4. Development, Confirmation, and Final-Evaluation Isolation

- `development set`: May be inspected and debugged repeatedly, but must be explicitly labeled and cannot support confirmation claims.
- `confirmation set`: Run after the candidate and thresholds are frozen to determine `capability`; once its results have been viewed, it becomes a development set.
- `sealed final set`: Used only for `claim`; generated or opened from an independent seed domain after the model and code are frozen.
- Evidence that was not run or is not applicable must be explicitly marked `not_run` or `not_applicable`; `valid: true`, `passed`, or equivalent states must not represent unrun confirmation/final evidence.
- The 24-Episode bank repeatedly used in R0001–R0018, whose outcomes are exposed, may only be used as a development set.
- Do not swap seeds, delete failed Episodes, or rename seen samples to fill cells, increase power, or meet a threshold.

A safety rejection, empty candidate set, or ordinary task failure must enter the ledger as a failure for that Episode. Unless infrastructure damage, data contamination, or evaluation leakage occurs, one failure must not automatically invalidate the Episode or stop the entire cohort. Final evaluation and hardware execution must not weaken safety constraints; simulation development may use a separate diagnostic configuration, but it must be accounted for separately from the final safety configuration and cannot support a claim of capability improvement.

## 5. Hard Rules Against Research Loops

1. A capability hypothesis may add at most one layer of prerequisite diagnosis.
   - If the diagnosis itself requires a new oracle, contract, lineage, or qualification gate, combine it into one minimal validation or abandon the route; do not continue a dependency chain like P79→P80→P83→P87→P88.
2. Two consecutive purely diagnostic rounds are not allowed.
   - In principle, every round must include at least one real `physics behavior run`: an action, contact event, task, or learning-policy closed loop.
   - The only exception is an explicit infrastructure blocker; its repair must remain in the same round and must not create a new round merely to give an impression of progress.
3. The resource target is at least 70% for behavior/training experiments, at most 20% for diagnosis, and at most 10% for evaluation infrastructure. If a round deviates, `03-summary.md` must explain why and provide a recovery plan.
4. After two consecutive **discriminating candidates** fail on the same bottleneck, compare another technical route; do not keep refining the same artifact, threshold, mask, hash, or evaluator. A discriminating candidate must actually execute at least one predeclared differentiated mechanism in normal `physics`; failure before reaching that mechanism is only implementation-readiness failure and does not count as route comparison.
5. The research loop must stop after three consecutive **countable rounds** without capability-ladder progress. A round is countable only when the candidate's differentiated mechanism has entered physics behavior and the result is sufficient for the predeclared route judgment:
   - Archived rounds under the old rules, `invalid` rounds, and purely infrastructural/diagnostic rounds are not counted;
   - `abandoned` is counted only when the differentiated mechanism was actually executed and the budget was sufficient to produce route evidence;
   - Multiple runs on the same seed after code changes are implementation iterations, not independent candidates, repeated evidence, or multiple rounds;
   - If the round explicitly says that no claim may be made and forbids using the result to judge or reject the route, it has no route-level evidence and must not simultaneously be counted as a no-progress round.
   After three countable rounds, report to the user:
   - the routes attempted;
   - the failure evidence;
   - whether to narrow the task, introduce demonstrations/pretraining, change the architecture, or terminate the project.
   Do not automatically create a fourth round without the user's explicit decision.
6. Do not automatically start the next session after a round ends. The next round must be explicitly started by the user.
7. `validated_infrastructure`, `accepted_measurement`, passing tests, lower loss, and better-looking trajectories are not capability improvement and cannot advance the capability ladder.

## 6. Proposal and Experiment Selection

The main Agent selects only one primary capability bottleneck per round. Rank candidates in the following order rather than by mechanical summation:

1. Does it directly change the probability of success at the next capability level?
2. Can one experiment distinguish two important routes?
3. Is there a simpler and stronger control baseline?
4. Is it executable within the current data and compute budget?
5. Will failure still substantially narrow the decision space?

Before using an end-to-end failure to reject a route or locate a system bottleneck, a candidate must implement and attempt the major stages required by the success state machine. If a candidate covers only sub-stages such as grasping, contact, or transport, the experiment must be predeclared as a subgoal and its conclusion may cover only that subgoal; do not attribute unimplemented later stages to the environment, robot, or entire technical route. This rule does not restrict the form of a `development` teacher; scripts, planning, manual actions, teleoperation, demonstrations, and exploratory tuning remain allowed.

`01-experiment.md` must state both the candidate's “differentiated mechanism” and the minimum behavior entry condition required for it to begin operating. Class fields, state-machine branches, unreached code, and static geometric solutions do not count as mechanism execution; there must be a record of a real action, dynamics, contact, or policy closed loop. If the run fails before the entry condition, the conclusion applies only to the current implementation and cannot be used to reject the main hypothesis, switch the entire route, or accumulate a no-progress round.

Behavior entry must include at least one action generated by the differentiated mechanism entering `physics` through the formal interface, with the post-action state observed. Merely satisfying a prerequisite contact gate, switching to a stage, declaring a later branch implemented, or immediately losing the prerequisite after the switch does not count as execution of the differentiated mechanism. If the differentiated mechanism is in `lift/transport`, entry cannot be defined only as entering `secure`; if the round tests only an acquire subgoal, explicitly narrow the mechanism and conclusion to acquire and do not use it simultaneously to accumulate a no-progress round for the full route.

When the capability baseline is 0 success:

- First establish a privileged oracle/teacher ceiling to show that the environment, robot, control horizon, and safety contract are solvable;
- then establish a state policy to isolate action representation, optimization, and control problems;
- then introduce vision, language, multitask settings, and OOD;
- single tasks, fixed layouts, curricula, demonstrations, behavior cloning, and pretraining are allowed as development or training methods;
- any route that “learns a long-horizon bimanual sparse-reward task directly from random actions” must be compared with the simple baselines above and cannot be the only formal route.

Evaluation repair and capability improvement must not be placed in the same causal comparison, but evaluation repair should not automatically consume a new research round. Source hashes, AST completeness, encrypted fixtures, or whole-program provenance unrelated to the primary metric may become hard gates only when a concrete leakage path exists and would change the experimental conclusion.

## 7. Agent Organization

- Main Agent: Maintains the capability ladder, current baseline, experiment selection, implementation integration, runs, and final decisions.
- Innovation Agent: Enabled only when there are at least two credible routes, normally 1–2 agents; independently proposes executable hypotheses.
- Red-team Agent: One agent is enabled for experiments that change behavior or carry confirmation claims; it focuses on leakage, unattributable changes, safety bypasses, and statistical errors.
- Implementation Agent: The sole owner for each candidate, with explicit file ownership; it must not expand to an unapproved second primary variable.
- Do not routinely launch 3 Innovation Agents, 2 screening Agents, and a cleanup Agent for every small fix. The number of agents must match decision value and must not make coordination cost exceed the experiment itself.

The main Agent must personally verify key evidence, the final diff, run commands, and results; it must not merely concatenate sub-Agent conclusions.

## 8. Per-Round Documents and Historical Archives

New rounds use four-digit incrementing directories: `docs/research-loop/<NNNN>/`. Create a directory only for a new capability hypothesis or when the user explicitly requests a new round; keep diagnosis, repairs, and reruns for the same hypothesis in the same round.

If the user explicitly requests continuation of a prematurely ended round whose main hypothesis has not changed, reopen the original directory, record the reasoned budget or stopping-condition revision in `01-experiment.md`, and preserve existing results without rewriting them; do not create a new number merely because the directory was marked “finished.”

Each round requires only four short documents:

- `00-context.md`: Starting commit, current capability level, shortest board, and development/confirmation/final data boundaries;
- `01-experiment.md`: One main hypothesis, controls, metrics, guards, seeds, budget, stopping conditions, and commands;
- `02-results.md`: All runs, failures, anomalies, resources, and an index of raw artifacts;
- `03-summary.md`: Whether the capability level changed, conclusion, retained/reverted items, and whether the next round is allowed.

For every supplemental probe that enters `03-summary.md`, affects bottleneck judgment, or changes a route decision, `02-results.md` must record its command, seed, key configuration, and raw-artifact index. Exploration that cannot preserve a minimal review record may only be labeled an “unarchived observation” and cannot support a formal conclusion.

Optional proposals or review materials go in `notes/`; no fixed number is required. Documents should preferably be in Chinese and use stable IDs, but do not keep adding infinite suffixes to finished old ideas instead of creating a new capability hypothesis.

`docs/research-loop/0001/`–`0018/` are archived in place with paths and bytes kept stable because historical evaluations are bound to these Git trees. See `docs/research-loop/archive/legacy-evidence-loop-0001-0018.md` for the archive index. New rounds must not modify these directories or continue copying their format as a mandatory template.

## 9. Implementation, Training, and Runs

- Behavioral changes must have tests proportionate to their risk; do not build a large framework for pure documentation or one-off analysis.
- Candidates and baselines use the same seeds, task definition, physics, safety configuration, and comparable interaction/compute budget.
- Record the source commit, command, configuration, random seeds, data source, and output directory before training; confirmation runs start from a clean, committed commit.
- Implement and validate a new multistage controller incrementally from the earliest failed stage; do not spend primary implementation resources on later state-machine stages that cannot execute before the grasp/contact behavior entry condition is met.
- Before freezing a debug implementation as a candidate, behavior entry must cover control continuity at the relevant stage handoff: execute at least one differentiated-mechanism action and record its successor observation. If a stage label is reached only at the terminal step of the last smoke, or a complete Episode immediately returns or loses contact before its first subsequent action, continue to treat this as an implementation-readiness issue rather than starting or ending the candidate-discrimination budget.
- Every new candidate first receives a bounded implementation/debug budget and then a candidate-discrimination budget. Bound the debug budget jointly by wall-clock time, compute, and maximum probe count; short `physics` smokes are allowed, but a few full Episodes lasting only seconds must not substitute for implementation readiness. Candidate-discrimination budget starts when the differentiated mechanism first meets its behavior entry condition. It may end immediately once sufficient evidence is obtained; there is no minimum research duration.
- A rerun on the same seed after code changes must be labeled an implementation iteration. Preserve a commit, patch, or minimal source hash for versions that affect the conclusion; unversioned intermediate runs are only unarchived debugging observations and cannot pose as independent repeated evidence.
- Run the smallest discriminating experiment first; expand data and compute only after the preset escalation condition is met.
- Long training may use tmux/host scheduling and a watchdog; the watchdog only keeps the same run alive and checks its state. It must not automatically create a new research round, change thresholds, or start a different candidate.
- Stop and retain failure evidence when training diverges, metrics are clearly hopeless, or the frozen budget is exceeded; do not restart the same run.
- Do not aim to keep all local compute permanently busy; utilization follows experiment value and attribution.

The recommended baseline order is: privileged teacher → state BC → visual BC/sequence policy → world-model/RL challenger. More complex methods become the mainline only after outperforming simpler baselines under the same data and compute budget.

## 10. Conclusion Vocabulary

Each candidate may use only one of the following:

- `accepted_capability`: Frozen closed-loop primary metrics improve with no unacceptable guard-metric regression, advancing the capability ladder;
- `rejected_capability`: The capability hypothesis is rejected or its net benefit is negative;
- `inconclusive_capability`: The `capability` experiment is insufficient to decide because of power or infrastructure problems; not used for pure `development` experiments;
- `validated_development`: The predeclared development/measurement/infrastructure evidence holds; the corresponding development milestone advances only when all preset minimum evidence for `L0` or `L1` is met, and it cannot become a deployable capability baseline;
- `invalid`: Leakage, contract errors, data contamination, lack of attribution, or implementation deviation invalidates the experiment;
- `abandoned`: The candidate is actively terminated because of insufficient value, recursive dependencies, or a route change.

Only `accepted_capability` can become a new deployable capability baseline; `validated_development` can advance `L0` and `L1` development milestones only under the conditions above. Every conclusion must state both “allowed claims” and “disallowed claims.”

## 11. Wrap-Up and Stopping

At the end of a round:

1. Update the four round documents and the current status in `docs/research-loop/README.md`;
2. Commit the implementation, configuration, and result index; large reconstructible artifacts should not all be added to Git merely for audit convenience;
3. Retain the current baseline, unique raw data, irreproducible results, the latest recoverable checkpoint, and confirmation-evaluation evidence;
4. Report whether the capability ladder advanced and the next recommendation, but do not execute the next round automatically;
5. If the round had no `physics behavior run`, or if three countable rounds have had no capability progress under Section 5, stop and request the user's decision.

## 12. New-Session Continuity

After receiving “read AGENTS.md and start a new round,” the Agent must inspect the workspace and archive index in the same turn and begin verifiable work; it must not merely reply with a plan. However, a new Agent must first confirm that the user explicitly specified a new round; an old session must not derive a new session on its own.

At startup, first read `docs/research-loop/README.md` and the latest round summary, and enumerate existing four-digit directories; the new directory is the next number after the current maximum, with no round number hard-coded in the long-term rules. Choose one objective from the current capability ladder and latest failure evidence. As long as the current status remains `L0 not passed`, by default continue seeking the shortest complete privileged teacher/oracle ceiling, but do not treat a subgoal result that does not cover the complete success state machine as an `L0` pass, and do not repeat an already failed candidate unchanged.
