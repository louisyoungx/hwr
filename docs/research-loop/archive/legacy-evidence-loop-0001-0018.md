# Archive of the Old Evidence-First Research Loop: R0001–R0018

## Archival Decision

- Archive date: 2026-08-31.
- Scope: `docs/research-loop/0001/`–`0018/` and their corresponding historical runs, commits, and experiment branches.
- Method: in-place read-only archive; do not move directories or rewrite R0001–R0017 contents.
- Reason: multiple historical evaluators, tests, and artifacts use the original paths, commit ancestors, and Git tree hashes as provenance; physically moving them would break reproducibility without adding research value.
- R0018 was an incomplete round when the mechanism was reset; only its results and summary were updated to record the intentional termination.

## Overall Conclusion

The old loop established substantial evaluation, measurement, provenance, and local-correctness evidence, but did not advance the capability baseline:

- the latest complete 3D world-model baseline remains 24 Episodes, 1,600 updates, and 0 success;
- the Actor was not unlocked;
- none of the three formal tasks passed based on a new closed-loop capability result;
- R0004–R0017 had no formal capability training, and R0018 had no formal physics/capability run either;
- historical `accepted as ...` statuses primarily mean that an evaluator, measurement, contract, or local fix was established; they must not be reinterpreted as `accepted_capability`.

Therefore, the old loop's final capability determination is: `L0 not passed`.

## Round Index

| Round | Main content | Archival interpretation |
|---|---|---|
| R0001–R0003 | World-model negative baseline, action causality, and model diagnostics | Preserve negative-baseline and plant-action causal evidence; no capability success |
| R0004–R0008 | Latency, safety, evaluation leakage, replay information, and contact ledger | Mostly measurement/evaluation contracts; not part of the capability baseline |
| R0009–R0013 | Coordinates, FK, candidate acquisition, reachability, interaction contract, and safety witness | Provide evidence of control-chain and task-expression defects; no training success |
| R0014–R0016 | v2 candidate fixes and artifact/selection lineage | Local correctness established, but the default runtime was not migrated and capability was unchanged |
| R0017–R0018 | Contract oracle and coordinate-oracle qualification chains | Had entered recursive prerequisite proof; intentionally terminated and the research mechanism was switched |

## Development Evidence Worth Retaining

The following evidence may inform the design of new routes, but the boundaries of the original conclusions must be respected:

- `R0001-P17`: plant actions in formal-task data have physical causal effects.
- `R0001-P40-E1/E2`: contact and entity-contact-graph measurements can serve as development diagnostics.
- `R0001-P51/P52`: Cartesian-frame fixes and policy FK/MuJoCo site agreement.
- `R0001-P57`: fixed-cohort dual-arm pre-contact command support is clearly insufficient; the current B2/B3/B4 horizons cannot serve as a feasible-control upper bound.
- `R0001-P61/P72`: generic candidate-centered primitives lack the information needed to express complete long-horizon tasks.
- `R0001-P66-E1`: one predicted-safety-rejection witness on a legacy path can inform the design of a safety-preserving teacher/controller, but cannot be extrapolated into overall safety capability.
- `R0001-P79/P83`: local evidence for an isolated v2 candidate and selection lineage, for development and historical reconstruction only; it does not automatically authorize a selector, training, or capability conclusion.

## Materials Downgraded to Development-Only

- The 24-Episode bank repeatedly used by R0001–R0018 and its derived candidates, selections, and prefix outcomes;
- sentinels whose outcomes have been exposed, public salts, fixed thresholds, and historical cohorts;
- source hashes, ASTs, encrypted fixtures, contract oracles, and lineage tools built for old evaluators;
- superseded and failed artifacts.

These materials may still be used for regression, debugging, and a counterexample library, but they must no longer be called unseen confirmation evidence.

## R0018 and P88 Preservation Status

R0018 was intentionally terminated before formal execution because the research mechanism was reset:

- frozen main-branch commit: `39f95f606659bf31eaccbc7b235060503a4ad5ad`;
- experiment branch: `exp/R0001-P88-E1-coordinate-oracle`;
- submitted implementation: `f9f417ea346755fcc60d0bc8332b42452b175b49`;
- worktree: `/Users/louis/Developer/AIWorkspace/50-housework-robot-r0018-p88`;
- at the archival check, the worktree still had 4 modified files, approximately `+47/-21`, uncommitted;
- the formal evaluator was not run, no formal artifact existed, and P88 had no scientific conclusion.

Do not delete, merge, or clean up the branch and dirty worktree above for now; await a separate user decision. They do not constitute a new baseline and should not block R0019's L0 capability reset.

## Inheritance Boundaries for the New Loop

The new loop inherits only:

1. the current real capability baseline is 0 success;
2. normal physics, the success state machine, and the independent safety layer must not be weakened in final capability evaluation;
3. the control-feasibility and task-expression gaps indicated by P57/P61 merit priority verification;
4. historical failures and counterexamples must not consume resources again.

The new loop does not inherit old-candidate approvals, the six-document template, a fixed multi-Agent roster, automatic next-round creation, or an expert-free/curriculum-free doctrine. It also does not inherit the requirement to continue completing the P88/P76/P68 prerequisite chain before action experiments.
