# R0019 Results

## Run Ledger

### 1. Generic B0–B7 Baseline

Command:

```bash
MUJOCO_GL=glfw .venv/bin/python -m hwr.apps.evaluate_bimanual_teacher \
  --mode development --controller paired \
  --seed 19001 --seed 19002 --seed 19003 \
  --seed 19004 --seed 19005 --seed 19006 \
  --output runs/research-loop/0019/development/paired-seeds-19001-19006-v2.json
```

The baseline uses simulator-private payload geometry to construct a unique candidate, deliberately
removing P61's perception/role ambiguity; the remaining B0–B7 horizon and action logic are
unchanged. Results:

- `0/6` success;
- `0/6` Episodes produced synchronized bimanual contact;
- `0` safety intervention;
- `0` actual severe collision;
- all 6 Episodes ended with `bimanual_task_timeout`.

The stage endpoint for seed 19001 shows that B2/B3/B4 still did not reach contact: the minimum
left and right handle reach distances were `0.1138905511m` and `0.1167015719m`, respectively,
and the final payload target distance was `0.8968926532m`. The earliest failure stage was
`B3_contact_approach`, consistent with P57's command-support deficit; the distances increased
again after B5/B6.

### 2. Privileged Teacher

Implementation:

- Run fixed-budget CEM for each hand on a copied `MjData`, using the signed distances from the
  four finger pads to their corresponding handles as the objective;
- the planner does not write to the authoritative `MjData`;
- generate normal 16-dimensional Cartesian actions using the backend's inverse Jacobian/DLS
  mapping;
- the state machine is `approach_base → acquire → secure → transport_probe/failed_hold`;
- complete target-guided `transport`, `place`, `release`, or `stabilize` is not implemented;
- all actions pass through the original `DualArmSafetySupervisor` and predictive collision
  filter.

Results:

| seed | success | first bimanual contact | longest synchronized contact | earliest failure stage | safety | severe |
|---|---:|---:|---:|---|---:|---:|
| 19001 | 0 | 123 | 83 | `transport_contact_lost` | 0 | 0 |
| 19002 | 0 | none | 0 | `acquire_timeout` | 0 | 0 |
| 19003 | 0 | none | 0 | `acquire_timeout` | 0 | 0 |
| 19004 | 0 | none | 0 | `acquire_timeout` | 0 | 0 |
| 19005 | 0 | none | 0 | `acquire_timeout` | 0 | 0 |
| 19006 | 0 | none | 0 | `acquire_timeout` | 0 | 0 |

Summary:

- `0/6` success;
- `1/6` Episode produced actual synchronized bimanual contact;
- seed 19001 had left/right contact steps of `272/83`, with a maximum synchronized window of
  `83` steps;
- seed 19001's maximum controlled target progress was only `0.0035074962m`;
- among the remaining seeds, 19003 had left-hand contact only, and 19006 had right-hand contact
  only;
- all teacher Episodes had `0` safety intervention and `0` actual severe collision;
- the maximum observed forbidden force was `54.0003924N`, below the `220N` severe gate.

The exploration phase also observed longer contact and partial lifting, but these one-off probes
did not preserve the command, complete configuration, or raw artifacts. Under this round's
closure rules, they are recorded as "unarchived observations" and are not used for formal
bottleneck assessment or route decisions.

### 3. Confirmation-Set Decision

The 100-seed confirmation status is `not_run`, and its `valid` field is `null`. The current
teacher lacks `lift/target_transport/place/release/stabilize`; the runner rejects this
implementation before the first confirmation Episode. The development teacher also has `0/6`
success, so no successful qualification report from the same commit is available. No
confirmation seed was consumed or viewed.

Runner v2 also enforces the following constraints:

- confirmation must use the paired controller and frozen 100-seed domain;
- it must start from a clean worktree and provide a complete development qualification report
  from the same commit and source hash with at least 1 teacher success;
- the teacher must declare coverage of the complete task's major stages;
- existing confirmation output must not be overwritten;
- a wall-time or infrastructure error writes a partial report and invalidates confirmation;
- a single controller failure still counts only as that Episode's failure and does not interrupt
  the ordinary cohort;
- the `invalid` command exit code is `2` and must not be misinterpreted as a pass by automation.

### 4. Raw Artifacts

- Final paired development artifact (schema v2):
  `runs/research-loop/0019/development/paired-seeds-19001-19006-v2.json`
- SHA-256:
  `629629437ff262bb523ca1e61260393e440a586f82c42a7960c0c31063057c8c`
- artifact size: `61,276` bytes;
- runtime: `44.5654s`;
- `run_status.completed=true`; all 12/12 are valid Episode results, with no infrastructure
  failure;
- `decision=invalid`, `l0_gate_passed=false`;
- `confirmation_evidence.status=not_run`, `valid=null`;
- `implementation_evidence.valid=false`; missing
  `lift/target_transport/place/release/stabilize`；
- the qualification hash covers 19 directly dependent files, including the teacher/runner,
  task/backend/safety, action wrapper, task configuration, scene XML, and its shared robot XML;
- the artifact records runtime source commit `5102d1411c23e3465dc11bf8891bfbc8505a43a7` and
  `source_worktree_dirty=true`, so it is development-only and cannot serve as a future
  confirmation qualification report;
- the runner also writes a `.sha256` sidecar; `runs/` is managed by `.gitignore` and is not
  included in Git;
- the old v1 artifact's `decision=validated_development` and confirmation `valid=true` semantics
  were incorrect, have been superseded by v2, and must not be used for conclusions.

### 5. Verification

```text
48 passed in 8.09s
Python size check passed: 461 files, file <= 800 lines, function <= 200 lines
Architecture check passed: engine, foundation, and core boundaries are intact
```
