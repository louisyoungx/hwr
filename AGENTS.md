# Repository Engineering Rules

## Architecture first

- Define a module's responsibility and dependency direction before adding it.
- Core schemas must not import simulation, training, evaluation, applications, or hardware adapters.
- Third-party engines and device SDKs belong behind adapters.
- Simulation and real hardware must share the same runtime contracts.
- Safety filtering must remain independent from learned policies.

## Python size limits

- A Python file must not exceed 800 physical lines.
- A Python function, async function, or method must not exceed 200 physical lines.
- Split code by responsibility before approaching either limit.
- Run `python3 scripts/check_python_size.py` before committing.

## Verification and commits

- Add or update tests for every behavior change.
- Run the relevant tests and the Python size check before each commit.
- Commit each independently verifiable stage separately.

