# Agent Instructions

- Implement the smallest complete change that satisfies the current requirement.
- Keep each change focused on one capability or bug.
- Inspect the relevant existing code path before editing.
- Do not perform unrelated refactors or dependency upgrades.
- Prefer clear data contracts; avoid abstractions for hypothetical future needs.
- Keep important research decisions observable.
- Do not silently swallow unexpected exceptions.
- Verify behavior with the narrowest meaningful test or smoke test.
- For bugs, reproduce before fixing.
- Once the requested behavior works and verification passes, stop making unrelated changes.
