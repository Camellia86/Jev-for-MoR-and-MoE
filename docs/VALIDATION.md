# Release validation — 2026-10-11

| Environment | Result | Evidence scope |
| --- | --- | --- |
| Python 3.12.14, Windows, no PyTorch | 27 discovered: 22 passed, 5 optional PyTorch checks skipped, zero failures/errors | Supported-version package and core routing/protocol checks |
| Python 3.9.12 with PyTorch 2.8.0 CPU | 27 passed, zero skipped/failures/errors | Supplementary source-level residual execution and gradient checks; not a claim of supported Canopy loading on Python 3.9 |
| Python 3.12 editable installation | Passed with preinstalled build tools and no build isolation | Packaging/import without optional model dependencies |
| Controlled probability replay | Passed; two targets, cost 3, math 2 steps and code 1 step | Fixture-driven plan compilation, not a model prediction |

The machine-readable receipts include actual Python versions, timestamps and
test outcomes: [core](../scripts/validation_receipt.json) and
[supplementary CPU PyTorch](../scripts/torch_cpu_validation_receipt.json).

Coverage includes exact resource allocation against brute force, capacity
charged per recursive expert step, negative/zero proxy scores selecting skip,
custom skips and targets with no compute offers, causal route invariance under
changed future tokens/scores, prefix-only adaptive generators/executors,
absorbing stops, multi-step execution observations, invalid probability
rejection, full-state request splitting, overlength rejection, the ten-candidate
Choice boundary, fourteen independent action Score questions, complete-plan
budget filtering, selected-only residual execution, teacher detachment and
student gradients, and causal distillation provenance.

Reproduce the lightweight checks with Python 3.10–3.12:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/offline_replay.py
```

Install `.[torch]` to exercise the optional tensor and gradient checks. The
receipt script can record a fresh verification with
`python scripts/validation_receipt.py --output your_receipt.json`.

Actual Canopy-Jev 27B inference was not run on the CPU validation host. There
are no new MoE/MoR training results, independent-seed quality measurements,
latency claims, or trained student weights in this release. The real Canopy
example and source-pinned adapter are supplied for its documented GPU runtime.
