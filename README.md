# Uncertainty Quantification for Risk Analysis (UQRA)

UQRA is a Python package for uncertainty quantification and risk analysis. This
repository combines the canonical UT Austin/MUSE implementation with the
audited adaptive sparse polynomial chaos expansion (PCE) compatibility work.

## Supported environments

- CPython 3.11
- CPython 3.12

Install a locked validation environment and the package in editable mode:

```bash
python -m pip install -r requirements/compatibility-py311.txt
python -m pip install --no-deps -e .
```

Use `requirements/compatibility-py312.txt` for Python 3.12.

## Validation

Run the compatibility gate locally:

```bash
python -m pytest tests/compatibility -q
```

Reproduce the deterministic benchmark and frozen publication manifest:

```bash
python -m uqra.adaptive.benchmark --scenario all --output artifacts/adaptive_phase8_manifest.json
python -m uqra.adaptive.publication --output artifacts/adaptive_phase11_publication_manifest.json
```

Run the versioned, configuration-driven reduced benchmark entry point:

```bash
python -m uqra.adaptive.run --config examples/configs/adaptive_reduced_smoke.json
python -m uqra.adaptive.run --config examples/configs/adaptive_reduced_full.json
```

The v1 examples above remain supported for the `v0.2.0` delivery contract. New
benchmark configurations use the v2 static registry contract:

```bash
python -m uqra.adaptive.run --config examples/configs/adaptive_registry_v2_smoke.json
python -m uqra.adaptive.run --config examples/configs/adaptive_registry_v2_full.json
```

The v2 `runner.benchmark` value must be a name published by UQRA's static
registry. Python module paths and configuration-driven imports are not accepted.

The v1/v2 configuration, runner-manifest, and trace contracts are published under
[`schemas/`](schemas/). This entry point intentionally accepts only
`purpose: software_benchmark` and `scale: reduced`; formal paper-production
experiments belong to the independent paper repository.

The algorithm decisions, phased audit summaries, frozen protocol, and known
historical-data limits are documented in [`specs/`](specs/README.md).

## Scope of the publication evidence

The frozen Phase 11 suite verifies the modern compatible implementation against
an independent portable Hermite evaluator on deterministic two-dimensional
inputs. Historical canonical FourBranch replay remains unavailable because the
original candidate/test inputs and RNG state were not recovered. See the Phase
9 and Phase 11 summaries before citing reproduction claims.

## License

Copyright (c) 2019, Jinsong Liu. Distributed under the BSD 3-Clause License; see
[`LICENSE`](LICENSE).
