# UQRA v0.3.0 release notes

Status: published 2026-08-06

`v0.3.0` is a backward-compatible software delivery release. It packages the
M2 benchmark work and the complete M3 contract, provenance, packaging, and
Windows/Python 3.12 quality gates accumulated after `v0.2.0`.

## Highlights

- static benchmark registry and config v2;
- deterministic reduced FourBranch, Ishigami, and Gayton software benchmarks;
- runner manifest v2 with Git/source-tree/environment provenance;
- byte-level candidate/test/reference, trace, result, and output-summary identity;
- distinct raw and cross-platform contract trace hashes;
- Draft 2020-12 validation for published config, manifest, and trace schemas;
- modern `pyproject.toml` metadata and `uqra.__version__` as the single version source;
- wheel and sdist delivery of all five published JSON schemas;
- repository-outside clean-install acceptance for wheel and sdist;
- UQRA-owned Python 3.12 warning regression gate;
- Windows with Python 3.12 as the sole formal, continuously validated environment.

## Compatibility

- Formal validation: Windows and CPython 3.12.
- CPython 3.11 remains installable but is not continuously validated and is not
  a release completion gate.
- Existing config v1 and manifest v1 artifacts remain supported; published old
  manifests are not silently rewritten.
- The manifest v2 `environment.uqra` version field is additive and optional in
  the schema so existing manifest v2 evidence remains valid.
- No mathematical contract change was made to `uqra/adaptive/` algorithms.

## Security and build tooling

Release preparation upgrades pytest, setuptools, and wheel to versions that
resolve the known advisories reported against the previous locked tooling.
The Python 3.12 lock is audited with `pip-audit`; the final result is recorded in
`UQRA_V0.3.0_SECURITY_AUDIT.json`.

## Claim boundaries

The included FourBranch, Ishigami, and Gayton runs are reduced software
benchmarks with `historical_replay: false` and `paper_production: false`. They
are not historical replay, paper-production results, or scientific reproduction.

## Release completion checklist

- [x] M3 scope and evidence complete.
- [x] Version source changed to `0.3.0`.
- [x] Packaging scripts generalized beyond `0.2.0`.
- [x] Vulnerable test/build tooling pins upgraded.
- [x] Final wheel and sdist rebuilt from release-preparation commit `2da9847`.
- [x] Final distribution hashes and clean-install evidence frozen.
- [x] Windows/Python 3.12 required gate passes on the release-preparation PR (`31068171070`).
- [x] Release-preparation PR merged to `master` (`7c2bb050dc3e02882929811b5dd9c8878d17e7d5`).
- [x] Annotated `v0.3.0` tag created from the approved merge commit.
- [x] [GitHub Release](https://github.com/Jinsongl/UQRA/releases/tag/v0.3.0) created with the frozen artifacts and these notes.
