"""Verify an installed UQRA distribution outside the source repository."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import uqra
from uqra._version import __version__ as source_version


SCHEMA_NAMES = (
    "adaptive-runner-config.schema.json",
    "adaptive-runner-config-v2.schema.json",
    "adaptive-runner-manifest.schema.json",
    "adaptive-runner-manifest-v2.schema.json",
    "adaptive-trace.schema.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validators(schema_directory: Path):
    schemas = {
        name: json.loads((schema_directory / name).read_text(encoding="utf-8"))
        for name in SCHEMA_NAMES
    }
    registry = Registry().with_resources(
        [(schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()]
    )
    return {
        name: Draft202012Validator(schema, registry=registry)
        for name, schema in schemas.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--install-kind", required=True, choices=("wheel", "sdist"))
    arguments = parser.parse_args(argv)

    artifact = arguments.source_artifact.resolve()
    evidence = arguments.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    schema_directory = Path(sys.prefix) / "schemas"
    missing = [name for name in SCHEMA_NAMES if not (schema_directory / name).is_file()]
    if missing:
        raise RuntimeError(f"installed schemas are missing: {', '.join(missing)}")

    installed_version = distribution_version("uqra")
    if not uqra.__version__ == source_version == installed_version:
        raise RuntimeError("runtime, source, and distribution versions differ")

    runner = Path(sys.prefix) / "Scripts" / "uqra-adaptive-runner.exe"
    version_output = subprocess.run(
        [str(runner), "--version"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    if version_output != f"uqra-adaptive-runner {installed_version}":
        raise RuntimeError(f"unexpected runner version: {version_output!r}")

    manifest_path = evidence / "runner-manifest.json"
    config_path = evidence / "runner-config.json"
    config = {
        "schema": "uqra.adaptive.runner-config/v1",
        "purpose": "software_benchmark",
        "scale": "reduced",
        "runner": {
            "kind": "deterministic_benchmark",
            "benchmark": "phase8_two_dimensional_hermite",
            "scenarios": ["converged"],
        },
        "output": {"manifest": manifest_path.as_posix()},
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [str(runner), "--config", str(config_path), "--output", str(manifest_path)],
        cwd=evidence, check=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = validators(schema_directory)
    published["adaptive-runner-config.schema.json"].validate(config)
    published["adaptive-runner-manifest-v2.schema.json"].validate(manifest)
    for scenario in manifest["run"]["scenarios"].values():
        for row in scenario["trace"]:
            published["adaptive-trace.schema.json"].validate(row)
    if manifest["provenance"]["environment"]["uqra"] != installed_version:
        raise RuntimeError("manifest version differs from installed version")

    identities = []
    artifacts = manifest["artifacts"]
    for collection in ("inputs", "traces", "results"):
        identities.extend(artifacts[collection].values())
    identities.append(artifacts["output_summary"])
    for identity in identities:
        path = manifest_path.parent / identity["path"]
        if path.stat().st_size != identity["size_bytes"] or sha256(path) != identity["sha256"]:
            raise RuntimeError(f"artifact identity mismatch: {path}")

    report = {
        "schema": "uqra.packaging.clean-install-evidence/v1",
        "install_kind": arguments.install_kind,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "versions": {
            "uqra_runtime": uqra.__version__,
            "uqra_source": source_version,
            "uqra_distribution": installed_version,
            "uqra_cli": version_output.rsplit(" ", 1)[-1],
            "uqra_manifest": manifest["provenance"]["environment"]["uqra"],
        },
        "source_artifact": {
            "name": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "schemas": list(SCHEMA_NAMES),
        "manifest": {
            "path": manifest_path.name,
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
    }
    (evidence / "clean-install-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
