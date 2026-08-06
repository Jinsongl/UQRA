"""Create a byte-level manifest for built UQRA distributions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


EXPECTED_SCHEMAS = {
    "adaptive-runner-config.schema.json",
    "adaptive-runner-config-v2.schema.json",
    "adaptive-runner-manifest.schema.json",
    "adaptive-runner-manifest-v2.schema.json",
    "adaptive-trace.schema.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def members(path: Path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path, "r:gz") as archive:
        return archive.getnames()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args(argv)

    paths = sorted(arguments.dist_dir.glob("uqra-0.2.0-*.whl"))
    paths += sorted(arguments.dist_dir.glob("uqra-0.2.0.tar.gz"))
    if len(paths) != 2:
        raise RuntimeError(f"expected one wheel and one sdist, found: {[p.name for p in paths]}")
    artifacts = []
    for path in paths:
        schema_members = sorted(
            name for name in members(path)
            if Path(name).name in EXPECTED_SCHEMAS
        )
        if {Path(name).name for name in schema_members} != EXPECTED_SCHEMAS:
            raise RuntimeError(f"schema set is incomplete in {path.name}")
        artifacts.append({
            "name": path.name,
            "kind": "wheel" if path.suffix == ".whl" else "sdist",
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "schema_members": schema_members,
        })
    manifest = {
        "schema": "uqra.packaging.distribution-manifest/v1",
        "version": "0.2.0",
        "source_commit": arguments.source_commit,
        "artifacts": artifacts,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
