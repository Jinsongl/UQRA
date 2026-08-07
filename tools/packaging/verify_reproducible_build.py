"""Compare two distribution directories at the byte level."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def identity(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def distributions(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.whl")) + sorted(directory.glob("*.tar.gz"))
    if len(paths) != 2:
        raise RuntimeError(f"expected one wheel and one sdist in {directory}, found {paths}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-date-epoch", required=True, type=int)
    arguments = parser.parse_args(argv)

    first = [identity(path) for path in distributions(arguments.first)]
    second = [identity(path) for path in distributions(arguments.second)]
    if first != second:
        raise RuntimeError(f"distribution builds differ: first={first!r}, second={second!r}")

    report = {
        "schema": "uqra.packaging.reproducible-build/v1",
        "source_commit": arguments.source_commit,
        "source_date_epoch": arguments.source_date_epoch,
        "artifacts": first,
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
