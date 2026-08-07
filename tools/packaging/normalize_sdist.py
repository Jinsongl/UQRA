"""Normalize sdist tar and gzip timestamps without changing member contents."""
from __future__ import annotations

import argparse
import gzip
import io
import os
from pathlib import Path
import tarfile
import tempfile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    arguments = parser.parse_args(argv)

    output = io.BytesIO()
    with tarfile.open(arguments.sdist, "r:gz") as source:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as target:
            for member in source.getmembers():
                member.mtime = arguments.epoch
                for key in ("mtime", "atime", "ctime"):
                    member.pax_headers.pop(key, None)
                stream = source.extractfile(member) if member.isfile() else None
                target.addfile(member, stream)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=arguments.sdist.parent, prefix=arguments.sdist.name + ".", delete=False
        ) as raw:
            temporary_path = Path(raw.name)
            with gzip.GzipFile(
                fileobj=raw, mode="wb", filename="", mtime=arguments.epoch
            ) as archive:
                archive.write(output.getvalue())
        os.replace(temporary_path, arguments.sdist)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
