"""Create an isolated build source from a Git commit or a clean copied tree."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


IGNORED_NAMES = {
    ".git", ".conda", ".conda-pkgs", ".pytest_cache", ".venv-py311",
    "artifacts", "build", "dist", "__pycache__",
}


def from_git(source: Path, destination: Path, revision: str) -> None:
    tree = subprocess.run(
        ["git", "-c", f"safe.directory={source.resolve().as_posix()}", "-C", str(source),
         "ls-tree", "-r", "-z", "--full-tree", revision],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tree.returncode:
        raise RuntimeError(tree.stderr.decode("utf-8", errors="replace").strip())

    entries = []
    for record in tree.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, object_type, object_id = metadata.split(b" ", 2)
        if object_type != b"blob":
            continue
        relative = Path(raw_path.decode("utf-8"))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe path in Git tree: {relative}")
        entries.append((object_id, relative))

    batch = subprocess.run(
        ["git", "-c", f"safe.directory={source.resolve().as_posix()}", "-C", str(source),
         "cat-file", "--batch"],
        input=b"\n".join(object_id for object_id, _relative in entries) + b"\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if batch.returncode:
        raise RuntimeError(batch.stderr.decode("utf-8", errors="replace").strip())

    cursor = 0
    for expected_id, relative in entries:
        header_end = batch.stdout.index(b"\n", cursor)
        header = batch.stdout[cursor:header_end].split(b" ")
        if len(header) != 3 or header[0] != expected_id or header[1] != b"blob":
            raise RuntimeError(f"unexpected git cat-file response for {relative}")
        size = int(header[2])
        content_start = header_end + 1
        content_end = content_start + size
        blob = batch.stdout[content_start:content_end]
        if batch.stdout[content_end:content_end + 1] != b"\n":
            raise RuntimeError(f"truncated git cat-file response for {relative}")
        cursor = content_end + 1
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)


def from_tree(source: Path, destination: Path) -> None:
    def ignore(_directory, names):
        return [
            name for name in names
            if name in IGNORED_NAMES or name.endswith(".egg-info") or name.endswith(".pyc")
        ]

    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignore)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--revision")
    arguments = parser.parse_args(argv)

    arguments.destination.mkdir(parents=True, exist_ok=False)
    if arguments.revision:
        from_git(arguments.source, arguments.destination, arguments.revision)
    else:
        from_tree(arguments.source, arguments.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
