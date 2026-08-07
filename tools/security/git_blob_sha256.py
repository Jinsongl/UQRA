"""Bind security evidence to the exact bytes of a canonical Git blob."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git_output(repository: Path, *arguments: str) -> bytes:
    safe_directory = repository.resolve().as_posix()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", "-C", str(repository), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def read_blob(repository: Path, revision: str, path: str) -> bytes:
    return git_output(repository, "cat-file", "blob", f"{revision}:{path}")


def resolve_identity(repository: Path, revision: str, path: str) -> tuple[str, str]:
    commit = git_output(repository, "rev-parse", f"{revision}^{{commit}}").decode("ascii").strip()
    blob_oid = git_output(repository, "rev-parse", f"{revision}:{path}").decode("ascii").strip()
    return commit, blob_oid


def validate_expected(actual: str, expected: str | None) -> None:
    if expected is not None and actual.lower() != expected.lower():
        raise RuntimeError(f"Git blob SHA-256 mismatch: expected {expected}, actual {actual}")


def audit_identity(path: Path) -> tuple[str, str]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    identity = evidence.get("input") or evidence.get("lock_file")
    try:
        return identity["path"], identity["git_blob_sha256"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("evidence lacks lock path or git_blob_sha256") from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--audit-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    expected = arguments.expected_sha256
    if expected is None and arguments.audit_evidence is None:
        parser.error("one of --expected-sha256 or --audit-evidence is required")
    if arguments.audit_evidence:
        audit_path, audit_hash = audit_identity(arguments.audit_evidence)
        if audit_path != arguments.path:
            raise RuntimeError(f"audit input path mismatch: expected {arguments.path}, found {audit_path}")
        if expected is not None and expected.lower() != audit_hash.lower():
            raise RuntimeError("command-line and audit evidence SHA-256 values differ")
        expected = audit_hash

    source_commit, blob_oid = resolve_identity(
        arguments.repository, arguments.revision, arguments.path
    )
    blob = read_blob(arguments.repository, source_commit, arguments.path)
    digest = hashlib.sha256(blob).hexdigest()
    validate_expected(digest, expected)
    report = {
        "schema": "uqra.security.git-blob-identity/v1",
        "revision": arguments.revision,
        "source_commit": source_commit,
        "git_blob_oid": blob_oid,
        "path": arguments.path,
        "size_bytes": len(blob),
        "sha256": digest,
        "audit_evidence": str(arguments.audit_evidence) if arguments.audit_evidence else None,
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
