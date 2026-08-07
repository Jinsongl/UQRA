"""Validate and verify the controlled UQRA GitHub Release contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


VERSION_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git", "-c", f"safe.directory={repository.resolve().as_posix()}",
            "-C", str(repository), *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8").strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_version(version: str) -> str:
    if not VERSION_PATTERN.fullmatch(version):
        raise RuntimeError(f"release version must be X.Y.Z: {version}")
    return f"v{version}"


def preflight(
    repository: Path, version: str, source_commit: str, expected_branch: str
) -> dict:
    tag = validate_version(version)
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise RuntimeError("source commit must be a full lowercase 40-character SHA")
    resolved = git(repository, "rev-parse", f"{source_commit}^{{commit}}")
    if resolved != source_commit:
        raise RuntimeError(f"source commit did not resolve exactly: {resolved}")
    head = git(repository, "rev-parse", "HEAD")
    if head != source_commit:
        raise RuntimeError(f"checked out HEAD differs from approved commit: {head}")
    version_text = git(repository, "show", f"{source_commit}:uqra/_version.py")
    if f'__version__ = "{version}"' not in version_text:
        raise RuntimeError("requested version differs from the committed package version")
    ancestor = subprocess.run(
        ["git", "-c", f"safe.directory={repository.resolve().as_posix()}",
         "-C", str(repository), "merge-base", "--is-ancestor", source_commit, expected_branch],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"source commit is not contained in {expected_branch}")
    tag_check = subprocess.run(
        ["git", "-c", f"safe.directory={repository.resolve().as_posix()}",
         "-C", str(repository), "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"],
        check=False,
    )
    if tag_check.returncode == 0:
        raise RuntimeError(f"release tag already exists: {tag}")
    if tag_check.returncode not in (0, 1):
        raise RuntimeError(f"unable to check release tag: {tag}")
    return {
        "schema": "uqra.release.preflight/v1",
        "version": version,
        "tag": tag,
        "source_commit": source_commit,
        "expected_branch": expected_branch,
        "status": "passed",
    }


def candidate(preflight_report: dict, build_report: dict, dist_dir: Path) -> dict:
    if preflight_report["source_commit"] != build_report["source_commit"]:
        raise RuntimeError("preflight and build evidence commits differ")
    expected = sorted(build_report["artifacts"], key=lambda item: item["name"])
    actual = []
    for identity in expected:
        path = dist_dir / identity["name"]
        if not path.is_file():
            raise RuntimeError(f"release artifact is missing: {path.name}")
        observed = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if observed != identity:
            raise RuntimeError(f"release artifact differs from build evidence: {path.name}")
        actual.append(observed)
    return {
        "schema": "uqra.release.candidate/v1",
        "version": preflight_report["version"],
        "tag": preflight_report["tag"],
        "source_commit": preflight_report["source_commit"],
        "artifacts": actual,
        "status": "approved_candidate",
    }


def verify_download(candidate_report: dict, directory: Path) -> dict:
    expected = sorted(candidate_report["artifacts"], key=lambda item: item["name"])
    expected_names = {item["name"] for item in expected}
    actual_paths = sorted(path for path in directory.iterdir() if path.is_file())
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        raise RuntimeError(
            f"downloaded release asset set differs: expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    actual = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in actual_paths
    ]
    if actual != expected:
        raise RuntimeError("downloaded release assets differ from the approved candidate")
    return {
        "schema": "uqra.release.download-readback/v1",
        "version": candidate_report["version"],
        "tag": candidate_report["tag"],
        "source_commit": candidate_report["source_commit"],
        "artifacts": actual,
        "status": "passed",
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--repository", required=True, type=Path)
    preflight_parser.add_argument("--version", required=True)
    preflight_parser.add_argument("--source-commit", required=True)
    preflight_parser.add_argument("--expected-branch", default="origin/master")
    preflight_parser.add_argument("--output", required=True, type=Path)

    candidate_parser = subparsers.add_parser("candidate")
    candidate_parser.add_argument("--preflight", required=True, type=Path)
    candidate_parser.add_argument("--build-evidence", required=True, type=Path)
    candidate_parser.add_argument("--dist-dir", required=True, type=Path)
    candidate_parser.add_argument("--output", required=True, type=Path)

    verify_parser = subparsers.add_parser("verify-download")
    verify_parser.add_argument("--candidate", required=True, type=Path)
    verify_parser.add_argument("--download-dir", required=True, type=Path)
    verify_parser.add_argument("--output", required=True, type=Path)

    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        report = preflight(
            arguments.repository, arguments.version, arguments.source_commit,
            arguments.expected_branch,
        )
    elif arguments.command == "candidate":
        report = candidate(
            read_json(arguments.preflight), read_json(arguments.build_evidence),
            arguments.dist_dir,
        )
    else:
        report = verify_download(read_json(arguments.candidate), arguments.download_dir)
    write_json(arguments.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
