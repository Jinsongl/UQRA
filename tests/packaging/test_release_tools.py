from __future__ import annotations

import hashlib
import importlib.util
from datetime import date
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


git_blob = load_tool("git_blob_sha256", "tools/security/git_blob_sha256.py")
repro = load_tool("verify_reproducible_build", "tools/packaging/verify_reproducible_build.py")
prepare = load_tool("prepare_build_source", "tools/packaging/prepare_build_source.py")
release = load_tool("release_contract", "tools/release/release_contract.py")
security = load_tool("create_security_audit", "tools/security/create_security_audit.py")


def test_lock_identity_uses_committed_blob_bytes_not_worktree_newlines():
    blob = git_blob.read_blob(ROOT, "HEAD", "requirements/compatibility-py312.txt")
    assert hashlib.sha256(blob).hexdigest() == (
        "51a49cdfc1ea25732789b5a1f5bc474acd95ee976483375c988880cbea4a5f78"
    )


def test_wrong_git_blob_sha256_is_rejected():
    with pytest.raises(RuntimeError, match="Git blob SHA-256 mismatch"):
        git_blob.validate_expected("a" * 64, "b" * 64)


def test_frozen_security_audit_is_bound_to_lock_blob():
    audit_path, audit_hash = git_blob.audit_identity(
        ROOT / "specs/releases/UQRA_V0.3.0_SECURITY_AUDIT.json"
    )
    blob = git_blob.read_blob(ROOT, "HEAD", audit_path)
    git_blob.validate_expected(hashlib.sha256(blob).hexdigest(), audit_hash)


def test_reproducibility_comparison_rejects_changed_bytes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, payload in ((first, b"one"), (second, b"two")):
        (directory / "uqra-1-py3-none-any.whl").write_bytes(payload)
        (directory / "uqra-1.tar.gz").write_bytes(b"same")
    with pytest.raises(RuntimeError, match="distribution builds differ"):
        repro.main([
            "--first", str(first), "--second", str(second),
            "--source-commit", "abc", "--source-date-epoch", "315532800",
        ])


def test_build_source_copy_excludes_generated_state(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "package").mkdir(parents=True)
    (source / "build").mkdir()
    (source / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "build" / "stale.py").write_text("stale\n", encoding="utf-8")
    destination.mkdir()

    prepare.from_tree(source, destination)

    assert (destination / "package" / "module.py").is_file()
    assert not (destination / "build").exists()


def test_build_source_archive_uses_committed_bytes(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    prepare.from_git(ROOT, destination, "HEAD")
    committed = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
         "show", "HEAD:pyproject.toml"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert (destination / "pyproject.toml").read_bytes() == committed


def test_release_version_requires_strict_semver():
    assert release.validate_version("1.2.3") == "v1.2.3"
    for invalid in ("v1.2.3", "1.2", "1.2.3rc1", "01.2.3"):
        with pytest.raises(RuntimeError, match="must be X.Y.Z"):
            release.validate_version(invalid)


def test_release_preflight_accepts_master_commit_then_rejects_existing_tag(tmp_path):
    repository = tmp_path / "repository"
    (repository / "uqra").mkdir(parents=True)
    (repository / "uqra" / "_version.py").write_text(
        '__version__ = "1.2.3"\n', encoding="utf-8"
    )
    commands = [
        ("init", "-b", "master"),
        ("config", "user.name", "Release Test"),
        ("config", "user.email", "release-test@example.invalid"),
        ("add", "uqra/_version.py"),
        ("commit", "-m", "release source"),
    ]
    for command in commands:
        subprocess.run(["git", "-C", str(repository), *command], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("ascii").strip()

    report = release.preflight(repository, "1.2.3", commit, "master")
    assert report["tag"] == "v1.2.3"
    subprocess.run(
        ["git", "-C", str(repository), "tag", "-a", "v1.2.3", "-m", "existing"],
        check=True,
    )
    with pytest.raises(RuntimeError, match="tag already exists"):
        release.preflight(repository, "1.2.3", commit, "master")


def release_candidate_inputs(tmp_path):
    distribution = tmp_path / "distribution"
    distribution.mkdir()
    wheel = distribution / "uqra-1.2.3-py3-none-any.whl"
    sdist = distribution / "uqra-1.2.3.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    artifacts = [
        {"name": path.name, "size_bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted((wheel, sdist))
    ]
    preflight = {
        "version": "1.2.3", "tag": "v1.2.3", "source_commit": "a" * 40,
    }
    build = {"source_commit": "a" * 40, "artifacts": artifacts}
    return distribution, preflight, build


def test_release_candidate_and_download_readback_are_byte_bound(tmp_path):
    distribution, preflight, build = release_candidate_inputs(tmp_path)
    candidate = release.candidate(preflight, build, distribution)
    readback = tmp_path / "readback"
    readback.mkdir()
    for path in distribution.iterdir():
        (readback / path.name).write_bytes(path.read_bytes())
    report = release.verify_download(candidate, readback)
    assert report["status"] == "passed"
    assert report["artifacts"] == candidate["artifacts"]


def test_release_candidate_rejects_commit_mismatch(tmp_path):
    distribution, preflight, build = release_candidate_inputs(tmp_path)
    build["source_commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="commits differ"):
        release.candidate(preflight, build, distribution)


def test_release_download_rejects_changed_or_extra_assets(tmp_path):
    distribution, preflight, build = release_candidate_inputs(tmp_path)
    candidate = release.candidate(preflight, build, distribution)
    (distribution / candidate["artifacts"][0]["name"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differ from the approved candidate"):
        release.verify_download(candidate, distribution)
    (distribution / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="asset set differs"):
        release.verify_download(candidate, distribution)


def test_release_workflow_keeps_write_permission_behind_environment():
    workflow = (ROOT / ".github/workflows/controlled-release.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "environment: uqra-release" in workflow
    assert "if: ${{ inputs.dry_run == false }}" in workflow
    assert workflow.count("contents: write") == 1
    assert "group: uqra-release-${{ inputs.version }}" in workflow
    assert "uqra-release must require at least one reviewer" in workflow
    assert "gh release delete" in workflow
    assert "refusing to clean a non-draft or unverifiable Release" in workflow
    assert 'git push origin ":refs/tags/$TAG"' in workflow


def test_git_identity_resolves_immutable_commit_and_blob():
    commit, blob_oid = git_blob.resolve_identity(
        ROOT, "HEAD", "requirements/compatibility-py312.txt"
    )
    assert len(commit) == 40
    assert len(blob_oid) == 40


def security_inputs(tmp_path, vulnerability, policy_updates=None):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8")
    policy = {
        "dependency_roles": {"runtime": ["demo"], "test_build": []},
        "severity_overrides": {},
        "accepted_vulnerabilities": {},
    }
    if policy_updates:
        policy.update(policy_updates)
    audit = {"dependencies": [{"name": "demo", "version": "1", "vulns": [vulnerability]}]}
    blob = {
        "source_commit": "a" * 40,
        "path": "requirements/compatibility-py312.txt",
        "git_blob_oid": "b" * 40,
        "sha256": "c" * 64,
    }
    return audit, policy, blob, workflow


def test_unresolved_high_security_finding_fails_gate(tmp_path):
    audit, policy, blob, workflow = security_inputs(
        tmp_path,
        {"id": "CVE-DEMO", "aliases": [], "fix_versions": []},
        {"severity_overrides": {"CVE-DEMO": "high"}},
    )
    report = security.create_report(
        audit, policy, blob, workflow, "a" * 40, "2.10.1", date(2026, 8, 7)
    )
    assert report["gate"]["status"] == "fail"
    assert report["gate"]["blocking_finding_count"] == 1


def test_accepted_high_security_finding_is_auditable(tmp_path):
    audit, policy, blob, workflow = security_inputs(
        tmp_path,
        {"id": "CVE-DEMO", "aliases": [], "fix_versions": []},
        {
            "severity_overrides": {"CVE-DEMO": "high"},
            "accepted_vulnerabilities": {
                "CVE-DEMO": {
                    "status": "accepted_risk",
                    "expires": "2026-09-01",
                    "rationale": "temporary test exception"
                }
            },
        },
    )
    report = security.create_report(
        audit, policy, blob, workflow, "a" * 40, "2.10.1", date(2026, 8, 7)
    )
    assert report["gate"]["status"] == "pass"
    assert report["findings"][0]["disposition"]["status"] == "accepted_risk"


def test_security_audit_classifies_actions_from_all_workflows(tmp_path):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "one.yml").write_text(
        "steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8"
    )
    (workflows / "two.yml").write_text(
        "steps:\n  - uses: actions/upload-artifact@" + "a" * 40 + "\n", encoding="utf-8"
    )
    actions = security.classify_actions(workflows)
    assert {item["workflow"] for item in actions} == {"one.yml", "two.yml"}
    assert {item["risk"] for item in actions} == {"mutable_major_tag", "none_detected"}
