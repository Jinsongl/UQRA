from __future__ import annotations

import hashlib
import importlib.util
from datetime import date
from pathlib import Path

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
