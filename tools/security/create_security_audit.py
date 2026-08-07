"""Create classified security evidence and enforce the M5 vulnerability gate."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re


FAIL_SEVERITIES = {"critical", "high", "unknown"}
ACTION_PATTERN = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)


def role_map(policy: dict) -> dict[str, str]:
    roles: dict[str, str] = {}
    for role, names in policy["dependency_roles"].items():
        for name in names:
            normalized = name.lower().replace("_", "-")
            if normalized in roles:
                raise RuntimeError(f"dependency has multiple roles: {name}")
            roles[normalized] = role
    return roles


def accepted_disposition(policy: dict, identifier: str, audit_date: date) -> dict:
    disposition = policy["accepted_vulnerabilities"].get(identifier)
    if disposition is None:
        return {"status": "unresolved"}
    if not isinstance(disposition, dict) or not disposition.get("rationale"):
        raise RuntimeError(f"accepted vulnerability lacks structured rationale: {identifier}")
    status = disposition.get("status")
    if status == "accepted_risk":
        try:
            expires = date.fromisoformat(disposition["expires"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"accepted risk lacks a valid expiry: {identifier}") from error
        if expires < audit_date:
            raise RuntimeError(f"accepted risk has expired: {identifier}")
    elif status != "false_positive":
        raise RuntimeError(f"unsupported vulnerability disposition: {identifier}")
    return disposition


def classify_dependencies(
    audit: dict, policy: dict, audit_date: date
) -> tuple[list[dict], list[dict]]:
    roles = role_map(policy)
    classified = []
    findings = []
    for dependency in audit["dependencies"]:
        name = dependency["name"].lower().replace("_", "-")
        if name not in roles:
            raise RuntimeError(f"pip-audit dependency is not classified: {name}")
        classified.append({
            "name": name,
            "version": dependency["version"],
            "role": roles[name],
            "vulnerability_count": len(dependency["vulns"]),
        })
        for vulnerability in dependency["vulns"]:
            identifier = vulnerability["id"]
            severity = policy["severity_overrides"].get(identifier, "unknown").lower()
            disposition = accepted_disposition(policy, identifier, audit_date)
            findings.append({
                "dependency": name,
                "role": roles[name],
                "id": identifier,
                "aliases": vulnerability.get("aliases", []),
                "fix_versions": vulnerability.get("fix_versions", []),
                "severity": severity,
                "disposition": disposition,
            })
    return classified, findings


def classify_actions(workflow: Path) -> list[dict]:
    actions = []
    paths = sorted(workflow.glob("*.yml")) if workflow.is_dir() else [workflow]
    for path in paths:
        for reference in ACTION_PATTERN.findall(path.read_text(encoding="utf-8")):
            target, separator, revision = reference.partition("@")
            immutable = bool(separator and re.fullmatch(r"[0-9a-fA-F]{40}", revision))
            actions.append({
                "workflow": path.name,
                "action": target,
                "revision": revision or None,
                "role": "github_actions",
                "immutable_revision": immutable,
                "risk": "none_detected" if immutable else "mutable_major_tag",
                "disposition": "monitored_by_dependabot" if not immutable else "none_required",
            })
    if not actions:
        raise RuntimeError(f"no GitHub Actions references found in {workflow}")
    return actions


def create_report(
    audit: dict,
    policy: dict,
    blob_identity: dict,
    workflow: Path,
    source_commit: str,
    audit_tool_version: str,
    audit_date: date,
) -> dict:
    if blob_identity["source_commit"] != source_commit:
        raise RuntimeError("lock identity commit differs from security audit commit")
    dependencies, findings = classify_dependencies(audit, policy, audit_date)
    blocking = [
        finding for finding in findings
        if finding["severity"] in FAIL_SEVERITIES
        and finding["disposition"]["status"] == "unresolved"
    ]
    return {
        "schema": "uqra.security.continuous-audit/v1",
        "source_commit": source_commit,
        "audit_date": audit_date.isoformat(),
        "audit_tool": {"name": "pip-audit", "version": audit_tool_version},
        "lock_file": {
            "path": blob_identity["path"],
            "git_blob_oid": blob_identity["git_blob_oid"],
            "sha256": blob_identity["sha256"],
        },
        "dependencies": dependencies,
        "github_actions": classify_actions(workflow),
        "findings": findings,
        "gate": {
            "status": "fail" if blocking else "pass",
            "blocking_finding_count": len(blocking),
            "rule": "unresolved critical/high/unknown severity findings fail",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pip-audit-json", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--blob-identity", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--audit-tool-version", required=True)
    parser.add_argument("--audit-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    report = create_report(
        json.loads(arguments.pip_audit_json.read_text(encoding="utf-8")),
        json.loads(arguments.policy.read_text(encoding="utf-8")),
        json.loads(arguments.blob_identity.read_text(encoding="utf-8")),
        arguments.workflow,
        arguments.source_commit,
        arguments.audit_tool_version,
        arguments.audit_date,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate"], sort_keys=True))
    return 1 if report["gate"]["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
