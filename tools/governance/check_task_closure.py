"""Enforce deterministic UQRA task-closure repository rules."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


BOARD = "specs/UQRA_PROJECT_PROGRESS_BOARD.md"
PLAN = "specs/UQRA_PROJECT_DEVELOPMENT_PLAN.md"
FROZEN_PREFIX = "specs/releases/UQRA_V0.3.0_"
TASK_SCOPES = ("uqra/adaptive/", "tests/compatibility/", "tools/performance/")

TASK_DECLARATIONS = (
    "本 PR 承载稳定任务 ID；任务在合并及 required master gate 通过前保持“🔄 进行中”",
    "本 PR 不承载稳定任务 ID（例如纯治理、维护或 closure 文档 PR）",
)
PLAN_DECLARATIONS = (
    "路线、优先级、里程碑状态和完成门均未改变",
    "存在上述变化，且已同步 `specs/UQRA_PROJECT_DEVELOPMENT_PLAN.md`",
)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], check=False, text=True, capture_output=True, encoding="utf-8"
    )


def checked(body: str, label: str) -> bool:
    pattern = rf"(?mi)^\s*-\s*\[[xX]\]\s*{re.escape(label)}\s*$"
    return re.search(pattern, body) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--event-path", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    diff = git("diff", "--name-only", args.base, args.head)
    if diff.returncode:
        print(diff.stderr, file=sys.stderr)
        return diff.returncode
    changed = {line.strip().replace("\\", "/") for line in diff.stdout.splitlines()}

    whitespace = git("diff", "--check", args.base, args.head)
    if whitespace.returncode:
        failures.append("git diff --check failed:\n" + whitespace.stdout + whitespace.stderr)

    frozen = sorted(path for path in changed if path.startswith(FROZEN_PREFIX))
    if frozen:
        failures.append("Frozen v0.3.0 evidence changed: " + ", ".join(frozen))

    task_changes = sorted(
        path for path in changed if any(path.startswith(prefix) for prefix in TASK_SCOPES)
    )
    if task_changes and BOARD not in changed:
        failures.append(
            f"Task-scope changes require {BOARD}; matched: " + ", ".join(task_changes)
        )

    if args.event_path:
        event = json.loads(args.event_path.read_text(encoding="utf-8"))
        pull_request = event.get("pull_request")
        if pull_request:
            body = pull_request.get("body") or ""
            task_count = sum(checked(body, label) for label in TASK_DECLARATIONS)
            plan_count = sum(checked(body, label) for label in PLAN_DECLARATIONS)
            if task_count != 1:
                failures.append("PR body must select exactly one task classification declaration.")
            if plan_count != 1:
                failures.append("PR body must select exactly one plan impact declaration.")
            if checked(body, PLAN_DECLARATIONS[1]) and PLAN not in changed:
                failures.append(f"PR declares plan impact but does not modify {PLAN}.")

    if failures:
        print("Task closure governance check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Task closure governance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
