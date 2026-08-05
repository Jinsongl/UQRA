"""Provenance and byte-level artifact identities for adaptive runner manifests."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
import sklearn

def _json_bytes(payload):
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _git(root, *arguments):
    process = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *arguments],
        capture_output=True, text=True, check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def source_tree_identity(root):
    """Hash the current bytes of every Git-tracked file in path order."""
    listing = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if listing is None:
        return {"algorithm": "sha256", "sha256": None, "tracked_files": None}
    paths = sorted(item for item in listing.split("\0") if item)
    digest = hashlib.sha256()
    count = 0
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        digest.update(relative.replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return {"algorithm": "sha256", "sha256": digest.hexdigest(), "tracked_files": count}


def provenance(config_path, output_path):
    root = Path(__file__).resolve().parents[2]
    dirty = _git(root, "status", "--porcelain")
    return {
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "worktree_dirty": None if dirty is None else bool(dirty),
        },
        "source_tree": source_tree_identity(root),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "reproduce_command": (
            f"python -m uqra.adaptive.run --config {Path(config_path).as_posix()} "
            f"--output {Path(output_path).as_posix()}"
        ),
    }


def _file_identity(relative_path, payload, **metadata):
    return {"path": relative_path, "size_bytes": len(payload), "sha256": _sha256(payload),
            **metadata}


def build_artifacts(manifest, arrays, output_path):
    """Return an artifact inventory and exact file payloads for the evidence package."""
    output_path = Path(output_path)
    directory = f"{output_path.stem}_artifacts"
    files = {}
    inputs = {}
    for role, array in sorted(arrays.items()):
        stream = BytesIO()
        np.save(stream, np.asarray(array), allow_pickle=False)
        payload = stream.getvalue()
        relative = f"{directory}/inputs/{role}.npy"
        files[relative] = payload
        inputs[role] = _file_identity(
            relative, payload, shape=list(array.shape), dtype=str(array.dtype),
            array_size_bytes=int(array.nbytes), role=role,
        )

    traces, results = {}, {}
    for name, scenario in sorted(manifest["run"]["scenarios"].items()):
        trace_payload = _json_bytes(scenario["trace"])
        trace_path = f"{directory}/traces/{name}.json"
        files[trace_path] = trace_payload
        raw_hash = scenario["trace_hash"]
        contract_hash = scenario.get("contract_trace_hash")
        if contract_hash is None:
            # Phase 8 predates the explicit field; derive the same cross-platform projection.
            projected = []
            for row in scenario["trace"]:
                item = dict(row); item.pop("cv_path", None); item.pop("qoi", None)
                projected.append(item)
            contract_hash = _sha256(json.dumps(
                projected, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode())
        traces[name] = _file_identity(
            trace_path, trace_payload, rows=len(scenario["trace"]),
            raw_trace_hash=raw_hash, raw_trace_hash_scope="full_precision_trace",
            contract_trace_hash=contract_hash,
            contract_trace_hash_scope="discrete_trace_without_cv_path_or_qoi",
        )

        result = {key: value for key, value in scenario.items() if key != "trace"}
        result_payload = _json_bytes(result)
        result_path = f"{directory}/results/{name}.json"
        files[result_path] = result_payload
        results[name] = _file_identity(result_path, result_payload)

    summary_payload = _json_bytes({
        "benchmark": manifest["run"]["benchmark"],
        "scenarios": {name: {"status": item["status"], "stop_reason": item["stop_reason"]}
                      for name, item in sorted(manifest["run"]["scenarios"].items())},
    })
    summary_path = f"{directory}/output-summary.json"
    files[summary_path] = summary_payload
    return {"inputs": inputs, "traces": traces, "results": results,
            "output_summary": _file_identity(summary_path, summary_payload)}, files


def write_artifacts(manifest_path, files):
    base = Path(manifest_path).parent
    for relative, payload in files.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
