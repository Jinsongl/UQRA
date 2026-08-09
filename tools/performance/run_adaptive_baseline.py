"""Measure deterministic reduced UQRA software benchmarks on Windows/Python 3.12."""
from __future__ import annotations

import argparse
import cProfile
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = (
    ROOT / "examples" / "configs" / "four_branch_reduced_v1.json",
    ROOT / "examples" / "configs" / "ishigami_reduced_v1.json",
    ROOT / "examples" / "configs" / "gayton_reduced_v1.json",
)
LOCK_PATH = ROOT / "requirements" / "compatibility-py312.txt"
SCHEMA = "uqra.adaptive.performance-baseline/v1"
PROFILE_TARGETS = {
    "controller_run": ("uqra/adaptive/controller.py", "run"),
    "order_design": ("uqra/adaptive/controller.py", "_order_design"),
    "fit": ("uqra/adaptive/controller.py", "_fit"),
    "inner_loop": ("uqra/adaptive/controller.py", "_inner_loop"),
    "optimal_design": ("uqra/adaptive/optimal_design.py", "greedy_optimal_ids"),
    "doi": ("uqra/adaptive/doi.py", "build_doi"),
    "sparse_pce": ("uqra/adaptive/sparse_pce.py", "fit_lars_path"),
    "artifact_build": ("uqra/adaptive/manifest.py", "build_artifacts"),
}


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_memory() -> dict[str, int]:
    if platform.system() != "Windows":
        raise RuntimeError("PERF-01 process memory evidence requires Windows")
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    handle = get_current_process()
    if not get_process_memory_info(
            handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
    }


def git(*arguments: str) -> str | None:
    process = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *arguments],
        capture_output=True, text=True, check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _profile_stats(profiler: cProfile.Profile) -> dict[tuple[str, int, str], tuple]:
    import pstats
    return pstats.Stats(profiler).stats


def extract_profile_times(profiler: cProfile.Profile) -> dict[str, float]:
    result = {name: 0.0 for name in PROFILE_TARGETS}
    for (filename, _line, function), values in _profile_stats(profiler).items():
        normalized = filename.replace("\\", "/")
        cumulative = float(values[3])
        for name, (suffix, expected_function) in PROFILE_TARGETS.items():
            if normalized.endswith(suffix) and function == expected_function:
                result[name] += cumulative
    return result


def child_measure(config_path: Path) -> dict:
    from uqra.adaptive.run import load_config, run_config, validate_manifest

    config_path = config_path.resolve()
    config = load_config(config_path)
    memory_before = process_memory()
    tracemalloc.start()
    started = time.perf_counter_ns()
    manifest = run_config(config)
    elapsed_ns = time.perf_counter_ns() - started
    _current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_after = process_memory()
    validate_manifest(manifest)

    profiler = cProfile.Profile()
    profiler.enable()
    profiled_manifest = run_config(config)
    profiler.disable()
    validate_manifest(profiled_manifest)
    if manifest["stable_manifest_hash"] != profiled_manifest["stable_manifest_hash"]:
        raise RuntimeError("profiled run changed the stable manifest identity")

    run = manifest["run"]
    scenario = next(iter(run["scenarios"].values()))
    return {
        "benchmark": run["benchmark"],
        "config": config_path.relative_to(ROOT).as_posix(),
        "total_seconds": elapsed_ns / 1_000_000_000,
        "python_traced_peak_bytes": int(traced_peak),
        "working_set_before_bytes": memory_before["working_set_bytes"],
        "working_set_after_bytes": memory_after["working_set_bytes"],
        "peak_working_set_bytes": memory_after["peak_working_set_bytes"],
        "peak_working_set_delta_bytes": max(
            0, memory_after["peak_working_set_bytes"] - memory_before["working_set_bytes"]
        ),
        "stage_cumulative_seconds": extract_profile_times(profiler),
        "identity": {
            "stable_manifest_hash": manifest["stable_manifest_hash"],
            "contract_manifest_hash": run.get("contract_manifest_hash"),
            "contract_trace_hash": scenario.get("contract_trace_hash"),
            "trace_hash": scenario["trace_hash"],
            "stop_reason": scenario["stop_reason"],
            "trace_rows": scenario["trace_rows"],
            "model_call_count": scenario["model_call_count"],
            "datasets": run["input"]["datasets"],
        },
    }


def summary(values: list[float | int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def run_parent(configs: list[Path], repetitions: int, output: Path) -> dict:
    if platform.system() != "Windows" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("formal PERF-01 baseline requires Windows and Python 3.12")
    if repetitions < 3:
        raise ValueError("formal PERF-01 baseline requires at least three repetitions")

    records = []
    for config in configs:
        for repetition in range(1, repetitions + 1):
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", "--config", str(config)],
                capture_output=True, text=True, check=True, cwd=ROOT,
            )
            record = json.loads(process.stdout)
            record["repetition"] = repetition
            records.append(record)

    aggregates = {}
    for benchmark in sorted({item["benchmark"] for item in records}):
        selected = [item for item in records if item["benchmark"] == benchmark]
        identities = [item["identity"] for item in selected]
        if any(identity != identities[0] for identity in identities[1:]):
            raise RuntimeError(f"behavior identity changed across repetitions: {benchmark}")
        aggregates[benchmark] = {
            "total_seconds": summary([item["total_seconds"] for item in selected]),
            "python_traced_peak_bytes": summary(
                [item["python_traced_peak_bytes"] for item in selected]
            ),
            "peak_working_set_bytes": summary(
                [item["peak_working_set_bytes"] for item in selected]
            ),
            "peak_working_set_delta_bytes": summary(
                [item["peak_working_set_delta_bytes"] for item in selected]
            ),
            "stage_cumulative_seconds_median": {
                stage: statistics.median(
                    item["stage_cumulative_seconds"][stage] for item in selected
                ) for stage in PROFILE_TARGETS
            },
            "identity": identities[0],
        }

    tool_path = Path(__file__).resolve()
    evidence = {
        "schema": SCHEMA,
        "purpose": "software_engineering_performance_acceptance",
        "paper_production": False,
        "scientific_reproduction": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "worktree_dirty": bool(git("status", "--porcelain")),
            "tool_path": tool_path.relative_to(ROOT).as_posix(),
            "tool_sha256": sha256(tool_path),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "uqra": importlib.metadata.version("uqra"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "lock_path": LOCK_PATH.relative_to(ROOT).as_posix(),
            "lock_sha256": sha256(LOCK_PATH),
        },
        "method": {
            "repetitions": repetitions,
            "process_isolation": "fresh child process per benchmark repetition",
            "total_scope": "load validated config before timer; run_config inside timer",
            "timer": "time.perf_counter_ns",
            "python_peak": "tracemalloc peak during the timed run",
            "process_peak": "Windows GetProcessMemoryInfo PeakWorkingSetSize after timed run",
            "process_delta": "peak working set minus working set immediately before timed run",
            "stage_scope": "second run under cProfile; cumulative/inclusive and non-additive",
            "warmup": "none",
        },
        "records": records,
        "aggregates": aggregates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.child:
        if not arguments.config or len(arguments.config) != 1:
            parser.error("child mode requires exactly one --config")
        print(json.dumps(child_measure(arguments.config[0]), sort_keys=True))
        return 0
    if arguments.output is None:
        parser.error("--output is required")
    configs = [path.resolve() for path in (arguments.config or DEFAULT_CONFIGS)]
    run_parent(configs, arguments.repetitions, arguments.output.resolve())
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
