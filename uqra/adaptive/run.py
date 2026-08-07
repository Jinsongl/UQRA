"""Configuration-driven entry point for reduced UQRA software benchmarks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from uqra._version import __version__

from .benchmark_registry import get_benchmark
from .manifest import build_artifacts, provenance, write_artifacts


CONFIG_SCHEMA = "uqra.adaptive.runner-config/v1"
CONFIG_SCHEMA_V2 = "uqra.adaptive.runner-config/v2"
SUPPORTED_CONFIG_SCHEMAS = {CONFIG_SCHEMA, CONFIG_SCHEMA_V2}
MANIFEST_SCHEMA = "uqra.adaptive.runner-manifest/v2"
TRACE_SCHEMA = "uqra.adaptive.trace/v1"
RUNNER_KIND = "deterministic_benchmark"
V1_BENCHMARK = "phase8_two_dimensional_hermite"


def _canonical_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_config(config):
    """Validate the supported v1 configuration subset and return a safe copy."""
    if not isinstance(config, dict):
        raise ValueError("runner config must be a JSON object")
    required = {"schema", "purpose", "scale", "runner", "output"}
    missing = sorted(required.difference(config))
    unknown = sorted(set(config).difference(required))
    if missing:
        raise ValueError(f"runner config missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"runner config has unknown fields: {', '.join(unknown)}")
    if config["schema"] not in SUPPORTED_CONFIG_SCHEMAS:
        raise ValueError(f"unsupported runner config schema: {config['schema']!r}")
    if config["purpose"] != "software_benchmark" or config["scale"] != "reduced":
        raise ValueError("this entry point only accepts reduced software_benchmark configs")

    runner = config["runner"]
    if not isinstance(runner, dict):
        raise ValueError("runner must be a JSON object")
    runner_required = {"kind", "benchmark", "scenarios"}
    if set(runner) != runner_required:
        raise ValueError("runner fields must be exactly: kind, benchmark, scenarios")
    if runner["kind"] != RUNNER_KIND:
        raise ValueError(f"unsupported runner kind: {runner['kind']!r}")
    if config["schema"] == CONFIG_SCHEMA and runner["benchmark"] != V1_BENCHMARK:
        raise ValueError(f"runner config v1 only supports benchmark: {V1_BENCHMARK}")
    benchmark = get_benchmark(runner["benchmark"])
    scenarios = runner["scenarios"]
    if (not isinstance(scenarios, list) or not scenarios
            or any(not isinstance(item, str) for item in scenarios)):
        raise ValueError("runner.scenarios must be a non-empty string array")
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("runner.scenarios must not contain duplicates")
    unsupported = sorted(set(scenarios).difference(benchmark.scenarios))
    if unsupported:
        raise ValueError(f"unsupported scenarios: {', '.join(unsupported)}")

    output = config["output"]
    if not isinstance(output, dict) or set(output) != {"manifest"}:
        raise ValueError("output must contain exactly one manifest path")
    if not isinstance(output["manifest"], str) or not output["manifest"].strip():
        raise ValueError("output.manifest must be a non-empty path string")
    return copy.deepcopy(config)


def load_config(path):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read runner config {path}: {error}") from error
    return validate_config(payload)


def validate_manifest(manifest):
    """Check invariants required by the published runner-manifest v1 contract."""
    required = {"schema", "trace_schema", "purpose", "scale", "config_hash",
                "config", "run", "provenance", "artifacts", "stable_manifest_hash"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("runner manifest fields do not match the v1 contract")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["trace_schema"] != TRACE_SCHEMA:
        raise ValueError("runner manifest uses an unsupported schema")
    if manifest["purpose"] != "software_benchmark" or manifest["scale"] != "reduced":
        raise ValueError("runner manifest has an invalid purpose or scale")
    if manifest["config_hash"] != _canonical_hash(manifest["config"]):
        raise ValueError("runner manifest config hash is invalid")
    run = manifest["run"]
    if not isinstance(run, dict):
        raise ValueError("runner manifest benchmark is invalid")
    benchmark = get_benchmark(run.get("benchmark"))
    scenarios = run.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("runner manifest has no scenario results")
    for name, result in scenarios.items():
        if name not in benchmark.scenarios or not isinstance(result, dict):
            raise ValueError(f"runner manifest scenario is invalid: {name!r}")
        if result.get("scenario") != name or not isinstance(result.get("trace"), list):
            raise ValueError(f"runner manifest trace is invalid: {name!r}")
        if len(result.get("trace_hash", "")) != 64:
            raise ValueError(f"runner manifest trace hash is invalid: {name!r}")
        if not all(isinstance(row, dict) and row.get("candidate_hash")
                   for row in result["trace"]):
            raise ValueError(f"runner manifest contains an invalid trace row: {name!r}")
    stable = copy.deepcopy(manifest)
    expected = stable.pop("stable_manifest_hash")
    stable.pop("provenance", None)
    stable.get("run", {}).pop("git", None)
    if expected != _canonical_hash(stable):
        raise ValueError("runner stable manifest hash is invalid")
    return True


def _run_config_bundle(config, *, config_path="<in-memory-config>", output_path=None):
    config = validate_config(config)
    benchmark = get_benchmark(config["runner"]["benchmark"])
    benchmark_manifest = benchmark.run(config["runner"]["scenarios"])
    output_path = Path(output_path or config["output"]["manifest"])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "trace_schema": TRACE_SCHEMA,
        "purpose": config["purpose"],
        "scale": config["scale"],
        "config_hash": _canonical_hash(config),
        "config": config,
        "run": benchmark_manifest,
    }
    manifest["provenance"] = provenance(config_path, output_path)
    manifest["artifacts"], files = build_artifacts(manifest, benchmark.inputs(), output_path)
    stable = copy.deepcopy(manifest)
    stable.pop("provenance", None)
    stable.get("run", {}).pop("git", None)
    manifest["stable_manifest_hash"] = _canonical_hash(stable)
    validate_manifest(manifest)
    return manifest, files


def run_config(config):
    """Execute a config and return its complete in-memory manifest."""
    return _run_config_bundle(config)[0]


def _write_status(message, stream=None):
    """Write a CLI status line without failing on a narrow Windows code page."""
    stream = stream or sys.stdout
    try:
        stream.write(message)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        escaped = message.encode(encoding, errors="backslashreplace").decode(encoding)
        stream.write(escaped)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="override output.manifest from the configuration")
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        output = arguments.output or Path(config["output"]["manifest"])
        manifest, files = _run_config_bundle(
            config, config_path=arguments.config, output_path=output,
        )
        payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        write_artifacts(output, files)
        output.write_text(payload, encoding="utf-8")
    except ValueError as error:
        parser.error(str(error))
    _write_status(f"wrote {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
