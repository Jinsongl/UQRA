import json
from pathlib import Path

import pytest

from uqra.adaptive.benchmark_registry import benchmark_names, get_benchmark
from uqra.adaptive.run import (CONFIG_SCHEMA, CONFIG_SCHEMA_V2, MANIFEST_SCHEMA, TRACE_SCHEMA,
                               load_config, main, run_config, validate_config,
                               validate_manifest)


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "examples" / "configs" / "adaptive_reduced_smoke.json"
V2_SMOKE_CONFIG = ROOT / "examples" / "configs" / "adaptive_registry_v2_smoke.json"


def test_published_json_schemas_and_examples_are_well_formed():
    schemas = {
        "adaptive-runner-config.schema.json": CONFIG_SCHEMA,
        "adaptive-runner-config-v2.schema.json": CONFIG_SCHEMA_V2,
        "adaptive-runner-manifest.schema.json": MANIFEST_SCHEMA,
        "adaptive-trace.schema.json": TRACE_SCHEMA,
    }
    for name in schemas:
        payload = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["$id"].endswith(name)
    examples = {
        SMOKE_CONFIG: CONFIG_SCHEMA,
        ROOT / "examples" / "configs" / "adaptive_reduced_full.json": CONFIG_SCHEMA,
        V2_SMOKE_CONFIG: CONFIG_SCHEMA_V2,
        ROOT / "examples" / "configs" / "adaptive_registry_v2_full.json": CONFIG_SCHEMA_V2,
    }
    for path, expected_schema in examples.items():
        assert load_config(path)["schema"] == expected_schema


def test_v2_schema_benchmark_enum_matches_static_registry():
    schema = json.loads(
        (ROOT / "schemas" / "adaptive-runner-config-v2.schema.json").read_text(encoding="utf-8")
    )
    configured_names = schema["properties"]["runner"]["properties"]["benchmark"]["enum"]
    assert configured_names == list(benchmark_names())
    assert get_benchmark(configured_names[0]).name == configured_names[0]


def test_registry_rejects_module_paths_and_unknown_benchmarks():
    config = load_config(V2_SMOKE_CONFIG)
    config["runner"]["benchmark"] = "package.module:benchmark"
    with pytest.raises(ValueError, match="unsupported benchmark"):
        validate_config(config)


def test_runner_rejects_paper_production_and_unknown_fields():
    config = load_config(SMOKE_CONFIG)
    config["purpose"] = "paper_production"
    with pytest.raises(ValueError, match="software_benchmark"):
        validate_config(config)
    config = load_config(SMOKE_CONFIG)
    config["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        validate_config(config)


def test_config_driven_smoke_manifest_is_valid_and_repeatable():
    config = load_config(SMOKE_CONFIG)
    first = run_config(config)
    second = run_config(config)
    assert validate_manifest(first)
    assert first["stable_manifest_hash"] == second["stable_manifest_hash"]
    assert first["purpose"] == "software_benchmark"
    assert first["scale"] == "reduced"
    assert set(first["run"]["scenarios"]) == {"converged"}


def test_v2_registry_config_is_valid_and_repeatable():
    config = load_config(V2_SMOKE_CONFIG)
    first = run_config(config)
    second = run_config(config)
    assert validate_manifest(first)
    assert first["stable_manifest_hash"] == second["stable_manifest_hash"]
    assert first["config"]["schema"] == CONFIG_SCHEMA_V2


def test_config_driven_cli_writes_manifest(tmp_path):
    output = tmp_path / "manifest.json"
    assert main(["--config", str(SMOKE_CONFIG), "--output", str(output)]) == 0
    assert validate_manifest(json.loads(output.read_text(encoding="utf-8")))
