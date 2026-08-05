import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from uqra.adaptive.benchmark_registry import benchmark_names, get_benchmark
from uqra.adaptive.four_branch_reduced import INPUT_HASHES
from uqra.adaptive.gayton_reduced import (
    EXPECTED_CONTRACT_MANIFEST_HASH as GAYTON_MANIFEST_HASH,
    EXPECTED_REFERENCE_FAILURE_PROBABILITY,
    EXPECTED_CONTRACT_TRACE_HASH as GAYTON_TRACE_HASH,
    gayton_limit_state,
)
from uqra.adaptive.ishigami_reduced import (
    EXPECTED_CONTRACT_MANIFEST_HASH as ISHIGAMI_MANIFEST_HASH,
    EXPECTED_REFERENCE_VARIANCE,
    EXPECTED_CONTRACT_TRACE_HASH as ISHIGAMI_TRACE_HASH,
    ishigami,
)
from uqra.adaptive.run import (CONFIG_SCHEMA, CONFIG_SCHEMA_V2, MANIFEST_SCHEMA, TRACE_SCHEMA,
                               load_config, main, run_config, validate_config,
                               validate_manifest)


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "examples" / "configs" / "adaptive_reduced_smoke.json"
V2_SMOKE_CONFIG = ROOT / "examples" / "configs" / "adaptive_registry_v2_smoke.json"
FOUR_BRANCH_CONFIG = ROOT / "examples" / "configs" / "four_branch_reduced_v1.json"
ISHIGAMI_CONFIG = ROOT / "examples" / "configs" / "ishigami_reduced_v1.json"
GAYTON_CONFIG = ROOT / "examples" / "configs" / "gayton_reduced_v1.json"


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
        FOUR_BRANCH_CONFIG: CONFIG_SCHEMA_V2,
        ISHIGAMI_CONFIG: CONFIG_SCHEMA_V2,
        GAYTON_CONFIG: CONFIG_SCHEMA_V2,
    }
    for path, expected_schema in examples.items():
        assert load_config(path)["schema"] == expected_schema


def _published_schema_validators():
    names = [
        "adaptive-runner-config.schema.json",
        "adaptive-runner-config-v2.schema.json",
        "adaptive-runner-manifest.schema.json",
        "adaptive-trace.schema.json",
    ]
    schemas = {name: json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
               for name in names}
    registry = Registry().with_resources(
        [(schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()]
    )
    return {name: Draft202012Validator(schema, registry=registry)
            for name, schema in schemas.items()}


@pytest.mark.parametrize("path", [SMOKE_CONFIG, V2_SMOKE_CONFIG, FOUR_BRANCH_CONFIG,
                                  ISHIGAMI_CONFIG, GAYTON_CONFIG])
def test_examples_and_generated_artifacts_pass_published_draft202012_schemas(path):
    validators = _published_schema_validators()
    config = load_config(path)
    config_schema = ("adaptive-runner-config.schema.json" if config["schema"] == CONFIG_SCHEMA
                     else "adaptive-runner-config-v2.schema.json")
    validators[config_schema].validate(config)
    manifest = run_config(config)
    validators["adaptive-runner-manifest.schema.json"].validate(manifest)
    for scenario in manifest["run"]["scenarios"].values():
        for row in scenario["trace"]:
            validators["adaptive-trace.schema.json"].validate(row)


def test_config_v2_schema_rejects_benchmark_scenario_mismatch():
    validators = _published_schema_validators()
    config = load_config(FOUR_BRANCH_CONFIG)
    config["runner"]["scenarios"] = ["converged"]
    assert list(validators["adaptive-runner-config-v2.schema.json"].iter_errors(config))


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


def test_v1_contract_cannot_select_v2_registry_benchmarks():
    config = load_config(SMOKE_CONFIG)
    config["runner"]["benchmark"] = "four_branch_reduced_v1"
    config["runner"]["scenarios"] = ["reduced"]
    with pytest.raises(ValueError, match="v1 only supports"):
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


def test_four_branch_reduced_config_is_registered_and_repeatable():
    config = load_config(FOUR_BRANCH_CONFIG)
    first = run_config(config)
    second = run_config(config)
    assert validate_manifest(first)
    assert first["stable_manifest_hash"] == second["stable_manifest_hash"]
    run = first["run"]
    assert run["input"]["historical_replay"] is False
    datasets = run["input"]["datasets"]
    assert len({item["seed"] for item in datasets.values()}) == 3
    assert len({item["sha256"] for item in datasets.values()}) == 3
    assert {name: item["sha256"] for name, item in datasets.items()} == INPUT_HASHES
    result = run["scenarios"]["reduced"]
    assert result["candidate_hash"] == datasets["candidate"]["sha256"]
    assert result["test_hash"] == datasets["test"]["sha256"]
    assert result["reference_hash"] == datasets["reference"]["sha256"]
    assert result["stage_counts"]["doi_constructed"] >= 1


@pytest.mark.parametrize("path", [FOUR_BRANCH_CONFIG, ISHIGAMI_CONFIG, GAYTON_CONFIG])
def test_m23_reduced_benchmark_contract(path):
    config = load_config(path)
    first, second = run_config(config), run_config(config)
    assert validate_manifest(first)
    assert first["stable_manifest_hash"] == second["stable_manifest_hash"]
    run = first["run"]
    assert run["input"]["purpose"] == "software_benchmark"
    assert run["input"]["scale"] == "reduced"
    assert run["input"]["historical_replay"] is False
    assert run["input"].get("paper_production", False) is False
    datasets = run["input"]["datasets"]
    assert set(datasets) == {"candidate", "test", "reference"}
    assert len({item["seed"] for item in datasets.values()}) == 3
    assert len({item["sha256"] for item in datasets.values()}) == 3
    scenario = run["scenarios"]["reduced"]
    assert scenario["stage_counts"]["doi_constructed"] >= 1
    assert scenario["trace_hash"] == second["run"]["scenarios"]["reduced"]["trace_hash"]


def test_ishigami_nonlinearity_interaction_and_frozen_results():
    xi = [[0.5, 0.5], [0.0, 0.0], [0.0, 1.0]]
    values = ishigami(xi)
    assert values[1] != values[0]  # x3 changes the response through x1*x3 interaction
    run = get_benchmark("ishigami_reduced_v1").run(["reduced"])
    scenario = run["scenarios"]["reduced"]
    assert scenario["reference_metric"]["value"] == EXPECTED_REFERENCE_VARIANCE
    assert scenario["contract_trace_hash"] == ISHIGAMI_TRACE_HASH
    assert run["contract_manifest_hash"] == ISHIGAMI_MANIFEST_HASH


def test_gayton_local_failure_domain_and_frozen_results():
    assert gayton_limit_state([[0.0], [0.0]])[0] > 0.0
    assert gayton_limit_state([[2.0], [2.0]])[0] < 0.0
    run = get_benchmark("gayton_reduced_v1").run(["reduced"])
    scenario = run["scenarios"]["reduced"]
    assert scenario["reference_metric"]["value"] == EXPECTED_REFERENCE_FAILURE_PROBABILITY
    assert 0.0 < scenario["reference_metric"]["value"] < 1.0
    assert scenario["contract_trace_hash"] == GAYTON_TRACE_HASH
    assert run["contract_manifest_hash"] == GAYTON_MANIFEST_HASH


def test_config_driven_cli_writes_manifest(tmp_path):
    output = tmp_path / "manifest.json"
    assert main(["--config", str(SMOKE_CONFIG), "--output", str(output)]) == 0
    assert validate_manifest(json.loads(output.read_text(encoding="utf-8")))
