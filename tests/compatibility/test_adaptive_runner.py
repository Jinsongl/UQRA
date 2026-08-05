import json
from pathlib import Path

import pytest

from uqra.adaptive.run import (CONFIG_SCHEMA, MANIFEST_SCHEMA, TRACE_SCHEMA,
                               load_config, main, run_config, validate_config,
                               validate_manifest)


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "examples" / "configs" / "adaptive_reduced_smoke.json"


def test_published_json_schemas_and_examples_are_well_formed():
    schemas = {
        "adaptive-runner-config.schema.json": CONFIG_SCHEMA,
        "adaptive-runner-manifest.schema.json": MANIFEST_SCHEMA,
        "adaptive-trace.schema.json": TRACE_SCHEMA,
    }
    for name in schemas:
        payload = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["$id"].endswith(name)
    for path in (SMOKE_CONFIG, ROOT / "examples" / "configs" / "adaptive_reduced_full.json"):
        assert load_config(path)["schema"] == CONFIG_SCHEMA


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


def test_config_driven_cli_writes_manifest(tmp_path):
    output = tmp_path / "manifest.json"
    assert main(["--config", str(SMOKE_CONFIG), "--output", str(output)]) == 0
    assert validate_manifest(json.loads(output.read_text(encoding="utf-8")))
