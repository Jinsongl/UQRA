import json

from uqra.adaptive.benchmark import main, run_suite


EXPECTED = {
    "converged": ("converged", "outer_qoi_converged"),
    "max_order": ("nonconverged", "max_order_reached"),
    "overfit_fallback": ("overfit_fallback", "overfit_rebuild_not_converged"),
    "runtime_failure": ("runtime_failure", "runtime_failure"),
}


def test_phase8_suite_is_repeatable_and_covers_four_outcomes():
    first = run_suite()
    second = run_suite()
    assert first["stable_manifest_hash"] == second["stable_manifest_hash"]
    assert first["input"] == second["input"]
    for name, expected in EXPECTED.items():
        scenario = first["scenarios"][name]
        assert (scenario["status"], scenario["stop_reason"]) == expected
        assert all(scenario["invariant_checks"].values())
        assert scenario["model_call_count"] == len(scenario["evaluated_global_ids"])
        assert scenario["model_call_count"] == len(scenario["evaluated_coordinate_hashes"])
        assert scenario["trace_hash"] == second["scenarios"][name]["trace_hash"]
    for name in ("converged", "max_order", "overfit_fallback"):
        stages = first["scenarios"][name]["stage_counts"]
        assert stages["global_refit"] >= 1
        assert stages["doi_constructed"] >= 1
        assert stages["doi_refit"] >= 1
        assert stages["order_completed"] >= 1
    assert first["scenarios"]["overfit_fallback"]["trigger_disclosure"] is not None
    assert first["scenarios"]["runtime_failure"]["stage_counts"]["runtime_failure"] == 1


def test_cli_writes_reproducible_json_manifest(tmp_path):
    output = tmp_path / "adaptive_phase8_manifest.json"
    assert main(["--scenario", "all", "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["reproduce_command"].startswith("python -m uqra.adaptive.benchmark")
    assert set(manifest["scenarios"]) == set(EXPECTED)
    assert len(manifest["stable_manifest_hash"]) == 64
