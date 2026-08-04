import json

from uqra.adaptive.publication import (SENSITIVITY_CASES,
                                       frozen_publication_protocol, main,
                                       run_publication_suite)


def test_publication_protocol_is_frozen_and_complete():
    first = frozen_publication_protocol()
    second = frozen_publication_protocol()
    assert first == second
    assert len(first["protocol_hash"]) == 64
    parameters = first["base_parameters"]
    for key in ("order_budget_factor", "doi_fallback", "cv_folds",
                "outer_stable_checks", "accuracy_max_cv_error", "validity"):
        assert parameters[key] is not None
    assert "overfit_fallback" in SENSITIVITY_CASES
    first["base_parameters"]["cv_folds"] = 99
    assert frozen_publication_protocol()["base_parameters"]["cv_folds"] == 4


def test_publication_sensitivity_separates_implementations_and_overfit():
    manifest = run_publication_suite()
    results = manifest["results"]
    assert results["canonical_uqra"]["status"] == "unavailable"
    assert set(results["modern_compatible"]) == set(SENSITIVITY_CASES)
    assert set(results["portable_control"]) == set(SENSITIVITY_CASES)
    assert manifest["overfit_fallback_counts"] == {
        "canonical_uqra": None, "modern_compatible": 1, "portable_control": 1,
    }
    assert manifest["portable_basis_max_abs_error"] == 0.0
    assert len(manifest["source_tree_hash"]) == 64
    for implementation in ("modern_compatible", "portable_control"):
        for row in results[implementation].values():
            assert row["validity_passed"]
            assert all(row["invariant_checks"].values())
    for comparison in manifest["modern_vs_portable"].values():
        assert all(comparison[key] for key in (
            "status_equal", "stop_reason_equal", "selected_ids_equal", "trace_hash_equal"))
        assert comparison["qoi_abs_difference"] in (None, 0.0)


def test_publication_manifest_is_repeatable_and_cli_writes_json(tmp_path):
    first = run_publication_suite()
    second = run_publication_suite()
    assert first["stable_manifest_hash"] == second["stable_manifest_hash"]
    output = tmp_path / "publication.json"
    assert main(["--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["stable_manifest_hash"] == first["stable_manifest_hash"]
