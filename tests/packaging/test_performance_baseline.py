import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "specs" / "performance" / "UQRA_PERF_01_BASELINE.json"
TOOL = ROOT / "tools" / "performance" / "run_adaptive_baseline.py"
LOCK = ROOT / "requirements" / "compatibility-py312.txt"
BENCHMARKS = {
    "four_branch_reduced_v1", "ishigami_reduced_v1", "gayton_reduced_v1",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_text_sha256(path):
    payload = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_perf01_evidence_is_bound_to_method_inputs_and_behavior_identity():
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema"] == "uqra.adaptive.performance-baseline/v1"
    assert evidence["purpose"] == "software_engineering_performance_acceptance"
    assert evidence["paper_production"] is False
    assert evidence["scientific_reproduction"] is False
    assert evidence["environment"]["python"].startswith("3.12.")
    assert evidence["environment"]["lock_sha256"] == sha256(LOCK)
    assert evidence["source"]["tool_canonical_sha256"] == canonical_text_sha256(TOOL)
    assert evidence["method"]["repetitions"] == 3
    assert set(evidence["aggregates"]) == BENCHMARKS
    assert len(evidence["records"]) == 3 * len(BENCHMARKS)

    for benchmark in BENCHMARKS:
        records = [item for item in evidence["records"] if item["benchmark"] == benchmark]
        assert [item["repetition"] for item in records] == [1, 2, 3]
        assert all(item["identity"] == records[0]["identity"] for item in records[1:])
        identity = records[0]["identity"]
        assert set(identity["datasets"]) == {"candidate", "test", "reference"}
        assert all(len(item["sha256"]) == 64 for item in identity["datasets"].values())
        assert len(identity["contract_trace_hash"]) == 64
        assert identity["trace_rows"] > 0
        assert identity["model_call_count"] > 0

        aggregate = evidence["aggregates"][benchmark]
        assert aggregate["identity"] == identity
        for metric in ("total_seconds", "python_traced_peak_bytes",
                       "peak_working_set_bytes", "peak_working_set_delta_bytes"):
            values = aggregate[metric]
            assert 0 < values["min"] <= values["median"] <= values["max"]
        assert all(value >= 0 for value in aggregate["stage_cumulative_seconds_median"].values())
