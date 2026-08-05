"""Static registry for supported reduced software benchmarks.

The registry deliberately contains callables owned by UQRA. Configuration files
select a public name and can never provide a Python module or import path.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .benchmark import BENCHMARK_NAME, run_suite
from .four_branch_reduced import (BENCHMARK_NAME as FOUR_BRANCH_NAME,
                                  SCENARIO as FOUR_BRANCH_SCENARIO,
                                  run_suite as run_four_branch_suite)


@dataclass(frozen=True)
class BenchmarkDefinition:
    """A configuration-visible benchmark and its supported scenarios."""

    name: str
    scenarios: tuple[str, ...]
    run: Callable[[Sequence[str] | None], dict]


_REGISTRY = {
    BENCHMARK_NAME: BenchmarkDefinition(
        name=BENCHMARK_NAME,
        scenarios=("converged", "max_order", "overfit_fallback", "runtime_failure"),
        run=run_suite,
    ),
    FOUR_BRANCH_NAME: BenchmarkDefinition(
        name=FOUR_BRANCH_NAME,
        scenarios=(FOUR_BRANCH_SCENARIO,),
        run=run_four_branch_suite,
    ),
}

BENCHMARK_REGISTRY: Mapping[str, BenchmarkDefinition] = MappingProxyType(_REGISTRY)


def benchmark_names() -> tuple[str, ...]:
    """Return registered public names in deterministic order."""
    return tuple(sorted(BENCHMARK_REGISTRY))


def get_benchmark(name: str) -> BenchmarkDefinition:
    """Resolve a public benchmark name without importing user-provided code."""
    try:
        return BENCHMARK_REGISTRY[name]
    except (KeyError, TypeError) as error:
        supported = ", ".join(benchmark_names())
        raise ValueError(f"unsupported benchmark: {name!r}; registered: {supported}") from error
