"""Explicit compatibility profiles for adaptive sparse PCE."""
from dataclasses import dataclass
from typing import Literal


ProfileName = Literal["literal_legacy", "dissertation", "publication"]


@dataclass(frozen=True)
class CompatibilityProfile:
    name: ProfileName
    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_seed: int = 0
    initial_sparsity: int | None = None
    order_budget_factor: float | None = None
    outer_stable_checks: int | None = None
    qoi_tolerance: float | None = None
    qoi_epsilon: float = 1e-12
    inner_qoi_tolerance: float | None = None
    inner_stable_checks: int | None = None
    max_inner_iterations: int | None = None
    doi_fallback: Literal["legacy_nearest", "expand", "global", "skip", "unresolved"] = "unresolved"
    minimum_doi_size: int | None = None
    literal_outer_break: bool = True
    overfit_rebuild: bool = False

    def batch_size(self, sparsity: int) -> int:
        return min(5, max(3, int(sparsity)))

    def validate_for_run(self) -> None:
        if self.order_budget_factor is None or self.order_budget_factor <= 0:
            raise ValueError("order_budget_factor must be positive")
        if self.outer_stable_checks is not None and self.outer_stable_checks < 1:
            raise ValueError("outer_stable_checks must be positive")
        if self.qoi_tolerance is not None and self.qoi_tolerance < 0:
            raise ValueError("qoi_tolerance must be nonnegative")
        if self.inner_qoi_tolerance is not None and self.inner_qoi_tolerance < 0:
            raise ValueError("inner_qoi_tolerance must be nonnegative")
        if self.inner_stable_checks is not None and self.inner_stable_checks < 1:
            raise ValueError("inner_stable_checks must be positive")
        if self.max_inner_iterations is not None and self.max_inner_iterations < 1:
            raise ValueError("max_inner_iterations must be positive")
        if self.name == "dissertation":
            unresolved = [
                name for name, value in (
                    ("initial_sparsity", self.initial_sparsity),
                    ("order_budget_factor", self.order_budget_factor),
                    ("outer_stable_checks", self.outer_stable_checks),
                    ("qoi_tolerance", self.qoi_tolerance),
                    ("inner_qoi_tolerance", self.inner_qoi_tolerance),
                    ("inner_stable_checks", self.inner_stable_checks),
                ) if value is None
            ]
            if unresolved:
                raise ValueError("unresolved dissertation fields: " + ", ".join(unresolved))


def literal_legacy_profile(**overrides) -> CompatibilityProfile:
    values = dict(name="literal_legacy", initial_sparsity=0, order_budget_factor=3.0,
                  literal_outer_break=False, doi_fallback="legacy_nearest",
                  inner_qoi_tolerance=None, inner_stable_checks=1)
    values.update(overrides)
    return CompatibilityProfile(**values)


def dissertation_profile(**resolved) -> CompatibilityProfile:
    values = dict(name="dissertation", literal_outer_break=True, doi_fallback="unresolved")
    values.update(resolved)
    return CompatibilityProfile(**values)


def publication_profile(**overrides) -> CompatibilityProfile:
    values = dict(name="publication", initial_sparsity=0, order_budget_factor=2.0, outer_stable_checks=1,
                  qoi_tolerance=1e-3, minimum_doi_size=1, doi_fallback="expand",
                  literal_outer_break=True, overfit_rebuild=True,
                  inner_qoi_tolerance=1e-3, inner_stable_checks=1)
    values.update(overrides)
    return CompatibilityProfile(**values)
