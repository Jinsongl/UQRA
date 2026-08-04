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
    doi_fallback: Literal["legacy_nearest", "expand", "global", "skip", "unresolved"] = "unresolved"
    minimum_doi_size: int | None = None
    literal_outer_break: bool = True
    overfit_rebuild: bool = False

    def batch_size(self, sparsity: int) -> int:
        return min(5, max(3, int(sparsity)))

    def validate_for_run(self) -> None:
        if self.name == "dissertation":
            unresolved = [
                name for name, value in (
                    ("initial_sparsity", self.initial_sparsity),
                    ("order_budget_factor", self.order_budget_factor),
                    ("outer_stable_checks", self.outer_stable_checks),
                    ("qoi_tolerance", self.qoi_tolerance),
                ) if value is None
            ]
            if unresolved:
                raise ValueError("unresolved dissertation fields: " + ", ".join(unresolved))


def literal_legacy_profile(**overrides) -> CompatibilityProfile:
    values = dict(name="literal_legacy", initial_sparsity=0, order_budget_factor=3.0,
                  literal_outer_break=False, doi_fallback="legacy_nearest")
    values.update(overrides)
    return CompatibilityProfile(**values)


def dissertation_profile(**resolved) -> CompatibilityProfile:
    values = dict(name="dissertation", literal_outer_break=True, doi_fallback="unresolved")
    values.update(resolved)
    return CompatibilityProfile(**values)


def publication_profile(**overrides) -> CompatibilityProfile:
    values = dict(name="publication", initial_sparsity=0, order_budget_factor=2.0, outer_stable_checks=1,
                  qoi_tolerance=1e-3, minimum_doi_size=1, doi_fallback="expand",
                  literal_outer_break=True, overfit_rebuild=True)
    values.update(overrides)
    return CompatibilityProfile(**values)
