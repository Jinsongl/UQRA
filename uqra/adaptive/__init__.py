"""UQRA-compatible adaptive sparse polynomial chaos expansion."""
from .controller import AdaptiveResult, AdaptiveSparsePCE, ControllerState, RoundTrace
from .doi import DomainOfInterest, build_doi
from .optimal_design import greedy_optimal_ids, optimality_scores, rrqr_initial_ids
from .profiles import (CompatibilityProfile, dissertation_profile,
                       literal_legacy_profile, publication_profile)
from .history import archived_array_identity, historical_literal_index_diagnostic
from .regression import (DOIBehaviorTrace, FrozenHermiteInputs, LarsBehaviorTrace,
                         OEDRoundTrace, canonical_legacy_lars_trace,
                         compare_doi, compare_oed_rounds, compare_preprocessing,
                         freeze_hermite_inputs, literal_index_bug_trace,
                         modern_lars_trace)
from .sparse_pce import SparsePCEFit, fit_lars_path, legacy_preprocess
from .state import AdaptiveState, array_hash, coordinate_hash

__all__ = ["AdaptiveResult", "AdaptiveSparsePCE", "AdaptiveState", "CompatibilityProfile", "ControllerState",
           "DomainOfInterest", "RoundTrace", "SparsePCEFit", "array_hash", "build_doi",
           "coordinate_hash", "dissertation_profile", "fit_lars_path", "greedy_optimal_ids",
           "legacy_preprocess", "literal_legacy_profile", "optimality_scores",
           "publication_profile", "rrqr_initial_ids", "DOIBehaviorTrace",
           "FrozenHermiteInputs", "LarsBehaviorTrace", "OEDRoundTrace",
           "canonical_legacy_lars_trace", "compare_doi", "compare_oed_rounds",
           "compare_preprocessing", "freeze_hermite_inputs", "literal_index_bug_trace",
           "modern_lars_trace", "archived_array_identity",
           "historical_literal_index_diagnostic"]
