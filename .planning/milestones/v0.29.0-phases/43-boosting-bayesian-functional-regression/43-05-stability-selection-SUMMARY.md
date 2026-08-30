---
phase: 43-boosting-bayesian-functional-regression
plan: "05"
subsystem: boosting_regression
tags: [functional-regression, stability-selection, subsampling, FDboost, REG-06-05]
requirements: [REG-06-05]
status: complete

dependency_graph:
  requires: [43-01-boosting-core-fosr]
  provides: [stability_selection, StabilityResult — per-learner selection frequencies + stable set + PFER bound]
  affects:
    - fdars-core/src/boosting_regression/stability.rs

tech_stack:
  added: []
  patterns:
    - "Subsampling ⌊n/2⌋ without replacement via partial Fisher–Yates, seeded per replicate (seed.wrapping_add(b))"
    - "iter_maybe_parallel! resample loop (deterministic regardless of parallelism)"
    - "Result-collecting parallel map: collect::<Result<Vec<Vec<bool>>, FdarError>>()"
    - "boost_fosr selected_learners → per-learner selection frequency aggregation"
    - "Meinshausen–Bühlmann PFER bound q²/((2·pi_thr−1)·p)"

key_files:
  created: []
  modified:
    - fdars-core/src/boosting_regression/stability.rs

decisions:
  - "Selection criterion: a base-learner counts as selected in a resample if it appears in boost_fosr's selected_learners at ANY iteration of the path (FDboost/stabs convention)."
  - "Subsampling ⌊n/2⌋ without replacement (Meinshausen-Bühlmann default), not bootstrap-with-replacement — documented divergence note vs stabs in rustdoc."
  - "Resample loop under iter_maybe_parallel!; each replicate owns an independently seeded StdRng so results are bit-identical whether run parallel or sequential."
  - "pi_thr validated to (0.5, 1.0] so the PFER denominator (2·pi_thr−1) is strictly positive."
  - "Local subsample_rows helper (avoids cross-module coupling to the explain/bootstrap copies)."

verification:
  module_tests: "5/5 pass — cargo test -p fdars-core --features linalg,parallel --lib boosting_regression::stability"
  tests:
    - stability_selects_strong_signal (strong predictor's selection freq exceeds every unrelated predictor's; strong predictor in the stable set)
    - stability_freqs_in_range (all frequencies in [0,1]; PFER bound finite and >= 0)
    - stability_is_deterministic_under_seed (two runs, same seed → identical freq/stable_set/pfer_bound)
    - stability_errors_on_invalid_params (pi_thr <= 0.5 and n_resamples = 0 → FdarError)
    - stability_errors_on_tiny_n (⌊n/2⌋ < 3 → FdarError)

notes:
  - "Implemented inline by the orchestrator after transient API (529 Overloaded) errors prevented executor-subagent dispatch."
  - "Full crate-wide clippy + fmt + test gate runs at phase end (out-of-band per repo 600s-watchdog convention)."

commits:
  - "ac4b16ca feat(43-05): FDboost-style stability selection (REG-06-05)"
---

# Plan 43-05 — FDboost-style stability selection (REG-06-05)

Implemented `stability_selection` in `fdars-core/src/boosting_regression/stability.rs`.
It draws `n_resamples` subsamples of ⌊n/2⌋ rows without replacement (seeded per replicate),
fits `boost_fosr` on each, and aggregates per-base-learner selection frequencies across
resamples. Base-learners with frequency ≥ `pi_thr` form the stable set; the
Meinshausen–Bühlmann PFER bound `q²/((2·pi_thr−1)·p)` is reported as an informational
diagnostic.

**Requirement REG-06-05** — "run FDboost-style stability selection over the boosting
base-learners and obtain per-learner selection frequencies / a stable predictor set" — is
satisfied.

## Verification

5/5 module tests pass, including strong-signal selection, frequency-range checks, and
bit-identical determinism under a fixed seed. Crate-wide clippy/fmt/test gate runs at
phase end.
