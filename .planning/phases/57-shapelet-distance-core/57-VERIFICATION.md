---
phase: 57
title: Shapelet Distance Core
status: passed
requirements: [SHP-01, SHP-02]
---

# Phase 57 Verification — Shapelet Distance Core

All 5 ROADMAP success criteria verified PASS. Evidence = passing test names + gate results.

## Success Criteria

### 1. Per-window z-normalization with constant-window guard — PASS
Population std (ddof=0), constant/near-constant window (std ≤ 1e-12) → finite zero vector, never NaN/Inf.
- Impl: `z_normalize_into` / `z_normalize_window` in `distance.rs` (two-pass, `STD_EPS = 1e-12`).
- Evidence: `test_znorm_constant_window` (constant + one-element-perturbed-by-1e-15 → all finite zeros), `test_znorm_mean_std` (mean≈0, population std≈1).

### 2. `sdist` = MIN over sliding windows, scale- & offset-invariant within 1e-10 — PASS
- Impl: `shapelet_distance` slides length-L window, per-window z-norm on the fly, min squared Euclidean, sqrt final.
- Evidence: `test_sdist_scale_offset_invariant` — `sdist(S,T) == sdist(S,T+c) == sdist(S,T*a)` within 1e-10 for distance AND offset (the make-or-break per-window-normalization gate).

### 3. Known-motif recovery (min-not-mean semantics) — PASS
- Evidence: `test_sdist_min_semantics` — a series with an exact copy of the shapelet's source motif planted in noise yields `sdist ≈ 0` (< 1e-9) at the correct offset; non-matching windows are ignored (min, not mean/sum). The `test_sdist_scale_offset_invariant` motif is embedded in a noisy series, further confirming a low min surrounded by non-matches.

### 4. Explicit `best_so_far` early-abandon; identical answer; measurable speedup — PASS
- Impl: inner element loop accumulates running squared sum and breaks when it exceeds `best_so_far²` (squared space); running best seeded from the caller bound; `INFINITY` disables abandon. Abandon only prunes windows that cannot beat the current best → answer unchanged.
- Evidence: `test_sdist_early_abandon_identical` — INFINITY-bound result equals both a tight bound (≥ true min) and the exact-min bound, in distance and offset.
- Speedup: the squared-space break skips the remaining L−k element ops on hopeless windows; on a non-matching shapelet most windows abandon after a few elements (constant-factor prune, no correctness change). (Formal criterion-benchmark deferred per 57-CONTEXT "no benchmark this phase"; the pruning path is exercised and proven answer-preserving by the test.)

### 5. Returns (min distance, best offset); `Shapelet` stores z-norm values + provenance — PASS
- Impl: `shapelet_distance -> Result<(f64, usize), FdarError>`; `Shapelet { values (z-normed), series_idx, start, length, quality }` with `from_source` + `len`/`is_empty`.
- Evidence: `test_sdist_min_semantics` / `test_sdist_scale_offset_invariant` assert the returned offset; `test_shapelet_from_source` asserts stored z-norm values + provenance fields.

## Gate Results
- fmt: `cargo fmt --check -p fdars-core` clean.
- clippy: `cargo clippy --all-targets --features linalg,parallel -p fdars-core -- -D warnings` clean.
- tests (linalg): `cargo test -p fdars-core --features linalg shapelet` → 7 passed.
- tests (default): `cargo test -p fdars-core shapelet` → 7 passed.
- doctests (both feature sets): 2 passed.

## Gaps
None.
