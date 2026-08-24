---
phase: 44
slug: fem-pde-smoothing-irregular-2d-domains
status: passed
verified: 2026-08-24
verifier: orchestrator-inline
reason: >
  Verification performed inline against objective, reproducible evidence
  (crate-wide clippy --all-targets clean after fixing one test-code map_clone
  warning the module-scoped runs missed, cargo fmt clean, full test suite 2568
  lib unit tests + all integration suites + 178 doctests passing with 0 failures,
  including the new fem_smoothing (14) and smooth_basis positive/monotone tests).
  Independent gsd-verifier subagent dispatch has been unreliable this session
  (transient API 529 / connection-drop); the objective gate results are authoritative.
requirements_verified: [REP-02-01, REP-02-02, REP-02-03, REP-02-04]
---

# Phase 44 — Verification (FEM/PDE Smoothing on Irregular 2D Domains)

**Goal:** A user can smooth scattered observations over an irregular 2D domain using a
finite-element basis with PDE (Laplacian) regularization — plus obtain shape-constrained
(positive, monotone) smoothers — capabilities absent from fdars' regular-grid 2D FOSR strength.

**Verdict: PASSED** — all four requirements are delivered as `Result`-returning public
functions (new `fem_smoothing.rs` module + additive `smooth_basis.rs` smoothers), re-exported
at the crate root and prelude, with inline recovery + error-path tests, and the whole crate
passes clippy `--all-targets`, `cargo fmt --check`, and the full test suite.

## Objective quality gates

| Gate | Command | Result |
|------|---------|--------|
| Lint (incl. test/bench code) | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | ✅ clean (one test-code `map_clone` warning fixed — the CI-style gate the module-scoped executor runs missed) |
| Format | `cargo fmt -p fdars-core --check` | ✅ clean |
| Full test suite | `cargo test -p fdars-core --features linalg,parallel` | ✅ 2568 lib + 12/55/50/107/77/1/56/16/34 integration + 178 doctests, **0 failures** |

## Per-requirement verdicts (goal-backward)

| Req | Must-have | Delivered symbol | Evidence | Verdict |
|-----|-----------|------------------|----------|---------|
| REP-02-01 | Linear P1 FE basis over a triangulated 2D mesh + basis eval + mass/stiffness assembly | `assemble_fem_matrices`, `fem_basis_eval`, `FemSmoothResult` (`fem_smoothing.rs`) | 7 tests: partition-of-unity, linear-field interpolation exactness, stiffness symmetry + row-sum≈0 (constant null space) + mass PD, and 4 error paths (bad index, outside mesh, empty mesh, degenerate triangle) | ✅ passed |
| REP-02-02 | PDE (Laplacian) -regularized surface smoothing of scattered obs + fitted surface + diagnostics | `fem_smooth`, `fem_smooth_gcv`, `fem_predict` (`fem_smoothing.rs`) | tests (14 total in module): SR-PDE surface recovery within tolerance, interpolation limit as λ→0, finite edf∈(0,n]/GCV, GCV λ-search, predict linear-exactness, obs-outside-mesh error | ✅ passed |
| REP-02-03 | Positive (nonnegative-guaranteed) smoother | `smooth_positive` + `SmoothPositiveResult` (`smooth_basis.rs`) | tests: fitted > 0 everywhere, recovers a known positive curve, non-positive input → FdarError (no ln NaN) | ✅ passed |
| REP-02-04 | Ramsay integral-of-exp monotone smoother | `smooth_monotone` + `SmoothMonotoneResult` (`smooth_basis.rs`) | tests: structurally monotone fit (f'=β₁·exp(w) constant sign), increasing recovery, decreasing direction auto-detect (β₁<0), bounded Gauss-Newton iterations, error paths | ✅ passed |

## Additive / non-breaking check

- New module `fdars-core/src/fem_smoothing.rs`; additive fns in `smooth_basis.rs`.
- Crate-root re-exports (`src/lib.rs:380`): `assemble_fem_matrices, fem_basis_eval, fem_smooth, fem_smooth_gcv, fem_predict, FemSmoothResult` + `smooth_positive, SmoothPositiveResult, smooth_monotone, SmoothMonotoneResult`.
- Prelude (`src/prelude.rs`): `FemSmoothResult`, `SmoothPositiveResult`, `SmoothMonotoneResult`.
- **Zero changes to existing public signatures** — existing `smooth_basis*` and `function_on_scalar_2d` signatures untouched; the irregular-mesh FEM is disjoint from the regular-grid tensor 2D FOSR (no A-6 overlap).
- **No new crate dependencies** — in-house linear-FE assembly + dense `cholesky_*`, `bspline_basis`, `bspline_penalty_matrix`, `smooth_basis`.

## Scope fences honored

- v1 = 2D triangulated meshes only (no 3D tetrahedral FEM).
- Neumann natural BC; isotropic Laplacian penalty; dense solve (sparse deferred).
- Documented divergences from fdaPDE / Ramsay baselines in rustdoc.

## Notes / tech debt

- Positive smoother carries a documented log-domain retransformation (Jensen) bias — accepted for v1.
- `smooth_monotone` recovery test uses `max_iter=100` for a steep logistic (documented); monotonicity is structural regardless of convergence.
- Dense O(N³) GCV inverse → v1 recommends mesh N ≲ 2000 (documented in rustdoc); sparse scaling deferred.
- `44-VALIDATION.md` remains `status: draft` (Nyquist per-task map seeded pre-plan; consistent with prior milestones' deferred Nyquist TODO).
