---
phase: 70-wavelet-domain-regressors-wcr-wnet
plan: 01
subsystem: regression
tags: [wavelet, dwt, pcr, pls, scalar-on-function, fda]

requires:
  - phase: 69-wavelet-dwt-primitive
    provides: DWT primitive (decompose_matrix, reconstruct, WaveletCoeffs, WaveletFamily, BoundaryMode, max_level)
provides:
  - "crate::wavelet::regression module: wcr() wavelet-domain scalar-on-function regressor (PCR + PLS)"
  - "WcrConfig, WcrMethod, WcrResult public types"
  - "pub(crate) curves_to_coeff_design() — curves -> concatenated wavelet-coefficient design (n x P) + CoeffLayout"
  - "pub(crate) coeff_weights_to_beta_t() — P-length coefficient-space weights -> time-domain beta(t) via inverse DWT"
  - "pub(crate) CoeffLayout struct"
affects: [70-02-wnet, 71-wavelet-exports-predict]

actuals:
  tokens: 8500
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Wavelet-coefficient design seam: DWT-transform every curve, concatenate [approx ++ details finest-first] into one P-length row, fit a reuse-first coefficient-space regressor"
    - "Method-agnostic plain-dot beta_coeff recovery: regress centered fitted contribution onto the full-rank coefficient design (normal equations) instead of the integration-weighted score projection"

key-files:
  created:
    - fdars-core/src/wavelet/regression.rs
  modified:
    - fdars-core/src/wavelet/mod.rs

key-decisions:
  - "beta_coeff recovered by regressing (fitted - intercept) onto the column-centered coefficient design, not via the recover_beta_t/PLS-weight projection — the projection recovers beta_coeff in each method's INTERNAL integration-weighted inner product (sqrt-weighted for PCR's SVD, int-weighted for PLS's NIPALS), which does not equal the plain functional dot product beta(t) must act by. Direct regression on the full-column-rank (P<=n) design is exact and method-agnostic."
  - "Coefficient index treated as an abstract basis: PCR/PLS receive a uniform 0..P argvals grid (Simpson weights over it)."
  - "Tasks 1-3 implemented together in a single cohesive submodule + one atomic feat commit (the tracer PCR path, the PLS expansion, and the validation gate all live in one file); the tracer path was verified green before adding PLS/validation."

patterns-established:
  - "Shared pub(crate) DWT-regression seams (curves_to_coeff_design, coeff_weights_to_beta_t, CoeffLayout) that Plan 70-02 wnet reuses verbatim — only the coefficient-space fit differs."

requirements-completed: [WAV-03]

coverage:
  - id: D1
    description: "wcr fits via PCR on the wavelet-coefficient design and recovers a known beta(t) on spanning full-rank synthetic data within 1e-6 relative L2"
    requirement: "WAV-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wcr_pcr_recovers_known_beta_t_on_spanning_design"
        status: pass
    human_judgment: false
  - id: D2
    description: "wcr fits via PLS on the same design and recovers the same known beta(t) within tolerance"
    requirement: "WAV-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wcr_pls_recovers_known_beta_t_on_spanning_design"
        status: pass
    human_judgment: false
  - id: D3
    description: "wcr validates dim/param mismatch to descriptive FdarError (never panics) and produces finite/NaN-free beta(t), fitted values, residuals"
    requirement: "WAV-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wcr_rejects_too_few_rows, wcr_rejects_mismatched_y_len, wcr_rejects_zero_ncomp, wcr_surfaces_unsupported_family, wcr_surfaces_level_out_of_range, wcr_finite_outputs_both_methods"
        status: pass
    human_judgment: false
  - id: D4
    description: "Shared curves_to_coeff_design + coeff_weights_to_beta_t seams exist and round-trip (concat design shape/layout correct; coeff->beta_t inverts decompose to <=1e-10)"
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#curves_to_coeff_design_layout_and_shape, coeff_weights_to_beta_t_inverts_decompose, coeff_weights_to_beta_t_rejects_wrong_length"
        status: pass
    human_judgment: false

duration: 26min
completed: 2026-09-04
status: complete
---

# Phase 70 Plan 01: wcr Wavelet-Domain Regressor Summary

**`wcr` scalar-on-function regressor fitting PCR or PLS on per-curve concatenated wavelet-coefficient designs, recovering beta(t) by inverse DWT — both paths recover a known beta(t) on a spanning full-rank design to <1e-6 rel L2.**

## Performance

- **Duration:** ~26 min
- **Completed:** 2026-09-04T20:21:43Z
- **Tasks:** 3 (implemented together, one atomic commit)
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- New `crate::wavelet::regression` submodule: `wcr(data, y, config)` transforms every curve to its wavelet-coefficient vector (Phase 69 `decompose_matrix`), fits PCR (`fdata_to_pc_1d`) or PLS (`fdata_to_pls_1d`) in coefficient space, and reconstructs beta(t) via inverse DWT (`reconstruct`).
- Public API: `WcrConfig` (db4/Periodic/auto-level/ncomp/PCR default), `WcrMethod {Pcr, Pls}`, `WcrResult` (intercept, beta_t, fitted_values, residuals, ncomp, method, coeff_weights, family, mode, level).
- Shared `pub(crate)` seams for Plan 70-02 (`wnet`): `curves_to_coeff_design`, `coeff_weights_to_beta_t`, `CoeffLayout`.
- Both PCR and PLS beta(t)-recovery tests green on an n=120, m=32 spanning full-rank pseudo-random (LCG) design; validation gate + finite-output tests green.

## Task Commits

1. **Tasks 1-3: wcr regressor (module + PCR tracer + PLS + validation)** - `dffec836` (feat)

_Tasks were implemented cohesively in one submodule; the PCR tracer path was verified green before expanding to PLS and the validation gate._

## Files Created/Modified
- `fdars-core/src/wavelet/regression.rs` - New submodule: `wcr`, `WcrConfig`, `WcrMethod`, `WcrResult`, `CoeffLayout`, shared seams `curves_to_coeff_design` / `coeff_weights_to_beta_t`, self-contained OLS/Cholesky helpers, plus 12 inline tests.
- `fdars-core/src/wavelet/mod.rs` - Added `pub mod regression;`.

## Decisions Made
- **beta_coeff recovery via direct regression, not score projection.** The plan's suggested `recover_beta_t`/PLS-weight projection (`Sigma_k coeff_k * rotation/weight_k`) recovers beta_coeff in each method's INTERNAL integration-weighted inner product — sqrt-weighted for PCR's weighted SVD, int-weighted for PLS's NIPALS. That does not equal the plain functional dot product beta(t) must act by (`decompose -> concatenate -> dot`), so the naive projection gave ~0.5 (PCR) / ~0.39 (PLS) relative recovery error. Since the reduced-rank fitted values lie exactly in the span of the full-column-rank (P<=n) coefficient design, regressing `(fitted - intercept)` onto the column-centered design recovers the exact, method-agnostic plain-dot beta_coeff. This makes both recoveries exact (<1e-6) and is documented inline. See Deviations.
- **Abstract-basis integration grid.** Wavelet coefficients carry no intrinsic spacing, so PCR/PLS receive a uniform `0..P` argvals grid (documented in the fn doc).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness] beta_coeff recovery method changed from score-projection to direct design regression**
- **Found during:** Task 1 (PCR tracer) — the recovery test fired at ~0.51 rel L2 error; Task 2 PLS at ~0.39.
- **Issue:** The plan's `recover_beta_t`-style projection recovers beta_coeff in the fit's internal integration-weighted inner product (sqrt-weighted SVD / int-weighted NIPALS), which is not the plain functional dot product that beta(t) — as the inverse DWT of the coefficient-space coefficient — must satisfy against the concatenated-coefficient design. No single y-generation aligns both methods (they use different internal weightings), and per project-memory rule the tolerance must NOT be loosened.
- **Fix:** Added `recover_coeff_weights(design, fitted, intercept)`: regress the centered fitted contribution onto the column-centered full-rank coefficient design via normal equations (tiny diagonal ridge for numerical stability). This yields the exact plain-dot beta_coeff independently of PCR/PLS internals. The PCR/PLS calls still supply the reduced-rank scores that produce the fit; only the beta_coeff extraction changed. The shared `coeff_weights_to_beta_t` seam is unchanged.
- **Files modified:** fdars-core/src/wavelet/regression.rs
- **Verification:** `wcr_pcr_recovers_known_beta_t_on_spanning_design` and `wcr_pls_recovers_known_beta_t_on_spanning_design` pass at <1e-6 rel L2; residuals ~0 on the noiseless full-rank fit.
- **Committed in:** dffec836

---

**Total deviations:** 1 auto-fixed (1 correctness).
**Impact on plan:** The reuse-first PCR/PLS fit is unchanged; only beta_coeff extraction was corrected to match the plain-dot semantics beta(t) requires. Shared seams (`curves_to_coeff_design`, `coeff_weights_to_beta_t`, `CoeffLayout`) are exactly as specified for Plan 70-02. No scope creep.

## Issues Encountered
None beyond the recovery-semantics correction documented above.

## Gate Results
- `cargo test -p fdars-core --features linalg,parallel wavelet` — **59 passed, 0 failed** (47 prior DWT + 12 new wcr/seam tests).
- `cargo test ... wavelet::regression` — **12 passed, 0 failed** (PCR + PLS recovery, validation gate, finite outputs, seam round-trips).
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` — **clean**.
- `cargo fmt --check` — **no diff**.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 70-02 (`wnet`) can reuse `curves_to_coeff_design`, `coeff_weights_to_beta_t`, and `CoeffLayout` verbatim — only the coefficient-space fit (elastic-net) differs.
- No crate-root/prelude re-exports were added (Phase 71 owns exports, as locked).
- `WcrResult` carries `family`/`mode`/`level`/`coeff_weights` for a future Phase 71 `predict`.

## Self-Check: PASSED

---
*Phase: 70-wavelet-domain-regressors-wcr-wnet*
*Completed: 2026-09-04*
