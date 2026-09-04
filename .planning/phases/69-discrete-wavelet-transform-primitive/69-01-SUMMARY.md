---
phase: 69-discrete-wavelet-transform-primitive
plan: 01
subsystem: wavelet
tags: [dwt, daubechies, haar, wavelet, signal-processing, orthonormal-filter-bank]

requires: []
provides:
  - "fdars-core/src/wavelet module (declared pub mod wavelet; in lib.rs, no re-exports)"
  - "WaveletFamily enum (Haar / Daubechies(N), from_db_order) + BoundaryMode enum (Periodic default / Symmetric)"
  - "pub(crate) FilterBank + filters::filter_bank(): orthonormal db1..db10 filter banks"
  - "pub(crate) single_level_analysis / single_level_synthesis: exact-adjoint single-level DWT step under periodic + symmetric boundaries"
affects: [wavelet, "70-multi-level-dwt", "71-wavelet-public-api", wavelet-regression]

actuals:
  tokens: 21000
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Single hardcoded dec_lo table per Daubechies family; dec_hi/rec_lo/rec_hi derived programmatically (single source of truth, no re-typed sign-flipped tables)"
    - "Both boundary modes route through one orthogonal even-length periodic core whose transpose is its exact inverse (perfect reconstruction proven, not approximated)"

key-files:
  created:
    - fdars-core/src/wavelet/mod.rs
    - fdars-core/src/wavelet/filters.rs
  modified:
    - fdars-core/src/lib.rs

key-decisions:
  - "Odd-length periodic handled by extending to n+1 (duplicate last sample) so the orthogonal even-length core applies; coefficient count stays ceil(n/2)"
  - "Symmetric mode implemented via half-point mirror extension to length 2n (n coefficients), reconstructed then truncated — independently perfect-reconstructing, decoupled from the periodic path"
  - "Synthesis is the transpose of the even-length periodic analysis core (uses dec_lo/dec_hi), which is the exact inverse because that operator is orthogonal; rec_lo/rec_hi are exposed on FilterBank for downstream use and verified by invariant tests"
  - "single_level_analysis/synthesis kept pub(crate) (per plan spec) to match the pub(crate) FilterBank they consume — no public API surface added this phase"

patterns-established:
  - "wavelet module dead_code allow scoped to the module (foundational pub(crate) API consumed by the next plan, reached only from own tests until then)"

requirements-completed: [WAV-01, WAV-02]

coverage:
  - id: D1
    description: "Haar single-level known-answer coefficients match hand-computed sum/difference over sqrt(2) (SC3)"
    requirement: WAV-01
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#haar_known_answer_coefficients"
        status: pass
    human_judgment: false
  - id: D2
    description: "Single-level analysis+synthesis round-trips within 1e-10 relative for Haar + db2..db10 under periodic AND symmetric on non-power-of-2 lengths (SC1, SC2)"
    requirement: WAV-01
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#round_trip_periodic_non_power_of_two"
        status: pass
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#round_trip_symmetric_non_power_of_two"
        status: pass
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#db4_even_and_odd_round_trip_both_modes"
        status: pass
    human_judgment: false
  - id: D3
    description: "Orthonormal db1..db10 filter tables: dec_lo sums to sqrt(2), unit L2 norm, QMF + time-reverse relations, filter_len==2N (WAV-02)"
    requirement: WAV-02
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/filters.rs#dec_lo_sums_to_sqrt2, dec_lo_has_unit_l2_norm, dec_hi_is_quadrature_mirror_of_dec_lo, synthesis_filters_are_time_reverse_of_analysis, filter_len_is_twice_the_order"
        status: pass
    human_judgment: false
  - id: D4
    description: "Unsupported family/order and empty/degenerate input return descriptive FdarError, no panic/NaN (SC4)"
    requirement: WAV-02
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/filters.rs#filter_bank_rejects_out_of_range_order + mod.rs#from_db_order_maps_correctly, empty_signal_is_invalid_parameter, reconstruction_has_no_nan_or_inf, synthesis_rejects_mismatched_coefficient_lengths"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-09-04
status: complete
---

# Phase 69 Plan 01: Discrete Wavelet Transform Primitive Summary

**Single-level orthonormal DWT engine (Haar + db2..db10) with exact-adjoint analysis/synthesis under periodic and symmetric boundaries, reconstructing any-length signals to ≤1e-10.**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-09-04T19:32:34Z
- **Tasks:** 3
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments
- Orthonormal Daubechies filter-coefficient tables for Haar/db1 through db10 (analysis-lowpass `dec_lo` hardcoded to full f64 precision in the PyWavelets orthonormal-DWT convention; `dec_hi`/`rec_lo`/`rec_hi` derived programmatically). Every family passes the sum==√2, unit-L2-norm, QMF-orthogonality, and time-reverse invariants within 1e-12.
- `WaveletFamily` (Haar / Daubechies(N), `from_db_order` mapping 1→Haar, 2..=10→Daubechies, else `InvalidParameter`) and `BoundaryMode` (Periodic default / Symmetric) config enums, serde-gated, `#[non_exhaustive]`.
- `single_level_analysis` / `single_level_synthesis` forming an exact adjoint pair under each boundary mode independently: Haar known-answer coefficients match, and round-trips for Haar + db2..db10 reconstruct within 1e-10 on even (n=36) and odd (n=37) non-power-of-2 lengths under both modes, with no NaN/inf.
- `pub mod wavelet;` declared in `lib.rs` alphabetically between `warping` and `wire`, with NO crate-root or prelude re-exports (deferred to Phase 71).

## Task Commits

1. **Task 1 (scaffold + Haar) & Task 2 (db2..db10 tables)** — `fd99d2e8` (feat) — lib.rs wiring + filters.rs (FilterBank, all filter tables, `filter_bank`)
2. **Task 1 (tracer engine) & Task 3 (full periodic+symmetric engine)** — `d4bf81ec` (feat) — mod.rs (enums + `single_level_analysis`/`single_level_synthesis` + all tests)

## Files Created/Modified
- `fdars-core/src/wavelet/filters.rs` - `FilterBank` struct, hardcoded db1..db10 `dec_lo` tables, `filter_bank()` lookup, filter-invariant tests
- `fdars-core/src/wavelet/mod.rs` - `WaveletFamily`/`BoundaryMode` enums, single-level analysis/synthesis engine, round-trip + known-answer + invalid-input tests
- `fdars-core/src/lib.rs` - `pub mod wavelet;` declaration (no re-exports)

## Decisions Made
- **Perfect-reconstruction architecture:** a naive transpose-synthesis is the exact inverse only for even-length periodic (orthogonal operator). To make BOTH modes reconstruct exactly for arbitrary length, both route through one orthogonal even-length periodic core: periodic extends odd n to n+1 (duplicate last sample, ceil(n/2) coeffs); symmetric mirror-extends to 2n (n coeffs). Synthesis reconstructs the even extension and truncates. Validated empirically (all families ≤1e-12) before wiring.
- **`dec_lo` numbers regenerated** to full f64 precision via spectral-factorization root construction (numpy), because two initial hand-transcribed tables (db3, db7) failed the sum==√2 / unit-L2 invariant — fixed the digits, never loosened tolerance (per guardrail).
- **`pub(crate)` engine fns** to match the `pub(crate) FilterBank` they consume (avoids a `private_interfaces` error and keeps zero public API this phase).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Synthesis was not an exact inverse for odd-length or symmetric mode**
- **Found during:** Task 3 (generalizing the single-level engine)
- **Issue:** The initial transpose-of-analysis synthesis (using an in-place reflected/wrapped index map) only inverts the even-length periodic case; odd-length and all symmetric round-trips diverged (rel err up to 0.34).
- **Fix:** Re-architected both modes to extend the signal to an even length and run one orthogonal even-length periodic core (`core_analysis`/`core_synthesis`), whose transpose is a proven exact inverse; synthesis truncates back to `output_len`. Verified via a standalone probe across families/lengths/modes before committing.
- **Files modified:** fdars-core/src/wavelet/mod.rs
- **Verification:** `round_trip_periodic_non_power_of_two`, `round_trip_symmetric_non_power_of_two`, `db4_even_and_odd_round_trip_both_modes` all pass ≤1e-10
- **Committed in:** d4bf81ec

**2. [Rule 1 - Bug] db3 and db7 dec_lo tables failed the normalization invariant**
- **Found during:** Task 2 (filter tables)
- **Issue:** Two hand-transcribed `dec_lo` tables did not sum to √2 / have unit L2 norm within 1e-12.
- **Fix:** Regenerated all db2..db10 `dec_lo` to full f64 precision via numpy spectral factorization; replaced tables. Did not relax the 1e-12 assert (guardrail).
- **Files modified:** fdars-core/src/wavelet/filters.rs
- **Verification:** `dec_lo_sums_to_sqrt2`, `dec_lo_has_unit_l2_norm` pass for all 11 families
- **Committed in:** fd99d2e8

**3. [Rule 1 - Lint] clippy fixes for the new module**
- **Found during:** Task 3 (final gate)
- **Issue:** `derivable_impls` (manual `Default for BoundaryMode`), `unreachable_pattern` (redundant `_` arm in `filter_bank`), `private_interfaces` (`pub` fns exposing `pub(crate) FilterBank`), `dead_code` (filter constructor/tables reached only from tests until the next plan wires them).
- **Fix:** `#[derive(Default)]` + `#[default]` on `Periodic`; removed the unreachable arm; made engine fns `pub(crate)`; scoped `#![allow(dead_code)]` to the wavelet module with an explanatory comment.
- **Files modified:** fdars-core/src/wavelet/mod.rs, fdars-core/src/wavelet/filters.rs
- **Verification:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean
- **Committed in:** fd99d2e8, d4bf81ec

**4. [Process] Tasks co-developed; committed by file grouping rather than strictly one-commit-per-task**
- **Found during:** All tasks
- **Issue:** `mod.rs` contains Task 1's tracer path and Task 3's full engine; `filters.rs` contains Task 1's Haar bank and Task 2's tables. A literal per-task commit would produce non-compiling intermediate states.
- **Fix:** Committed in two atomic, compiling commits grouped by file (filters+lib, then mod.rs), each mapping to the plan tasks it delivers. Both commits use `--no-verify` per repo protocol (pre-commit hook runs the full suite and times out); fmt + scoped tests + clippy run out-of-band and are green.
- **Impact:** No functional deviation; every task's acceptance criteria verified by passing tests.

---

**Total deviations:** 4 (2 correctness bugs auto-fixed, 1 lint cleanup, 1 process note).
**Impact on plan:** All auto-fixes essential for correctness (perfect reconstruction, filter normalization) and a clean clippy gate. No scope creep; no public API added; no new crate dependency.

## Issues Encountered
- The `serde` feature build fails on a **pre-existing** `ClassifFit`/shapelet break (documented in project memory, Phase 60) — unrelated to this phase. The new wavelet types' serde-gated derives compile fine; no new serde break introduced.

## Next Phase Readiness
- The exact-adjoint single-level DWT step is proven for every in-scope family under both boundary modes at arbitrary length — Plan 02 can iterate it into the Mallat multi-level pyramid.
- `filter_bank`, `single_level_analysis`, `single_level_synthesis` are `pub(crate)` and ready to be consumed by the multi-level/batch surface; the `#![allow(dead_code)]` can be dropped once Plan 02 wires them.

---
*Phase: 69-discrete-wavelet-transform-primitive*
*Completed: 2026-09-04*
