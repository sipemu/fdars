---
phase: 69-discrete-wavelet-transform-primitive
plan: 02
subsystem: api
tags: [wavelet, dwt, mallat-pyramid, daubechies, fdmatrix, rust]

# Dependency graph
requires:
  - phase: 69-01
    provides: single_level_analysis/synthesis exact-adjoint engine, FilterBank/filter_bank, WaveletFamily/BoundaryMode enums
provides:
  - "WaveletCoeffs multi-level DWT result struct (non_exhaustive, serde-gated, must_use accessors)"
  - "max_level(signal_len, family) pyramid-depth helper"
  - "decompose/reconstruct multi-level Mallat pyramid (auto + explicit level, exact signal_len round-trip)"
  - "decompose_matrix FdMatrix batch path (one WaveletCoeffs per row, identical to per-row decompose)"
affects: [70-wavelet-regressors, 71-wavelet-public-surface]

# Actuals
actuals:
  tokens: 15000
  tasks: 3
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Multi-level pyramid stores per-level input lengths (level_lens) so reconstruct recovers each synthesis target exactly"
    - "Batch path == per-row slice path (asserted by parity test), respecting column-major FdMatrix row iteration"

key-files:
  created: []
  modified:
    - fdars-core/src/wavelet/mod.rs

key-decisions:
  - "Detail bands stored finest-first (details[0] == level-1 detail); documented in field doc"
  - "Store per-level input lengths (level_lens, pub(crate)) rather than deriving via halving — keeps reconstruction exact for non-power-of-2 lengths"
  - "Single consolidated implementation commit (all 3 tasks touch the one file mod.rs; splitting hunks risked fmt drift for no traceability gain)"

patterns-established:
  - "Pyramid-depth guard: max_level = floor(log2(n/(filter_len-1))), InvalidParameter on n==0/too-short/unsupported family"
  - "WaveletCoeffs mirrors kshape #[non_exhaustive] + serde-gated + #[must_use] accessor house pattern"

requirements-completed: [WAV-01, WAV-02]

coverage:
  - id: D1
    description: "max_level returns floor(log2(n/(filter_len-1))) and errors on n==0 / sub-one-level / unsupported family"
    requirement: WAV-02
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#max_level_haar_is_log2_of_n, max_level_db4_uses_filter_len_minus_one, max_level_rejects_zero_length, max_level_rejects_too_short_signal, max_level_rejects_unsupported_family"
        status: pass
    human_judgment: false
  - id: D2
    description: "Multi-level decompose->reconstruct round-trips <=1e-10 for Haar+db2..db10 across >=2 levels under both boundary modes on non-power-of-2 lengths"
    requirement: WAV-01
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#multi_level_round_trip_periodic_all_families, multi_level_round_trip_symmetric_non_power_of_two, multi_level_round_trip_periodic_non_power_of_two, signal_len_preserved_odd_and_even"
        status: pass
    human_judgment: false
  - id: D3
    description: "Auto/explicit level handling: auto == max_level; level 0 or >max returns InvalidParameter; empty signal errors"
    requirement: WAV-02
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#auto_level_equals_max_level, explicit_level_out_of_range_is_invalid, decompose_rejects_empty_signal, reconstruct_rejects_inconsistent_coeffs"
        status: pass
    human_judgment: false
  - id: D4
    description: "decompose_matrix batch path returns one WaveletCoeffs per row identical to per-row decompose, each round-trips both modes; empty matrix and unsupported order error"
    requirement: WAV-01
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/mod.rs#decompose_matrix_matches_per_row_slice_path, decompose_matrix_round_trip_both_modes, decompose_matrix_rejects_empty_matrix, unsupported_order_surfaces_invalid_parameter"
        status: pass
    human_judgment: false

# Metrics
duration: 15min
completed: 2026-09-04
status: complete
---

# Phase 69-02: Multi-level DWT Pyramid & Batch Path Summary

**Multi-level Mallat pyramid (WaveletCoeffs / decompose / reconstruct) plus the FdMatrix batch path, reconstructing <=1e-10 for Haar+db2..db10 across >=2 levels, both boundary modes, and non-power-of-2 lengths**

## Performance

- **Duration:** ~15 min
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- `max_level(signal_len, family)` pyramid-depth helper — `floor(log2(n/(filter_len-1)))`, with descriptive `InvalidParameter` on empty/too-short/unsupported input.
- `WaveletCoeffs` result struct following the house pattern (`#[non_exhaustive]`, serde-gated, `Debug/Clone/PartialEq`, `#[must_use]` accessors `levels`/`signal_len`/`approx`/`detail`/`family`/`mode`); detail bands finest-first; per-level input lengths stored for exact inversion.
- `decompose`/`reconstruct` multi-level orthogonal DWT — auto level defaults to `max_level`, explicit level validated `1..=max_level`; round-trip returns exactly `signal_len` samples and reconstructs <=1e-10 for Haar + db2..db10 across >=2 levels under both periodic and symmetric modes on non-power-of-2 lengths.
- `decompose_matrix` FdMatrix batch path — one `WaveletCoeffs` per row (curve), asserted byte-identical to per-row `decompose`.
- Full invalid-input gate: empty signal/matrix, unsupported family order, level > max — each a specific `FdarError` with no NaN/panic.

## Task Commits

1. **Tasks 1-3 (max_level + WaveletCoeffs, decompose/reconstruct, decompose_matrix + invalid-input gate)** - `84ef914b` (feat)
2. **Plan summary** - (docs commit for this file)

_Note: all three tasks modify the single file `fdars-core/src/wavelet/mod.rs`; see Deviations._

## Files Created/Modified
- `fdars-core/src/wavelet/mod.rs` - added multi-level pyramid (`WaveletCoeffs`, `max_level`, `decompose`, `reconstruct`, `decompose_matrix`) and 17 new tests; refreshed the scoped `#![allow(dead_code)]` rationale to point at the still-unused `rec_lo`/`rec_hi` synthesis-filter surface.

## Decisions Made
- **Detail bands finest-first** (`details[0]` == level-1 detail), documented on the field.
- **Store per-level input lengths** (`level_lens`, `pub(crate)`) instead of re-deriving by halving, so `reconstruct` recovers each level's exact synthesis target length and stays exact on non-power-of-2 lengths (the key length-bookkeeping link the plan flagged).
- **Single implementation commit for all 3 tasks** — the three tasks all edit the one file with interleaved code/tests; splitting the hunks across three commits gave no traceability benefit and risked `cargo fmt` drift under `--no-verify`.

## Deviations from Plan

### Test-fixture adjustments (self-corrected during Task 2)
**1. Round-trip test signal lengths raised to guarantee >=2 useful levels**
- **Found during:** Task 2 (multi-level round-trip tests)
- **Issue:** Initial tests used n=64 with `Some(3)` and n=37 auto, but `max_level` for large filters (db6/db8/db10, `filter_len-1` up to 19) legitimately yields fewer than 3 (and fewer than 2 at n=37) levels — the correct behavior of the `floor(log2(n/(filter_len-1)))` formula, not a bug.
- **Fix:** Periodic >=2-level test uses n=256 with `Some(2)`; symmetric non-power-of-2 test uses n=201 (auto, >=2 for all families); added a periodic non-power-of-2 (n=201) case so SC2 covers both modes. No tolerance was loosened; the 1e-10 gate is intact.
- **Files modified:** fdars-core/src/wavelet/mod.rs (tests only)
- **Verification:** all 36 wavelet tests pass.
- **Committed in:** 84ef914b

**Impact on plan:** Test fixtures only; the `max_level` formula and all production logic match the plan exactly. No scope creep.

## Issues Encountered
- **Pre-existing serde build break (unrelated):** `cargo build -p fdars-core --features serde` fails with 4 errors, all from `shapelet/classifier.rs` embedding a non-serde `ClassifFit` (documented in MEMORY.md). Confirmed zero errors originate from the new wavelet types — `WaveletCoeffs` serde-derives cleanly. Task 1's "not newly broken" criterion is satisfied; no second break introduced.

## Gate Results
- `cargo test -p fdars-core --features linalg,parallel wavelet::` — **36 passed, 0 failed**.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` — **clean**.
- `cargo fmt --check` — **clean**.
- `cargo build -p fdars-core --features serde` — pre-existing shapelet break only; no wavelet-originated errors.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The multi-level slice API and the FdMatrix batch seam are ready for Phase 70's wavelet regressors.
- Public wavelet surface remains reachable only as `crate::wavelet::...`; crate-root/prelude re-exports are still deferred to Phase 71 (not added here).

---
*Phase: 69-discrete-wavelet-transform-primitive*
*Completed: 2026-09-04*
