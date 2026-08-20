---
phase: 30-interval-testing-procedure-family
verified: 2026-08-20T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 30: Interval Testing Procedure Family — Verification Report

**Phase Goal:** A user can run the ITP family that fdatest provides — one-population and two-population interval-wise tests over B-spline and Fourier bases with domain-selective adjusted p-values, plus interval-wise FLM coefficient testing — via a new inference/itp.rs, reusing the shipped INF-01 permutation infrastructure and basis/ projection, without any existing inference code changing.

**Verified:** 2026-08-20T00:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Three public Result-returning entry points in inference/itp.rs, crate-root re-exported: `itp_one_pop`, `itp_two_pop`, `itp_flm`, each accepting a basis choice and returning per-component adjusted p-values via `ItpResult` | VERIFIED | All three functions exist with `pub fn … -> Result<ItpResult, FdarError>` signatures at lines 280, 490, 651 of itp.rs; re-exported in inference/mod.rs line 40 and crate root lib.rs lines 227–228 |
| 2 | Each test returns domain-selective adjusted p-values (ITP interval-wise closure adjustment) — sub-interval identification, not just a global p-value | VERIFIED | `pval_correct()` at line 183 implements the cone-walk closure: `adjusted_pvalues[k]` = max over all contiguous intervals containing k; `ItpResult.adjusted_pvalues` carries per-component values (one per basis function) |
| 3 | On synthetic data with a localized between-group difference on a known sub-interval, adjusted p-values are small on the differing interval; in null case non-significant everywhere | VERIFIED | Tests `one_population_localized` (min adj-p < 0.05 with shift on [0.4,0.6]), `one_population_null` (max adj-p > 0.10), `two_population_localized` (min adj-p < 0.05), `two_population_null` (max adj-p > 0.10), `flm_effect` (min adj-p < 0.05), `flm_null` (max adj-p > 0.10) — all 13 tests passed, confirmed by live test run |
| 4 | ITP family reuses INF-01 permutation infra + basis/ projection, adds no new crate dependency, invalid inputs return FdarError not panic | VERIFIED | `fdata_to_basis` called directly; `StdRng::seed_from_u64(seed)` / `(n_ge+1)/(n_perm+1)` formula match INF-01 pattern; `shuffle_itp` is an inline private copy of `permutation::shuffle_labels` (that fn is private); git diff shows no change to fdars-core/Cargo.toml; error-path tests for all three entry points (`one_population_error_paths`, `two_population_error_paths`, `flm_error_paths`) confirmed to pass |
| 5 | Existing inference/ (INF-01/INF-02) and basis/ projection keep public signatures unchanged; full suite + clippy --all-targets --features linalg,parallel green | VERIFIED | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` exits clean (0 warnings); `ProjectionBasisType` received only an additive `#[cfg_attr(feature = "serde", …)]` attribute (line 18 projection.rs) — no variant or signature change; inference/mod.rs additions are purely additive (`mod itp; pub use itp::{…}`) |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/inference/itp.rs` | New file with ItpResult + three entry points + helpers + 13 inline tests | VERIFIED | 1454 lines; fully substantive with rank_transform, fisher_cf, build_pval_matrix, pval_correct helpers; itp_one_pop, itp_two_pop, itp_flm public fns; 13 tests |
| `fdars-core/src/inference/mod.rs` | `mod itp; pub use itp::{itp_flm, itp_one_pop, itp_two_pop, ItpResult}` | VERIFIED | Line 33: `mod itp;`, line 40: `pub use itp::{itp_flm, itp_one_pop, itp_two_pop, ItpResult};` |
| `fdars-core/src/lib.rs` | Crate-root re-export of all three entry points + ItpResult | VERIFIED | Lines 227–228: `itp_flm, itp_one_pop, itp_two_pop` and `ItpResult` included in inference pub use block |
| `fdars-core/src/basis/projection.rs` | Additive serde attribute on ProjectionBasisType | VERIFIED | Line 18: `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` on `ProjectionBasisType` enum |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `itp_one_pop` / `itp_two_pop` / `itp_flm` | `basis::projection::fdata_to_basis` | Direct call at lines 320, 515, 521, 697 | WIRED | All three entry points call `fdata_to_basis` with the caller-supplied `basis_type` |
| `itp_one_pop` / `itp_two_pop` / `itp_flm` | `pval_correct` closure adjustment | Internal call chain: rank_transform → build_pval_matrix → pval_correct | WIRED | All three fns terminate with `pval_correct(&pval_matrix, p)` → `adjusted_pvalues` in ItpResult |
| `inference/itp` | `inference/mod.rs` | `mod itp; pub use` | WIRED | Confirmed in mod.rs |
| `inference` module | `lib.rs` crate root | `pub use inference::{…}` block | WIRED | Lines 226–229 of lib.rs |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 13 ITP inline tests pass | `cargo test -p fdars-core --features linalg,parallel --lib inference::itp` | 13 passed, 0 failed, 0.05s | PASS |
| Clippy clean across all targets | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | Finished with 0 warnings | PASS |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None | — | — | No TBD/FIXME/XXX/TODO/HACK markers found in itp.rs; no stub patterns detected |

### Documented Intentional Divergences (not gaps)

The following are documented in rustdoc and are by design, not defects:

1. **`itp_flm` response-permutation vs R partial-residual** (RESEARCH Assumption A2): Uses `shuffle y` instead of `ITPlmbspline` partial-residual method. Documented in rustdoc at line 621–633.
2. **Raw p = `(n_ge+1)/(n_perm+1)`** vs R's `n_ge/B`: INF-01 convention avoiding zero p-values. Documented in ItpResult rustdoc and `pval_correct` docstring.
3. **Internal closure matrix uses `n_ge/n_perm`** (no +1): Matches R source for the pval_matrix; only the returned `raw_pvalues` field uses the +1 correction.

---

## Summary

All 5 success criteria are met with direct codebase evidence:

- `inference/itp.rs` exists and is fully implemented (1454 lines, not a stub).
- Three public `Result<ItpResult, FdarError>`-returning entry points with correct signatures are present.
- Crate-root re-export wiring is confirmed in both `inference/mod.rs` and `lib.rs`.
- Domain-selective closure adjustment is implemented via `pval_correct` (cone walk), not a global p-value.
- Inline tests assert localized-difference significance and null non-significance for all three test families.
- `pval_correct_hand_computed` unit test locks the index arithmetic with a hand-traced p=4 example.
- No new Cargo.toml dependency added.
- All 13 ITP tests pass live; clippy clean.
- No existing public signatures in inference/ or basis/ changed (additive only).

---

_Verified: 2026-08-20T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
