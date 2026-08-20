---
phase: 30-interval-testing-procedure-family
plan: "01"
subsystem: inference/itp
status: complete
tags: [inference, itp, permutation, closure-adjustment, basis-projection, functional-testing]

dependency_graph:
  requires:
    - basis/projection.rs (fdata_to_basis, ProjectionBasisType)
    - inference/mod.rs (TestResult, existing pattern)
    - inference/permutation.rs (RNG seeding pattern, p-value formula)
  provides:
    - inference/itp.rs (ItpResult, itp_one_pop, rank_transform, build_pval_matrix, pval_correct)
    - itp_one_pop + ItpResult at crate root
  affects:
    - fdars-core/src/inference/mod.rs (mod itp; pub use block)
    - fdars-core/src/lib.rs (inference re-export block)
    - fdars-core/src/basis/projection.rs (conditional serde on ProjectionBasisType — additive only)

tech_stack:
  added: []
  patterns:
    - rank-transform (descending rank / B for pseudo-p-values per permutation)
    - Fisher combining function (-2 * sum(log(p_i)), clamped at 1e-300)
    - O(p²) interval p-value matrix with circular doubling trick
    - Interval-wise closure max-adjustment (cone walk on doubled+reversed matrix)
    - sign-flip permutation null for one-population mean test
    - (n_ge + 1) / (n_perm + 1) raw p-value formula (INF-01 convention)

key_files:
  created:
    - fdars-core/src/inference/itp.rs
  modified:
    - fdars-core/src/inference/mod.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/basis/projection.rs

decisions:
  - "ItpResult fields: adjusted_pvalues, raw_pvalues, basis_type, n_basis, n_perm — fixed by CONTEXT/RESEARCH"
  - "build_pval_matrix uses n_ge / n_perm (no +1) for internal closure matrix, matching R source; only ItpResult.raw_pvalues uses (n_ge+1)/(n_perm+1)"
  - "rank_transform parallelized with iter_maybe_parallel! over components k; permutation loop is sequential (single RNG state)"
  - "pval_correct cone walk ends with corrected.reverse() restoring natural component order (RESEARCH Pitfall 3, confirmed by hand-computed test)"
  - "ProjectionBasisType gains conditional serde (additive attribute only; no variant/signature change) to enable ItpResult conditional serde"

metrics:
  duration_minutes: 45
  completed: "2026-08-20"
  tasks_completed: 2
  commits: 3
  tests_added: 6

actuals:
  tokens: 9800
  tasks: 2
  commits: 3
---

# Phase 30 Plan 01: ITP Tracer (pval_correct + itp_one_pop) Summary

**One-liner:** Interval-wise closure helper `pval_correct` (cone-walk on doubled+reversed matrix) and `itp_one_pop` wired end-to-end — basis projection to ItpResult — with 6 inline tests locking arithmetic and behavior.

## What Was Built

### New file: `fdars-core/src/inference/itp.rs`

**Private helpers (tracer core):**
- `rank_transform(t_perm, p, b)` — converts `(b, p)` permutation stat matrix to pseudo-p values via descending rank / b; parallelized over components with `iter_maybe_parallel!`
- `fisher_cf(vals)` — `-2 * Σ log(max(v, 1e-300))`; clamped to prevent NaN on 0.0 p-values (T-30-02)
- `build_pval_matrix(raw_pvalues, l, p, n_perm)` — O(p²) interval loop using circular doubling; row `p-1` = raw p-values; row `p - interval_len` = joint Fisher p-value for length-`interval_len` intervals
- `pval_correct(pval_matrix, p)` — cone walk on doubled+reversed matrix (`get_2x_rev`), computes max over all contiguous intervals containing each component, then `.reverse()` to restore natural order

**Public API:**
- `pub struct ItpResult` — `adjusted_pvalues`, `raw_pvalues`, `basis_type`, `n_basis`, `n_perm`; derives `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde
- `pub fn itp_one_pop(data, argvals, mu0, basis_type, nbasis, n_perm, seed) -> Result<ItpResult, FdarError>` — sign-flip permutation for one-population mean test; single sequential RNG loop; full validation

### Modified files
- `inference/mod.rs`: `mod itp; pub use itp::{itp_one_pop, ItpResult};`
- `lib.rs`: extended inference re-export block with `itp_one_pop`, `ItpResult`
- `basis/projection.rs`: `#[cfg_attr(feature = "serde", derive(...))]` added to `ProjectionBasisType`

## Tests Added (6 total)

| Test | Purpose |
|------|---------|
| `pval_correct_hand_computed` | Locks cone-walk index math: p=4 pval_matrix, expected values traced by hand |
| `fisher_cf_log_safe` | Verifies finite output on 0.0 p-value input |
| `one_population_localized` | Constant shift on [0.4, 0.6] → min adjusted p < 0.05 |
| `one_population_null` | No shift → max adjusted p > 0.10 |
| `one_population_deterministic` | Same seed → bit-identical ItpResult |
| `one_population_error_paths` | n<2, argvals mismatch, nbasis<2, n_perm==0 → FdarError |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Clippy manual_memcpy in build_pval_matrix**
- **Found during:** clippy --all-targets gate after Task 1 commit
- **Issue:** `for j in 0..p { mat[p-1][j] = raw_pvalues[j]; }` triggered `clippy::manual-memcpy`
- **Fix:** `mat[p - 1][..p].copy_from_slice(&raw_pvalues[..p]);`
- **Files modified:** `fdars-core/src/inference/itp.rs`
- **Commit:** edb02a8a

**2. [Rule 1 - Bug] Clippy useless_vec in test**
- **Found during:** clippy --all-targets gate after Task 1 commit
- **Issue:** `let expected = vec![0.60, 0.55, 0.48, 0.42];` in test triggered `clippy::useless-vec`
- **Fix:** `let expected = [0.60, 0.55, 0.48, 0.42];`
- **Files modified:** `fdars-core/src/inference/itp.rs`
- **Commit:** edb02a8a

## Key Decisions Made

1. **build_pval_matrix uses n_ge / n_perm (no +1) for internal closure matrix** — matching R source for internal p-value matrix; only `ItpResult.raw_pvalues` uses the `(n_ge+1)/(n_perm+1)` correction. This ensures the closure adjustment is consistent with the fdatest baseline.

2. **rank_transform returns assignments as Vec<Vec<(usize, f64)>> then scatters** — needed for parallelism: `iter_maybe_parallel!` over components collects per-component index↔rank pairs, then a sequential scatter loop fills the `l` matrix. Direct parallel mutation of `l` would require unsafe or a Mutex.

3. **pval_correct.reverse() confirmed by hand-computed test** — the hand-traced execution (Python script) for p=4 with a monotone pval_matrix confirms the reversal produces natural component order; without it the test would fail.

## Gate Results

| Gate | Result |
|------|--------|
| `cargo test --lib inference::itp::tests::pval_correct_hand_computed` | PASS (1/1) |
| `cargo test --lib inference::itp` (all 6 tests) | PASS (6/6) |
| `cargo build --features serde` | PASS |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | PASS |

## Known Stubs

None — all outputs are wired: `itp_one_pop` runs the full pipeline (project → permute → rank → build matrix → adjust → ItpResult).

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or trust boundary crossings introduced.

## Self-Check

- [x] `fdars-core/src/inference/itp.rs` — FOUND
- [x] `fdars-core/src/inference/mod.rs` — FOUND, contains `mod itp; pub use itp::{itp_one_pop, ItpResult}`
- [x] `fdars-core/src/lib.rs` — FOUND, contains `itp_one_pop, ItpResult` in inference re-export block
- [x] `fdars-core/src/basis/projection.rs` — FOUND, contains conditional serde on `ProjectionBasisType`
- [x] Commit d673a586 — FOUND (feat: pval_correct helpers + ItpResult + wiring)
- [x] Commit edb02a8a — FOUND (fix: clippy manual_memcpy + useless_vec)

## Self-Check: PASSED
