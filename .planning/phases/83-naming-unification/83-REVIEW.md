---
phase: 83-naming-unification
reviewed: 2026-09-07T11:47:18Z
depth: deep
files_reviewed: 21
files_reviewed_list:
  - fdars-core/src/fdata.rs
  - fdars-core/src/metric/lp.rs
  - fdars-core/src/metric/tests.rs
  - fdars-core/src/metric/mod.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
  - fdars-core/src/gmm/mod.rs
  - fdars-core/src/gmm/em.rs
  - fdars-core/src/gmm/cluster.rs
  - fdars-core/src/gmm/subspace.rs
  - fdars-core/src/function_on_scalar_2d.rs
  - fdars-core/src/fpca_variants.rs
  - fdars-core/src/alignment/nd.rs
  - fdars-core/src/alignment/srsf.rs
  - fdars-core/src/seasonal/peak.rs
  - fdars-core/tests/validate_against_r.rs
  - fdars-core/benches/depth_benchmarks.rs
  - fdars-core/examples/02_functional_operations/main.rs
  - fdars-core/examples/06_distances_and_metrics/main.rs
  - fdars-core/examples/02_functional_operations/README.md
  - fdars-core/examples/06_distances_and_metrics/README.md
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues_found
---

# Phase 83: Code Review Report

**Reviewed:** 2026-09-07T11:47:18Z
**Depth:** deep
**Files Reviewed:** 21
**Status:** issues_found

## Summary

Reviewed the API-04 naming-unification phase (commits ce5c3a6e, 24ec997c, 7d7d2c66): three hard renames and two grid-enum dispatcher collapses (`deriv`, `lp`). The stated invariant — API-shape change only, numeric output byte-identical — holds. All four highest-risk correctness checks pass:

1. **Dispatcher routing is correct with no swaps.**
   - `deriv` (fdata.rs): `DerivDomain::OneD { argvals, nderiv }` → `deriv_1d_impl(data, argvals, nderiv)` → `DerivResult::OneD`; `DerivDomain::TwoD { argvals_s, argvals_t, m1, m2 }` → `deriv_2d_impl(data, argvals_s, argvals_t, m1, m2)` with `.map(DerivResult::TwoD).unwrap_or(DerivResult::None)`. Payload fields reach the impl in the correct positions; `Some → TwoD` / `None → None` semantics match the former `Option<Deriv2DResult>` guard exactly. `deriv_2d_impl` still returns `Option<Deriv2DResult>`; `Deriv2DResult` derives `PartialEq`, so `DerivResult` derives it and `assert_eq!(result, DerivResult::None)` is sound.
   - `lp_self`/`lp_cross` (metric/lp.rs): `lp_self` routes only to `lp_self_*_impl`, `lp_cross` only to `lp_cross_*_impl`; `OneD → *_1d_impl`, `TwoD → *_2d_impl`. No self/cross or 1d/2d swap. `p` and `user_weights` are forwarded unchanged in the correct argument positions.

2. **`_impl` bodies are byte-identical to the originals.** Verified against `ca2b32f5` via body diff: `deriv_1d_impl`, `deriv_2d_impl`, `lp_cross_1d_impl`, `lp_self_1d_impl`, `lp_cross_2d_impl`, `lp_self_2d_impl` differ from the pre-phase functions in the signature line only (`pub fn NAME` → `fn NAME_impl`). No math altered.

3. **Hand-reconstructed `test_lp_cross` block (metric/tests.rs) is faithful.** Full-body comparison against `ca2b32f5:test_lp_cross_2d` confirms both `FdMatrix::from_column_major` calls are byte-identical (data1: `(i*0.1).sin()`, `n1`, `n_points`; data2: `(i*0.2).cos()`, `n2`, `n_points`), `argvals_s`/`argvals_t` unchanged, and only the two `lp_cross_2d(...)` calls became `lp_cross(.., LpDomain::TwoD { .. }, ..)`. This test asserts shape / non-negative / finite only (no hardcoded expected numbers), so there is no altered-expected-value vector here. The R-value tests (`validate_against_r.rs`) that DO carry R-computed expected constants (`test_lp_l2_distance_matrix` 6e-3 tolerance, `test_lp_cross_values_vs_r`) retained their expected values and correct argument order across the migration.

4. **Rename completeness is correct.** Targets `Fosr2dResult`, `GmmFitResult`, `fun_hddc_cluster` are present everywhere including struct defs, impls, return types, field types (`GmmClusterResult.best: GmmFitResult`), re-exports (lib.rs, gmm/mod.rs, metric/mod.rs, prelude.rs), doctests, and intra-doc links. The pre-phase names `FosrResult2d`, `GmmResult`, `funhddC_cluster` no longer appear as public identifiers anywhere in the crate. `GmmClusterResult` was NOT accidentally renamed (still `pub struct GmmClusterResult`). No old dispatcher public names (`deriv_1d`, `deriv_2d`, `lp_self_1d/2d`, `lp_cross_1d/2d`) survive; only the private `*_impl` helpers remain, as intended. All call sites in library, tests, benches, and examples were migrated to the new enum-dispatch forms.

No correctness issues (no routing swap, no altered test expectation, no missed rename). One stale-string warning and two cosmetic info items follow.

## Warnings

### WR-01: Error `operation` string still uses the old function name `funhddC_cluster`

**File:** `fdars-core/src/gmm/subspace.rs:668`
**Issue:** The `fun_hddc_cluster` function was renamed from `funhddC_cluster`, but the `FdarError::ComputationFailed` raised on total EM-restart failure still reports the pre-rename name:
```rust
    ) = best.ok_or_else(|| FdarError::ComputationFailed {
        operation: "funhddC_cluster",
        detail: "all EM restarts failed".to_string(),
    })?;
```
This is not a correctness/routing bug (it is a diagnostic string, not control flow), but it is a user-facing leak of a now-nonexistent public API name. A caller matching on or logging `operation` would see a name that no longer resolves in the crate, undermining the point of the unification.
**Fix:** Update the string to match the new function name:
```rust
        operation: "fun_hddc_cluster",
```

## Info

### IN-01: Internal test function names retain the old `funhddC` spelling

**File:** `fdars-core/src/gmm/subspace.rs:759,776,806,826,837,848,859,871`
**Issue:** Test functions (`test_funhddC_recovery`, `test_funhddC_bic_finite`, `test_funhddC_deterministic`, `test_funhddC_invalid_empty`, `test_funhddC_invalid_k_zero`, `test_funhddC_invalid_k_exceeds_n`, `test_funhddC_invalid_dk_ge_m`, `test_funhddC_invalid_argvals_mismatch`) keep the pre-rename `funhddC` spelling while the function under test is now `fun_hddc_cluster`. Purely cosmetic — test names are not public API — but inconsistent with the rename intent.
**Fix:** Optionally rename to `test_fun_hddc_*` for consistency. No behavior impact.

### IN-02: `unreachable!` guards on `DerivResult::OneD` matches are correct but repeated

**File:** `fdars-core/src/alignment/nd.rs:70-77`, `fdars-core/src/alignment/srsf.rs:42-44`, `fdars-core/src/seasonal/peak.rs:114-117`, `fdars-core/src/fpca_variants.rs:231-234,936-944`, `fdars-core/examples/02_functional_operations/main.rs`
**Issue:** Every 1D caller of the new `deriv` dispatcher destructures with `let DerivResult::OneD(m) = deriv(.., DerivDomain::OneD { .. }) else { unreachable!(...) }` (or a `match` with `_ => unreachable!(...)`). This is logically sound — a `OneD` domain provably yields a `OneD` result — and is not a bug. It is noted only as a maintainability observation: the enum dispatcher trades a direct typed return for a runtime `unreachable!` at each 1D call site. Acceptable given the phase's shape-unification goal; no change required.
**Fix:** None required. (If a future phase wants to eliminate the `unreachable!` boilerplate, a thin typed helper e.g. `deriv_1d_result(...) -> FdMatrix` wrapping the dispatcher could restore ergonomics without reintroducing the old public name.)

---

_Reviewed: 2026-09-07T11:47:18Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
