---
type: summary
plan: 81-02
phase: 81
requirements: [API-01]
status: complete
---

# 81-02 SUMMARY — Hard-remove the 6 deprecated forms (API-01)

## What was done

Hard-removed the 6 already-deprecated delegation shims, all their re-exports, and
migrated every internal consumer to the `Dim`/`_seeded` replacements. Behavior is
preserved bit-for-bit (the `fanova` callers pin the legacy seed 42 explicitly).

### Removed forms + replacements

| Removed | Defined at | Replacement (behavior-preserving) |
|---------|-----------|-----------------------------------|
| `fdata::mean_2d` | src/fdata.rs | `mean(data, Dim::Two)` |
| `function_on_scalar::fanova` | src/function_on_scalar.rs | `fanova_seeded(data, groups, n_perm, 42)` |
| `depth::fraiman_muniz_2d` | src/depth/fraiman_muniz.rs | `fraiman_muniz(obj, ori, scale, Dim::Two)` |
| `depth::modal_2d` | src/depth/modal.rs | `modal(obj, ori, h, Dim::Two)` |
| `depth::random_tukey_2d` | src/depth/random_tukey.rs | `random_tukey(obj, ori, nproj, Dim::Two)` |
| `depth::random_projection_2d` | src/depth/random_projection.rs | `random_projection(obj, ori, nproj, Dim::Two)` |

### Task 1 — definitions + re-exports + dead allow(deprecated)
- Deleted all 6 `pub fn` definitions and their `#[deprecated]`/`#[must_use]` blocks.
- Removed every re-export: `src/depth/mod.rs`, `src/lib.rs` (fanova / depth / fdata
  blocks), `src/prelude.rs`.
- Removed the now-dead `#[allow(deprecated)]` guards on those re-export blocks and
  refreshed the stale explanatory comments that referenced the removed forms.
- Replacement functions (`mean`, `fanova_seeded`, `fraiman_muniz`, `modal`,
  `random_tukey`, `random_projection`) left untouched.

### Task 2 — production callers + doctests
- Migrated every `fanova(...)` call to `fanova_seeded(..., 42)` in
  `inference/anova.rs` (2 test callers), `inference/permutation.rs` (1),
  `function_on_scalar.rs` (5 in-module test callers).
- Fixed intra-doc `[`fanova`]` links (which would break the doc build) to
  `fanova_seeded` in `function_on_scalar.rs` (module doc + integrated_f_statistic
  doc), `inference/anova.rs` (module doc + 3 links), `inference/permutation.rs`.
- Removed the now-stale `#[allow(deprecated)]` on the migrated test fns.
- Library crate compiles clean.

### Task 3 — shim-only tests + example 21
Removed the delegation/equivalence tests whose sole purpose was proving a removed
shim matches its replacement (replacement-side coverage already exists and remains):
- `src/depth/tests.rs`: `test_fraiman_muniz_2d_delegates`, `test_modal_2d_delegates`,
  `test_random_projection_2d_returns_valid`.
- `src/fdata.rs`: `test_mean_2d_delegates`.
- `tests/equivalence_phase50.rs`: `fanova_shim_seed42_bit_identical`,
  `dispatch_modal_equals_2d`, `dispatch_fraiman_muniz_equals_2d`,
  `dispatch_mean_equals_2d`, `dispatch_random_projection_2d_is_valid`,
  `dispatch_random_tukey_2d_is_valid`.
- `tests/validate_against_r.rs`: `test_2d_delegates_to_1d_fm`, `test_2d_mean_valid`.
- Updated the equivalence_phase50.rs module-doc/comment prose to reflect the shims
  are gone (kept the `fanova_seeded`-side goldens).
- `examples/21_function_on_scalar/main.rs`: updated the `//!` doc line to reference
  `fanova_seeded` (example body already used `fanova_seeded`; no code call migrated).

**Tests removed:** 12 (shim-only). **Callers migrated:** 8 fanova call sites.
**Files changed:** 14.

## Final gate results

1. `cargo fmt --check` — **PASS** (`FMT_OK`)
2. `cargo clippy --all-targets --features linalg,parallel -- -D warnings` —
   **PASS** (`Finished`, zero warnings)
3. `cargo test` — lib unittests: **`test result: ok. 2850 passed; 0 failed`**;
   migrated integration files: `equivalence_phase50` **7 passed / 0 failed**,
   `validate_against_r` **171 passed / 0 failed**. All other binaries green EXCEPT
   two PRE-EXISTING failures unrelated to API-01 (see below).
4. `cargo build --features serde` — **PASS** (`Finished`)
5. `cargo build --examples` — **PASS** (`Finished`; all 28 examples compile)

### Pre-existing failures (NOT caused by this plan)
- `tests/equivalence_phase48.rs`: `golden_co_cluster_parallel`,
  `golden_co_cluster_below_threshold` (coclustering golden mismatch).
- `tests/equivalence_phase49.rs`: `svd_sign_fpca_two_matrix_bit_identical`
  (SVD sign-convention golden mismatch).

Verified by `git stash`-ing the plan's changes and re-running: all three fail
IDENTICALLY on the baseline commit e2f74df4, touch none of the 6 removed forms,
and are outside API-01 scope. This plan introduces zero new failures.

## Commits
- `f92466ff7c4873e833845e66bc23b278e0e1e165` refactor(81-02): hard-remove 6 deprecated forms + re-exports (API-01)
- `e2f74df4eb7cca2363a575275ffff0e73c40b4b7` refactor(81-02): migrate fanova callers to fanova_seeded(...,42) (API-01)
- `3d9a21431dcbe676c3127dce6ea8b89f074bec1a` test(81-02): remove shim-only delegation tests + migrate example 21 doc (API-01)

## Deviations from plan
None functional. The plan's Task 3 grep for a bare `fanova(` in tests would match a
single historical prose comment in `equivalence_phase50.rs`; it was reworded so no
code call or intra-doc link references the removed forms. Two golden test failures
surfaced (co_cluster, svd_sign) but were confirmed pre-existing and out of scope.
