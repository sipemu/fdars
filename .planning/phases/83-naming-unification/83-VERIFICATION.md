---
phase: 83-naming-unification
verified: 2026-09-07T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 83: Naming Unification Verification Report

**Phase Goal:** The approved naming unification is applied — `_1d`/`_2d`/`_nd` suffix sprawl and config/result naming collapse into consistent dispatchers across the whole crate (API-04). Breaking is API-SHAPE ONLY — no numeric change.
**Verified:** 2026-09-07
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth (Success Criterion) | Status | Evidence |
|---|---------------------------|--------|----------|
| 1 | Approved `_1d`/`_2d`/`_nd` + config/result naming changes applied; surface exposes consistent dispatchers (AUD-14..18 only; AUD-19..23 untouched) | ✓ VERIFIED | All 5 new symbols present; all 9 old public names gone; AUD-19..23 retain old names |
| 2 | All call sites, all 28 examples, all doctests updated; crate compiles | ✓ VERIFIED | Grep clean (only error-string literal + private `*_impl`); both READMEs updated; `cargo build --examples` (28) exit 0; doctests 209 passed |
| 3 | Whole-crate test suite passes with no numeric/behavioral change | ✓ VERIFIED | Full `cargo test` exit 0, 2857 lib + 209 doc passed, 0 failed; impl bodies preserved verbatim; migrated deriv (21) + lp (31) numeric-parity tests pass |
| 4 | Whole-crate gates green (fmt, clippy, test, serde build) | ✓ VERIFIED | fmt exit 0; clippy `--all-targets --features linalg,parallel -- -D warnings` exit 0 / 0 warnings; test exit 0; serde build exit 0; examples build exit 0 |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Criterion-by-criterion detail

#### Criterion 1 — Approved naming changes applied (AUD-14..18), scope discipline held

New public symbols present (grep over `fdars-core/src`):
- AUD-14: `pub fn fun_hddc_cluster` — `src/gmm/subspace.rs:554`
- AUD-15: `pub struct Fosr2dResult` — `src/function_on_scalar_2d.rs:69`
- AUD-16: `pub struct GmmFitResult` — `src/gmm/mod.rs:39` (and `GmmClusterResult.best: GmmFitResult` at `:71` — the intended disambiguation)
- AUD-17: `pub enum DerivDomain<'a>` (`fdata.rs:844`), `pub enum DerivResult` (`:871`), `pub fn deriv(...) -> DerivResult` (`:912`)
- AUD-18: `pub enum LpDomain<'a>` (`metric/lp.rs:16`), `pub fn lp_self` (`:43`), `pub fn lp_cross` (`:66`)

Re-exports updated at every site: crate-root `src/lib.rs` (lines 396, 478-479, 636-638, 674-676), `src/prelude.rs` (line 47), `src/metric/mod.rs` (line 150), `src/gmm/mod.rs` barrel (line 81). Old suffixed names removed from all re-export lists.

Scope discipline (AUD-19..23 deliberately NOT touched) — old names confirmed still present:
- AUD-19: `geometric_median_1d`/`_2d` (`fdata.rs:1076,1101`)
- AUD-20: `hausdorff_self_1d`/`_cross_1d`/`_self_2d`/`_cross_2d`/`_3d` (`metric/hausdorff.rs`)
- AUD-21: `functional_spatial_1d`/`_2d` (`depth/spatial.rs:18,77`)
- AUD-22 sample: `soft_dtw_self_1d` (`metric/soft_dtw.rs:176`), `center_1d` (`fdata.rs:222`)
- AUD-23: `LpeerResult` (`peer.rs:193`)

#### Criterion 2 — Call sites / examples / doctests updated; crate compiles

Grep gate: `rg -n --glob '*.rs' '\b(funhddC_cluster|FosrResult2d|deriv_1d|deriv_2d|lp_self_1d|lp_cross_1d|lp_self_2d|lp_cross_2d)\b'` over `fdars-core/{src,tests,examples,benches}` returns a **single** hit: `src/gmm/subspace.rs:668` — the error-string literal `operation: "funhddC_cluster"` (not an API symbol; explicitly out of scope per plan/CONTEXT). Bare `GmmResult` search returns nothing (`GmmClusterResult` correctly retained).

The 12 remaining `*_impl` references are all private `fn` defs (`deriv_1d_impl`, `deriv_2d_impl`, `lp_self_1d_impl`, `lp_cross_1d_impl`, `lp_self_2d_impl`, `lp_cross_2d_impl`) and the dispatcher's internal routing calls — exactly as designed.

Example READMEs updated: `examples/02_functional_operations/README.md` (`fdata::deriv()` with `DerivDomain`), `examples/06_distances_and_metrics/README.md` (`metric::lp_self()`/`lp_cross()` with `LpDomain`).

Compile proof: `cargo build --examples` exit 0 (28 `[[example]]` entries in `fdars-core/Cargo.toml`); `cargo test --doc` = 209 passed / 0 failed / 4 ignored (compile-checked doctests, including migrated deriv/lp doctests).

#### Criterion 3 — Test suite passes, no numeric/behavioral change

Full `cargo test --features linalg,parallel --no-fail-fast` → **exit 0**, all 20 binaries green:
- lib: `2857 passed; 0 failed`
- doc: `209 passed; 0 failed; 4 ignored`
- every integration binary reported `ok`

Notably the known environmental golden flake (`golden_co_cluster_parallel`, `golden_co_cluster_below_threshold` in equivalence_phase48; `svd_sign_fpca_two_matrix_bit_identical` in equivalence_phase49) did NOT even manifest this run — both binaries ran green in the full parallel run. No isolated re-run was needed.

No-numeric-change evidence (behavior-dependent truth): git diff of commits `24ec997c` (deriv) and `7d7d2c66` (lp) shows NO algorithm-body lines removed — only doc comments, old fn signatures, and test/example caller lines were changed. The computation bodies were relocated verbatim into private `*_impl` helpers; the dispatchers only `match` on the domain enum and route. Behavioral parity is exercised by the migrated tests that assert concrete numeric values: 21 deriv tests (e.g. `test_deriv_1d_linear` asserting derivative ≈ 1.0, `test_deriv_2d_linear_surface`) and 31 `lp_` tests (self/cross, 1D/2D, symmetry, diagonal-zero, user-weights) — all pass. This is behavioral evidence, not presence alone.

#### Criterion 4 — Whole-crate gates green

| Gate | Command | Result |
|------|---------|--------|
| fmt | `cargo fmt --check` | exit 0 (clean) |
| clippy | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | exit 0, 0 warnings |
| test | `cargo test --features linalg,parallel --no-fail-fast` | exit 0, 0 failed |
| serde build | `cargo build --features serde` | exit 0 (historic shapelet/ClassifFit break no longer reproduces) |
| examples | `cargo build --examples` | exit 0 (28 examples) |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/gmm/subspace.rs` | `fun_hddc_cluster` def | ✓ VERIFIED | Line 554, wired via barrel + lib re-export |
| `src/function_on_scalar_2d.rs` | `Fosr2dResult` type | ✓ VERIFIED | Line 69, re-exported lib.rs:396 |
| `src/gmm/mod.rs` | `GmmFitResult` type | ✓ VERIFIED | Line 39; `GmmClusterResult.best` uses it |
| `src/fdata.rs` | `deriv` + `DerivDomain` + `DerivResult` | ✓ VERIFIED | Lines 844/871/912; `*_impl` helpers preserved |
| `src/metric/lp.rs` | `lp_self`/`lp_cross` + `LpDomain` | ✓ VERIFIED | Lines 16/43/66; four `*_impl` helpers preserved |
| `src/lib.rs`, `src/prelude.rs`, `src/metric/mod.rs`, `src/gmm/mod.rs` | re-exports updated | ✓ VERIFIED | New names added, old names removed at all 4 sites |
| example 02 + 06 READMEs | migrated to new API | ✓ VERIFIED | Both updated |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `deriv` dispatcher | `deriv_1d_impl`/`deriv_2d_impl` | `match DerivDomain` | ✓ WIRED | fdata.rs:912-926 routes to verbatim bodies |
| `lp_self`/`lp_cross` dispatchers | four `lp_*_impl` | `match LpDomain` | ✓ WIRED | metric/lp.rs:43-82 routes to verbatim bodies |
| public API | callers/tests/examples/doctests | new names | ✓ WIRED | crate compiles; grep clean; 209 doctests pass |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| API-04 | 83-01/02/03 | Naming unification (AUD-14..18) | ✓ SATISFIED | All 5 renames applied, gates green, scope held |

### Anti-Patterns Found

None. No `TODO`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER`/debt markers in any of the 10 changed source files. The lone `funhddC_cluster` occurrence is an error-string literal, explicitly documented as out of scope.

### Human Verification Required

None. All criteria verified programmatically with behavioral test evidence.

### Gaps Summary

No gaps. All 4 success criteria hold with direct codebase evidence:
- New dispatchers/types exist and are wired at every re-export site.
- All 9 old public names are gone (the only survivor is an error-string literal).
- Scope discipline preserved — AUD-19..23 untouched.
- Whole-crate gates all green (fmt, clippy `-D warnings`, full test exit 0, serde build, 28 examples).
- No-numeric-change proven: computation bodies preserved verbatim as private `*_impl`; migrated numeric-assertion tests (21 deriv + 31 lp) pass. The known golden flake did not even manifest in the full run.

---

_Verified: 2026-09-07_
_Verifier: Claude (gsd-verifier)_
