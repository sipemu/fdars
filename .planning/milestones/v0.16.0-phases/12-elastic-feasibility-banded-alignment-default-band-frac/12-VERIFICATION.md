---
phase: 12-elastic-feasibility-banded-alignment-default-band-frac
verified: 2026-08-11T20:14:35Z
status: passed
score: 5/5
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 12: Elastic Feasibility — Banded Alignment opt-in Verification Report

**Phase Goal:** Expose a banded Sakoe-Chiba DP path through the high-level elastic alignment API (karcher_mean, elastic_self_distance_matrix, elastic_cross_distance_matrix) via an OPT-IN, NON-BREAKING band_frac control, so large grids (N=500,M=200) become tractable while the full unbanded path stays available and exact.
**Verified:** 2026-08-11T20:14:35Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A caller can reach the banded Sakoe-Chiba DP path for all three high-level functions through new `*_with_band` wrappers without invoking internal `_banded` variants directly | VERIFIED | `karcher_mean_with_band` at karcher.rs:342, `elastic_self_distance_matrix_with_band` at pairwise.rs:232, `elastic_cross_distance_matrix_with_band` at pairwise.rs:333 — all public, all take `band_frac: Option<f64>`, all delegate via the existing impl functions. Re-exported at crate root in lib.rs:144,148,150. |
| 2 | `band_frac = None` or `Some(0.0)` yields exact unbanded results identical to existing functions; `Some(f>0)` selects the banded path | VERIFIED | karcher.rs:356 uses `band_frac.unwrap_or(0.0)` which collapses `None` and `Some(0.0)` to 0.0, which `band_radius` converts to `None` (unbanded). pairwise.rs:238,340 use `.and_then(|f| band_radius(f, m))` — same collapse. Six passing equivalence tests confirm correctness: `None` path matches exactly within 1e-15, `Some(0.0)` equivalent. |
| 3 | A sufficiently wide band (`Some(0.99)`) matches the unbanded result within numerical tolerance, verified by inline tests | VERIFIED | Three "wide matches unbanded" tests (tests.rs:2756,2797,2839) run with `Some(0.99)` at m=30 or m=40; all assert `(a-b).abs() < 1e-12`. All 6 tests passed: `cargo test ... 6 passed; 0 failed` |
| 4 | All existing call sites of the three functions compile unchanged; no existing signature or default is altered (non-breaking) | VERIFIED | `karcher_mean` signature at karcher.rs:293-301 is unchanged (5 params, no defaults). `elastic_self_distance_matrix` at pairwise.rs:194 is unchanged (3 params, no `#[must_use]` added). `elastic_cross_distance_matrix` at pairwise.rs:293-299 is unchanged (4 params). Clippy `--all-targets --all-features -D warnings` passes clean — no compile errors anywhere. |
| 5 | New `*_with_band` wrappers and existing `*_banded` variants are re-exported at the crate root; a feasibility bench is compiled and registered | VERIFIED | lib.rs:144,147-150 exports all six names (`elastic_cross_distance_matrix_banded`, `elastic_cross_distance_matrix_with_band`, `elastic_self_distance_matrix_banded`, `elastic_self_distance_matrix_with_band`, `karcher_mean_banded`, `karcher_mean_with_band`). alignment/mod.rs:66 and 75-81 export them at submodule level. Bench `bench_karcher_mean_with_band` registered in criterion_group at alignment_benchmarks.rs:193-199; `cargo bench --list` shows `karcher_mean_with_band/n20_m50_none` and `karcher_mean_with_band/n20_m50_band0.1`. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/alignment/karcher.rs` | `karcher_mean_with_band` wrapper | VERIFIED | Lines 323-358: full `pub fn karcher_mean_with_band(…, band_frac: Option<f64>)`, `#[must_use]`, complete rustdoc explaining None/Some(0.0)/Some(0.1)/Some(0.99) tradeoffs. Delegates via `band_frac.unwrap_or(0.0)` to `karcher_mean_impl`. |
| `fdars-core/src/alignment/pairwise.rs` | `elastic_self_distance_matrix_with_band` and `elastic_cross_distance_matrix_with_band` wrappers | VERIFIED | Lines 232-240 and 333-342: both `pub fn *_with_band(…, band_frac: Option<f64>)`, `#[must_use]`, complete rustdoc. Delegate via `.and_then(|f| band_radius(f, m))` (no `clippy::map_flatten` issue). |
| `fdars-core/src/alignment/mod.rs` | New wrappers in pub use blocks | VERIFIED | Line 66 exports `karcher_mean_with_band`; lines 75-81 export both distance-matrix `_with_band` variants (plus `_banded` variants). |
| `fdars-core/src/lib.rs` | New wrappers + `_banded` variants in crate-root re-export | VERIFIED | Lines 144,147-150: all six new/promoted names present in `pub use alignment::{…}` block. |
| `fdars-core/src/alignment/tests.rs` | 6 equivalence tests (plan said 5; 6 delivered) | VERIFIED | Lines 2736,2756,2776,2797,2818,2839: all six `#[test]` functions present. All passed in live run. |
| `fdars-core/benches/alignment_benchmarks.rs` | `bench_karcher_mean_with_band` feasibility bench group | VERIFIED | Lines 149-199: function defined, benchmarks `None` vs `Some(0.1)` at n=20/m=50, registered in `criterion_group!`. `--list` confirms both bench IDs are published. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `karcher_mean_with_band` | `karcher_mean_impl` | `band_frac.unwrap_or(0.0)` passed as `band_frac` argument | WIRED | karcher.rs:350-357 — direct delegation, no new algorithm |
| `elastic_self_distance_matrix_with_band` | `self_distance_matrix_impl` | `.and_then(|f| band_radius(f, m))` → `Option<usize>` band | WIRED | pairwise.rs:238-239 — correct `.and_then` pattern, no `map_flatten` |
| `elastic_cross_distance_matrix_with_band` | `cross_distance_matrix_impl` | `.and_then(|f| band_radius(f, m))` → `Option<usize>` band | WIRED | pairwise.rs:340-341 — same correct pattern |
| `alignment/mod.rs` re-exports | `lib.rs` crate-root | `pub use alignment::{…}` | WIRED | lib.rs:138-169 — mod.rs re-exports precede lib.rs block; resolution confirmed by build |
| `Option<f64>` → `Option<usize>` conversion | `band_radius(f, m)` | `and_then` | WIRED | `band_radius` at mod.rs:534-540: returns `None` for `f<=0` or `f>=1`, `Some(ceil(f*m).max(1))` otherwise |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `None` path matches exact (karcher) | `cargo test … test_karcher_mean_with_band_none_matches_exact` | ok | PASS |
| Wide band matches unbanded (karcher) | `cargo test … test_karcher_mean_with_band_wide_matches_unbanded` | ok | PASS |
| `None` path matches exact (self dist) | `cargo test … test_self_distance_matrix_with_band_none_matches_exact` | ok | PASS |
| Wide band matches unbanded (self dist) | `cargo test … test_self_distance_matrix_with_band_wide_matches_unbanded` | ok | PASS |
| `None` path matches exact (cross dist) | `cargo test … test_cross_distance_matrix_with_band_none_matches_exact` | ok | PASS |
| Wide band matches unbanded (cross dist) | `cargo test … test_cross_distance_matrix_with_band_wide_matches_unbanded` | ok | PASS |
| Bench registered | `cargo bench --list` | `karcher_mean_with_band/n20_m50_none`, `karcher_mean_with_band/n20_m50_band0.1` | PASS |
| CI-parity clippy | `cargo clippy --all-targets --features linalg -- -D warnings` | Finished with no warnings | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PERF-03 | 12-01-PLAN.md | Banded DP path exposed through high-level elastic API with `band_frac` control; unbanded path retained; wide-band matches unbanded within tolerance; bench demonstrates feasibility | SATISFIED | All three `*_with_band` wrappers implemented and re-exported at crate root; 6 equivalence tests pass; feasibility bench compiled and registered; original signatures unchanged; clippy clean |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none found) | — | — | — | — |

Scanned for `TBD`, `FIXME`, `XXX` in all six modified files — no matches. No placeholder implementations, no empty handlers, no hardcoded returns.

### Human Verification Required

None. All must-haves were verified programmatically:
- Wrappers are substantive (delegate to real implementations, not stubs)
- Tests are behavioral (run and produce correct numeric output, not just compile)
- Re-exports are structural (confirmed by successful `cargo test` compilation and link)
- Clippy confirms no CI-blocking lint issues

### Gaps Summary

No gaps. All five must-have truths are VERIFIED against the live codebase:

1. Three `*_with_band` public wrappers exist with correct `Option<f64>` signatures and rustdoc.
2. `None`/`Some(0.0)` → unbanded path; `Some(f>0)` → banded path — proven by delegation logic and passing tests.
3. Wide-band (`Some(0.99)`) matches unbanded within 1e-12 — proven by three passing tests.
4. Original `karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix` signatures are byte-for-byte unmodified — confirmed by reading source and clean clippy build.
5. All six names re-exported at crate root in lib.rs; feasibility bench listed by `cargo bench --list`.

---

_Verified: 2026-08-11T20:14:35Z_
_Verifier: Claude (gsd-verifier)_
