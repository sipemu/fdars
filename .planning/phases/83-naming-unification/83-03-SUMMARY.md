---
type: summary
phase: 83
plan: 03
requirements: [API-04]
status: complete
commit: 7d7d2c66
---

# 83-03 SUMMARY — lp grid-enum collapse (AUD-18) + phase completion

## What was done

Collapsed `lp_self_1d` / `lp_cross_1d` / `lp_self_2d` / `lp_cross_2d` into two public
dispatchers `lp_self` and `lp_cross` via the approved grid-enum design. The four
original bodies were kept VERBATIM as private helpers `lp_self_1d_impl`,
`lp_cross_1d_impl`, `lp_self_2d_impl`, `lp_cross_2d_impl`; the dispatchers only route —
numeric output is byte-identical. Both return `FdMatrix` (no output enum needed).

## New public symbols (all in `src/metric/lp.rs`)

- `pub enum LpDomain<'a>` — `OneD { argvals: &'a [f64] }` /
  `TwoD { argvals_s, argvals_t }`. Derives Debug/Clone/PartialEq. NOT
  `#[non_exhaustive]`. Carries a lifetime.
- `pub fn lp_self(data: &FdMatrix, domain: LpDomain<'_>, p: f64, user_weights: &[f64]) -> FdMatrix`
- `pub fn lp_cross(data1: &FdMatrix, data2: &FdMatrix, domain: LpDomain<'_>, p: f64, user_weights: &[f64]) -> FdMatrix`

Re-exports updated at three sites: `src/metric/mod.rs` (`pub use lp::{lp_cross, lp_self, LpDomain};`),
`src/lib.rs` (added `lp_cross, lp_self, LpDomain`; removed the four suffixed names),
`src/prelude.rs` (`lp_cross, lp_self, LpDomain`).

## Call sites migrated (~44 references)

- `src/metric/tests.rs` — all 1D/2D self+cross calls + empty-input assertions.
- `tests/validate_against_r.rs` — 3 call sites (lp_self 1D, lp_cross 1D ×2) via
  fully-qualified `fdars_core::metric::lp_*` + `LpDomain`.
- `examples/06_distances_and_metrics/main.rs` — import + 4 calls (L1/L2/Linf self +
  cross); printed output preserved.
- `examples/06_distances_and_metrics/README.md` — `lp_self_1d()`/`lp_cross_1d()` →
  `lp_self()`/`lp_cross()` (with `LpDomain`).
- `benches/depth_benchmarks.rs` — import + call + benchmark-group label (surfaced by
  `--all-targets` clippy; not in the original plan file list but required to keep the
  gate + phase-wide grep green).
- Two lp doctests in `lp.rs` migrated to `LpDomain::OneD { .. }`.
- Doc-comment mentions of the old suffixed names neutralized for the grep gate.

## Deviation

The perl multi-line regex used to rewrite `metric/tests.rs` corrupted one
`test_lp_cross` block whose first `lp_cross_2d` argument spanned multiple lines
(two `FdMatrix::from_column_major(...)` calls got mangled). Detected immediately by the
compile gate; reconstructed the block by hand and verified the crate + tests compile.
No other site was affected (all others had single-line args). Also had to migrate
`depth_benchmarks.rs` (a bench, not in the plan's file list) because `--all-targets`
clippy compiles benches.

## Gate results (whole-crate — the phase-completion proof for all 5 AUD items)

- `cargo fmt --check` — clean
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean, zero warnings
- `cargo test` — lib: `test result: ok. 2850 passed; 0 failed`. `--no-fail-fast`
  confirms ONLY the 3 pre-existing env golden flakes fail
  (`golden_co_cluster_parallel`, `golden_co_cluster_below_threshold`,
  `svd_sign_fpca_two_matrix_bit_identical`); all 18 other binaries ok. Flakes are
  hardware/BLAS-dependent goldens (verified fail on clean checkout), touch no
  lp/deriv/gmm/fosr symbol.
- `cargo test --doc` — `test result: ok. 209 passed; 0 failed; 4 ignored`
  (both migrated lp doctests compile + run).
- `cargo build --features serde` — green
- `cargo build --examples` — all 28 compile
- Phase-wide grep gate — clean across `src/`, `tests/`, `examples/`, `benches/`
  (`--include='*.rs'`) AND example READMEs: NONE of the 9 old public names survive
  (`funhddC_cluster` as callable, `FosrResult2d`, bare `GmmResult`, `deriv_1d`,
  `deriv_2d`, `lp_self_1d`, `lp_cross_1d`, `lp_self_2d`, `lp_cross_2d`) — only the six
  private `*_impl` helpers remain.

## Files changed

- `fdars-core/src/metric/lp.rs`, `fdars-core/src/metric/mod.rs`,
  `fdars-core/src/lib.rs`, `fdars-core/src/prelude.rs`,
  `fdars-core/src/metric/tests.rs`, `fdars-core/tests/validate_against_r.rs`,
  `fdars-core/examples/06_distances_and_metrics/main.rs`,
  `fdars-core/examples/06_distances_and_metrics/README.md`,
  `fdars-core/benches/depth_benchmarks.rs`

## Commit

`7d7d2c66` — refactor(83-03): collapse lp_self/lp_cross _1d/_2d into grid-enum dispatchers (API-04)
