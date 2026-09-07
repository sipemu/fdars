---
type: summary
phase: 83
plan: 02
requirements: [API-04]
status: complete
commit: 24ec997c
---

# 83-02 SUMMARY — deriv grid-enum collapse (AUD-17)

## What was done

Collapsed `deriv_1d` + `deriv_2d` into a single public dispatcher `deriv` via the
approved grid-enum design. The original computation bodies were kept VERBATIM as
private helpers `deriv_1d_impl` / `deriv_2d_impl`; the dispatcher only routes — numeric
output is byte-identical.

## New public symbols (all in `src/fdata.rs`, re-exported at crate root)

- `pub enum DerivDomain<'a>` — `OneD { argvals: &'a [f64], nderiv: usize }` /
  `TwoD { argvals_s, argvals_t, m1, m2 }`. Derives Debug/Clone/PartialEq. NOT
  `#[non_exhaustive]` (user constructs/matches directly). Carries a lifetime.
- `pub enum DerivResult` — `OneD(FdMatrix)` / `TwoD(Deriv2DResult)` / `None`. The
  `None` variant represents the old `deriv_2d` bad-dimension `Option::None` case
  without inventing numeric data.
- `pub fn deriv(data: &FdMatrix, domain: DerivDomain<'_>) -> DerivResult` — routes
  OneD → `DerivResult::OneD(deriv_1d_impl(..))`; TwoD →
  `deriv_2d_impl(..).map(DerivResult::TwoD).unwrap_or(DerivResult::None)`.

Crate-root re-export (`src/lib.rs`): removed `deriv_1d, deriv_2d`; added
`deriv, DerivDomain, DerivResult` (Deriv2DResult already exported).

## Call sites migrated (~29 references)

- `src/alignment/nd.rs` (1), `src/alignment/srsf.rs` (import + 1 call),
  `src/seasonal/peak.rs` (import + 1 call), `src/fpca_variants.rs` (2 doc mentions,
  1 comment, 1 impl call, 1 test call), `src/fdata.rs` unit tests (4×1D + 3×2D +
  migrated module doctest), `tests/validate_against_r.rs` 2D case (1),
  `examples/02_functional_operations/main.rs` (import + 2 calls),
  `examples/02_functional_operations/README.md` (`deriv_1d()` → `deriv()`).
- Doc-comment mentions of the old suffixed names were also neutralized so the
  phase-wide grep gate (which excludes only `*_impl`) stays clean.

## Gate results (whole-crate)

- `cargo fmt --check` — clean
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean, zero warnings
- `cargo test` — lib: `test result: ok. 2850 passed; 0 failed`. Ran `--no-fail-fast`:
  ONLY the 3 pre-existing env golden flakes fail (`golden_co_cluster_parallel`,
  `golden_co_cluster_below_threshold` in equivalence_phase48;
  `svd_sign_fpca_two_matrix_bit_identical` in equivalence_phase49) — confirmed
  pre-existing (also fail on clean checkout), touch none of the deriv symbols. All 18
  other test binaries report ok.
- `cargo test --doc` — `test result: ok. 209 passed; 0 failed; 4 ignored` (migrated
  fdata deriv doctest compiles + runs).
- `cargo build --features serde` — green
- `cargo build --examples` — all 28 compile
- Grep gate — no `deriv_1d`/`deriv_2d` reference remains except the two private
  `*_impl` helpers.

## Files changed

- `fdars-core/src/fdata.rs`, `fdars-core/src/lib.rs`, `fdars-core/src/alignment/nd.rs`,
  `fdars-core/src/alignment/srsf.rs`, `fdars-core/src/seasonal/peak.rs`,
  `fdars-core/src/fpca_variants.rs`, `fdars-core/tests/validate_against_r.rs`,
  `fdars-core/examples/02_functional_operations/main.rs`,
  `fdars-core/examples/02_functional_operations/README.md`

## Commit

`24ec997c` — refactor(83-02): collapse deriv_1d/deriv_2d into deriv grid-enum dispatcher (API-04)
