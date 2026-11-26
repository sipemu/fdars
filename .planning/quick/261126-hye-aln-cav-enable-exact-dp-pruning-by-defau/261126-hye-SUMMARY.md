---
quick_id: 261126-hye
issue: 66
status: complete
commits:
  - 18487a68 feat(quick-261126-hye): exact DP pruning on by default in every elastic DP path (closes #66)
---

# Summary — ALN-CAV: exact DP pruning on by default (GH #66)

## What changed
- `WarpOptimConfig::dp_prune` default `false -> true`; `false` remains the explicit switch to the unpruned reference fill.
- New internal gate `alignment::dp_prune_is_exact(argvals, lambda)` = `lambda >= 0 && argvals finite && strictly increasing` (exactly the condition under which every DP edge cost is non-negative). Negative/NaN lambda never prunes.
- Wired into every user of the shared engine: `dp_alignment_core_banded` (-> `dp_alignment_core`, used by `karcher_mean*`, closed, multires, geodesic, bayesian, elastic regression/logistic/scalar-on-shape, conformal), `optim::dp_warp` (all `*_with_config` and the classic pair/distance-matrix paths via `classic_dp`), `dp_alignment_core_with_penalty` (RBFGS coarse start, new `prune` arg), `dp_grid_solve` (new `prune` arg: nd, constrained, partial_match, curve_shape geometry with `h > 0`).
- `dp_grid_solve_banded` folded into `dp_grid_solve_moves` (was only a pass-through); `DpMoves::classic(prune)` const ctor.
- Bench: `dp_options` rows renamed (`default`/`linear7`/`nbhd10_linear`/`scl3` = pruned, `*_unpruned` = reference); `karcher_mean/n20_m50_unpruned` added.
- CHANGELOG `[Unreleased]` entries (root + fdars-core); `dp_prune` field docs + doc example updated.

## Validation
- `alignment/prune_default_tests.rs` (6 tests): 79 FNV-1a hashes recorded on unchanged HEAD 7d8cefe2 (only the test module registered) — classic 1-D on uniform + non-uniform grids, lambda 0 / 0.25 / -0.5, binding band 0.06 for pair / self-dm / karcher; config L2Gam, L2Psi, DP2 (nbhd 10 + PiecewiseLinear), dp_scl 3, RBFGS (L2Psi/L2Gam coarse start) for pair / cross-dm / karcher; closed pair + karcher; N-d pair + karcher (uniform/non-uniform); curve_shape open / penalised / closed; constrained; partial match; geodesic. All match after the change, for default and for `dp_prune = false`.
- Engine: 480 random cases (continuous >= 0 and heavily tied integer costs, square/rectangular, full/banded, nbhd 7/10) give identical paths pruned vs unpruned; a negative-cost counterexample is found (justifies the gate); gate unit test (negative, NaN, flat, reversed, non-finite grid).
- Existing hash tests unchanged and passing (optim_tests, hygiene_tests, dp_options_tests).

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib alignment`: 562 passed, 2 ignored. `--lib -- elastic conformal shape`: 419 passed.
- Integration: validate_against_r (171), validate_new_modules (56), validate_phase_bands (16), equivalence_phase114 (2), alloc_audit_fpca — all pass.
- `cargo test --doc WarpOptimConfig`: 2 passed. `RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps`: clean.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean (only the pre-existing manifest `homepage` warning).

## Bench (criterion `--quick`, machine under background load ~8-13 on 20 cores; 3 interleaved reps, same build)
| bench | unpruned | pruned (default) | speed-up |
|---|---|---|---|
| pair m=100 (`dp_options/default*`) | 6.32-6.57 ms | 4.20-4.26 ms | 1.53x |
| pair m=200 | 24.7-27.4 ms | 16.0-16.6 ms | 1.64x |
| `karcher_mean` n20 m50 | 87.3-95.1 ms | 58.2-63.0 ms | 1.44x |

Against the pre-change saved baseline `pre66` (single run, noisier): `elastic_align_pair/m/100` 7.02 -> 4.39 ms, `m/200` 49.2 (load-inflated) -> 17.9 ms, `m100_lambda0.1` 7.39 -> 4.00 ms, `karcher_mean/n20_m50` 104.1 -> 66.3 ms, `n10_m50` 58.9 -> 40.6 ms.

## Deviations
- None functionally. `dp_grid_solve` / `dp_alignment_core_with_penalty` (pub(super)/pub(crate)) gained a `prune` argument.
