# Quick 261126-hye — ALN-CAV: exact DP pruning on by default (GH #66)

**Goal:** turn `WarpOptimConfig::dp_prune` (#49) on by default and apply it in every entry point routed through the shared DP engine (`alignment/mod.rs`), bit-identically.

**API:** no new public API. `WarpOptimConfig::dp_prune` default `false -> true` (explicit `false` = unpruned reference). Internal: `dp_prune_is_exact(argvals, lambda)` gate, `DpMoves::classic(prune)`, `dp_grid_solve(nrows, ncols, prune, cost)`, `dp_alignment_core_with_penalty(.., prune, ..)`; `dp_grid_solve_banded` folded into `dp_grid_solve_moves`.

**Algorithm/exactness:** pruning skips source `s` when `e[s] >= e[t]`; with edge cost `w >= 0`, `e[s] + w >= e[s] >= e[t]` (IEEE rounding is monotone) can never win the strict `<`, and NaN costs never win either way, so only non-winning evaluations are skipped. All engine costs are squared residuals x span + lambda x non-negative slope penalty x span, so non-negativity holds iff lambda >= 0 and all spans > 0 (finite strictly increasing grid). Gate = exactly that; negative/NaN lambda never prunes.

**Validation:**
1. Record FNV-1a hashes of all affected entry points on HEAD 7d8cefe2 (unpruned) *before* changing code: classic 1-D (uniform/non-uniform, lambda 0/0.25/-0.5, band 0.06 binding), config (L2Gam, L2Psi, DP2 nbhd10 linear, scl3, RBFGS L2Psi/L2Gam; pair/cross-dm/karcher), closed, N-d, curve_shape (open/penalised/closed), constrained, partial, geodesic.
2. After the change the same hashes must hold, for default (pruned) and `dp_prune = false`.
3. Engine: random non-negative continuous + tied-integer costs, rectangular grids, bands, nbhd 7/10: pruned path == unpruned. Negative-cost counterexample exists (gate is necessary). Gate unit test.
4. Existing hash tests (optim_tests, hygiene_tests, dp_options_tests) unchanged.

**Tasks:** record hashes -> implement gate + wiring -> tests/gates -> bench A/B (`--quick`, elastic_align_pair, karcher_mean, dp_options) -> CHANGELOG/docs -> commit.
