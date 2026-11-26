---
quick_id: 261126-2ez
issue: 49
backlog_id: ALN-DPACC
status: complete
---
# ALN-DPACC (#49) — configurable DP neighbourhood, `dp_scl` refinement, DP2 cost, pruning

## Commits
- d81d1671 feat(quick-261126-2ez): configurable elastic DP — nbhd_dim, dp_scl refinement,
  DP2 piecewise-linear cost, exact pruning (closes #49)

## Public API (additive)
- `WarpOptimConfig` new DP-only fields (sealed; defaults reproduce the classic engine):
  `nbhd_dim: usize = 7` (1..=64), `dp_scl: usize = 1` (1..=64), `dp_cost: DpEdgeCost =
  PiecewiseConstant`, `dp_prune: bool = false`; `validate()` checks the ranges.
- New `#[non_exhaustive]` enum `DpEdgeCost { PiecewiseConstant, PiecewiseLinear }`,
  re-exported from `alignment` and crate root.
- Applies to every `*_with_config` entry point (pair, distance, self/cross distance
  matrices, Karcher mean). RBFGS's internal DP start is unchanged.

## Engine (alignment/mod.rs)
- `DpMoves { nbhd, prune }` threaded through `dp_grid_solve_moves` → `dp_fill_grid` →
  `dp_fill_grid_impl::<PRUNE>` → `dp_relax_cell::<PRUNE>`; classic wrappers pass
  `DP_MOVES_CLASSIC` (`COPRIME_NBHD_7`, no pruning). `generate_coprime_nbhd` is no longer
  test-only; `coprime_nbhd(d)` borrows the const for d = 7.
- `dp_alignment_core_opts` / `dp_path_opts` / `dp_path_weighted`: cost model + refinement
  dispatched once, so each edge closure is monomorphic and the classic path runs the
  original arithmetic; √(dr/dc) table generalised (stack `[[f64;65];65]`, filled to w).
- `dp_edge_weight_linear_core`: exact `dt·(f0²+f0f1+f1²)/3` per merged sub-interval in
  the same integer walk (fdasrsf `DynamicProgrammingQ2` semantics).
- `dp_scl`: SRSFs linearly resampled index-uniformly ×scl; coarse edges evaluated on fine
  indices (same √γ', span) → warp path stays on the input grid, cost keeps integral scale
  (so `lambda` is scl-independent; fdasrsf's CostFn2 sums without dt — deliberate deviation).

## Validation (fdars-core/src/alignment/dp_options_tests.rs, 10 tests)
- Bit-identity: pre-existing FNV hash tests in optim_tests.rs pass unchanged (classic and
  default config, λ ∈ {0, 0.3}, pair/self/cross/Karcher/band/penalized); explicit default
  options == classic on uniform + non-uniform grids; `dp_prune = true` bit-identical to
  `false` across both cost models, scl 3, L2Gam/L2Psi, λ > 0, band, non-uniform grid, self
  distance matrix, Karcher mean (pruning is exact: edge costs ≥ 0).
- Linear edge weight vs 200k-point midpoint integral (< 1e-9 rel); refined constant cost
  converges monotonically to the linear cost; linear cost is scl-invariant.
- DP's own objective non-increasing over nbhd_dim 3→5→7→10→12 (both cost models, λ 0/0.1).
- DP-independent reference cost (64× midpoint, linear SRSFs) of the warp: nbhd10+linear,
  nbhd10+scl4, linear7 all ≤ default on 4 cases.
- Warped self-distance f vs f∘γ (γ = t+0.25t(1−t), m = 80, the #48 warp; 1-D elastic
  distance scale): default 0.1200 → nbhd10+linear 0.0699, nbhd10+scl4 0.0712.
- Golden: fdasrsf 2.7.5 `optimum_reparam(method="DP2", grid_dim=G, penalty, lam)` on 5
  cases (G = 10 and 7, m = 41/61/101, incl. l2gam λ = 0.05): sup diff ≤ 2.3e-16 with and
  without pruning (tolerance 1e-12). Classic engine vs DP2: 0.008–0.05. Script:
  `golden_dp2.py` (this dir) → `fdars-core/src/alignment/dp_options_golden.rs`.

## Timings (criterion `dp_options`, `--quick`, loaded 20-core machine, one pair)
| variant | m=100 | m=200 |
|---|---|---|
| default (classic) | 6.45 ms | 27.5 ms |
| default + prune | 4.14 ms | 17.3 ms |
| linear, nbhd 7 | 13.8 ms | 59.7 ms |
| nbhd 10 + linear (= DP2) | 34.6 ms | 139 ms |
| nbhd 10 + linear + prune | 19.9 ms | 79.6 ms |
| dp_scl 3 | 19.4 ms | 77.0 ms |
| dp_scl 3 + prune | 11.8 ms | 44.2 ms |

Classic `elastic_align_pair/m/{50,100,200}` vs a pre-change HEAD worktree, alternating
runs: within run-to-run noise (±4%) after monomorphising the weight dispatch and pruning
switch (a first draft with a runtime branch + heap table was ~8–12% slower; fixed).
Worktree and its target/ removed.

## Gates
- `cargo fmt --all` clean; `cargo clippy -p fdars-core --all-targets --features linalg,parallel
  -- -D warnings` clean; `cargo test --lib alignment` 362 passed, `--lib elastic` 147 passed;
  `cargo test --doc WarpOptimConfig` 2 passed; `cargo doc` no new warnings; pmat complexity
  of touched fns ≤ 22 (dp_relax_cell 22, validate 18).
- No serde types added.

## Deviations / notes
- fdasrsf `DP` (DP.c) also uses spline-resampled q and a different path/warp
  reconstruction; fdars's `dp_scl` refines the existing dp_grid-style engine instead
  (linear resampling, exact-integral scale). Exact agreement is targeted at `DP2` (achieved).
- Pruning is opt-in although provably exact; it could become the default (or be applied
  in the classic `dp_alignment_core*` when λ ≥ 0) in a later change for a free ~1.6× speedup.
- On non-uniform grids `PiecewiseLinear` uses the same index-space sub-interval split as the
  classic model (exact only on uniform grids) — documented on the enum.
- `dp_options` bench runs for m = 100/200 only; not run as part of the whole suite.
