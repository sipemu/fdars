---
quick_id: 261126-6po
issue: 53
backlog_id: ALN-SPARSE
status: complete
issue_fully_done: true
---
# ALN-SPARSE (GH #53, part 2) summary: closed sparse elastic means + robust smooth goldens

## Commits
- `76a75d07` feat(quick-261126-6po): closed sparse elastic means and robust smooth-mean goldens
  (closes #53)

## Public API (additive, sealed-config fields with old-behaviour defaults)
- `SparseElasticMeanConfig { closed: false, pen_factor: 100.0, closure_steps: 1 }`.
- `SparseElasticMeanResult { closed: bool, start_indices: Vec<usize> }`,
  `SparseElasticMeanResult::closure_residual() -> Vec<f64>` (`∫_0^1 p|p|`).
- `srv_eval` / `mean_curve` handle the periodic (closed smooth) spline; closed smooth `coefs`
  are `d × K` (elasdics layout).

## Algorithm (elasdics 1.1.3 source: fit_mean_closed, pre_align_closed_srv_curves,
## one_step_grad, get_loss_pen, get_grad_penalty_fun, make_design(closed = TRUE))
New `alignment/sparse/closed_mean.rs`. Curves closed (append start); all-pairs starting-point
pre-alignment (parallel), reference = argmin Σ d²; curves rotated to their start; LS fit on the
closed design (periodic hat basis / K indicators); per iteration: open alignment of the rotated
curves (smooth: eps·100/i; polygon: fixed 1e-3 as elasdics), refit, stop criterion vs the
previous penalised coefs, one step along `(XᵀXc − Xᵀy)/(pen·i) + ∇|e|²` with Brent line search
on [-1, 1] (exact port of R `Brent_fmin`); on convergence one pure-closure step. Generalised to
any d (elasdics hard-codes 2 columns). Closure-penalty gradient/Jacobian analytic (elasdics:
numDeriv on numerical integrals). Addition: `closure_steps > 1` = Gauss–Newton projections
`δ = −Jᵀ(JJᵀ)⁻¹e` -> closure residual at round-off (elasdics leaves ~1e-4..2e-3).
params_optim un-rotated to the original point order exactly as elasdics.

## Validation
- R elasdics 1.1.3 (R 4.6.1, ~/.cache/R-lib), script `golden_elasdics_closed_mean.R` here;
  literals in `sparse/golden_closed.rs`:
  - closed polygon means (help-page spirals 15 knots; noisy closed shapes set.seed(7), 13 knots):
    coefficients, all t_optims, closure residual to 1e-9 (observed ~1e-13), start indices equal
    R's, distances = exact re-scoring of R's solution to 1e-9.
  - closed smooth means (help-page example `closed = TRUE`, 11 knots; shapes, 9 knots): same
    start indices; our Fréchet objective ≤ R's re-scored by the exact integrator (3.565 vs
    3.612; 1.873 vs 2.312); triangle-inequality per-curve bounds; mean-to-mean sanity bound
    0.15·‖μ_R‖ (observed 8.2 % / 10.6 % — R's constrOptim runs with reltol 1 and 0.5 in its
    1–2 iterations, leaving warps near the pre-alignment); closure residuals < 1 % of length.
  - `brent_fmin` reproduces R `optimise(...)$minimum` bit-for-bit (tolerance-limited path).
- Invariants: FD checks of the closure Jacobian/penalty gradient and LS gradient;
  rotation/unrotation; relabelling non-reference curves' points -> identical mean (1e-6) and
  same physical start; relabelling all curves -> same closed curve up to spline resolution;
  one closed shape sampled on 4 grids (120–200 pts) with different starts -> smooth mean closer
  to the shape (0.034) than every observation (≥ 0.05), polygon 0.115 (60-segment limit);
  closed-distance reparametrisation invariance; 3-D closed mean; closure_steps -> |e| < 1e-10;
  periodic SRV; error paths (2 knots, bad pen_factor, non-closing param).
- Open path bit-identity: open smooth + polygon spiral means (coefs, distances, params)
  compared bit-for-bit against the pre-change build (HEAD sources swapped in) — identical.

## Smooth-mean golden robustness (task 2)
`golden_smooth_mean_spirals` now asserts: (a) fdars' exact integrator reproduces R's
rel.tol = 1e-13 recomputation of R's distances (1e-9; observed 1e-15); (b) Σd²(fdars) ≤ Σd²(R)
re-scored exactly (+1e-9); (c) mean-curve distance < 0.05·‖μ_R‖ = 0.126 (rel.tol/2 on each √H_k;
observed 0.097, was a fixed 0.1); (d) |d_i − d*_i(R)| ≤ d(μ, μ_R) + 1e-3 (triangle inequality,
d* with optimal warps to R's mean); (e) looser sanity: coefficient rel L2 < 0.1 (observed 0.049,
was < 0.05), t_optim within 0.02 (observed 0.006). The former per-curve 0.03 absolute bound is
replaced by (d). Not asserted (documented): "our mean beats R's after optimal re-alignment" —
true for open (2.847 vs 2.904) but false for closed spirals, so not a well-defined golden.

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib alignment::sparse` 28/28; `alignment::hygiene` 3/3;
  `cargo test -p fdars-core --doc sparse` 7/7.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo check -p fdars-core --features serde` and `--no-default-features` clean.
- `pmat analyze complexity --path fdars-core/src/alignment/sparse --max-cognitive 23`: 0.

## Deviations from elasdics (documented in module docs)
- any d (elasdics 2-D only); analytic penalty gradient; `closure_steps > 1` is an addition;
  distances are to the returned mean (as for open means).

## Remaining
None for #53 (`fit_elastic_regression` was out of scope per the issue).
