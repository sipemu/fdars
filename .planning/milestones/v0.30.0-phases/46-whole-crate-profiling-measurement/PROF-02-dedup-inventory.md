# PROF-02 — Duplication / Consolidation Inventory

**Phase:** 46 (Whole-Crate Profiling & Measurement) · **Requirement:** PROF-02 · **Consumer:** Phase 49 (CONS-01 / CONS-02)
**Method:** static grep + source read against the live tree (2026-08-30). No benches, no `src/` edits.
**Ranking metric (locked):** dedup leverage = (# call sites × complexity/drift-risk) — NOT raw duplicate-LOC count.

Anchors below were re-verified against the current tree; counts strip comment lines.

---

## Leverage Table

| # | Category | Call sites | Files | Complexity | Drift risk | Leverage | CONS target |
|---|----------|-----------|-------|-----------|-----------|----------|-------------|
| 1 | χ²/F survival + regularized incomplete gamma (2 independent impls) | ~17 (`inference`) + ~53 (`spm`, mostly intra-module) | `inference/dist.rs`, `spm/chi_squared.rs` | High (numerical special functions) | **High** (two hand-rolled gamma kernels can diverge in accuracy) | **HIGH** | CONS-01 |
| 2 | Permutation-test loops (build null dist, count exceedances, p-value) | 6 loop sites across 6 fns | `inference/permutation.rs`, `frechet/anova.rs`, `function_on_scalar.rs`, `famm.rs`, `explain_generic/importance.rs`, `explain/importance.rs` | Medium | **Medium-High** (3 sequential, 3 parallel — parallelization + seeding drift) | **MEDIUM-HIGH** | CONS-02 |
| 3 | Per-thread seeded RNG `StdRng::seed_from_u64(seed + k)` | 10 thread-offset sites (98 `seed_from_u64` total across 20+ files) | `gmm/em.rs`, `clustering.rs`, `coclustering.rs`, `alignment/*`, `explain/*`, `scalar_on_function/bootstrap.rs`, … | Low | Medium (determinism contract must be preserved on consolidation) | **MEDIUM** | CONS-02 |
| 4 | SVD sign-fix (`fix_svd_signs`) + one inline mirror | 3 (`regression.rs`) + 1 inline mirror (`pace_fpca.rs`) | `regression.rs`, `pace_fpca.rs` | Low-Medium | **Correctness** (silent sign-flip if the two diverge) | **LOW (correctness-critical)** | CONS-01 |
| — | Simpson/quadrature weights (`simpsons_weights`) | 161 hits | `helpers.rs` (canonical) | — | None | **ALREADY CONSOLIDATED** | — |
| — | Cholesky (`cholesky_factor` / `_forward_back` / `_solve`) | canonical + reused | `linalg.rs` (canonical), `frechet/regression.rs` reuses | — | None | **ALREADY CONSOLIDATED** | — |
| — | FPCA scoring (`fdata_to_pc_1d`, `.project()`) | 144 + 68 hits | `regression.rs` (canonical) | — | Low | **ALREADY CONSOLIDATED** | — |

---

## Detailed Findings

### 1. χ²/F survival + regularized incomplete gamma — HIGH, CONS-01

Two independent implementations of the **same** underlying regularized incomplete gamma machinery:

- `src/inference/dist.rs:99` — `pub(crate) fn chi_square_sf(x, k)` = Q(k/2, x/2) via its own `gamma_p_series` / `gamma_q_cf`. Plus `chi_square_sf_df(x, df)` at `dist.rs:118` (real-valued df, Satterthwaite).
- `src/spm/chi_squared.rs:164` — `pub(super) fn chi2_cdf(x, k)` = P(k/2, x/2) via its own `regularized_gamma_p`; `chi2_quantile(p, k)` at `:189` (Wilson-Hilferty + Newton-Raphson).

Both compute the regularized lower/upper incomplete gamma from scratch with separate series/continued-fraction code. **Drift risk is real**: the two kernels have independent accuracy tuning and could diverge under refactor. `spm` is CDF-oriented, `inference` is SF-oriented — a shared kernel exposes both.

> `dist.rs:99`: `if xx < a + 1.0 { 1.0 - gamma_p_series(a, xx) } else { gamma_q_cf(a, xx) }`
> `chi_squared.rs:164`: `regularized_gamma_p(k as f64 / 2.0, x / 2.0)`

### 2. Permutation-test loops — MEDIUM-HIGH, CONS-02

Six sites repeat the "shuffle → recompute statistic → count exceedances → p = (1+count)/(1+n_perm)" scaffold:

- `src/inference/permutation.rs:175, 238` — **sequential**
- `src/frechet/anova.rs:171` — **sequential**
- `src/explain/importance.rs:131, 221` — **sequential**
- `src/function_on_scalar.rs:831, 847` — parallel (`par_iter`)
- `src/famm.rs:861` — parallel
- `src/explain_generic/importance.rs:68` — parallel

The sequential/parallel split is itself a drift smell — the same statistical operation is parallelized in half the sites and not the others. A shared `permutation_pvalue(stat_fn, n_perm, seed)` helper (with a feature-gated parallel loop via the `parallel.rs` macros) would unify counting semantics and the (1+count)/(1+n_perm) convention.

### 3. Per-thread seeded RNG — MEDIUM, CONS-02

The thread-offset seeding contract `StdRng::seed_from_u64(seed + k as u64)` is copied at 10 sites (98 `seed_from_u64` calls total). This is the determinism-under-parallelism convention documented project-wide. A `seed_for_thread(seed, k)` helper (or `rng_for_thread`) in `helpers.rs` centralizes the contract so a future change to the offset scheme touches one place. **Must preserve determinism** — consolidation is behavior-preserving only if the offset formula is identical.

### 4. SVD sign-fix — LOW leverage, correctness-critical, CONS-01

- Canonical: `src/regression.rs:180` `fn fix_svd_signs(rotation, scores, ncomp)` — flips each component so its largest-abs loading is positive. Called at `regression.rs:381, 991`.
- Inline mirror: `src/pace_fpca.rs:219` reimplements the same convention inline, with a comment explicitly acknowledging it "mirrors `fix_svd_signs` in regression.rs".

Low call-site count, but a **silent numerical-sign divergence** if the two ever disagree (scores/loadings sign is user-visible). Promote `fix_svd_signs` to `pub(crate)` and migrate `pace_fpca` to it.

---

## Proposed Consolidation

| Item | Target location | Proposed `pub(crate)` signature |
|------|-----------------|--------------------------------|
| χ²/F survival + gamma | **new** `src/distributions.rs` (or extend `helpers.rs`) | `pub(crate) fn reg_gamma_p(a: f64, x: f64) -> f64`, `pub(crate) fn reg_gamma_q(a: f64, x: f64) -> f64`, `pub(crate) fn chi2_sf(x: f64, df: f64) -> f64`, `pub(crate) fn chi2_cdf(x: f64, df: f64) -> f64`, `pub(crate) fn chi2_quantile(p: f64, k: usize) -> f64` — migrate `inference/dist.rs` + `spm/chi_squared.rs` call sites |
| Permutation loops | `src/helpers.rs` (or `src/permutation.rs`) | `pub(crate) fn permutation_pvalue<F: Fn(&[usize]) -> f64 + Sync>(observed: f64, n: usize, n_perm: usize, seed: u64, stat: F) -> f64` (feature-gated parallel via `iter_maybe_parallel!`) |
| Seeded RNG | `src/helpers.rs` | `pub(crate) fn seed_for_thread(seed: u64, k: usize) -> StdRng` returning `StdRng::seed_from_u64(seed + k as u64)` |
| SVD sign-fix | `src/regression.rs` → `pub(crate)` (or `src/linalg.rs`) | promote existing `fn fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)` to `pub(crate)`; migrate `pace_fpca.rs:219` |

---

## Already-Consolidated (No Action)

- **`simpsons_weights` / `simpsons_weights_2d`** (`helpers.rs:57, 154`) — 161 call sites, canonical. Grep of the 9 new subsystems found **no** local Simpson/quadrature reimplementation. (`frechet` `weighted_average`/`resolve_weights` and `fts` `bartlett_weight` are *different* machinery — Fréchet barycenter weights and spectral-kernel weights, not integration quadrature — do not conflate.)
- **Cholesky** (`linalg.rs:85/113/131`) — `frechet/regression.rs:77,82,145` correctly calls `cholesky_factor` / `cholesky_forward_back` / `cholesky_solve`. No local Cholesky in the new subsystems.
- **FPCA scoring** (`fdata_to_pc_1d`, `.project()`) — 144 + 68 call sites route through the canonical `regression.rs` path; the `FpcPredictor` trait (`explain_generic/mod.rs`) already unifies projection across models.

### 9-subsystem canonical-helper confirmation

| Subsystem | Simpson | Cholesky | Verdict |
|-----------|---------|----------|---------|
| `inference` | canonical | n/a | ✓ shares `dist.rs` (itself the χ² dedup target #1) |
| `fts` | canonical | n/a | ✓ (own `bartlett_weight` is a spectral kernel, not quadrature) |
| `frechet` | canonical | reuses `crate::linalg` | ✓ |
| `density_fda` | canonical | n/a | ✓ |
| `fpca_variants` | canonical | n/a | ✓ (SVD sign-fix mirror is the separate `pace_fpca` case) |
| `face`/`irreg_fdata` | canonical | n/a | ✓ |
| `boosting_regression` | canonical | n/a | ✓ |
| `fem_smoothing` | canonical | n/a | ✓ |
| `coclustering` | canonical | n/a | ✓ (uses `seed_from_u64` — feeds RNG item #3) |

---

## Phase 49 Hand-off

Top target: **#1 χ²/F survival + gamma** (HIGH — 2 independent numerical kernels, drift-prone) → CONS-01, new `src/distributions.rs`.
Then **#2 permutation loops** (CONS-02), **#3 seeded RNG** (CONS-02), **#4 SVD sign-fix** (CONS-01, correctness-critical). Items marked "Already Consolidated" are explicitly out of scope for Phase 49.
