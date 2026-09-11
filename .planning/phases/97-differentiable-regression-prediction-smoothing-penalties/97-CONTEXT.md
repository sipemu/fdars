# Phase 97: Differentiable Regression Prediction & Smoothing Penalties - Context

**Gathered:** 2026-09-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Make scalar-on-function regression PREDICTION (`fregre_lm` / FPCR path) differentiable w.r.t. the input curve, and roughness/smoothing-PENALTY VALUE evaluation differentiable w.r.t. the spline coefficients — both generic over `Scalar`, with f64 numerics preserved (families 2 & 3 of DIF-F2 / DOP-02 + DOP-03). FD-checked at forward-mode `Dual` and reverse-mode `Var`.

**In scope:**
- **DOP-02:** a new per-curve generic prediction `predict_curve_generic<S: Scalar>(curve: &[S], fit: &FregreLmResult) -> S` composing Phase 94's `project_scores_generic<S>` + the coefficient combination `intercept + Σ_k coefficients[1+k]·score_k`. Differentiable w.r.t. `curve`.
- **DOP-03:** a new standalone penalty-value evaluation `penalty_value_generic<S: Scalar>(coef: &[S], penalty: &FdMatrix, lambda: f64) -> S` computing `λ·cᵀRc`, differentiable w.r.t. the coefficients `coef`.
- f64-parity tests + Dual/Var FD-checked gradient tests for both.

**Out of scope (deferred / stays f64):**
- The public batch `predict_fregre_lm(fit, new_data: &FdMatrix, new_scalar) -> Vec<f64>` signature stays UNCHANGED (may internally delegate to the generic per-curve fn at `S=f64`).
- Scalar covariates in prediction stay f64 (the natural differentiable input is the functional curve, not the scalar z).
- Penalty MATRIX construction (`bspline_penalty_matrix`, `fourier_penalty_matrix`, `difference_matrix`, `integrate_symmetric_penalty`, `differentiate_basis_columns`) stays f64 — R is a fixed design matrix, not curve data.
- Differentiating the penalty w.r.t. raw curve values through the smoothing solve (Cholesky / linear solve), or w.r.t. the smoothing parameter λ — explicitly deferred (user decision: differentiate w.r.t. coefficients c only). Do NOT make the smoothing FIT differentiable.
- Depth + curve distances → Phase 98 (DOP-04).
</domain>

<decisions>
## Implementation Decisions

### DOP-02 — Differentiable prediction (user-confirmed)
- **New generic per-curve fn** `predict_curve_generic<S: Scalar>(curve: &[S], fit: &FregreLmResult) -> S`:
  1. `let scores = project_scores_generic::<S>(curve, &fit.fpca.mean, &fit.fpca.rotation, &fit.fpca.weights, fit.ncomp);` (Phase 94, reused unchanged)
  2. `let mut yhat = S::from_f64(fit.intercept); for k in 0..fit.ncomp { yhat += S::from_f64(fit.coefficients[1 + k]) * scores[k]; }`
  3. return `yhat`.
- **Differentiable input: the curve `&[S]`.** The prediction is linear in the curve; `d(ŷ)/d(x[j]) = Σ_k coef[1+k]·rotation[j,k]·weights[j]` (constant). All model params (`intercept`, `coefficients`, fpca mean/rotation/weights) stay f64, lifted via `S::from_f64` only where combined with `S`.
- **Scalar covariates stay f64** and are NOT part of the generic per-curve prediction (they're a separate additive `Σ gamma_j·z_j` term the batch f64 path keeps). If a generic prediction must include them for parity with a specific test, pass them as f64 constants lifted via `from_f64` — but the differentiable path is the curve only.
- **`predict_fregre_lm` public signature UNCHANGED.** Refactor its inner per-curve loop to call `predict_curve_generic::<f64>` so the batch result is identical (bit-for-bit — `project_scores_generic::<f64>` reproduces the current 3-mul kernel; verify the existing `test_predict_fregre_lm_on_training_data` still passes to 1e-6). This is the DOP-02 non-breaking guard.

### DOP-03 — Differentiable roughness penalty (user decision: differentiate w.r.t. coefficients c)
- **New standalone fn** `penalty_value_generic<S: Scalar>(coef: &[S], penalty: &FdMatrix, lambda: f64) -> S` computing `λ · Σ_i Σ_j coef[i] · R[i,j] · coef[j]` where `R` (the penalty matrix, e.g. from `bspline_penalty_matrix` / `fourier_penalty_matrix` / `DᵀD`) stays f64, `lambda` stays f64, and `coef` is the differentiable `S` input.
  - Analytic gradient: `d(λ cᵀRc)/d(c_i) = λ·((R + Rᵀ)c)_i = 2λ(Rc)_i` for symmetric R.
  - Mechanics: `sum = S::zero(); for i { for j { sum += coef[i] * S::from_f64(R[i,j]) * coef[j]; } } return S::from_f64(lambda) * sum;` — fold the f64 `R[i,j]` once via `from_f64`; keep the accumulation order stable for f64 parity.
- **Differentiable input: the coefficients `coef: &[S]`.** R matrix + lambda stay f64.
- **f64 parity:** since this is a NEW function, "f64 penalty values unchanged" means `penalty_value_generic::<f64>` equals a direct hand-written `λ·cᵀRc` f64 computation (bit-identical / 1e-12). It does NOT change any existing smoothing fit — the fit's normal-equations path is untouched (this is an additive evaluation function, not a rewrite of the solve).
- **Penalty matrix location:** accept `penalty: &FdMatrix` (or `&[f64]` column-major + dim) — Claude's discretion on the exact matrix carrier; it must interoperate with the existing `bspline_penalty_matrix`/`fourier_penalty_matrix` (which return `Vec<f64>` column-major) and `difference_matrix` (returns nalgebra `DMatrix<f64>` → `DᵀD`). Provide a small adapter if needed; do NOT change those public constructors.

### Pattern & non-breaking (continues Phase 94/95/96 decisions)
- Generic via plain `<S: Scalar>` (no `= f64` default — Rust 1.97 rejects it on free fns). New functions are ADDITIVE; existing public signatures (`predict_fregre_lm`, penalty-matrix constructors) unchanged.
- Curve/coefficients (the optimization variables) → `S`; model params / R / lambda / mean / rotation / weights → `f64`, lifted via `S::from_f64` only where mixed.
- Non-breaking proof: full suite + 28 examples + serde + wasm + clippy `--all-targets` green; churn confined to the touched files (scalar_on_function/fregre_lm.rs, a penalty-eval location, + their tests); no new dependency.

### Tests & validation
- f64 parity: `predict_curve_generic::<f64>` matches `predict_fregre_lm` per-curve (and the existing prediction test stays green to 1e-6); `penalty_value_generic::<f64>` matches a direct `λ·cᵀRc` reference (bit-identical / 1e-12).
- Autodiff FD: gradient of prediction w.r.t. curve, and gradient of penalty w.r.t. coef, match central FD (h=1e-6, tol robust `1e-6*(1.0+fd.abs())` — the absolute-floor form adopted in Phase 96 to avoid false failures at zero-gradient points) at BOTH `Dual` and `Var(vjp)`.
- Mirror the Phase 94/95/96 tiered test pattern; RED-first.

### Claude's Discretion
- Exact module placement of `predict_curve_generic` (scalar_on_function/fregre_lm.rs beside `predict_fregre_lm`) and `penalty_value_generic` (smooth_basis.rs, or a small penalty module); the penalty-matrix carrier type (`&FdMatrix` vs `&[f64]`+dims); whether to expose both at crate root/prelude (prefer yes, additive, for Phase 99's unified API). Guided by the no-new-dependency / non-breaking constraints.
</decisions>

<code_context>
## Existing Code Insights

### Reuse (already generic — do NOT re-touch)
- `project_scores_generic<S: Scalar>(curve: &[S], mean: &[f64], rotation: &FdMatrix, weights: &[f64], ncomp) -> Vec<S>` — regression.rs:232 (Phase 94). The core of FPCR prediction. `FpcaResult::project_generic<S>` method wraps it (regression.rs:139).
- `Scalar` trait autodiff/mod.rs:38-85; `Dual` forward.rs; `Var`/`vjp` reverse.rs; `f64::from_f64` identity.

### Generalization targets (NEW work)
- `predict_fregre_lm(fit: &FregreLmResult, new_data: &FdMatrix, new_scalar: Option<&FdMatrix>) -> Vec<f64>` — scalar_on_function/fregre_lm.rs:444. Inner per-curve kernel at 454–471: `yhat = intercept + Σ_k coef[1+k]·(Σ_j (x_j-mean_j)·rotation[j,k]·weights[j]) + Σ scalar terms`. Extract the functional part as `predict_curve_generic<S>`.
- `FregreLmResult` — scalar_on_function/mod.rs:62–98 (intercept, fpca: FpcaResult, coefficients [α,γ₁..γ_K,z₁..z_p], gamma, ncomp).
- Penalty value: does NOT exist standalone — currently baked into smoothing normal equations (smooth_basis.rs:~234 `Φ'Φ + λR`; pspline.rs:~90 `btb + λ·DᵀD`). Create `penalty_value_generic<S>`.

### Stay f64 (do NOT generalize)
- `bspline_penalty_matrix` (smooth_basis.rs:112), `fourier_penalty_matrix` (smooth_basis.rs:159) — public, return Vec<f64>.
- `differentiate_basis_columns` (smooth_basis.rs:988, pub(crate)), `integrate_symmetric_penalty` (smooth_basis.rs:1011, pub(crate)) — R construction.
- `difference_matrix` (basis/pspline.rs:9, public, returns DMatrix<f64>).
- The smoothing FIT / linear solve (smooth_basis, pspline_fit) — untouched.

### Public API / non-breaking anchors
- `scalar_on_function/mod.rs:46` re-exports `predict_fregre_lm` (public, Vec<f64> — signature frozen).
- Crate-root / prelude re-exports — add the new generic fns additively (Claude's discretion).

### Tests
- `scalar_on_function/tests.rs:117` `test_predict_fregre_lm_on_training_data` (tol 1e-6) — must stay green (DOP-02 parity guard).
- `smooth_basis.rs:1318+` penalty-matrix tests (symmetry, PSD, tol 1e-10) — unchanged; add a new penalty-value parity + FD test.

</code_context>

<specifics>
## Specific Ideas

- DOP-02 prediction is LINEAR in the curve (gradient constant) — `project_scores_generic` already encodes exactly this; the new fn just adds the coefficient combination. Cheap and clean.
- DOP-03: the penalty is a NEW standalone evaluation `λ·cᵀRc` — it does not alter the existing smoothing solve; "f64 parity" is against a direct reference computation, not a pre-existing standalone value (there wasn't one).
- Differentiable inputs: curve (DOP-02), coefficients (DOP-03). Everything else f64.
- Reuse the Phase 96 robust FD tolerance `1e-6*(1.0+fd.abs())`.
</specifics>

<deferred>
## Deferred Ideas

- Differentiating the penalty w.r.t. raw curve values through the smoothing solve, or w.r.t. λ — explicitly deferred (user chose coefficients-only). Would require differentiating a Cholesky/linear solve.
- Making scalar covariates a differentiable input — deferred (functional curve is the differentiable input).
- Depth + curve distances → Phase 98 (DOP-04). Unified gradient API + composition demo → Phase 99.
</deferred>
