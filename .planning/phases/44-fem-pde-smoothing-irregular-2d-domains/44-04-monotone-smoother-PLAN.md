---
phase: 44-fem-pde-smoothing-irregular-2d-domains
plan: 04
type: execute
wave: 4
depends_on: ["44-03"]
files_modified:
  - fdars-core/src/smooth_basis.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [REP-02-04]
estimate:
  tokens: 58000
  raw_tokens: 32000
  tasks: 3
  confidence: med
must_haves:
  truths:
    - "smooth_monotone fits Ramsay's f(t)=β0+β1·∫₀ᵗ exp(w(u))du and returns a fit that is monotone (nondecreasing or nonincreasing) at EVERY consecutive pair of grid points — structurally guaranteed since f'=β1·exp(w) has constant sign"
    - "smooth_monotone recovers a known monotone curve within tolerance and tracks the data's direction (increasing vs decreasing) automatically"
    - "The Gauss-Newton fit terminates in a bounded number of iterations (converged flag + iteration count reported); even an underconverged iterate is monotone"
    - "Degenerate input (fewer than 3 points, argvals length mismatch, nbasis < 2) returns FdarError (no panic)"
  artifacts:
    - "smooth_monotone public fn + SmoothMonotoneResult struct added to fdars-core/src/smooth_basis.rs"
    - "crate-root re-export of smooth_monotone + SmoothMonotoneResult in fdars-core/src/lib.rs; SmoothMonotoneResult added to src/prelude.rs"
  key_links:
    - "w(u)=Σ_j α_j Ψ_j(u) via bspline_basis; W(t)=cumulative-trapezoid of exp(w); Jacobian cols ∂f/∂β0=1, ∂f/∂β1=W, ∂f/∂α_j=β1·cumtrap(exp(w)·Ψ_j); GN normal equations solved with pub(crate) cholesky_solve"
---

<objective>
Deliver REP-02-04: a monotone smoother added additively to `smooth_basis.rs`, using Ramsay's integral-of-exponential representation `f(t) = β₀ + β₁ ∫₀ᵗ exp(w(u)) du` with `w` expanded in a B-spline basis, fit by Gauss-Newton nonlinear least squares (no optimization crate). Monotonicity is structural: `f'(t) = β₁·exp(w(t))` has constant sign, so the fit is monotone for ANY parameter values — even an underconverged iterate. Additive — zero changes to existing `smooth_basis*` signatures.

Purpose: gives users a shape-constrained (monotone) smoother — a capability fdars lacks — matching `fda::smooth.monotone`.
Output: `smooth_monotone` public fn + `SmoothMonotoneResult` in `smooth_basis.rs`, crate-root + prelude re-exports, inline tests (monotonicity, recovery, direction auto-detect, bounded convergence, error paths).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-CONTEXT.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-RESEARCH.md

@fdars-core/src/smooth_basis.rs
@fdars-core/src/basis/bspline.rs
@fdars-core/src/linalg.rs
</context>

<artifacts_this_phase_produces>
New public symbols introduced by THIS plan, added to `smooth_basis.rs`:
- `SmoothMonotoneResult { fitted: Vec<f64> (monotone), beta0: f64, beta1: f64, w_coefficients: Vec<f64>, iterations: usize, converged: bool }` — `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]` + serde `cfg_attr` matching SmoothBasisResult.
- `smooth_monotone(data: &[f64], argvals: &[f64], nbasis: usize, order: usize, lambda: f64, max_iter: usize) -> Result<SmoothMonotoneResult, FdarError>` — `#[must_use]`.
- Crate-root re-export additions: `smooth_monotone`, `SmoothMonotoneResult`. Prelude addition: `SmoothMonotoneResult`.

Reuses (existing, unchanged): `bspline_basis(t: &[f64], nbasis, order) -> Vec<f64>` (src/basis/bspline.rs — returns m×nbasis design flat), `bspline_penalty_matrix` (roughness penalty on w), `cholesky_solve`/`cholesky_factor` (linalg, pub(crate), ROW-MAJOR), `FdMatrix`.
</artifacts_this_phase_produces>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end Gauss-Newton monotone fit on the basis grid; monotonicity oracle</name>
  <files>fdars-core/src/smooth_basis.rs</files>
  <read_first>
    - fdars-core/src/basis/bspline.rs:100 — `bspline_basis(t, nbasis, order) -> Vec<f64>` layout (m×nbasis; confirm row/col order by reading, and how to index Ψ_j(t_i)).
    - fdars-core/src/linalg.rs:85-134 — `cholesky_solve(a,b,p)` (ROW-MAJOR flat p×p SPD; returns solution p-vector).
    - RESEARCH §3 (Ramsay monotone: model f=β0+β1∫exp(w); Jacobian columns; Gauss-Newton scheme; monotonicity is structural; ≤50 iters; the numerically-safe cumulative-integral construction).
  </read_first>
  <action>
Add `SmoothMonotoneResult` to `smooth_basis.rs` (derives + `#[non_exhaustive]` + serde cfg_attr as above). Fields: `fitted: Vec<f64>`, `beta0: f64`, `beta1: f64`, `w_coefficients: Vec<f64>` (length nbasis), `iterations: usize`, `converged: bool`.

Implement `pub fn smooth_monotone(data: &[f64], argvals: &[f64], nbasis: usize, order: usize, lambda: f64, max_iter: usize) -> Result<SmoothMonotoneResult, FdarError>` with `#[must_use = "..."]`.

Validation: `let m = data.len();` require `m >= 3`, `argvals.len() == m`, `nbasis >= 2`, `order >= 1`, `lambda >= 0.0`, `max_iter >= 1` (else `InvalidDimension`/`InvalidParameter`).

**Basis:** `let psi = bspline_basis(argvals, nbasis, order);` — Ψ_j(t_i). Access `psi_at(i,j)` per the layout you read (document it). `let k = nbasis;`. Precompute the roughness penalty `let r = bspline_penalty_matrix(argvals, nbasis, order, 2);` (K×K column-major) for the α block.

**Model & cumulative integrals** (helper `fn build_W_and_integrals(alpha, psi, argvals, m, k) -> (W: Vec<f64>, Iexp_psi: Vec<f64> /* m*k */)`):
- `w_i = Σ_j alpha[j]*psi_at(i,j)`; `e_i = w_i.exp()` (finite; clamp w_i to e.g. [-30, 30] before exp to avoid overflow).
- `W[i] = ∫₀^{t_i} exp(w) du` via cumulative trapezoid over argvals: `W[0]=0`; `W[i] = W[i-1] + 0.5*(e_{i-1}+e_i)*(t_i - t_{i-1})`.
- `Iexp_psi[i*k+j] = ∫₀^{t_i} exp(w)·Ψ_j du` via the same cumulative trapezoid of `e_i*psi_at(i,j)`.

**Initialization:** `beta0 = data[0]`; `beta1 = (data[m-1]-data[0]) / (argvals[m-1]-argvals[0]).max(1e-12)` (auto-detects direction: sign(beta1) = data trend — increasing or decreasing); `alpha = vec![0.0; k]` (so w=0, e=1, W=t initially → f linear).

**Gauss-Newton loop** (`for iter in 0..max_iter`):
- Recompute `(W, Iexp_psi)` from current `alpha`.
- `f_i = beta0 + beta1*W[i]`; residual `r_i = data[i] - f_i`.
- Jacobian J is m×(2+k): col0 = 1; col1 = W[i]; col(2+j) = beta1 * Iexp_psi[i*k+j].
- Form normal equations `A = JᵀJ` ((2+k)×(2+k) ROW-MAJOR) and `g = Jᵀr`. Add Levenberg ridge to the diagonal (`A[d*P+d] += 1e-6*(1+A[d*P+d].abs())`) and add `lambda*R` into the α–α sub-block (`A[(2+a)*P + (2+b)] += lambda*r[a + b*k]`) for w-roughness regularization. P = 2+k.
- Solve `let delta = crate::linalg::cholesky_solve(&A, &g, P)?;` and update `beta0 += delta[0]; beta1 += delta[1]; alpha[j] += delta[2+j];`.
- Convergence: if `delta.iter().map(|d| d*d).sum::<f64>().sqrt() < 1e-8` → set converged=true, break. Track `iterations = iter+1`.

**Final fit:** recompute `(W,_)` from converged `alpha`; `fitted[i] = beta0 + beta1*W[i]`. Return `SmoothMonotoneResult { fitted, beta0, beta1, w_coefficients: alpha, iterations, converged }`.

Tracer test `test_smooth_monotone_is_monotone` (inline, `uniform_grid`): sample a monotone-increasing target e.g. `g(t) = t^2` (or a logistic) on m≈41 points; call `smooth_monotone(&y, &t, 8, 4, 1e-3, 50)`; assert the fitted sequence is nondecreasing: `for i in 1..m { assert!(fitted[i] >= fitted[i-1] - 1e-9); }` and all finite. (Monotonicity must hold structurally regardless of `converged`.)
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis::tests::test_smooth_monotone_is_monotone 2>&1 | tail -20</automated>
  </verify>
  <done>smooth_monotone runs the Gauss-Newton integral-of-exp fit and returns a structurally nondecreasing fit on increasing data; SmoothMonotoneResult defined.</done>
</task>

<task type="auto">
  <name>Task 2: Recovery + direction auto-detect + bounded convergence oracles</name>
  <files>fdars-core/src/smooth_basis.rs</files>
  <read_first>
    - smooth_basis.rs Task 1 — smooth_monotone init (beta1 sign = data trend), the GN loop, the converged flag.
    - RESEARCH §3 / §Validation REP-02-04 (recovery within tolerance; decreasing-direction handling via sign of beta1).
  </read_first>
  <action>
Add tests:
- `test_smooth_monotone_recovers_increasing`: on a smooth increasing target (e.g. a logistic `1/(1+exp(-8*(t-0.5)))`), assert mean-abs error between `fitted` and the target is below a tolerance (e.g. 0.1), and `converged` is true within `max_iter=50`.
- `test_smooth_monotone_decreasing`: on a decreasing target (e.g. `g(t)=1.0-t`), assert `beta1 < 0.0` (direction auto-detected) and the fitted sequence is NONincreasing (`fitted[i] <= fitted[i-1] + 1e-9`).
- `test_smooth_monotone_bounded_iterations`: assert `result.iterations <= 50` and, on noisy monotone data, the fitted sequence is still monotone even if `converged == false` (structural guarantee).

If any recovery tolerance proves too tight for the GN scheme on a fixed grid, widen the tolerance to a defensible value (document why) — do NOT weaken the monotonicity assertion, which must always hold.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis::tests::test_smooth_monotone_recovers_increasing smooth_basis::tests::test_smooth_monotone_decreasing smooth_basis::tests::test_smooth_monotone_bounded_iterations 2>&1 | tail -20</automated>
  </verify>
  <done>smooth_monotone recovers a known monotone curve within tolerance, auto-detects decreasing direction (beta1<0 → nonincreasing fit), and terminates within max_iter with monotonicity holding regardless of convergence.</done>
</task>

<task type="auto">
  <name>Task 3: Error paths + crate-root/prelude re-exports</name>
  <files>fdars-core/src/smooth_basis.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/lib.rs:379 — `pub use smooth_basis::{ ... }` block (already extended by Plan 03; add the two monotone symbols).
    - fdars-core/src/prelude.rs — re-export style.
    - smooth_basis.rs Task 1 — smooth_monotone validation branches.
  </read_first>
  <action>
Extend the `pub use smooth_basis::{ ... }` block in `lib.rs` to include `smooth_monotone` and `SmoothMonotoneResult` (additive to Plan 03's additions). Add `SmoothMonotoneResult` to the `smooth_basis::{...}` re-export in `src/prelude.rs`.

Add error-path tests:
- `test_smooth_monotone_errors_on_short_input`: `data.len() == 2` → `Err` (needs >= 3).
- `test_smooth_monotone_errors_on_argvals_mismatch`: `argvals.len() != data.len()` → `Err`.
- `test_smooth_monotone_errors_on_bad_params`: `nbasis == 1` (or `max_iter == 0`) → `Err(FdarError::InvalidParameter { .. })`. No panic in any case.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis::tests::test_smooth_monotone 2>&1 | tail -25</automated>
  </verify>
  <done>Degenerate/invalid inputs return FdarError with no panic; smooth_monotone + SmoothMonotoneResult re-exported at crate root and prelude; full smooth_basis::tests module green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → smooth_monotone | Untrusted numeric response + grid cross into an iterative Gauss-Newton solve |

Attack surface: none — pure in-process numeric library. Concerns are numerical only.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-44-08 | Elevation (incorrect output) | exp(w) overflow for large w | medium | mitigate | clamp w to [-30,30] before exp; finite by construction |
| T-44-09 | Denial (non-termination) | Gauss-Newton fails to converge | low | mitigate | hard cap max_iter; Levenberg ridge on JᵀJ; converged flag reported; fit is monotone regardless |
| T-44-10 | Elevation | singular JᵀJ (rank-deficient Jacobian) | medium | mitigate | Levenberg ridge + lambda·R on the α block make A SPD; cholesky_solve errors surface as ComputationFailed |
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis` — all module tests green.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
</verification>

<success_criteria>
- `smooth_monotone` returns a structurally monotone fit (nondecreasing or nonincreasing) for any input, recovers a known monotone curve within tolerance, and auto-detects direction.
- Gauss-Newton terminates within max_iter (converged flag + iteration count reported).
- Degenerate/invalid inputs return FdarError with no panic.
- `smooth_monotone` + `SmoothMonotoneResult` re-exported at crate root and prelude; existing `smooth_basis*` signatures unchanged.
</success_criteria>

<output>
Create `.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-04-SUMMARY.md` when done.
</output>
