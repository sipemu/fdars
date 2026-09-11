---
phase: 97-differentiable-regression-prediction-smoothing-penalties
plan: 02
type: execute
wave: 2
depends_on: [97-01]
files_modified:
  - fdars-core/src/smooth_basis.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [DOP-03]
estimate:
  tokens: 50000
  raw_tokens: 27000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "penalty_value_generic::<f64>(coef, R, lambda) is bit-identical to a direct inline λ·cᵀRc reference within 1e-12 (DOP-03)"
    - "Gradient of penalty_value_generic w.r.t. coef matches central FD at Dual (forward) within 1e-6*(1+|fd|)"
    - "Gradient of penalty_value_generic w.r.t. coef matches central FD at Var (reverse/vjp) within 1e-6*(1+|fd|)"
    - "bspline_penalty_matrix / fourier_penalty_matrix / difference_matrix constructor signatures stay UNCHANGED"
  artifacts:
    - "fdars-core/src/smooth_basis.rs — pub fn penalty_value_generic<S: Scalar>(coef: &[S], penalty: &FdMatrix, lambda: f64) -> S"
    - "fdars-core/src/smooth_basis.rs #[cfg(test)] — parity(1e-12) + Dual FD + Var FD tests for penalty_value_generic"
    - "fdars-core/src/lib.rs + prelude.rs — additive re-exports of penalty_value_generic"
  key_links:
    - "penalty carrier is &FdMatrix; call sites wrap bspline_penalty_matrix's Vec<f64> via FdMatrix::from_column_major(vec,k,k)"
    - "penalty_value_generic re-exported from crate root + prelude (additive, for Phase 99 unified API)"
---

<objective>
Add the standalone differentiable roughness-penalty evaluation `penalty_value_generic<S: Scalar>(coef: &[S], penalty: &FdMatrix, lambda: f64) -> S` (DOP-03), computing `λ · Σ_i Σ_j coef[i]·R[i,j]·coef[j]` where the penalty matrix R and lambda stay f64 and only `coef` is the differentiable `S` input. Expands out from the tracer (Plan 01) proved the differentiable substrate; this adds the second differentiable family (penalty w.r.t. coefficients) with the same Scalar / from_f64-lift idiom.

Purpose: enables autodiff of the roughness penalty w.r.t. spline coefficients — the DOP-03 requirement.
Output: `penalty_value_generic` (additive), its three tests, additive crate-root + prelude re-exports. Penalty-matrix constructors and the smoothing solve stay untouched.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/97-differentiable-regression-prediction-smoothing-penalties/97-RESEARCH.md
@.planning/phases/97-differentiable-regression-prediction-smoothing-penalties/97-PATTERNS.md

@fdars-core/src/smooth_basis.rs
@fdars-core/src/matrix.rs
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED tests — penalty f64 bit-identity parity (1e-12) + combined Dual/Var FD w.r.t. coef</name>
  <files>fdars-core/src/smooth_basis.rs</files>
  <read_first>
    - 97-PATTERNS.md lines 365–477 (the three DOP-03 test skeletons: f64 parity 1e-12, Dual FD, Var FD — copy verbatim)
    - 97-RESEARCH.md lines 480–551 (test code examples) and lines 595–599 (verbatim tolerances)
    - fdars-core/src/smooth_basis.rs lines 1312–1315 (test module: `use super::*;` + `use crate::test_helpers::uniform_grid;`) and lines 1318+ (existing frozen penalty-matrix tests)
    - fdars-core/src/matrix.rs lines 50–100 (from_column_major / nrows / ncols API)
  </read_first>
  <behavior>
    - test_penalty_value_generic_f64_parity: uniform_grid(51); penalty_vec = bspline_penalty_matrix(&t,8,4,2); k=sqrt(len); penalty_mat = FdMatrix::from_column_major(penalty_vec.clone(),k,k).unwrap(); lambda=2.0; coef[i]=(i*0.2+0.5).cos(); reference = lambda·Σ_i Σ_j coef[i]·penalty_vec[i+j*k]·coef[j]; assert (penalty_value_generic::<f64>(&coef,&penalty_mat,lambda) - reference).abs() < 1e-12.
    - test_penalty_value_generic_dual_fd_check: bspline_penalty_matrix(&t,10,4,2); lambda=0.5; coef[i]=(i*0.3+0.1).sin(); per index seed Dual, .extract() tangent, compare to central FD (h=1e-6) tol 1e-6*(1+|fd|).
    - test_penalty_value_generic_var_fd_check: same fixture; vjp(|c:&[Var]| penalty_value_generic::<Var>(c,&pm_clone,lambda), &coef_f64); compare grad[i] to central FD, same tol. Note vjp closure must capture an owned clone of the FdMatrix (pm_clone = penalty_mat.clone()).
  </behavior>
  <action>Append the three tests to the existing `#[cfg(test)] mod tests` block at the bottom of smooth_basis.rs (module already has `use super::*;` and `use crate::test_helpers::uniform_grid;`, so bspline_penalty_matrix and FdMatrix are in scope). Add the autodiff imports each test needs inside the test fns per the PATTERNS.md skeletons: `use crate::autodiff::Dual;` and `use crate::autodiff::{vjp, Var};`. Copy the parity test (tol 1e-12 / bit-identical — DOP-03 IS bit-identical per CRITICAL_DESIGN_NOTES because R is symmetric by construction and f64::from_f64 is the identity), the Dual FD test, and the Var FD test from 97-PATTERNS.md lines 373–477 verbatim, adjusting only names. The reference computation must index penalty_vec column-major as `penalty_vec[i + j * k]`. Do NOT modify the existing penalty-matrix tests. These tests reference penalty_value_generic which does not yet exist — they MUST fail RED (compile error naming penalty_value_generic). Run `cargo fmt` then commit `git commit --no-verify`.</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel smooth_basis::tests::test_penalty_value_generic 2>&1 | tail -20</automated>
    <fails_when>The three tests compile and pass (they must fail RED — penalty_value_generic is not defined, so this is expected to be a compile error naming penalty_value_generic).</fails_when>
  </verify>
  <done>Three new tests exist referencing penalty_value_generic; the build fails RED with an unresolved `penalty_value_generic`. Existing penalty-matrix tests untouched.</done>
  <reversibility rating="reversible">Additive test code; revertible by deleting the three test fns.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add penalty_value_generic<S: Scalar> + additive re-exports — GREEN</name>
  <files>fdars-core/src/smooth_basis.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - 97-PATTERNS.md lines 139–234 (import to add + the penalty_value_generic body + accumulation idiom + adapter pattern)
    - 97-RESEARCH.md lines 226–291 (recipe Q-1, carrier decision Q-1a, symmetry Q-1b) and Pitfall 3 (lines 327–333) + Pitfall 5 (lines 343–349)
    - fdars-core/src/smooth_basis.rs lines 12–16 (current imports — already has `use crate::matrix::FdMatrix;`) and lines ~159–188 (fourier_penalty_matrix — insert the new fn after it)
    - fdars-core/src/lib.rs line 495 (bspline_penalty_matrix / fourier_penalty_matrix re-export line) and fdars-core/src/prelude.rs (add beside the smooth_basis / regression re-exports)
  </read_first>
  <action>In smooth_basis.rs add one import beside the existing ones: `use crate::autodiff::Scalar;` (FdMatrix is already imported at line 14). After fourier_penalty_matrix (~line 188) add the additive `#[must_use] pub fn penalty_value_generic<S: Scalar>(coef: &[S], penalty: &FdMatrix, lambda: f64) -> S` exactly per 97-PATTERNS.md lines 199–217: `let k = coef.len();`, `debug_assert_eq!(penalty.nrows(), k, ...)`, `let mut sum = S::zero();`, double loop `for i in 0..k { for j in 0..k { sum += coef[i] * S::from_f64(penalty[(i, j)]) * coef[j]; } }`, return `S::from_f64(lambda) * sum`. Derive k from `coef.len()` (RESEARCH Pitfall 5), lift each R[i,j] once via from_f64 and lift lambda once outside the loops (RESEARCH Pitfall 3), i-outer/j-inner order for f64 parity. Plain `<S: Scalar>` no `= f64` default. Include the doc comment from PATTERNS.md lines 184–198. Do NOT touch bspline_penalty_matrix / fourier_penalty_matrix / integrate_symmetric_penalty / difference_matrix or the smoothing solve. In lib.rs add `penalty_value_generic` to the re-export list beside bspline_penalty_matrix / fourier_penalty_matrix (line ~495). In prelude.rs add `pub use crate::smooth_basis::penalty_value_generic;`. Run `cargo fmt` then commit `git commit --no-verify`.</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core && cargo test -p fdars-core --features linalg,parallel smooth_basis::tests::test_penalty_value_generic 2>&1 | tail -20</automated>
    <fails_when>Any penalty_value_generic test fails — parity beyond 1e-12 (check column-major indexing / accumulation order), or a Dual/Var gradient beyond 1e-6*(1+|fd|).</fails_when>
  </verify>
  <done>All three penalty_value_generic tests pass GREEN: f64 parity within 1e-12 (bit-identical), Dual gradient matches FD, Var gradient matches FD. penalty_value_generic reachable via crate root and prelude. Penalty-matrix constructors unchanged.</done>
  <reversibility rating="reversible">Additive function + additive re-exports; no existing signature or body changed.</reversibility>
</task>

<task type="auto">
  <name>Task 3: Confirm penalty-matrix constructors + existing smooth_basis tests unchanged, fmt, no-verify commit</name>
  <files>fdars-core/src/smooth_basis.rs</files>
  <read_first>
    - 97-RESEARCH.md lines 632 (existing penalty-matrix tests that must stay green: symmetric, PSD, fourier diagonal)
    - fdars-core/src/smooth_basis.rs lines 112–188 (bspline_penalty_matrix / fourier_penalty_matrix signatures — must stay unchanged)
  </read_first>
  <action>Run the full smooth_basis module test suite to confirm the existing penalty-matrix tests (test_bspline_penalty_matrix_symmetric, positive_semidefinite, fourier diagonal) and every other smooth_basis test stay green alongside the three new ones. Then grep to confirm bspline_penalty_matrix and fourier_penalty_matrix public signatures are unchanged. `cargo fmt` then `git commit --no-verify` any residual fmt changes (skip commit if nothing changed).</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel smooth_basis:: 2>&1 | tail -15 && grep -nE "pub fn bspline_penalty_matrix\(|pub fn fourier_penalty_matrix\(" fdars-core/src/smooth_basis.rs</automated>
    <fails_when>Any smooth_basis test fails (test result: FAILED in the tail), OR the grep prints fewer than two lines (a penalty-matrix constructor signature was renamed/removed).</fails_when>
  </verify>
  <done>Full smooth_basis test suite is green (0 failed) including the existing penalty-matrix tests and the three new penalty_value_generic tests; both `pub fn bspline_penalty_matrix(` and `pub fn fourier_penalty_matrix(` still present unchanged.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure in-crate numeric quadratic-form evaluation over a caller-provided scalar slice + f64 matrix; no external input, network, IO, or auth. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-97-02 | (n/a) | penalty_value_generic | low | accept | No attack surface — pure numeric penalty evaluation over caller-provided scalar/f64 slices; no external input/network/IO/auth. No packages installed (in-crate only). |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel smooth_basis::` green (0 failed), including the three new tests and the frozen penalty-matrix tests.
- `pub fn bspline_penalty_matrix(` and `pub fn fourier_penalty_matrix(` signatures present and unchanged.
- penalty_value_generic reachable from crate root and prelude (additive re-exports).
</verification>

<success_criteria>
- penalty_value_generic::<f64> is bit-identical to a direct λ·cᵀRc reference within 1e-12.
- Penalty gradient w.r.t. coef matches central FD (h=1e-6) at both Dual and Var within 1e-6*(1+|fd|).
- Penalty-matrix constructors + smoothing solve byte-unchanged; existing suite green.
- No new crate dependency; git diff confined to the three owned files.
</success_criteria>

<output>
Create `.planning/phases/97-differentiable-regression-prediction-smoothing-penalties/97-02-SUMMARY.md` when done.
</output>