---
phase: "64"
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/optimal_design.rs
autonomous: true
requirements: [FOD-01, FOD-03]
estimate:
  tokens: 55000
  raw_tokens: 55000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "design_criterion(model, &[], DesignCriterion::Trajectory) returns Σ_k λ_k (per D-Trajectory, FOD-01)"
    - "MSE(∅) is grid-invariant across m=21/51/101 because simpsons_weights is used, not uniform 1/m (FOD-01)"
    - "Adding any grid point to the design does not increase the trajectory criterion: criterion(S∪{t}) ≤ criterion(S) + 1e-12 (FOD-01)"
    - "Out-of-range index, sigma2 <= 0, and ncomp == 0 each return FdarError::InvalidParameter (FOD-03)"
    - "Near-singular Σ_d triggers a 1e-8 ridge-retry and never panics (FOD-01)"
  artifacts:
    - "fdars-core/src/optimal_design.rs (new file: DesignCriterion + OptimalityKind enums, design_criterion, build_sigma_design, trajectory branch)"
  key_links:
    - "build_sigma_design assembles p×p Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p (p = selected.len()), NOT K×K — the shared solve both branches depend on"
    - "trajectory branch calls helpers::simpsons_weights(&model.argvals) — the link that makes MSE grid-invariant"
    - "cholesky_solve / cholesky_factor from linalg.rs (pub(crate), NOT behind linalg feature) — the solve backing the quadratic form"
---

<objective>
Land the shared numerical core of the FOptDes criterion machinery: the new `fdars-core/src/optimal_design.rs` file carrying the public `DesignCriterion` / `OptimalityKind` enum pair, the `design_criterion` entry point with full input validation, the shared private `build_sigma_design` helper (with ridge-retry), and ONE fully-verified criterion branch — the trajectory-reconstruction BLUP-MSE (FOD-01) — with its known-answer tests green.

This is the tracer slice: prove the Σ_d assembly, the cholesky_solve wiring, the Simpson-weighted integral, the validation guards, and the ridge-retry end-to-end on the trajectory path before the second criterion branch expands from the same helper in plan 64-02.

Purpose: Front-load every numerical make-or-break gate. Every downstream selection result (Phase 65 greedy loop) inherits any bug in Σ_d assembly or integration weighting, so it must be proven here first.

Output: New file `fdars-core/src/optimal_design.rs` compiling standalone (not yet wired into lib.rs — that is plan 64-02), with the trajectory branch and its known-answer/grid-invariance/monotonicity/validation/ridge-retry tests passing.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md

@.planning/phases/64-criterion-machinery-core/64-CONTEXT.md
@.planning/phases/64-criterion-machinery-core/64-RESEARCH.md
@.planning/phases/64-criterion-machinery-core/64-VALIDATION.md

@fdars-core/src/pace_fpca.rs
@fdars-core/src/linalg.rs
@fdars-core/src/helpers.rs
@fdars-core/src/kshape.rs
@fdars-core/src/matrix.rs
</context>

<artifacts_produced>
## Artifacts this plan produces

NEW file `fdars-core/src/optimal_design.rs` containing:
- `pub enum DesignCriterion` — variants `Trajectory` and `Score(OptimalityKind)`. Derives `Debug, Clone, PartialEq`; serde-gated `Serialize, Deserialize` via `#[cfg_attr(feature = "serde", ...)]`. (Plan 64-01 defines both variants; the `Score` branch body is implemented in 64-02.)
- `pub enum OptimalityKind` — variants `A` and `D`. Same derives + serde gating.
- `pub fn design_criterion(model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion) -> Result<f64, FdarError>` — `#[must_use]`, entry-point validation, dispatch. In this plan the `Score` arm may delegate to a `score` impl stub that returns `Ok(0.0)` ONLY as a compile placeholder if needed; plan 64-02 replaces it with the real posterior-covariance math. (Do NOT ship the stub as a shortcut — 64-02 must overwrite it.)
- `build_sigma_design(model: &PaceFpcaResult, selected: &[usize]) -> Result<Vec<f64>, FdarError>` — private; p×p Σ_d, row-major.
- private trajectory branch (e.g. `trajectory_criterion`) + any ridge-retry helper.
- `#[cfg(test)] mod tests` with the trajectory known-answer gates.

NOTE: `lib.rs` re-export is intentionally NOT in this plan — it lands in 64-02 so the two enums, `design_criterion`, AND the complete Score branch go public together.
</artifacts_produced>

<tasks>

<task type="tracer">
  <name>Task 1: build_sigma_design + validation + trajectory branch — one path end-to-end</name>
  <files>fdars-core/src/optimal_design.rs</files>
  <read_first>
    - fdars-core/src/pace_fpca.rs — PaceFpcaResult fields (`eigenvalues: Vec<f64>` len ncomp, `eigenfunctions: FdMatrix` m×ncomp column-major, `argvals: Vec<f64>` len m, `sigma2: f64`, `ncomp: usize`); Σ_yi assembly ~461–474; ridge-retry ~480–490.
    - fdars-core/src/linalg.rs — `cholesky_solve(a, b, p)`, `cholesky_factor(a, p)`, `cholesky_forward_back(l, b, p)`; all `pub(crate)`, row-major, NOT behind the linalg feature. Cholesky fails when a diagonal element <= 1e-12.
    - fdars-core/src/helpers.rs — `simpsons_weights(&argvals)` returns weights of length m; returns `vec![1.0; n]` for n < 2.
    - fdars-core/src/matrix.rs — FdMatrix column-major: element (row, col) at `row + col * nrows`; access via `model.eigenfunctions[(j, k)]`.
    - fdars-core/src/kshape.rs — peer-module convention: module doc `//!`, `#[non_exhaustive]` on public result structs, standard derives, serde-gated derives, inline `#[cfg(test)] mod tests`.
    - fdars-core/src/error.rs — FdarError variants `InvalidParameter { parameter, message }`, `ComputationFailed { operation, detail }`.
  </read_first>
  <action>
Create `fdars-core/src/optimal_design.rs` with a module doc comment (`//!`) summarizing the FOptDes criterion machinery and citing Ji and Müller (2017).

Declare `pub enum DesignCriterion { Trajectory, Score(OptimalityKind) }` and `pub enum OptimalityKind { A, D }`, each with `#[derive(Debug, Clone, PartialEq)]` and `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`. Doc-comment every variant.

Implement `#[must_use] pub fn design_criterion(model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion) -> Result<f64, FdarError>`. At entry, validate defensively and return `FdarError::InvalidParameter` with a contextual message on each failure: `model.ncomp == 0`; `model.sigma2 <= 0.0`; any `idx` in `selected` where `idx >= model.argvals.len()`. Duplicate indices are tolerated (document this in a doc comment; greedy excludes them upstream in Phase 65). Then `match criterion` and dispatch: `Trajectory` to the trajectory impl. For the `Score` arm, delegate to a `score_criterion(model, selected, kind)` that this plan MAY leave as a minimal placeholder returning `Ok(0.0)` purely so the file compiles — plan 64-02 overwrites it with the real math and its tests. Do not test the Score path in this plan.

Implement private `build_sigma_design(model, selected) -> Result<Vec<f64>, FdarError>`: let `p = selected.len()`, `ncomp = model.ncomp`. Allocate `vec![0.0_f64; p * p]` row-major. For each `row in 0..p` (design point `selected[row]`) and `col in 0..p` (design point `selected[col]`), accumulate over `k in 0..ncomp`: `model.eigenfunctions[(selected[row], k)] * model.eigenvalues[k] * model.eigenfunctions[(selected[col], k)]`, storing into `sigma_d[row * p + col]`. After the col loop, add `model.sigma2` to `sigma_d[row * p + row]` (the σ²I_p diagonal — mirror of pace_fpca.rs:461–474). Returning the empty Vec for `p == 0` is fine; the empty-set fast path in the trajectory branch never calls the solve.

Implement a ridge-retry solve helper mirroring pace_fpca.rs:480–490: attempt `cholesky_solve(&sigma_d, &rhs, p)`; on `Err`, add `1e-8` to every diagonal `sigma_d[i*p+i]` and retry once; if the retry still fails, return `FdarError::ComputationFailed { operation: "optimal_design Sigma_d Cholesky", detail: "Cholesky failed after 1e-8 ridge; sigma2 may be too small" }`. Never panic. For per-grid-point efficiency prefer factoring Σ_d ONCE via `cholesky_factor` (with the same ridge-retry-on-fail wrapper) then calling `cholesky_forward_back` inside the grid loop, so the trajectory branch is O(m·p²), not O(m·p³).

Implement the private trajectory branch (FOD-01): the integrated Simpson-weighted conditional BLUP-MSE. Let `m = model.argvals.len()`, `ncomp = model.ncomp`, `weights = helpers::simpsons_weights(&model.argvals)` (length m — NEVER a uniform `1.0/m`; that is the grid-scale bug). Let `p = selected.len()`. If `p == 0`, return `Σ_j weights[j] · (Σ_k λ_k · model.eigenfunctions[(j,k)]²)` — the empty-set prior. Otherwise, for each grid point `j in 0..m`: (1) `prior_var_j = Σ_k model.eigenvalues[k] * model.eigenfunctions[(j,k)].powi(2)`; (2) build the cross-covariance p-vector `rhs_j[i] = Σ_k model.eigenvalues[k] * model.eigenfunctions[(j,k)] * model.eigenfunctions[(selected[i],k)]` for `i in 0..p`; (3) solve `Σ_d · v = rhs_j` using the pre-factored Cholesky (forward/back) and compute `reduction_j = dot(rhs_j, v)`; (4) accumulate `mse += weights[j] * (prior_var_j - reduction_j)`. Return `mse`. Optionally cache the p×ncomp `Φ_d` sub-matrix (`Φ_d[i,k] = model.eigenfunctions[(selected[i],k)]`) once before the grid loop to avoid O(m·p·K) re-reads.

Do NOT add `#[cfg(feature = "linalg")]` anywhere — the Cholesky helpers are always compiled. Do NOT use `linalg::cholesky_d` (wrong threshold); use `cholesky_solve` / `cholesky_factor` / `cholesky_forward_back`. Follow project conventions: no panics on input validation, `#[must_use]` on `design_criterion`, standard derives, MSRV 1.81.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core 2>&1 | tail -5</automated>
    <fails_when>Output contains "error[" or "error:" or "cannot find" — the module does not compile standalone.</fails_when>
  </verify>
  <acceptance_criteria>
    - `fdars-core/src/optimal_design.rs` exists with `DesignCriterion`, `OptimalityKind`, `design_criterion`, `build_sigma_design`, and the trajectory branch.
    - `design_criterion` validates ncomp==0, sigma2<=0, and out-of-range index at entry, each returning `FdarError::InvalidParameter`.
    - `build_sigma_design` returns a p×p (p = selected.len()) row-major Vec with `+= sigma2` on the diagonal.
    - The trajectory branch uses `helpers::simpsons_weights(&model.argvals)` (no uniform weights) and the ridge-retry solve.
    - `cargo build -p fdars-core` succeeds (the crate still compiles; module is not yet re-exported from lib.rs).
  </acceptance_criteria>
  <done>The new file compiles as part of the crate; Σ_d assembly, validation, ridge-retry, and the Simpson-weighted trajectory integral are all implemented on the single proven path.</done>
  <reversibility rating="reversible">Additive new file, no published contract touched; deletable without affecting any other module.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Trajectory known-answer, grid-invariance, monotonicity, validation and ridge-retry tests</name>
  <files>fdars-core/src/optimal_design.rs</files>
  <read_first>
    - fdars-core/src/optimal_design.rs — the trajectory branch and validation from Task 1.
    - fdars-core/src/pace_fpca.rs — PaceFpcaResult field list, to construct a synthetic model in-test (it derives Clone; build the struct literal directly or via `..` from a helper).
    - fdars-core/src/matrix.rs — FdMatrix constructor to build the m×ncomp column-major eigenfunctions for the synthetic model.
    - .planning/phases/64-criterion-machinery-core/64-RESEARCH.md — "Known-Answer Test Architecture" section (synthetic orthonormal model: 2 eigenfunctions, λ=[2.0,1.0], σ²=0.5, MSE(∅)=3.0).
  </read_first>
  <behavior>
    - test_trajectory_empty_set: MSE(∅) == Σλ_k (2.0+1.0=3.0) within 1e-10 on a synthetic orthonormal model (m=51).
    - test_trajectory_grid_invariance: MSE(∅) equal within 1e-10 across grids m=21, 51, 101 (proves simpsons_weights, not uniform 1/m).
    - test_trajectory_reduces_on_point: MSE(&[25]) <= MSE(&[]) + 1e-12.
    - test_monotonicity_trajectory: MSE(&[10,30]) <= MSE(&[10]) + 1e-12.
    - test_validation_index_range: design_criterion(&model, &[m], Trajectory) is Err (InvalidParameter).
    - test_validation_sigma2: model with sigma2<=0 returns Err.
    - test_validation_ncomp: model with ncomp==0 returns Err.
    - test_ridge_retry: model with sigma2 = 1e-12 (near-singular) on Trajectory returns Ok (no panic).
  </behavior>
  <action>
Add an inline `#[cfg(test)] mod tests` block. Write a private test helper that constructs a synthetic `PaceFpcaResult` with exactly-orthonormal eigenfunctions under the grid's Simpson weights: build a uniform grid on [0,1] of length `m`, use closed-form functions normalized so `Σ_j w_j φ_k(t_j)² = 1` (e.g. scaled Fourier cosines; normalize each column by `sqrt(Σ_j w_j φ_k(t_j)²)` using `helpers::simpsons_weights`), set `eigenvalues = vec![2.0, 1.0]`, `sigma2 = 0.5`, `ncomp = 2`, and fill the remaining `PaceFpcaResult` fields (`mean`, `scores`, `fitted`, `fitted_lower`, `fitted_upper`) with valid-shape zero/placeholder FdMatrix values (they are unused by the trajectory branch). Parameterize the helper by `m` so grid-invariance can vary it.

Implement the eight tests named in `<behavior>` as exact known-answer assertions with the tolerances given (1e-10 for equalities, 1e-12 for monotonicity slack). For validation tests, construct degenerate models (sigma2=0.0, ncomp=0 with empty eigenvalues) and assert `.is_err()`. For test_ridge_retry, set `sigma2 = 1e-12` and assert `.is_ok()` on a small multi-point design (e.g. `&[10, 20, 30]`).

Run the RED step first if practical: the tests reference the already-implemented Task-1 code, so they should go GREEN immediately; if any fails, fix the trajectory branch (common cause: uniform weights, missing σ²I, or wrong Σ_d dimension revealing MSE(∅)=K·Σλ_k).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg optimal_design 2>&1 | tail -20</automated>
    <fails_when>Output contains "test result: FAILED", "0 passed", or any "FAILED" line — a trajectory known-answer, grid-invariance, monotonicity, validation, or ridge-retry gate did not hold.</fails_when>
  </verify>
  <acceptance_criteria>
    - All eight tests in `<behavior>` pass under `cargo test -p fdars-core --features linalg optimal_design`.
    - test_trajectory_empty_set and test_trajectory_grid_invariance together prove `MSE(∅) = Σλ_k` grid-invariantly.
    - test_ridge_retry passes with sigma2=1e-12 (no panic, returns Ok).
    - `cargo test` reports `test result: ok` with 0 failed for the optimal_design filter.
  </acceptance_criteria>
  <done>The trajectory criterion is proven end-to-end: prior recovery, grid-invariance, monotonicity, validation guards, and ridge-retry robustness all pass as automated known-answer tests.</done>
</task>

</tasks>

<security_note>
Numerical-robustness-only surface: no external API/SDK, no network, no untrusted input, no schema/DB. The only security-relevant controls are ASVS V5 input validation (index range, sigma2>0, ncomp>0 → InvalidParameter) and no-panic robustness (ridge-retry on near-singular Σ_d), both first-class acceptance criteria above. No `<threat_model>` STRIDE register required for this phase.
</security_note>

<verification>
- `cargo build -p fdars-core` — module compiles as part of the crate.
- `cargo test -p fdars-core --features linalg optimal_design` — all trajectory-branch known-answer gates green.
- Sanity: `grep -v '^\s*//' fdars-core/src/optimal_design.rs | grep -c 'simpsons_weights'` ≥ 1 (Simpson weights actually used, not commented).
</verification>

<success_criteria>
- New file `fdars-core/src/optimal_design.rs` exists and compiles within the crate.
- `DesignCriterion` and `OptimalityKind` enums declared with standard + serde-gated derives.
- `build_sigma_design` produces p×p Σ_d with σ²I diagonal; ridge-retry never panics.
- Trajectory branch: `MSE(∅) = Σλ_k` (grid-invariant), monotone non-increasing, validation guards enforced.
- All trajectory-path tests pass under the quick module test command.
</success_criteria>

<output>
Create `.planning/phases/64-criterion-machinery-core/64-01-SUMMARY.md` when done.
</output>
