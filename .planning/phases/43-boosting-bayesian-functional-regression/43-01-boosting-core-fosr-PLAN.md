---
phase: 43-boosting-bayesian-functional-regression
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/boosting_regression/mod.rs
  - fdars-core/src/boosting_regression/boost_fosr.rs
  - fdars-core/src/boosting_regression/boost_fofr.rs
  - fdars-core/src/boosting_regression/gamlss.rs
  - fdars-core/src/boosting_regression/bayesian.rs
  - fdars-core/src/boosting_regression/stability.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [REG-06-01]
estimate:
  tokens: 78000
  raw_tokens: 39000
  tasks: 3
  confidence: low
must_haves:
  truths:
    - "boost_fosr fits a function-on-scalar boosted model, selecting one base-learner per iteration (REG-06-01), reducing training RSS monotonically along the boosting path and recovering a known β(t) on synthetic data"
    - "The new boosting_regression module compiles with mod.rs declaring all five submodules and all config/result types, and the four sibling submodules (boost_fofr, gamlss, bayesian, stability) present as skeletons that later plans fill in"
    - "boost_fosr_one_step (internal helper) performs a single component-wise boosting iteration reusable by gamlss.rs"
    - "Public symbols are registered in src/lib.rs (pub mod + crate-root re-exports) and key result types in src/prelude.rs; existing public signatures are unchanged"
  artifacts:
    - fdars-core/src/boosting_regression/mod.rs
    - fdars-core/src/boosting_regression/boost_fosr.rs
    - fdars-core/src/boosting_regression/boost_fofr.rs
    - fdars-core/src/boosting_regression/gamlss.rs
    - fdars-core/src/boosting_regression/bayesian.rs
    - fdars-core/src/boosting_regression/stability.rs
  key_links:
    - "src/lib.rs `pub mod boosting_regression;` + crate-root re-export block wires the module into the crate API"
    - "boost_fosr base-learner fit calls bspline_basis (arbitrary predictor values) + cholesky_factor/cholesky_forward_back for the penalized normal-equation solve"
    - "boost_fosr_one_step is the shared boosting primitive gamlss.rs (Plan 03) and stability.rs (Plan 05) build on via boost_fosr"
---

<objective>
Deliver the boosting core and boosted FOSR estimator (REG-06-01), the foundation the rest of Phase 43 builds on. This plan creates the new `fdars-core/src/boosting_regression/` folder: `mod.rs` (module barrel + ALL config structs + ALL result structs + ALL submodule declarations + ALL barrel re-exports), the fully-implemented `boost_fosr.rs`, and minimal compiling skeletons for the four sibling submodules (`boost_fofr.rs`, `gamlss.rs`, `bayesian.rs`, `stability.rs`) so the crate compiles end-to-end while later plans fill each in. It also registers the module in `src/lib.rs` and `src/prelude.rs`.

Purpose: Component-wise gradient boosting with penalized functional base-learners for a function-on-scalar response is absent from fdars today; it is also the shared primitive GAMLSS (Plan 03) and stability selection (Plan 05) reuse. Locking it first de-risks the whole phase.
Output: New module `boosting_regression` with a working `boost_fosr()` (REG-06-01), shared `BoostingConfig`/`BayesianConfig`/`StabilityConfig` and all five result structs, plus crate-root + prelude re-exports.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-CONTEXT.md

Build/test note (MEMORY.md): prefix every cargo invocation with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` to avoid /tmp tmpfs exhaustion during doctest linking. Cholesky helpers (`cholesky_factor`, `cholesky_forward_back`, `cholesky_solve`, `compute_xtx`) are `pub(crate)` in `src/linalg.rs` — callable from the new in-crate module. `bspline_basis(t, nknots, order)` and `bspline_basis_from_knots(t, knots, order)` in `src/basis/bspline.rs` evaluate at arbitrary points `t` (verified) — use these to build the base-learner design matrix at scalar predictor values. `rand_distr = "0.4"` is a direct dependency (verified Cargo.toml:37).
</context>

<artifacts_produced>
New public symbols this plan introduces (drift verification MUST exclude these as newly-created):

Config structs (in `boosting_regression/mod.rs`, all `#[derive(Debug, Clone, PartialEq)]`):
- `BoostingConfig { mstop: usize, nu: f64, nbasis: usize, order: usize, lfd_order: usize, lambda: f64, ncomp_x: usize, seed: u64 }`
- `BayesianConfig { ncomp: usize, tau2: f64, ig_a0: f64, ig_b0: f64, n_iter: usize, burn_in: usize, thin: usize, seed: u64 }`
- `StabilityConfig { n_resamples: usize, pi_thr: f64, seed: u64 }`

Result structs (all `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]`):
- `BoostFosrResult` (fields per RESEARCH §Algorithm 1: `intercept`, `beta`, `fitted`, `residuals`, `r_squared_t`, `r_squared`, `mstop`, `nu`, `selected_learners`, `gcv_path`)
- `BoostFofrResult`, `GamlssResult`, `BayesianFosrResult`, `StabilityResult` (defined here in mod.rs per PATTERNS.md; field lists as in RESEARCH §Algorithms 2/3/4/5 — later plans consume them)

Public functions:
- `pub fn boost_fosr(data: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &BoostingConfig) -> Result<BoostFosrResult, FdarError>`

Internal (crate-visible) helper:
- `pub(crate) fn boost_fosr_one_step(...)` — a single component-wise boosting iteration (signature at implementer discretion; must let gamlss.rs run one μ-step and one σ-step). Document its contract in rustdoc.

Skeleton public functions (created here, IMPLEMENTED in later plans — return a valid result or an explicit `FdarError::ComputationFailed{ operation, detail: "not yet implemented (Plan NN)" }` placeholder so the crate compiles and clippy passes):
- `pub fn boost_fofr(...)` (Plan 02), `pub fn gamlss_fosr(...)` (Plan 03), `pub fn bayesian_fosr(...)` (Plan 04), `pub fn stability_selection(...)` (Plan 05)

Module paths: `crate::boosting_regression::{mod, boost_fosr, boost_fofr, gamlss, bayesian, stability}`
</artifacts_produced>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end boost_fosr slice — module skeleton + one working base-learner boosting path</name>
  <files>fdars-core/src/boosting_regression/mod.rs, fdars-core/src/boosting_regression/boost_fosr.rs, fdars-core/src/boosting_regression/boost_fofr.rs, fdars-core/src/boosting_regression/gamlss.rs, fdars-core/src/boosting_regression/bayesian.rs, fdars-core/src/boosting_regression/stability.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md (mod.rs section: gmm/mod.rs analog for barrel + config structs; boost_fosr.rs section: function_on_scalar.rs analog for penalized_solve + pointwise_r_squared + error handling; lib.rs/prelude.rs registration sections)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Algorithm 1 steps + BoostFosrResult fields + Integration Points exact re-export lists)
    - fdars-core/src/function_on_scalar.rs lines 25-55 (FosrResult derives + field convention), lines 121-168 (penalized_solve + pointwise_r_squared to copy/adapt)
    - fdars-core/src/gmm/mod.rs lines 1-20 (module barrel + submodule declaration pattern)
    - fdars-core/src/basis/bspline.rs lines 62-140 (bspline_basis / bspline_basis_from_knots — evaluate basis at arbitrary predictor values)
    - fdars-core/src/lib.rs lines 82-103 (module list; insert `pub mod boosting_regression;` after `pub mod regression;` line 103) and lines 445-458 (re-export block region — add boosting re-exports after the regression re-exports)
    - fdars-core/src/prelude.rs lines 14-19 (regression results block — add boosting results after it)
  </read_first>
  <action>
Create the `boosting_regression` folder with `mod.rs` and five sibling files, wire it into the crate, and implement ONE end-to-end boosting path so a recovery test passes.

1. `mod.rs`: module doc comment (per PATTERNS.md, cite Hothorn et al. 2010, Hofner et al. 2016, Jiang et al. 2025 and note divergences documented per-function). Declare `pub mod boost_fosr; pub mod boost_fofr; pub mod gamlss; pub mod bayesian; pub mod stability;` and `#[cfg(test)] mod tests;` only if you add mod-level config-validation tests. Define ALL THREE config structs (`BoostingConfig`, `BayesianConfig`, `StabilityConfig`) and ALL FIVE result structs (`BoostFosrResult`, `BoostFofrResult`, `GamlssResult`, `BayesianFosrResult`, `StabilityResult`) with exact fields from `<artifacts_produced>` / RESEARCH §Algorithms 1-5, each `#[derive(Debug, Clone, PartialEq)]`; result structs also `#[non_exhaustive]`. Add barrel re-exports: `pub use self::boost_fosr::boost_fosr; pub use self::boost_fofr::boost_fofr; pub use self::gamlss::gamlss_fosr; pub use self::bayesian::bayesian_fosr; pub use self::stability::stability_selection;`.

2. `boost_fosr.rs`: implement `boost_fosr(data, predictors, argvals, config)`. Validate inputs first (dimension: n>=3, m>0, predictors.nrows()==n → `FdarError::InvalidDimension`; parameters: mstop>=1, 0<nu<=1, nbasis>=4, lambda>0, order>=1 → `FdarError::InvalidParameter`) per the PATTERNS error-handling block. Algorithm 1: initialize F0(t)=pointwise column mean of Y; for each of p scalar predictor columns build the B-spline design matrix Φ_j (n×K) by evaluating `bspline_basis` at that column's values; form A_j = Φ_j'Φ_j + lambda·R_j + 1e-10·I where R_j = `bspline_penalty_matrix(...)`; `cholesky_factor(&A_j, k)` ONCE per learner (cache across the mstop loop since Φ_j is constant). Each boosting iteration: compute residual U = Y − F; for each learner j solve per response time point t via `cholesky_forward_back(&L_j, Φ_j'·U[:,t], k)` to get c_j(t) and fitted Ĥ_j (n×m_t); select j* = argmin Σ (U − Ĥ_j)²; update F += nu·Ĥ_{j*}, accumulate beta_{j*}(t) += nu·(reconstructed pointwise coefficient), push j* to `selected_learners`, push ‖U‖_F² to `gcv_path`. Use `data.column(t)` (contiguous) in the inner time loop per RESEARCH Pitfall 5. Assemble `BoostFosrResult` with `pointwise_r_squared` (copied/adapted from function_on_scalar.rs) and integrated `r_squared`. Mark `#[must_use = "expensive computation whose result should not be discarded"]`. Rustdoc must document the divergence from FDboost `bbs()` (fixed nu, fixed mstop, GCV-path tracking not early-stopping — cite CONTEXT decisions).

3. Extract the single-iteration body into `pub(crate) fn boost_fosr_one_step(...)` that `boost_fosr` calls in its loop, so gamlss.rs (Plan 03) can drive one μ-step / one σ-step. Document its inputs/outputs in rustdoc.

4. Skeleton the four sibling files (`boost_fofr.rs`, `gamlss.rs`, `bayesian.rs`, `stability.rs`): each with the exact public fn signature from `<artifacts_produced>` / PATTERNS.md, body returning `Err(FdarError::ComputationFailed { operation: "<name>", detail: "not yet implemented (Plan NN)".to_string() })`. Add `#[must_use]`. These MUST compile clean under `cargo clippy --all-targets`. Do NOT add tests to skeletons yet.

5. `src/lib.rs`: insert `pub mod boosting_regression;` after `pub mod regression;` (line 103). Insert the crate-root re-export block after the regression re-exports (line 452 region): `pub use boosting_regression::{ bayesian_fosr, boost_fofr, boost_fosr, gamlss_fosr, stability_selection, BayesianConfig, BayesianFosrResult, BoostFofrResult, BoostFosrResult, BoostingConfig, GamlssResult, StabilityConfig, StabilityResult, };`.

6. `src/prelude.rs`: after the regression results block (line 19) add `pub use crate::boosting_regression::{BayesianFosrResult, BoostFosrResult};`.

Do NOT place fenced code blocks inside this action — follow the signatures and equations in RESEARCH §Algorithm 1 and the copy/adapt targets named in read_first.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build -p fdars-core --features linalg,parallel` exits 0 (whole crate compiles with the new module wired in)
    - `grep -q "pub fn boost_fosr" fdars-core/src/boosting_regression/boost_fosr.rs` succeeds
    - `grep -q "pub mod boosting_regression;" fdars-core/src/lib.rs` succeeds
    - `grep -q "pub(crate) fn boost_fosr_one_step" fdars-core/src/boosting_regression/boost_fosr.rs` succeeds
    - The four skeleton files each contain their `pub fn` signature (`boost_fofr`, `gamlss_fosr`, `bayesian_fosr`, `stability_selection`)
  </acceptance_criteria>
  <done>The crate compiles with the boosting_regression module registered; boost_fosr is implemented end-to-end; the four sibling submodules exist as compiling skeletons; lib.rs and prelude.rs re-export the public symbols.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: boost_fosr recovery, monotonicity, and error-path tests</name>
  <files>fdars-core/src/boosting_regression/boost_fosr.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Validation Architecture → REG-06-01 test rows)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-VALIDATION.md (test oracles)
    - fdars-core/src/function_on_scalar.rs (inline `#[cfg(test)] mod tests` for shape + recovery assertions to mirror)
    - fdars-core/src/test_helpers.rs (`uniform_grid`)
  </read_first>
  <behavior>
    - RSS monotonicity: `gcv_path` (‖U‖_F² per iteration) is non-increasing across the mstop iterations for a signal-bearing synthetic dataset
    - Recovery: on synthetic Y_i(t) = x_i · beta_true(t) + small noise (single informative scalar predictor), fitted values track Y and integrated `r_squared` exceeds a threshold (e.g. > 0.8)
    - r_squared and every r_squared_t entry lie within a sane range (r_squared in [0,1] up to tiny numerical slack)
    - selected_learners has length == mstop and every entry is a valid predictor index (< p)
    - Error path: predictors.nrows() != data.nrows() returns `FdarError::InvalidDimension` (not a panic)
    - Error path: config with mstop==0 (or nu<=0, or lambda<=0, or nbasis<4) returns `FdarError::InvalidParameter`
  </behavior>
  <action>
Add an inline `#[cfg(test)] mod tests { use super::*; use crate::test_helpers::uniform_grid; ... }` block to `boost_fosr.rs`. Build a synthetic dataset with n>=20 curves on a uniform grid (m>=15), one informative scalar predictor driving a smooth beta_true(t) plus a couple of noise predictors, and small additive noise. Write tests covering every bullet in `<behavior>`: `boost_fosr_reduces_rss_monotonically`, `boost_fosr_recovers_known_beta`, `boost_fosr_r_squared_in_range`, `boost_fosr_selected_learners_valid`, `boost_fosr_errors_on_dimension_mismatch`, `boost_fosr_errors_on_invalid_params`. Use `assert!(matches!(err, FdarError::InvalidDimension { .. }))` style for error paths. Keep tolerances loose enough to be deterministic (fixed synthetic data, no RNG needed here).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::boost_fosr 2>&1 | tail -15</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test -p fdars-core --features linalg,parallel boosting_regression::boost_fosr` exits 0
    - Test output shows the six named tests running and passing (recovery, monotonicity, r_squared range, selected-learners validity, two error paths)
    - `grep -c "#\[test\]" fdars-core/src/boosting_regression/boost_fosr.rs` returns >= 6
  </acceptance_criteria>
  <done>REG-06-01 has inline recovery + monotonicity + range + error-path tests, all green.</done>
</task>

<task type="auto">
  <name>Task 3: Full-suite + clippy gate for the wired module</name>
  <files>fdars-core/src/boosting_regression/mod.rs</files>
  <read_first>
    - MEMORY.md pointer: CI clippy uses `--all-targets` (a plain `-p ... -D warnings` false-greens on test/bench code)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Project Constraints: #[must_use], #[non_exhaustive], derive boilerplate, crate-level clippy allows already present)
  </read_first>
  <action>
Run the full clippy gate and the whole test suite to confirm the new module (implemented boost_fosr + four skeletons + registration) does not regress anything and is warning-clean. Fix any clippy findings in the new files (e.g. missing `#[must_use]`, needless clones, unused imports in skeletons). If the skeleton files trip `dead_code` for unused params, prefix with `_` or add `#[allow(unused_variables)]` locally with a `// filled in Plan NN` note — do NOT suppress at module scope. Confirm config-struct validation is exercised: if you added mod-level tests, ensure they pass; otherwise this task is the gate only.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -8 && TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -8</automated>
  </verify>
  <acceptance_criteria>
    - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` exits 0 (no warnings in new or existing code)
    - `cargo test -p fdars-core --features linalg,parallel` exits 0 (full suite green; ~1654+ prior tests plus the new boost_fosr tests, no regressions)
  </acceptance_criteria>
  <done>The full CI clippy gate and whole test suite pass with the new module wired in.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure numeric in-process Rust library function set. No I/O, no network, no deserialization of untrusted data, no auth, no external attack surface. Inputs are in-memory `FdMatrix` / config structs supplied by the calling Rust code. |

## STRIDE Threat Register

Attack surface: NONE — pure numeric computation on in-memory `FdMatrix`. The only failure modes are numerical, handled as `FdarError` returns / numerical guards, not security threats.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-43-01 | Tampering (numerical) | `boost_fosr` penalized normal equations | low | mitigate | Ridge jitter `+1e-10·I` on `A_j` before `cholesky_factor`; `cholesky_factor` returns `FdarError::ComputationFailed` on non-positive diagonal (ill-conditioned base-learner) |
| T-43-02 | DoS (numerical) | `BoostingConfig` params | low | mitigate | Validate `mstop>=1`, `0<nu<=1`, `nbasis>=4`, `lambda>0`, `nbasis<=n` at entry → `FdarError::InvalidParameter`/`InvalidDimension` (no unbounded allocation / deep recursion) |

No package installs in this plan (no new crate dependency — `rand`/`rand_distr` already direct deps). No `T-43-SC` supply-chain checkpoint required.
</threat_model>

<verification>
Phase-level checks for this plan:
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --features linalg,parallel` compiles the whole crate with `boosting_regression` wired in.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::boost_fosr` — REG-06-01 recovery/monotonicity/error-path tests green.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.
- No existing public signature changed (additive-only): `git diff --stat` shows only additions to `src/lib.rs`/`src/prelude.rs` re-export/module regions plus the new `boosting_regression/` files.
</verification>

<success_criteria>
- REG-06-01 satisfied: `boost_fosr` fits a component-wise boosted FOSR selecting one base-learner per iteration, with inline recovery + error-path tests passing.
- The `boosting_regression` module compiles with all config/result types and five submodule files present (one implemented, four compiling skeletons).
- `boost_fosr_one_step` exists as the shared boosting primitive for Plan 03.
- Crate-root + prelude re-exports present; existing public signatures untouched.
- Full clippy gate + full test suite green.
</success_criteria>

<output>
Create `.planning/phases/43-boosting-bayesian-functional-regression/43-01-boosting-core-fosr-SUMMARY.md` when done.
</output>
