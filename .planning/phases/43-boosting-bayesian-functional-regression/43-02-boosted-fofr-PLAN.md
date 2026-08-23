---
phase: 43-boosting-bayesian-functional-regression
plan: 02
type: execute
wave: 2
depends_on: [43-01]
files_modified:
  - fdars-core/src/boosting_regression/boost_fofr.rs
autonomous: true
requirements: [REG-06-02]
estimate:
  tokens: 55000
  raw_tokens: 27500
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "boost_fofr fits component-wise gradient boosting for a function-on-function predictor/response (REG-06-02) through the same boosting framework, compressing each functional predictor to FPC scores and selecting one base-learner per iteration"
    - "boost_fofr fitted values have shape (n, m_y) and residuals decrease over the boosting path; a reconstructed coefficient surface β_j(s,t) of shape (m_x, m_y) is available per predictor"
    - "boost_fofr returns FdarError on dimension mismatch / invalid parameters rather than panicking"
  artifacts:
    - fdars-core/src/boosting_regression/boost_fofr.rs
  key_links:
    - "boost_fofr calls fdata_to_pc_1d to compress each functional predictor into an (n × K_j) score design matrix that replaces the B-spline design of boost_fosr"
    - "boost_fofr reuses BoostingConfig + BoostFofrResult from Plan 01's mod.rs (no mod.rs edits — collision-free)"
---

<objective>
Implement boosted function-on-function regression (REG-06-02): replace the boosted-FOSR scalar-predictor B-spline design with FPC-score signal compression of each functional predictor, run the same component-wise boosting loop, and reconstruct coefficient surfaces β_j(s,t). This fills the `boost_fofr.rs` skeleton created in Plan 01.

Purpose: Extends the boosting framework to function-on-function predictors/response (the bfpc / signal-compression variant), a capability fdars lacks. Reuses the boosting machinery and FPCA already in the crate — no new numerical primitives.
Output: A working `boost_fofr()` returning `BoostFofrResult`, with inline recovery + shape + error-path tests.
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
@.planning/phases/43-boosting-bayesian-functional-regression/43-01-boosting-core-fosr-SUMMARY.md

Depends on Plan 01: `BoostingConfig`, `BoostFofrResult`, and the boosting loop pattern in `boost_fosr.rs`. `boost_fofr.rs` already exists as a compiling skeleton (returns `FdarError::ComputationFailed{ detail: "not yet implemented (Plan 02)" }`) — this plan replaces that body. Prefix cargo with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`.
</context>

<artifacts_produced>
New public symbol implemented (signature already declared in Plan 01; drift verification excludes it):
- `pub fn boost_fofr(x_data: &[&FdMatrix], x_argvals: &[&[f64]], y_data: &FdMatrix, y_argvals: &[f64], config: &BoostingConfig) -> Result<BoostFofrResult, FdarError>`

`BoostFofrResult` (defined in Plan 01 mod.rs) fields consumed/populated: `intercept`, `fitted`, `residuals`, `r_squared_t`, `r_squared`, `fpca_x: Vec<FpcaResult>`, `score_coefs: Vec<FdMatrix>`, `beta_surfaces: Vec<FdMatrix>`, `selected_learners`, `gcv_path`, `mstop`, `nu`.

Private helper (crate-internal, this file): score-space per-time-point OLS solve reusing `compute_xtx` / `cholesky_factor` / `cholesky_forward_back` (or the simpler `S_j'S_j` normal equations from RESEARCH §Algorithm 2 step 2).
</artifacts_produced>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end boost_fofr — FPC-score signal compression + boosting loop</name>
  <files>fdars-core/src/boosting_regression/boost_fofr.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Algorithm 2: model, FPC-compression steps, coefficient-surface reconstruction, and the documented divergence from FDboost bsignal)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md (boost_fofr.rs section: fof_regression.rs analog, double-FPCA precomputation, BoostFofrResult shape, function signature)
    - fdars-core/src/fof_regression.rs lines 55-160 (double-FPCA → regress scores → reconstruct β(s,t) pattern)
    - fdars-core/src/boosting_regression/boost_fosr.rs (Plan 01 — the boosting loop + selection + update pattern to mirror in score space)
    - fdars-core/src/regression.rs lines 25-38, 287-292 (FpcaResult fields + fdata_to_pc_1d signature; rotation is m_x × K, scores are n × K)
  </read_first>
  <action>
Replace the `boost_fofr.rs` skeleton body with the full implementation per RESEARCH §Algorithm 2.

Validate inputs first: `x_data` non-empty; every predictor and the response share the same n (== y_data.nrows()); each `x_argvals[j].len()` matches that predictor's m_x; `y_argvals.len()` == y_data.ncols(); config params (mstop>=1, 0<nu<=1, ncomp_x>=1) — else `FdarError::InvalidDimension`/`InvalidParameter`.

Preprocessing: for each functional predictor j, `let fpca_j = fdata_to_pc_1d(x_data[j], config.ncomp_x, x_argvals[j])?;` and use `fpca_j.scores` (n × K_j) as that base-learner's design matrix S_j. Collect `fpca_x: Vec<FpcaResult>`.

Boosting loop (mirror boost_fosr from Plan 01, but in score space): init F0(t) = pointwise column mean of Y (m_y-vector); residual U = Y − F. Per iteration, for each learner j solve the score-space normal equations per response time point t — `(S_j'S_j + 1e-10·I) c_j(t) = S_j'·U[:,t]` via `cholesky_factor` (factored once per learner, cached) + `cholesky_forward_back` — giving fitted Ĥ_j = S_j·c_j (n × m_y). Select j* = argmin Σ(U − Ĥ_j)²; update F += nu·Ĥ_{j*}; accumulate `score_coefs[j*]` (K_{j*} × m_y) += nu·c_{j*}; push j* + ‖U‖_F².

Coefficient-surface reconstruction: for each predictor with any selection, β_j(s,t) = rotation_j (m_x × K_j) · score_coefs_j (K_j × m_y) = (m_x × m_y) matrix → `beta_surfaces[j]`. For never-selected predictors, produce a zero surface of the correct shape.

Assemble `BoostFofrResult` with `fitted`, `residuals = Y − fitted`, pointwise + integrated R², `fpca_x`, `score_coefs`, `beta_surfaces`, `selected_learners`, `gcv_path`, `mstop`, `nu`. Mark `#[must_use]`. Rustdoc must document the divergence from FDboost `bsignal` (bfpc truncated-KL compression instead of joint B-spline β(s,t) with trapezoidal integration) per RESEARCH §Algorithm 2.

Do NOT inline fenced code — follow RESEARCH §Algorithm 2 equations and the fof_regression.rs reconstruction pattern named in read_first.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build -p fdars-core --features linalg,parallel` exits 0
    - `boost_fofr.rs` no longer contains the string `not yet implemented`
    - `grep -q "fdata_to_pc_1d" fdars-core/src/boosting_regression/boost_fofr.rs` succeeds (FPC compression wired in)
    - `grep -q "beta_surfaces" fdars-core/src/boosting_regression/boost_fofr.rs` succeeds (coefficient-surface reconstruction present)
  </acceptance_criteria>
  <done>boost_fofr is implemented end-to-end: FPC-compressed base-learners, boosting loop, reconstructed coefficient surfaces; the crate compiles.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: boost_fofr shape, monotonicity, and error-path tests + gate</name>
  <files>fdars-core/src/boosting_regression/boost_fofr.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Validation Architecture → REG-06-02 rows: fitted shape == (n, m_y); residuals decrease over iterations)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-VALIDATION.md (test oracles for boosted FoFR)
    - fdars-core/src/fof_regression.rs inline tests (shape + recovery assertion style to mirror)
    - fdars-core/src/test_helpers.rs (`uniform_grid`)
  </read_first>
  <behavior>
    - fitted.shape() == (n, m_y) and residuals.shape() == (n, m_y)
    - gcv_path (‖U‖_F² per iteration) is non-increasing across iterations for a signal-bearing synthetic dataset
    - r_squared lies within [0,1] (up to tiny numerical slack) on a synthetic dataset where the response is generated from a known functional predictor
    - beta_surfaces has one entry per predictor, each of shape (m_x, m_y)
    - Error path: predictor/response row-count mismatch returns FdarError::InvalidDimension
    - Error path: ncomp_x == 0 (or mstop == 0) returns FdarError::InvalidParameter
  </behavior>
  <action>
Add an inline `#[cfg(test)] mod tests` block. Build a synthetic function-on-function dataset on uniform grids (n>=20, m_x>=15, m_y>=15) where Y is generated by integrating a known predictor against a smooth β(s,t) plus small noise, and 1-2 noise predictors. Write tests for every `<behavior>` bullet: `boost_fofr_fitted_shape`, `boost_fofr_residuals_decrease`, `boost_fofr_r_squared_in_range`, `boost_fofr_beta_surface_shape`, `boost_fofr_errors_on_dimension_mismatch`, `boost_fofr_errors_on_invalid_params`. Then run the full clippy gate + full test suite and fix any findings in this file.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::boost_fofr 2>&1 | tail -12 && TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -6</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test -p fdars-core --features linalg,parallel boosting_regression::boost_fofr` exits 0
    - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` exits 0
    - `grep -c "#\[test\]" fdars-core/src/boosting_regression/boost_fofr.rs` returns >= 5
  </acceptance_criteria>
  <done>REG-06-02 has inline shape + monotonicity + surface-shape + error-path tests, all green; clippy gate clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure numeric in-process Rust library function. No I/O, network, deserialization, auth, or external attack surface — inputs are in-memory `FdMatrix` slices + config supplied by calling Rust code. |

## STRIDE Threat Register

Attack surface: NONE — pure numeric computation on in-memory `FdMatrix`. Only failure modes are numerical, handled as `FdarError` returns.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-43-02a | Tampering (numerical) | score-space normal equations | low | mitigate | Ridge jitter `+1e-10·I` before `cholesky_factor`; `FpcaResult`/Cholesky failures propagate as `FdarError::ComputationFailed` |
| T-43-02b | DoS (numerical) | `ncomp_x` / `mstop` params | low | mitigate | Validate `ncomp_x>=1`, `mstop>=1`, per-predictor dimension agreement at entry → `FdarError` (no unbounded work) |

No package installs. No supply-chain checkpoint required.
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::boost_fofr` green (REG-06-02).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.
- Only `boost_fofr.rs` modified (no mod.rs edit — collision-free with sibling wave-2 plans).
</verification>

<success_criteria>
- REG-06-02 satisfied: `boost_fofr` fits a boosted function-on-function model via FPC-compressed base-learners through the shared boosting framework, with fitted shape (n, m_y), decreasing residuals, reconstructed coefficient surfaces, and error-path handling.
- Inline tests green; full clippy gate clean.
</success_criteria>

<output>
Create `.planning/phases/43-boosting-bayesian-functional-regression/43-02-boosted-fofr-SUMMARY.md` when done.
</output>
