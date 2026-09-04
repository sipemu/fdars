---
phase: "67"
plan: "01"
type: execute
wave: 1
depends_on: ["66-01"]
files_modified: ["fdars-core/src/peer.rs"]
files_deleted: []
autonomous: true
requirements: ["PER-03"]
tags: ["rust", "regression", "smoothing", "gcv", "reml", "peer"]

estimate:
  tokens: 92000
  raw_tokens: 61000
  tasks: 4
  confidence: low

must_haves:
  truths:
    - "PER-03: GCV runs an automatic grid search and returns the GCV-minimizing λ recorded in PeerResult.lambda + PeerResult.gcv; two runs on the same data pick the same λ (bit-exact deterministic)."
    - "PER-03: REML fits λ via self-contained mixed-model EM (eigendecomposition of Q), returns a positive finite λ, and REML-vs-GCV β(t) agree within a documented tolerance on synthetic SNR data."
    - "PER-03: Explicit LambdaChoice::Fixed(λ) is used verbatim (no search runs) and appears unchanged in PeerResult.lambda with PeerResult.gcv == None and lambda_method == Fixed."
    - "PER-03: On synthetic SNR data both selectors pick a non-degenerate λ (neither ~0 nor over-smoothed flat β) and recover the known β(t) within tolerance."
  artifacts:
    - "fdars-core/src/peer.rs — LambdaChoice enum {Fixed(f64), Gcv, Reml} (Default = Gcv)"
    - "fdars-core/src/peer.rs — LambdaMethod enum {Fixed, Gcv, Reml}"
    - "fdars-core/src/peer.rs — PeerConfig.lambda field of type LambdaChoice (replaces f64)"
    - "fdars-core/src/peer.rs — PeerResult.gcv: Option<f64> + PeerResult.lambda_method: LambdaMethod"
    - "fdars-core/src/peer.rs — private select_lambda_gcv_peer + select_lambda_reml_peer + gcv_lambda_grid helpers"
  key_links:
    - "peer() dispatch on config.lambda → selected λ flows into the existing cholesky_solve of (WtW + λQ)."
    - "GCV inner loop reuses compute_peer_trace_hat (Phase 66) for tr(H)."
    - "REML eigendecomposition of Q via nalgebra symmetric_eigen partitions null (fixed) vs range (random) space."
---

<objective>
Add automatic smoothing-parameter (λ) selection to the Phase 66 `peer()` estimator: GCV grid search and a self-contained REML EM, selectable via a new `LambdaChoice` enum, with an explicit `Fixed(λ)` honored verbatim. All changes land in the single tracked file `fdars-core/src/peer.rs` (additive, non-breaking within the unpublished v0.36.0 milestone). Implements PER-03.

Purpose: Delivers PER-03 — matches refund's automatic λ selection so users no longer hand-pick λ, while keeping both selectors fully deterministic for reproducibility.

Output: Updated `PeerConfig`/`PeerResult`, two new selector enums, private GCV + REML helpers, `peer()` dispatch, mechanically-migrated Phase 66 tests, and new known-answer/determinism tests — all in `fdars-core/src/peer.rs`.
</objective>

<assumption_delta_decision>
This phase changes `PeerConfig.lambda: f64` → `lambda: LambdaChoice` — a derived→chosen / scalar→structured transition. Noun now primary: "λ as a selection choice (Fixed | Gcv | Reml)". Decision = **promote** (replace the raw `f64` field; update Phase 66 tests). Rationale: a single field both pins and selects; the crate is unpublished until the v0.36.0 tag (after Phase 68), so evolving the field is non-breaking. This is the LOCKED CONTEXT decision — recorded, not re-decided.
</assumption_delta_decision>

<reversibility rating="one-way">
The `LambdaChoice` enum + `PeerResult.gcv`/`lambda_method` field additions become the published-on-tag public contract. Rated one-way, but per REVERSIBILITY_GATES guidance NO blocking checkpoint is inserted: the change is additive within the milestone and this is an unattended run. Flagged for awareness only.
</reversibility>

<artifacts_this_phase_produces>
New/changed public symbols in `fdars-core/src/peer.rs`:
- `pub enum LambdaChoice { Fixed(f64), Gcv, Reml }` — `#[derive(Debug, Clone, PartialEq)]` + conditional serde; `impl Default` returns `Gcv`.
- `pub enum LambdaMethod { Fixed, Gcv, Reml }` — same derives + conditional serde (marker recording which path ran).
- `PeerConfig.lambda` field type changes `f64` → `LambdaChoice`; `PeerConfig::default()` now yields `LambdaChoice::Gcv`.
- `PeerResult` gains `pub gcv: Option<f64>` (GCV score at the selected λ; `None` when GCV did not run) and `pub lambda_method: LambdaMethod`. `PeerResult.lambda` continues to hold the value actually used.

New private helpers in `peer.rs`:
- `gcv_lambda_grid() -> Vec<f64>` — 40-point log-spaced grid on [1e-6, 1e4].
- `select_lambda_gcv_peer(...) -> (f64, f64)` — GCV grid argmin (best λ, GCV at best), ties → smaller grid index.
- `select_lambda_reml_peer(...) -> f64` — REML EM via `nalgebra::symmetric_eigen` of Q (null=fixed, range=random b~N(0,σ²_u I)), λ = σ²_e/σ²_u.

Phase 66 test constructions migrated from `lambda: <f64>` to `lambda: LambdaChoice::Fixed(<f64>)` (8 explicit `PeerConfig{...}` literals + the module-header doctest). `PeerConfig::default()` call sites keep compiling but now default to `Gcv`.
</artifacts_this_phase_produces>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/67-automatic-lambda-selection-gcv-reml/67-CONTEXT.md
@.planning/phases/67-automatic-lambda-selection-gcv-reml/67-RESEARCH.md
@fdars-core/src/peer.rs
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Tracer — evolve PeerConfig.lambda to LambdaChoice and wire Fixed(lambda) end-to-end</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/peer.rs:41-119` — PeerPenalty (derive + conditional-serde template to copy), PeerConfig (lambda: f64 to LambdaChoice), PeerResult (add two fields; note `#[non_exhaustive]` + `#[must_use]`).
    - `fdars-core/src/peer.rs:146-283` — peer() body; the lambda is read at line ~251 (`let lambda = config.lambda;`) and flows into `A = WtW + lambda*Q`, `cholesky_solve`, `compute_peer_trace_hat`, and the returned PeerResult.
    - `fdars-core/src/peer.rs:16-25` — module-header doctest constructing `PeerConfig { ..., lambda: 1.0 }`.
    - `fdars-core/src/peer.rs:347-773` — the entire `#[cfg(test)] mod tests` block: the 8 `PeerConfig { ..., lambda: <f64> }` literals to migrate (see RESEARCH "Tests that must be mechanically updated" for the exact line list) and the `PeerConfig::default()` call sites (no code change, but their default lambda path now becomes Gcv — see note below).
  </read_first>
  <action>
Add two public enums next to PeerPenalty, copying its exact derive attributes (`#[derive(Debug, Clone, PartialEq)]` plus `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`):
`LambdaChoice { Fixed(f64), Gcv, Reml }` with an `impl Default for LambdaChoice` returning `LambdaChoice::Gcv`; and `LambdaMethod { Fixed, Gcv, Reml }` (the which-path-ran marker). Doc-comment each variant.

Change `PeerConfig.lambda` from `f64` to `LambdaChoice`; update `impl Default for PeerConfig` to use `lambda: LambdaChoice::default()` (= Gcv). Update the field doc-comment to describe the choice semantics.

Add two fields to `PeerResult`: `pub gcv: Option<f64>` (GCV score at the selected lambda; `None` when GCV did not run) and `pub lambda_method: LambdaMethod` (which path ran). Doc-comment both. `PeerResult.lambda` keeps its meaning (value actually used). PeerResult is already `#[non_exhaustive]`.

In `peer()`, after `wtw`/`wty` are built and before the `A = WtW + lambda*Q` solve, introduce a dispatch that produces `(lambda, gcv, lambda_method)`. In THIS tracer task wire ONLY the `LambdaChoice::Fixed(lam)` arm: it maps to `(*lam, None, LambdaMethod::Fixed)`. For the `Gcv` and `Reml` arms, temporarily route them through the Fixed path using a placeholder lambda equal to the old default (`1.0`) so the code compiles and the tracer proves the API evolution end-to-end; leave a `Task 2/Task 3 wire selectors here` marker comment on those arms. Do NOT inline any grid/EM code in this task. Feed the resulting `lambda` into the existing solve/trace path unchanged, and populate the two new PeerResult fields from the dispatch.

Mechanically migrate every `PeerConfig { ..., lambda: <f64>, ... }` literal in the test module to `lambda: LambdaChoice::Fixed(<f64>)` (the 8 occurrences listed in RESEARCH API Evolution). Migrate the module-header doctest the same way (`lambda: LambdaChoice::Fixed(1.0)`). The `PeerConfig::default()` call sites (`test_peer_argvals_mismatch`, `test_peer_rejects_single_observation`, `test_peer_rejects_non_monotonic_argvals`, `test_peer_rejects_non_finite_y`) rely only on the validation error path which fires before selection, so the default-to-Gcv change does not affect them — verify each still asserts an `Err(...)` and not a numeric lambda.
  </action>
  <acceptance_criteria>
    - `LambdaChoice` and `LambdaMethod` enums exist with the specified variants, derives, conditional serde, and `Default = Gcv` for LambdaChoice.
    - `PeerConfig.lambda: LambdaChoice`; `PeerConfig::default()` yields `LambdaChoice::Gcv`.
    - `PeerResult` carries `gcv: Option<f64>` and `lambda_method: LambdaMethod`; a `Fixed` fit sets `gcv == None`, `lambda_method == LambdaMethod::Fixed`, and `lambda` equal to the supplied value bit-exact.
    - All 12 Phase 66 tests compile and pass after mechanical migration; the module-header doctest compiles.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer:: -- --nocapture</automated>
    <fails_when>non-zero exit, or output contains "test result: FAILED", or a compile error naming `lambda` / `LambdaChoice` / a `PeerResult` field mismatch</fails_when>
  </verify>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel --doc peer</automated>
    <fails_when>non-zero exit, or output contains "test result: FAILED" from the migrated module-header doctest</fails_when>
  </verify>
  <done>The API evolution compiles, the `Fixed(lambda)` path is wired verbatim through peer() end-to-end, and every Phase 66 test plus the module doctest is green with no selector logic yet added.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Wire the GCV grid-search selector</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/peer.rs:238-267` — how `wtw`, `wty`, `wc`, `q`, `m`, `n` are in scope at the dispatch point, and the existing `compute_peer_trace_hat(&wtw, &q, lambda, m, n)` call (reuse verbatim inside the GCV loop).
    - `fdars-core/src/peer.rs:324-340` — `compute_peer_trace_hat` clamps tr(H) to `n as f64`; the GCV denominator still needs a `<= 0` guard.
    - `fdars-core/src/function_on_scalar.rs:171-178` — `compute_fosr_gcv` (crate GCV convention to match); `:371-397` — `select_lambda_gcv` (grid-loop pattern, coarse 9-pt).
    - `fdars-core/src/linalg.rs:131-134` — `cholesky_solve(&a, &wty, m)` used per grid point.
  </read_first>
  <behavior>
    - Test: two `peer()` calls with `LambdaChoice::Gcv` on `make_fixture()` return bit-exact equal `result.lambda` (determinism; ties resolve to the smaller grid index).
    - Test: on `make_fixture()` the GCV lambda is non-degenerate — strictly greater than the grid-minimum sentinel and finite, and beta(t) recovered with max abs error below the Phase 66 tolerance (0.15).
    - Test: `result.gcv.is_some()` and `result.lambda_method == LambdaMethod::Gcv` when Gcv is selected.
  </behavior>
  <action>
Add a private `gcv_lambda_grid() -> Vec<f64>` returning 40 log-spaced points `10^x` for `x` in `linspace(-6, 4, 40)` (i.e. `-6.0 + 10.0 * i as f64 / 39.0`), documented as the fixed internal grid on [1e-6, 1e4] (not user-tunable this phase).

Add a private `select_lambda_gcv_peer(wc: &FdMatrix, yc: &[f64], wtw: &[f64], wty: &[f64], q: &[f64], m: usize, n: usize) -> (f64, f64)` returning `(best_lambda, gcv_at_best)`. For each grid lambda: build `A = wtw + lambda*q` (flat m*m), solve `beta = cholesky_solve(&a, wty, m)` (on `Err`, `continue` that grid point), compute `RSS = sum_i (yc[i] - sum_j wc[(i,j)]*beta[j])^2`, get `trh = compute_peer_trace_hat(wtw, q, lambda, m, n)`, set `denom = n as f64 - trh`; if `denom <= 0.0` skip that grid point; else `gcv = n as f64 * RSS / (denom*denom)`. Track the argmin with a strict `<` comparison initialized to `f64::INFINITY` so ties keep the earlier (smaller-lambda) grid index. Guard: if no grid point produced a finite score, fall back to the smallest grid lambda with its computed GCV (or `f64::INFINITY`). Document the tie-break and the tr(H) >= n guard in the fn doc-comment.

Replace the Task-1 placeholder `Gcv` arm in the `peer()` dispatch with a real call: `LambdaChoice::Gcv => { let (lam, g) = select_lambda_gcv_peer(&wc, &yc, &wtw, &wty, &q, m, n); (lam, Some(g), LambdaMethod::Gcv) }`. The selected `lam` then flows into the existing solve unchanged.

Add tests: `test_peer_gcv_deterministic` (two Gcv runs yield a bit-exact `result.lambda`), `test_peer_gcv_recovers_beta` (non-degenerate lambda: `result.lambda > 1e-10 && result.lambda < 1e6`, max abs beta error below 0.15, `result.gcv.is_some()`, `result.lambda_method == LambdaMethod::Gcv`). Reuse `make_fixture()`.
  </action>
  <acceptance_criteria>
    - `select_lambda_gcv_peer` iterates the fixed 40-point grid, computes `n*RSS/(n-tr(H))^2`, returns the argmin with documented tie-break, and guards `denom <= 0`.
    - `peer()` with `LambdaChoice::Gcv` records the selected lambda in `PeerResult.lambda`, the GCV score in `PeerResult.gcv` (Some), and `LambdaMethod::Gcv` in `lambda_method`.
    - Two Gcv runs on identical data yield bit-exact identical lambda; beta(t) recovered within 0.15 on the SNR fixture.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_gcv_deterministic peer::tests::test_peer_gcv_recovers_beta -- --nocapture</automated>
    <fails_when>non-zero exit, or output contains "test result: FAILED" (non-deterministic lambda, degenerate lambda, or beta recovery error >= 0.15)</fails_when>
  </verify>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer:: -- --nocapture</automated>
    <fails_when>non-zero exit, or output contains "test result: FAILED" from any regressed Phase 66 test</fails_when>
  </verify>
  <done>GCV runs an automatic deterministic grid search, records the argmin lambda plus GCV score in the result, and recovers beta(t) on the SNR fixture without regressing any Phase 66 test.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Wire the self-contained REML EM selector</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/fpca_variants.rs:473-488` — `nalgebra::DMatrix::symmetric_eigen()` usage plus the explicit eigenvalue sort pattern (PEER sorts ASCENDING so the null space comes first).
    - `fdars-core/src/linalg.rs:85-134` — `cholesky_factor`, `cholesky_forward_back`, `cholesky_solve` (used for the r*r E-step Sigma_b inverse and diagonal trace).
    - `fdars-core/src/peer.rs:238-267` — the dispatch point (`wc`, `yc`, `q`, `m`, `n` in scope) and the placeholder `Reml` arm from Task 1.
    - `fdars-core/src/famm.rs:395-499` — `reml_variance_update` plus `fit_scalar_mixed_model`: read to CONFIRM the subject-grouped structure that must NOT be called; port only the M-step variance-update math, not the function.
  </read_first>
  <behavior>
    - Test: two `peer()` calls with `LambdaChoice::Reml` on `make_fixture()` return bit-exact equal `result.lambda` (deterministic; fixed init, fixed iteration cap, no RNG).
    - Test: REML lambda on `make_fixture()` is positive and finite (`result.lambda > 0.0 && result.lambda.is_finite()`), `result.gcv.is_none()`, `result.lambda_method == LambdaMethod::Reml`.
    - Test: on a moderate-SNR fixture, REML and GCV beta(t) agree within a documented tolerance (max abs difference below 0.2), and REML recovers true beta within 0.15.
    - Test (edge): Ridge penalty (Q = I_m, null space empty, s = 0) yields a finite positive lambda with no panic; a zero-Q Decree (range space empty, r = 0) returns the documented fallback lambda.
  </behavior>
  <action>
Add `use nalgebra::DMatrix;` to the top of `peer.rs` if not already present.

Add a private `select_lambda_reml_peer(wc: &FdMatrix, yc: &[f64], q: &[f64], m: usize, n: usize) -> f64`:
1. Eigendecompose Q: `DMatrix::from_row_slice(m, m, q).symmetric_eigen()` (Q is symmetric so row-major equals column-major). Build `idx_sorted` ascending by eigenvalue; `max_ev = max abs(eigenvalue)`; `tol = 1e-8 * max_ev.max(1.0)`. Partition into `null_idx` (abs(ev) < tol, fixed/unpenalized, count `s`) and `range_idx` (abs(ev) >= tol, random effect, count `r`). Document the tolerance choice.
2. Edge — range-space empty (`r == 0`, zero-Q Decree): return a documented small fallback lambda (`1e-4`) — no random effect to estimate.
3. Build `Z_null` (n*s) and `Z_range` (n*r) as `W_c * V_null` and `W_c * V_range` using `eigen.eigenvectors[(j, ev_idx)]` (see RESEARCH "Form the projected designs").
4. Init deterministically: `y_var = sum((yc - ybar)^2) / (n-1).max(1)`; `sigma2_e = y_var.max(1e-12)`; `sigma2_u = (sigma2_e * 0.1).max(1e-12)`; `alpha = [0.0; s]` (OLS-init from Z_null when `s > 0` via `cholesky_solve` on `Z_null'Z_null`).
5. EM loop, cap 100 iterations. E-step: form `ZtZ_range` (r*r), `M = ZtZ_range/sigma2_e + (1/sigma2_u)*I_r`, invert via Cholesky to get `Sigma_b`; `b_hat = Sigma_b * Z_range' * r_alpha / sigma2_e` where `r_alpha = yc - Z_null*alpha`. M-step: `sigma2_u = (b_hat'b_hat + tr(Sigma_b)) / r`; `sigma2_e = (norm2(r_alpha - Z_range*b_hat) + tr(Z_range Sigma_b Z_range')) / n` (compute `tr(Z_range Sigma_b Z_range') = tr(Sigma_b ZtZ_range)`). When `s > 0`, re-estimate `alpha` via a GLS/Woodbury update (per RESEARCH "GLS update of alpha"; add a 1e-10 diagonal ridge for stability); when `s == 0` skip the alpha-update and use `r_alpha = yc`. After each M-step clamp `sigma2_u = sigma2_u.max(1e-12)`, `sigma2_e = sigma2_e.max(1e-12)`. Break when `abs(delta sigma2_u) + abs(delta sigma2_e) < 1e-8 * (sigma2_u + sigma2_e)`.
6. Return `(sigma2_e / sigma2_u).max(1e-15)`.

Guard all the RESEARCH edge cases: Ridge s=0 (skip GLS), zero-Q r=0 (fallback), sigma2_u toward 0 (clamp keeps lambda finite-large), non-negative variance components. NO RNG anywhere — determinism comes from the fixed init plus the fixed cap.

Replace the Task-1 placeholder `Reml` arm in the `peer()` dispatch: `LambdaChoice::Reml => { let lam = select_lambda_reml_peer(&wc, &yc, &q, m, n); (lam, None, LambdaMethod::Reml) }`.

Add tests: `test_peer_reml_deterministic` (two Reml runs yield bit-exact lambda), `test_peer_reml_lambda_positive` (lambda > 0 and finite, `gcv.is_none()`, `lambda_method == Reml`), `test_peer_reml_gcv_beta_agreement` (Gcv vs Reml beta agreement below 0.2 AND REML max abs beta error below 0.15 on a moderate-noise fixture), and `test_peer_reml_ridge_and_zeroq_edges` (Ridge yields finite positive lambda no panic; zero-Q Decree yields the fallback lambda). Reuse `make_fixture()`; add a moderate-SNR variant inline (noise 0.1) for the agreement test.
  </action>
  <acceptance_criteria>
    - `select_lambda_reml_peer` eigendecomposes Q (ascending sort), partitions null/range by tolerance, runs a fixed-init capped EM, returns `sigma2_e/sigma2_u > 0` finite.
    - Ridge (s=0) and zero-Q Decree (r=0) edge cases return finite positive / documented-fallback lambda with no panic and no NaN.
    - Two Reml runs on identical data yield bit-exact identical lambda; `gcv == None`; `lambda_method == LambdaMethod::Reml`.
    - REML and GCV beta(t) agree within 0.2 and REML recovers true beta within 0.15 on the moderate-SNR fixture.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_reml_deterministic peer::tests::test_peer_reml_lambda_positive peer::tests::test_peer_reml_gcv_beta_agreement peer::tests::test_peer_reml_ridge_and_zeroq_edges -- --nocapture</automated>
    <fails_when>non-zero exit, or output contains "test result: FAILED" (non-deterministic lambda, non-positive/non-finite lambda, beta disagreement >= 0.2, or a panic in an edge case)</fails_when>
  </verify>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer:: -- --nocapture</automated>
    <fails_when>non-zero exit, or output contains "test result: FAILED" from any regressed Phase 66 or Task 2 test</fails_when>
  </verify>
  <done>REML fits lambda via a self-contained deterministic EM over the eigendecomposition of Q, returns a positive finite lambda recorded with LambdaMethod::Reml, agrees with GCV on beta within tolerance, and never panics on the Ridge/zero-Q edges.</done>
</task>

<task type="auto">
  <name>Task 4: Phase gate — full suite, clippy, fmt</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/peer.rs` — the complete post-Task-3 module (enums, helpers, dispatch, all tests) to confirm no leftover placeholder arms, no unused imports, and no `f64` lambda literals remain in tests.
  </read_first>
  <action>
Run the full crate test suite and the CI-parity lint/format gates, and fix any warnings or drift surfaced (per MEMORY: CI clippy uses `--all-targets`; `--no-verify` commits leave fmt drift, so run `cargo fmt` explicitly). Confirm both `Gcv` and `Reml` dispatch arms call their real selectors (no placeholder `1.0` remaining), that the `Fixed` arm is unchanged, and that the module doctest still reflects the migrated `LambdaChoice::Fixed(...)` construction. No new behavior — this task only proves the whole file is green under CI-equivalent gates and resolves any lint/format issues.
  </action>
  <acceptance_criteria>
    - Full `fdars-core` test suite passes (lib + doc) with the `linalg,parallel` features.
    - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` is clean.
    - `cargo fmt -- --check` reports no drift.
    - No placeholder selector arm remains; the Fixed/Gcv/Reml dispatch all route to their intended paths.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel && cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo fmt -- --check</automated>
    <fails_when>non-zero exit, or output contains "test result: FAILED", "warning:" escalated to error by `-D warnings`, or a `Diff in` line from `cargo fmt -- --check`</fails_when>
  </verify>
  <done>The whole crate builds, tests, lints (all-targets), and format-checks clean under CI-equivalent gates with the λ-selection feature fully wired.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → `peer()` | Untrusted numeric inputs (`data`, `y`, `argvals`, `config.penalty` Q, `config.lambda`) cross into the estimator. This is the only boundary — pure in-process library, no I/O, network, or auth. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-67-01 | Tampering | `peer()` numeric inputs (NaN/Inf y, non-monotone argvals, wrong-dim Q) | low | mitigate | Existing Phase 66 entry validation returns `FdarError::InvalidDimension`/`InvalidParameter` before any selection runs; unchanged by this phase. |
| T-67-02 | Denial of Service | `select_lambda_reml_peer` EM loop / degenerate variance components | low | mitigate | Fixed iteration cap (100) plus σ² clamps (1e-12) bound runtime and prevent divide-by-zero / λ→∞ blowups; r=0 fallback returns immediately. |
| T-67-03 | Denial of Service | `select_lambda_gcv_peer` grid loop with tr(H) ≥ n | low | mitigate | `denom <= 0.0` guard skips degenerate grid points; fixed 40-point grid bounds the loop. |
| T-67-04 | Tampering | non-finite λ / β leaking into the returned result | low | mitigate | Existing post-solve β NaN guard (`FdarError::ComputationFailed`) plus `.max(1e-15)` on the REML λ keep the returned λ finite and positive. |

No high-severity threats — pure numerical computation on already-validated inputs (ASVS L1). No new external dependency, no package-install task, so `T-{phase}-SC` is not applicable.
</threat_model>

<verification>
- Per task: `cargo test -p fdars-core --features linalg,parallel peer:: -- --nocapture` green (module tests, fast).
- Determinism: `test_peer_gcv_deterministic` and `test_peer_reml_deterministic` assert bit-exact λ across two runs.
- Known-answer: `test_peer_gcv_recovers_beta`, `test_peer_reml_lambda_positive`, `test_peer_reml_gcv_beta_agreement` cover the four ROADMAP success criteria.
- Phase gate (Task 4): full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt -- --check` all clean.
</verification>

<success_criteria>
- GCV runs an automatic grid search, records the argmin λ + GCV score, and is bit-exact deterministic across runs.
- REML fits λ via the self-contained eigendecomposition-based EM, returns a positive finite λ, and agrees with GCV on β within the documented tolerance.
- `LambdaChoice::Fixed(λ)` is used verbatim with `gcv == None` and `lambda_method == Fixed`.
- Both selectors pick a non-degenerate λ and recover the known β(t) within tolerance on the SNR fixture.
- All Phase 66 tests remain green after the mechanical `LambdaChoice::Fixed` migration; CI-parity lint/format gates pass.
</success_criteria>

<output>
Create `.planning/phases/67-automatic-lambda-selection-gcv-reml/67-01-SUMMARY.md` when done.
</output>
