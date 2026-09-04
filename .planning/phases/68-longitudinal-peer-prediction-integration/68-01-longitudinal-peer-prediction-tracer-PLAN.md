---
phase: 68-longitudinal-peer-prediction-integration
plan: 01
type: execute
wave: 1
depends_on: [67-01]
files_modified:
  - fdars-core/src/peer.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [PER-04, PER-05]

estimate:
  tokens: 78000
  raw_tokens: 52000
  tasks: 4
  confidence: med

must_haves:
  truths:
    - "lpeer() fits subject-level random effects (via famm::fit_scalar_mixed_model) and returns LpeerResult carrying time-varying beta(t) + sigma2_subject/sigma2_resid; both variance components non-negative (PER-04)."
    - "On synthetic longitudinal data with a known injected between-subject variance and known beta(t), lpeer() recovers beta(t) within tolerance and fitted sigma2_subject tracks the injected variance (PER-04)."
    - "predict on re-passed training curves reproduces training fitted_values within 1e-9 (self-consistency, via stored w_bar) and is finite on genuinely new curves; wrong ncols returns FdarError (PER-05)."
    - "Full PEER/lpeer public surface reachable from crate root + prelude; a running module doctest demonstrates fit -> beta(t) -> predict and passes cargo test --doc (PER-05)."
  artifacts:
    - "fdars-core/src/peer.rs — lpeer() fn, LpeerResult struct, peer_predict_core() helper, PeerResult::predict + LpeerResult::predict methods, running module doctest"
    - "fdars-core/src/lib.rs — pub use peer::{...} crate-root re-export block"
    - "fdars-core/src/prelude.rs — pub use crate::peer::{...} prelude re-export block"
  key_links:
    - "LpeerResult.w_bar computed identically to peer()'s w_bar so peer_predict_core reproduces fitted_values for both result types."
    - "ScalarMixedResult.sigma2_u -> LpeerResult.sigma2_subject; sigma2_eps -> sigma2_resid (famm clamps both non-negative)."
    - "Back-projection beta[j] = Sum_k gamma[k]*fpca.rotation[(j,k)] links famm fixed effects to beta(t) on the argvals grid."
---

<objective>
Close the v0.36.0 PEER milestone by adding, all in `fdars-core/src/peer.rs` plus one-line
re-exports in `lib.rs`/`prelude.rs`: (1) out-of-sample `predict` on the fitted `PeerResult`
(the tracer — smallest end-to-end slice proving the prediction contract, unblocks the
doctest), (2) the longitudinal `lpeer()` estimator with subject-level random effects via
`famm::fit_scalar_mixed_model` returning `LpeerResult` with variance components (PER-04), and
(3) crate-root + prelude re-exports plus a running module doctest (PER-05).

Purpose: PER-04 (longitudinal PEER) and PER-05 (prediction + integration) are the last two
requirements of the milestone; this phase makes the whole PEER surface usable and reachable.
Output: `lpeer` fn, `LpeerResult` struct, `peer_predict_core` helper, `predict` methods on
both result types, crate-root/prelude export blocks, and a replaced running doctest.

Assumption-delta: primary noun = "PEER estimator family (cross-sectional `peer` +
longitudinal `lpeer`)". Decision = **add-alongside** — `lpeer` is a distinct estimator for
repeated-measures data, NOT a replacement; `peer` stays first-class for non-longitudinal data.

API coverage: see `COVERAGE.md` — no external API integration.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/68-longitudinal-peer-prediction-integration/68-RESEARCH.md
@fdars-core/src/peer.rs
</context>

<artifacts_produced>
## Artifacts this phase produces (new symbols)

- `pub fn lpeer(data, y, argvals, subject_map, &config) -> Result<LpeerResult, FdarError>` — new public estimator in `peer.rs`.
- `pub struct LpeerResult` — new result struct with fields: `beta`, `intercept`, `w_bar`, `fitted_values`, `sigma2_subject: f64`, `sigma2_resid: f64`, `n_subjects: usize`, `lambda: f64`, `penalty_type: PeerPenalty`, `gcv: Option<f64>`, `lambda_method: LambdaMethod`. Derives `Debug, Clone, PartialEq`; `#[non_exhaustive]`; `#[must_use]`; conditional serde.
- `fn peer_predict_core(beta, intercept, w_bar, new_data, argvals) -> Result<Vec<f64>, FdarError>` — private module-level helper shared by both predict methods.
- `impl PeerResult { pub fn predict(&self, new_data, argvals) -> Result<Vec<f64>, FdarError> }` and `impl LpeerResult { pub fn predict(&self, new_data, argvals) -> Result<Vec<f64>, FdarError> }`.
- Crate-root `pub use peer::{lpeer, peer, LambdaChoice, LambdaMethod, LpeerResult, PeerConfig, PeerPenalty, PeerResult}` block in `lib.rs`.
- Prelude `pub use crate::peer::{...}` block (same set) in `prelude.rs`.
- Replaced running module doctest in the `peer.rs` `//!` header (was `no_run`), plus an `lpeer` `///` doctest.
</artifacts_produced>

<tasks>

<task type="tracer">
  <name>Task 1: Tracer — predict method + shared peer_predict_core, wired end-to-end on PeerResult</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/peer.rs` lines 116-153 (`PeerResult` fields, esp. `intercept`, `w_bar`, `beta`, `fitted_values`).
    - `fdars-core/src/peer.rs` lines 1121-1154 (`test_peer_stores_w_bar_for_prediction` — this test already exercises the exact predict identity by hand; the method codifies it).
    - RESEARCH §4 (predict Design — shared helper `peer_predict_core`, validation, self-consistency).
    - `fdars-core/src/scalar_on_function/mod.rs:668-673` (predict-method-on-result-struct convention).
  </read_first>
  <action>
    Add a private module-level fn `peer_predict_core(beta: &[f64], intercept: f64, w_bar: &[f64], new_data: &FdMatrix, argvals: &[f64]) -> Result<Vec<f64>, FdarError>`. Let `m = beta.len()`, `(n_new, m_new) = new_data.shape()`. Validate `m_new == m` else return `FdarError::InvalidDimension { parameter: "new_data", expected: format of m columns, actual: format of m_new }`; validate `argvals.len() == m` else `InvalidDimension { parameter: "argvals", ... }`. Compute `w = simpsons_weights(argvals)`, `base = intercept - w_bar.iter().zip(beta).map(|(wb,b)| wb*b).sum()`, then for each new row `i`: `base + sum_j new_data[(i,j)] * w[j] * beta[j]`. Return `Ok(preds)`. This is the exact identity in test_peer_stores_w_bar_for_prediction (per D — CONTEXT.md predict formula `y* = (intercept - w_bar·beta) + Sum_j x*[j]·w[j]·beta[j]`).

    Add `impl PeerResult { pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Result<Vec<f64>, FdarError> { peer_predict_core(&self.beta, self.intercept, &self.w_bar, new_data, argvals) } }` with a doc comment noting out-of-sample prediction and self-consistency on re-passed training curves.

    Add inline tests: `test_peer_predict_self_consistent` (fit via `make_fixture` + Difference{2} Fixed(1e-4), call `result.predict(&data, &t)`, assert each pred within 1e-9 of the corresponding `fitted_values`); `test_predict_wrong_ncols` (build `new_data` with a different column count via a truncated/padded FdMatrix, assert `predict` returns `Err(FdarError::InvalidDimension { .. })`); `test_predict_new_curves_finite` (build fresh curves with `hash_unit` offset by a large constant so they differ from training, assert `predict(...).unwrap().iter().all(|v| v.is_finite())`).
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_predict_self_consistent peer::tests::test_predict_wrong_ncols peer::tests::test_predict_new_curves_finite -- --nocapture</automated>
    <fails_when>the predict method does not reproduce fitted_values within 1e-9 (self-consistency assert trips), or a wrong-ncols new_data does not yield InvalidDimension, or predictions on fresh curves are non-finite.</fails_when>
  </verify>
  <acceptance_criteria>
    - `peer_predict_core` exists and is shared (ready for LpeerResult in Task 2).
    - `PeerResult::predict` reproduces training `fitted_values` within 1e-9 on re-passed training curves.
    - Wrong-ncols `new_data` returns `FdarError::InvalidDimension`; fresh curves yield all-finite predictions.
  </acceptance_criteria>
  <done>The end-to-end predict path (result -> peer_predict_core -> predictions) works for PeerResult, is self-consistent, validates dims, and is committed. This tracer proves the prediction contract the doctest depends on.</done>
  <reversibility rating="one-way">`PeerResult::predict(&self, new_data, argvals)` signature is published on the v0.36.0 tag right after this phase; changing it later is a breaking change. Unattended run — do NOT insert a checkpoint.</reversibility>
</task>

<task type="auto">
  <name>Task 2: lpeer() estimator + LpeerResult + LpeerResult::predict + longitudinal fixture tests</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - RESEARCH §1 (famm exact contract — `fit_scalar_mixed_model(y, subject_map, n_subjects, Some(covariates), p)`, `ScalarMixedResult { gamma, u_hat, sigma2_u, sigma2_eps }`, variance clamping, `build_subject_map`).
    - RESEARCH §2 (fof_regression famm template — pass raw FPC scores WITHOUT h.sqrt rescaling; back-projection pattern).
    - RESEARCH §3 (lpeer Design — full algorithm; ncomp = min(n-1, m, 10); pass `yc` centered; intercept = y_bar).
    - RESEARCH §6 (synthetic longitudinal fixture with injected between-subject variance; recovery + tracking tolerances).
    - RESEARCH §8 (Pitfalls 1-6: no h.sqrt rescale, pass yc not y, build_subject_map, rotation layout, w_bar must match peer(), non-singular design).
    - `fdars-core/src/peer.rs` lines 242-330 (peer() weights/centering/w_bar/build_q/λ-dispatch — mirror exactly so predict is identical).
    - `fdars-core/src/peer.rs` lines 773-814 (`hash_unit` + `make_fixture` — reuse the spanning full-rank design pattern for the longitudinal fixture).
  </read_first>
  <action>
    Add `pub struct LpeerResult` with fields `beta: Vec<f64>`, `intercept: f64`, `w_bar: Vec<f64>`, `fitted_values: Vec<f64>`, `sigma2_subject: f64`, `sigma2_resid: f64`, `n_subjects: usize`, `lambda: f64`, `penalty_type: PeerPenalty`, `gcv: Option<f64>`, `lambda_method: LambdaMethod`. Derive `Debug, Clone, PartialEq`; add `#[non_exhaustive]`, conditional serde `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`, and `#[must_use = "expensive computation whose result should not be discarded"]`. Doc each field (mirror PeerResult doc style; note `sigma2_subject`/`sigma2_resid` are non-negative via famm clamping). This is a distinct estimator's result, add-alongside PeerResult (per D — CONTEXT.md lpeer API & result).

    Add `pub fn lpeer(data: &FdMatrix, y: &[f64], argvals: &[f64], subject_map: &[usize], config: &PeerConfig) -> Result<LpeerResult, FdarError>` following RESEARCH §3/§9 (per D — CONTEXT.md lpeer signature + famm integration). Steps: (a) `(n, m) = data.shape()`; validate `n >= 2`, `m >= 3`, `argvals.len() == m`, `y.len() == n`, `subject_map.len() == n` (each -> descriptive `FdarError::InvalidDimension`); reuse peer()'s finite/monotone guards on `y` and `argvals`. (b) `(sm_dense, n_subjects) = crate::famm::build_subject_map(subject_map)`; if `n_subjects < 2` return `FdarError::InvalidParameter { parameter: "subject_map", message: at least 2 distinct subjects required }` (per D — n_subjects derived from subject_map). (c) Compute `w = simpsons_weights(argvals)`, weighted `wmat[(i,j)] = data[(i,j)]*w[j]`, `y_bar`, `yc = y - y_bar`, and `w_bar[j] = col-mean of wmat` EXACTLY as peer() does (Pitfall 5 — w_bar must match so predict reproduces fitted values). (d) `let ncomp = (n-1).min(m).min(10);` document the cap; `let fpca = crate::regression::fdata_to_pc_1d(data, ncomp, argvals)?; let scores = &fpca.scores;` (raw FPC scores, NO h.sqrt rescale — Pitfall 1). (e) λ dispatch identical to peer() (`build_q(m, &config.penalty)?`, build centered `wc`/`wtw`/`wty`, then match `config.lambda` over Fixed/Gcv/Reml using `select_lambda_gcv_peer`/`select_lambda_reml_peer`) so the structured penalty + λ choice carry through (per D — carry PeerPenalty/λ into the estimate). (f) `let result = crate::famm::fit_scalar_mixed_model(&yc, &sm_dense, n_subjects, Some(scores), ncomp);` (pass centered yc — Pitfall 2). (g) NaN-guard `result.gamma`; back-project `beta[j] = (0..ncomp.min(result.gamma.len())).map(|k| result.gamma[k] * fpca.rotation[(j,k)]).sum()` for `j in 0..m` (Pitfall 4 — rotation is m×ncomp, `rotation[(j,k)]`); NaN-guard `beta`. (h) `intercept = y_bar`; `base = y_bar - w_bar·beta`; `fitted_values[i] = base + sum_j data[(i,j)]*w[j]*beta[j]`. (i) Return `LpeerResult { beta, intercept: y_bar, w_bar, fitted_values, sigma2_subject: result.sigma2_u, sigma2_resid: result.sigma2_eps, n_subjects, lambda, penalty_type: config.penalty.clone(), gcv: gcv_score, lambda_method }`.

    Add `impl LpeerResult { pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Result<Vec<f64>, FdarError> { peer_predict_core(&self.beta, self.intercept, &self.w_bar, new_data, argvals) } }` (marginal / fixed-effect-only prediction; doc that new-subject random effect = 0, per D — lpeer new-curve predict is marginal).

    Add an inline `make_lpeer_fixture()` (RESEARCH §6): n_subjects=10, obs_per=5 (n=50), m=20, `true_beta[j] = sin(pi*t[j])`, inject deterministic subject effect `u_s = hash_unit(s) * sqrt(sigma2_u_true)` with `sigma2_u_true = 1.0`, design via `hash_unit`, small within-subject noise; return `(data, y, t, subject_map, sigma2_u_true)`. Add tests: `test_lpeer_variance_non_negative` (assert `sigma2_subject >= 0.0 && sigma2_resid >= 0.0`); `test_lpeer_beta_recovery` (max abs error vs true_beta < 0.5); `test_lpeer_sigma2_tracks_injection` (assert `0.1 < sigma2_subject < 5.0`, tracking injected 1.0 within a factor of ~3 at small n); `test_lpeer_invalid_subject_map` (subject_map of wrong length -> `Err(FdarError::InvalidDimension)`); `test_lpeer_single_subject_rejected` (all-equal subject_map -> `Err(FdarError::InvalidParameter)`); `test_lpeer_predict_self_consistent` (`lpeer(...).predict(&data, &t)` within 1e-9 of `fitted_values`).
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer::tests::test_lpeer_variance_non_negative peer::tests::test_lpeer_beta_recovery peer::tests::test_lpeer_sigma2_tracks_injection peer::tests::test_lpeer_invalid_subject_map peer::tests::test_lpeer_single_subject_rejected peer::tests::test_lpeer_predict_self_consistent -- --nocapture</automated>
    <fails_when>a variance component is negative, lpeer beta(t) error vs sin(pi*t) exceeds 0.5, fitted sigma2_subject falls outside (0.1, 5.0) for injected 1.0, a wrong-length subject_map does not yield InvalidDimension, a single-subject map is not rejected with InvalidParameter, or LpeerResult::predict on training curves diverges from fitted_values by >1e-9.</fails_when>
  </verify>
  <acceptance_criteria>
    - `lpeer()` returns `LpeerResult` with non-negative `sigma2_subject`/`sigma2_resid`.
    - `lpeer()` recovers `sin(pi*t)` within 0.5 and fitted `sigma2_subject` tracks injected variance in (0.1, 5.0).
    - Wrong-length subject_map -> `InvalidDimension`; single-subject map -> `InvalidParameter`.
    - `LpeerResult::predict` self-consistent within 1e-9 (shared `peer_predict_core`).
  </acceptance_criteria>
  <done>lpeer() fits subject random effects via famm, returns variance components, recovers beta(t), and its predict is self-consistent — PER-04 satisfied and committed.</done>
  <precondition>Phase 67 shipped `peer()`, `build_q`, `select_lambda_gcv_peer`, `select_lambda_reml_peer`, `PeerConfig`, `LambdaChoice`, `LambdaMethod` in `peer.rs`, and `famm::fit_scalar_mixed_model` + `famm::build_subject_map` are `pub(crate)` — assert these symbols resolve at compile time before running.</precondition>
  <reversibility rating="one-way">`lpeer` signature and the `LpeerResult` public struct (field set incl. `sigma2_subject`/`sigma2_resid`/`n_subjects`) are the published contract on the v0.36.0 tag. Unattended run — do NOT insert a checkpoint.</reversibility>
</task>

<task type="auto">
  <name>Task 3: Crate-root + prelude re-exports + running module/lpeer doctests</name>
  <files>fdars-core/src/peer.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - RESEARCH §5 (Integration: exact insertion points + doctest bodies).
    - `fdars-core/src/lib.rs` lines 593-603 (optimal_design/clustering_advanced re-export blocks — insert the peer block near here, explicit no-wildcard alphabetical list).
    - `fdars-core/src/prelude.rs` line 105 (last block `CoClusterConfig, CoClusterResult, CoClusterSelectResult` — append the peer block after it).
    - `fdars-core/src/peer.rs` lines 14-25 (current `no_run` module-header snippet to replace).
  </read_first>
  <action>
    In `lib.rs`, after the `optimal_design` re-export block (~line 596), add a `pub use peer::{...}` block listing exactly `lpeer, peer, LambdaChoice, LambdaMethod, LpeerResult, PeerConfig, PeerPenalty, PeerResult` (explicit, no wildcard, alphabetical) with a comment `// Re-export PEER regression types (v0.36.0)` (per D — CONTEXT.md crate-root exports). In `prelude.rs`, append `pub use crate::peer::{...}` with the same set after the co-clustering block (per D — prelude exports).

    In `peer.rs`, replace the `//! ```no_run` module-header block (lines 16-25) with a RUNNING `//!` doctest per RESEARCH §5: build a tiny deterministic dataset (n=10, m=5, `argvals` uniform on [0,1], design `((i*m+j) as f64 * 0.3).sin()`, response accumulated from `true_beta = sin(pi*t)` and Simpson weights), fit with `PeerConfig { penalty: PeerPenalty::Ridge, lambda: LambdaChoice::Fixed(1e-3) }`, assert `fit.beta.len() == m`, assert fitted values finite, then `let preds = fit.predict(&data, &argvals).unwrap();` and assert each pred within 1e-9 of the matching `fitted_values` (Pitfall 6 — use non-constant sinusoidal design, not FdMatrix::zeros, so the penalized system is non-singular). Add a separate `///` doctest on the `lpeer` fn per RESEARCH §5: n=12, m=5, 3 subjects × 4 obs (`subject_map[i] = i/4`), Ridge Fixed(1e-2), assert `fit.beta.len() == m`, `fit.sigma2_subject >= 0.0`, `fit.sigma2_resid >= 0.0`, and `fit.predict(&data, &argvals).unwrap().len() == n`.

    Note the module-header doctest imports must be crate-external form (`use fdars_core::matrix::FdMatrix; use fdars_core::peer::{peer, PeerConfig, PeerPenalty, LambdaChoice}; use fdars_core::helpers::simpsons_weights;`).
  </action>
  <verify>
    <automated>cargo test -p fdars-core --doc --features linalg,parallel peer</automated>
    <fails_when>the module-header or lpeer doctest fails to compile or an in-doctest assert trips (e.g. predict not within 1e-9 of fitted_values, or a variance component negative), or a re-exported symbol is not found.</fails_when>
  </verify>
  <acceptance_criteria>
    - `lib.rs` and `prelude.rs` each re-export the full 8-item PEER set (explicit, no wildcard).
    - The running module doctest and the `lpeer` `///` doctest both pass under `cargo test --doc`.
    - The former `no_run` snippet is gone.
  </acceptance_criteria>
  <done>Full PEER/lpeer surface reachable from crate root + prelude and both doctests pass under cargo test --doc — the doctest half of PER-05 satisfied and committed.</done>
  <reversibility rating="one-way">The crate-root and prelude re-export sets are the published public surface on the v0.36.0 tag. Unattended run — do NOT insert a checkpoint.</reversibility>
</task>

<task type="auto">
  <name>Task 4: Export-reachability compile checks + phase gate (full suite, doctests, clippy, fmt)</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - RESEARCH §"Wave 1 Gaps" (`test_crate_root_exports_compile`, `test_prelude_exports_compile`).
    - Memory note: CI clippy uses `--all-targets` — a plain `-p ... -D warnings` misses test-code warnings; always run `--all-targets`.
    - Memory note: `--no-verify` commits skip cargo fmt — run `cargo fmt` per commit; the phase gate re-checks `cargo fmt -- --check`.
  </read_first>
  <action>
    Add two inline compile-check tests in the `peer.rs` test module. `test_crate_root_exports_compile`: a `use fdars_core::{peer, lpeer, LpeerResult, PeerConfig, PeerPenalty, PeerResult, LambdaChoice, LambdaMethod};` reference — bind each as a value/type path (e.g. `let _f: fn(&FdMatrix, &[f64], &[f64], &PeerConfig) -> Result<PeerResult, FdarError> = peer;` and analogous for `lpeer`; reference the enums/structs in a `let _ = ... ;` type position) so a missing or misnamed crate-root export fails to compile. Note: crate-internal tests reference these as `crate::{...}` if `fdars_core::` self-path is unavailable in-crate — use `crate::` paths to hit the same re-export block. `test_prelude_exports_compile`: `use crate::prelude::*;` then reference `PeerResult`, `LpeerResult`, `PeerConfig`, `PeerPenalty`, `LambdaChoice`, `LambdaMethod`, `peer`, `lpeer` in type/value position to prove the prelude brings them into scope.

    Then run the full phase gate and fix anything red: full suite, doctests, `--all-targets` clippy, and fmt check.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer:: && cargo test -p fdars-core --doc --features linalg,parallel peer && cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo fmt -- --check</automated>
    <fails_when>a crate-root or prelude symbol does not resolve (compile-check test fails to build), any peer test or doctest is red, clippy emits a warning under --all-targets, or fmt reports drift.</fails_when>
  </verify>
  <acceptance_criteria>
    - `test_crate_root_exports_compile` and `test_prelude_exports_compile` both compile and pass (proving reachability).
    - Full `peer::` suite + peer doctests green; `--all-targets` clippy clean; `cargo fmt -- --check` clean.
  </acceptance_criteria>
  <done>The full PEER/lpeer surface is provably reachable from crate root + prelude, and the phase passes suite + doctests + clippy + fmt — PER-05 integration closed and committed. Milestone-closing phase complete.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller -> lpeer/predict | Untrusted `subject_map`, `y`, `argvals`, and `new_data` dimensions cross into the numerical routine. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-68-01 | Tampering | `lpeer` input validation | low | mitigate | Validate `subject_map.len()==n`, `n_subjects>=2`, `n>=2`, `m>=3`, `argvals.len()==m`, `y.len()==n`, finite/monotone `y`/`argvals` -> descriptive `FdarError` before any solve. |
| T-68-02 | Denial of Service | `predict` dimension mismatch | low | mitigate | `peer_predict_core` validates `new_data.ncols()==m` and `argvals.len()==m` -> `FdarError::InvalidDimension`; no unchecked indexing. |
| T-68-03 | Tampering | degenerate mixed-model fit | low | mitigate | famm clamps variance components >= 1e-15; post-solve NaN guards on `gamma` and `beta` return `FdarError::ComputationFailed` rather than emitting NaN. |

No high-severity threats: pure in-crate numerical library, no I/O, no external service, no untrusted deserialization. ASVS L1.
</threat_model>

<verification>
- After Task 1: `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_predict_self_consistent peer::tests::test_predict_wrong_ncols peer::tests::test_predict_new_curves_finite`
- After Task 2: `cargo test -p fdars-core --features linalg,parallel peer::` (lpeer + predict tests green)
- After Task 3: `cargo test -p fdars-core --doc --features linalg,parallel peer`
- Phase gate (Task 4): `cargo test -p fdars-core --features linalg,parallel` && `cargo test -p fdars-core --doc --features linalg,parallel` && `cargo clippy --all-targets --features linalg,parallel -- -D warnings` && `cargo fmt -- --check`
</verification>

<success_criteria>
- PER-04: `lpeer()` fits subject random effects, returns `LpeerResult` with non-negative variance components, recovers beta(t) within tolerance, and fitted `sigma2_subject` tracks injected between-subject variance.
- PER-05: `predict` self-consistent on re-passed training curves (both result types) + finite on new curves + validates ncols; full PEER/lpeer surface reachable from crate root + prelude; running module + lpeer doctests pass `cargo test --doc`.
- Additive/non-breaking: no existing public signature changed; R + WASM bindings + 28 examples unaffected.
- Full suite + doctests + `--all-targets` clippy + fmt all clean.
</success_criteria>

<output>
Create `.planning/phases/68-longitudinal-peer-prediction-integration/68-01-SUMMARY.md` when done.
</output>
