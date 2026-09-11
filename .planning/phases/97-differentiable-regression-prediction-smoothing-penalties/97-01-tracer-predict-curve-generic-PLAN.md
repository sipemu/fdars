---
phase: 97-differentiable-regression-prediction-smoothing-penalties
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/scalar_on_function/fregre_lm.rs
  - fdars-core/src/scalar_on_function/mod.rs
  - fdars-core/src/scalar_on_function/tests.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [DOP-02]
estimate:
  tokens: 55000
  raw_tokens: 30000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "predict_curve_generic::<f64>(curve, fit) matches predict_fregre_lm per-curve within 1e-9 (DOP-02)"
    - "Gradient of predict_curve_generic w.r.t. the curve matches central FD at Dual (forward) within 1e-6*(1+|fd|)"
    - "Gradient of predict_curve_generic w.r.t. the curve matches central FD at Var (reverse/vjp) within 1e-6*(1+|fd|)"
    - "predict_fregre_lm public signature and body stay UNCHANGED; test_predict_fregre_lm_on_training_data stays green at 1e-6"
  artifacts:
    - "fdars-core/src/scalar_on_function/fregre_lm.rs — pub fn predict_curve_generic<S: Scalar>"
    - "fdars-core/src/scalar_on_function/tests.rs — parity + Dual FD + Var FD tests for predict_curve_generic"
    - "fdars-core/src/lib.rs + prelude.rs — additive re-exports of predict_curve_generic"
  key_links:
    - "predict_curve_generic composes crate::regression::project_scores_generic (Phase 94, reused unchanged)"
    - "predict_curve_generic re-exported through scalar_on_function/mod.rs:46 so tests (use super::*) and prelude reach it"
---

<objective>
Add the differentiable per-curve FPCR prediction entry point `predict_curve_generic<S: Scalar>(curve: &[S], fit: &FregreLmResult) -> S` (DOP-02), composing Phase 94's `project_scores_generic` with the coefficient combination `intercept + Σ_k coefficients[1+k]·score_k`. This is the phase tracer: it wires the whole differentiable-prediction path (Scalar curve → project_scores_generic → coefficient fold → scalar output) end-to-end, proven by an f64-parity test plus forward-mode (`Dual`) and reverse-mode (`Var`/`vjp`) FD gradient checks w.r.t. the curve.

Purpose: proves the differentiable-prediction loop end-to-end before the penalty work in Plan 02, on the executor's best early-context tokens.
Output: `predict_curve_generic` (additive), its three tests, and additive crate-root + prelude re-exports. `predict_fregre_lm` and all its callers stay byte-identical.
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

# Analog (reused unchanged — the primary template):
@fdars-core/src/regression.rs
# Target file + its FregreLmResult owner:
@fdars-core/src/scalar_on_function/fregre_lm.rs
@fdars-core/src/scalar_on_function/mod.rs
@fdars-core/src/scalar_on_function/tests.rs
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end differentiable-prediction slice — RED tests first (f64 parity 1e-9 + combined Dual/Var FD w.r.t. curve)</name>
  <files>fdars-core/src/scalar_on_function/tests.rs</files>
  <read_first>
    - 97-PATTERNS.md lines 238–361 (the three DOP-02 test skeletons: parity, Dual FD, Var FD — copy their structure verbatim)
    - 97-RESEARCH.md lines 416–478 (test code examples) and lines 595–599 (verbatim tolerances)
    - fdars-core/src/scalar_on_function/tests.rs lines 1–4 (module uses `use super::*;`) and lines 100–128 (existing generate_test_data + test_predict_fregre_lm_on_training_data — the frozen regression guard)
    - fdars-core/src/autodiff/forward.rs lines 177–200 (Dual::seed / Dual::constant / .extract() API)
    - fdars-core/src/autodiff/reverse.rs line 505 (`pub fn vjp<F: FnOnce(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)`)
  </read_first>
  <behavior>
    - test_predict_curve_generic_f64_parity: fit fregre_lm on generate_test_data(30,50,42) ncomp=3; for each of 30 curves assert (predict_curve_generic::<f64>(&curve,&fit) - predict_fregre_lm(&fit,&data,None)[i]).abs() < 1e-9.
    - test_predict_curve_generic_dual_fd_check: fit on generate_test_data(20,30,7) ncomp=2; per curve coordinate seed a Dual (Dual::seed for the active index, Dual::constant elsewhere), call predict_curve_generic::<Dual>, .extract() the tangent; compare to central FD (h=1e-6) with tol 1e-6*(1.0+fd.abs()).
    - test_predict_curve_generic_var_fd_check: same fixture; vjp(|c:&[Var]| predict_curve_generic::<Var>(c,&fit), &curve_f64) returns (value, grad); compare grad[j] to central FD (h=1e-6), tol 1e-6*(1.0+fd.abs()).
  </behavior>
  <action>Append the three tests to the existing `#[cfg(test)] mod tests` block in scalar_on_function/tests.rs (module already has `use super::*;`). Add the autodiff imports the new tests need inside each test fn per the PATTERNS.md skeletons: `use crate::autodiff::Dual;` (Dual test) and `use crate::autodiff::{vjp, Var};` (Var test); `Scalar` is only needed if you call `::<f64>` turbofish that requires the trait in scope — it is reachable via crate::autodiff::Scalar if the compiler asks. Copy the parity test (tol 1e-9 per CRITICAL_DESIGN_NOTES — NOT assert_eq!, NOT 1e-12: the batch kernel's 3-mul accumulation diverges from project_scores_generic's pre-folded rotation·weights by ~1e-14, documented in RESEARCH §Risk P-1a), the Dual FD test, and the Var FD test from 97-PATTERNS.md lines 268–361 verbatim, adjusting only names. Do NOT modify generate_test_data or test_predict_fregre_lm_on_training_data. These tests reference predict_curve_generic which does not yet exist — they MUST fail to compile / fail RED now. Run `cargo fmt` then commit `git commit --no-verify` (pre-commit hook times out at 30s; remove a stale .git/index.lock if present).</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel scalar_on_function::tests::test_predict_curve_generic 2>&1 | tail -20</automated>
    <fails_when>The three new tests compile and pass (they must fail RED — predict_curve_generic is not defined yet, so this is expected to be a compile error naming predict_curve_generic).</fails_when>
  </verify>
  <done>Three new tests exist referencing predict_curve_generic; the build fails RED with an unresolved `predict_curve_generic` (compile error) — proving the tests are wired to the not-yet-written function. generate_test_data and test_predict_fregre_lm_on_training_data untouched.</done>
  <reversibility rating="reversible">Additive test code; revertible by deleting the three test fns.</reversibility>
</task>

<task type="tracer" tdd="true">
  <name>Task 2: Add predict_curve_generic<S: Scalar> + additive re-exports — GREEN</name>
  <files>fdars-core/src/scalar_on_function/fregre_lm.rs, fdars-core/src/scalar_on_function/mod.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - 97-PATTERNS.md lines 24–135 (exact imports to add + the predict_curve_generic body + the frozen predict_fregre_lm kernel reference at lines 118–135 — do NOT refactor it)
    - 97-RESEARCH.md lines 155–188 (recipe P-1) and lines 220–224 (Recommendation: Option 1 — do NOT refactor predict_fregre_lm's inner loop)
    - fdars-core/src/scalar_on_function/fregre_lm.rs lines 1–10 (current imports) and lines 440–473 (predict_fregre_lm — stays UNCHANGED)
    - fdars-core/src/scalar_on_function/mod.rs line 46 (`pub use fregre_lm::{...}` — add predict_curve_generic here) and lines 62–98 (FregreLmResult fields)
    - fdars-core/src/lib.rs line 404 (predict_fregre_lm re-export list) and fdars-core/src/prelude.rs line 18 (project_scores_generic re-export)
  </read_first>
  <action>In fregre_lm.rs, add two imports beside the existing ones (neither is currently imported): `use crate::autodiff::Scalar;` and `use crate::regression::project_scores_generic;` (use the crate::regression:: path, NOT the lib re-export). After `predict_fregre_lm` (ends ~line 473) add the additive `#[must_use] pub fn predict_curve_generic<S: Scalar>(curve: &[S], fit: &FregreLmResult) -> S` exactly per 97-PATTERNS.md lines 99–113: call project_scores_generic(curve, &fit.fpca.mean, &fit.fpca.rotation, &fit.fpca.weights, fit.ncomp), init `let mut yhat = S::from_f64(fit.intercept);`, then `for k in 0..fit.ncomp { yhat += S::from_f64(fit.coefficients[1 + k]) * scores[k]; }`, return yhat. Coefficient index is `1 + k` (index 0 is the separately-stored intercept — RESEARCH Pitfall 4). Plain `<S: Scalar>` with no `= f64` default (Rust 1.97 rejects defaults on free fns per D-DOP-02). Do NOT touch predict_fregre_lm's inner loop — it stays the frozen f64 3-mul batch path (RESEARCH Option 1, Risk P-1a). Include the doc comment from PATTERNS.md lines 88–98. In mod.rs line 46 add `predict_curve_generic` to the `pub use fregre_lm::{...}` list (alphabetical or beside predict_fregre_lm) so `use super::*` in tests and the prelude reach it. In lib.rs line 404 add `predict_curve_generic` to the re-export list beside predict_fregre_lm. In prelude.rs add `pub use crate::scalar_on_function::predict_curve_generic;`. Run `cargo fmt` then commit `git commit --no-verify`.</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core && cargo test -p fdars-core --features linalg,parallel scalar_on_function::tests::test_predict_curve_generic 2>&1 | tail -20</automated>
    <fails_when>Any of the three predict_curve_generic tests fails — parity beyond 1e-9 (check coefficient index / project_scores_generic args), or a Dual/Var gradient beyond 1e-6*(1+|fd|).</fails_when>
  </verify>
  <done>All three predict_curve_generic tests pass GREEN: f64 parity within 1e-9, Dual gradient matches FD, Var gradient matches FD. predict_curve_generic reachable via crate root, prelude, and `use super::*` in tests. predict_fregre_lm body unchanged.</done>
  <reversibility rating="reversible">Additive function + additive re-exports; no existing signature or body changed.</reversibility>
</task>

<task type="auto">
  <name>Task 3: Confirm predict_fregre_lm + its callers unchanged, scalar_on_function suite green, fmt, no-verify commit</name>
  <files>fdars-core/src/scalar_on_function/fregre_lm.rs</files>
  <read_first>
    - 97-RESEARCH.md lines 620–632 (Non-Breaking Proof — the three predict_fregre_lm callers: tests.rs:117, conformal/regression.rs:109/117, mod.rs:670)
    - fdars-core/src/scalar_on_function/tests.rs lines 117–127 (test_predict_fregre_lm_on_training_data — the DOP-02 non-breaking guard)
  </read_first>
  <action>Run the full scalar_on_function module test suite to confirm the existing regression guard (test_predict_fregre_lm_on_training_data, tol 1e-6) and every other scalar_on_function test stay green alongside the three new ones. Then assert predict_fregre_lm's public signature is unchanged by grepping it. Then confirm the git diff for this plan is confined to the five files this plan owns. `cargo fmt` then `git commit --no-verify` any residual fmt changes (this task is a verification gate — if no code changed, skip the commit).</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel scalar_on_function:: 2>&1 | tail -15 && grep -nE "pub fn predict_fregre_lm\(" fdars-core/src/scalar_on_function/fregre_lm.rs</automated>
    <fails_when>Any scalar_on_function test fails (test result: FAILED in the tail), OR the grep prints zero lines (predict_fregre_lm signature was renamed/removed).</fails_when>
  </verify>
  <done>Full scalar_on_function test suite is green (0 failed) including test_predict_fregre_lm_on_training_data and the three new predict_curve_generic tests; `pub fn predict_fregre_lm(` still present with its original signature.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure in-crate numeric prediction over a caller-provided scalar slice; no external input, network, IO, or auth. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-97-01 | (n/a) | predict_curve_generic | low | accept | No attack surface — pure numeric prediction over caller-provided scalar/f64 slices; no external input/network/IO/auth. No packages installed (in-crate only). |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel scalar_on_function::` green (0 failed), including the three new tests and the frozen regression guard.
- `pub fn predict_fregre_lm(` signature present and unchanged.
- predict_curve_generic reachable from crate root and prelude (additive re-exports).
</verification>

<success_criteria>
- predict_curve_generic::<f64> matches predict_fregre_lm per-curve within 1e-9.
- Prediction gradient w.r.t. the curve matches central FD (h=1e-6) at both Dual and Var within 1e-6*(1+|fd|).
- predict_fregre_lm and its callers byte-unchanged; existing suite green.
- No new crate dependency; git diff confined to the five owned files.
</success_criteria>

<output>
Create `.planning/phases/97-differentiable-regression-prediction-smoothing-penalties/97-01-SUMMARY.md` when done.
</output>
