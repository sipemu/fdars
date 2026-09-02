---
phase: "64"
plan: "02"
type: execute
wave: 2
depends_on: ["64-01"]
files_modified:
  - fdars-core/src/optimal_design.rs
  - fdars-core/src/lib.rs
autonomous: true
requirements: [FOD-02, FOD-03]
estimate:
  tokens: 50000
  raw_tokens: 50000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "design_criterion(model, &[], DesignCriterion::Score(OptimalityKind::A)) returns Σ_k λ_k (per D-Score, FOD-02)"
    - "design_criterion(model, &[], DesignCriterion::Score(OptimalityKind::D)) returns Σ_k log λ_k (negative for λ_k<1) (FOD-02)"
    - "Cov(ξ|∅) = diag(λ): the empty-set posterior equals the prior for both A and D (FOD-02)"
    - "A-opt and D-opt are monotone non-increasing: criterion(S∪{t}) ≤ criterion(S) + 1e-12 (FOD-02)"
    - "All three DesignCriterion variants dispatch to the correct branch (FOD-03)"
    - "DesignCriterion, OptimalityKind, and design_criterion are additively re-exported from lib.rs; existing signatures unbroken; serde-gated derives compile under --features serde (FOD-03)"
  artifacts:
    - "fdars-core/src/optimal_design.rs (Score A/D branch implemented, replacing any 64-01 placeholder)"
    - "fdars-core/src/lib.rs (additive: pub mod optimal_design; + pub use of the two enums and design_criterion)"
  key_links:
    - "score branch computes K×K posterior Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ via the shared build_sigma_design Σ_d — the A_mat/Ω_i pattern from pace_fpca.rs:547–558"
    - "D-opt uses linalg::cholesky_factor + linalg::log_det_from_cholesky on the K×K Cov; result is NEGATIVE (never negated)"
    - "lib.rs `pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};` — the public surface for hand-chosen designs and the Phase 65 greedy wrapper"
---

<objective>
Expand from the proven trajectory slice (64-01) to complete the FOptDes criterion core: implement the Score-criterion branch (A-optimality = trace, D-optimality = log-det of the K×K posterior FPC-score covariance, FOD-02) sharing the same `build_sigma_design` helper, prove it with known-answer tests, then additively re-export the two enums and `design_criterion` from `lib.rs` (FOD-03).

Purpose: FOD-02 lives in this core phase so the Phase 65 greedy wrapper stays pure orchestration with no new math. The re-export makes `design_criterion` publicly usable for hand-chosen designs today and available to the Phase 65 greedy loop.

Output: The completed `optimal_design.rs` (Score branch + tests) and the additive `lib.rs` re-export; full crate test suite and clippy/fmt gates green.
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
@.planning/phases/64-criterion-machinery-core/64-01-SUMMARY.md

@fdars-core/src/optimal_design.rs
@fdars-core/src/pace_fpca.rs
@fdars-core/src/linalg.rs
@fdars-core/src/lib.rs
</context>

<artifacts_produced>
## Artifacts this plan produces

MODIFIED `fdars-core/src/optimal_design.rs`:
- private `score_criterion(model, selected, kind: OptimalityKind) -> Result<f64, FdarError>` — computes the K×K posterior covariance and returns trace (A) or log-det (D). Replaces any placeholder left by 64-01.
- additional `#[cfg(test)] mod tests` gates for the Score branch and enum dispatch.

MODIFIED `fdars-core/src/lib.rs` (additive only — no existing line removed):
- `pub mod optimal_design;`
- `pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};`

No greedy loop, no `OptDesConfig`/`OptDesResult`, no prelude change, no benchmark — all Phase 65.
</artifacts_produced>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement Score A/D posterior-covariance branch with known-answer tests</name>
  <files>fdars-core/src/optimal_design.rs</files>
  <read_first>
    - fdars-core/src/optimal_design.rs — the Task-1 `build_sigma_design`, ridge-retry helper, validation, dispatch, and the in-test synthetic-model helper from 64-01.
    - fdars-core/src/pace_fpca.rs — A_mat/Ω_i posterior-covariance assembly at ~547–558 (the pattern to mirror, substituting Φ_d/Σ_d for Φ_i/Σ_yi).
    - fdars-core/src/linalg.rs — `cholesky_solve(a,b,p)` for the p×p solves per component; `cholesky_factor(a,p)` + `log_det_from_cholesky(l,d)` (= 2·Σ ln L_ii) for the K×K log-det. All pub(crate), always compiled.
    - .planning/phases/64-criterion-machinery-core/64-RESEARCH.md — "Score criterion (FOD-02)" math and "Known-Answer Test Architecture" tests 4 and 6; Pitfalls 4 (D-opt sign) and 5 (solve the p×p Σ_d, not K×K).
  </read_first>
  <behavior>
    - test_score_a_empty_set: A(∅) == Σλ_k (3.0) within 1e-10.
    - test_score_d_empty_set: D(∅) == Σ log λ_k (ln 2.0 + ln 1.0 = ln 2 ≈ 0.6931) within 1e-10.
    - test_score_prior_recovery: with selected=&[], both A and D recover the prior (Cov = diag(λ)); D(∅) matches Σ log λ_k exactly.
    - test_monotonicity_a_opt: A(&[10,30]) <= A(&[10]) + 1e-12.
    - test_monotonicity_d_opt: D(&[10,30]) <= D(&[10]) + 1e-12.
  </behavior>
  <action>
Implement (or replace the 64-01 placeholder for) the private `score_criterion(model, selected, kind)`.

Empty-set fast path (`selected.is_empty()`): the posterior equals the prior `Cov = diag(λ)`. For `OptimalityKind::A` return `model.eigenvalues.iter().sum::<f64>()`. For `OptimalityKind::D` return `model.eigenvalues.iter().map(|&lam| lam.ln()).sum::<f64>()` directly (avoid a K×K Λ Cholesky); if any `lam <= 0.0`, return `FdarError::ComputationFailed { operation: "optimal_design D-optimality", detail: "non-positive eigenvalue in prior" }` defensively.

Non-empty path: let `p = selected.len()`, `ncomp = model.ncomp`. Call `build_sigma_design(model, selected)` to get the p×p Σ_d, then factor it ONCE with the same ridge-retry-on-fail wrapper used in 64-01 (`cholesky_factor` + retry with +1e-8 diagonal). Build the p×ncomp `Φ_d` where `Φ_d[i,k] = model.eigenfunctions[(selected[i], k)]`. Following pace_fpca.rs:547–558: for each component `k in 0..ncomp`, solve `Σ_d · x_k = Φ_d[:,k]` (p-vector) using the pre-factored Cholesky forward/back. Then assemble the K×K `a_mat[k*ncomp + l] = model.eigenvalues[k] * dot(Φ_d[:,k], x_l) * model.eigenvalues[l]` (equivalently accumulate `λ_k · Φ_d[:,k]ᵀ · x_l · λ_l`). Form the K×K posterior covariance `cov[k*ncomp + l] = (if k==l { model.eigenvalues[k] } else { 0.0 }) - a_mat[k*ncomp + l]`.

For `OptimalityKind::A` return `trace(cov) = Σ_k cov[k*ncomp + k]`. For `OptimalityKind::D` compute `log det(cov)` via `linalg::cholesky_factor(&cov, ncomp)` then `linalg::log_det_from_cholesky(&l, ncomp)`; return it DIRECTLY — it is NEGATIVE for an informative design and must NOT be negated or abs'd (Pitfall 4). If the Cov Cholesky fails (should not for a valid PSD posterior), return `FdarError::ComputationFailed { operation: "optimal_design D-optimality log-det", detail: ... }`.

CRITICAL: the per-component solve is against the p×p `Σ_d` (`cholesky_solve`/forward-back with dimension p), NOT a K×K system (Pitfall 5). Do not drop the σ²I term — it lives inside `build_sigma_design`.

Add the five Score tests named in `<behavior>` into the existing `#[cfg(test)] mod tests`, reusing the 64-01 synthetic-model helper. Use tolerance 1e-10 for equalities, 1e-12 slack for monotonicity.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg optimal_design 2>&1 | tail -25</automated>
    <fails_when>Output contains "test result: FAILED", "0 passed", or any "FAILED" line — a Score known-answer, prior-recovery, or monotonicity gate did not hold (e.g. D-opt sign flip or wrong solve dimension).</fails_when>
  </verify>
  <acceptance_criteria>
    - `score_criterion` implements the K×K `Cov = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ` via the p×p Σ_d solve and the A_mat/Ω_i pattern.
    - A returns trace(Cov); D returns log-det(Cov) unnegated (negative for informative designs).
    - Empty-set fast path returns Σλ_k (A) and Σ log λ_k (D) without a Σ_d solve.
    - All five Score tests plus the 64-01 trajectory tests pass together under the quick module command.
  </acceptance_criteria>
  <done>The Score branch is proven: A(∅)=Σλ_k, D(∅)=Σ log λ_k, prior recovery exact, both monotone non-increasing.</done>
  <reversibility rating="reversible">Fills in a private branch of a not-yet-public module; no external contract affected.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Enum-dispatch test + additive lib.rs re-export + full-suite/clippy/fmt gates</name>
  <files>fdars-core/src/optimal_design.rs, fdars-core/src/lib.rs</files>
  <read_first>
    - fdars-core/src/lib.rs — the peer `pub mod` block near line 105 (`pub mod kernel_kmeans; pub mod kshape;`) and the re-export block near lines 586–589 (`pub use kshape::{...};`). Add the new `pub mod` near the peer declarations and the new `pub use` near the other module re-exports. Additive only — remove no existing line.
    - fdars-core/src/optimal_design.rs — the completed `design_criterion` dispatch (Trajectory + Score(A|D)).
  </read_first>
  <behavior>
    - test_enum_dispatch: on the synthetic model, `design_criterion(&m, &[10], Trajectory)`, `..Score(A)`, and `..Score(D)` each return Ok and produce three distinct finite values, confirming each variant routes to its own branch (not all to one).
  </behavior>
  <action>
Add `test_enum_dispatch` to the `#[cfg(test)] mod tests`: call `design_criterion` with all three `DesignCriterion` variants on a shared synthetic model and assert each is `Ok`, finite, and that the three results are mutually distinct (route-correctness). No new math.

Edit `fdars-core/src/lib.rs` additively: add `pub mod optimal_design;` alongside the peer module declarations (near `pub mod kshape;` at line ~105–106), and add `pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};` alongside the other module re-exports (near the `pub use kshape::{...};` at line ~589). Do NOT add a prelude entry, greedy fn, or config/result type (Phase 65). Use `Edit` (scoped), never a whole-file rewrite of lib.rs.

Then run the full CI gates and fix any fallout (fmt drift is the usual culprit — run `cargo fmt` before the fmt-check):
1. `cargo fmt -p fdars-core` then `cargo fmt -p fdars-core --check`.
2. `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`.
3. `cargo build -p fdars-core --features serde` (serde-gated derives compile).
4. `cargo test -p fdars-core --features linalg,parallel` (full suite — all existing tests plus the new optimal_design tests pass).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core --check 2>&1 | tail -3</automated>
    <fails_when>Output is non-empty (a diff is printed) or exit code is non-zero — formatting drift remains; run `cargo fmt -p fdars-core` first.</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -5</automated>
    <fails_when>Output contains "error:" or "warning:" — clippy is not clean under the CI `--all-targets` gate.</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core --features serde 2>&1 | tail -3</automated>
    <fails_when>Output contains "error[" or "error:" — the serde-gated derives on DesignCriterion/OptimalityKind do not compile.</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -15</automated>
    <fails_when>Output contains "test result: FAILED" or any "FAILED" line — a new optimal_design test or a pre-existing test regressed.</fails_when>
  </verify>
  <acceptance_criteria>
    - `test_enum_dispatch` passes: all three DesignCriterion variants return distinct finite Ok values.
    - `lib.rs` additively declares `pub mod optimal_design;` and re-exports `design_criterion`, `DesignCriterion`, `OptimalityKind`; no existing line removed.
    - `cargo fmt -p fdars-core --check` clean, `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
    - `cargo build -p fdars-core --features serde` succeeds; `cargo test -p fdars-core --features linalg,parallel` reports 0 failed.
  </acceptance_criteria>
  <done>The Score branch is public, the module is re-exported additively, and every CI gate (fmt, clippy --all-targets, serde build, full test suite) is green.</done>
  <reversibility rating="reversible">Additive re-export; removable without breaking any existing public signature.</reversibility>
</task>

</tasks>

<security_note>
Numerical-robustness-only surface (same as plan 64-01): no external API/SDK, no network, no untrusted input, no schema/DB. ASVS V5 input validation and no-panic robustness are the only security-relevant controls and are covered by the shared `design_criterion` validation and the ridge-retry. No `<threat_model>` STRIDE register required.
</security_note>

<verification>
- `cargo test -p fdars-core --features linalg optimal_design` — Score + dispatch known-answer gates green.
- `cargo fmt -p fdars-core --check` — no drift.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` — CI lint gate clean.
- `cargo build -p fdars-core --features serde` — serde-gated derives compile.
- `cargo test -p fdars-core --features linalg,parallel` — full suite unbroken.
- Sanity: `grep -c 'optimal_design' fdars-core/src/lib.rs` ≥ 2 (module decl + re-export both present).
</verification>

<success_criteria>
- Score branch: `A(∅)=Σλ_k`, `D(∅)=Σ log λ_k` (negative), `Cov(ξ|∅)=diag(λ)`, both monotone non-increasing.
- Enum dispatch routes all three variants correctly.
- `lib.rs` additively re-exports the enums + `design_criterion`; nothing broken.
- fmt, clippy `--all-targets`, serde build, and the full test suite all green.
</success_criteria>

<output>
Create `.planning/phases/64-criterion-machinery-core/64-02-SUMMARY.md` when done.
</output>
