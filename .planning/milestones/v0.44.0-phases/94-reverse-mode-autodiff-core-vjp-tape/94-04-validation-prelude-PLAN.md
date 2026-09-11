---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: 04
type: execute
wave: 4
depends_on: [94-03]
files_modified:
  - fdars-core/src/autodiff/reverse.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [RAD-03]
estimate:
  tokens: 65000
  raw_tokens: 32000
  tasks: 3
  confidence: low
must_haves:
  truths:
    - "vjp gradients through `soft_dtw_distance_generic::<Var>` match central finite differences within 1e-6 (RAD-03)"
    - "vjp gradients through `project_scores_generic::<Var>` (FPCA scores) match central finite differences within 1e-6 (RAD-03)"
    - "A composed objective `soft_dtw + λ·Σscores²` differentiated via vjp matches central FD per component within 1e-6 (RAD-03)"
    - "`soft_dtw_distance_generic::<Var>` and `project_scores_generic::<Var>` instantiate with ZERO call-site changes to those functions"
    - "Var, vjp are re-exported from the prelude; a running module doctest demonstrates vjp"
  artifacts:
    - "fdars-core/src/autodiff/reverse.rs — Tier-4 FD cross-check tests (soft_dtw, project_scores, composed) + a vjp doctest"
    - "fdars-core/src/prelude.rs — extended re-export line including vjp, Var"
  key_links:
    - "Var implements Scalar completely, so the two already-generic subset functions accept &[Var] unchanged (the RAD-03 validation path)"
    - "The f64 instantiations soft_dtw_distance_generic::<f64> / project_scores_generic::<f64> serve as the FD oracle"
    - "prelude.rs:21 extends the existing autodiff re-export without breaking the forward-mode exports"
    - "No existing f64 call site, R/WASM binding, or example signature is changed (non-breaking, additive-only)"
---

<objective>
Close RAD-03: validate reverse-mode gradients against central finite differences on the existing differentiable subset — `soft_dtw_distance_generic` and `project_scores_generic` (FPCA scores), both called unchanged at `Var` — plus a composed `soft_dtw + λ·Σscores²` objective checked per component. Then expose the reverse-mode surface: re-export `Var` and `vjp` from the prelude and add a running module doctest demonstrating `vjp`.

Purpose: RAD-03 is the payoff of the whole phase — reverse-mode gradients must agree with FD (and, via Plan 03, with forward-mode Dual) on real functional-data objectives, proving `Var` is a correct drop-in `Scalar`. The prelude re-export + doctest make the new capability discoverable and lock a runnable usage example. All changes are strictly additive: the validation targets are called with zero signature changes.

Output: green Tier-4 FD cross-check tests, prelude re-exports, and a passing `vjp` doctest.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-PATTERNS.md
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-03-SUMMARY.md
@fdars-core/src/metric/soft_dtw.rs
@fdars-core/src/regression.rs
</context>

<build_and_commit_constraints>
HARD project hazards (from MEMORY.md) — apply to EVERY task:
- Pre-commit hook times out / gets killed — commit with `git commit --no-verify`.
- Run `cargo fmt` BEFORE each commit. Doctests link in a small `/tmp` tmpfs — if a doctest build fails with a bogus "No space left", free `/tmp` first.
- Gates OUT-OF-BAND, per-gate, FOREGROUND, 600s timeout.
- Clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- `rm -rf target/debug/{incremental,examples}` if `/home` is tight.
- Keep `--features serde` build green. NO new crate dependency.
- The MEMORY `soft_dtw_backward zero-gradient bug` is in the OLD hand-rolled f64 barycenter backward — it does NOT affect this path (vjp goes through the generic Var tape). Do NOT call the old f64 `soft_dtw_backward`.
</build_and_commit_constraints>

<artifacts_this_phase_produces>
This plan finalizes: the prelude re-exports (`vjp`, `Var`) and a running module doctest for `vjp`. `Tape` stays opaque (NOT re-exported) per CONTEXT.md discretion — it is fully hidden behind `vjp`. No new production symbols beyond the re-exports.
</artifacts_this_phase_produces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Tier-4 FD cross-check — vjp through soft_dtw_distance_generic and project_scores_generic</name>
  <files>fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/metric/soft_dtw.rs (line 158: `pub fn soft_dtw_distance_generic<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S` — validation target #1, called unchanged; the `softmin3_generic` recurrence relies on value-only PartialOrd, already satisfied)
    - fdars-core/src/regression.rs (line 232: `pub fn project_scores_generic<S: Scalar>(curve: &[S], mean: &[f64], rotation: &FdMatrix, weights: &[f64], ncomp: usize) -> Vec<S>` — validation target #2; line 387 `fdata_to_pc` builds the FPCA model; FpcaResult fields `rotation`@54, `mean`@58, `weights`@62)
    - fdars-core/src/autodiff/forward.rs (the composed-objective test `grad_composed_objective_matches_finite_diff` ~1029–1124 — the EXACT setup to mirror with vjp: m=24, n=40, ncomp=3, gamma=0.1, seed 20260906, argvals on [0.1,0.9], spanning full-rank training data; imports `crate::metric::soft_dtw_distance_generic`, `crate::regression::{fdata_to_pc, project_scores_generic}`)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md ("Central FD Cross-Check on soft_dtw_distance_generic" code example; Tolerances: h=1e-8 simple, tol 1e-6)
  </read_first>
  <behavior>
    - `vjp_soft_dtw_matches_finite_diff`: build a curve of m=16 sample points and a reference; `vjp(|vars| soft_dtw_distance_generic(vars, &ref_vars, gamma), &curve)` where `ref_vars` are `Scalar::from_f64(r)` constants; for each component j, central FD via `soft_dtw_distance_generic::<f64>(&plus, &reference, gamma)` with h=1e-8; assert `(gradient[j] - fd).abs() < 1e-6`.
    - `vjp_project_scores_matches_finite_diff`: build a small FPCA model with `fdata_to_pc`, take a curve, define the scalar objective `Σ project_scores_generic(curve, mean, rotation, weights, ncomp)²`, vjp its gradient, and central-FD-check each component within 1e-6 (FD oracle uses `project_scores_generic::<f64>`).
  </behavior>
  <action>
    Add Tier-4 FD cross-check tests to `#[cfg(test)] mod tests` in `reverse.rs`, importing the targets by absolute path exactly as the forward composed test does: `use crate::metric::soft_dtw_distance_generic;`, `use crate::regression::{fdata_to_pc, project_scores_generic};`, `use crate::matrix::FdMatrix;`.

    Test 1 (soft-DTW): mirror the "Central FD Cross-Check" example from 94-RESEARCH.md — curve of m=16 sin-valued points, cos-valued reference, gamma=0.1, `ref_vars: Vec<Var> = reference.iter().map(|&r| Scalar::from_f64(r)).collect()`, call `vjp(|vars: &[Var]| soft_dtw_distance_generic(vars, &ref_vars, gamma), &curve)`, central FD with h=1e-8 per component, assert within 1e-6. This calls `soft_dtw_distance_generic::<Var>` with ZERO changes to that function.

    Test 2 (FPCA scores): build a spanning full-rank training set (reuse the forward composed test's seed 20260906, m=24, n=40, ncomp=3, argvals on [0.1,0.9], column-major FdMatrix construction), `let fpca = fdata_to_pc(&data, ncomp, &argvals).unwrap();`, extract `mean`/`rotation`/`weights` from the FpcaResult, pick a curve, and define the scalar objective as the sum of squared scores: `let scores = project_scores_generic(vars, &mean, &rotation, &weights, ncomp); let mut acc = Scalar::zero(); for s in &scores { acc = acc + *s * *s; } acc`. vjp its gradient, FD-check each component with h=1e-8 (FD oracle: same objective at `project_scores_generic::<f64>`) within 1e-6.

    Do NOT modify `soft_dtw.rs` or `regression.rs` — they are read-only integration points; the only proof needed is that `::<Var>` instantiates and the gradients match FD.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff::reverse 2>&1 | tail -25</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" / "error[" in output, or any FD-vs-reverse component disagreement exceeding 1e-6, or a compile error instantiating the generic at Var</fails_when>
  </verify>
  <acceptance_criteria>
    - `soft_dtw_distance_generic::<Var>` and `project_scores_generic::<Var>` compile and run with no changes to those functions.
    - vjp gradients match central FD (h=1e-8) within 1e-6 on both targets, per component.
    - `soft_dtw.rs` and `regression.rs` are unmodified (`git diff` on them is empty).
  </acceptance_criteria>
  <done>Reverse-mode gradients match finite differences within 1e-6 on both existing differentiable-subset functions, called unchanged — RAD-03's FD half is proven on the real functional-data path.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Composed-objective vjp FD cross-check (soft_dtw + λ·Σscores²)</name>
  <files>fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/autodiff/reverse.rs (from Task 1 — the FPCA-model setup and imports to reuse)
    - fdars-core/src/autodiff/forward.rs (the `grad_composed_objective_matches_finite_diff` test ~1029–1124 — mirror EXACTLY, swapping grad→vjp, Dual→Var, `Dual::constant(r)`→`Scalar::from_f64(r)`, `Dual::constant(0.0)`→`Scalar::zero()`, `Dual::constant(lambda)`→`Scalar::from_f64(lambda)`; keep m=24, n=40, ncomp=3, gamma=0.1, lambda=1.0, seed 20260906, h=1e-6)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md ("Composed-Objective vjp Test Pattern"; Tolerances: composed objective h=1e-6, tol 1e-6)
  </read_first>
  <behavior>
    - `vjp_composed_objective_matches_finite_diff`: objective(c) = `soft_dtw_distance_generic(c, reference, gamma) + lambda * Σ project_scores_generic(c, ...)²`; `vjp(objective, &curve)`; for each component j, central FD with h=1e-6 on the f64 objective; assert `(gradient[j] - fd).abs() < 1e-6`.
  </behavior>
  <action>
    Add `vjp_composed_objective_matches_finite_diff` to the test module, mirroring the forward-mode `grad_composed_objective_matches_finite_diff` (forward.rs ~1029) verbatim in structure but using `vjp` and `Var`. Reuse the identical setup (same seed, m, n, ncomp, gamma, lambda, argvals). Build the composed scalar objective as a `|c: &[Var]| -> Var` closure: `let sdtw = soft_dtw_distance_generic(c, &reference_vars, gamma); let scores = project_scores_generic(c, &mean, &rotation, &weights, ncomp); let mut acc = Scalar::zero(); for s in &scores { acc = acc + *s * *s; } sdtw + Scalar::from_f64(lambda) * acc`. Run `let (value, gradient) = vjp(objective, &curve);`. Build an f64 version of the SAME objective as the FD oracle and central-FD-check each component with h=1e-6 within 1e-6.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff::reverse::tests::vjp_composed_objective_matches_finite_diff 2>&1 | tail -20</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" in output, or any component FD disagreement exceeding 1e-6</fails_when>
  </verify>
  <acceptance_criteria>
    - The composed objective (soft_dtw + λ·Σscores²) is differentiated via a single vjp call.
    - Every gradient component matches central FD (h=1e-6) within 1e-6.
    - Setup mirrors the forward composed test (same seed/dims) for direct comparability.
  </acceptance_criteria>
  <done>A composed functional-data objective built from both subset functions is correctly differentiated in one reverse sweep, FD-checked per component — the capstone RAD-03 validation.</done>
</task>

<task type="auto">
  <name>Task 3: Prelude re-exports (Var, vjp) + running vjp module doctest</name>
  <files>fdars-core/src/prelude.rs, fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/prelude.rs (line 21: `pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};` — the exact line to extend, preceded by the comment `// Forward-mode automatic-differentiation core (v0.39.0 DIF-04)`)
    - fdars-core/src/autodiff/reverse.rs (the `vjp` fn — where the `///` doctest goes)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-PATTERNS.md ("prelude.rs (line 21 — extend re-export)" — target line; "Tape is left at autodiff::Tape only, not re-exported")
    - fdars-core/src/autodiff/mod.rs (confirm `pub use reverse::{vjp, Var};` is present so the prelude path resolves)
  </read_first>
  <action>
    Extend `fdars-core/src/prelude.rs` line 21 to additionally re-export the reverse-mode surface: `pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var};`. Add a brief comment noting reverse-mode (v0.44.0 DIF-F1) alongside the existing forward-mode comment. Do NOT re-export `Tape` — it stays opaque and fully hidden behind `vjp` (CONTEXT.md discretion, 94-RESEARCH Open Question 1). This is a strictly additive change — the existing forward-mode exports remain.

    Add a running `///` module/function doctest on `vjp` in `reverse.rs` demonstrating a real gradient, e.g.:
    - Use `use fdars_core::prelude::*;` (or `use fdars_core::autodiff::{vjp, Var};`).
    - Show `let (value, grad) = vjp(|x: &[Var]| x[0] * x[0] + x[1], &[3.0, 5.0]);` with `assert!((value - 14.0).abs() < 1e-10);` and `assert!((grad[0] - 6.0).abs() < 1e-10);` `assert!((grad[1] - 1.0).abs() < 1e-10);`.
    The doctest must be a runnable ```` ```rust ```` block (not `ignore`/`no_run`) so `cargo test --doc` executes it.

    Verify no existing f64 call site, R/WASM binding, or example changed — this plan only ADDS re-exports and a doctest.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel --doc autodiff 2>&1 | tail -20</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" in output, or the doctest not being collected (means it was marked ignore/no_run), or a bogus "No space left" (free /tmp and retry)</fails_when>
  </verify>
  <acceptance_criteria>
    - `prelude.rs` re-exports `vjp` and `Var` (additively; forward-mode exports intact); `Tape` NOT re-exported.
    - A runnable `vjp` doctest exists and passes under `cargo test --doc`.
    - No existing f64 signature, binding, or example changed.
  </acceptance_criteria>
  <done>Var and vjp are discoverable via the prelude, Tape stays opaque, and a running doctest demonstrates the reverse-mode gradient API — the reverse-mode surface is exposed non-breakingly.</done>
</task>

</tasks>

<threat_model>
No attack surface — pure numerical library code operating on caller-provided f64 slices; no I/O, network, deserialization, filesystem, auth, or privilege boundary. The only correctness hazards are numerical (FD vs reverse disagreement), covered by the Tier-4 cross-check tests here, not by security controls.

| Boundary | Description |
|----------|-------------|
| (none) | No trust boundary crossed — in-crate numeric computation on f64 slices |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel autodiff` — Tier-4 FD cross-checks + composed objective green.
- `cargo test -p fdars-core --features linalg,parallel --doc` — the vjp doctest passes.
- Full suite: `cargo test -p fdars-core --features linalg,parallel` green (non-regression proof).
- Out-of-band gates: `cargo fmt --check`; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`; `cargo build -p fdars-core --features serde` green.
- `git diff Cargo.toml fdars-core/Cargo.toml fdars-core/src/metric/soft_dtw.rs fdars-core/src/regression.rs` empty (no dependency added, validation targets unmodified).
</verification>

<success_criteria>
- Reverse-mode gradients match FD within 1e-6 on soft_dtw + FPCA scores + composed objective (RAD-03).
- Validation targets called at Var with zero signature changes; strictly additive.
- Var/vjp in prelude, Tape opaque, running vjp doctest; no new dependency; clippy + fmt + serde clean.
</success_criteria>

<output>
Create `.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-04-SUMMARY.md` when done, noting final FD tolerances observed and confirming RAD-01/02/03 all satisfied.
</output>
