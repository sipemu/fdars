---
phase: "95"
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/helpers.rs
autonomous: true
requirements: [GEN-01]
estimate:
  tokens: 55000
  raw_tokens: 28000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "l2_distance accepts `<T: Scalar = f64>`; the unannotated caller l2_distance_matrix (distance.rs:70) resolves T=f64 unchanged (GEN-01)."
    - "l2_distance<f64> reproduces the pre-change f64 result bit-identically (GEN-01)."
    - "l2_distance flows forward-mode Dual and reverse-mode Var gradients matching central finite differences within 1e-5 (GEN-01)."
    - "The crate + distance.rs (the read-only caller) still compile with no edit to distance.rs (GEN-01)."
  artifacts:
    - "fdars-core/src/helpers.rs — l2_distance generalized in place + inline parity/Dual/Var tests"
  key_links:
    - "l2_distance ↔ crate::autodiff::Scalar bound"
    - "l2_distance ↔ l2_distance_matrix (distance.rs:70) read-only caller at T=f64"
  prohibitions:
    - statement: "No new crate dependency added to Cargo.toml"
    - statement: "No edit to distance.rs or any l2_distance caller — only helpers.rs changes"
---

<objective>
TRACER slice for GEN-01: generalize ONE shared kernel — `l2_distance` (helpers.rs:56) — in place to `<T: Scalar = f64>`, and prove the entire non-breaking-generalization loop end to end before expanding to the other three kernels: f64 bit-parity, forward-mode `Dual` flow, reverse-mode `Var` flow, and the untouched f64 caller `l2_distance_matrix` (distance.rs:70) plus the whole crate still compile.

Purpose: The whole architecture (Scalar bound, f64/generic boundary, from_f64 lift, T=f64 inference at existing call sites) is validated on one kernel on the agent's freshest context. A dead end here is caught after one commit, not after all four kernels ship.
Output: `l2_distance<T: Scalar = f64>` in helpers.rs + 3 inline tests (parity, Dual FD, Var FD). No new files. No call-site edits.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/95-generic-scalar-hot-path-signatures/95-CONTEXT.md
@.planning/phases/95-generic-scalar-hot-path-signatures/95-RESEARCH.md
@.planning/phases/95-generic-scalar-hot-path-signatures/95-PATTERNS.md
</context>

<artifacts_this_phase_produces>
Phase 95 produces (across all three plans) exactly four generalized signatures, all in place, no new files:
- `l2_distance<T: Scalar = f64>` — helpers.rs (THIS plan)
- `trapz<T: Scalar = f64>` — helpers.rs (plan 02)
- `inner_product<T: Scalar = f64>` — utility.rs (plan 02)
- `inner_product_l2<T: Scalar = f64>` — warping.rs (plan 02)
Plus their inline f64-parity + Dual/Var FD spot-check tests, and (plan 03) the non-breaking compile-gate proof.
This plan delivers the FIRST signature (l2_distance) and its three tests.
</artifacts_this_phase_produces>

<build_and_commit_constraints>
- Commit with `git commit --no-verify` (pre-commit hook runs the full cargo gate and TIMES OUT/gets killed). Remove a stale `.git/index.lock` if present before committing.
- Run `cargo fmt` before EVERY commit.
- Run each cargo gate OUT-OF-BAND, per-gate, FOREGROUND, with a 600s timeout. NEVER combine gates into one backgrounded run.
- On "No space left" / "linking with cc failed": `rm -rf target/debug/{incremental,examples}` then retry (not a code bug).
- Add NO new crate dependency.
</build_and_commit_constraints>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Generalize l2_distance in place to `<T: Scalar = f64>` + write RED parity/Dual/Var tests first</name>
  <files>fdars-core/src/helpers.rs</files>
  <read_first>
    - fdars-core/src/helpers.rs:56-63 (current f64 l2_distance body) and the existing `test_l2_distance_different` in the file's `#[cfg(test)]` block for baseline values.
    - fdars-core/src/regression.rs:232-251 (`project_scores_generic` — the f64-fold-before-lift boundary template).
    - fdars-core/src/autodiff/mod.rs:38-85 (`Scalar` trait: zero, from_f64, sqrt, Mul/Sub/AddAssign supertraits).
    - fdars-core/src/autodiff/forward.rs:177 (`Dual::seed`), :187 (`Dual::constant`), :197 (`Dual::extract() -> (f64, f64)`).
    - fdars-core/src/autodiff/reverse.rs:505 (`vjp<F: FnOnce(&[Var]) -> Var>(f, x: &[f64]) -> (f64, Vec<f64>)`) and the vjp doc example at :499 for the closure idiom.
  </read_first>
  <behavior>
    - test_l2_distance_parity: l2_distance(&c1,&c2,&w) at T=f64 equals the inlined original loop (`Σ (c1-c2)²·w` then `.sqrt()`) with `assert_eq!` (bit-identical, tolerance 0.0).
    - test_l2_distance_dual: seed one curve coordinate with `Dual::seed`, others `Dual::constant`; the extracted tangent matches a central finite difference (h=1e-5) within `1e-5 * fd.abs().max(1e-10)`.
    - test_l2_distance_var: `vjp(|x| l2_distance(x, &c2_var, &w), &c1_f64)` where c2 is lifted via `<Var as Scalar>::from_f64`; the returned grad matches central FD per coordinate within the same tolerance.
  </behavior>
  <action>
    Add `use crate::autodiff::Scalar;` to the top of helpers.rs (do not remove existing imports; keep the import ordering convention already in the file). Rewrite l2_distance in place to the exact signature `pub fn l2_distance<T: Scalar = f64>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T`, keeping its existing doc comment and any `#[must_use]`/`#[inline]` attribute it already carries. Body per RESEARCH.md §Code Examples: start `let mut dist_sq = T::zero();`, loop `let diff = curve1[i] - curve2[i]; dist_sq += diff * diff * T::from_f64(weights[i]);`, return `dist_sq.sqrt()` (this resolves to `Scalar::sqrt`, not `f64::sqrt`). The f64 weight is lifted once via `T::from_f64` — never write `T * f64` directly. Add a one-line doc note that the `= f64` default is documentary: T is inferred from the `&[T]` arguments at every existing call site (per D-01 in CONTEXT.md — in-place generic, not a `_generic` companion). Write the three tests described in `<behavior>` into the file's existing `#[cfg(test)] mod tests` block; write them BEFORE the signature change so they fail to compile (RED), then land the generalization so they pass (GREEN). Do NOT touch distance.rs or any other caller (per D-01 boundary + RESEARCH.md §Pitfall 4). Do NOT generalize `simpsons_weights` or any row op.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel helpers::tests::test_l2_distance 2>&1 | tail -20</automated>
    <fails_when>Any of the three l2_distance tests fails or does not compile; a `Sum`/`cannot multiply T by f64`/`E0393` error appears; or the parity assert_eq! reports a non-bit-identical value.</fails_when>
  </verify>
  <acceptance_criteria>
    - l2_distance signature is exactly `pub fn l2_distance<T: Scalar = f64>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T`.
    - All three tests (parity, dual, var) pass; parity is bit-identical (assert_eq!).
    - `git diff --name-only HEAD` after this task shows ONLY fdars-core/src/helpers.rs.
  </acceptance_criteria>
  <done>l2_distance is generic over T with the f64 default; parity is bit-identical; Dual and Var gradients match central FD within 1e-5.</done>
  <reversibility rating="reversible">Additive in-place generalization — the f64 path is preserved (T=f64 identity), trivially revertible to the f64-only signature.</reversibility>
</task>

<task type="auto">
  <name>Task 2: Confirm the read-only caller l2_distance_matrix + the whole crate compile unchanged</name>
  <files>fdars-core/src/helpers.rs</files>
  <read_first>
    - fdars-core/src/distance.rs:67-71 (`l2_distance_matrix` — calls `l2_distance(&data.row(i), &data.row(j), &weights)` where `data.row(i): Vec<f64>` → T=f64 infers).
    - fdars-core/src/helpers.rs:56 (the l2_distance re-exported path in lib.rs — confirm the pub signature is still exported unchanged in KIND).
  </read_first>
  <action>
    Do NOT edit any file in this task — it is the tracer's end-to-end compile proof. Run the pre-edit safety greps from RESEARCH.md §Pre-Edit Audit scoped to `l2_distance` to confirm zero fn-pointer coercions and zero `: f64 = l2_distance(` annotations remain (they were absent at research time; re-confirm post-edit). Then compile the crate and lint all targets to prove distance.rs and every other l2_distance caller resolve T=f64 with zero churn. If a genuine unavoidable break surfaces (turbofish now required at a call site), STOP and surface it — do not silently edit a caller's behavior (per D-01 residual-risk rule). Free disk first if a link error appears: `rm -rf target/debug/{incremental,examples}`.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -20</automated>
    <fails_when>clippy emits any warning/error, distance.rs fails to compile against the generalized l2_distance, or a call site now requires a turbofish (E0282/E0283/E0393).</fails_when>
  </verify>
  <acceptance_criteria>
    - clippy --all-targets is clean with -D warnings.
    - `grep -rn ": f64 = l2_distance" fdars-core/src fdars-core/tests fdars-core/examples --include="*.rs"` returns nothing.
    - distance.rs is NOT in `git diff --name-only HEAD`.
  </acceptance_criteria>
  <done>The generalized l2_distance compiles cleanly across all targets; l2_distance_matrix and every other caller resolve T=f64 with no source churn outside helpers.rs.</done>
</task>

<task type="auto">
  <name>Task 3: fmt, per-kernel test sample, and no-verify commit</name>
  <files>fdars-core/src/helpers.rs</files>
  <read_first>
    - MEMORY.md hazards: `--no-verify` commits leave fmt drift → run `cargo fmt` first; pre-commit hook times out.
  </read_first>
  <action>
    Run `cargo fmt -p fdars-core`. Confirm `git diff --name-only HEAD` lists ONLY fdars-core/src/helpers.rs (the tracer non-churn check). Remove `.git/index.lock` if present, then commit with `git commit --no-verify -m "feat(95): generalize l2_distance to <T: Scalar = f64> (GEN-01 tracer)"`.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && git show --stat HEAD | grep -E "helpers\.rs" && test -z "$(git diff --name-only HEAD -- fdars-core/src/distance.rs)" && echo CHURN_OK</automated>
    <fails_when>The commit touches any file other than helpers.rs, or distance.rs shows in the diff, or CHURN_OK is not printed.</fails_when>
  </verify>
  <acceptance_criteria>
    - HEAD commit contains only fdars-core/src/helpers.rs.
    - Working tree is clean (fmt applied, nothing unstaged in helpers.rs).
  </acceptance_criteria>
  <done>The tracer is committed; only helpers.rs changed; the full generalization loop (parity + Dual + Var + crate compile) is proven on one kernel.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No attack surface — in-place generalization of a pure numeric kernel over caller-provided f64/scalar slices; no I/O, network, deserialization, or privilege boundary.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-95-01 | Tampering | l2_distance numeric result | low | accept | f64-parity test asserts bit-identical output at T=f64; no external input crosses any boundary. |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel helpers::tests::test_l2_distance` — all three tests green.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` — clean.
- `git diff --name-only HEAD` — only helpers.rs.
</verification>

<success_criteria>
l2_distance is generic over `T: Scalar = f64`; f64 parity is bit-identical; Dual and Var gradients match central FD within 1e-5; distance.rs and the crate compile unchanged with no call-site churn.
</success_criteria>

<output>
Create `.planning/phases/95-generic-scalar-hot-path-signatures/95-01-SUMMARY.md` when done.
</output>
