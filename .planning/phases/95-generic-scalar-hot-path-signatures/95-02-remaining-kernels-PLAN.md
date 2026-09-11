---
phase: "95"
plan: "02"
type: execute
wave: 2
depends_on: ["95-01"]
files_modified:
  - fdars-core/src/helpers.rs
  - fdars-core/src/utility.rs
  - fdars-core/src/warping.rs
autonomous: true
requirements: [GEN-01]
estimate:
  tokens: 70000
  raw_tokens: 36000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "trapz, inner_product, and inner_product_l2 each accept `<T: Scalar = f64>`; unannotated f64 call sites resolve T=f64 unchanged (GEN-01)."
    - "Each generalized kernel at T=f64 reproduces its pre-change f64 result bit-identically (GEN-01)."
    - "inner_product's iterator `.sum()` is rewritten to an explicit `T::zero()` accumulator loop (GEN-01)."
    - "trapz, inner_product, inner_product_l2 flow forward-mode Dual gradients matching central FD within 1e-5; inner_product_l2 also flows reverse-mode Var (GEN-01)."
  artifacts:
    - "fdars-core/src/helpers.rs — trapz generalized in place + inline tests"
    - "fdars-core/src/utility.rs — inner_product generalized in place (accumulator-loop rewrite) + inline tests"
    - "fdars-core/src/warping.rs — inner_product_l2 generalized in place + inline tests"
  key_links:
    - "inner_product_l2 ↔ generic trapz (T inferred from &[T])"
    - "inner_product ↔ simpsons_weights (stays f64) via T::from_f64 lift"
    - "trapz ↔ all f64 callers (density_fda, frechet, alignment, fts) at T=f64"
  prohibitions:
    - statement: "No new crate dependency added to Cargo.toml"
    - statement: "No edit to any caller of trapz/inner_product/inner_product_l2 — only helpers.rs/utility.rs/warping.rs change"
    - statement: "simpsons_weights and cumulative_trapz stay f64 — not generalized"
---

<objective>
Expand from the proven tracer to the remaining three shared kernels, IN THE DEPENDENCY ORDER trapz → inner_product → inner_product_l2 (inner_product_l2 delegates to the generalized trapz). Each is generalized in place to `<T: Scalar = f64>` with its inline f64-parity + Dual FD spot-check (and a Var FD check for inner_product_l2), mirroring the tracer's boundary rules.

Purpose: Complete the four-kernel shared substrate the DOP families (Phases 96/97/98) are written against, keeping every existing f64 call site and the f64 numeric results unchanged.
Output: `trapz<T>`, `inner_product<T>`, `inner_product_l2<T>` generalized in place + their tests. No new files. No call-site edits.
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
@.planning/phases/95-generic-scalar-hot-path-signatures/95-01-SUMMARY.md
</context>

<build_and_commit_constraints>
- Commit with `git commit --no-verify` (pre-commit hook runs the full cargo gate and TIMES OUT/gets killed). Remove a stale `.git/index.lock` if present.
- `cargo fmt` before EVERY commit.
- Run each cargo gate OUT-OF-BAND, per-gate, FOREGROUND, 600s timeout. NEVER one combined backgrounded gate.
- On "No space left" / "linking with cc failed": `rm -rf target/debug/{incremental,examples}` then retry.
- Add NO new crate dependency.
</build_and_commit_constraints>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Generalize trapz in place to `<T: Scalar = f64>` + parity/Dual tests</name>
  <files>fdars-core/src/helpers.rs</files>
  <read_first>
    - fdars-core/src/helpers.rs:253-259 (current f64 trapz body) and the file's `#[cfg(test)]` block (the l2_distance tests from plan 01 are the pattern to mirror).
    - fdars-core/src/regression.rs:232-251 (f64-fold-before-lift rule).
    - fdars-core/src/autodiff/forward.rs:177,187,197 (Dual::seed/constant/extract).
    - 95-PATTERNS.md §`trapz<T: Scalar>` (exact target body + ordering invariant).
  </read_first>
  <behavior>
    - test_trapz_parity: trapz(&y,&x) at T=f64 equals the inlined original loop `Σ 0.5·(y[k]+y[k-1])·(x[k]-x[k-1])` with `assert_eq!` (bit-identical).
    - test_trapz_dual: seed one y coordinate with `Dual::seed`, rest `Dual::constant`; extracted tangent matches central FD (h=1e-5) within `1e-5 * fd.abs().max(1e-10)`.
  </behavior>
  <action>
    `use crate::autodiff::Scalar;` is already present in helpers.rs from plan 01. Rewrite trapz in place to `pub fn trapz<T: Scalar = f64>(y: &[T], x: &[f64]) -> T`, keeping its doc comment and existing attributes. Body per 95-PATTERNS.md: `let mut sum = T::zero();`, loop `let half_dx = T::from_f64(0.5 * (x[k] - x[k - 1])); sum += half_dx * (y[k] + y[k - 1]);`, return `sum`. The ordering invariant is load-bearing: fold `0.5 * (x[k] - x[k-1])` entirely in f64 FIRST, then `T::from_f64` once — this preserves bit-identical accumulation order at T=f64 (per 95-RESEARCH.md §Pitfall 3). `x` stays `&[f64]` (grid spacings are quadrature constants). Add a doc note that the `= f64` default is documentary; T infers from `&[T]`. Write the two tests into the existing `#[cfg(test)]` block, RED before the change. Do NOT touch cumulative_trapz (stays f64, out of scope per 95-RESEARCH.md anti-patterns) or any trapz caller.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel helpers::tests::test_trapz 2>&1 | tail -20</automated>
    <fails_when>Either trapz test fails/won't compile; a `cannot multiply T by f64` or `Sum` error appears; or parity is not bit-identical.</fails_when>
  </verify>
  <acceptance_criteria>
    - trapz signature is exactly `pub fn trapz<T: Scalar = f64>(y: &[T], x: &[f64]) -> T`.
    - Both trapz tests pass; parity is bit-identical.
    - cumulative_trapz is unchanged (still f64).
  </acceptance_criteria>
  <done>trapz is generic over T with bit-identical f64 parity and FD-correct Dual gradients; all f64 trapz callers still infer T=f64.</done>
  <reversibility rating="reversible">Additive in-place generalization preserving the f64 path.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Generalize inner_product in place — rewrite `.sum()` to accumulator loop — + parity/Dual tests</name>
  <files>fdars-core/src/utility.rs</files>
  <read_first>
    - fdars-core/src/utility.rs:34-46 (current inner_product body — note the trailing `.sum()`).
    - fdars-core/src/regression.rs:240-248 (accumulator-loop analog).
    - 95-PATTERNS.md §`inner_product<T: Scalar>` (CRITICAL `.sum()` incompatibility note + target body).
    - fdars-core/src/autodiff/forward.rs:177,187,197 (Dual API).
  </read_first>
  <behavior>
    - test_inner_product_parity: inner_product(&c1,&c2,&argvals) at T=f64 equals the inlined original (`simpsons_weights` then `Σ c1·c2·w`) with `assert_eq!` (bit-identical, or `< 1e-12` if the `.sum()`→loop reorder shifts a low bit).
    - test_inner_product_dual: seed one c1 coordinate; extracted tangent matches central FD within `1e-5 * fd.abs().max(1e-10)`.
  </behavior>
  <action>
    Add `use crate::autodiff::Scalar;` to the top of utility.rs. Rewrite inner_product in place to `pub fn inner_product<T: Scalar = f64>(curve1: &[T], curve2: &[T], argvals: &[f64]) -> T`, keeping its doc comment and attributes. Preserve the early-return guard but return `T::zero()` instead of `0.0`. REPLACE the iterator `.sum()` chain with an explicit accumulator loop (mandatory — `T: Scalar` does NOT implement `std::iter::Sum`, per 95-RESEARCH.md §Pitfall 1): keep `let weights = simpsons_weights(argvals);` (stays `Vec<f64>`), then `let mut acc = T::zero(); for i in 0..curve1.len() { acc += curve1[i] * curve2[i] * T::from_f64(weights[i]); } acc`. Add a doc note that `= f64` is documentary. Write the two tests RED-first into the existing `#[cfg(test)]` block. Do NOT generalize simpsons_weights. Do NOT touch inner_product_matrix or any other caller.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel utility::tests::test_inner_product 2>&1 | tail -20</automated>
    <fails_when>A `the trait bound T: Sum is not satisfied` error appears (the `.sum()` was not rewritten), a `cannot multiply T by f64` error appears, or a parity/Dual test fails.</fails_when>
  </verify>
  <acceptance_criteria>
    - inner_product signature is exactly `pub fn inner_product<T: Scalar = f64>(curve1: &[T], curve2: &[T], argvals: &[f64]) -> T`.
    - The body uses an explicit `T::zero()` accumulator loop — no `.sum()`.
    - Both tests pass; parity within tolerance (bit-identical preferred).
  </acceptance_criteria>
  <done>inner_product is generic over T via an explicit accumulator loop; f64 parity holds; Dual gradient matches central FD; simpsons_weights and all callers are untouched.</done>
  <reversibility rating="reversible">Additive in-place generalization preserving the f64 path.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Generalize inner_product_l2 in place (delegates to generic trapz) + parity/Dual/Var tests, fmt, commit</name>
  <files>fdars-core/src/warping.rs</files>
  <read_first>
    - fdars-core/src/warping.rs:83-86 (current inner_product_l2 body) and its existing `trapz` import.
    - fdars-core/src/warping.rs:90,96,177 (f64 callers chaining `.max(0.0).sqrt()` / `.clamp(-1.0,1.0)` — must stay valid at T=f64).
    - 95-PATTERNS.md §`inner_product_l2<T: Scalar>` (mechanical target body).
    - fdars-core/src/autodiff/reverse.rs:499,505 (vjp closure idiom for the Var test).
  </read_first>
  <behavior>
    - test_inner_product_l2_parity: inner_product_l2(&psi1,&psi2,&time) at T=f64 equals the inlined original (`prod = psi1·psi2` then `trapz(prod,time)`) with `assert_eq!` (bit-identical).
    - test_inner_product_l2_dual: seed one psi1 coordinate; extracted tangent matches central FD within `1e-5 * fd.abs().max(1e-10)`.
    - test_inner_product_l2_var: `vjp(|x| inner_product_l2(x, &psi2_var, &time), &psi1_f64)` grad matches central FD per coordinate.
  </behavior>
  <action>
    Add `use crate::autodiff::Scalar;` to the top of warping.rs (confirm trapz is already imported from helpers; it is used at :83). Rewrite inner_product_l2 in place to `pub fn inner_product_l2<T: Scalar = f64>(psi1: &[T], psi2: &[T], time: &[f64]) -> T`, keeping its doc comment and attributes. Body per 95-PATTERNS.md: `let prod: Vec<T> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect(); trapz(&prod, time)` — `a * b` is `T * T → T` (no lift), `collect::<Vec<T>>()` works via `T: Copy`, and `trapz::<T>` infers from `&[T]`. Add a doc note that `= f64` is documentary. Write the three tests RED-first into the existing `#[cfg(test)]` block; for the Var test lift psi2 constants via `<Var as Scalar>::from_f64`. The callers at :90/:96/:177 must NOT be edited — they pass `&[f64]`, infer T=f64, and keep `.max(0.0)`/`.clamp(-1.0,1.0)` on the f64 return (per 95-RESEARCH.md §Risk 3). Then run `cargo fmt -p fdars-core`, confirm `git diff --name-only HEAD` lists ONLY helpers.rs, utility.rs, warping.rs, remove `.git/index.lock` if present, and `git commit --no-verify -m "feat(95): generalize trapz/inner_product/inner_product_l2 to <T: Scalar = f64> (GEN-01)"`.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel warping::tests::test_inner_product_l2 2>&1 | tail -20</automated>
    <fails_when>Any inner_product_l2 test fails/won't compile, trapz fails to infer T, or a caller at warping.rs:90/96/177 no longer compiles.</fails_when>
  </verify>
  <acceptance_criteria>
    - inner_product_l2 signature is exactly `pub fn inner_product_l2<T: Scalar = f64>(psi1: &[T], psi2: &[T], time: &[f64]) -> T`.
    - All three tests pass; parity is bit-identical.
    - The commit touches ONLY helpers.rs, utility.rs, warping.rs; warping.rs callers at :90/:96/:177 are unedited.
  </acceptance_criteria>
  <done>All four Phase-95 kernels are now generic over T; inner_product_l2 delegates to generic trapz; f64 parity + Dual/Var gradients validated; the three kernel files committed with no call-site churn.</done>
  <reversibility rating="reversible">Additive in-place generalization preserving the f64 path.</reversibility>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No attack surface — in-place generalization of pure numeric kernels over caller-provided f64/scalar slices; no I/O, network, deserialization, or privilege boundary.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-95-02 | Tampering | trapz/inner_product/inner_product_l2 numeric results | low | accept | f64-parity tests assert bit-identical output at T=f64; the accumulator-loop rewrite of inner_product preserves value semantics; no external input crosses any boundary. |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel helpers::tests::test_trapz utility::tests::test_inner_product warping::tests::test_inner_product_l2` — all green.
- `git diff --name-only HEAD` (for this plan's commit) — only helpers.rs, utility.rs, warping.rs.
</verification>

<success_criteria>
trapz, inner_product, inner_product_l2 are generic over `T: Scalar = f64`; f64 parity is bit-identical; Dual (and Var for inner_product_l2) gradients match central FD within 1e-5; every existing f64 caller and simpsons_weights are unchanged.
</success_criteria>

<output>
Create `.planning/phases/95-generic-scalar-hot-path-signatures/95-02-SUMMARY.md` when done.
</output>
