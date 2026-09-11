---
phase: 96-differentiable-basis-evaluation-inner-products
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/basis/bspline.rs
  - fdars-core/src/basis/tests.rs
autonomous: true
requirements: [DOP-01]

estimate:
  tokens: 55000
  raw_tokens: 28000
  tasks: 3
  confidence: high

must_haves:
  truths:
    - "B-spline basis evaluation (`bspline_basis_from_knots` + `evaluate_order_zero` + `bspline_recurrence_step`) is generic over `Scalar`; at `f64` it reproduces current numerics bit-for-bit (assert_eq! parity) and preserves partition-of-unity within 1e-10 (DOP-01 criterion #1)."
    - "A combined objective (B-spline eval at `t` → generic `inner_product` against a fixed curve → scalar) is differentiable w.r.t. `t`; gradients match central finite differences within 1e-6 at both `Dual` and reverse-mode `Var(vjp)` (DOP-01 criterion #3, tracer-proves the whole loop)."
    - "Every existing `bspline_basis_from_knots` caller compiles unchanged at inferred `T = f64` (helpers.rs spline_interpolate, basis/pspline.rs)."
  artifacts:
    - "fdars-core/src/basis/bspline.rs — generic `bspline_basis_from_knots<T: Scalar>`, `evaluate_order_zero<T: Scalar>`, `bspline_recurrence_step<T: Scalar>`"
    - "fdars-core/src/basis/tests.rs — new bspline f64-parity test + combined-objective Dual FD test + Var(vjp) FD test"
  key_links:
    - "bspline_basis_from_knots<T> feeds a T-valued basis matrix into crate::utility::inner_product<T> — the tracer path proving basis-eval → inner-product → scalar differentiability."
    - "f64::from_f64 is the identity → the generic path at T=f64 is bit-identical to the pre-change f64 path (assert_eq! parity holds)."
---

<objective>
TRACER: generalize the B-spline basis evaluation path in place over `T: Scalar`, proving the whole generic-basis-eval + differentiability loop end-to-end on B-spline before Fourier (Plan 02) or the gate (Plan 03). Generalize the three core evaluators (`bspline_basis_from_knots`, `evaluate_order_zero`, `bspline_recurrence_step`), add f64-parity (`assert_eq!`), confirm the existing partition-of-unity guard still passes, and add a combined objective (basis-eval → `inner_product` → scalar) finite-difference-checked at both forward `Dual` and reverse `Var`.

Purpose: This is the thinnest slice that touches every layer the phase modifies — generic basis kernel, the f64/generic boundary discipline, the parity guard, and both autodiff modes wired end-to-end through a real inner product. Proving it on B-spline catches any architectural dead-end after one commit. Implements DOP-01.
Output: Generic B-spline evaluators in fdars-core/src/basis/bspline.rs + parity/Dual/Var tests in basis/tests.rs, all committed.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/96-differentiable-basis-evaluation-inner-products/96-CONTEXT.md
@.planning/phases/96-differentiable-basis-evaluation-inner-products/96-PATTERNS.md
@fdars-core/src/basis/bspline.rs
@fdars-core/src/helpers.rs
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end B-spline generic slice — RED tests first (parity + combined Dual/Var FD objective)</name>
  <files>fdars-core/src/basis/tests.rs</files>
  <read_first>
    - fdars-core/src/basis/tests.rs (lines 1-130: imports `use super::*`, `use crate::matrix::FdMatrix`, existing bspline dimension/partition-of-unity/non-negative/boundary tests to mirror)
    - fdars-core/src/helpers.rs (lines 1224-1300: `test_l2_distance_dual` and `test_l2_distance_var` — the exact Dual::seed/Dual::constant/.extract() forward pattern and the `vjp(|x: &[Var]| ..., &t_f64)` reverse pattern with per-coordinate central FD)
    - fdars-core/src/utility.rs (lines 44-55: `inner_product<T: Scalar>(c1: &[T], c2: &[T], argvals: &[f64]) -> T` — the generic inner product this objective composes against)
    - fdars-core/src/basis/bspline.rs (lines 4-83: `construct_bspline_knots`, `bspline_basis_from_knots`, `evaluate_order_zero`, `bspline_recurrence_step` current f64 bodies)
  </read_first>
  <behavior>
    - Test A (f64 parity): a hardcoded reference `Vec<f64>` (the pre-change `bspline_basis_from_knots` output for a fixed t/knots/order, captured as a literal or recomputed once) equals `bspline_basis_from_knots::<f64>(&t, &knots, order)` via `assert_eq!` (bit-identical).
    - Test B (Dual forward FD): define an inline objective `fn obj<T: Scalar>(t: &[T], knots: &[f64], order: usize, curve: &[f64], argvals: &[f64]) -> T` that calls `bspline_basis_from_knots(t, knots, order)` (T-valued matrix), extracts one column j as `&[T]`, lifts `curve` with `T::from_f64`, and returns `inner_product(&col, &curve_lifted, argvals)`. Seed each `t[idx]` in turn with `Dual::seed`, others `Dual::constant`; the extracted tangent matches central FD (h=1e-6) of the f64 objective within tol 1e-6.
    - Test C (Var reverse FD): `vjp(|t: &[Var]| obj(t, &knots, order, &curve, &argvals), &t_f64)` returns a gradient whose every coordinate matches per-coordinate central FD (h=1e-6) within tol 1e-6.
  </behavior>
  <action>
    Write three new `#[test]` functions in fdars-core/src/basis/tests.rs, appended near the existing bspline tests. Follow the Dual/Var analog in helpers.rs verbatim for the seed/extract/vjp mechanics. Reference the generic `inner_product` as `crate::utility::inner_product`; import `Dual`, `Var`, `vjp`, `Scalar` inside each test via `use crate::autodiff::{Dual, Var, vjp, Scalar};`. Use `construct_bspline_knots` to build knots and a non-degenerate `t` grid (e.g. uniform 8 points on [0,1], order 4) so basis columns are non-trivial. Choose a middle basis column index j (not an all-zero boundary column) for the objective so gradients are non-zero. Use a fixed pseudo-arbitrary `curve` (length = nbasis) and `argvals` (length = nbasis, uniform) as the inner-product operands. These tests MUST reference the generic call form `bspline_basis_from_knots::<f64>(...)` / `bspline_basis_from_knots(&t_dual, ...)`, which does not compile until Task 2 lands `<T: Scalar>` — that is the RED state. Implements DOP-01.
    Do NOT place any fenced code block in this file's non-test prose; write real Rust test bodies. Bit-parity uses `assert_eq!` (not abs-diff). FD tolerance follows the Phase 95 convention: `let tol = 1e-6 * fd.abs().max(1e-10);` then `assert!((grad - fd).abs() <= tol, ...)`.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel basis::tests::test_bspline 2>&1 | tail -25</automated>
    <fails_when>The new bspline parity/Dual/Var tests fail to COMPILE (expected RED at this task: `bspline_basis_from_knots` is not yet generic, so `::<f64>` / Dual / Var instantiation errors). This is the intended failing state before Task 2 — confirm the failure is the type-generic instantiation error, not a typo in the test bodies.</fails_when>
  </verify>
  <done>Three new tests exist (f64-parity assert_eq!, combined-objective Dual FD, combined-objective Var vjp FD) referencing the generic call form; they currently fail to compile because the evaluators are still f64-only (RED).</done>
</task>

<task type="tracer" tdd="true">
  <name>Task 2: Generalize the three B-spline evaluators in place over `<T: Scalar>` — GREEN</name>
  <files>fdars-core/src/basis/bspline.rs</files>
  <read_first>
    - fdars-core/src/basis/bspline.rs (lines 19-83: exact current bodies of `evaluate_order_zero`, `bspline_recurrence_step`, `bspline_basis_from_knots` — the edit targets)
    - fdars-core/src/helpers.rs (line 3 `use crate::autodiff::Scalar;` import; lines 66-73 `l2_distance<T>` for the `T::zero()`/`T::from_f64` accumulator idiom)
    - .planning/phases/96-differentiable-basis-evaluation-inner-products/96-PATTERNS.md (lines 131-150: exact generic target forms and change points; lines 418-423 division-before-multiply bit-parity rule)
  </read_first>
  <action>
    Add `use crate::autodiff::Scalar;` at the top of fdars-core/src/basis/bspline.rs. Generalize IN PLACE (no `_generic` companions, no `= f64` default on the free functions — Rust 1.97 rejects that):
    - `evaluate_order_zero<T: Scalar>(t_val: T, knots: &[f64], t_max_knot_idx: usize) -> Vec<T>`: `vec![0.0; ...]` → `vec![T::zero(); ...]`; span comparisons become `t_val >= T::from_f64(knots[j]) && t_val < T::from_f64(knots[j + 1])` (and the `<=` boundary arm likewise lifts the knot) — the `PartialOrd` is T-vs-T; the indicator `b0[j] = 1.0` → `b0[j] = T::one()`. Keep the `break` and the `j == t_max_knot_idx - 1` boundary branch structurally unchanged (discrete control flow, not differentiated).
    - `bspline_recurrence_step<T: Scalar>(b: &[T], knots: &[f64], t_val: T, k: usize) -> Vec<T>`: keep `d1` and `d2` as f64 (pure knot subtraction) and keep the `d1.abs() > 1e-10` / `d2.abs() > 1e-10` guards in f64; the `0.0` fallback arms become `T::zero()`. Left arm: `(t_val - T::from_f64(knots[j])) / T::from_f64(d1) * b[j]` — division THEN multiply, order preserved (NOT reciprocal-multiply; that breaks ULP parity). Right arm: `(T::from_f64(knots[j + k]) - t_val) / T::from_f64(d2) * b[j + 1]`.
    - `bspline_basis_from_knots<T: Scalar>(t: &[T], knots: &[f64], order: usize) -> Vec<T>`: `t: &[f64]` → `t: &[T]`, return `Vec<T>`, `vec![0.0; n * nbasis]` → `vec![T::zero(); n * nbasis]`; loop body unchanged structurally (types flow from callee signatures). Keep the column-major flatten `basis[ti + j * n] = b[j]`.
    Leave `construct_bspline_knots` and `bspline_basis` (the auto-deriving f64 wrapper, lines 100+) UNCHANGED — they stay f64-only per the locked decision. Basis eval is infallible on in-domain input (no Result). Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core && cargo test -p fdars-core --features linalg,parallel basis::tests::test_bspline 2>&1 | tail -25</automated>
    <fails_when>Any of the three Task-1 tests fails: parity `assert_eq!` fails (a reciprocal-multiply or reordered f64-fold broke bit-parity), or a Dual/Var FD assertion fails (tangent/adjoint diverges from central FD beyond 1e-6, indicating a mis-lifted constant). Also fails if the crate does not compile (generic signature error).</fails_when>
  </verify>
  <done>The three evaluators are generic over `T: Scalar`; the f64-parity test passes bit-identically (`assert_eq!`), and the combined-objective Dual and Var FD tests pass within 1e-6.</done>
</task>

<task type="auto">
  <name>Task 3: Confirm existing bspline callers + partition-of-unity regression tests unchanged, fmt, no-verify commit</name>
  <files>fdars-core/src/basis/bspline.rs, fdars-core/src/basis/tests.rs</files>
  <read_first>
    - .planning/phases/96-differentiable-basis-evaluation-inner-products/96-PATTERNS.md (lines 245-259: `bspline_basis_from_knots` callers — helpers.rs:523, helpers.rs:671, basis/pspline.rs:168 — all pass `&[f64]`, infer T=f64, no source change; lines 373-380 existing regression guards)
  </read_first>
  <action>
    Do NOT edit any caller. Confirm via grep that all `bspline_basis_from_knots` call sites pass f64 slices (T=f64 inferred) and compile unchanged. Confirm the existing regression guards still pass: `test_bspline_basis_dimensions`, `test_bspline_basis_partition_of_unity` (1e-10), `test_bspline_basis_non_negative`, `test_bspline_basis_boundary`. Run `cargo fmt -p fdars-core` before committing. Commit with `git commit --no-verify` (pre-commit hook times out at 30s; remove a stale `.git/index.lock` first if present). If a build fails with "No space left"/"linking with cc failed", run `rm -rf target/debug/{incremental,examples}` and retry. Commit message: `feat(96): generalize B-spline basis evaluation over Scalar (DOP-01 tracer)`. Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel basis:: 2>&1 | tail -20 && grep -rn "bspline_basis_from_knots" fdars-core/src/ | grep -v "src/basis/bspline.rs\|src/basis/tests.rs"</automated>
    <fails_when>Any existing basis:: test fails (partition-of-unity, non-negativity, boundary, or dimensions regression — signals the generalization changed f64 numerics), OR the grep reveals a caller outside bspline.rs/tests.rs that needed a source change (would mean the generalization was not caller-transparent).</fails_when>
  </verify>
  <done>All basis:: tests pass; every `bspline_basis_from_knots` caller compiles unchanged at inferred T=f64; changes committed with `--no-verify` after `cargo fmt`.</done>
</task>

</tasks>

<threat_model>
No attack surface — pure numeric B-spline basis-evaluation code over caller-provided f64/scalar slices. No external input, network, IO, auth, or untrusted data. No new crate dependency.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-96-01 | Tampering | in-crate numeric kernel | low | accept | Pure deterministic numeric transform; f64-parity `assert_eq!` guards against unintended numeric change. |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel basis::` — all bspline tests pass (parity, Dual FD, Var FD, and the four existing regression guards).
- grep confirms no `bspline_basis_from_knots` caller outside bspline.rs/tests.rs changed.
</verification>

<success_criteria>
- The three B-spline evaluators are generic over `T: Scalar`.
- f64 parity is bit-identical (`assert_eq!`); partition-of-unity within 1e-10.
- Combined basis-eval → inner_product objective gradients match central FD within 1e-6 at both `Dual` and `Var(vjp)`.
- No caller churn; committed with `--no-verify` after `cargo fmt`.
</success_criteria>

<output>
Create `.planning/phases/96-differentiable-basis-evaluation-inner-products/96-01-SUMMARY.md` when done.
</output>
