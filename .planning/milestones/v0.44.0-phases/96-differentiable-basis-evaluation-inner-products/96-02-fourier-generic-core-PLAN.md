---
phase: 96-differentiable-basis-evaluation-inner-products
plan: 02
type: execute
wave: 2
depends_on: ["96-01"]
files_modified:
  - fdars-core/src/basis/fourier.rs
  - fdars-core/src/basis/tests.rs
autonomous: true
requirements: [DOP-01]

estimate:
  tokens: 50000
  raw_tokens: 25000
  tasks: 3
  confidence: high

must_haves:
  truths:
    - "Fourier basis evaluation is generic over `Scalar` via a NEW additive core `fourier_basis_eval<T: Scalar>(t: &[T], nbasis, period: f64, t_min: f64) -> Vec<T>`; at `f64` it reproduces the current `fourier_basis_with_period` numerics bit-for-bit (assert_eq! parity) (DOP-01 criterion #1)."
    - "A combined objective (Fourier eval at `t` → generic `inner_product` against a fixed curve → scalar) is differentiable w.r.t. `t`; gradients match central finite differences within 1e-6 at both `Dual` and reverse-mode `Var(vjp)` (DOP-01 criterion #3)."
  artifacts:
    - "fdars-core/src/basis/fourier.rs — NEW generic core `fourier_basis_eval<T: Scalar>`; `fourier_basis` and `fourier_basis_with_period` refactored to delegate at f64 with their public signatures UNCHANGED"
    - "fdars-core/src/basis/tests.rs — new fourier f64-parity test + combined-objective Dual FD test + Var(vjp) FD test"
  key_links:
    - "fourier_basis_with_period (f64, signature unchanged) derives t_min as today and delegates to fourier_basis_eval — zero caller churn, public API frozen."
    - "fourier_basis_eval<T> is the generic-over-Scalar basis evaluation DOP-01 requires; it feeds a T-valued basis matrix into crate::utility::inner_product<T>."
  prohibitions:
    - statement: "Public signatures of fourier_basis and fourier_basis_with_period MUST NOT change (generic work lives in the new additive fourier_basis_eval core)."
    - statement: "No new crate dependency."
---

<objective>
Add a NEW additive generic core `fourier_basis_eval<T: Scalar>` to fdars-core/src/basis/fourier.rs and refactor the two existing f64 wrappers (`fourier_basis`, `fourier_basis_with_period`) to delegate to it — WITHOUT changing their public signatures. This is the ADDITIVE approach mandated by the phase design override: it avoids threading a `t_min` parameter into the public `fourier_basis_with_period` (which would churn 4+ external call sites). Add f64-parity (`assert_eq!`) and a combined basis-eval → inner-product objective FD-checked at both `Dual` and `Var`.

Purpose: Complete the second basis family (Fourier) generic over `Scalar`, reusing the tracer-proven basis-eval → inner-product → differentiability loop from Plan 01. Implements DOP-01.
Output: `fourier_basis_eval<T>` core + delegating f64 wrappers in fdars-core/src/basis/fourier.rs + parity/Dual/Var tests in basis/tests.rs, all committed.
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
@fdars-core/src/basis/fourier.rs
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED tests — Fourier f64-parity + combined Dual/Var FD objective referencing the new core</name>
  <files>fdars-core/src/basis/tests.rs</files>
  <read_first>
    - fdars-core/src/basis/tests.rs (lines 1-12 imports; lines 72-130 existing fourier dimension/DC-constant/sin-cos-range/period tests to preserve)
    - fdars-core/src/basis/fourier.rs (lines 22-69: current `fourier_basis` and `fourier_basis_with_period` f64 bodies)
    - fdars-core/src/helpers.rs (lines 1224-1300: `test_l2_distance_dual` / `test_l2_distance_var` seed/extract/vjp mechanics to mirror)
    - fdars-core/src/utility.rs (lines 44-55: generic `inner_product<T>`)
  </read_first>
  <behavior>
    - Test A (f64 parity): `fourier_basis_eval::<f64>(&t, nbasis, period, t_min)` equals the existing `fourier_basis_with_period(&t, nbasis, period)` output (same t/nbasis/period, with `t_min` = min(t) as the wrapper derives it) via `assert_eq!` (bit-identical).
    - Test B (Dual forward FD): inline objective `fn obj<T: Scalar>(t: &[T], nbasis, period: f64, t_min: f64, curve: &[f64], argvals: &[f64]) -> T` calls `fourier_basis_eval(t, nbasis, period, t_min)`, extracts a middle harmonic column (a sin or cos column, not the DC column) as `&[T]`, lifts `curve` with `T::from_f64`, returns `inner_product(&col, &curve_lifted, argvals)`. Seed each `t[idx]` with `Dual::seed` (others `Dual::constant`); tangent matches central FD (h=1e-6) within tol 1e-6.
    - Test C (Var reverse FD): `vjp(|t: &[Var]| obj(...), &t_f64)` gradient matches per-coordinate central FD (h=1e-6) within 1e-6.
  </behavior>
  <action>
    Append three new `#[test]` functions to fdars-core/src/basis/tests.rs. Import inside each test: `use crate::autodiff::{Dual, Var, vjp, Scalar};` and reference `crate::utility::inner_product`. Pick a non-degenerate `t` grid (uniform ~8 points on [0,1]), `nbasis` odd (e.g. 5), a fixed `period` and `t_min` = min(t). Choose a harmonic column index (1 = first sin, 2 = first cos) so the gradient w.r.t. `t` is non-zero (the DC column is constant → zero gradient, avoid it for the FD objective). Use a fixed `curve` (length nbasis) and uniform `argvals` (length nbasis). These tests reference `fourier_basis_eval`, which does not exist until Task 2 — that is the intended RED state. Bit-parity uses `assert_eq!`. FD tolerance: `let tol = 1e-6 * fd.abs().max(1e-10);`. Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel basis::tests::test_fourier 2>&1 | tail -25</automated>
    <fails_when>The new fourier tests fail to COMPILE because `fourier_basis_eval` does not yet exist (intended RED). Confirm the failure is the unresolved-name / signature error for `fourier_basis_eval`, not a typo in the test bodies.</fails_when>
  </verify>
  <done>Three new fourier tests exist (f64-parity assert_eq!, combined Dual FD, combined Var vjp FD) referencing `fourier_basis_eval`; they fail to compile because the core does not yet exist (RED).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add `fourier_basis_eval<T: Scalar>` core; refactor f64 wrappers to delegate (signatures UNCHANGED) — GREEN</name>
  <files>fdars-core/src/basis/fourier.rs</files>
  <read_first>
    - fdars-core/src/basis/fourier.rs (full file: `use std::f64::consts::PI`, `fourier_basis` lines 22-27, `fourier_basis_with_period` lines 42-69 — the delegation targets)
    - .planning/phases/96-differentiable-basis-evaluation-inner-products/96-PATTERNS.md (lines 200-239 fold-f64-lift-once discipline; lines 425-428 t_min-as-explicit-f64 rule)
    - fdars-core/src/utility.rs (lines 44-55: the `T::from_f64` per-element lift idiom)
  </read_first>
  <action>
    Add `use crate::autodiff::Scalar;` at the top of fdars-core/src/basis/fourier.rs (after the existing `use std::f64::consts::PI;`). Add a NEW public generic core:
    `pub fn fourier_basis_eval<T: Scalar>(t: &[T], nbasis: usize, period: f64, t_min: f64) -> Vec<T>`.
    Its body is the current `fourier_basis_with_period` loop with these lifts: `let n = t.len();` unchanged; `vec![0.0; n * nbasis]` → `vec![T::zero(); n * nbasis]`; precompute the f64 scale ONCE outside the loop `let scale = 2.0 * PI / period;`; per point `let x = T::from_f64(scale) * (ti - T::from_f64(t_min));` (fold the f64 sub-expression before the single lift); DC term `basis[i] = T::one();`; harmonics `basis[i + k * n] = (T::from_f64(freq as f64) * x).sin();` and `.cos();`. Keep the exact `while k < nbasis` / `freq` loop structure so column ordering and count are identical.
    Then refactor the two f64 wrappers to DELEGATE, keeping their EXACT current public signatures:
    - `pub fn fourier_basis_with_period(t: &[f64], nbasis: usize, period: f64) -> Vec<f64>`: derive `t_min` exactly as today (`t.iter().copied().fold(f64::INFINITY, f64::min)`) then `return fourier_basis_eval(t, nbasis, period, t_min);` (T=f64 inferred).
    - `pub fn fourier_basis(t: &[f64], nbasis: usize) -> Vec<f64>`: unchanged — it already delegates to `fourier_basis_with_period` (which now delegates onward). Leave its body as-is.
    Do NOT add `t_min` to any public signature. Re-export the new core in basis/mod.rs is OPTIONAL and NOT required for DOP-01 (the generic core is reached in-crate); do not touch lib.rs re-exports. Basis eval is infallible (no Result). No new dependency. Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core && cargo test -p fdars-core --features linalg,parallel basis::tests::test_fourier 2>&1 | tail -25</automated>
    <fails_when>The fourier f64-parity `assert_eq!` fails (the scale precompute or the `ti - t_min` fold changed f64 numerics vs the original `2.0 * PI * (ti - t_min) / period`), OR a Dual/Var FD assertion fails (a mis-lifted `freq`/`scale`/`t_min` constant), OR the crate fails to compile.</fails_when>
  </verify>
  <done>`fourier_basis_eval<T>` exists; f64 wrappers delegate with unchanged public signatures; the fourier parity + Dual + Var FD tests pass.</done>
</task>

<task type="auto">
  <name>Task 3: Confirm Fourier public signatures + callers unchanged, existing fourier regression tests pass, fmt, no-verify commit</name>
  <files>fdars-core/src/basis/fourier.rs, fdars-core/src/basis/tests.rs</files>
  <read_first>
    - .planning/phases/96-differentiable-basis-evaluation-inner-products/96-PATTERNS.md (lines 284-293 `fourier_basis` callers unchanged; lines 373-380 existing fourier regression guards)
  </read_first>
  <action>
    Do NOT edit any caller. Confirm via grep that `fourier_basis_with_period` and `fourier_basis` public signatures are byte-identical to the pre-change forms (only their bodies changed to delegate) and that all their callers (seasonal/strength.rs:54, smooth_basis.rs:1042, seasonal/period.rs:185, basis/projection.rs:68, basis/auto_select.rs:93, basis/fourier_fit.rs:66, elastic_regression/scalar_on_shape.rs:116) compile unchanged. Confirm existing fourier regression guards pass: `test_fourier_basis_dimensions`, `test_fourier_basis_constant_first_column`, `test_fourier_basis_sin_cos_range`, `test_fourier_basis_with_period`, `test_fourier_basis_period_affects_frequency` (NOTE: these existing tests call `fourier_basis_with_period(&t, nbasis, period)` with THREE args — the signature is unchanged, so they need NO edit). Run `cargo fmt -p fdars-core`; commit with `git commit --no-verify` (remove stale `.git/index.lock` if present; on "No space left" run `rm -rf target/debug/{incremental,examples}` and retry). Commit message: `feat(96): add generic Fourier basis-eval core, delegate f64 wrappers (DOP-01)`. Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel basis:: 2>&1 | tail -20 && grep -nE "pub fn fourier_basis\(|pub fn fourier_basis_with_period\(" fdars-core/src/basis/fourier.rs</automated>
    <fails_when>Any existing fourier `basis::` test fails (regression in f64 numerics), OR the grep shows `fourier_basis`/`fourier_basis_with_period` gained a parameter (public signature changed — violates the non-breaking prohibition), OR the crate does not compile.</fails_when>
  </verify>
  <done>All basis:: tests pass; `fourier_basis`/`fourier_basis_with_period` public signatures unchanged; all callers compile unchanged; committed with `--no-verify` after `cargo fmt`.</done>
</task>

</tasks>

<threat_model>
No attack surface — pure numeric Fourier basis-evaluation code over caller-provided f64/scalar slices. No external input, network, IO, auth, or untrusted data. No new crate dependency.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-96-02 | Tampering | in-crate numeric kernel | low | accept | Pure deterministic numeric transform; f64-parity `assert_eq!` guards against unintended numeric change. |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel basis::` — all fourier tests pass (parity, Dual FD, Var FD, and existing regression guards).
- grep confirms `fourier_basis` / `fourier_basis_with_period` public signatures unchanged.
</verification>

<success_criteria>
- `fourier_basis_eval<T: Scalar>` is the generic Fourier evaluation core.
- f64 wrappers delegate with public signatures frozen (zero caller churn).
- f64 parity bit-identical (`assert_eq!`); combined objective gradients match central FD within 1e-6 at both `Dual` and `Var(vjp)`.
- Committed with `--no-verify` after `cargo fmt`.
</success_criteria>

<output>
Create `.planning/phases/96-differentiable-basis-evaluation-inner-products/96-02-SUMMARY.md` when done.
</output>
