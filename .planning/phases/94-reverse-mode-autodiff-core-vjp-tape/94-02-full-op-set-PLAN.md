---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: 02
type: execute
wave: 2
depends_on: [94-01]
files_modified:
  - fdars-core/src/autodiff/reverse.rs
autonomous: true
requirements: [RAD-01]
estimate:
  tokens: 65000
  raw_tokens: 32000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "Var supports the full forward-mode operation set: ±, ×, ÷, neg, sqrt, exp, ln, sin, cos, powf, abs, signum, and AddAssign/SubAssign/MulAssign (RAD-01)"
    - "Var fully implements the `Scalar` trait (all 12 methods + all supertrait ops) with no remaining unimplemented!() stubs"
    - "Every op has a Tier-1 known-answer backward test at tolerance 1e-10"
    - "Singular points (sqrt/ln/powf at 0) propagate NaN/Inf adjoints, not panics, matching forward-mode Dual"
    - "abs uses the subdifferential convention (0 at 0) and signum has zero gradient everywhere, matching Dual"
  artifacts:
    - "fdars-core/src/autodiff/reverse.rs — complete `impl Scalar for Var` + all op impls + Tier-1 & Tier-2 tests"
  key_links:
    - "Every binary op short-circuits const*const to a SENTINEL result; every unary op short-circuits a SENTINEL input"
    - "AddAssign/SubAssign/MulAssign delegate to the binary ops (`*self = *self OP rhs`) so push_binary runs and node updates"
    - "Div handles all three constant cases (both const, RHS const, LHS const) with correct quotient-rule partials"
---

<objective>
Expand the tracer into the full differentiable operation set on `Var`: all arithmetic ops (Add/Sub/Div/Neg, and the assign-ops), all transcendentals (sqrt/exp/ln/sin/cos/powf/abs/signum), completing the `Scalar for Var` impl so `Var` is a drop-in scalar type. Add Tier-1 known-answer backward tests (1e-10) for every op and Tier-2 singular-point guard tests, mirroring the forward-mode `Dual` test tiers exactly.

Purpose: RAD-01 requires the reverse-mode tape to support the same operation set as `Dual`. After this plan, `Var: Scalar` is complete, which is the precondition for the already-generic `soft_dtw_distance_generic<Var>` and `project_scores_generic<Var>` to instantiate (validated in Plan 04).

Output: a fully-implemented `Scalar for Var` with per-op known-answer and singular-point tests, all green.
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
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-01-SUMMARY.md
</context>

<build_and_commit_constraints>
HARD project hazards (from MEMORY.md) — apply to EVERY task:
- Pre-commit hook times out / gets killed — commit with `git commit --no-verify`.
- Run `cargo fmt` BEFORE each commit (--no-verify skips the fmt hook).
- Gates OUT-OF-BAND, per-gate, FOREGROUND, 600s timeout — never one combined backgrounded gate.
- Clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- `rm -rf target/debug/{incremental,examples}` if `/home` is tight.
- Keep `--features serde` build green. NO new crate dependency.
</build_and_commit_constraints>

<artifacts_this_phase_produces>
This plan completes: all `Add`/`Sub`/`Div`/`Neg`/`AddAssign`/`SubAssign`/`MulAssign` impls for `Var`, and the full `impl Scalar for Var` transcendentals (`sqrt`/`exp`/`ln`/`sin`/`cos`/`powf`/`abs`/`signum`), replacing the Plan 01 `unimplemented!()` stubs.
</artifacts_this_phase_produces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement all arithmetic ops on Var (Add, Sub, Div, Neg, assign-ops)</name>
  <files>fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/autodiff/reverse.rs (from Plan 01 — Var, Node, SENTINEL, push_binary, push_unary, the existing Mul impl as the template, and the `unimplemented!()` op stubs to replace)
    - fdars-core/src/autodiff/forward.rs (Dual's Add/Sub/Div/Neg impls and the assign-op impls at ~322–341 which delegate `*self = *self OP rhs` — mirror the delegation exactly)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md (Pitfall 4 assign-op delegation; Open Question 2 — Div's three constant cases; Pitfall 3 sentinel guard)
  </read_first>
  <behavior>
    - Add: `vjp(|x| x[0] + x[0], &[2.0])` → gradient [2.0]; partials ∂(u+v)/∂u = 1, ∂/∂v = 1.
    - Sub: `vjp(|x| x[0] - x[0]*Scalar::from_f64(0.0), ...)` sanity; ∂(u-v)/∂u = 1, ∂/∂v = -1.
    - Div: `vjp(|x| x[0] / x[0], &[3.0])` value 1.0, gradient 0.0 (u/u); and RHS-constant `x/c` gradient 1/c; quotient rule ∂(u/v)/∂u = 1/v, ∂/∂v = -u/v^2.
    - Neg: `vjp(|x| -(x[0]*x[0]), &[3.0])` gradient [-6.0].
    - AddAssign/SubAssign/MulAssign propagate gradient (must call push_binary via delegation).
  </behavior>
  <action>
    Replace the Plan 01 `unimplemented!()` stubs in `reverse.rs` with real impls, mirroring the `Mul for Var` pattern and the `Dual` structural analogs:
    - `Add`: value = self.value + rhs.value; const+const → SENTINEL; else `push_binary(self.node, 1.0, rhs.node, 1.0)`.
    - `Sub`: value = self.value - rhs.value; const-const → SENTINEL; else `push_binary(self.node, 1.0, rhs.node, -1.0)`.
    - `Div`: value = self.value / rhs.value. Handle all three constant cases (94-RESEARCH Open Question 2): both SENTINEL → SENTINEL constant; RHS SENTINEL only → `push_unary(self.node, 1.0 / rhs.value)`; else `push_binary(self.node, 1.0 / rhs.value, rhs.node, -self.value / (rhs.value * rhs.value))` (quotient rule). (LHS-SENTINEL-only still falls through the general binary push with self.node = SENTINEL, which the sentinel guard handles — but pushing via push_binary with dep0 = SENTINEL is harmless; prefer the explicit RHS-const branch for the common `1/gamma` softmin case.)
    - `Neg`: value = -self.value; SENTINEL → SENTINEL; else `push_unary(self.node, -1.0)`.
    - `AddAssign`/`SubAssign`/`MulAssign`: delegate `*self = *self + rhs;` (resp. `-`, `*`) — do NOT hand-write value-only mutation (Pitfall 4: skips the tape push). Do NOT implement `DivAssign` — the `Scalar` supertrait bounds require only `AddAssign + SubAssign + MulAssign` (Pitfall 7).
    Every op guards SENTINEL operands before indexing (Pitfall 3).

    Add Tier-1 known-answer tests to `#[cfg(test)] mod tests` for Add, Sub, Div (incl. RHS-constant), Neg, and one assign-op, each asserting value and gradient within `TOL` (1e-10) per the behavior block.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff::reverse 2>&1 | tail -20</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" in output, or any arithmetic-op gradient off by more than 1e-10 from the known answer</fails_when>
  </verify>
  <acceptance_criteria>
    - Add/Sub/Div/Neg impls record correct partials and short-circuit constant operands.
    - AddAssign/SubAssign/MulAssign delegate to binary ops (gradient flows).
    - No `DivAssign` impl (not required by Scalar).
    - Per-op Tier-1 tests pass at 1e-10.
  </acceptance_criteria>
  <done>All arithmetic ops on Var are implemented with correct backward partials and constant-folding, each proven by a known-answer test.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Complete Scalar for Var transcendentals + Tier-2 singular-point guards</name>
  <files>fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/autodiff/reverse.rs (from Task 1 — full arithmetic; the constant constructors from Plan 01; the transcendental `unimplemented!()` stubs to replace)
    - fdars-core/src/autodiff/forward.rs (Dual's `impl Scalar` transcendentals at ~394–468: sqrt@394, abs@444 subdifferential, signum@459 zero-tangent — the exact local-partial formulas to reuse; and the Tier-2 guard tests ~847–889)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-PATTERNS.md (transcendental mirror examples: sqrt/abs/signum; Tier-2 guard shape checking `grad[0].is_infinite()`/`.is_nan()`)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md (Op Push Patterns — sqrt example; "Singular points match Dual behavior exactly, no clamping")
  </read_first>
  <behavior>
    - sqrt: `vjp(|x| Scalar::sqrt(x[0]), &[4.0])` → (2.0, [0.25]); local partial 1/(2√v).
    - exp: at x=1 → (e, [e]); partial exp(v).
    - ln: at x=2 → (ln 2, [0.5]); partial 1/v.
    - sin/cos: at x → partials cos(v) / -sin(v).
    - powf(x, 3.0): at x=2 → (8, [12]); partial p·v^(p-1) w.r.t. base.
    - abs: at x=-3 → (3, [-1]); subdifferential 0 at 0.
    - signum: gradient 0 everywhere.
    - Tier-2 guards: sqrt(0) adjoint non-finite (Inf), ln(0) adjoint non-finite, powf(0, 0.5) adjoint non-finite — NaN/Inf propagated, NOT a panic.
  </behavior>
  <action>
    Replace the transcendental `unimplemented!()` stubs in `impl Scalar for Var` with real impls, each computing the forward value and pushing ONE unary node with the local partial (reusing the exact formulas from Dual in forward.rs); each short-circuits a SENTINEL input to a SENTINEL result:
    - `sqrt`: s = value.sqrt(); partial 1.0/(2.0*s).
    - `exp`: e = value.exp(); partial e.
    - `ln`: l = value.ln(); partial 1.0/value.
    - `sin`: partial value.cos().
    - `cos`: partial -value.sin().
    - `powf(p)`: v = value.powf(p); partial p * value.powf(p - 1.0) (w.r.t. the base; exponent is an f64 arg, not a Var).
    - `abs`: sub = if value == 0.0 { 0.0 } else { value.signum() }; partial sub (subdifferential convention, matches Dual@444).
    - `signum`: value.signum() with a zero-gradient result — return a SENTINEL constant (gradient 0 everywhere, matching Dual@459; no tape node needed).
    Do NOT clamp singular points — let 1/(2·0)=Inf, 1/0=Inf propagate exactly as Dual does.

    Confirm the `impl Scalar for Var` now has zero `unimplemented!()` remaining. Add Tier-1 known-answer tests for each transcendental (1e-10) and Tier-2 singular-point guard tests (assert `grad[0].is_infinite()` or `!grad[0].is_finite()` for sqrt(0)/ln(0)/powf(0,0.5); assert the call does not panic) per the behavior block.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff::reverse 2>&1 | tail -25</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" / "panicked" in output, or any transcendental gradient off by more than 1e-10, or a singular-point test observing a finite adjoint where Inf is expected</fails_when>
  </verify>
  <acceptance_criteria>
    - All 8 transcendentals implemented with correct unary partials; SENTINEL inputs short-circuit.
    - Zero `unimplemented!()` remain in `impl Scalar for Var` or the op impls.
    - `abs` subdifferential (0 at 0); `signum` zero gradient.
    - Tier-2 guards confirm Inf/NaN adjoint propagation without panic on sqrt/ln/powf at 0.
    - `cargo check -p fdars-core --features linalg` succeeds (Var is a complete Scalar).
  </acceptance_criteria>
  <done>Var fully implements Scalar with the complete op set; every op has a known-answer test at 1e-10 and singular points propagate non-finite adjoints exactly like Dual.</done>
</task>

</tasks>

<threat_model>
No attack surface — pure numerical library code operating on caller-provided f64 slices; no I/O, network, deserialization, filesystem, auth, or privilege boundary. The only correctness hazards are numerical (NaN/Inf adjoints at singular points) and are covered by the Tier-2 guard tests in this plan, not by security controls.

| Boundary | Description |
|----------|-------------|
| (none) | No trust boundary crossed — in-crate numeric computation on f64 slices |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel autodiff::reverse` — all Tier-1 + Tier-2 reverse tests pass.
- `cargo check -p fdars-core --features linalg` — Var: Scalar is complete.
- Out-of-band gates: `cargo fmt --check`; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- `git diff Cargo.toml fdars-core/Cargo.toml` empty.
</verification>

<success_criteria>
- Var supports the full Dual op set with correct backward partials; Scalar impl complete.
- Every op has a Tier-1 known-answer test (1e-10); singular points guarded (Tier-2).
- No new dependency; clippy + fmt clean.
</success_criteria>

<output>
Create `.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-02-SUMMARY.md` when done.
</output>
