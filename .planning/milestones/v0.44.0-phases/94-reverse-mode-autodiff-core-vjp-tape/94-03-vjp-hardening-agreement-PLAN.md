---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: 03
type: execute
wave: 3
depends_on: [94-02]
files_modified:
  - fdars-core/src/autodiff/reverse.rs
autonomous: true
requirements: [RAD-02]
estimate:
  tokens: 50000
  raw_tokens: 25000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "vjp seeds the output adjoint and accumulates all input gradients in ONE backward sweep, efficient for many-input→scalar objectives (RAD-02)"
    - "vjp is panic-safe across calls via the double-clear tape lifecycle (clear before AND after)"
    - "vjp handles edge cases: empty input, constant-only closure, single input, and repeated calls on the same closure without gradient drift"
    - "For every op, vjp's gradient agrees with forward-mode grad's gradient within 1e-10 (Tier-3 reverse-vs-Dual agreement)"
  artifacts:
    - "fdars-core/src/autodiff/reverse.rs — hardened vjp lifecycle + Tier-3 agreement tests + Tier-5 edge-case tests"
  key_links:
    - "vjp clears the tape both before the forward pass and after reading adjoints (Pitfall 1 panic-safety)"
    - "The many-input→scalar backward sweep accumulates every input's adjoint in a single reverse iteration (the efficiency claim vs grad's m forward passes)"
---

<objective>
Harden the `vjp` entry point for RAD-02: guarantee the double-clear tape lifecycle is panic-safe, confirm the single-backward-sweep many-input→scalar accumulation, and lock the behavior with Tier-3 reverse-vs-`Dual` agreement tests (every op, 1e-10) and Tier-5 edge-case tests (empty input, constant-only closure, single input, repeated calls). This closes RAD-02: the backward pass seeds the output adjoint and accumulates input gradients through the `vjp` entry point.

Purpose: The tracer proved one path works; this plan proves `vjp` is robust across the full op set and the tricky lifecycle/edge cases (tape leakage, all-zero-gradient traps) that the research flagged as the real risk. Reverse-vs-Dual agreement is the first half of RAD-03's cross-validation (the FD half lands in Plan 04).

Output: a hardened, edge-case-tested `vjp` with per-op reverse-vs-Dual agreement.
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
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-02-SUMMARY.md
</context>

<build_and_commit_constraints>
HARD project hazards (from MEMORY.md) — apply to EVERY task:
- Pre-commit hook times out / gets killed — commit with `git commit --no-verify`.
- Run `cargo fmt` BEFORE each commit.
- Gates OUT-OF-BAND, per-gate, FOREGROUND, 600s timeout.
- Clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- `rm -rf target/debug/{incremental,examples}` if `/home` is tight.
- Keep `--features serde` build green. NO new crate dependency.
</build_and_commit_constraints>

<artifacts_this_phase_produces>
This plan finalizes: the hardened `vjp` lifecycle (double-clear, edge-case handling) — no new public symbols beyond what Plan 01 introduced; the surface of `vjp` is unchanged (`#[must_use] pub fn vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)`).
</artifacts_this_phase_produces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Harden vjp lifecycle + Tier-5 edge-case tests</name>
  <files>fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/autodiff/reverse.rs (from Plan 02 — the existing `vjp` body from Plan 01 and the full op set; verify the double-clear is already present, add edge-case robustness)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md (Pitfall 1 tape leakage / double-clear; Pitfall 2 real leaf nodes; the "Backward Pass" 7-step reference; the empty-input guard mirroring grad)
  </read_first>
  <behavior>
    - Empty input: `vjp(|_| Scalar::from_f64(5.0), &[])` → (5.0, empty vec), no panic.
    - Constant-only closure: `vjp(|_x| Scalar::from_f64(2.0), &[1.0, 2.0])` → value 2.0, gradient [0.0, 0.0] (output.node == SENTINEL).
    - Single input: `vjp(|x| x[0] * x[0] * x[0], &[2.0])` → (8.0, [12.0]).
    - Repeated calls: calling the SAME closure via vjp 3 times in a row yields the identical gradient each time (no drift → proves no tape leakage).
    - Many-input→scalar: `vjp(|x| x[0]*x[1] + x[2], &[2.0,3.0,4.0])` → value 10.0, gradient [3.0, 2.0, 1.0] — all three accumulated in one sweep.
  </behavior>
  <action>
    Review and harden the `vjp` body in `reverse.rs`:
    - Confirm the tape is cleared at the START (before seeding leaves) AND at the END (after collecting the gradient) — the double-clear for panic-safety (Pitfall 1). If Plan 01's tracer omitted either clear, add it now.
    - Confirm the empty-input guard returns `(f(&[]).value, Vec::new())` without touching the tape.
    - Confirm the constant-only case: when `output.node == SENTINEL`, the adjoint seed is skipped and every input's gradient is 0.0 (the leaf nodes were pushed but never consumed).
    - Confirm inputs are seeded as REAL leaf nodes via `push_binary(SENTINEL, 0.0, SENTINEL, 0.0)`, never via `from_f64` (Pitfall 2).
    - Confirm the backward sweep accumulates ALL input adjoints in one `for i in (0..n).rev()` pass — this is the many-input→scalar efficiency win over grad's m forward passes (RAD-02).

    Add Tier-5 edge-case tests to `#[cfg(test)] mod tests` per the behavior block: empty input, constant-only closure, single input (cube), repeated-call stability (assert three successive gradients are bit-identical), and the many-input→scalar accumulation. Region-scope any negative assertions to the specific test.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff::reverse 2>&1 | tail -25</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" / "panicked" in output, or the repeated-call test showing gradient drift between calls (tape leakage), or the constant-only test returning a non-zero gradient</fails_when>
  </verify>
  <acceptance_criteria>
    - vjp clears the tape before AND after; empty/constant-only/single-input/repeated-call cases all handled.
    - Many-input→scalar gradient accumulated in one backward sweep; correct on the 3-input product+sum example.
    - Repeated vjp calls on the same closure produce identical gradients (no drift).
  </acceptance_criteria>
  <done>vjp is panic-safe and correct across all edge cases; the single-sweep many-input→scalar accumulation is proven, satisfying RAD-02's efficiency and correctness claims.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Tier-3 reverse-vs-Dual agreement tests for every op</name>
  <files>fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/autodiff/reverse.rs (from Task 1 — the full op set and hardened vjp)
    - fdars-core/src/autodiff/forward.rs (the `grad` entry point — the reverse gradient must match `grad`'s on the same closure shape)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md ("Reverse-vs-Dual Agreement Test Pattern" — grad(|d:&[Dual]|...) vs vjp(|v:&[Var]|...), assert both value and gradient agree at 1e-10)
  </read_first>
  <behavior>
    - For each op (mul, add, sub, div, neg, sqrt, exp, ln, sin, cos, powf, abs, and a mixed composed chain), build the SAME scalar closure once as `|d: &[Dual]| ...` and once as `|v: &[Var]| ...`, evaluate `grad(fwd, x)` and `vjp(rev, x)` at the same point x, and assert `(fwd_val - rev_val).abs() < 1e-10` and each `(fwd_grad[k] - rev_grad[k]).abs() < 1e-10`.
    - Include at least one multi-input composed closure (e.g. `sin(x0)*exp(x1) + sqrt(x2)`) to exercise cross-op agreement.
  </behavior>
  <action>
    Add a Tier-3 test group to `#[cfg(test)] mod tests` in `reverse.rs` following the "Reverse-vs-Dual Agreement" pattern from 94-RESEARCH.md. For each op and one mixed composed closure, write the closure twice (Dual and Var forms — the bodies are structurally identical since both are `Scalar`), run `grad` and `vjp` at a fixed in-domain point (avoid singular points), and assert value+gradient agreement at `TOL` (1e-10). Import `Dual` and `grad` from `super::super::forward` or via `crate::autodiff::{Dual, grad}` as the existing tests do. Keep points in the interior of each op's domain (positive args for sqrt/ln/powf).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff 2>&1 | tail -25</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" in output, or any reverse-vs-Dual disagreement exceeding 1e-10</fails_when>
  </verify>
  <acceptance_criteria>
    - Every op has a reverse-vs-Dual agreement test passing at 1e-10.
    - At least one multi-input composed closure agrees value+gradient at 1e-10.
    - Both forward and reverse tests coexist green under `cargo test ... autodiff`.
  </acceptance_criteria>
  <done>vjp's gradients match forward-mode grad's within 1e-10 for every op and a composed chain — the reverse-vs-Dual half of RAD-03 is locked.</done>
</task>

</tasks>

<threat_model>
No attack surface — pure numerical library code operating on caller-provided f64 slices; no I/O, network, deserialization, filesystem, auth, or privilege boundary. The only hazards are numerical/lifecycle (tape leakage → corrupt gradients), covered by the repeated-call and agreement tests here, not by security controls.

| Boundary | Description |
|----------|-------------|
| (none) | No trust boundary crossed — in-crate numeric computation on f64 slices |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel autodiff` — Tier-3 agreement + Tier-5 edge cases green alongside forward tests.
- Out-of-band gates: `cargo fmt --check`; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- `git diff Cargo.toml fdars-core/Cargo.toml` empty.
</verification>

<success_criteria>
- vjp is panic-safe (double-clear) and correct on all edge cases; single-sweep many-input→scalar proven.
- Reverse-vs-Dual agreement at 1e-10 for every op + a composed closure.
- No new dependency; clippy + fmt clean.
</success_criteria>

<output>
Create `.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-03-SUMMARY.md` when done.
</output>
