---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/autodiff.rs
  - fdars-core/src/autodiff/mod.rs
  - fdars-core/src/autodiff/forward.rs
  - fdars-core/src/autodiff/reverse.rs
autonomous: true
requirements: [RAD-01, RAD-02]
estimate:
  tokens: 70000
  raw_tokens: 35000
  tasks: 3
  confidence: low
must_haves:
  truths:
    - "A reverse-mode Wengert-list tape records ONE op (Mul) on a `Var` scalar type (RAD-01)"
    - "A minimal `vjp` seeds the output adjoint and runs one backward sweep, returning `(f64, Vec<f64>)` (RAD-02)"
    - "The end-to-end record→seed→backward→read-adjoint loop is proven by a known-answer test: vjp(|x| x[0]*x[0], &[3.0]) == (9.0, [6.0])"
    - "All public autodiff paths still resolve after the directory refactor (`autodiff::Dual`, `autodiff::Scalar`, `autodiff::grad`, etc.)"
    - "Existing forward-mode `Dual` tests still pass unchanged after the move"
  artifacts:
    - "fdars-core/src/autodiff/mod.rs (Scalar trait + shared re-exports)"
    - "fdars-core/src/autodiff/forward.rs (Dual + forward entry points, moved verbatim)"
    - "fdars-core/src/autodiff/reverse.rs (Var, Node, TAPE thread-local, push_binary/push_unary, minimal vjp, Mul impl, one test)"
  key_links:
    - "autodiff/mod.rs re-exports forward::* and reverse::* so lib.rs:76 `pub mod autodiff;` resolves to the directory unchanged"
    - "Var implements enough of Scalar (from_f64/zero/one/infinity + Mul) that the tracer test compiles"
    - "vjp pushes real leaf nodes (not SENTINEL constants) for inputs so adjoints accumulate"
---

<objective>
TRACER SLICE: prove the entire reverse-mode autodiff architecture end-to-end on one path before expanding. Refactor the monolithic `autodiff.rs` into an `autodiff/` directory (`mod.rs` + `forward.rs`) preserving every public path and passing all existing `Dual` tests, then stand up the `reverse.rs` skeleton — `Var`, `Node`, thread-local `TAPE`, `push_binary`/`push_unary`, exactly ONE op (`Mul`), the constant `Scalar` constructors needed to compile, and a minimal `vjp` — and prove the full record→seed→backward→read-adjoint loop with one known-answer test.

Purpose: The pivotal architectural risk (thread-local tape + Copy `Var` handle + sentinel constants + one backward sweep) is caught here, on the agent's best early-context tokens, after one commit — not after ten already-committed op impls. The tracer is production-quality: `Var`, `Tape`, `Node`, and `vjp` written here are kept and expanded, never thrown away.

Output: `autodiff/{mod.rs,forward.rs,reverse.rs}` with a green end-to-end reverse-mode multiply-gradient test and an unchanged public API.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-CONTEXT.md
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md
@.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-PATTERNS.md
</context>

<build_and_commit_constraints>
HARD project hazards (from MEMORY.md) — apply to EVERY task in this plan:
- The pre-commit hook runs the full cargo gate and TIMES OUT / gets killed mid-run. Commit with `git commit --no-verify`.
- Because `--no-verify` skips the fmt hook, run `cargo fmt` BEFORE each commit or CI fmt-check fails.
- Run gates OUT-OF-BAND, per-gate, FOREGROUND, with a 600s timeout — never one backgrounded combined gate.
- Clippy gate is `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code).
- If `/home` disk is tight, `rm -rf target/debug/{incremental,examples}` before heavy builds.
- Keep the `--features serde` build green.
- NO new crate dependency — the tape is hand-written in-crate. Adding a dep is a hard failure.
</build_and_commit_constraints>

<artifacts_this_phase_produces>
NEW symbols introduced across Phase 94 (this plan seeds the italicized subset):
- *`Var` struct* (`{ value: f64, node: usize }`, Copy), *`Node` struct* (`{ deps: [usize;2], weights: [f64;2] }`), *`Tape` = thread-local `Vec<Node>`*, *`push_binary`/`push_unary` helpers*, *`SENTINEL` const*, *minimal `vjp`*, *`autodiff/mod.rs`*, *`autodiff/forward.rs`*, *`autodiff/reverse.rs`*, *`Mul for Var`*.
- Later plans add: full op set + `Scalar for Var` (Plan 02), hardened `vjp` (Plan 03), prelude re-exports of `Var`/`vjp` (Plan 04).
</artifacts_this_phase_produces>

<tasks>

<task type="tracer">
  <name>Task 1: Refactor autodiff.rs into autodiff/{mod.rs, forward.rs} preserving all public paths</name>
  <files>fdars-core/src/autodiff.rs, fdars-core/src/autodiff/mod.rs, fdars-core/src/autodiff/forward.rs</files>
  <read_first>
    - fdars-core/src/autodiff.rs (full file, 1143 lines — the monolith being split; the `Scalar` trait is at line 110, `impl Scalar for f64` at 158, `Dual` + impls at 225–468, entry points `diff`@484 `grad`@511 `jacobian`@559 `directional_derivative`@611, tests at 629–1143)
    - fdars-core/src/lib.rs (line 76: `pub mod autodiff;` — MUST stay byte-identical; Cargo resolves it to `autodiff/mod.rs` automatically)
    - fdars-core/src/classification/mod.rs (lines 1–30: barrel `//!` doc + `pub mod` + `pub use` re-export style to mirror)
    - fdars-core/src/depth/mod.rs (lines 29–31: `pub use` re-export style)
  </read_first>
  <action>
    Split the monolithic `fdars-core/src/autodiff.rs` into a directory module. Use `git mv fdars-core/src/autodiff.rs fdars-core/src/autodiff/forward.rs` first to preserve history, then create `fdars-core/src/autodiff/mod.rs`.

    In `autodiff/mod.rs`: add a module-level `//!` doc summarizing forward-mode (Dual) and reverse-mode (Var) AD. Declare `pub mod forward;` and `pub mod reverse;`. MOVE the `Scalar` trait definition (currently forward.rs lines ~110–156) OUT of forward.rs and INTO mod.rs so it is the shared trait both files reference. Re-export so every existing public path is preserved exactly: `pub use forward::{diff, directional_derivative, grad, jacobian, Dual};` and `pub use reverse::{vjp, Var};`. The `Scalar` trait, being defined in mod.rs, is already at `autodiff::Scalar`.

    In `autodiff/forward.rs` (the moved file): DELETE the `Scalar` trait definition (now in mod.rs) and add `use super::Scalar;` near the top so `impl Scalar for f64`, `impl Scalar for Dual`, and the entry points still compile. Leave `impl Scalar for f64`, the `Dual` struct + all its op/Scalar impls, and `diff`/`grad`/`jacobian`/`directional_derivative` otherwise verbatim. Keep the existing `#[cfg(test)] mod tests` in forward.rs unchanged (it references `Dual`, `grad`, etc. which stay here). NOTE: the composed-objective test at forward.rs ~1029 imports `crate::metric::soft_dtw_distance_generic` and `crate::regression::{fdata_to_pc, project_scores_generic}` by absolute path — those absolute paths are unaffected by the move.

    Do NOT edit `lib.rs` line 76 — verify it still reads `pub mod autodiff;` (a no-op after the refactor; Cargo resolves it to the directory). This module refactor is reversible-with-effort (git mv + re-split) and preserves the published contract, so it is NOT a one-way door.
  </action>
  <reversibility rating="reversible">Directory split preserves all public paths; re-collapsing to a single file is a mechanical git mv back.</reversibility>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff::forward 2>&1 | tail -20</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" / "error[" / "cannot find" in output — any existing Dual test failing to compile or pass means the move broke a public path</fails_when>
  </verify>
  <acceptance_criteria>
    - `fdars-core/src/autodiff.rs` no longer exists; `autodiff/mod.rs` and `autodiff/forward.rs` exist.
    - `autodiff::Dual`, `autodiff::Scalar`, `autodiff::grad`, `autodiff::diff`, `autodiff::jacobian`, `autodiff::directional_derivative` all resolve (compile check).
    - All pre-existing forward-mode `Dual` tests pass unchanged.
    - `lib.rs:76` still reads exactly `pub mod autodiff;`.
  </acceptance_criteria>
  <done>The autodiff module is a directory; the `Scalar` trait lives in mod.rs; `forward.rs` holds Dual + forward entry points and its tests pass; no public path changed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Stand up reverse.rs skeleton — Var, Node, TAPE, push helpers, Mul, minimal constants</name>
  <files>fdars-core/src/autodiff/reverse.rs, fdars-core/src/autodiff/mod.rs</files>
  <read_first>
    - fdars-core/src/autodiff/mod.rs (from Task 1 — the `Scalar` trait definition and the `pub use reverse::{vjp, Var};` line this task's symbols must satisfy)
    - fdars-core/src/alignment/mod.rs (lines 465–513: the `thread_local! { static ...: RefCell<Vec<...>> = const { RefCell::new(Vec::new()) }; }` idiom and the `.with(|cell| { let mut x = cell.borrow_mut(); ... })` access pattern to copy exactly — proven rayon-compatible)
    - fdars-core/src/autodiff/forward.rs (Dual's `#[derive(Debug, Clone, Copy)]` struct shape; the `Mul for Dual` impl as the structural analog; the value-only `PartialEq`/`PartialOrd` hand-written impls that must NOT be derived)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md (sections "Node Record Layout", "Thread-Local Tape", "Op Push Patterns", "Constants and Sentinel")
  </read_first>
  <behavior>
    - Test (added in Task 3): a single Mul recorded on the tape produces the correct backward gradient via vjp.
    - Constant `Var` values (from_f64/zero/one/infinity) carry the SENTINEL node index and push NO tape node.
    - `const * const` Mul short-circuits to a SENTINEL result (no tape node pushed).
  </behavior>
  <action>
    Create `fdars-core/src/autodiff/reverse.rs` with a module-level `//!` doc. Add `use super::Scalar;` and the ops it needs. Define, per 94-RESEARCH.md and 94-PATTERNS.md:
    - `const SENTINEL: usize = usize::MAX;`
    - `#[derive(Debug, Clone, Copy)] pub struct Var { pub(crate) value: f64, pub(crate) node: usize }`
    - `#[derive(Clone, Copy)] struct Node { deps: [usize; 2], weights: [f64; 2] }` (private).
    - `thread_local! { static TAPE: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) }; }` — copy the const-init idiom from alignment/mod.rs:473.
    - `fn push_binary(dep0: usize, w0: f64, dep1: usize, w1: f64) -> usize` and `fn push_unary(dep0: usize, w0: f64) -> usize` — each borrows TAPE mutably, records the node, returns `tape.len()` before push (the new node's index).
    - Hand-write value-only `PartialEq` (compare `value` only) and `PartialOrd` (`self.value.partial_cmp(&other.value)`) for `Var` — do NOT derive; deriving would compare `node` and corrupt `softmin3_generic` control flow later.
    - A PARTIAL `impl Scalar for Var` sufficient to compile the tracer: the four constant constructors `from_f64`/`zero`/`one`/`infinity` (all set `node: SENTINEL`, push nothing). Leave the transcendental methods (`sqrt`/`exp`/`ln`/`sin`/`cos`/`powf`/`abs`/`signum`) as `unimplemented!()` stubs for now — they are FUNCTIONALITY gaps filled in Plan 02, not architectural gaps. (This is the one allowed tracer stub: the trait shape is proven, the bodies come next.)
    - `impl Mul for Var`: compute `value = self.value * rhs.value`; if both nodes are SENTINEL return a SENTINEL constant; else `push_binary(self.node, rhs.value, rhs.node, self.value)` (partials ∂(u*v)/∂u = v, ∂(u*v)/∂v = u) and return `Var { value, node }`. Mul is the only op impl in this plan.

    Add the other supertrait op impls (`Add`/`Sub`/`Div`/`Neg`/`AddAssign`/`SubAssign`/`MulAssign`) as `unimplemented!()` stubs so `Var: Scalar`'s supertrait bounds are satisfied and the crate compiles — Plan 02 fills them. Do NOT wire any real op besides Mul.

    Ensure `autodiff/mod.rs`'s `pub use reverse::{vjp, Var};` resolves (vjp comes in Task 3).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -15</automated>
    <fails_when>non-zero exit, or "error[" / "cannot find" / "unresolved" in output — the skeleton must compile even with unimplemented stubs</fails_when>
  </verify>
  <acceptance_criteria>
    - `Var`, `Node`, `TAPE`, `push_binary`, `push_unary`, `SENTINEL` exist in reverse.rs.
    - `Var: Copy` and hand-written value-only `PartialEq`/`PartialOrd` (not derived).
    - `Mul for Var` records a binary node with product-rule partials and short-circuits const*const.
    - Constant constructors set `node: SENTINEL` and push nothing.
    - Crate compiles with `--features linalg,parallel`.
  </acceptance_criteria>
  <done>reverse.rs holds the Var/Node/TAPE skeleton with a working Mul and sentinel constants; the crate builds.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Minimal vjp + end-to-end known-answer tracer test</name>
  <files>fdars-core/src/autodiff/reverse.rs</files>
  <read_first>
    - fdars-core/src/autodiff/reverse.rs (from Task 2 — Var, Node, TAPE, push_binary, SENTINEL, Mul)
    - fdars-core/src/autodiff/forward.rs (the `grad` entry point body ~511–535 — the signature template `#[must_use] pub fn grad<F: Fn(&[Dual]) -> Dual>(f: F, x: &[f64]) -> (f64, Vec<f64>)`; and the `#[cfg(test)] mod tests` header with `const TOL: f64 = 1e-10` to mirror)
    - .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-RESEARCH.md (section "Backward Pass (the vjp implementation)" — the exact 7-step body: clear→seed leaves→forward→seed adjoint→backward sweep→collect→clear; and Pitfall 1 double-clear, Pitfall 2 real leaf nodes)
  </read_first>
  <behavior>
    - Test `var_mul_known_answer`: `vjp(|x| x[0] * x[0], &[3.0])` returns `(9.0, vec![6.0])` within TOL — f(x)=x^2, f'(3)=6. Proves record→seed→backward→read end-to-end.
  </behavior>
  <action>
    In `reverse.rs` add `#[must_use] pub fn vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)` mirroring `grad`'s signature. Implement the 7-step lifecycle from 94-RESEARCH.md:
    1. Clear TAPE (borrow_mut().clear()) — safe re-entry after a prior panic (double-clear, Pitfall 1).
    2. Seed one REAL leaf node per input via `push_binary(SENTINEL, 0.0, SENTINEL, 0.0)` — leaves have a valid tape index so their adjoint accumulates, but propagate nothing (both deps SENTINEL, both weights 0.0). Build `Vec<Var> { value: x[k], node: leaf_idx }`. Do NOT use `from_f64` for inputs (that makes them SENTINEL constants → all-zero gradient, Pitfall 2).
    3. Run `let output = f(&vars); let primal = output.value;` (forward pass records ops).
    4–6. Borrow TAPE, allocate `adjoints = vec![0.0; tape.len()]`, seed `adjoints[output.node] = 1.0` (guard `output.node != SENTINEL`), reverse-iterate `for i in (0..n).rev()` accumulating `adjoints[deps[slot]] += adjoints[i] * weights[slot]` under the `deps[slot] != SENTINEL` guard, then collect `gradient[k] = if vars[k].node == SENTINEL { 0.0 } else { adjoints[vars[k].node] }`.
    7. Clear TAPE again (cleanup for the next call).
    Handle `x.len() == 0` by returning `(f(&[]).value, Vec::new())` (mirror grad's empty guard).

    Add `#[cfg(test)] mod tests { use super::*; const TOL: f64 = 1e-10; ... }` with ONE test `var_mul_known_answer` per the behavior block. This is the tracer's end-to-end proof — the full record→seed→backward→read-adjoint loop through a real op.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel autodiff::reverse 2>&1 | tail -20</automated>
    <fails_when>non-zero exit, or "FAILED" / "0 passed" in output, or the reported gradient is 0.0 (Pitfall 2: leaf not a real node) or grows across runs (Pitfall 1: tape leakage)</fails_when>
  </verify>
  <acceptance_criteria>
    - `vjp` exists, is `#[must_use]`, returns `(f64, Vec<f64>)` with `gradient.len() == x.len()`.
    - `var_mul_known_answer` passes: value ≈ 9.0, gradient[0] ≈ 6.0 within 1e-10.
    - Tape is cleared before AND after the backward pass.
    - One forward pass + one reverse sweep (no per-input loop).
  </acceptance_criteria>
  <done>The full reverse-mode loop is proven end-to-end: one Mul recorded, output adjoint seeded, one backward sweep, input gradient read back correctly. The architecture holds.</done>
</task>

</tasks>

<threat_model>
No attack surface — pure numerical library code operating on caller-provided f64 slices; no I/O, network, deserialization, filesystem, auth, or privilege boundary. The thread-local tape is process-local with no exfiltration vector. The only correctness hazards are numerical (NaN/Inf adjoints at singular points) and are covered by guard tests in Plan 02, not by security controls.

| Boundary | Description |
|----------|-------------|
| (none) | No trust boundary crossed — in-crate numeric computation on f64 slices |
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg,parallel autodiff` — all forward (`Dual`) tests plus the one reverse tracer test pass.
- Out-of-band gates (per-gate, foreground, 600s): `cargo fmt --check`; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- `git diff Cargo.toml fdars-core/Cargo.toml` is empty (no new dependency).
- `lib.rs:76` unchanged.
</verification>

<success_criteria>
- autodiff refactored into a directory; all existing public paths resolve; all Dual tests green.
- reverse.rs skeleton (Var/Node/TAPE/push helpers/Mul/minimal vjp) compiles and the end-to-end multiply-gradient tracer test passes.
- No new crate dependency; clippy + fmt clean.
</success_criteria>

<output>
Create `.planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-01-SUMMARY.md` when done, noting the final node-record layout chosen and any deviation from the research recommendation.
</output>
