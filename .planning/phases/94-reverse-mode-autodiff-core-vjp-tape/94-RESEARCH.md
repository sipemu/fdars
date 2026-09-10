# Phase 94: Reverse-Mode Autodiff Core (VJP Tape) - Research

**Researched:** 2026-09-10
**Domain:** In-crate hand-written Wengert-list reverse-mode automatic differentiation in Rust
**Confidence:** HIGH (all findings grounded in direct codebase reads or authoritative external source)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `Var` implements the existing `Scalar` trait (autodiff.rs:110–156, 12 methods + Copy/Add/Sub/Mul/Div/Neg/AddAssign/SubAssign/MulAssign supertraits).
- Thread-local tape access model. `Var { value: f64, node: usize }` (Copy handle). Tape reached via thread-local, not a stored borrow.
- Constants use a sentinel node index (`usize::MAX`) — `from_f64`/`zero`/`one`/`infinity` produce off-tape constants; backward pass skips sentinel parents.
- Value-only `PartialEq`/`PartialOrd` on `Var`, mirroring `Dual`'s semantics.
- `vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)` mirrors `grad`'s signature.
- `vjp` (many-input→scalar) only — reverse-mode `jacobian` out of scope.
- Each tape node records local partial derivative(s) and parent node indices; backward accumulates `adjoint[parent] += adjoint[node] * local_partial`.
- Module refactor into `autodiff/` directory: `autodiff/mod.rs`, `autodiff/forward.rs`, `autodiff/reverse.rs`. Public paths preserved.
- Type name `Var`, tape type `Tape`. Prelude extends with `Var`, `Tape`, `vjp`.

### Claude's Discretion
- Exact node record layout (enum vs struct-of-arrays), tape growth strategy, thread-local reset/scoping ergonomics, whether `Tape` is publicly constructible or fully hidden behind `vjp`, and internal helper naming.

### Deferred Ideas (OUT OF SCOPE)
- Reverse-mode `jacobian` (vector-output)
- Differentiating `elastic_distance` DP argmin (DIF-02, permanently deferred)
- `soft_dtw_barycenter` optimizer redesign (SDTW-O1)
- Generalizing further hot-path signatures (Phase 95 / GEN-01)
- Unified `grad`/`jacobian`/`vjp` demo API (Phase 99 / API-01)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RAD-01 | In-crate reverse-mode tape (Wengert-list) recording operations on `Var`/tape scalar type, supporting same op set as `Dual` | Node record layout (§Node Layout), op push patterns (§Op Push Patterns), sentinel constant design (§Constants & Sentinel) |
| RAD-02 | Backward pass seeding output adjoint, accumulating input gradients, exposing `vjp` entry point efficient for many-input→scalar objectives | Backward pass algorithm (§Backward Pass), `vjp` implementation shape (§vjp Entry Point), efficiency analysis (§Efficiency Claim) |
| RAD-03 | Reverse-mode gradients match forward-mode `Dual` path and finite differences within tolerance on existing differentiable subset | Validation architecture (§Validation Architecture), tolerance strategy (§Tolerances), both target functions verified generic in codebase (§Validation Targets) |
</phase_requirements>

---

## Summary

Phase 94 delivers a hand-written reverse-mode (VJP) tape alongside the existing forward-mode `Dual`. The architecture is fully decided in CONTEXT.md — this research answers the implementation-risk questions a planner needs to sequence tasks correctly.

The canonical Wengert-list pattern in Rust uses a `Vec<Node>` on the tape, where each node stores two local partials and two parent indices (covering both unary and binary ops). Constants use a sentinel index (`usize::MAX`) that the backward pass skips. The codebase already uses `thread_local! { static ...: RefCell<...> }` in `alignment/mod.rs` for per-rayon-worker scratch buffers — the tape follows the exact same idiom and is rayon-compatible for the same reason: each rayon worker thread has its own tape copy, so `vjp` calls that happen on different threads never see each other's tapes.

The key implementation risk is **not** algorithmic — it is ordering and cleanup. A tape built during the forward closure is already in topological order (evaluation order = topological order for a DAG built by sequential execution), so the backward pass is a plain reverse iteration with no graph analysis needed. The lifecycle risk is tape leakage across `vjp` calls: the `vjp` function must clear the tape after reading the adjoints, not before (adjoint read happens last). The `soft_dtw_backward` zero-gradient bug documented in project MEMORY is irrelevant here — that bug is in the old hand-rolled `f64` backward function; the new path goes through the generic `Var` tape.

**Primary recommendation:** Implement the node record as a flat `Vec<Node>` where `Node { deps: [usize; 2], weights: [f64; 2] }`, use `usize::MAX` as the sentinel for both `deps` slots of constant nodes, write the backward pass as a single `for i in (0..tape_len).rev()` loop, and wrap the entire forward+backward+clear lifecycle inside `vjp`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Tape storage & node push | `autodiff/reverse.rs` | — | All state is in the thread-local tape; no other module touches it |
| `Var` type + `Scalar` impl | `autodiff/reverse.rs` | `autodiff/mod.rs` (re-export) | `Var` is the public handle type |
| `vjp` entry point | `autodiff/reverse.rs` | `autodiff/mod.rs` (re-export) | Mirrors `grad` which lives in forward.rs |
| `Scalar` trait definition | `autodiff/mod.rs` | — | Shared between forward and reverse; moves from autodiff.rs to mod.rs |
| `Dual` + forward entry points | `autodiff/forward.rs` | `autodiff/mod.rs` (re-export) | Unchanged move from autodiff.rs |
| Public re-exports | `autodiff/mod.rs` + `prelude.rs` | `lib.rs` (unchanged `pub mod autodiff`) | Preserves all existing public paths |
| Validation tests | `autodiff/reverse.rs` (`#[cfg(test)]`) | Integration tests in `tests/` | Mirrors inline test pattern from `Dual` |

---

## Standard Stack

No external packages. This phase is in-crate only.

### No New Dependency

The implementation is entirely in-crate Rust — `Vec`, `RefCell`, `thread_local!`, standard arithmetic traits. The `Cargo.toml` is unchanged. [VERIFIED: fdars-core/Cargo.toml:34-46]

---

## Package Legitimacy Audit

No external packages are installed in this phase. Audit: N/A.

---

## Architecture Patterns

### System Architecture Diagram

```
vjp(f, x: &[f64]) -> (f64, Vec<f64>)
         │
         ├─ 1. TAPE CLEAR ── TAPE.with(|c| c.borrow_mut().clear())
         │
         ├─ 2. SEED VARS ── x.iter().map(|&v| push_leaf(v)) → Vec<Var>
         │                                │
         │                    TAPE grows: one node per input
         │
         ├─ 3. FORWARD PASS ── f(&vars) → Var (output)
         │                                │
         │                    TAPE grows: one node per op
         │                    (evaluation order = topological order)
         │
         ├─ 4. SEED ADJOINT ── adjoints[output.node] = 1.0
         │
         ├─ 5. BACKWARD PASS ── for i in (0..tape_len).rev()
         │                          for slot in 0..2
         │                              if deps[i][slot] != SENTINEL
         │                                  adjoints[deps[i][slot]] +=
         │                                      adjoints[i] * weights[i][slot]
         │
         ├─ 6. COLLECT GRADIENT ── input_vars.iter().map(|v| adjoints[v.node])
         │
         └─ 7. TAPE CLEAR ── reset for next call (prevent leakage)
```

### Node Record Layout (recommended: struct approach)

```rust
// Source: Rufflewind 2016 tutorial + codebase alignment/mod.rs thread-local pattern
// [VERIFIED: fdars-core/src/alignment/mod.rs:473] — identical RefCell idiom already in use

/// A single record on the Wengert-list tape.
///
/// Covers both binary ops (both slots populated) and unary ops (slot 1 = SENTINEL,
/// weight 1 = 0.0). Constants produce no node — they use SENTINEL as their `node` index.
#[derive(Clone, Copy)]
struct Node {
    /// Parent node indices. `usize::MAX` = sentinel (no contribution for that slot).
    deps: [usize; 2],
    /// Local partial derivative w.r.t. each parent:
    ///   deps[0] contributes `weights[0]`, deps[1] contributes `weights[1]`.
    weights: [f64; 2],
}
```

**Why this layout:** Every op — binary or unary — fits the same 4-word record. The backward loop is a single uniform `for j in 0..2 { if deps[j] != SENTINEL { adjoints[deps[j]] += adjoint * weights[j] } }` — no branching on op type.

### Thread-Local Tape

```rust
// Source: alignment/mod.rs:473 — identical pattern already proven in this codebase
// [VERIFIED: fdars-core/src/alignment/mod.rs:471-473]
use std::cell::RefCell;

thread_local! {
    /// Per-thread Wengert-list tape.
    ///
    /// Each rayon worker thread has its own tape, so vjp calls on different
    /// threads never interfere. Cleared at the start and end of every vjp call.
    static TAPE: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) };
}
```

**Access pattern** (from alignment/mod.rs:506):
```rust
TAPE.with(|cell| {
    let mut tape = cell.borrow_mut();
    // ... push nodes, backward pass
})
```

### `Var` Type

```rust
// [VERIFIED: 94-CONTEXT.md — Var { value: f64, node: usize } (Copy)]
#[derive(Debug, Clone, Copy)]
pub struct Var {
    /// The primal (forward) value.
    pub(crate) value: f64,
    /// Index into the thread-local tape. `usize::MAX` = constant (off-tape).
    pub(crate) node: usize,
}

const SENTINEL: usize = usize::MAX;
```

### Constants and Sentinel

Constants (`from_f64`, `zero`, `one`, `infinity`) set `node = SENTINEL`. They push **no node** to the tape. This is critical: `soft_dtw_distance_generic` and `project_scores_generic` call `S::from_f64(...)`, `S::infinity()`, and `S::zero()` extensively (initialization, weight scaling) — each such call on `Var` must be cheap and produce no tape record.

The backward pass guard `if deps[j] != SENTINEL` skips constants automatically.

```rust
// Pattern: constant Var, no tape push
impl Scalar for Var {
    fn from_f64(v: f64) -> Self { Var { value: v, node: SENTINEL } }
    fn zero() -> Self { Var { value: 0.0, node: SENTINEL } }
    fn one() -> Self { Var { value: 1.0, node: SENTINEL } }
    fn infinity() -> Self { Var { value: f64::INFINITY, node: SENTINEL } }
}
```

### Op Push Patterns

Every arithmetic op on `Var` runs the forward value computation and pushes one `Node`. Helper: `push_binary` / `push_unary`.

```rust
// Source: Rufflewind 2016 canonical pattern
fn push_binary(dep0: usize, w0: f64, dep1: usize, w1: f64) -> usize {
    TAPE.with(|cell| {
        let mut tape = cell.borrow_mut();
        let idx = tape.len();
        tape.push(Node {
            deps: [dep0, dep1],
            weights: [w0, w1],
        });
        idx
    })
}

fn push_unary(dep0: usize, w0: f64) -> usize {
    TAPE.with(|cell| {
        let mut tape = cell.borrow_mut();
        let idx = tape.len();
        tape.push(Node {
            deps: [dep0, SENTINEL],
            weights: [w0, 0.0],
        });
        idx
    })
}
```

**Binary ops** — product rule example:

```rust
impl Mul for Var {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let value = self.value * rhs.value;
        if self.node == SENTINEL && rhs.node == SENTINEL {
            // const * const: no node
            return Var { value, node: SENTINEL };
        }
        // General: ∂(u*v)/∂u = v, ∂(u*v)/∂v = u
        let node = push_binary(self.node, rhs.value, rhs.node, self.value);
        Var { value, node }
    }
}
```

**Unary ops** — sqrt example:

```rust
impl Scalar for Var {
    fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        if self.node == SENTINEL {
            return Var { value: s, node: SENTINEL };
        }
        // d/dx sqrt(v) = 1/(2*sqrt(v))
        let node = push_unary(self.node, 1.0 / (2.0 * s));
        Var { value: s, node }
    }
}
```

**Singular points** — match `Dual` behavior exactly (NaN/Inf adjoint, not panics). `sqrt` at 0: `1/(2*0) = Inf`; `ln` at 0: `1/0 = Inf`. No clamping.

**Unary/binary constant-folding guard:** When both inputs are sentinel (or the sole input is sentinel), skip the `push_*` call and return a constant `Var`. This keeps the tape from bloating on constant-only subexpressions (e.g., `S::from_f64(gamma)` scaled quantities in `softmin3_generic`).

### Backward Pass (the `vjp` implementation)

```rust
pub fn vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
    // 1. Clear tape from any prior call.
    TAPE.with(|cell| cell.borrow_mut().clear());

    // 2. Seed inputs: one leaf node per input, no parents.
    let vars: Vec<Var> = x.iter().map(|&v| {
        let node = push_binary(SENTINEL, 0.0, SENTINEL, 0.0); // leaf
        Var { value: v, node }
    }).collect();

    // 3. Forward pass: tape records all ops.
    let output = f(&vars);
    let primal = output.value;

    // 4. Backward pass.
    let gradient = TAPE.with(|cell| {
        let tape = cell.borrow();
        let n = tape.len();
        let mut adjoints = vec![0.0f64; n];

        if output.node != SENTINEL {
            adjoints[output.node] = 1.0; // seed
        }

        for i in (0..n).rev() {
            let node = tape[i];
            let adj = adjoints[i];
            for slot in 0..2 {
                if node.deps[slot] != SENTINEL {
                    adjoints[node.deps[slot]] += adj * node.weights[slot];
                }
            }
        }

        // 5. Collect gradient: adjoint at each input leaf node.
        vars.iter().map(|v| {
            if v.node == SENTINEL { 0.0 } else { adjoints[v.node] }
        }).collect()
    });

    // 6. Clear tape after reading: prevent leakage into next call.
    TAPE.with(|cell| cell.borrow_mut().clear());

    (primal, gradient)
}
```

**Lifecycle invariant:** Clear before AND after. "Before" handles the case where a prior `vjp` call panicked without clearing. "After" is the normal cleanup. This is the safe double-clear pattern.

**Leaf node design:** Input leaves are pushed as `push_binary(SENTINEL, 0.0, SENTINEL, 0.0)` — they are real nodes (with valid indices) so `vars[k].node` indexes into `adjoints[]` correctly. They do not contribute to any backward propagation (both weights are 0.0, both deps are sentinel), but they accumulate adjoint from the ops that consumed them.

### Module Refactor (directory split)

```
fdars-core/src/autodiff/
├── mod.rs       — Scalar trait, shared re-exports (Dual, Var, Tape, diff, grad, jacobian,
│                  directional_derivative, vjp). Public paths unchanged.
├── forward.rs   — Dual struct + all Scalar impls + diff/grad/jacobian/directional_derivative.
│                  Moved verbatim from autodiff.rs; no content changes.
└── reverse.rs   — Var struct, Node struct, TAPE thread-local, push_binary/push_unary helpers,
                   Scalar impl for Var (all ops), vjp entry point, #[cfg(test)] mod tests.
```

`lib.rs:76` remains `pub mod autodiff;` — unchanged. [VERIFIED: fdars-core/src/lib.rs:76]

`prelude.rs:21` extends from:
```rust
pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};
```
to:
```rust
pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var};
```
`Tape` may be re-exported from `prelude` too if it has any public surface (e.g., a `clear()` method), or left at `autodiff::Tape` only if it's purely internal to `vjp`. [VERIFIED: fdars-core/src/prelude.rs:21]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Topological sort of the tape | A graph traversal or DFS | None needed — the tape is already in evaluation order | A Wengert list built by sequential execution is already a topological sort of the computation DAG. Reverse iteration is sufficient. |
| Per-op dispatch at backward | An enum with a match arm per op | The uniform 2-slot node record | Every op maps to at most 2 parents + 2 local partials. The backward loop is data-driven, not op-dispatch-driven. |
| Thread-safety of the tape | Mutex / Arc | `thread_local! { RefCell<Vec<Node>> }` | Each thread owns its tape. Already proven compatible with rayon in `alignment/mod.rs`. |
| Constant folding / dead-node pruning | A separate graph optimizer | Sentinel check in push | The sentinel guard in `push_binary`/`push_unary` is sufficient for correctness. Full dead-node pruning is a performance optimization out of scope. |

**Key insight:** Reverse-mode AD on a Wengert list requires no graph infrastructure beyond a `Vec<Node>` and a reverse `for` loop. The topological-order property comes free from sequential execution.

---

## Runtime State Inventory

Not applicable — greenfield addition, no rename/refactor.

---

## Common Pitfalls

### Pitfall 1: Tape Leakage Between `vjp` Calls

**What goes wrong:** If `vjp` (or a function it calls) panics after the forward pass but before the tape clear, the next `vjp` call sees stale nodes from the prior call. Adjoint indices are wrong; gradients are silently corrupt.

**Why it happens:** Single clear at the end is not panic-safe. A `?` propagation or an `assert!` in user closure triggers unwinding with the tape non-empty.

**How to avoid:** Clear at the **start** of `vjp` (before the forward pass) as well as at the end. This makes re-entry idempotent. If Rust unwinds through `vjp`, the next call clears the stale state before proceeding.

**Warning signs:** Gradient components that are suspiciously large or that grow with each repeated `vjp` call on the same function.

---

### Pitfall 2: Leaf Nodes Without Real Tape Indices

**What goes wrong:** If input leaves are `Var { node: SENTINEL }` (constants), their adjoint is never accumulated — the gradient is all zeros.

**Why it happens:** `vjp` must push a real leaf node for each input, even though the leaf has no parents. `Var::leaf(v)` ≠ `Scalar::from_f64(v)`. These are distinct:
- `from_f64(v)` → constant, `node = SENTINEL`, gradient contribution = 0
- `leaf(v)` → real node at index k, gradient = `adjoints[k]` after backward

**How to avoid:** `vjp` explicitly calls `push_binary(SENTINEL, 0.0, SENTINEL, 0.0)` for each input, giving it a real tape index. Never call `Var::from_f64` or `Scalar::from_f64` on inputs inside `vjp`.

**Warning signs:** All gradient components are 0.0 even on a function with obvious dependence.

---

### Pitfall 3: Constant-vs-Var Arithmetic Silently Drops Gradient

**What goes wrong:** When one operand of a binary op is a constant (`node = SENTINEL`), the corresponding parent slot should contribute 0 adjoint. If instead the code treats `SENTINEL` as a valid node index, it will corrupt `adjoints[usize::MAX]` (or panic on OOB access).

**Why it happens:** `push_binary` receives `SENTINEL` for one dep. The backward loop must check `dep != SENTINEL` before indexing.

**How to avoid:** The sentinel guard `if node.deps[slot] != SENTINEL` in the backward loop handles this. Additionally, the constant-folding short-circuit in binary ops (if both `self.node == SENTINEL && rhs.node == SENTINEL`) avoids pushing a node at all for pure constant arithmetic.

**Warning signs:** Index-out-of-bounds panic in the backward pass; or incorrect gradient on a function mixing constants and variables.

---

### Pitfall 4: `AddAssign` / `SubAssign` / `MulAssign` Not Composing Chain Rule

**What goes wrong:** Deriving `AddAssign` from the `Add` impl is correct. But if `AddAssign` is hand-written as `self.value += rhs.value` without updating `self.node`, the tape record is skipped and gradient does not flow.

**Why it happens:** Assign-ops on `Dual` correctly delegate to the binary ops (`*self = *self + rhs`). The same must hold for `Var`.

**How to avoid:** Implement `AddAssign`, `SubAssign`, `MulAssign` as delegation to the binary ops:

```rust
impl AddAssign for Var {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
```

This ensures `push_binary` is called, updating both `self.value` and `self.node`. [VERIFIED: autodiff.rs:322-341 — identical Dual pattern]

---

### Pitfall 5: `soft_dtw_backward` Bug Is Not Relevant

**What goes wrong (would-be confusion):** Project MEMORY documents a `soft_dtw_backward` zero-gradient bug. A planner might worry this affects Phase 94 validation.

**Why it does not apply:** That bug is in the old hand-rolled `f64` backward function for `soft_dtw_barycenter`. Phase 94's validation runs `soft_dtw_distance_generic<Var>` — the generic Scalar path — through the Wengert-list tape. The generic function has no hand-rolled backward; gradients come entirely from the tape's chain-rule accumulation. The two code paths are completely separate.

**Warning signs (if confused):** Writing tests that call the old `f64` `soft_dtw_backward` instead of `vjp(|vars| soft_dtw_distance_generic(vars, ...), ...)`.

---

### Pitfall 6: `PartialOrd` / `PartialEq` Must Be Value-Only

**What goes wrong:** `softmin3_generic` uses `<=` and `>=` comparisons on `Var` values to select the minimum and handle infinity sentinels. If `PartialOrd` compared `node` fields, control flow would branch on tape indices rather than primal values — producing wrong softmin selection and corrupt gradients.

**How to avoid:** Hand-write `PartialEq` and `PartialOrd` comparing `value` only, exactly as `Dual` does. [VERIFIED: autodiff.rs:347-362 — Dual's identical hand-written PartialOrd]

---

### Pitfall 7: `DivAssign` Is Not Required by `Scalar`

**What goes wrong:** The `Scalar` trait requires `AddAssign + SubAssign + MulAssign` but NOT `DivAssign`. Implementing `DivAssign` for completeness is fine but not gating on it.

**How to avoid:** Check the `Scalar` supertrait bounds list before writing impls. [VERIFIED: autodiff.rs:110-122 — supertrait list verbatim: `AddAssign + SubAssign + MulAssign`, no `DivAssign`]

---

## Code Examples

### Tier 1 Known-Answer Test Pattern (mirror from Dual tests)

```rust
// Source: autodiff.rs:640-651 — Dual test structure to mirror exactly
// [VERIFIED: fdars-core/src/autodiff.rs:629-651]
#[test]
fn var_mul_known_answer() {
    // f(x) = x^2, df/dx = 2x. At x = 3: f = 9, df/dx = 6.
    let (value, grad) = vjp(|x| x[0] * x[0], &[3.0]);
    assert!((value - 9.0).abs() < 1e-10);
    assert!((grad[0] - 6.0).abs() < 1e-10);
}

#[test]
fn var_sqrt_known_answer() {
    // f(x) = sqrt(x), df/dx = 1/(2*sqrt(x)). At x = 4: f = 2, df/dx = 0.25.
    let (value, grad) = vjp(|x| Scalar::sqrt(x[0]), &[4.0]);
    assert!((value - 2.0).abs() < 1e-10);
    assert!((grad[0] - 0.25).abs() < 1e-10);
}
```

### Reverse-vs-Dual Agreement Test Pattern

```rust
// vjp result must agree with grad (forward-mode) at 1e-10 for every op
#[test]
fn var_vs_dual_mul_agreement() {
    let x = &[3.0f64];
    let (fwd_val, fwd_grad) = grad(|d: &[Dual]| d[0] * d[0], x);
    let (rev_val, rev_grad) = vjp(|v: &[Var]| v[0] * v[0], x);
    assert!((fwd_val - rev_val).abs() < 1e-10);
    assert!((fwd_grad[0] - rev_grad[0]).abs() < 1e-10);
}
```

### Central FD Cross-Check on `soft_dtw_distance_generic`

```rust
// Source: autodiff.rs:1029-1124 — grad_composed_objective_matches_finite_diff pattern
// [VERIFIED: fdars-core/src/autodiff.rs:1029-1124]
#[test]
fn vjp_soft_dtw_matches_finite_diff() {
    use crate::metric::soft_dtw_distance_generic;
    let m = 16usize;
    let gamma = 0.1f64;
    let curve: Vec<f64> = (0..m).map(|j| (j as f64 / m as f64).sin()).collect();
    let reference: Vec<f64> = (0..m).map(|j| (j as f64 / m as f64).cos()).collect();

    let (_, gradient) = vjp(|vars: &[Var]| {
        let ref_vars: Vec<Var> = reference.iter().map(|&r| Scalar::from_f64(r)).collect();
        soft_dtw_distance_generic(vars, &ref_vars, gamma)
    }, &curve);

    let h = 1e-8f64;
    for j in 0..m {
        let mut plus = curve.clone();
        let mut minus = curve.clone();
        plus[j] += h;
        minus[j] -= h;
        let fd = (soft_dtw_distance_generic::<f64>(&plus, &reference, gamma)
                  - soft_dtw_distance_generic::<f64>(&minus, &reference, gamma))
                 / (2.0 * h);
        assert!((gradient[j] - fd).abs() < 1e-6,
            "component {j}: rev {} vs FD {fd}", gradient[j]);
    }
}
```

### Composed-Objective `vjp` Test Pattern

```rust
// Mirrors grad_composed_objective_matches_finite_diff with vjp instead of grad
// [VERIFIED: autodiff.rs:1029 structure]
#[test]
fn vjp_composed_objective_matches_finite_diff() {
    // soft_dtw_distance_generic(curve, ref, gamma) + lambda * sum(scores^2)
    // h = 1e-6 for composed test (matching autodiff.rs:1111)
    let h = 1e-6f64;
    // ... (full setup mirrors the existing test at autodiff.rs:1029-1124)
    // gradient from vjp must match FD per component within 1e-6
}
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` harness |
| Config file | none (inline `#[cfg(test)] mod tests`) |
| Quick run command | `cargo test -p fdars-core autodiff --features linalg` |
| Full suite command | `cargo test -p fdars-core --features linalg` |

### Five Tiers — Mirroring autodiff.rs:629–1143

The existing Dual tests run in 5 tiers. The `Var` tests must mirror all 5, in the same file (`autodiff/reverse.rs`).

| Tier | Purpose | Tolerance | Count |
|------|---------|-----------|-------|
| 1. Known-answer | One test per op: `+`, `-`, `*`, `/`, neg, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, `signum`, assign-ops, composed chain | 1e-10 | ~14 tests |
| 2. Singular-point guards | `sqrt(0)` adjoint = Inf, `ln(0)` adjoint = Inf, `powf(0, 0.5)` adjoint = Inf — NaN/Inf propagated, not panics | exact Inf/NaN check | ~3 tests |
| 3. Reverse-vs-`Dual` agreement | For every op: `vjp` gradient == `grad` gradient at 1e-10 | 1e-10 | ~6 tests |
| 4. Reverse-vs-central-FD on real functions | `soft_dtw_distance_generic`, `project_scores_generic`, composed objective | 1e-6 (h=1e-8 simple; h=1e-6 composed) | ~3 tests |
| 5. Lifecycle / edge cases | Empty input, constant-only closure, repeated `vjp` calls, `vjp` on single-input | exact | ~4 tests |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| RAD-01 | `Var` records ops on tape correctly, all 12 methods + arithmetic | unit (Tier 1 + 2) | `cargo test -p fdars-core autodiff::reverse::tests --features linalg` |
| RAD-01 | `Var: Scalar` compiles — `soft_dtw_distance_generic::<Var>` and `project_scores_generic::<Var>` instantiate | compile check | `cargo check -p fdars-core --features linalg` |
| RAD-02 | `vjp` returns `(f64, Vec<f64>)` in one backward pass; gradient length == input length | unit (Tier 5) | `cargo test -p fdars-core autodiff --features linalg` |
| RAD-03 | Reverse == `Dual` path within 1e-10 | unit (Tier 3) | `cargo test -p fdars-core autodiff --features linalg` |
| RAD-03 | Reverse == central FD within 1e-6 on `soft_dtw_distance_generic` | integration (Tier 4) | `cargo test -p fdars-core autodiff --features linalg` |
| RAD-03 | Reverse == central FD within 1e-6 on `project_scores_generic` | integration (Tier 4) | `cargo test -p fdars-core autodiff --features linalg` |
| RAD-03 | Composed objective matches FD per component within 1e-6 | integration (Tier 4) | `cargo test -p fdars-core autodiff --features linalg` |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core autodiff --features linalg` (fast, < 5 s)
- **Per wave merge:** `cargo test -p fdars-core --features linalg`
- **Phase gate:** Full suite green: `cargo fmt --check && cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test -p fdars-core --features linalg`

### Wave 0 Gaps

- [ ] `fdars-core/src/autodiff/mod.rs` — does not exist yet (directory refactor)
- [ ] `fdars-core/src/autodiff/forward.rs` — does not exist yet (moved from autodiff.rs)
- [ ] `fdars-core/src/autodiff/reverse.rs` — does not exist yet (new file)

*(No new test framework needed — Rust built-in harness already configured.)*

---

## Efficiency Claim (RAD-02)

Forward-mode `grad` runs the closure **m times** (once per input) to compute a full gradient for a many-input→scalar function. Reverse-mode `vjp` runs the closure **once** (one forward pass to build the tape) and then does **one backward sweep** to compute all m gradients simultaneously. The asymptotic complexity for a many-input→scalar function of m inputs is:

- `grad`: O(m · cost(f)) — m forward passes
- `vjp`: O(cost(f) + tape_len) — 1 forward pass + 1 reverse sweep, where `tape_len ≤ O(cost(f))`

For the two validation targets:
- `soft_dtw_distance_generic(x, y, gamma)`: m = len(x) (e.g., 24–512 sample points). Forward cost O(n·m) for the DP. `vjp` is ~m× faster than `grad`.
- `project_scores_generic(curve, ...)`: m = len(curve). Forward cost O(m·ncomp). `vjp` is ~m× faster.

The efficiency win is real but not the primary concern for Phase 94 (correctness is). No benchmarks are required; this phase is correctness-only.

---

## Rayon Compatibility

The thread-local tape is **natively rayon-compatible** for the same reason `DP_SCRATCH` in `alignment/mod.rs` is: each rayon worker thread has its own `thread_local!` slot. Two calls to `vjp` executing on different threads in a rayon parallel region will access different tape instances and never interfere.

**Constraint:** `vjp` itself is sequential (single tape, single backward pass). It MUST NOT be called concurrently from multiple threads on the same closure — but this is the expected single-call API anyway. Callers that use rayon over independent `vjp` calls (e.g., per-curve gradient computation) are safe because each call is on its own thread with its own tape.

[VERIFIED: fdars-core/src/alignment/mod.rs:471-473 — identical thread_local RefCell pattern proven rayon-compatible in production]

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| Borrowed-tape `Var<'t>` with lifetime | Thread-local `Var { node: usize }` (Copy) | Lifetime approach breaks `Scalar::from_f64`/`zero`/`one` constructors (decided in CONTEXT.md) |
| `Rc<RefCell<Tape>>` stored in `Var` | Thread-local | Rc breaks `Copy`, adds runtime ref-count overhead |
| Per-op match dispatch in backward | Uniform 2-slot node record | Uniform record removes branching in the hot backward loop |
| Separate allocations per tape | `Vec<Node>` with `.clear()` reuse | Amortized allocation: tape `Vec` grows to high-water mark and stays there |

**The canonical Wengert-list paper:** Wengert (1964), "A simple automatic derivative evaluation program" — the tape is the original contribution. The thread-local + Copy handle variant is a modern Rust idiom documented in Rufflewind (2016). [CITED: https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation]

---

## Validation Targets — Codebase Verification

Both functions are confirmed generic over `Scalar` in the current codebase:

- `soft_dtw_distance_generic<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S` [VERIFIED: fdars-core/src/metric/soft_dtw.rs:158]
- `project_scores_generic<S: Scalar>(curve: &[S], mean: &[f64], rotation: &FdMatrix, weights: &[f64], ncomp: usize) -> Vec<S>` [VERIFIED: fdars-core/src/regression.rs:232]

Both functions use only `Scalar` methods (`from_f64`, `zero`, `infinity`, `sqrt`, `exp`, `ln`, `sin`, `cos`, arithmetic ops) — no `f64`-specific methods. Instantiating at `Var` requires no call-site changes.

The `soft_dtw_distance_inner` DP kernel uses `S::infinity()` for initialization and `softmin3_generic<S>` for the recurrence. `softmin3_generic` uses `PartialOrd` (`<=`, `>=`) for branching — correct for `Var` if `PartialOrd` is value-only. [VERIFIED: fdars-core/src/metric/soft_dtw.rs:57-94]

---

## Tolerances

| Test type | Tolerance | Step h | Matches existing |
|-----------|-----------|--------|-----------------|
| Known-answer Tier 1 | 1e-10 | — | `const TOL: f64 = 1e-10` in autodiff.rs:634 [VERIFIED] |
| Reverse-vs-Dual agreement | 1e-10 | — | Same as Tier 1 |
| Central FD (simple function) | 1e-6 | h = 1e-8 | autodiff.rs:895: `let h = 1e-8_f64` [VERIFIED] |
| Central FD (composed objective) | 1e-6 | h = 1e-6 | autodiff.rs:1111: `let h = 1e-6_f64` [VERIFIED] |
| Singular-point guards | NaN/Inf exact | — | autodiff.rs:848-889 [VERIFIED] |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Constant-folding short-circuit (skip push when both operands are sentinel) is an optimization the planner may include or omit without correctness risk — correctness requires only the backward sentinel guard | Op Push Patterns | If omitted from planning, tape may bloat on constant-heavy computations (softmin initialization), but gradients remain correct |
| A2 | `Tape` struct can remain fully internal (no public constructor needed) since `vjp` manages the lifecycle entirely | Module Layout | If user needs direct tape access (not required by RAD-01/02/03), `Tape` may need a thinner public surface |

**If this table has only 2 entries:** All other claims were verified directly from source files this session.

---

## Open Questions

1. **`Tape` public surface**
   - What we know: CONTEXT.md says "whether `Tape` is publicly constructible or fully hidden behind `vjp`" is Claude's discretion.
   - What's unclear: Phase 99 (API-01) might want `Tape` to be publicly inspectable (node count for debugging). Nothing in RAD-01/02/03 requires it.
   - Recommendation: Make `Tape` opaque — not publicly constructible, not re-exported from prelude. Expose only `vjp`. If Phase 99 needs inspection, add it then.

2. **`Div` implementation when RHS is a constant**
   - What we know: The quotient rule for `u/v` is `(u' * v - u * v') / v^2`. When `v` is a constant (`node = SENTINEL`), `v' = 0` and the formula simplifies to `u' / v`.
   - What's unclear: The constant-folding short-circuit in `Div` needs to distinguish three cases: both sentinel (result is constant), RHS sentinel only (simpler partial), LHS sentinel only (result has gradient only through `v`).
   - Recommendation: Implement all three cases explicitly in `Div` for correctness and to avoid pushing unnecessary tape nodes for constant-denominator division (common in softmin's `1/gamma` scaling).

---

## Environment Availability

Step 2.6: Not applicable — this phase has no external tool dependencies beyond the existing Rust toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (stable) | Compilation | ✓ | 1.97.0 (dev; MSRV 1.81) | — |
| `cargo test` | Tests | ✓ | bundled | — |
| `cargo clippy` | CI gate | ✓ | bundled | — |

---

## Security Domain

> `security_enforcement` not explicitly set in config. However, this phase is a pure numeric library addition with no I/O, no user input, no network, no auth, no secret handling, and no external data. ASVS categories do not apply.

No security domain concerns for this phase. The thread-local tape is local to the process; there is no data exfiltration vector.

---

## Project Constraints (from CLAUDE.md)

| Constraint | Impact on Phase 94 |
|---|---|
| No new crate dependency | Confirmed: tape is pure in-crate Rust. Cargo.toml unchanged. |
| Strictly additive / non-breaking | Module refactor preserves all public paths; `pub mod autodiff;` unchanged in lib.rs. |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Clippy runs on test code; `Var` impls must pass. Key: `#[must_use]` on `vjp` (matches `grad`/`diff` at autodiff.rs:483,510). |
| `cargo fmt --check` | Format all new files before committing. |
| `rust-version = "1.81"` (MSRV) | `thread_local!` with `const` initializer (`= const { RefCell::new(...) }`) is stable since Rust 1.21 — no MSRV issue. |
| All public types: `#[derive(Debug, Clone, PartialEq)]` | `Var` needs `Debug + Clone + Copy`. `PartialEq` must be hand-written (value-only), not derived. |
| `#[must_use]` on expensive computations (74+ functions) | Add `#[must_use]` to `vjp`. |
| GSD workflow enforcement | Use `/gsd-execute-phase` for execution, not direct edits. |

---

## Sources

### Primary (HIGH confidence — direct codebase reads this session)
- `fdars-core/src/autodiff.rs:110-156` — `Scalar` trait verbatim (12 methods + supertraits) [VERIFIED]
- `fdars-core/src/autodiff.rs:225-468` — `Dual` struct + all `Scalar` impls (structural template for `Var`) [VERIFIED]
- `fdars-core/src/autodiff.rs:511-535` — `grad` entry point (template for `vjp`) [VERIFIED]
- `fdars-core/src/autodiff.rs:629-1143` — tiered test structure (Tier 1–5, tolerances, composed test) [VERIFIED]
- `fdars-core/src/alignment/mod.rs:471-513` — `thread_local! { RefCell<Vec<...>> }` + `.with(|c| ...)` pattern proven rayon-compatible in production [VERIFIED]
- `fdars-core/src/metric/soft_dtw.rs:57-158` — `softmin3_generic<S: Scalar>` and `soft_dtw_distance_generic` signatures and implementation [VERIFIED]
- `fdars-core/src/regression.rs:232-251` — `project_scores_generic<S: Scalar>` signature and implementation [VERIFIED]
- `fdars-core/src/prelude.rs:21` — current autodiff prelude export line (extension target) [VERIFIED]
- `fdars-core/src/lib.rs:76` — `pub mod autodiff;` (unchanged after refactor) [VERIFIED]
- `fdars-core/Cargo.toml:34-46` — no new dependencies; MSRV 1.81 [VERIFIED]

### Secondary (MEDIUM confidence)
- Rufflewind (2016): "Reverse-mode automatic differentiation: a tutorial" — canonical Wengert-list node layout `{ deps: [usize; 2], weights: [f64; 2] }`, backward loop pattern. [CITED: https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation]

### Tertiary (LOW confidence)
- None required — all implementation decisions are grounded in codebase reads or the canonical external tutorial.

---

## Metadata

**Confidence breakdown:**
- Node record layout: HIGH — directly follows Rufflewind canonical + Dual structural template from codebase read
- Thread-local idiom: HIGH — verbatim from alignment/mod.rs already in production
- Backward pass algorithm: HIGH — standard Wengert-list reverse sweep, confirmed against codebase Dual tests
- Tolerances: HIGH — read directly from autodiff.rs test constants
- Validation targets: HIGH — both functions read directly from source

**Research date:** 2026-09-10
**Valid until:** Indefinite for the algorithmic patterns (Wengert-list AD is stable); re-verify if `Scalar` trait bounds change.
