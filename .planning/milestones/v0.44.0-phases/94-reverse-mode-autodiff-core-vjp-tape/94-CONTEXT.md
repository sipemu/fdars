# Phase 94: Reverse-Mode Autodiff Core (VJP Tape) - Context

**Gathered:** 2026-09-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver an in-crate reverse-mode (vector-Jacobian-product) automatic-differentiation core alongside the shipped v0.39.0 forward-mode `Dual`. The core is a hand-written Wengert-list tape recording a `Var` scalar type that supports the full forward-mode operation set; a backward pass seeds the output adjoint and accumulates input gradients, exposed through a `vjp` entry point efficient for many-input→scalar objectives. Reverse-mode gradients are validated against the forward-mode `Dual` path and central finite differences on the *existing* differentiable subset (elastic soft-DTW distance + FPCA scores). No new crate dependency.

**In scope:** `Var` scalar type + `Tape`, full op set, backward pass, `vjp` entry point, validation vs `Dual`/FD on soft-DTW + FPCA scores, module refactor into an `autodiff/` directory, prelude re-exports.

**Out of scope (deferred):** generalizing additional hot-path signatures (Phase 95 / GEN-01), the four DOP families (96–98), the unified `grad`/`jacobian`/`vjp` demo API (Phase 99 / API-01), differentiating the discrete `elastic_distance` DP argmin (permanently deferred, DIF-02), `soft_dtw_barycenter` optimizer redesign (SDTW-O1). A reverse-mode `jacobian` (vector-output) is out of scope for RAD-02 — `vjp` (many-input→scalar) only.
</domain>

<decisions>
## Implementation Decisions

### Tape & `Var` Architecture
- **`Var` implements the existing `Scalar` trait** (autodiff.rs:110–156, 12 methods + Copy/Add/Sub/Mul/Div/Neg/AddAssign/SubAssign/MulAssign supertraits). This is the pivotal decision: it lets `soft_dtw_distance_generic<S: Scalar>` (metric/soft_dtw.rs:158) and `project_scores_generic<S: Scalar>` (regression.rs:232) run on `Var` **unchanged**, giving a direct path to RAD-03 validation.
- **Thread-local tape** access model. Because `Scalar: Copy` forces `Var: Copy`, and `Scalar::from_f64`/`zero`/`one`/`infinity` take no tape argument, `Var` must be a small `Copy` handle — `Var { value: f64, node: usize }` — that reaches the recording tape via a thread-local, not a stored borrow. (Rejected: `&Tape` borrow `Var<'t>` — breaks the tape-less `from_f64`/`zero`/`one` constructors; `Rc<RefCell<Tape>>` — runtime cost.)
- **Constants use a sentinel node index** (e.g. `usize::MAX`) meaning "off-tape / zero adjoint contribution". `from_f64`/`zero`/`one`/`infinity` produce off-tape constants; the backward pass skips sentinel parents. (Rejected: pushing a real zero-adjoint node per constant — tape bloat.)
- Value-only `PartialEq`/`PartialOrd` on `Var` (primal decides control flow), mirroring `Dual`'s semantics.

### Backward Pass, API Surface & Validation
- **`vjp` mirrors `grad`'s signature:** `vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)`. Runs the closure once to build the tape, seeds the output adjoint = 1.0, reverse-iterates the Wengert list accumulating input adjoints in a `Vec<f64>` indexed by node. This is **one** backward pass vs forward-mode's m passes — the efficiency win for many-input→scalar objectives (RAD-02).
- **`vjp` (many-input→scalar) only** for RAD-02 — a reverse-mode `jacobian` (one backward per output) is out of scope this phase; vector-output stays with forward-mode `jacobian`.
- Each tape node records the op's local partial derivative(s) w.r.t. its parent(s) (unary: 1 parent; binary: 2 parents) plus parent node indices; backward accumulates `adjoint[parent] += adjoint[node] * local_partial`.
- **f64 stays the non-tape passthrough** — no separate f64 reverse path. Parity is checked as reverse-vs-`Dual` and reverse-vs-FD; the f64 `Scalar` impl is unchanged.

### Module Layout (user decision)
- **Refactor into an `autodiff/` directory:** `autodiff/mod.rs` (re-exports + the `Scalar` trait + shared `grad`/`jacobian`/`diff`/`directional_derivative` entry points), `autodiff/forward.rs` (move `Dual` + its impls), `autodiff/reverse.rs` (`Var`, `Tape`, `vjp`, backward pass). Public paths **preserved** — `autodiff::Dual`, `autodiff::Var`, `autodiff::Scalar`, `autodiff::vjp` all resolve; `lib.rs:76` `pub mod autodiff;` unchanged. Use `git mv`-style moves to keep history.
- **Type name `Var`** for the reverse-mode scalar (per roadmap goal wording); tape type `Tape`; backward internal.
- **Prelude re-exports:** add `Var`, `Tape`, `vjp` to `prelude.rs:21` alongside the forward-mode exports.

### Tests & Tolerances (mirror the existing tiered layout in autodiff.rs:629–1143)
- Tier 1 known-answer unit derivatives for every `Var` op, tol 1e-10.
- Guard tests locking singular-point (sqrt/ln/powf at 0) NaN/Inf adjoint behavior, identical to forward-mode.
- Reverse-vs-`Dual` agreement, bit-close (1e-10/1e-12).
- Reverse-vs-central-FD cross-check, tol 1e-6 (h=1e-8 unit; h=1e-6 composed), on `soft_dtw_distance_generic` and `project_scores_generic`.
- A composed-objective test mirroring `grad_composed_objective_matches_finite_diff` (autodiff.rs:1029): `soft_dtw(curve, ref) + λ·Σ(scores²)` gradient via `vjp`, FD-checked per component.
- Module/function doctests for `vjp` and `Var` using `use fdars_core::prelude::*;` / `use fdars_core::autodiff::...`.

### Claude's Discretion
- Exact node record layout (enum vs struct-of-arrays), tape growth strategy, thread-local reset/scoping ergonomics, whether `Tape` is publicly constructible or fully hidden behind `vjp`, and internal helper naming — all at Claude's discretion, guided by the `Dual` conventions and the no-new-dependency / non-breaking constraints.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`Scalar` trait** — `fdars-core/src/autodiff.rs:110–156`. 12 methods: `zero`, `one`, `from_f64`, `infinity`, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, `signum`. Supertraits: `Copy + Clone + Debug + PartialOrd + Add/Sub/Mul/Div/Neg(Output=Self) + AddAssign + SubAssign + MulAssign`. `Var` must satisfy all of these.
- **`Dual` type** — `autodiff.rs:225–262` (fields `value`, `tangent`; `seed`/`constant`/`extract`). Op impls at 264–468. The structural template for `Var`.
- **Entry points** — `diff` (484), `grad` (511), `jacobian` (559), `directional_derivative` (611). `vjp` mirrors `grad`.
- **`soft_dtw_distance_generic<S: Scalar>(x:&[S], y:&[S], gamma:f64) -> S`** — `metric/soft_dtw.rs:158`. Already generic; validation target #1. DP kernel `soft_dtw_distance_inner<S>` (102), softmin `softmin3_generic<S>` (57).
- **`project_scores_generic<S: Scalar>(curve:&[S], mean:&[f64], rotation:&FdMatrix, weights:&[f64], ncomp:usize) -> Vec<S>`** — `regression.rs:232`. Already generic; validation target #2. Model built by `fdata_to_pc` (regression.rs:387).

### Established Patterns
- `#[cfg(test)] mod tests` inline at module bottom; `const TOL`; tiered tests (known-answer 1e-10 → central FD 1e-6 → f64 parity bit-exact → composed objective).
- `central_fd` helper h=1e-8 (autodiff.rs:895); composed test h=1e-6 (1111).
- All public functions return values directly here (AD is infallible on in-domain input); domain restrictions are caller-owned, matching `Dual`.
- `#[derive(Debug, Clone, Copy)]` on scalar types.

### Integration Points
- `lib.rs:76` — `pub mod autodiff;` (unchanged after dir refactor).
- `prelude.rs:21` — `pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};` → extend with `Var, Tape, vjp`.
- No downstream code references the internal layout of autodiff.rs, so the dir split is safe as long as public re-exports are preserved.

</code_context>

<specifics>
## Specific Ideas

- Reverse-mode `Var` MUST implement the *same* `Scalar` trait so the two already-generic subset functions (`soft_dtw_distance_generic`, `project_scores_generic`) validate RAD-03 with zero call-site changes.
- The `vjp` efficiency claim (single backward pass) is the reason reverse-mode exists here vs forward-mode's per-input passes — the backward pass must genuinely accumulate all input gradients in one reverse sweep.
- Thread-local tape is the enabling trick that keeps `Var: Copy` while giving the tape-less `Scalar` constructors somewhere to record.
</specifics>

<deferred>
## Deferred Ideas

- Reverse-mode `jacobian` (vector-output, one backward pass per output row) — not needed for RAD-02; could be added in Phase 99 if the unified API wants it.
- Differentiating the warp-searched `elastic_distance` DP argmin — permanently deferred (DIF-02); soft-DTW / amplitude-at-warp surrogates remain the differentiable paths.
- Generalizing further hot-path signatures over the scalar type — Phase 95 (GEN-01).
</deferred>
