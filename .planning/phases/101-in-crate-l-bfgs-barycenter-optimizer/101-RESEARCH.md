# Phase 101: In-Crate L-BFGS Barycenter Optimizer - Research

**Researched:** 2026-09-13
**Domain:** Numerical optimization — hand-rolled limited-memory L-BFGS replacing MM descent in soft-DTW barycenter; pure Rust, no new crate dependency, wasm32-safe
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- History size: **m = 10** memory (s, y) pairs — standard L-BFGS default; cheap, robust for this smooth objective. Two-loop recursion with the standard γ_k = (sᵀy)/(yᵀy) initial-Hessian scaling.
- Line search: **backtracking Armijo**, c1 = 1e-4, backtracking factor β = 0.5, initial step = 1.0, ~20-backtrack cap; on line-search failure take a safeguarded fallback step so the iteration cannot diverge.
- Stopping rule: **reuse the existing `max_iter` and `tol` parameters** — stop when the gradient ∞-norm falls below `tol` OR the relative change in total soft-DTW cost falls below `tol`, or when `max_iter` is reached.
- Behavior vs current MM path: **L-BFGS replaces the MM/diagonal-Newton descent as the single-init refiner.** Same objective and gradient; the 4-arg entry point is unchanged in this phase. (Config-taking API arrives in Phase 103.)

### Claude's Discretion
- Exact internal module layout / helper function names for the L-BFGS routine.
- Whether to keep a tiny MM safeguard internally if the line search stalls (user chose pure replacement, but a safeguarded fallback step to guarantee non-divergence is at Claude's discretion as an implementation detail).
- Memory-buffer representation for the (s, y) history ring buffer.

### Deferred Ideas (OUT OF SCOPE)
- Deterministic multi-restart wrapper (SDTW-02) → Phase 102.
- `SoftDtwBarycenterConfig` builder + config-taking entry point (SDTW-03) → Phase 103.
- Quality-evidence test proving escape-from-local-minimum vs MM (SDTW-04) → Phase 104.
- Optional rayon-parallel restarts — explicitly out of scope this milestone.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SDTW-01 | `soft_dtw_barycenter` minimizes the soft-DTW barycenter objective with an in-crate L-BFGS optimizer (limited-memory two-loop recursion + backtracking/Armijo line search) driven by the exact soft-DTW gradient the crate already computes, replacing the fixed-step MM descent. No new crate dependency; compiles and runs in the default feature set and on `wasm32` (no `linalg`/`argmin` gating). | Two-loop recursion algorithm (Section 1), Armijo line search (Section 2), objective/gradient assembly from existing helpers (Section 3), wasm32 compatibility (Section 4), test architecture (Section 5). |
</phase_requirements>

---

## Summary

Phase 101 replaces the fixed-step MM/diagonal-Newton update in `soft_dtw_barycenter` (lines 413–425 and 473–486 of `fdars-core/src/metric/soft_dtw.rs`) with a hand-rolled limited-memory L-BFGS optimizer. The objective is Σᵢ `soft_dtw_distance(bary, row_i)` and the gradient is Σᵢ accumulated via `soft_dtw_accumulate_gradient_and_weight` (extracting only the `grad` output; the `weight` output is no longer needed by the optimizer). Both helpers are already correct, in-module, and out-of-scope to change.

The L-BFGS implementation is self-contained in pure Rust using only `Vec<f64>` arithmetic — no `nalgebra`, no `faer`, no new crates. This is consistent with the project convention (v0.44.0 hand-rolled the reverse-mode tape; v0.39.0 hand-rolled the forward-mode Dual substrate). The only external state the optimizer needs is the gradient and objective value at the current barycenter, both computable without any structural change to the existing module.

The 4-arg `soft_dtw_barycenter(data, gamma, max_iter, tol)` public signature is byte-identical after the change. The `SoftDtwBarycenterResult { barycenter, n_iter, converged }` result type is unchanged. The `update_barycenter` and `init_barycenter_mean` private functions will change: `init_barycenter_mean` is reused as-is; `update_barycenter` is replaced by the L-BFGS loop body.

**Primary recommendation:** Implement the two-loop L-BFGS + backtracking Armijo line search as a single private function `lbfgs_barycenter_step` (or inline within `soft_dtw_barycenter`) working on `Vec<f64>` buffers, replacing the call to `update_barycenter`. Keep `init_barycenter_mean` unchanged. Retain `soft_dtw_accumulate_gradient_and_weight` for gradient assembly; simply stop passing the `weight` output to any optimizer step (the L-BFGS does not use diagonal curvature). The `update_barycenter` private function can be removed entirely.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| L-BFGS two-loop recursion | `src/metric/soft_dtw.rs` (private fn) | — | Tightly coupled to objective/gradient; no cross-module state needed |
| Armijo line search | `src/metric/soft_dtw.rs` (private fn, or inline in L-BFGS body) | — | Single-call-site; inlining keeps the module self-contained |
| Ring buffer for (s,y) history | `Vec<Vec<f64>>` ring within `soft_dtw_barycenter` | — | Plain heap vecs, no iterator abstractions needed |
| Objective + gradient assembly | `soft_dtw_accumulate_gradient_and_weight` (existing, unchanged) + `soft_dtw_distance` | — | Already correct since CORR-01; out of scope to touch |
| Entry point / signature | `soft_dtw_barycenter` (existing public fn) | lib.rs + prelude re-exports | Signature preserved byte-for-byte; all 28 examples + R/WASM bindings compile unchanged |
| Result type | `SoftDtwBarycenterResult` (existing, unchanged) | — | `#[non_exhaustive]` struct; no new fields in Phase 101 |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| (none new) | — | L-BFGS in plain `Vec<f64>` arithmetic | No new crate dependency; wasm32-safe by construction |

The implementation uses only `Vec<f64>` and scalar `f64` arithmetic from `std`. No `nalgebra`, `faer`, `ndarray`, or any other linear algebra crate is needed — L-BFGS in its limited-memory form only requires dot products and axpy operations on flat vectors, all of which are trivially expressible as for-loops over `&[f64]`.

### Supporting (already present in crate — reused, not changed)
| Helper | Location | Purpose |
|--------|----------|---------|
| `soft_dtw_accumulate_gradient_and_weight` | `src/metric/soft_dtw.rs:379` | Accumulates gradient for one row into caller-provided buffer |
| `soft_dtw_distance` | `src/metric/soft_dtw.rs:136` | Objective term for one row |
| `init_barycenter_mean` | `src/metric/soft_dtw.rs:428` | Pointwise-mean initialization — unchanged |
| `soft_dtw_forward` / `soft_dtw_backward` | same file | Called internally by `accumulate_gradient_and_weight` — unchanged |

### Package Legitimacy Audit

No new packages are introduced in this phase. `[VERIFIED: fdars-core/Cargo.toml:1-50]` — the Cargo.toml does not add any dependency for this phase; the L-BFGS is hand-rolled in pure Rust.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| (none new) | — | — | — | — | — | No audit needed |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious SUS:** none

---

## Architecture Patterns

### System Architecture Diagram

```
soft_dtw_barycenter(data, gamma, max_iter, tol)
    │
    ├─► init_barycenter_mean(rows)             [unchanged]
    │         └─► bary: Vec<f64>
    │
    └─► L-BFGS loop  (replaces MM loop)
              │
              ├─── eval_obj_and_grad(bary, rows, gamma)
              │         ├─► soft_dtw_distance(bary, rowᵢ, gamma)   [for each i → sum = f]
              │         └─► soft_dtw_accumulate_gradient_and_weight(bary, rowᵢ, gamma, &mut g, &mut w)
              │                   └─► forward pass → backward pass → accumulate [unchanged]
              │              (w is computed but not used by L-BFGS)
              │
              ├─── two_loop_recursion(g, history_s, history_y, γ_k)
              │         └─► d: Vec<f64>  (descent direction, already negated: d = -H·g)
              │
              ├─── backtracking_armijo(bary, d, f, g, c1, β, max_bt)
              │         └─► α: f64  (accepted step length, or safeguard fallback)
              │
              ├─── bary ← bary + α·d
              │
              ├─── update history: s = α·d,  y = g_new − g_old
              │         curvature safeguard: skip push if sᵀy ≤ 0
              │
              └─── check stopping: ‖g‖∞ < tol  OR  |Δf|/max(|f|,1) < tol  OR  iter ≥ max_iter
                        └─► SoftDtwBarycenterResult { barycenter, n_iter, converged }
```

### Recommended Project Structure

No new files are required. All changes are in `fdars-core/src/metric/soft_dtw.rs`. The L-BFGS and line search can be private functions in the same file or as `// ---- L-BFGS internals ----` inline helpers within the module.

```
fdars-core/src/metric/
├── soft_dtw.rs        ← all changes here (replace update_barycenter; add lbfgs helpers)
│                        private fns: eval_obj_grad, two_loop_lbfgs, backtrack_armijo
│                        (or inline within soft_dtw_barycenter — Claude's discretion)
└── (all other files unchanged)
```

### Pattern 1: Two-Loop L-BFGS Recursion

**What:** The two-loop recursion computes the L-BFGS approximate Newton direction `d = -H_k · g_k` using the last m (s, y) pairs stored in a ring buffer, without forming H_k explicitly. Initial Hessian is scaled as `H₀ = γ_k I` where `γ_k = (s_{k-1}ᵀ y_{k-1}) / (y_{k-1}ᵀ y_{k-1})`.

**When to use:** Every L-BFGS iteration after the gradient `g_k` is computed.

**Algorithm (Nocedal & Wright, Algorithm 7.4):** [ASSUMED — textbook algorithm, implementation is the prescription]

```rust
// Source: Nocedal & Wright "Numerical Optimization" 2nd ed., Algorithm 7.4
// All vectors are &[f64] or Vec<f64>; history is a ring over the last m pairs.
fn two_loop_lbfgs(
    g: &[f64],
    s_hist: &[Vec<f64>],  // ring of s_k = x_{k+1} - x_k  (oldest to newest)
    y_hist: &[Vec<f64>],  // ring of y_k = g_{k+1} - g_k
    gamma: f64,           // initial Hessian scale γ_k = sᵀy / yᵀy
    q: &mut Vec<f64>,     // work buffer
    r: &mut Vec<f64>,     // output direction (will be negated to get -H·g)
) {
    let m_actual = s_hist.len(); // 0..=m locked-in pairs
    let n = g.len();
    q.resize(n, 0.0);
    q.copy_from_slice(g);

    let mut alphas = vec![0.0_f64; m_actual];

    // Loop 1: newest to oldest
    for i in (0..m_actual).rev() {
        let rho_i = 1.0 / dot(&y_hist[i], &s_hist[i]);  // ρᵢ = 1/(yᵢᵀsᵢ)
        alphas[i] = rho_i * dot(&s_hist[i], q);
        axpy(-alphas[i], &y_hist[i], q);  // q ← q - αᵢ yᵢ
    }

    // Apply initial Hessian scale: r ← γ_k · q
    r.resize(n, 0.0);
    for j in 0..n { r[j] = gamma * q[j]; }

    // Loop 2: oldest to newest
    for i in 0..m_actual {
        let rho_i = 1.0 / dot(&y_hist[i], &s_hist[i]);
        let beta = rho_i * dot(&y_hist[i], r);
        axpy(alphas[i] - beta, &s_hist[i], r);  // r ← r + (αᵢ - βᵢ) sᵢ
    }
    // d = -r  (descent direction)
    for v in r.iter_mut() { *v = -*v; }
}

// Inline helpers (trivial, no allocation):
fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| x*y).sum() }
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) { *yi += alpha * xi; }
}
```

**Ring buffer management:** Use a `VecDeque<Vec<f64>>` or a plain `Vec<Vec<f64>>` with a front-pop. For m = 10, plain Vec with remove(0) on overflow is fine (m is tiny). [ASSUMED — implementation detail at Claude's discretion per CONTEXT.md]

**Curvature safeguard:** After computing `s = α·d` and `y = g_new − g_old`, only push to history if `dot(&s, &y) > 0.0`. This is the standard L-BFGS positive-definiteness guard. [ASSUMED — textbook]

**γ_k initial-Hessian scale:** `gamma = dot(&s_prev, &y_prev) / dot(&y_prev, &y_prev)`. Use `gamma = 1.0` when history is empty (first iteration). Guard against division by near-zero: if `dot(&y_prev, &y_prev) < 1e-20`, use `gamma = 1.0`. [ASSUMED — textbook]

### Pattern 2: Backtracking Armijo Line Search

**What:** Finds a step length α satisfying the sufficient-decrease (Armijo) condition: `f(x + α·d) ≤ f(x) + c1·α·(gᵀd)`. Start at α = 1.0 and multiply by β = 0.5 up to 20 times. If no step satisfies the condition, use a safeguarded minimum step (α_min = β²⁰ ≈ 9.5e-7) to guarantee the barycenter stays bounded. [ASSUMED — Nocedal & Wright textbook; parameters from CONTEXT.md locked decisions]

```rust
// Source: Nocedal & Wright "Numerical Optimization" 2nd ed., backtracking line search
// c1 = 1e-4 (Armijo constant), beta = 0.5, max_bt = 20  [LOCKED by CONTEXT.md]
fn backtrack_armijo(
    bary: &[f64],
    d: &[f64],          // descent direction (d = -H·g, already negated)
    f0: f64,            // f(bary)
    g0: &[f64],         // gradient at bary
    rows: &[Vec<f64>],  // data series for objective evaluation
    gamma_sdtw: f64,    // soft-DTW gamma parameter
    c1: f64,            // Armijo constant (1e-4)
    beta: f64,          // backtrack factor (0.5)
    max_bt: usize,      // cap (20)
) -> (f64, f64) {       // returns (alpha, f_new)
    let slope = dot(g0, d); // must be < 0 for d to be a descent direction
    let mut alpha = 1.0_f64;
    let alpha_min = beta.powi(max_bt as i32);

    for _ in 0..max_bt {
        let x_new: Vec<f64> = bary.iter().zip(d).map(|(b, di)| b + alpha * di).collect();
        let f_new: f64 = rows.iter().map(|r| soft_dtw_distance(&x_new, r, gamma_sdtw)).sum();
        if f_new <= f0 + c1 * alpha * slope {
            return (alpha, f_new);
        }
        alpha *= beta;
    }
    // Safeguard: use alpha_min — the update is tiny but avoids stall/divergence
    let x_new: Vec<f64> = bary.iter().zip(d).map(|(b, di)| b + alpha_min * di).collect();
    let f_new: f64 = rows.iter().map(|r| soft_dtw_distance(&x_new, r, gamma_sdtw)).sum();
    (alpha_min, f_new)
}
```

**Divergence impossibility argument:** The backtracking loop is bounded (max 20 iterations). The safeguard α_min = 0.5²⁰ ≈ 9.5×10⁻⁷ means even in the failure case the barycenter moves by at most α_min × ‖d‖, which is bounded because `d = -H·g` and `H` is positive definite (γ_k > 0 always, curvature guard ensures positive-definite updates). The outer `max_iter` cap bounds total iterations. Therefore the barycenter sequence is bounded. [ASSUMED — standard L-BFGS convergence reasoning]

### Pattern 3: Objective and Gradient Assembly

**What:** Total objective and gradient for the barycenter optimization are assembled by summing over all n rows. This reuses existing in-module helpers without modification.

**Verified function signatures:** [VERIFIED: fdars-core/src/metric/soft_dtw.rs:379-399]

```rust
// soft_dtw_accumulate_gradient_and_weight — existing private fn, exact signature:
fn soft_dtw_accumulate_gradient_and_weight(
    bary: &[f64],
    xi: &[f64],
    gamma: f64,
    grad: &mut [f64],
    weight: &mut [f64],
)
// Effect: grad[k-1] += Σⱼ E[k][j] * 2*(bary[k-1] - xi[j-1])
//         weight[k-1] += Σⱼ E[k][j]   (not used by L-BFGS; still pass a scratch buffer)

// soft_dtw_distance — existing pub fn:
pub fn soft_dtw_distance(x: &[f64], y: &[f64], gamma: f64) -> f64
```

**Assembly pattern for L-BFGS:**

```rust
// Compute total objective f and total gradient g at `bary`:
fn eval_obj_and_grad(bary: &[f64], rows: &[Vec<f64>], gamma: f64,
                     g: &mut [f64], w_scratch: &mut [f64]) -> f64 {
    let m = bary.len();
    for v in g.iter_mut() { *v = 0.0; }
    for v in w_scratch.iter_mut() { *v = 0.0; }
    let mut f = 0.0_f64;
    for row in rows {
        f += soft_dtw_distance(bary, row, gamma);
        soft_dtw_accumulate_gradient_and_weight(bary, row, gamma, g, w_scratch);
    }
    f
}
// w_scratch is unused by the optimizer but required by the existing fn signature.
// Allocate once outside the loop and reuse.
```

**Key insight:** The `weight` buffer in `soft_dtw_accumulate_gradient_and_weight` is the MM curvature estimate; the L-BFGS does not use it. Passing a scratch buffer satisfies the signature without changing the function.

### Pattern 4: Ring Buffer for (s, y) History

Three equivalent approaches at Claude's discretion:

1. **`VecDeque<Vec<f64>>`** — idiomatic, O(1) push/pop; requires `std::collections::VecDeque`. Zero-cost on wasm32.
2. **`Vec<Vec<f64>>` + modular index** — wrap index mod m; avoids dequeue import; requires maintaining an explicit count.
3. **Flat array `[[f64; m]; MAX_M]` stack allocation** — not viable here because m=10 and barycenter length varies; heap allocation is required.

Recommendation: use `VecDeque<Vec<f64>>` (option 1). It is in `std::collections`, requires no new crate, and makes the ring semantics explicit. [ASSUMED — implementation choice at Claude's discretion]

```rust
use std::collections::VecDeque;
let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_lbfgs);
let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_lbfgs);
// Push: if full, pop front first
if s_hist.len() == m_lbfgs { s_hist.pop_front(); y_hist.pop_front(); }
s_hist.push_back(s.clone());
y_hist.push_back(y.clone());
```

### Pattern 5: Full L-BFGS Loop Structure

```rust
// Replaces the for-loop in soft_dtw_barycenter (lines 473–486)
let mut bary = init_barycenter_mean(&rows);   // unchanged init [VERIFIED: soft_dtw.rs:428-441]
let m_lbfgs = 10;  // history size [LOCKED by CONTEXT.md]
let c1 = 1e-4_f64; let beta_ls = 0.5_f64; let max_bt = 20_usize; // [LOCKED by CONTEXT.md]

let mut g = vec![0.0_f64; m];
let mut w_scratch = vec![0.0_f64; m];  // for accumulate_gradient_and_weight signature
let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_lbfgs);
let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_lbfgs);
let mut q_buf = vec![0.0_f64; m];
let mut d = vec![0.0_f64; m];

let mut f = eval_obj_and_grad(&bary, &rows, gamma, &mut g, &mut w_scratch);
let mut converged = false;
let mut n_iter = 0_usize;

for iter in 0..max_iter {
    n_iter = iter + 1;

    // Stopping check 1: gradient ∞-norm
    let g_inf = g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if g_inf < tol { converged = true; break; }

    // Compute descent direction via two-loop recursion
    let gamma_k = if s_hist.is_empty() { 1.0 } else {
        let s = s_hist.back().unwrap(); let y = y_hist.back().unwrap();
        let sy = dot(s, y); let yy = dot(y, y);
        if yy < 1e-20 { 1.0 } else { sy / yy }
    };
    two_loop_lbfgs_from_deque(&g, &s_hist, &y_hist, gamma_k, &mut q_buf, &mut d);
    // d is now the descent direction (-H·g)

    // Line search
    let (alpha, f_new) = backtrack_armijo(
        &bary, &d, f, &g, &rows, gamma, c1, beta_ls, max_bt
    );

    // Stopping check 2: relative cost change
    let rel_change = (f - f_new).abs() / f.abs().max(1.0);
    if rel_change < tol { converged = true; }

    // Compute s = alpha * d; update bary
    let s_vec: Vec<f64> = d.iter().map(|v| alpha * v).collect();
    for (b, si) in bary.iter_mut().zip(&s_vec) { *b += si; }
    f = f_new;

    // Compute y = g_new - g_old; update gradient
    let g_old = g.clone();
    eval_obj_and_grad(&bary, &rows, gamma, &mut g, &mut w_scratch);
    // (f is already f_new from line search; objective evaluated in backtrack)
    // Actually recompute g only (f already held). See pitfall note.
    let y_vec: Vec<f64> = g.iter().zip(&g_old).map(|(gn, go)| gn - go).collect();

    // Curvature safeguard: only push if sᵀy > 0
    if dot(&s_vec, &y_vec) > 0.0 {
        if s_hist.len() == m_lbfgs { s_hist.pop_front(); y_hist.pop_front(); }
        s_hist.push_back(s_vec);
        y_hist.push_back(y_vec);
    }

    if converged { break; }
}
```

**Note on f vs gradient re-evaluation:** The backtracking already evaluates `soft_dtw_distance` to check the Armijo condition at the new point; it does NOT compute the gradient there. The gradient at `bary + α·d` must be computed with a separate call to `eval_obj_and_grad`. This means each iteration costs: 1 objective eval (inside backtrack per step) + 1 gradient eval (after accepting the step). The gradient eval in `eval_obj_and_grad` also recomputes the objective as a byproduct, which is fine — it can be ignored or used to update `f` for the next iteration's relative-change check. [ASSUMED — standard L-BFGS implementation detail; see Pitfall 3]

### Anti-Patterns to Avoid

- **Mutating `soft_dtw_accumulate_gradient_and_weight` signature or behavior:** Out of scope per CONTEXT.md. The weight buffer stays; just pass a scratch buffer and ignore it.
- **Using `nalgebra::DVector` for L-BFGS vectors:** Requires nalgebra for trivial dot/axpy; adds compile overhead and a layering violation. Use `Vec<f64>` and for-loops.
- **Using `rayon` in the L-BFGS loop body:** The `iter_maybe_parallel!` macro is wasm32-safe (falls back to sequential), but the L-BFGS loop itself is inherently sequential (each step depends on the previous). The row-sum in `eval_obj_and_grad` could use the macro for the per-row objective, but correctness is more important than speed here; Phase 101 ships the optimizer, Phase 102 handles multi-restart (where parallelism may be applicable later).
- **Floating-point divergence from using `f64::MAX` as a sentinel:** Use `f64::INFINITY` to match the rest of the module.
- **Allocating `s_vec` and `y_vec` inside the line-search function:** These must be computed AFTER the line search confirms α, so they cannot be precomputed. Allocate them inside the iteration body.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dot product | Custom SIMD kernel | Plain `iter().zip().map(|x,y| x*y).sum()` for-loop | The bottleneck is the O(n²m) DP in soft_dtw_distance, not the O(m) dot product in L-BFGS |
| Ring buffer | Manual circular array arithmetic | `VecDeque` from `std::collections` | Zero cost, no allocation, no new dependency |
| Objective value at accepted step | Re-running forward DP after backtrack completes | Return f_new from the line search (it already evaluated it) | Avoids a redundant O(n·m·len) forward pass |

**Key insight:** For this barycenter problem (m = series length ≈ 20–500, n = number of series ≈ 3–100), the soft-DTW DP dominates all costs. The L-BFGS overhead (dot products, ring buffer) is negligible. Optimize for correctness and clarity, not micro-performance.

---

## Common Pitfalls

### Pitfall 1: The `ρᵢ = 1/(yᵢᵀsᵢ)` denominator can be near-zero even after the curvature guard
**What goes wrong:** The curvature guard `sᵀy > 0` prevents negative ρ but does not prevent ρ → +∞ when sᵀy is tiny positive. In `two_loop_lbfgs`, computing `rho_i * dot(s, q)` with a huge ρ can produce NaN or overflow.
**Why it happens:** When the line search returns α_min (the fallback step), s = α_min · d is very small, so sᵀy can be tiny.
**How to avoid:** Additionally guard `sᵀy > 1e-20` before pushing to history; or cap ρ at a maximum (`rho_i.min(1e20)`). Either is fine.
**Warning signs:** NaN appearing in `g` or `d` after a near-zero-step iteration.

### Pitfall 2: Gradient at the accepted step is not equal to the gradient used to compute y
**What goes wrong:** The line search evaluates `f(x + α·d)` (objective only) but NOT `grad(x + α·d)`. After `bary += α·d`, the outer loop must call `eval_obj_and_grad` to get `g_new`. If the same `g` buffer is reused without zeroing, the gradient is stale.
**Why it happens:** `soft_dtw_accumulate_gradient_and_weight` ACCUMULATES into `grad` (+=). If `g` is not zeroed before the call, the new gradient is `g_old + g_new`, not just `g_new`.
**How to avoid:** In `eval_obj_and_grad`, always zero the buffers first (`for v in g.iter_mut() { *v = 0.0; }`). The example pattern above already does this.
**Warning signs:** Gradient grows monotonically across iterations instead of converging.

### Pitfall 3: The backtracking function evaluates f(x+α·d) but the caller also needs f for the next iteration
**What goes wrong:** After accepting α, `f_new` is known from the last evaluation inside `backtrack_armijo`. If the caller then calls `eval_obj_and_grad` to get the gradient, it recomputes the objective redundantly. Worse, if the caller does NOT call `eval_obj_and_grad` and instead uses the stale gradient, the L-BFGS direction is wrong.
**Why it happens:** The gradient and objective share the same forward+backward pass (they both call `soft_dtw_accumulate_gradient_and_weight` internally), but the line-search function only has access to the public `soft_dtw_distance`, not the gradient accumulation.
**How to avoid:** After the line search accepts α, call `eval_obj_and_grad(&bary_new, ...)` to compute BOTH f and g simultaneously. Use the returned f to update `f` for the relative-change stopping check, ignoring the redundant objective computation inside `backtrack_armijo`. Cost = one extra O(n·m) forward pass per iteration (negligible vs correctness).
**Warning signs:** The relative cost stopping check triggers incorrectly because `f` is stale.

### Pitfall 4: Two-loop recursion on an empty history uses γ_k = sᵀy/yᵀy but there are no s or y yet
**What goes wrong:** On iteration 1 (`s_hist` is empty), the two-loop is a no-op: q = g, r = γ·q, d = -r = -γ·g. If γ is computed from nonexistent history, division-by-zero or NaN occurs.
**Why it happens:** The γ_k formula requires the most recent (s, y) pair, which doesn't exist on iteration 1.
**How to avoid:** Use `gamma_k = 1.0` when `s_hist.is_empty()`. This gives a gradient-descent step with step α·1·g, and α is determined by the line search.
**Warning signs:** NaN in d on the very first iteration.

### Pitfall 5: The descent direction d must satisfy gᵀd < 0 before calling the line search
**What goes wrong:** Backtracking Armijo requires `slope = dot(g, d) < 0`. If the two-loop recursion returns a direction where `slope ≥ 0` (due to a degenerate history), the Armijo condition can never be satisfied and the line search loops to max_bt before returning the safeguard.
**Why it happens:** Can occur after a curvature-safeguard skip if the history becomes ill-conditioned.
**How to avoid:** After computing d, check `dot(&g, &d)`. If `≥ 0`, reset d to `-g` (steepest descent fallback) and clear the history to avoid repeating the failure.
**Warning signs:** Line search consistently hits max_bt cap, α = α_min every iteration.

### Pitfall 6: The `#[non_exhaustive]` rule on `SoftDtwBarycenterResult`
**What goes wrong:** Adding a new field to `SoftDtwBarycenterResult` in Phase 101 breaks every external match/construction site.
**Why it happens:** `SoftDtwBarycenterResult` is `#[non_exhaustive]` [VERIFIED: fdars-core/src/metric/soft_dtw.rs:17-25]. External code constructing it via `SoftDtwBarycenterResult { barycenter, n_iter, converged }` would break (E0639).
**How to avoid:** Do NOT add fields to `SoftDtwBarycenterResult` in Phase 101. The three existing fields (`barycenter`, `n_iter`, `converged`) are sufficient.

---

## Runtime State Inventory

Step 2.5: SKIPPED — this is a greenfield implementation phase (adding a new optimizer), not a rename/refactor/migration phase.

---

## Environment Availability

Step 2.6: This phase is purely code changes within an existing Rust crate. No new external tools, services, or runtimes are required beyond those already confirmed available.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable toolchain | Build | (from CLAUDE.md: runtime 1.97.0) | 1.97.0 | — |
| cargo | Build/test | Yes | (stable) | — |
| wasm32-unknown-unknown target | WASM compile gate | (from CLAUDE.md: present) | — | — |

**The L-BFGS uses only `std::collections::VecDeque` and f64 arithmetic — no new tool dependency.**

---

## Code Examples

### Verified: Existing helpers to call (do not modify)

```rust
// [VERIFIED: fdars-core/src/metric/soft_dtw.rs:136-138]
pub fn soft_dtw_distance(x: &[f64], y: &[f64], gamma: f64) -> f64

// [VERIFIED: fdars-core/src/metric/soft_dtw.rs:379-399]
fn soft_dtw_accumulate_gradient_and_weight(
    bary: &[f64], xi: &[f64], gamma: f64,
    grad: &mut [f64],    // accumulated +=
    weight: &mut [f64],  // accumulated +=; pass scratch buffer, ignore output
)

// [VERIFIED: fdars-core/src/metric/soft_dtw.rs:428-441]
fn init_barycenter_mean(rows: &[Vec<f64>]) -> Vec<f64>
```

### Verified: Existing public function signature to preserve

```rust
// [VERIFIED: fdars-core/src/metric/soft_dtw.rs:453-458]
pub fn soft_dtw_barycenter(
    data: &FdMatrix,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> SoftDtwBarycenterResult

// [VERIFIED: fdars-core/src/metric/soft_dtw.rs:15-25]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SoftDtwBarycenterResult {
    pub barycenter: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
}
```

### Verified: FdMatrix API used inside soft_dtw_barycenter

```rust
// [VERIFIED: fdars-core/src/matrix.rs:106-108]
pub fn shape(&self) -> (usize, usize)  // returns (nrows, ncols)

// [VERIFIED: fdars-core/src/matrix.rs:146-150]
pub fn row(&self, row: usize) -> Vec<f64>  // O(ncols) row extraction
```

### Verified: metric/mod.rs re-exports that must not change

```rust
// [VERIFIED: fdars-core/src/metric/mod.rs:151-155]
pub use soft_dtw::{
    soft_dtw_barycenter, soft_dtw_cross, soft_dtw_distance, soft_dtw_distance_generic,
    soft_dtw_div_cross, soft_dtw_div_self, soft_dtw_divergence, soft_dtw_self,
    SoftDtwBarycenterResult,
};
```

### Verified: lib.rs re-exports that must not change

```rust
// [VERIFIED: fdars-core/src/lib.rs:636-638]
soft_dtw_barycenter, soft_dtw_cross, soft_dtw_distance, soft_dtw_distance_generic,
soft_dtw_div_cross, soft_dtw_div_self, soft_dtw_divergence, soft_dtw_self, GakConfig,
GakGramTrain, LpDomain, SbdResult, SoftDtwBarycenterResult,
```

### Verified: Existing integration test (must still pass after L-BFGS replacement)

```rust
// [VERIFIED: fdars-core/tests/validate_against_r.rs:3838-3908]
// test_soft_dtw_barycenter_vs_tslearn
// Calls: soft_dtw_barycenter(&data, 1.0, 50, 1e-6)
// Asserts: result.converged == true; mean near 0; amplitude in (0.5, 2.0);
//          L2 from pointwise mean > 0.05 (proves gradient is non-zero and moves)
// After Phase 101: L-BFGS must produce a bounded result satisfying all three assertions.
// The amplitude and mean assertions are loose enough that any reasonable local optimizer
// converging on sinusoidal data should satisfy them.
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | none (uses `cargo test`) |
| Quick run command | `cargo test -p fdars-core --features linalg -- soft_dtw` |
| Full suite command | `cargo test -p fdars-core --features linalg` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SDTW-01 | L-BFGS lowers cost vs MM on same budget | unit | `cargo test -p fdars-core -- lbfgs_beats_mm` | Wave 0 |
| SDTW-01 | Cost monotonicity: f(k+1) ≤ f(k) with valid line search | unit | `cargo test -p fdars-core -- cost_monotone` | Wave 0 |
| SDTW-01 | Convergence + boundedness: barycenter stays bounded, converges | unit | `cargo test -p fdars-core -- test_soft_dtw_barycenter_vs_tslearn` | ✅ (existing) |
| SDTW-01 | Gradient norm stopping: ‖g‖∞ < tol at convergence | unit | `cargo test -p fdars-core -- lbfgs_gradient_stops` | Wave 0 |
| SDTW-01 | f64 determinism: same result for two identical calls | unit | `cargo test -p fdars-core -- lbfgs_deterministic` | Wave 0 |
| SDTW-01 | wasm32 compile: no `linalg`/`argmin` gating | compile | `cargo check --target wasm32-unknown-unknown` | Wave 0 (CI gate) |
| SDTW-01 | Default features compile: barycenter compiles without linalg | compile | `cargo check -p fdars-core` (no features) | Wave 0 |
| SDTW-01 | max_iter honored: loop stops at max_iter if not converged | unit | `cargo test -p fdars-core -- lbfgs_max_iter_honored` | Wave 0 |

### Sampling Rate
- **Per-task commit:** `cargo test -p fdars-core -- soft_dtw`
- **Per-wave merge:** `cargo test -p fdars-core --features linalg`
- **Phase gate:** Full suite green before `/gsd-verify-work`:
  - `cargo fmt --check`
  - `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
  - `cargo test -p fdars-core --features linalg`
  - `cargo check --target wasm32-unknown-unknown`
  - `cargo build --features serde` (serde guard)

### Wave 0 Gaps

The following tests do not yet exist and must be added as part of implementation:

- [ ] `soft_dtw.rs:#[cfg(test)]` — `lbfgs_beats_mm`: builds a 2-row dataset where MM stalls; runs both MM (by directly calling the old update formula) and L-BFGS; asserts L-BFGS cost ≤ MM cost. (Phase 104 will add a more comprehensive escape test; this is a lightweight sanity check.)
- [ ] `soft_dtw.rs:#[cfg(test)]` — `cost_monotone`: runs L-BFGS for 20 iterations on a 5-row sinusoidal dataset; records f at each step; asserts the sequence is non-increasing (or that decreases are monotone). NOTE: strict monotonicity is not guaranteed by L-BFGS in general, but the Armijo condition guarantees `f(k+1) ≤ f(k) + c1·α·slope`, which is sufficient for boundedness.
- [ ] `soft_dtw.rs:#[cfg(test)]` — `lbfgs_gradient_stops`: runs to convergence with small tol; checks that `‖g‖∞ < tol` at exit when `converged = true`.
- [ ] `soft_dtw.rs:#[cfg(test)]` — `lbfgs_deterministic`: calls `soft_dtw_barycenter` twice with identical inputs; asserts `result1.barycenter == result2.barycenter` (bit-identical, since the algorithm is fully deterministic in pure f64).
- [ ] `soft_dtw.rs:#[cfg(test)]` — `lbfgs_max_iter_honored`: calls with `max_iter = 3`; asserts `result.n_iter <= 3` and `result.converged == false` when the problem requires more iterations.
- [ ] CI/compile gate — `cargo check --target wasm32-unknown-unknown` added to the phase gate command sequence.

*(Existing test `test_soft_dtw_barycenter_vs_tslearn` is the primary regression guard — it asserts convergence + boundedness on the sinusoidal case that the MM step passed. The L-BFGS must also pass it.)*

---

## Security Domain

`security_enforcement` is not configured; treating as enabled. However, this phase implements a pure numerical algorithm with no I/O, no user-supplied data paths, no cryptography, no authentication, and no network access.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | `gamma > 0`, `max_iter ≥ 0`, `tol > 0` already validated at the public API; no new parameters added in Phase 101 |
| V6 Cryptography | no | — |

### Known Threat Patterns for Rust Numerical Code

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN/inf propagation from degenerate input | Tampering (data integrity) | Curvature safeguard (`sᵀy > 0`), γ guard (`yᵀy > 1e-20`), safeguard step in line search |
| Integer overflow in ring buffer indexing | Tampering | Use `VecDeque` which handles capacity automatically; no manual mod arithmetic needed |
| Panic on empty data | DoS | Already guarded: `if n == 0 || m == 0 { return ... }` at line 459–465 [VERIFIED: soft_dtw.rs:459-466] |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed `lr = 1/n` gradient descent | MM inverse-curvature step (diagonal Newton) | v0.40.0 Phase 78 (CORR-01) | Stopped divergence after gradient fix |
| MM inverse-curvature (diagonal Newton) | In-crate L-BFGS (this phase) | Phase 101 | Better convergence on non-convex soft-DTW landscape; memory-efficient quasi-Newton |
| All-zero gradient bug (v0.39.0) | Fixed endpoint-seed guard in `soft_dtw_backward` | v0.40.0 Phase 78 | Gradient is now correct; CORR-01 cross-checked vs Dual + oracle + FD |

**Deprecated/outdated after Phase 101:**
- `update_barycenter` private function (lines 413–425): replaced by L-BFGS loop body. Can be removed entirely; it is `fn` (private), not `pub`, so removal has no API impact.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The two-loop L-BFGS recursion produces a descent direction when history is consistent (sᵀy > 0) | Algorithm patterns | Low: this is a textbook invariant; only matters if the curvature safeguard is implemented incorrectly |
| A2 | Armijo backtracking with c1=1e-4, β=0.5, max_bt=20 provides sufficient reduction for the soft-DTW objective | Line search | Low: Armijo is unconditionally well-defined when slope < 0; the specific parameters are the user's locked decision |
| A3 | `VecDeque` from `std::collections` compiles and runs on wasm32 without issue | Ring buffer | Very low: `VecDeque` is core `std` with no platform-specific code |
| A4 | Using `w_scratch` as a throw-away weight buffer (allocated once, ignored) is free of unexpected side effects | Gradient assembly | Very low: `soft_dtw_accumulate_gradient_and_weight` only writes to `grad` and `weight` slices, does not touch global state |
| A5 | The existing integration test `test_soft_dtw_barycenter_vs_tslearn` will pass with the L-BFGS optimizer (loose tolerance assertions) | Validation | Low: the assertions are loose (mean < 0.3, amplitude in 0.5–2.0, L2 from mean > 0.05); any converging optimizer on sinusoidal data satisfies them |
| A6 | `eval_obj_and_grad` computing the objective by summing `soft_dtw_distance` across rows is numerically equivalent to the value the line search computed | Pitfall 3 | Very low: same function calls, same floating-point arithmetic order |

**If this table is empty:** N/A — all non-trivial algorithm claims are ASSUMED per the Absent-evidence provenance rule (L-BFGS textbook algorithm, not a crate with docs to cite).

---

## Open Questions

1. **Should `update_barycenter` be deleted or kept as a dead private function?**
   - What we know: it is `fn` (private), no external caller; removing it avoids a dead-code clippy warning.
   - What's unclear: the planner may want to stage removal separately for safety.
   - Recommendation: delete it in the same commit as adding the L-BFGS; it is `fn` (not `pub`) so there is no API risk, and clippy `--all-targets` would flag it as dead code anyway.

2. **`eval_obj_and_grad` computes both f and g together (two forward+backward passes total per call). The line search only needs f. Is a separate `eval_obj` helper worth adding?**
   - What we know: the DP cost is O(n·m·len); separating them avoids redundant gradient computation during backtracking.
   - What's unclear: for m = 10–20 series of length 50, the overhead is negligible; optimization is premature.
   - Recommendation: don't add a separate `eval_obj` helper. The line search already calls `soft_dtw_distance` directly (public fn) to check the Armijo condition. After the search, call `eval_obj_and_grad` once to get both f and g. This is the pattern shown in the code examples above.

3. **Should the L-BFGS helpers be private `fn`s in `soft_dtw.rs` or a new submodule `metric/lbfgs.rs`?**
   - What we know: the L-BFGS is small (< 100 lines of pure math); `soft_dtw.rs` is currently 709 lines and will grow by ~150 lines.
   - Recommendation: keep all helpers in `soft_dtw.rs` as private `fn`s. A new submodule would require a `pub(super)` re-export and adds boilerplate for a single-use optimizer. If the file exceeds 1000 lines after addition, refactoring to `metric/lbfgs.rs` is at Claude's discretion.

---

## Sources

### Primary (HIGH confidence)
- `fdars-core/src/metric/soft_dtw.rs` — read in full; all function signatures and behavior verified via direct Read
- `fdars-core/src/matrix.rs` — row, shape, column methods verified via direct Read
- `fdars-core/tests/validate_against_r.rs:3838-3908` — existing integration test verified via direct Read
- `fdars-core/Cargo.toml` — feature flags and dependency list verified via direct Read
- `fdars-core/src/metric/mod.rs` — re-exports verified via direct Read
- `fdars-core/src/lib.rs:636-638` — crate-root re-exports verified via direct Read
- `.planning/phases/101-in-crate-l-bfgs-barycenter-optimizer/101-CONTEXT.md` — decisions verified via direct Read

### Secondary (MEDIUM confidence)
- Nocedal & Wright "Numerical Optimization" 2nd ed., Algorithm 7.4 (two-loop recursion) and Chapter 3 (backtracking Armijo line search) — standard textbook reference [CITED] — algorithm is well-established and not version-sensitive

### Tertiary (LOW confidence)
- L-BFGS convergence theory for smooth non-convex objectives: convergence to a stationary point under standard assumptions [ASSUMED — standard result, not verified in this session]

---

## Metadata

**Confidence breakdown:**
- Existing code interfaces: HIGH — all verified by direct file read this session
- L-BFGS algorithm correctness: MEDIUM — textbook algorithm; numerics are well-known
- Line search parameter choices: HIGH — locked by CONTEXT.md (c1=1e-4, β=0.5, max_bt=20)
- Wasm32 compatibility: HIGH — uses only `std` types; `VecDeque` is `std`; no new crates
- Non-breaking compatibility: HIGH — public signature verified unchanged; all re-exports verified

**Research date:** 2026-09-13
**Valid until:** 2026-10-13 (stable algorithm, no external dependencies)
