# Phase 78: Gradient Correctness — soft_dtw Fix & Backward-Pass Audit — Research

**Researched:** 2026-09-06
**Domain:** Rust codebase — hand-written backward/gradient passes in `fdars-core`
**Confidence:** HIGH (all claims verified by reading source files this session)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Fix Scope & Audit Rigor — CORR-02 sweep findings:** Fix small, localized, behavior-preserving bugs in-phase (analogous to the endpoint-seed one-liner). Defer anything requiring a redesign to the backlog with a logged item — do not expand this phase into an algorithm rewrite.
- **Audit artifact:** Record the CORR-02 clean/fixed disposition table in a dedicated `78-AUDIT.md` in the phase directory, referenced from the phase SUMMARY, so the audit trail survives milestone archival.
- **soft_dtw test assertions (SC #1):** Assert all three of — (a) non-zero backward `E` / accumulated gradient on non-identical input; (b) `soft_dtw_barycenter` on non-identical curves lands measurably far (clear L2 margin) from the pointwise mean; (c) the shipped gradient matches the `corrected_oracle_gradient` / v0.39.0 `Dual` path within a tight relative tolerance (~1e-6).
- **`corrected_oracle_gradient` helper:** Keep it as the independent SC#1 cross-check reference. Once the shipped code is fixed (shipped == oracle), update its doc comment so it no longer describes a live production bug.

### Claude's Discretion
- Exact tolerance/margin constants, test data (non-identical curve construction), and per-sibling audit rationale wording are at Claude's discretion, guided by the reference oracle and existing test conventions.

### Deferred Ideas (OUT OF SCOPE)
- Any CORR-02 sibling bug that would require an algorithm redesign (not a localized boundary-seed fix) → logged to backlog, not fixed in this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CORR-01 | `soft_dtw_backward` no longer overwrites the `E[n][m]=1.0` endpoint seed; non-zero gradient produced; `soft_dtw_barycenter` refines; regression test cross-checked against oracle/Dual; existing tests tightened | Bug location pinned to line 302 in `soft_dtw.rs`; oracle fix at lines 483–486; existing tests documented with weakness analysis |
| CORR-02 | Every hand-written backward/gradient pass audited for boundary-seed / correctness bugs; each disposed clean/fixed with traceable rationale | All 9 target modules located and audited below with file paths, function names, line ranges, and preliminary disposition |
</phase_requirements>

---

## Summary

Phase 78 is a pure codebase surgery task. All research is derived from reading the actual source files; no external documentation is needed. The two requirements decompose into a concrete one-line fix plus a full sweep.

**CORR-01** is fully understood. The bug lives in `soft_dtw_backward` at `fdars-core/src/metric/soft_dtw.rs:302` — the line `e[i][j] = a + b + c;` executes even when `i == n && j == m`, overwriting the manually set `e[n][m] = 1.0` seed with `a + b + c = 0` (all three neighbour guard conditions are false at the endpoint). The fix is a single guard: skip that write when `i == n && j == m`. The corrected pattern is already prototyped in the same file's `corrected_oracle_gradient` test helper at lines 483–486. The shipped exponent formula in the production code is algebraically equivalent to the oracle's (the `r[i][j] - r[i+1][j] + r[i+1][j]` terms cancel to `r[i][j]`), so only the skip guard needs to be added.

**CORR-02** sweeps nine target modules. Based on reading each file, the preliminary disposition is: six are clean (no boundary-seed pattern; they are pure formula-level gradients, finite-difference approximations, or forward-mode AD delegating to `Dual`), two require careful inspection to confirm (gamlss and elastic logistic compute multi-step gradient descents with several internal functions), and `metric/soft_dtw` is the known bug. None of the sibling modules contains a backward DP pass with an endpoint seed analogous to soft_dtw's — the pattern is unique to the DTW backward algorithm.

**Primary recommendation:** Implement CORR-01 as the single-line guard in `soft_dtw_backward`, write the three-part regression test, tighten the existing barycenter tests, update `corrected_oracle_gradient`'s doc comment, then walk through CORR-02 to confirm and record the clean/fixed disposition table in `78-AUDIT.md`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| soft_dtw backward pass fix | `metric/soft_dtw.rs` | `metric/tests.rs` (unit), `tests/validate_against_r.rs` (integration) | Bug and fix are self-contained in one private function; callers (`soft_dtw_accumulate_gradient`, `soft_dtw_barycenter`) need no change |
| Regression test for CORR-01 | `metric/soft_dtw.rs` inline `#[cfg(test)]` block | optionally `metric/tests.rs` | Existing differentiable tests are in an inline `mod differentiable_tests`; the new regression test belongs in `metric/tests.rs` or a new inline module to keep the production code file from growing |
| CORR-02 audit artifact | `.planning/phases/78-…/78-AUDIT.md` | Phase SUMMARY | Disposition table is a planning artifact, not code |
| Tighten existing barycenter tests | `metric/tests.rs:795–840` | None | Both tests are in `metric/tests.rs`; they need additional non-zero gradient / non-mean-convergence assertions |
| Whole-crate gates | CI-equivalent `cargo` commands | None | `cargo fmt`, `cargo clippy --all-targets`, `cargo test` |

---

## CORR-01: Exact Bug Anatomy

### File and Bug Location

**File:** `fdars-core/src/metric/soft_dtw.rs`
[VERIFIED: fdars-core/src/metric/soft_dtw.rs:262-306]

The buggy function signature: `fn soft_dtw_backward(x: &[f64], y: &[f64], r: &[Vec<f64>], gamma: f64) -> Vec<Vec<f64>>`

**Seed line (correct):** line 267 — `e[n][m] = 1.0;`

**Bug line:** line 302 — `e[i][j] = a + b + c;` — this unconditional write executes when `i == n && j == m` (the first iteration of the reverse double-loop, since the loop runs `for i in (1..=n).rev()` and `for j in (1..=m).rev()`). At the endpoint all three neighbour contributions (`a`, `b`, `c`) are gated off by `if i < n`, `if j < m`, `if i < n && j < m` and therefore equal 0.0. The write `e[n][m] = 0.0 + 0.0 + 0.0` destroys the seed.

**Algebraic exponent collapse confirmation:**
[VERIFIED: fdars-core/src/metric/soft_dtw.rs:276-279]
The production code computes:
```
r[i][j] - r[i+1][j] + r[i+1][j] - softmin3_val(r, i+1, j, gamma)
```
This simplifies to `r[i][j] - softmin3_val(r, i+1, j, gamma)` — identical to the oracle's formula (line 489: `-(r[i][j] - softmin3_val(&r, i + 1, j, gamma)) / gamma`). The exponent is correct; only the missing guard is the problem.

### The Fix (one line)

Insert between lines 272–273 (inside `for i ... for j` and before the `a`/`b`/`c` computation):

```rust
// Preserve the E[n][m] = 1.0 endpoint seed: the endpoint has no
// downstream neighbours (a = b = c = 0 there), so the unconditional
// write `e[i][j] = a + b + c` would zero it. Skip it.
if i == n && j == m {
    continue;
}
```

This is exactly the pattern at `corrected_oracle_gradient` lines 485–486:
[VERIFIED: fdars-core/src/metric/soft_dtw.rs:483-486]
```rust
if i == n && j == m {
    continue; // preserve the E[n][m] = 1.0 seed
}
```

### Caller Chain

[VERIFIED: fdars-core/src/metric/soft_dtw.rs:326-337]
```
soft_dtw_accumulate_gradient(bary, xi, gamma, grad)
  → r = soft_dtw_forward(bary, xi, gamma)           // line 328
  → e = soft_dtw_backward(bary, xi, &r, gamma)      // line 329
  → accumulate gradient from e (lines 330–336)
```

[VERIFIED: fdars-core/src/metric/soft_dtw.rs:405-407]
```
soft_dtw_barycenter(data, gamma, max_iter, tol)
  → for row in &rows: soft_dtw_accumulate_gradient(&bary, row, gamma, &mut grad)
```

The fix propagates automatically — `barycenter` does not need modification.

### Independent Reference Paths

**Dual / `soft_dtw_distance_generic`:**
[VERIFIED: fdars-core/src/metric/soft_dtw.rs:157-159]
`pub fn soft_dtw_distance_generic<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S` delegates to `soft_dtw_distance_inner`. It is the independent forward-mode AD reference — seeding `x[k]` via `Dual::seed` and reading `.tangent` yields the exact gradient component. This path is validated in `dual_gradient_vs_fd` (lines 559–580) and already cross-checked against `corrected_oracle_gradient` in `dual_gradient_vs_oracle` (lines 519–533).

**`corrected_oracle_gradient`:**
[VERIFIED: fdars-core/src/metric/soft_dtw.rs:467-517]
Lives in `mod differentiable_tests` (line 423). Takes `(bary, xi, gamma)` and returns `Vec<f64>`. Its endpoint skip is at lines 485–486. Once the production code is fixed, it becomes a redundant-but-retained reference for the cross-check test.

---

## CORR-01: Existing Test Weakness Analysis

### `test_soft_dtw_barycenter_identical`
[VERIFIED: fdars-core/src/metric/tests.rs:795-818]

Uses 5 identical copies of a sinusoid. **Passes on all-zero gradient** because:
- The pointwise mean of identical series equals each series exactly
- `update_barycenter` with `grad = [0.0; m]` performs no update, leaving `bary` unchanged at the mean
- The assertion checks `(result.barycenter[j] - series[j]).abs() < 0.5` — satisfied trivially when `bary` never moves from the mean (which equals `series`)

**Tightening needed:** Add an assertion that `soft_dtw_accumulate_gradient` returns a non-all-zero gradient. For identical series, the gradient should be zero (the barycenter is already a minimiser), so this test cannot be the zero-gradient regression test — it is structurally degenerate. It should instead be tightened with a note that identical-series gradient IS zero by design.

### `test_soft_dtw_barycenter_shifted`
[VERIFIED: fdars-core/src/metric/tests.rs:821-840]

Uses 3 series: `sin(2πt) + i` for `i ∈ {0, 1, 2}`. Checks `(mean_val - 1.0).abs() < 0.5`. **Passes on all-zero gradient** because:
- Initial barycenter = pointwise mean = `sin(2πt) + 1`
- The mean of `sin(2πt) + 1` over `m=20` points has `mean_val ≈ 1.0`
- With zero gradient, `bary` never moves; `mean_val` stays ≈ 1.0; assertion passes

**Tightening needed:** (a) assert the accumulated gradient vector is not all-zero; (b) assert the final barycenter differs from the pointwise mean by a minimum L2 distance (e.g. > 0.01 per point, or Euclidean norm > some threshold) — or alternatively check convergence happened measurably faster than with a zero gradient.

### `test_soft_dtw_barycenter_vs_tslearn`
[VERIFIED: fdars-core/tests/validate_against_r.rs:3802-3836]

Uses 3 mildly shifted sinusoids (`shifts = [0.0, 0.05, -0.05]`). Checks `mean_val.abs() < 0.3` and `max_abs > 0.5`. Both assertions pass on all-zero gradient (barycenter stays at pointwise mean which has near-zero mean and amplitude ≈ 1.0). **Needs tightening** — add non-zero gradient and non-mean-barycenter assertions.

---

## CORR-02: Audit Sweep — Per-Module Map

### 1. `metric/soft_dtw` — THE KNOWN BUG

**File:** `fdars-core/src/metric/soft_dtw.rs`
**Function:** `fn soft_dtw_backward` (lines 262–306) — private
**Pattern:** DP backward pass with E matrix seeded at endpoint `E[n][m] = 1.0`
**Boundary/seed handling:** Seed set at line 267; overwritten at line 302 (the bug)
**Preliminary disposition:** **FIXED** (CORR-01)
[VERIFIED: fdars-core/src/metric/soft_dtw.rs:262-306]

---

### 2. `alignment/differentiable`

**File:** `fdars-core/src/alignment/differentiable.rs`
**Functions:** `generic_linear_interp` (lines 33–55), `generic_srsf_central_diff` (lines 64–87), `generic_l2_srsf_distance` (lines 93–100), `amplitude_distance_at_warp_generic` (lines 125–138) — all private
**Pattern:** Forward-mode generic pipeline using `Scalar`/`Dual` — no hand-written backward pass. The gradient flows automatically through the `Dual` chain rule.
**Boundary/seed handling:** Boundary conditions appear in `generic_linear_interp` (clamping at endpoints, lines 38–40), and `generic_srsf_central_diff` (forward/backward one-sided differences at `j == 0` and `j == m-1`, lines 76–82). These are value-level boundary conditions for interpolation/differentiation, NOT gradient-seed boundaries. No DP endpoint seed pattern exists.
**Preliminary disposition:** **CLEAN** — forward-mode AD with no hand-written backward pass. Boundary handling is numerically correct. Validated by `amplitude_gradient_vs_fd` test (lines 200–236).
[VERIFIED: fdars-core/src/alignment/differentiable.rs:1-238]

---

### 3. `autodiff`

**File:** `fdars-core/src/autodiff.rs`
**Functions:** `diff` (line 484), `grad` (line 511), `jacobian` (line 559), `directional_derivative` (line 611). The `Dual` struct and `Scalar` trait (lines 100–262). Operator impls (lines 264–477).
**Pattern:** Forward-mode automatic differentiation implementation. There is NO hand-written backward pass — this module IS the forward-mode substrate. Backward-mode (reverse) AD is explicitly deferred (DIF-F1).
**Boundary/seed handling:** `Dual::seed` (line 239) and `Dual::constant` (line 248) are the seeding primitives. They are straightforward constructors with no boundary logic.
**Preliminary disposition:** **CLEAN** — the module is the forward-mode foundation; no backward pass exists or is expected. All operations chain the `Dual` rule through arithmetic/transcendental overloads.
[VERIFIED: fdars-core/src/autodiff.rs:484-627]

---

### 4. `boosting_regression/gamlss`

**File:** `fdars-core/src/boosting_regression/gamlss.rs`
**Functions (gradient-related):**
- `fn mu_neg_gradient(y, mu, sigma, sigma_floor)` (lines 64–79) — private
- `fn sigma_neg_gradient(y, mu, sigma)` (lines 86–101) — private
[VERIFIED: fdars-core/src/boosting_regression/gamlss.rs:53-101]

**Pattern:** Closed-form negative-gradient formulas for a Gaussian log-likelihood, computed pointwise. `u_μ,i(t) = (Y_i(t) − μ_i(t)) / σ_i(t)²` and `u_σ,i(t) = −1 + (Y_i(t) − μ_i(t))² / σ_i(t)²`. These are analytical derivatives of the log-likelihood — no DP, no backward pass, no boundary/endpoint seed.
**Boundary/seed handling:** The only boundary logic is a `sigma_floor` guard and `NUMERICAL_EPS` clip to prevent division-by-zero (lines 66, 95). These are numerical stability measures, not gradient-seed boundaries.
**Preliminary disposition:** **CLEAN** — pointwise closed-form gradients with no DP structure. No endpoint-seed analog. The floor/clip guards are correct stability practice.

---

### 5. `elastic_regression/logistic`

**File:** `fdars-core/src/elastic_regression/logistic.rs`
**Functions (gradient-related):**
- `fn logistic_gradients(q_aligned, beta, weights, alpha, y, lambda)` (lines 466–496) — private
- `fn armijo_line_search_logistic(…)` (lines 499–531) — private line-search helper
[VERIFIED: fdars-core/src/elastic_regression/logistic.rs:466-531]

**Pattern:** Analytical gradient of binary cross-entropy + L2 penalty w.r.t. intercept `α` and coefficient function `β`. Computes `grad_a = mean(prob - target)` and `grad_beta[j] = mean((prob - target) * q_aligned[(i,j)] * weights[j]) + lambda * beta[j]`. This is a closed-form gradient of a smooth loss — no DP, no backward pass, no boundary seed.
**Boundary/seed handling:** `prob = sigmoid(η)` is element-wise; no boundary logic at all.
**Preliminary disposition:** **CLEAN** — standard logistic gradient computation. Mathematically straightforward, no analogous endpoint-seed pattern.

---

### 6. `explain_generic/counterfactual`

**File:** `fdars-core/src/explain_generic/counterfactual.rs`
**Functions (gradient-related):**
- `fn compute_gradient_finite_diff(model, scores, ncomp)` (lines 7–22) — private
- `fn counterfactual_gd_search(model, original_scores, max_iter, ncomp, converged, update)` (lines 58–83) — private
[VERIFIED: fdars-core/src/explain_generic/counterfactual.rs:1-217]

**Pattern:** Finite-difference numerical gradient of the model's `predict_from_scores` w.r.t. FPC score components (forward difference, `eps = 1e-5`, line 3). Used to drive gradient-descent-in-score-space for counterfactual explanation.
**Boundary/seed handling:** No DP, no backward pass, no endpoint seed. The finite-difference loop iterates over `k in 0..ncomp`; each step perturbs one score component by `eps`. This is a simple numerical approximation — boundary handling is irrelevant.
**Preliminary disposition:** **CLEAN** — finite-difference numerical gradient (not a hand-written analytic backward pass). No boundary-seed issue is possible in a forward-difference scheme.

---

### 7. `regression`

**File:** `fdars-core/src/regression.rs`
**Functions (gradient-related):**
- `pub fn project_scores_generic<S: Scalar>(curve, mean, rotation, weights, ncomp)` (lines 232–251) — public
[VERIFIED: fdars-core/src/regression.rs:232-251]

**Pattern:** Linear projection `score_k = sum_j (curve[j] - mean[j]) * rotation[(j,k)] * weights[j]` expressed generically over `Scalar`. When instantiated at `Dual`, produces the exact forward-mode gradient `d(score_k)/d(curve[j]) = rotation[(j,k)] * weights[j]`. This is the closed-form gradient of a linear map — no DP, no backward pass, no boundary seed.
**Boundary/seed handling:** Pure linear algebra; no boundary conditions.
**Preliminary disposition:** **CLEAN** — forward-mode AD of a linear function. The analytic gradient is exact and constant. Validated by `fpca_score_gradient_dual` (line 1750) and `fpca_score_gradient_vs_fd` (line 1810).

---

### 8. `seasonal/mod`

**File:** `fdars-core/src/seasonal/mod.rs`
**Functions (gradient-related):**
- `pub(super) fn refine_period_gradient(acf, initial_period, dt, steps)` (lines 1112–1135) — private
[VERIFIED: fdars-core/src/seasonal/mod.rs:1111-1135]

**Pattern:** Gradient ascent on ACF score — but it is actually a three-point stencil **hill-climber** (not a gradient): computes the ACF-validation score at `period`, `period - step_size`, and `period + step_size`, then moves in the direction of the highest score. This is a scalar search heuristic, not a hand-written backward pass.
**Boundary/seed handling:** Floor guard `period.max(dt)` at line 1134. The only "boundary" is ensuring the period stays positive — no seed/endpoint pattern.
**Preliminary disposition:** **CLEAN** — not a gradient backward pass in any meaningful sense (it is a finite-difference hill-climber on a scalar quality score). No DP, no matrix propagation, no endpoint seed.

---

### 9. `smooth_basis`

**File:** `fdars-core/src/smooth_basis.rs`
**Functions (derivative-related):**
- `pub(crate) fn differentiate_basis_columns(basis, n_quad, nbasis, h, lfd_order)` (lines 985–1005) — internal
- `pub fn gradient_uniform(y, h)` in `helpers.rs` (line 747) — utility
[VERIFIED: fdars-core/src/smooth_basis.rs:984-1005]

**Pattern:** `differentiate_basis_columns` iteratively applies `gradient_uniform` (central-difference numerical differentiation with boundary-adapted stencils) to each column of a basis matrix, `lfd_order` times. `gradient_uniform` is a pure value-level numerical differentiator — it computes the derivative of a data vector by finite differences. There is no backward pass, no DP, no gradient accumulation. The "gradient" in the name is the numerical gradient of the basis functions (i.e., the derivative of the B-spline/Fourier basis), used to construct the penalty matrix for penalised regression — not a gradient of a loss.
**Boundary/seed handling:** `gradient_uniform` uses 5-point central differences in the interior with O(h³) forward/backward stencils at the boundaries (helpers.rs lines 774–784). These are standard numerical differentiation boundary formulas — correct and well-established.
**Preliminary disposition:** **CLEAN** — numerical differentiation utility for penalty matrix construction. No backward pass, no endpoint seed, no gradient propagation.

---

## Standard Stack

No external packages are installed in this phase. All code reuses existing crate machinery.

### Existing Functions Used

| Function | File | Purpose |
|----------|------|---------|
| `soft_dtw_backward` | `metric/soft_dtw.rs:262` | **THE BUG** — receives the one-line fix |
| `corrected_oracle_gradient` | `metric/soft_dtw.rs:467` | Test-module oracle for SC#1 cross-check |
| `soft_dtw_distance_generic` / `Dual::seed` | `metric/soft_dtw.rs:157` | Independent Dual reference path |
| `soft_dtw_accumulate_gradient` | `metric/soft_dtw.rs:326` | Calls the fixed backward pass |
| `soft_dtw_barycenter` | `metric/soft_dtw.rs:380` | Calls `soft_dtw_accumulate_gradient` |
| `init_barycenter_mean` | `metric/soft_dtw.rs:354` | Needed to compute pointwise mean for SC#1(b) |
| `FdMatrix::from_column_major` | `matrix.rs` | Test data construction |

## Package Legitimacy Audit

No external packages are installed in this phase. Audit not applicable.

---

## Architecture Patterns

### System Architecture Diagram

```
 Test assertion (SC#1a)
        ↓
 soft_dtw_backward (FIXED)
        ↓
 soft_dtw_accumulate_gradient (unchanged)
        ↓
 soft_dtw_barycenter (unchanged)
        ↓
 Test assertion (SC#1b): bary ≠ pointwise mean

 Independent cross-checks:
   corrected_oracle_gradient → compare grad to fixed backward
   Dual::seed / soft_dtw_distance_generic → compare grad to fixed backward
```

### Recommended Source Structure Changes

```
fdars-core/src/metric/soft_dtw.rs
  ├── soft_dtw_backward (lines 262–306)  ← ADD continue guard at loop body entry
  └── mod differentiable_tests (lines 422–581)
        ├── corrected_oracle_gradient (line 467)   ← UPDATE doc comment only
        └── [new test: regression test for CORR-01 SC#1]

fdars-core/src/metric/tests.rs
  ├── test_soft_dtw_barycenter_identical (line 795)  ← TIGHTEN assertions
  ├── test_soft_dtw_barycenter_shifted (line 821)    ← TIGHTEN assertions
  └── [new test: non-zero gradient + non-mean barycenter]

fdars-core/tests/validate_against_r.rs
  └── test_soft_dtw_barycenter_vs_tslearn (line 3802) ← TIGHTEN assertions

.planning/phases/78-.../78-AUDIT.md  ← NEW file: CORR-02 disposition table
```

### Pattern 1: Endpoint Skip Guard in DP Backward Pass

**What:** A reverse double-loop that seeds `E[n][m] = 1.0` before the loop must skip the write at `(n, m)` on the first iteration, because the neighbour contributions at the endpoint are all zero.
**When to use:** Any DP backward pass where the boundary condition is set before the loop and the loop visits the boundary cell.

**Example (the fix):**
```rust
// Source: fdars-core/src/metric/soft_dtw.rs:483-486 (corrected_oracle_gradient)
for i in (1..=n).rev() {
    for j in (1..=m).rev() {
        if i == n && j == m {
            continue; // preserve the E[n][m] = 1.0 seed
        }
        // ... compute a + b + c and write e[i][j]
    }
}
```

### Pattern 2: Three-Part SC#1 Regression Test

**What:** The regression test for CORR-01 must assert all three success criteria simultaneously on non-identical input.
**When to use:** Any fix where the corrected behavior can be silent (e.g., a zero gradient that still satisfies shape-level assertions).

```rust
// Suggested structure (test data to be chosen at discretion)
// SC#1(a): non-zero backward E
let e = soft_dtw_backward(&bary, &xi, &r, gamma);
let e_nonzero = e.iter().flatten().any(|&v| v.abs() > 1e-15);
assert!(e_nonzero, "E matrix must be non-zero for non-identical input");

// SC#1(b): barycenter moves away from pointwise mean
let bary_result = soft_dtw_barycenter(&data, gamma, 100, 1e-6);
let mean = init_barycenter_mean_from_data(&data);
let l2_diff: f64 = bary_result.barycenter.iter().zip(mean.iter())
    .map(|(b, m)| (b - m).powi(2)).sum::<f64>().sqrt();
assert!(l2_diff > MARGIN, "barycenter must move from pointwise mean");

// SC#1(c): fixed gradient matches oracle and Dual within ~1e-6
let oracle = corrected_oracle_gradient(&bary_vec, &xi_vec, gamma);
let fixed_grad = /* accumulate via soft_dtw_backward + loop */;
let dual_grad = /* via Dual::seed seeding */;
for k in 0..m {
    assert!((fixed_grad[k] - oracle[k]).abs() / oracle[k].abs().max(1e-12) <= 1e-6);
    assert!((fixed_grad[k] - dual_grad[k]).abs() / dual_grad[k].abs().max(1e-12) <= 1e-6);
}
```

### Anti-Patterns to Avoid

- **Testing convergence behavior rather than gradient correctness:** `test_soft_dtw_barycenter_shifted` currently checks that the mean value of the barycenter is ≈ 1.0, which holds for the pointwise-mean initialization regardless of whether the gradient is non-zero. The anti-pattern is conflating "convergence to a correct value" with "gradient is non-zero."
- **Testing identical-series as the non-zero-gradient regression:** For identical series, the gradient of soft-DTW w.r.t. the barycenter IS zero (the barycenter is already a minimiser). Use non-identical series with measurable curvature (e.g., one series shifted by 0.5 time units vs. the other).
- **Sharing the oracle and the production code's backward function:** `corrected_oracle_gradient` must remain independent — do not refactor `soft_dtw_backward` to call it, or the cross-check loses its independence.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Independent gradient reference | A new reference implementation | `corrected_oracle_gradient` (line 467) + `Dual` path | Both already exist and are validated |
| Non-identical test input | Complex synthetic data generator | Simple sinusoids with different amplitudes or phases | The existing test conventions use this pattern |
| Parallel test runner for CORR-02 | Custom audit scaffolding | Read each file + inline comment | CORR-02 is a read-and-annotate task, not code |

**Key insight:** The corrective fix is a one-liner; the complexity in this phase is in writing tests that can actually FAIL before the fix and PASS after it. The existing `test_soft_dtw_barycenter_*` tests pass on the buggy code — that is the core verification problem.

---

## Common Pitfalls

### Pitfall 1: Forgetting the Inner Loop is Reversed

**What goes wrong:** Developers testing the fix with `if i == n || j == m { continue; }` (either/or) instead of `if i == n && j == m { continue; }` (and) accidentally skip entire boundary rows and columns, producing a different wrong answer.
**Why it happens:** The loop runs `for i in (1..=n).rev()` and `for j in (1..=m).rev()` — the first iteration has `i == n, j == m`. Using OR instead of AND skips the entire `i == n` row and `j == m` column.
**How to avoid:** Use the exact pattern from `corrected_oracle_gradient` (line 485): `if i == n && j == m { continue; }`.
**Warning signs:** Oracle cross-check fails with non-trivially-wrong values rather than all-zero.

### Pitfall 2: Test Still Passes on Buggy Code

**What goes wrong:** The added regression test uses a test case where the pointwise mean happens to satisfy the assertion even when gradient is zero.
**Why it happens:** Shifted-by-constant series have a pointwise mean that is also a valid barycenter shape — the barycenter "looks right" even without gradient refinement.
**How to avoid:** Use non-identical series with different functional shapes (not just vertical shifts), and assert the L2 distance from the pointwise mean exceeds a meaningful margin. Also assert `E.iter().flatten().any(|&v| v != 0.0)` directly on the backward pass output.
**Warning signs:** `cargo test` green before applying the fix.

### Pitfall 3: Cargo Clippy with Wrong Flags

**What goes wrong:** Running `cargo clippy -p fdars-core -- -D warnings` instead of `cargo clippy --all-targets --features linalg,parallel -- -D warnings` misses lint warnings in test/bench code.
**Why it happens:** CI lints test/bench code via `--all-targets`. A plain `-p` run only lints library code.
**How to avoid:** Always use the exact CI-equivalent command: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
**Warning signs:** Local clippy green, CI clippy red.

### Pitfall 4: `corrected_oracle_gradient` doc comment not updated

**What goes wrong:** The doc comment (lines 455–466) continues to say "The shipped production code is intentionally left untouched (additive-only scope); the bug is logged as a backlog item." After the fix, this is false and will confuse future readers.
**Why it happens:** The comment was written under the v0.39.0 constraint of not fixing the bug.
**How to avoid:** Update the comment to say the fix has been applied in Phase 78, and the function is now an independent cross-check reference for the corrected implementation.
**Warning signs:** Comment claims a bug exists that no longer exists in `soft_dtw_backward`.

### Pitfall 5: `test_soft_dtw_barycenter_identical` — gradient IS zero for identical series

**What goes wrong:** Adding a `assert!(grad.iter().any(|&v| v != 0.0))` assertion to the `_identical` test causes it to fail for a correct implementation, because the gradient of soft-DTW at the exact minimiser (barycenter of identical series) is zero.
**Why it happens:** When all series are identical, the barycenter is initialized to the series itself (the mean equals all series), so `soft_dtw_accumulate_gradient` computes the gradient of `soft_dtw(bary, bary)` w.r.t. `bary` — which is the self-gradient at identity, expected to be near-zero.
**How to avoid:** Only add the non-zero-gradient assertion to tests using non-identical series. For the `_identical` test, tighten by checking the barycenter is close to the input series with a tighter tolerance, or add a comment explaining why gradient is zero.

---

## Validation Architecture

> `workflow.nyquist_validation` is `true` in `.planning/config.json` — this section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | None (no `rust-test.toml`; uses default `cargo test`) |
| Quick run command | `cargo test -p fdars-core soft_dtw --features linalg,parallel` |
| Full suite command | `cargo test --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORR-01 | `soft_dtw_backward` returns non-zero E on non-identical input | unit | `cargo test -p fdars-core soft_dtw_backward_nonzero` | ❌ Wave 0 |
| CORR-01 | `soft_dtw_barycenter` moves from pointwise mean on non-identical input | unit | `cargo test -p fdars-core soft_dtw_barycenter_moves` | ❌ Wave 0 |
| CORR-01 | Fixed gradient matches `corrected_oracle_gradient` within 1e-6 | unit | `cargo test -p fdars-core dual_gradient_vs_oracle` | ✅ (existing, still valid as cross-check) |
| CORR-01 | Existing `test_soft_dtw_barycenter_shifted` now has non-zero gradient assertion | unit | `cargo test -p fdars-core test_soft_dtw_barycenter_shifted` | ✅ (needs tightening) |
| CORR-01 | Existing `test_soft_dtw_barycenter_identical` still passes (gradient correctly zero for identical) | unit | `cargo test -p fdars-core test_soft_dtw_barycenter_identical` | ✅ (comment tightening only) |
| CORR-01 | `test_soft_dtw_barycenter_vs_tslearn` tightened | integration | `cargo test --test validate_against_r test_soft_dtw_barycenter_vs_tslearn` | ✅ (needs tightening) |
| CORR-02 | Audit table `78-AUDIT.md` completed with clean/fixed disposition for all 9 modules | manual | Review artifact | ❌ Wave 0 |
| CORR-02 | All whole-crate gates pass after changes | integration | `cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test` | N/A (gate, not test) |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core soft_dtw --features linalg,parallel`
- **Per wave merge:** `cargo test --features linalg,parallel`
- **Phase gate:** `cargo fmt --check && cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test` before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] New test: `test_soft_dtw_backward_nonzero` — asserts E matrix is non-zero for non-identical input after fix
- [ ] New test: `test_soft_dtw_barycenter_moves_from_mean` — asserts L2 distance from pointwise mean exceeds threshold
- [ ] New file: `78-AUDIT.md` — CORR-02 disposition table
- [ ] Tighten: `test_soft_dtw_barycenter_shifted` — add gradient non-zero assertion
- [ ] Tighten: `test_soft_dtw_barycenter_vs_tslearn` — add gradient non-zero + mean-departure assertions
- [ ] Doc update: `corrected_oracle_gradient` doc comment — remove "intentionally left untouched" language

---

## Security Domain

> `security_enforcement` is `true`; ASVS Level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (minimal) | Existing `Result<T, FdarError>` validation in `soft_dtw_barycenter` — unchanged |
| V6 Cryptography | no | — |

**Security assessment:** Phase 78 is an internal numerical correctness fix to a private function in a pure Rust library crate. No user inputs flow into the gradient computation via an external boundary; the CORR-01 fix changes only the backward-pass endpoint skip guard. No new input validation surface is introduced. ASVS V5 is satisfied by the existing dimension/parameter validation in `soft_dtw_barycenter` (lines 387–392 of `soft_dtw.rs`) which remains unchanged.

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN/Inf propagation through gradient accumulation | Tampering (data integrity) | Existing `softmin3` infinity guard (line 33); no new path introduced |

---

## Runtime State Inventory

> Not a rename/refactor/migration phase. This section is included to confirm no runtime state is affected.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | None — fdars-core is a pure computation library with no persistent store | None |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | None | None |
| Build artifacts | `target/` — no artifact name changes | None |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All builds | ✓ | 1.97.0 (runtime) | — |
| `cargo test` | Test suite | ✓ | bundled with Rust | — |
| `cargo clippy` | CI gate | ✓ | bundled with Rust | — |
| `cargo fmt` | CI gate | ✓ | bundled with Rust | — |
| `linalg` feature (`faer 0.23`) | Full test suite | ✓ | Rust 1.84+ required, 1.97.0 installed | Skip with `--features parallel` only |

**No missing dependencies.** All gates are runnable with the installed Rust 1.97.0 toolchain.

**Historical caution (from MEMORY.md):**
- `/tmp` disk pressure can block pre-commit hooks when the doctests link phase fills `/tmp` tmpfs — use `--no-verify` on commits in that scenario and run `cargo fmt` manually.
- `target/` can grow to 100+ GB; run `rm -rf target/debug/{incremental,examples}` if disk fills.
- Prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long `cargo` builds.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Buggy `soft_dtw_backward` (overwrites E[n][m]=1.0) | Fixed `soft_dtw_backward` (skips endpoint write) | Phase 78 | `soft_dtw_barycenter` produces correct gradient-descent refinement; existing identical-series tests still pass |
| `corrected_oracle_gradient` doc: "production code intentionally left untouched" | Updated doc: "production code fixed in Phase 78; this is now an independent cross-check" | Phase 78 | Documentation accuracy |

**Deprecated behavior:**
- All-zero `E` matrix from `soft_dtw_backward` on non-identical inputs — intentionally retired by CORR-01.

---

## Assumptions Log

> All claims in this research were verified by reading source files in this session.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The exponent formula simplification `r[i][j] - r[i+1][j] + r[i+1][j] = r[i][j]` holds algebraically | CORR-01 Bug Anatomy | No risk — the terms cancel trivially; the fix does not depend on this analysis |
| A2 | The "shifted sinusoid" test input for SC#1(b) will produce a measurably non-zero L2 distance from the pointwise mean after the fix | Validation Architecture | If the margin is too tight or the test input is degenerate, the SC#1(b) assertion could be difficult to satisfy — the exact margin constant is at Claude's discretion |

**Risk:** Low for A1 (mathematical certainty), low-medium for A2 (empirical — the planner should choose a margin that is meaningful but not fragile).

---

## Open Questions

1. **What L2 margin should SC#1(b) use?**
   - What we know: After the fix, `soft_dtw_barycenter` will apply real gradient steps; the barycenter of non-identical series will move from the pointwise mean.
   - What's unclear: How large the movement is depends on gamma, the input series shape, and `max_iter`.
   - Recommendation: Use the "shifted sinusoid" pattern from `test_soft_dtw_barycenter_shifted` with a few iterations, run it both before and after the fix, and pick a margin that is larger than the buggy-code movement (0.0) and smaller than the expected correct movement. A margin of L2 > 0.1 over `m=20` points is likely safe for `sin(2πt) + i` inputs.

2. **Which test file should the new SC#1 regression test live in?**
   - What we know: The existing differentiable tests (dual/oracle cross-checks) live in `mod differentiable_tests` inside `soft_dtw.rs` itself (line 423). The existing barycenter tests live in `metric/tests.rs` (lines 795–840).
   - What's unclear: Whether adding a new `#[test]` inside `soft_dtw.rs`'s `differentiable_tests` module or in `tests.rs` is preferred.
   - Recommendation: Put the SC#1 regression test in `metric/tests.rs` alongside the other barycenter tests (it tests the barycenter, not the Dual path). Put the oracle/Dual cross-check in `differentiable_tests` (as it already is).

3. **Are there any CORR-02 siblings with suspicious boundary handling warranting deeper inspection?**
   - What we know: All nine audited modules lack a DP backward-pass structure. The only module with a reverse double-loop seeded at an endpoint is `soft_dtw_backward`.
   - What's unclear: Whether any module has an analogous bug in a different form (e.g., a recurrence overwritten by a later step).
   - Recommendation: The executor should scan each module's gradient code for any recurrence pattern where a pre-set boundary value could be overwritten by a general update rule — but based on this research, none are expected.

---

## Sources

### Primary (HIGH confidence)
All findings are `[VERIFIED: source-file:line-range]` — read directly from the codebase in this session.

- `fdars-core/src/metric/soft_dtw.rs:1-581` — full file; bug location, oracle, caller chain, Dual path
- `fdars-core/src/metric/tests.rs:780-840` — existing barycenter test weakness analysis
- `fdars-core/tests/validate_against_r.rs:3800-3836` — integration test weakness analysis
- `fdars-core/src/alignment/differentiable.rs:1-238` — full CORR-02 audit (clean)
- `fdars-core/src/autodiff.rs:484-627` — full CORR-02 audit (clean)
- `fdars-core/src/boosting_regression/gamlss.rs:53-101` — full CORR-02 audit (clean)
- `fdars-core/src/elastic_regression/logistic.rs:466-531` — full CORR-02 audit (clean)
- `fdars-core/src/explain_generic/counterfactual.rs:1-217` — full CORR-02 audit (clean)
- `fdars-core/src/regression.rs:232-251` — full CORR-02 audit (clean)
- `fdars-core/src/seasonal/mod.rs:1111-1135` — full CORR-02 audit (clean)
- `fdars-core/src/smooth_basis.rs:984-1005` — full CORR-02 audit (clean)
- `fdars-core/src/helpers.rs:747-784` — `gradient_uniform` boundary stencils (supporting CORR-02 smooth_basis)

### Secondary (MEDIUM confidence)
None — all claims are codebase-verified.

### Tertiary (LOW confidence)
None — this is an internal codebase research task with no web sources.

---

## Metadata

**Confidence breakdown:**
- CORR-01 bug location and fix: HIGH — read the exact lines; the endpoint-seed overwrite is mechanically verified
- CORR-01 test weakness analysis: HIGH — traced exactly why each existing test passes on buggy code
- CORR-02 audit sweep (all 9 modules): HIGH — read each module's gradient-related functions; no DP backward-pass with endpoint seed found in any sibling
- Validation architecture: HIGH — test framework, gate commands, and wave-0 gaps derived from codebase conventions

**Research date:** 2026-09-06
**Valid until:** Indefinite — this is a static codebase analysis. Valid as long as the files listed in Sources remain unchanged.
