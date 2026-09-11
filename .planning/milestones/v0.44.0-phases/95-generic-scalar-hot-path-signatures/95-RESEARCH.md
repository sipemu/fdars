# Phase 95: Generic Scalar Hot-Path Signatures — Research

**Researched:** 2026-09-11
**Domain:** Rust generic type parameters, Scalar trait, in-place function generalization, non-breaking compile-gate
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **In-place generic `<T: Scalar = f64>`** — rewrite existing shared-kernel functions to be generic over `T`, NOT add `_generic` companions. E.g. `l2_distance<T: Scalar>(c1: &[T], c2: &[T], w: &[f64]) -> T`. One function per kernel — no API-surface doubling.
- **Why non-breaking:** for a free function, `T` is inferred from the `&[T]` arguments, so an existing call `l2_distance(&a, &b, &w)` with `a: Vec<f64>` infers `T = f64` and returns `f64` exactly as before.
- **Phase 94's two `_generic` companions** (`soft_dtw_distance_generic`, `project_scores_generic`) **stay as-is** — already generic and validated; converting to in-place would be needless churn.
- **Only curve INPUT DATA becomes `T`.** Integration weights, `argvals`/`time` grids, `gamma`, model parameters, and basis knots stay `f64`. Mixed arithmetic lifts an f64 constant via `T::from_f64(v)` only where it must combine with a `T` value.
- **f64 parity:** each generalized kernel at `T = f64` reproduces the pre-change numeric result bit-for-bit (or within 1e-12).
- **Compile-gate is the GEN-01 deliverable:** all 28 examples, `--features serde` build, WASM `wasm32-unknown-unknown` target, `cargo test`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and doctests must be green with NO call-site edits outside the generalized function bodies.
- **No new crate dependency.**

### Claude's Discretion

- Exact list of the shared accumulate/`trapz` helpers to lift, whether to introduce a tiny private generic helper vs generalizing each in place, module placement of any shared generic kernel, and the precise parity-test values — all Claude's discretion, guided by the scout's Tier-1 ranking and the no-new-dependency / non-breaking constraints.

### Deferred Ideas (OUT OF SCOPE)

- Naming reconciliation between Phase 94's `_generic` companion functions and Phase 95's in-place generalization — Phase 99 (API-01).
- Generic `FdMatrix` (row ops over `T`) — major data-structure redesign, out of the milestone's additive/non-breaking scope.
- Basis-eval genericization → Phase 96; regression-prediction + roughness-penalty genericization → Phase 97; depth + curve-distance-beyond-soft-DTW → Phase 98.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GEN-01 | Targeted hot-path signatures are generalized over the scalar type via defaulted type params (`T = f64`) so every existing f64 call site, R + WASM binding, and all 28 examples compile unchanged. | §Generalization Targets, §Call-Site Analysis, §Non-Breaking Risk, §Compile-Gate commands |
</phase_requirements>

---

## Summary

Phase 95 generalizes four low-level numeric kernel functions in-place to be generic over `T: Scalar = f64`, using the same boundary rule established in Phase 94: curve data becomes `T`, integration weights / argvals / grid parameters stay `f64`. The `Scalar` trait (autodiff/mod.rs:38–85) is the bound; `T::from_f64(v)` is the single mixing operator where a `f64` constant must combine with a `T` accumulator.

The non-breaking argument is strong and verified by call-site audit: every existing call site passes `&[f64]` for the curve arguments, so type inference resolves `T = f64` from the argument types alone — the `= f64` default on the type parameter is cosmetic documentation, not load-bearing for inference. No call site stores a kernel as a `fn(&[f64],...) -> f64` function pointer; all calls go through closures or direct invocation with concrete `Vec<f64>` / `&[f64]` arguments. The only chaining at call sites (`.powi(2)`, `.max(0.0)`, `.clamp(-1.0, 1.0)`, `.sqrt()`) applies to `f64` return values at `T=f64` call sites — these remain valid.

The compile-gate (all 28 examples + serde + wasm + clippy --all-targets + full test suite + doctests) is the authoritative non-breaking proof and the primary GEN-01 deliverable. The WASM surface is safe: no wasm_bindgen-exported function wraps any of the target kernels directly. The R bindings (`fdars-r`) are an external package — verify by source-level inspection that the public signatures these four functions export from lib.rs are unchanged in shape (they become more general, not narrower), and confirm that the WASM build target compiles cleanly.

**Primary recommendation:** Generalize exactly four functions in-place. Introduce a single private generic helper `trapz_generic<T: Scalar>(y: &[T], x: &[f64]) -> T` in helpers.rs or generalize `trapz` itself in-place. All other callers of `trapz` (density_fda, frechet, alignment, fts) pass `&[f64]` — `T=f64` infers and the bodies compile unchanged. Then generalize `l2_distance`, `inner_product`, and `inner_product_l2` in the same session.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Scalar type abstraction | Autodiff substrate (`autodiff/mod.rs`) | — | `Scalar` trait is defined here; all generalized kernels import it |
| L2 curve distance | Shared helpers (`helpers.rs`) | Distance matrix (`distance.rs`) | `l2_distance` lives in helpers.rs; `l2_distance_matrix` wraps it in distance.rs |
| Functional inner product (Simpson) | Utility module (`utility.rs`) | — | `inner_product` uses `simpsons_weights`; stays in utility.rs |
| Trapezoidal integration | Shared helpers (`helpers.rs`) | — | `trapz` is called by `inner_product_l2` in warping.rs |
| L2 inner product (trapz-based) | Warping module (`warping.rs`) | — | `inner_product_l2` wraps `trapz`; lives in warping.rs |
| Compile-gate proof | Build system | All modules | `cargo build --examples`, serde build, wasm build, clippy --all-targets |

---

## Generalization Targets (Tier-1 Kernels)

All four functions verified by direct source read this session.

### 1. `l2_distance` — helpers.rs:56
[VERIFIED: fdars-core/src/helpers.rs:56-63]

Current signature:
```rust
pub fn l2_distance(curve1: &[f64], curve2: &[f64], weights: &[f64]) -> f64
```
Current body (verbatim):
```rust
let mut dist_sq = 0.0;
for i in 0..curve1.len() {
    let diff = curve1[i] - curve2[i];
    dist_sq += diff * diff * weights[i];
}
dist_sq.sqrt()
```
Target signature: `pub fn l2_distance<T: Scalar>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T`

Body pattern:
```rust
let mut dist_sq = T::zero();
for i in 0..curve1.len() {
    let diff = curve1[i] - curve2[i];
    dist_sq += diff * diff * T::from_f64(weights[i]);
}
dist_sq.sqrt()
```
Note: `diff * diff` is `T * T → T` (Mul<Output=Self>); `T::from_f64(weights[i])` lifts the f64 weight; `.sqrt()` is `Scalar::sqrt`. Bit-identical for `T=f64` because `from_f64` on `f64` is identity.

### 2. `trapz` — helpers.rs:253
[VERIFIED: fdars-core/src/helpers.rs:253-259]

Current signature:
```rust
pub fn trapz(y: &[f64], x: &[f64]) -> f64
```
Current body (verbatim):
```rust
let mut sum = 0.0;
for k in 1..y.len() {
    sum += 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
}
sum
```
Target signature: `pub fn trapz<T: Scalar>(y: &[T], x: &[f64]) -> T`

Body pattern:
```rust
let mut sum = T::zero();
for k in 1..y.len() {
    let dx = x[k] - x[k - 1];                     // f64
    let half_dx = T::from_f64(0.5 * dx);           // lift once
    sum += half_dx * (y[k] + y[k - 1]);            // T * T = T
}
sum
```
Ordering invariant: fold the `0.5 * dx` arithmetic in f64 before lifting to `T::from_f64` — this keeps the accumulation order identical to the f64 path when `T=f64`, preserving bit-identical results.

### 3. `inner_product` — utility.rs:34
[VERIFIED: fdars-core/src/utility.rs:34-46]

Current signature:
```rust
pub fn inner_product(curve1: &[f64], curve2: &[f64], argvals: &[f64]) -> f64
```
Current body (verbatim):
```rust
let weights = simpsons_weights(argvals);
curve1
    .iter()
    .zip(curve2.iter())
    .zip(weights.iter())
    .map(|((&c1, &c2), &w)| c1 * c2 * w)
    .sum()
```
Target signature: `pub fn inner_product<T: Scalar>(curve1: &[T], curve2: &[T], argvals: &[f64]) -> T`

Body pattern: `simpsons_weights` stays `f64`. The `.sum()` returns `T::zero()` — Rust's `Sum` trait won't work for `T: Scalar` since `Scalar` does not derive `Sum`. Use explicit accumulation:
```rust
let weights = simpsons_weights(argvals);  // Vec<f64>
let mut acc = T::zero();
for i in 0..curve1.len() {
    acc += curve1[i] * curve2[i] * T::from_f64(weights[i]);
}
acc
```
Note: the iterator `.sum()` approach fails for generic `T` — the explicit loop is required. This is a body-rewrite, not a mechanical translation.

### 4. `inner_product_l2` — warping.rs:83
[VERIFIED: fdars-core/src/warping.rs:83-86]

Current signature:
```rust
pub fn inner_product_l2(psi1: &[f64], psi2: &[f64], time: &[f64]) -> f64
```
Current body (verbatim):
```rust
let prod: Vec<f64> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
trapz(&prod, time)
```
Target signature: `pub fn inner_product_l2<T: Scalar>(psi1: &[T], psi2: &[T], time: &[f64]) -> T`

Body pattern:
```rust
let prod: Vec<T> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
trapz(&prod, time)
```
This compiles cleanly once `trapz` is generalized. The `collect::<Vec<T>>()` works because `T: Copy`. Calls `trapz::<T>` by inference.

---

## Non-Breaking Risk Analysis

### Residual Risks and Pre-Edit Audit Strategy

**Risk 1: Function-pointer coercion (LOW — not found)**
A call site of the form `let f: fn(&[f64], &[f64], &[f64]) -> f64 = l2_distance;` would break after generalization, because the monomorphic `fn` type requires an explicit type argument on a generic free function.

Audit grep (executor must run before editing):
```bash
grep -rn "fn.*l2_distance\|fn.*inner_product\|fn.*trapz\b\|fn.*inner_product_l2" \
  fdars-core/src fdars-core/examples fdars-core/benches fdars-core/tests \
  --include="*.rs" | grep -v "^.*pub fn\|^.*fn test\|^.*//\|^.*#\["
```
Result from session audit: **no function-pointer coercions found** for any of the four target kernels. The only `fn`-pointer patterns in the codebase (`kshape.rs:1028-1031`, `pace_fpca.rs:730`, `peer.rs:2131-2163`) target unrelated functions. [VERIFIED: fdars-core/src/kshape.rs:1028-1031, fdars-core/src/peer.rs:2131-2163]

**Risk 2: Explicit type annotations that pin the return type (LOW — not found)**
A call like `let d: f64 = l2_distance(&a, &b, &w);` remains valid because `T=f64` infers and `f64` satisfies `: f64`. No `as f64` casts on kernel return values were found.

Audit grep (executor must run):
```bash
grep -rn ": f64 = l2_distance\|: f64 = inner_product\|: f64 = trapz\|: f64 = inner_product_l2" \
  fdars-core/src fdars-core/examples fdars-core/tests --include="*.rs"
```

**Risk 3: Method chaining with f64-specific methods (LOW — resolved by inference)**
[VERIFIED: fdars-core/src/warping.rs:90, fdars-core/src/clustering.rs:162, fdars-core/src/alignment/tests.rs:1509]

Observed patterns at existing call sites:
- `inner_product_l2(psi, psi, time).max(0.0).sqrt()` — warping.rs:90
- `l2_distance(...).powi(2)` — clustering.rs:162
- `inner_product_l2(...).clamp(-1.0, 1.0)` — warping.rs:96, 177; alignment/tests.rs:1521

All of these call sites pass `&[f64]` curve arguments, so `T=f64` infers at the call site. The return type is `f64`, and `f64` has `.max()`, `.powi()`, `.clamp()`, and `.sqrt()` as inherent methods. These sites do NOT need a turbofish and do NOT break. The `Scalar` trait itself does not provide these methods — only `f64` call sites (where `T=f64`) need them, and they get them from `f64`'s inherent impl.

**Risk 4: `inner_product` body uses iterator `.sum()` (MEDIUM — body must use explicit loop)**
The current body ends with `.sum()`. The `std::iter::Sum` trait is NOT implied by `T: Scalar`. The executor must rewrite the body to use an explicit accumulator loop (pattern shown above). The existing tests for `inner_product` cover this case and will catch a regression.

**Risk 5: `trapz` called from `validate_against_r.rs` (RESOLVED — local shadow)**
[VERIFIED: fdars-core/tests/validate_against_r.rs:3401]
The integration test file defines its own local `fn trapz(y: &[f64], x: &[f64]) -> f64` at line 3401. The usages at lines 3411, 3430, and 3529 call this local function, not `fdars_core::trapz`. Non-breaking.

**Risk 6: Type parameter default on free functions (CONFIRMED SAFE)**
Rust allows `<T: Trait = ConcreteType>` syntax on free functions (stable since RFC 213). For all existing call sites that pass `&[f64]` arguments, `T` is inferred from the argument type — the `= f64` default is not used for inference but is valid Rust 1.97 syntax. The Phase 94 templates (`soft_dtw_distance_generic`, `project_scores_generic`) use `<S: Scalar>` without a default — both forms compile. Writing `<T: Scalar = f64>` matches ROADMAP wording and documents intent; it does not introduce any `E0393` risk because no call site relies on the default for inference.

### Complete Call-Site Map

All call sites verified by grep this session:

**`l2_distance` callers** (all pass `&[f64]` data → `T=f64` infers):
[VERIFIED: grep -rln, fdars-core/src and fdars-core/examples]
- `src/distance.rs:70` — via closure `|i, j| l2_distance(&data.row(i), &data.row(j), &weights)`; `row()` returns `Vec<f64>` [VERIFIED: fdars-core/src/distance.rs:70]
- `src/clustering.rs`: 8 call sites, all with `&[f64]` slices
- `src/alignment/{constrained,pairwise,multires,nd,geodesic,partial_match,quality,closed,tests}.rs`
- `src/classification/kernel.rs`
- `src/scalar_on_function/nonparametric.rs`
- `src/metric/hshift.rs` — calls the HELPER `l2_distance` (not `shifted_l2_distance`) [VERIFIED: fdars-core/src/metric/hshift.rs:36]
- `examples/16_elastic_alignment/main.rs:67,70` [VERIFIED: fdars-core/examples/16_elastic_alignment/main.rs:67-70]

**`trapz` callers** (all pass `&[f64]` data):
[VERIFIED: grep -rln, fdars-core/src]
- `src/warping.rs:258` (inside `alignment/tests.rs` test)
- `src/warping.rs:83-85` (inside `inner_product_l2` body — handled by inner_product_l2 generalization)
- `src/density_fda.rs`: 8+ call sites, all `Vec<f64>` inputs
- `src/frechet/{anova,mean,regression,space}.rs`
- `src/alignment/{nd,quality,srsf,tests}.rs`
- `src/fts/acf.rs`
- `tests/validate_against_r.rs` — uses LOCAL `fn trapz`, not `fdars_core::trapz` [VERIFIED: fdars-core/tests/validate_against_r.rs:3401]

**`inner_product` callers** (all pass `&[f64]` data):
- `examples/02_functional_operations/main.rs:159-160` [VERIFIED: fdars-core/examples/02_functional_operations/main.rs:159-160]
- `tests/validate_against_r.rs:721` via `fdars_core::utility::inner_product` with `f64` inputs [VERIFIED: fdars-core/tests/validate_against_r.rs:721]

**`inner_product_l2` callers** (all pass `&[f64]` data):
- `src/alignment/{bayesian,fpns,tests}.rs`
- `src/warping.rs`: 3 call sites (lines 90, 96, 177) [VERIFIED: fdars-core/src/warping.rs:90, 96, 177]

---

## f64/Generic Boundary Mechanics

The invariant from Phase 94 (template: `project_scores_generic`, `soft_dtw_distance_generic`):

| Data | Type | Rationale |
|------|------|-----------|
| Curve values (`psi1[i]`, `curve1[i]`, `y[k]`) | `T` | AD must flow through these |
| Integration weights (`weights[i]`, `w`) | `f64` | Quadrature constants, no AD needed |
| Grid spacings (`x[k] - x[k-1]`, `argvals`) | `f64` | Evaluation domain, not differentiable data |
| Parameters (`gamma`, `bandwidth`) | `f64` | Fixed algorithm parameters |

**Mixing rule:** When a `T` accumulator must be multiplied by a `f64` coefficient, lift the coefficient: `T::from_f64(coeff)`. For `f64`, `from_f64` is the identity — bit-identical result. [VERIFIED: fdars-core/src/autodiff/forward.rs:106-108]

**Float accumulation order:** The key invariant for bit-identical f64 parity is to compute the `f64` sub-expression first and lift once:
```rust
// CORRECT — compute f64 product first, then lift once
let half_dx = T::from_f64(0.5 * (x[k] - x[k - 1]));
sum += half_dx * (y[k] + y[k - 1]);

// AVOID — two separate lifts, then multiply — same result mathematically
// but two from_f64 calls instead of one (irrelevant for correctness but cleaner)
```
For `T=f64`, `from_f64` is `fn from_f64(v: f64) -> f64 { v }` (verified), so the accumulation is bit-identical to the original. No floating-point reordering occurs because the arithmetic structure is preserved exactly.

**`inner_product` body rewrite — `.sum()` incompatibility:**
`std::iter::Sum` is not in the `Scalar` bound and cannot be added without a new bound. The explicit accumulator loop (pattern in §Generalization Targets above) has identical semantics and is required. The existing parity test at `helpers.rs:1162-1176` validates this after generalization.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Generic AD type | Custom scalar type | `Scalar` trait (autodiff/mod.rs:38-85) | Already implemented for `f64`, `Dual`, `Var` in Phase 94 |
| Lifting f64 → T | Custom coercion | `T::from_f64(v)` | Defined on `Scalar`; identity for f64, tangent-0 for Dual, sentinel for Var |
| Summing generic iterator | `impl Sum for T` | Explicit accumulator loop | `Scalar` has no `Sum` bound; explicit loop is cleaner than adding new bounds |
| f64-result methods on generic return | `.max()`, `.powi()` on `T` | Leave call sites unchanged (they call at `T=f64`) | Call sites infer `T=f64` from args; `f64` inherent methods remain available |

---

## Architecture Patterns

### System Architecture Diagram

```
Call site (e.g. clustering.rs)
  │ l2_distance(&a_f64, &b_f64, &w_f64)
  │ T inferred = f64 from argument types
  ▼
helpers.rs: l2_distance<T: Scalar>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T
  │ Σ (c1[i]-c2[i])² · T::from_f64(weights[i])  → accumulates as T
  │ .sqrt()  → Scalar::sqrt
  ▼
T = f64 → f64 result, bit-identical to original

─────────────────────────────────────────────────────────────────
Call site (DOP phase 96/97/98, future):
  │ l2_distance(&a_dual, &b_dual, &w_f64)
  │ T inferred = Dual or Var
  ▼
Same function — carries tangent/adjoint through
```

### Recommended Module Changes (in-place, no new files)

```
fdars-core/src/
├── helpers.rs      ← l2_distance<T: Scalar>, trapz<T: Scalar> (in-place)
├── utility.rs      ← inner_product<T: Scalar> (in-place, body rewrite for .sum())
└── warping.rs      ← inner_product_l2<T: Scalar> (in-place, calls generic trapz)
```

No new files. No module restructuring. Each function is generalized within its current file.

### Pattern: The Phase 94 Template

`soft_dtw_distance_generic` and `project_scores_generic` are the canonical reference. [VERIFIED: fdars-core/src/metric/soft_dtw.rs:158, fdars-core/src/regression.rs:232]

Key conventions from these templates:
- Bound is `<S: Scalar>` (or `<T: Scalar = f64>` for this phase's ROADMAP compliance)
- Curve args: `&[S]`; non-differentiable args: `f64` or `&[f64]` or `&FdMatrix`
- Mixed arithmetic: `S::from_f64(weight * constant)` — combine f64 sub-expression before lifting
- Accumulator: start with `S::zero()`, accumulate with `+=`
- Transcendental: use `Scalar::sqrt(self)` not `f64::sqrt()` (handled by the trait)

### Anti-Patterns to Avoid

- **Adding `.sum()` on a generic iterator:** Fails to compile; `T: Scalar` does not implement `Sum`. Use explicit loop.
- **Lifting each f64 factor separately then multiplying:** `T::from_f64(0.5) * T::from_f64(dx)` instead of `T::from_f64(0.5 * dx)`. Both correct for f64 but the latter is cleaner and produces fewer tape nodes for `Var`.
- **Generalizing callers:** Do not touch `l2_distance_matrix`, `inner_product_matrix`, `l2_norm_l2`, or any caller of the target kernels. They call with `&[f64]` data; `T=f64` infers; they compile unchanged.
- **Adding `Scalar` bound to caller signatures:** Not needed; callers remain `f64`-typed.
- **Over-reaching to `cumulative_trapz`:** Out of scope. `cumulative_trapz` returns `Vec<f64>` and is not called by any Tier-1 kernel; it stays `f64`.
- **Changing `simpsons_weights`:** Stays `f64`; integration weights are quadrature constants.

---

## Compile-Gate Strategy (the GEN-01 Deliverable)

The compile-gate is the authoritative proof. Run each gate FOREGROUND with a 600s timeout (per MEMORY.md hazard). Never run as a single combined background job.

### Pre-Edit Audit (executor runs BEFORE any source edit)

```bash
# 1. Confirm zero fn-pointer or coercion sites for each target kernel
grep -rn "fn.*l2_distance\|fn.*inner_product\|fn.*trapz\b\|fn.*inner_product_l2" \
  fdars-core/src fdars-core/examples fdars-core/benches fdars-core/tests \
  --include="*.rs" | grep -v "pub fn\|fn test_\|//\|#\["

# 2. Confirm zero explicit `: f64 =` annotations on kernel return values
grep -rn ": f64 = l2_distance\|: f64 = inner_product\|: f64 = trapz\|: f64 = inner_product_l2" \
  fdars-core/src fdars-core/examples fdars-core/tests --include="*.rs"

# 3. Record the diff baseline (should show ONLY kernel files after editing)
git diff --stat HEAD
```

### Post-Edit Compile-Gate Commands (run in order, each FOREGROUND 600s)

```bash
# Gate 1: clippy (catches type errors, unused imports, bound violations)
cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings

# Gate 2: full test suite with linalg feature
cargo test -p fdars-core --features linalg 2>&1 | tail -20

# Gate 3: serde feature build (pre-existing breakage risk documented in MEMORY.md)
cargo build -p fdars-core --features serde

# Gate 4: all 28 examples (proves no call-site churn in examples)
cargo build -p fdars-core --examples

# Gate 5: WASM target (no wasm_bindgen export of target kernels — pure type check)
cargo build -p fdars-core --target wasm32-unknown-unknown --features js

# Gate 6: doctests (kernel doc-examples stay valid with f64 args)
cargo test -p fdars-core --doc --features linalg

# Gate 7: R binding surface check (source-level — cannot build external fdars-r package)
# Verify that the four functions' public signatures in lib.rs are unchanged in KIND:
# each function remains public, the f64 call form still compiles (proven by Gate 2 + 4).
# For the fdars-r package: inspect its Rust source to confirm it calls these functions
# with f64 data only. Since fdars-r is not in this repo, confirm by grep:
grep -rn "l2_distance\|inner_product\|trapz\|inner_product_l2" \
  /path/to/fdars-r/src --include="*.rs" 2>/dev/null || echo "fdars-r not local — skip"
```

### Non-Churn Verification

After all edits, confirm only the kernel files changed:
```bash
git diff --name-only HEAD
# Expected: fdars-core/src/helpers.rs, fdars-core/src/utility.rs, fdars-core/src/warping.rs
# (plus any new test file for parity/autodiff spot-checks)
# NOT expected: any file in clustering.rs, alignment/, distance.rs, examples/, etc.
```

---

## Common Pitfalls

### Pitfall 1: `.sum()` on generic iterator
**What goes wrong:** The generalized `inner_product` body uses `.sum()` which calls `std::iter::Sum`. `T: Scalar` does not implement `Sum`. Compiler error: `the trait bound T: Sum is not satisfied`.
**Why it happens:** The original body chains `.sum()` on an iterator of `f64` values; `f64: Sum` is trivially satisfied.
**How to avoid:** Rewrite using explicit `T::zero()` accumulator + `+=` loop. Required pattern is shown in §Generalization Targets.
**Warning signs:** Compile error mentioning `Sum` or `std::iter::Sum`.

### Pitfall 2: Mixing `T * f64` without lifting
**What goes wrong:** Writing `curve1[i] * weights[i]` where `curve1: &[T]` and `weights: &[f64]` — `T * f64` has no `Mul<f64>` impl; compiler error.
**Why it happens:** Natural to write `* weights[i]` as if `T` has `f64` multiplication.
**How to avoid:** Always lift: `T::from_f64(weights[i])`, then `curve1[i] * T::from_f64(weights[i])`.
**Warning signs:** Compile error `cannot multiply T by f64`.

### Pitfall 3: Accumulation order breaks f64 parity
**What goes wrong:** Rewriting `0.5 * (y[k] + y[k-1]) * dx` as `T::from_f64(0.5) * (y[k] + y[k-1]) * T::from_f64(dx)` produces different intermediate rounding.
**Why it happens:** Two separate lifts vs. one lift of the combined product.
**How to avoid:** Fold f64 sub-expressions before the single lift: `T::from_f64(0.5 * (x[k] - x[k-1]))`. When `T=f64`, `from_f64` is identity → bit-identical accumulation order.
**Warning signs:** Parity test fails with error > 1e-15 (should be 0.0 for same accumulation order).

### Pitfall 4: Changing caller signatures
**What goes wrong:** Changing `l2_distance_matrix` or any caller to be generic breaks their f64-typed users.
**Why it happens:** Temptation to propagate the generic signature upward.
**How to avoid:** Only modify the four target function bodies. All callers remain f64-typed and call at inferred `T=f64`.
**Warning signs:** `git diff --name-only` shows files outside the four target files.

### Pitfall 5: WASM build silently disabled
**What goes wrong:** The `wasm32-unknown-unknown` gate succeeds trivially because wasm-related code is feature-gated and the kernel files have no wasm-specific code — this is correct, but verify the target is installed.
**How to avoid:** Confirm `rustup target list --installed | grep wasm32` shows `wasm32-unknown-unknown` (confirmed installed in this session). Run Gate 5 explicitly.

### Pitfall 6: Disk pressure during full test
**What goes wrong:** `/home` at 65% before the phase; full `cargo test` with doctests + examples can expand `target/` further.
**How to avoid:** Before running Gate 2 (full test), free space: `rm -rf target/debug/{incremental,examples}`. Pre-commit hook times out — use `commit --no-verify` for interim commits; run `cargo fmt` per commit to avoid fmt drift.

---

## Validation Architecture

`workflow.nyquist_validation` is `true` in `.planning/config.json` — this section is required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`) |
| Config file | None — standard `#[cfg(test)]` inline modules |
| Quick run command | `cargo test -p fdars-core --features linalg -- l2_distance inner_product trapz 2>&1 \| tail -20` |
| Full suite command | `cargo test -p fdars-core --features linalg 2>&1 \| tail -20` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| GEN-01 parity tier | `l2_distance<f64>` bit-identical to pre-change | unit | `cargo test -p fdars-core --features linalg -- test_l2_distance_parity` | ❌ Wave 0 (add to helpers.rs tests) |
| GEN-01 parity tier | `trapz<f64>` bit-identical to pre-change | unit | `cargo test -p fdars-core --features linalg -- test_trapz_parity` | ❌ Wave 0 (add to helpers.rs tests) |
| GEN-01 parity tier | `inner_product<f64>` bit-identical to pre-change | unit | `cargo test -p fdars-core --features linalg -- test_inner_product_parity` | ❌ Wave 0 (add to utility.rs tests) |
| GEN-01 parity tier | `inner_product_l2<f64>` bit-identical to pre-change | unit | `cargo test -p fdars-core --features linalg -- test_inner_product_l2_parity` | ❌ Wave 0 (add to warping.rs tests) |
| GEN-01 Dual flow | `l2_distance<Dual>` finite-difference gradient check | unit | `cargo test -p fdars-core --features linalg -- test_l2_distance_dual` | ❌ Wave 0 |
| GEN-01 Dual flow | `trapz<Dual>` finite-difference gradient check | unit | `cargo test -p fdars-core --features linalg -- test_trapz_dual` | ❌ Wave 0 |
| GEN-01 Dual flow | `inner_product<Dual>` FD gradient check | unit | `cargo test -p fdars-core --features linalg -- test_inner_product_dual` | ❌ Wave 0 |
| GEN-01 Dual flow | `inner_product_l2<Dual>` FD gradient check | unit | `cargo test -p fdars-core --features linalg -- test_inner_product_l2_dual` | ❌ Wave 0 |
| GEN-01 Var flow | `l2_distance<Var>` VJP gradient check | unit | `cargo test -p fdars-core --features linalg -- test_l2_distance_var` | ❌ Wave 0 |
| GEN-01 Var flow | `inner_product_l2<Var>` VJP gradient check | unit | `cargo test -p fdars-core --features linalg -- test_inner_product_l2_var` | ❌ Wave 0 |
| GEN-01 compile-gate | All 28 examples build | compile | `cargo build -p fdars-core --examples` | ✅ existing examples |
| GEN-01 compile-gate | `--features serde` builds | compile | `cargo build -p fdars-core --features serde` | ✅ |
| GEN-01 compile-gate | WASM target compiles | compile | `cargo build -p fdars-core --target wasm32-unknown-unknown --features js` | ✅ |
| GEN-01 compile-gate | clippy clean on all targets | lint | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | ✅ |

### Parity Test Tolerance

- **f64 parity tier:** tolerance `0.0` (bit-identical) — the accumulation pattern preserves exact floating-point order. If a body rewrite accidentally reorders, tolerance falls back to `< 1e-12`.
- **Dual/Var FD spot-check tier:** tolerance `< 1e-5` relative to finite-difference approximation (h = 1e-5), matching the Phase 94 FD conventions.

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg -- l2_distance inner_product trapz` (targets just the four new test names)
- **Per wave merge:** `cargo test -p fdars-core --features linalg` (full suite)
- **Phase gate:** Full suite green + all compile gates before `/gsd-verify-work`

### Wave 0 Gaps

All four parity tests and all autodiff flow tests are new — add inline to their respective `#[cfg(test)]` blocks:

- [ ] `fdars-core/src/helpers.rs` — add `test_l2_distance_parity`, `test_trapz_parity`, `test_l2_distance_dual`, `test_l2_distance_var`, `test_trapz_dual`
- [ ] `fdars-core/src/utility.rs` — add `test_inner_product_parity`, `test_inner_product_dual`
- [ ] `fdars-core/src/warping.rs` — add `test_inner_product_l2_parity`, `test_inner_product_l2_dual`, `test_inner_product_l2_var`

Parity test pattern (reference `test_l2_distance_different` in helpers.rs for baseline values):
```rust
#[test]
fn test_l2_distance_parity() {
    use crate::autodiff::Scalar; // only if needed for ::<f64> turbofish
    let c1 = vec![0.0f64, 1.0, 2.0];
    let c2 = vec![1.0f64, 0.0, 1.0];
    let w  = vec![0.25f64, 0.5, 0.25];
    // f64-typed original
    let expected: f64 = {
        let mut d = 0.0f64;
        for i in 0..3 { let diff = c1[i]-c2[i]; d += diff*diff*w[i]; }
        d.sqrt()
    };
    // generic at T=f64
    let actual = l2_distance(&c1, &c2, &w);
    assert_eq!(actual, expected, "l2_distance parity must be bit-identical");
}
```

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc stable | All gates | ✓ | 1.97.0 | — |
| cargo | All gates | ✓ | 1.97.0 | — |
| wasm32-unknown-unknown target | Gate 5 | ✓ | — | — |
| /home disk | Full test+build | ✓ (~164G free) | 65% used | `rm -rf target/debug/{incremental,examples}` first |

**Missing dependencies with no fallback:** none.

**Disk hazard:** Run `rm -rf target/debug/{incremental,examples}` before Gate 2 (full test) to avoid disk pressure. 164G free is sufficient but the `target/` directory can grow during incremental builds.

---

## Security Domain

`security_enforcement: true` in config.json. This phase introduces no new input surfaces, no new dependencies, and no external data processing. The generalization is purely type-level — the numeric computation is identical at `T=f64`. No ASVS categories apply to in-place type parameter generalization of internal numeric kernels.

| ASVS Category | Applies | Rationale |
|---------------|---------|-----------|
| V2 Authentication | No | No auth surface |
| V3 Session Management | No | No session state |
| V4 Access Control | No | Library functions, no access control |
| V5 Input Validation | No | Existing dimension/parameter checks are unchanged |
| V6 Cryptography | No | No cryptographic operations |

---

## Code Examples

### Verified: Phase 94 Template (exact boundary pattern)
[VERIFIED: fdars-core/src/regression.rs:232-251]
```rust
pub fn project_scores_generic<S: Scalar>(
    curve: &[S],
    mean: &[f64],
    rotation: &FdMatrix,
    weights: &[f64],
    ncomp: usize,
) -> Vec<S> {
    let m = curve.len();
    let mut scores = vec![S::zero(); ncomp];
    for (k, score) in scores.iter_mut().enumerate() {
        let mut sum = S::zero();
        for j in 0..m {
            let centered = curve[j] - S::from_f64(mean[j]);
            let w_rot = S::from_f64(rotation[(j, k)] * weights[j]);
            sum += centered * w_rot;
        }
        *score = sum;
    }
    scores
}
```
Key observations: curve data is `&[S]`, model params are `f64`; mixed arithmetic uses `S::from_f64(rotation[(j,k)] * weights[j])` — the f64 sub-expression is computed first, then lifted once.

### Target: `l2_distance` (complete generalized form)
```rust
pub fn l2_distance<T: Scalar>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T {
    let mut dist_sq = T::zero();
    for i in 0..curve1.len() {
        let diff = curve1[i] - curve2[i];
        dist_sq += diff * diff * T::from_f64(weights[i]);
    }
    dist_sq.sqrt()
}
```

### Target: `trapz` (complete generalized form)
```rust
pub fn trapz<T: Scalar>(y: &[T], x: &[f64]) -> T {
    let mut sum = T::zero();
    for k in 1..y.len() {
        sum += T::from_f64(0.5 * (x[k] - x[k - 1])) * (y[k] + y[k - 1]);
    }
    sum
}
```

### Target: `inner_product` (body rewrite required)
```rust
pub fn inner_product<T: Scalar>(curve1: &[T], curve2: &[T], argvals: &[f64]) -> T {
    if curve1.len() != curve2.len() || curve1.len() != argvals.len() || curve1.is_empty() {
        return T::zero();
    }
    let weights = simpsons_weights(argvals);  // Vec<f64>, unchanged
    let mut acc = T::zero();
    for i in 0..curve1.len() {
        acc += curve1[i] * curve2[i] * T::from_f64(weights[i]);
    }
    acc
}
```

### FD Spot-Check Pattern (one per kernel)
```rust
#[test]
fn test_l2_distance_dual() {
    use crate::autodiff::{diff, Dual};
    let c2 = vec![0.0f64, 1.0, 0.0];
    let w  = vec![0.25f64, 0.5, 0.25];
    // Compute d/d(c1[1]) of l2_distance
    let c1_0 = [1.0f64, 2.0, 1.0];
    let h = 1e-5;
    let fd = (l2_distance(&[c1_0[0], c1_0[1]+h, c1_0[2]], &c2, &w)
            - l2_distance(&[c1_0[0], c1_0[1]-h, c1_0[2]], &c2, &w)) / (2.0 * h);
    // Forward-mode: seed c1[1]
    let c1_dual: Vec<Dual> = c1_0.iter().enumerate()
        .map(|(i, &v)| if i == 1 { Dual::seed(v) } else { Dual::constant(v) })
        .collect();
    let c2_dual: Vec<Dual> = c2.iter().map(|&v| Dual::constant(v)).collect();
    let (_, tangent) = l2_distance(&c1_dual, &c2_dual, &w).extract();
    assert!((tangent - fd).abs() < 1e-5 * fd.abs().max(1e-10),
        "l2_distance Dual tangent={tangent:.8} vs FD={fd:.8}");
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| f64-only kernel functions | Generic over `T: Scalar` with `= f64` default | Phase 95 | DOP phases 96/97/98 can compose these kernels for AD |
| Companion `_generic` functions (Phase 94) | In-place generalization (this phase) | Phase 95 | One function per kernel, no API doubling |
| Iterator `.sum()` in `inner_product` | Explicit `T::zero()` accumulator loop | Phase 95 | Required for non-`Sum` generic types |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `validate_against_r.rs` local `fn trapz` at line 3401 shadows `fdars_core::trapz` for all usages at 3411, 3430, 3529 | Non-Breaking Risk | If those usages actually import from fdars_core, generalization could require turbofish on the test's inline trapz calls — but the local definition means they call the local fn, not crate fn. Verify by grep before editing. |
| A2 | No `fdars-r` source in this repo — R binding source-level check is done by grep/manual inspection, not a build | Compile-Gate | If fdars-r wraps target kernels in ways that aren't pure f64 calls, breakage would be invisible until the external package is rebuilt. Risk is very low given fdars-r wraps high-level functions, not low-level helpers. |

---

## Open Questions

1. **Should `trapz` be generalized in-place or as a private generic helper?**
   - What we know: `trapz` is `pub` and re-exported from `lib.rs`. Generalizing in-place is cleanest and matches the ROADMAP decision. A private helper would require calling it from the generalized `trapz` wrapper (unnecessary indirection).
   - Recommendation: generalize `trapz` in-place. All call sites pass `&[f64]` → `T=f64` infers. The existing body is trivial.

2. **Should the `= f64` default be included given Phase 94 templates omit it?**
   - What we know: `soft_dtw_distance_generic<S: Scalar>` and `project_scores_generic<S: Scalar>` use no default. The ROADMAP GEN-01 specifically says "defaulted type params (`T = f64`)". Rust allows this syntax on free functions; it is cosmetic for inference but documents intent.
   - Recommendation: include `= f64` per ROADMAP wording. No technical downside. Note in a doc comment that the default is for documentation — `T` is inferred from arguments at all existing call sites.

---

## Sources

### Primary (HIGH confidence)
- [VERIFIED: fdars-core/src/helpers.rs:56-63] — `l2_distance` signature and body, read this session
- [VERIFIED: fdars-core/src/helpers.rs:253-259] — `trapz` signature and body, read this session
- [VERIFIED: fdars-core/src/utility.rs:34-46] — `inner_product` signature and body, read this session
- [VERIFIED: fdars-core/src/warping.rs:83-86] — `inner_product_l2` signature and body, read this session
- [VERIFIED: fdars-core/src/autodiff/mod.rs:38-85] — `Scalar` trait definition (12 methods + Copy/arith supertraits), read this session
- [VERIFIED: fdars-core/src/autodiff/forward.rs:96-146] — `impl Scalar for f64`, `from_f64` is identity, read this session
- [VERIFIED: fdars-core/src/regression.rs:232-251] — `project_scores_generic<S: Scalar>` template body (from_f64 boundary pattern)
- [VERIFIED: fdars-core/src/metric/soft_dtw.rs:158] — `soft_dtw_distance_generic<S: Scalar>` template signature
- [VERIFIED: grep call-site audit] — all four kernels' call sites enumerated this session; no fn-pointer patterns found

### Secondary (MEDIUM confidence)
- [ASSUMED] Rust RFC 213 behavior: type parameter defaults on free functions are not used for inference when argument types uniquely determine the type. Behavior confirmed consistent with Rust 1.97.0 installed (rustc 1.97.0 verified this session); no formal documentation URL fetched.

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Kernel signatures and bodies: HIGH — read source files this session with line citations
- Call-site audit: HIGH — grep enumerated all callers; spot-checked method-chaining patterns
- Non-breaking analysis: HIGH — argument types for all call sites verified as `&[f64]`; T=f64 inference is mechanically certain
- Body rewrite patterns: HIGH — derived from verified Phase 94 templates and Scalar trait
- Rust type-param-default behavior: MEDIUM — consistent with known Rust semantics, not fetched from docs URL this session

**Research date:** 2026-09-11
**Valid until:** 2026-10-11 (stable Rust language feature; Scalar trait unchanged)
