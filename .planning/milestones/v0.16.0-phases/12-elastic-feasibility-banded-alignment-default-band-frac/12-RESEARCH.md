# Phase 12: Elastic Feasibility — Banded Alignment `band_frac` Exposure - Research

**Researched:** 2026-08-11
**Domain:** Rust codebase mapping — API surfacing, no new algorithm
**Confidence:** HIGH (all findings verified by reading source files this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **API shape: opt-in, non-breaking.** Expose `band_frac: Option<f64>` on the three high-level functions (or via their config), where `None` (the default) preserves today's **exact unbanded** behavior. Existing callers are unchanged and get identical numerical results; they opt into the 4–6× banded path explicitly by passing `Some(0.1)`.
  - Do **NOT** silently flip the default to banded (`band_frac=0.1`) — that would change existing callers' results (band-approximation) and was explicitly declined. The default-flip may be revisited in a future milestone.

### Claude's Discretion (guided by codebase conventions + the audit)
- Prefer threading `band_frac` through the existing `ElasticConfig`/config struct if one is already the parameter-passing idiom for these functions, rather than adding a positional argument (positional additions are breaking — avoid). If a config struct exists, add `band_frac: Option<f64>` (default `None`) to it; if the public functions take loose params, add `band_frac: Option<f64>` as a trailing optional in a non-breaking way (or a new `*_with_band` wrapper if a clean non-breaking signature is otherwise impossible — but a config struct is strongly preferred).
- `None` → call the existing unbanded impl (current path). `Some(f)` with `f > 0` → `band_radius(f, m)` → banded impl. `Some(0.0)` → treat as exact/unbanded (equivalent to `None`).
- rustdoc on all three functions must document: default `None` = exact; `Some(0.1)` ≈ 4–6× faster with small band-approximation error; band width is a fraction of M.

### Deferred Ideas (OUT OF SCOPE)
- Flipping the default to `band_frac=0.1` (banded-by-default) — declined for this milestone; candidate for a future milestone once users have adopted the opt-in.
- Parallelizing the elastic-FPCA inner loops (PERF-PAR-ELFPCA) — separate deferred backlog item.
</user_constraints>

---

## Summary

Phase 12 is API surfacing only. The banded Sakoe-Chiba DP implementations (`karcher_mean_banded`, `elastic_self_distance_matrix_banded`, `elastic_cross_distance_matrix_banded`) and the `band_radius` helper already exist and are correct. The three target high-level functions (`karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix`) currently take only positional parameters with no config struct — no `ElasticConfig` or equivalent is used at their call sites. The `ElasticConfig` struct belongs to `elastic_regression`/`elastic_logistic`, not to these alignment functions.

**The cleanest non-breaking approach is:** introduce a new `BandedAlignConfig` struct (or equivalently named) with `band_frac: Option<f64>` and add three new `*_with_band` wrapper functions that accept it, delegating to the appropriate `_banded` or unbanded `_impl`. Simultaneously, add the three existing `_banded` variants plus the three new `*_with_band` variants to `lib.rs` crate-root re-exports. The existing `karcher_mean`, `elastic_self_distance_matrix`, and `elastic_cross_distance_matrix` signatures are **not touched**, so all ~30+ internal call sites remain valid.

An alternative simpler approach — add `band_frac: Option<f64>` as a new trailing parameter to the existing three functions — is **breaking in Rust** (no default parameters). Do not use this approach.

**Primary recommendation:** New `*_with_band` wrappers + crate-root re-exports. See Architecture Patterns for the exact pattern.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Band-radius computation | `alignment/mod.rs` (shared helper) | — | `band_radius` is `pub(super)` in mod.rs, used by karcher.rs and pairwise.rs |
| Banded DP path (algorithm) | `alignment/karcher.rs` + `alignment/pairwise.rs` | — | Already implemented; `karcher_mean_impl` and `*_impl` fns accept `Option<usize>` band |
| Public API surface (high-level fns) | `alignment/karcher.rs` + `alignment/pairwise.rs` | `src/lib.rs` (crate-root re-export) | New wrappers go in same files as the unbanded functions |
| Crate-root discoverability | `src/lib.rs` | — | `pub use alignment::{}` block; _banded variants not currently listed |
| Equivalence/regression tests | `alignment/tests.rs` | `fdars-core/tests/` | Inline unit tests for wide-band equivalence; integration test if cross-module |
| Feasibility timing bench | `fdars-core/benches/audit_hotpaths.rs` | — | Existing Phase-3 bench infrastructure already covers N=500/M=200 cells |

---

## Standard Stack

No new dependencies. This phase touches only existing Rust source files within `fdars-core`.

| Component | File | Role |
|-----------|------|------|
| `band_radius` helper | `fdars-core/src/alignment/mod.rs:533-539` | Converts `band_frac: f64` to `Option<usize>` band radius |
| `karcher_mean_impl` | `fdars-core/src/alignment/karcher.rs:323-442` | Shared impl accepting `band_frac: f64`; 0.0 → unbanded |
| `self_distance_matrix_impl` | `fdars-core/src/alignment/pairwise.rs:215-251` | Shared impl accepting `band: Option<usize>` |
| `cross_distance_matrix_impl` | `fdars-core/src/alignment/pairwise.rs:289-330` | Shared impl accepting `band: Option<usize>` |

**Installation:** No new packages. No `Cargo.toml` changes.

---

## Package Legitimacy Audit

Not applicable — no external packages are installed in this phase.

---

## Exact Current Signatures

All signatures read from source this session.

### `karcher_mean`
**File:** `fdars-core/src/alignment/karcher.rs:293-300`
**Verbatim:**
```rust
pub fn karcher_mean(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> KarcherMeanResult {
    karcher_mean_impl(data, argvals, max_iter, tol, lambda, 0.0)
}
```
[VERIFIED: fdars-core/src/alignment/karcher.rs:293-300]

Has `#[must_use = "expensive computation whose result should not be discarded"]` at line 292.

### `karcher_mean_banded`
**File:** `fdars-core/src/alignment/karcher.rs:312-321`
**Verbatim:**
```rust
pub fn karcher_mean_banded(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
    band_frac: f64,
) -> KarcherMeanResult {
    karcher_mean_impl(data, argvals, max_iter, tol, lambda, band_frac)
}
```
[VERIFIED: fdars-core/src/alignment/karcher.rs:312-321]

### `elastic_self_distance_matrix`
**File:** `fdars-core/src/alignment/pairwise.rs:194-196`
**Verbatim:**
```rust
pub fn elastic_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix {
    self_distance_matrix_impl(data, argvals, None, lambda)
}
```
[VERIFIED: fdars-core/src/alignment/pairwise.rs:194-196]

No `#[must_use]` on the unbanded variant (the `_banded` variant has it).

### `elastic_self_distance_matrix_banded`
**File:** `fdars-core/src/alignment/pairwise.rs:205-213`
**Verbatim:**
```rust
pub fn elastic_self_distance_matrix_banded(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    band_frac: f64,
) -> FdMatrix {
    let band = band_radius(band_frac, argvals.len());
    self_distance_matrix_impl(data, argvals, band, lambda)
}
```
[VERIFIED: fdars-core/src/alignment/pairwise.rs:205-213]

### `elastic_cross_distance_matrix`
**File:** `fdars-core/src/alignment/pairwise.rs:266-273`
**Verbatim:**
```rust
pub fn elastic_cross_distance_matrix(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
) -> FdMatrix {
    cross_distance_matrix_impl(data1, data2, argvals, None, lambda)
}
```
[VERIFIED: fdars-core/src/alignment/pairwise.rs:266-273]

### `elastic_cross_distance_matrix_banded`
**File:** `fdars-core/src/alignment/pairwise.rs:278-287`
**Verbatim:**
```rust
pub fn elastic_cross_distance_matrix_banded(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    band_frac: f64,
) -> FdMatrix {
    let band = band_radius(band_frac, argvals.len());
    cross_distance_matrix_impl(data1, data2, argvals, band, lambda)
}
```
[VERIFIED: fdars-core/src/alignment/pairwise.rs:278-287]

### `band_radius` (shared helper)
**File:** `fdars-core/src/alignment/mod.rs:533-539`
**Verbatim:**
```rust
pub(super) fn band_radius(band_frac: f64, m: usize) -> Option<usize> {
    if band_frac > 0.0 && band_frac < 1.0 {
        Some(((band_frac * m as f64).ceil() as usize).max(1))
    } else {
        None
    }
}
```
[VERIFIED: fdars-core/src/alignment/mod.rs:533-539]

**Semantics:** `band_frac <= 0.0` or `>= 1.0` → `None` (full unbanded DP). `0.0 < band_frac < 1.0` → `Some(max(1, ceil(band_frac * m)))`. `band_frac = 0.99` at `m=40` → `Some(40)` which equals or exceeds `m-1=39`, covering the full grid exactly. `band_frac = 0.0` (current hardcoded value in `karcher_mean`) → `None` → unbanded. [VERIFIED: fdars-core/src/alignment/mod.rs:533-539]

---

## Band-Plumbing Flow

### `karcher_mean_impl` band threading
**File:** `fdars-core/src/alignment/karcher.rs:323-442`

The impl receives `band_frac: f64`. Key plumbing:
- Line 333: `let fine_band = band_radius(band_frac, m);` — computes `Option<usize>` for the full grid
- Line 340: `pre_center_template(..., fine_band)` passes band to initial centering
- Line 358: `let coarse_band = band_radius(band_frac, m_c);` — recomputes for the coarse-grid phase (coarse-to-fine strategy active when `m > 50 && max_iter >= 10`)
- Inner alignment loops at lines ~185, 376, 413, 432 use `iter_maybe_parallel!` + the computed band option

`karcher_mean()` hard-codes `band_frac=0.0` at line 300: `karcher_mean_impl(data, argvals, max_iter, tol, lambda, 0.0)`. This is the change point: a new `karcher_mean_with_band` wrapper would pass `band_frac.unwrap_or(0.0)` instead. [VERIFIED: fdars-core/src/alignment/karcher.rs:293-342]

### `self_distance_matrix_impl` / `cross_distance_matrix_impl` band threading
The `_banded` variants call `band_radius(band_frac, argvals.len())` to produce `Option<usize>` and pass it directly to `*_impl`. The impl passes it to `elastic_distance_from_srsf(..., band, ...)` which routes to `dp_alignment_core_banded(..., band)`. The `None` path is the unbanded O(m²) DP; `Some(r)` path is O(m·r). [VERIFIED: fdars-core/src/alignment/pairwise.rs:194-330]

---

## Architecture Patterns

### Config Struct Idiom in This Codebase

The `ElasticConfig` struct (`fdars-core/src/elastic_regression/mod.rs:38-57`) serves `elastic_regression` and `elastic_logistic`. It is **not** related to `karcher_mean` or the distance-matrix functions. [VERIFIED: fdars-core/src/elastic_regression/mod.rs:38-57]

Fields of `ElasticConfig` verbatim:
```rust
pub struct ElasticConfig {
    pub ncomp_beta: usize,
    pub lambda: f64,
    pub max_iter: usize,
    pub tol: f64,
}
```

The alignment functions (`karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix`) take only loose positional parameters. **No config struct exists for them.** [VERIFIED: fdars-core/src/alignment/karcher.rs:293-300, fdars-core/src/alignment/pairwise.rs:194-273]

Precedent for paired functions: `elastic_regression` (positional) + `elastic_regression_with_config` (config wrapper) at `fdars-core/src/elastic_regression/regression.rs:65-177`. [VERIFIED: fdars-core/src/elastic_regression/regression.rs:160-177]

### Recommended Non-Breaking Change Pattern

**Recommendation: Three new `*_with_band(…, band_frac: Option<f64>)` wrappers in the same source files, plus crate-root re-exports.**

This avoids touching any existing signature, leaves all 30+ internal `karcher_mean` call sites intact, and does not require a config struct (which would be heavier for a single parameter).

#### `karcher_mean_with_band` (new, in `karcher.rs`)
```rust
/// Karcher mean with optional Sakoe–Chiba band. `None` preserves exact
/// unbanded behavior; `Some(0.1)` is typically 4–6× faster.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn karcher_mean_with_band(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
    band_frac: Option<f64>,
) -> KarcherMeanResult {
    karcher_mean_impl(data, argvals, max_iter, tol, lambda, band_frac.unwrap_or(0.0))
}
```

`Some(0.0)` → `unwrap_or(0.0)` → `band_radius(0.0, m)` → `None` → unbanded (correct per CONTEXT spec). `Some(f)` with `f > 0` → banded path.

#### `elastic_self_distance_matrix_with_band` (new, in `pairwise.rs`)
```rust
/// Self-distance matrix with optional Sakoe–Chiba band. `None` = exact unbanded.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_self_distance_matrix_with_band(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    band_frac: Option<f64>,
) -> FdMatrix {
    let band = band_frac.map(|f| band_radius(f, argvals.len())).flatten();
    self_distance_matrix_impl(data, argvals, band, lambda)
}
```

#### `elastic_cross_distance_matrix_with_band` (new, in `pairwise.rs`)
```rust
/// Cross-distance matrix with optional Sakoe–Chiba band. `None` = exact unbanded.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_cross_distance_matrix_with_band(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    band_frac: Option<f64>,
) -> FdMatrix {
    let band = band_frac.map(|f| band_radius(f, argvals.len())).flatten();
    cross_distance_matrix_impl(data1, data2, argvals, band, lambda)
}
```

#### Crate-root re-exports to add to `src/lib.rs` (lines 138-167 block)

Add to the `pub use alignment::{…}` block:
```
elastic_self_distance_matrix_with_band,
elastic_cross_distance_matrix_with_band,
karcher_mean_banded,
karcher_mean_with_band,
elastic_self_distance_matrix_banded,
elastic_cross_distance_matrix_banded,
```

Also add the new functions to `alignment/mod.rs` re-exports from `karcher` and `pairwise`.

**Why not a config struct?** A config struct is heavier API surface for a single `Option<f64>` parameter. The `*_with_band` pattern mirrors the codebase's existing `_banded` variant convention (caller-facing, same module), while `Option<f64>` cleanly encodes "absent = exact". No struct needed.

**Alternative considered (rejected):** Adding `band_frac: Option<f64>` as a trailing positional parameter to the existing three functions. **This is breaking in Rust** — all existing callers (`elastic_changepoint.rs:95,137,204`, `elastic_fpca.rs` (14+ calls), `tsrvf.rs:155,315`, `lambda_cv.rs:130`, `transfer.rs:111,120`, `persistence.rs:140`, `robust_karcher.rs:95,193,225`, `tolerance/elastic.rs:113,247,315`, `spm/elastic_spm.rs:233`, 3 examples, and tests) would fail to compile.

### Recommended Project Structure — Files Changed

```
fdars-core/src/alignment/
├── karcher.rs           # Add: karcher_mean_with_band()
├── pairwise.rs          # Add: elastic_self_distance_matrix_with_band()
│                        #      elastic_cross_distance_matrix_with_band()
└── mod.rs               # Add _with_band fns to pub use blocks

fdars-core/src/
└── lib.rs               # Add _with_band + _banded variants to pub use alignment::{…}

fdars-core/src/alignment/
└── tests.rs             # Add: equivalence test for each _with_band fn

fdars-core/benches/
└── alignment_benchmarks.rs  # Add: timing bench for _with_band at large (N, M)
                              # (or extend audit_hotpaths.rs — see below)
```

---

## Crate-Root / Prelude Re-exports

**Current state (lib.rs:138-167):** `karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix` are re-exported at crate root. The `_banded` variants and the `_with_band` wrappers (which don't exist yet) are **not**. [VERIFIED: fdars-core/src/lib.rs:138-167]

**`alignment/mod.rs` current state:** `karcher_mean_banded` re-exported at line 66; `elastic_self_distance_matrix_banded` and `elastic_cross_distance_matrix_banded` re-exported at lines 77-79. [VERIFIED: fdars-core/src/alignment/mod.rs:66-80]

**Prelude (`src/prelude.rs`):** Does not export any alignment functions. No change needed there. [VERIFIED: fdars-core/src/prelude.rs:1-40]

**Change required in lib.rs:** Add the six new/existing items to the `pub use alignment::{…}` block. The block spans lines 138-167; add to the function list (alphabetical order preferred to match existing style). No other file at crate-root level requires changes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Sakoe-Chiba DP with band | Custom banded DP | `karcher_mean_banded` / `elastic_*_banded` (already correct) |
| Band radius computation | Custom `ceil(f*m)` formula | `band_radius(band_frac, m)` in `alignment/mod.rs:533` |
| `Option<f64>` → `Option<usize>` | Inline conversion | `.map(|f| band_radius(f, m)).flatten()` or `unwrap_or(0.0)` into `band_radius` |

**Key insight:** Every banded algorithm is already implemented and tested. The only new code is thin delegation wrappers and re-export lines.

---

## Existing Tests and Benches

### Unit tests (inline in `alignment/tests.rs`)

All tests verified by reading source this session:

| Test | Line | Covers |
|------|------|--------|
| `test_banded_align_wide_matches_unbanded` | 2664 | `elastic_align_pair_banded` at `band_frac=0.99` matches unbanded within `1e-12` |
| `test_banded_align_is_constrained_valid_warp` | 2682 | Narrow band produces valid monotone warp; distance >= unbanded |
| `test_self_distance_matrix_banded_wide_matches` | 2704 | `elastic_self_distance_matrix_banded` at `band_frac=0.99`, n=6, m=40: matches unbanded within `1e-12` |
| `test_karcher_mean_banded_runs` | 2718 | `karcher_mean_banded` at `band_frac=0.2`, n=12, m=50: runs, valid warp monotonicity |

[VERIFIED: fdars-core/src/alignment/tests.rs:2664-2733]

**Gap:** No wide-band equivalence test for `karcher_mean_with_band(…, Some(0.99))` vs `karcher_mean(…)`, and no equivalence test for the `_cross_distance_matrix_with_band`. The plan must add these.

**Gap:** No test asserting `karcher_mean_with_band(…, None)` produces byte-identical results to `karcher_mean(…)`. Add this.

### Integration tests (`fdars-core/tests/`)

| File | Relevance |
|------|-----------|
| `validate_phase_bands.rs` | Calls `karcher_mean` at line 1685; pattern for wide-band equivalence integration tests |
| `validate_against_r.rs` | Calls `karcher_mean` and `elastic_self_distance_matrix` for R-validation |

[VERIFIED: fdars-core/tests/validate_phase_bands.rs:1-55, fdars-core/tests/validate_against_r.rs:1680,1768]

### Benchmarks

| File | Bench function | What it covers |
|------|---------------|----------------|
| `alignment_benchmarks.rs:112-147` | `bench_karcher_mean` | `karcher_mean` at n=10,20 m=50 — no banded variant |
| `alignment_benchmarks.rs:84-110` | `bench_self_distance_matrix` | `elastic_self_distance_matrix` + `_banded` at n=10,30,50 m=50 |
| `audit_hotpaths.rs:269-434` | `bench_p3_karcher_banded` | `karcher_mean_banded` at N∈{100,500}×M∈{50,200}, `band_frac=0.1` |
| `audit_hotpaths.rs:436-580` | `bench_p3_elastic_self_banded` | `elastic_self_distance_matrix_banded` same grid |
| `audit_hotpaths.rs:582+` | `bench_p3_elastic_self` | Unbanded cross-distance at full grid |

[VERIFIED: fdars-core/benches/alignment_benchmarks.rs:84-155, fdars-core/benches/audit_hotpaths.rs:269-580]

**Gap for the plan:** The Phase-3 audit benches already benchmark `karcher_mean_banded` at N=500/M=200 (the "infeasible" cell). To demonstrate feasibility via the new `_with_band` API, the plan should add a bench that calls `karcher_mean_with_band(..., Some(0.1))` at a large cell (N=500, M=200) so users can see the result is the same as `karcher_mean_banded`. The simplest approach: extend `alignment_benchmarks.rs` with a `bench_karcher_mean_with_band` group (small-to-medium sizes) and point readers to `audit_hotpaths.rs` for the large cells. **Do not add another N=500/M=200 bench** — `audit_hotpaths.rs` already has it.

---

## Common Pitfalls

### Pitfall 1: Touching existing positional signatures (BREAKING)
**What goes wrong:** Adding `band_frac: Option<f64>` as a trailing parameter to `karcher_mean`, `elastic_self_distance_matrix`, or `elastic_cross_distance_matrix`. All 30+ internal call sites (elastic_changepoint, elastic_fpca, tsrvf, lambda_cv, transfer, persistence, robust_karcher, tolerance/elastic, spm/elastic_spm, 3 examples, many tests) fail to compile.
**How to avoid:** New `*_with_band` wrappers only; existing signatures untouched.
**Warning signs:** Compiler errors in `src/elastic_changepoint.rs`, `src/elastic_fpca.rs`, etc.

### Pitfall 2: Forgetting to add `_with_band` to `alignment/mod.rs` pub use
**What goes wrong:** New functions compile but are not accessible via `fdars_core::alignment::*`, so the crate-root re-export in `lib.rs` also fails.
**How to avoid:** Add to `pub use karcher::{…}` line 66 and `pub use pairwise::{…}` lines 75-80 in `alignment/mod.rs` before touching `lib.rs`.

### Pitfall 3: `band_frac=0.0` semantics with `Some(0.0)`
**What goes wrong:** Caller passes `Some(0.0)` expecting exact unbanded behavior, but the impl calls `band_radius(0.0, m)` → `None` → correct (unbanded). This is actually fine per `band_radius` definition, but must be documented explicitly to prevent caller confusion.
**How to avoid:** Rustdoc on `_with_band` fns must say: "`Some(0.0)` is treated as exact/unbanded (equivalent to `None`)."

### Pitfall 4: Wide-band equivalence threshold is not `band_frac=0.5`
**What goes wrong:** Test writer assumes any `band_frac > 0.5` covers the full grid. The threshold is `ceil(f*m) >= m-1`, i.e., `f >= (m-2)/m`. At `m=50` this is `f >= 0.96`; at `m=40` it is `f >= 0.95`. `band_frac=0.5` at `m=50` gives `radius=25`, which is `< m-1=49`, so the band does NOT cover the full grid and results may differ.
**How to avoid:** Use `band_frac=0.99` in all wide-band equivalence tests. Document the threshold formula.
**Warning signs:** Equivalence test fails at `band_frac=0.5` — this is expected, not a bug.

### Pitfall 5: `elastic_self_distance_matrix` missing `#[must_use]`
**What goes wrong:** The unbanded `elastic_self_distance_matrix` and `elastic_cross_distance_matrix` currently lack `#[must_use]`. The `_banded` and `_with_band` variants should have it for consistency. Do not add it retroactively to the existing unbanded fns in this phase (that's a separate lint change); just apply it to the new wrappers.
**How to avoid:** Add `#[must_use = "…"]` to all three `*_with_band` wrappers.

### Pitfall 6: Parallelism feature-gate interaction
**What goes wrong:** Tests run single-threaded by default; `parallel` feature gates `rayon` usage. Banded and unbanded results can differ at floating-point level between parallel and single-threaded runs due to summation order. Do not compare parallel-mode vs sequential-mode outputs in equivalence tests.
**How to avoid:** Both legs of a wide-band equivalence test (`_with_band(None)` vs `karcher_mean`) run in the same test binary with the same feature flags. Tolerance `1e-12` used in existing equivalence tests is appropriate.

---

## Numerical-Equivalence Test Design

The existing `test_self_distance_matrix_banded_wide_matches` test (line 2704) establishes the correct pattern:
- Use `band_frac=0.99` (safe above the `(m-2)/m` threshold for any `m >= 10`)
- Use small-to-medium `m` (m=30 to 50) so the test runs fast
- Assert element-wise difference `< 1e-12`

New tests to add (in `alignment/tests.rs`):

1. **`test_karcher_mean_with_band_none_matches_exact`**: `karcher_mean_with_band(..., None)` == `karcher_mean(...)` element-wise within `1e-15` (they call identical code paths — `unwrap_or(0.0)` → `band_radius(0.0,m)` → `None` → same impl). Use n=8, m=30.

2. **`test_karcher_mean_with_band_wide_matches_unbanded`**: `karcher_mean_with_band(..., Some(0.99))` matches `karcher_mean(...)` within `1e-12`. Use n=8, m=30. (band covers full grid at m=30: `ceil(0.99*30)=30 >= 29=m-1`).

3. **`test_self_distance_matrix_with_band_none_matches_exact`**: analogous, n=5, m=30, tolerance `1e-15`.

4. **`test_cross_distance_matrix_with_band_none_matches_exact`**: analogous.

5. **`test_cross_distance_matrix_with_band_wide_matches_unbanded`**: `Some(0.99)` matches unbanded within `1e-12`.

---

## Validation Architecture

`nyquist_validation` is enabled (not `false` in `.planning/config.json`). [VERIFIED: .planning/config.json]

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) |
| Config file | None (uses `#[test]`, `#[cfg(test)]`) |
| Quick run command | `cargo test -p fdars-core alignment::tests` |
| Full suite command | `cargo test -p fdars-core` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PERF-03 | `karcher_mean_with_band(None)` = exact unbanded | unit | `cargo test -p fdars-core test_karcher_mean_with_band_none` | ❌ Wave 0 |
| PERF-03 | `karcher_mean_with_band(Some(0.99))` ≈ unbanded within 1e-12 | unit | `cargo test -p fdars-core test_karcher_mean_with_band_wide` | ❌ Wave 0 |
| PERF-03 | `elastic_self_distance_matrix_with_band(None)` = exact | unit | `cargo test -p fdars-core test_self_distance_matrix_with_band_none` | ❌ Wave 0 |
| PERF-03 | `elastic_cross_distance_matrix_with_band(None)` = exact | unit | `cargo test -p fdars-core test_cross_distance_matrix_with_band_none` | ❌ Wave 0 |
| PERF-03 | `elastic_cross_distance_matrix_with_band(Some(0.99))` ≈ unbanded | unit | `cargo test -p fdars-core test_cross_distance_matrix_with_band_wide` | ❌ Wave 0 |
| PERF-03 | Crate compiles with all existing callers untouched | compile | `cargo build -p fdars-core` | ✅ existing |
| PERF-03 | New fns accessible via `fdars_core::karcher_mean_with_band` | compile | `cargo build -p fdars-core` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --lib alignment 2>&1 | tail -5`
- **Per wave merge:** `cargo test -p fdars-core`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `fdars-core/src/alignment/tests.rs` — add 5 new `_with_band` equivalence tests
- [ ] `fdars-core/src/alignment/karcher.rs` — add `karcher_mean_with_band` function
- [ ] `fdars-core/src/alignment/pairwise.rs` — add two `*_with_band` functions
- [ ] `fdars-core/src/alignment/mod.rs` — add `_with_band` fns to pub use blocks
- [ ] `fdars-core/src/lib.rs` — add 6 items to `pub use alignment::{…}` block

---

## Security Domain

No security surface is introduced. This is a pure algorithmic API extension with no I/O, no network, no user-controlled string parsing, and no unsafe code. `security_enforcement` is enabled but no ASVS categories apply to this phase. The only parameter added is a bounded `Option<f64>` processed through the existing `band_radius` function (which clamps the result to `max(1, ceil(f*m))`).

---

## Environment Availability

This phase is code-only changes within `fdars-core`. No external tools beyond the standard Rust toolchain are required.

| Dependency | Required By | Available | Notes |
|------------|------------|-----------|-------|
| Rust 1.81+ | MSRV | ✓ (1.97.0 in dev) | Confirmed in CLAUDE.md |
| `cargo test` | Tests | ✓ | Standard toolchain |
| `cargo bench` | Benches | ✓ | criterion 0.5 already in Cargo.toml |
| `linalg` feature | Not required | — | This phase does not touch linalg/faer code |

**Missing dependencies with no fallback:** None.

---

## Call Site Inventory (Non-Breaking Verification)

All existing callers of the three functions use positional args and will be unaffected by adding new `*_with_band` wrappers. The 30+ call sites verified are:

**`karcher_mean` call sites in `fdars-core/src/`:**
- `elastic_changepoint.rs:95,137,204` [VERIFIED]
- `elastic_fpca.rs:977,1008,1022,1035,1062,1086,1110,1126,1163,1179,1189,1205,1228` (13 calls) [VERIFIED]
- `alignment/tsrvf.rs:155,315` [VERIFIED]
- `alignment/lambda_cv.rs:130` [VERIFIED]
- `alignment/transfer.rs:111,120` [VERIFIED]
- `alignment/persistence.rs:140` [VERIFIED]
- `alignment/robust_karcher.rs:95,193,225` [VERIFIED]
- `alignment/fpns.rs:238,254,282,299` (4 calls) [VERIFIED]
- `alignment/generative.rs:357,443` [VERIFIED]
- `tolerance/elastic.rs:113,247,315` [VERIFIED]
- `spm/elastic_spm.rs:233` [VERIFIED]
- `elastic_regression/pcr.rs` (imports only, no direct call to 3-function set) [VERIFIED]
- Examples: `16_elastic_alignment/main.rs:122`, `19_tsrvf/main.rs:75`, `26_elastic_analysis/main.rs:48` [VERIFIED]
- Tests: `alignment/tests.rs` (multiple), `fdars-core/tests/*.rs` [VERIFIED]

**`elastic_self_distance_matrix` call sites:**
- `alignment/shape.rs:284`, `alignment/pairwise.rs:368`, `alignment/clustering.rs:429`, `alignment/tests.rs:667,692,708,718,2708` [VERIFIED]

**`elastic_cross_distance_matrix` call sites:**
- `alignment/tests.rs:960,977,993` [VERIFIED]

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full O(m²) DP for all grids | Opt-in `_banded` variants at O(m·r) | Already implemented (audit milestone) | 4–6× speedup measurable in benches |
| `_banded` accessible only via `fdars_core::alignment::*` | After this phase: `fdars_core::*_with_band` at crate root | Phase 12 | Users don't need to know about alignment submodule to opt in |

---

## Assumptions Log

All key facts were verified by reading source files this session. No `[ASSUMED]` claims in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | (none) | — | — |

**All claims verified:** No user confirmation needed before planning.

---

## Open Questions

1. **Naming: `*_with_band` vs `*_banded_opt` vs keeping only `_banded`**
   - What we know: `_banded` variants already exist and take `band_frac: f64`. The `_with_band` suffix uses `Option<f64>` for ergonomic "absent = exact" semantics. Codebase has `_with_config` precedent (elastic_regression_with_config) but no `_with_band` precedent.
   - What's unclear: Which suffix the maintainer prefers.
   - Recommendation: Use `_with_band` to signal the `Option<f64>` distinction from the `f64`-taking `_banded` variants. Both sets are then in the public API: `_banded` (fast, f64) and `_with_band` (opt-in, Option<f64>).

2. **Should `_banded` variants also be added to crate-root re-exports?**
   - What we know: `_banded` variants are in `alignment/mod.rs` but not in `lib.rs`. Currently only accessible as `fdars_core::alignment::karcher_mean_banded`.
   - Recommendation: Yes, add them alongside `_with_band` for completeness. The diff to `lib.rs` is minimal.

---

## Sources

### Primary (HIGH confidence)
All findings verified by `Read` tool against source files this session.

- `fdars-core/src/alignment/karcher.rs:293-321, 323-442` — `karcher_mean`, `karcher_mean_banded`, `karcher_mean_impl` signatures
- `fdars-core/src/alignment/pairwise.rs:194-330` — all three distance-matrix function signatures and impls
- `fdars-core/src/alignment/mod.rs:533-539, 60-106` — `band_radius` semantics, module re-export block
- `fdars-core/src/lib.rs:138-167` — crate-root re-export block (confirmed `_banded` variants absent)
- `fdars-core/src/elastic_regression/mod.rs:38-57` — `ElasticConfig` fields (confirmed: unrelated to karcher/pairwise)
- `fdars-core/src/alignment/tests.rs:2664-2733` — existing banded equivalence tests
- `fdars-core/benches/alignment_benchmarks.rs:84-155` — existing alignment bench coverage
- `fdars-core/benches/audit_hotpaths.rs:269-580` — Phase-3 large-grid bench coverage
- `.planning/config.json` — `nyquist_validation: true`

### Secondary (MEDIUM confidence)
- `fdars-core/src/elastic_regression/regression.rs:160-177` — `elastic_regression_with_config` as `*_with_config` naming precedent

---

## Metadata

**Confidence breakdown:**
- Existing signatures: HIGH — read from source this session with file:line citations
- Band-plumbing flow: HIGH — traced from public fn through `_impl` to `band_radius`
- Non-breaking approach: HIGH — positional-param limitation is a Rust language constraint, not assumption
- Recommended wrapper design: HIGH — follows existing `_banded` precedent exactly
- Equivalence test design: HIGH — extrapolated from existing passing tests at line 2704, same tolerance

**Research date:** 2026-08-11
**Valid until:** Stable — source code doesn't change until implementation starts
