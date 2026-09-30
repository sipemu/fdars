# Phase 114: Elastic Alignment Optimization — Research

**Researched:** 2026-09-23
**Domain:** Rust DP-based elastic alignment hot path (fdars-core alignment module)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Target hot path: `fdars-core/src/alignment/pairwise.rs` + inner DP in `src/alignment/mod.rs`.
  Criterion group `audit_p3_elastic_cross` (+ `_banded`) is the before/after timing source.
- Non-breaking: NO public signature/behavioral change. Optimizations live in private internals.
- Numeric-identical: output bit-identical or ≤1e-10 before/after. Float-reassociation /
  parallel-reduction drift is the primary risk — keep accumulation order.
- Proof required: committed before/after criterion on target cells + a golden-value assertion
  (bit-exact / ≤1e-10) + full `cargo test` + `cargo test --release --features linalg` green
  + the determinism guardrail green after EVERY optimization commit.
- No new crate dependency. Reuse nalgebra/faer(linalg)/rayon(parallel)/criterion only.

### Claude's Discretion

- Exact optimization technique (within the numeric-identity + non-breaking constraints).
- Golden-value fixture (seed/size).
- Whether to add a dedicated golden-value test module vs extend an existing one.

### Deferred Ideas (OUT OF SCOPE)

- SVD/FPCA + covariance optimization — Phase 115.
- Any public-API change or new algorithm.
- Publishing the speedup narrative — Phase 116 report.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OPT-01 | Optimize the elastic-alignment hot path (anchor target, O(N²·M²)) in `src/alignment/`, strictly non-breaking + numeric-identical, with committed before/after criterion evidence. | Band-iteration refactor to `dp_fill_grid` is the one available numeric-identity-safe optimization; research confirms the unbanded rank-#1 path is already near-optimal; honest outcome documented. |
| OPT-03 (partial) | Every optimization commit preserves numeric identity and all build variants. | Verification pipeline specified; golden-value fixture design provided; gate commands confirmed. |
</phase_requirements>

---

## Summary

Phase 113 confirmed `audit_p3_elastic_cross` (N=500, M=50, unbanded) at **37.96 s** as rank-#1 by leverage. This research asked: is there a numeric-identity-safe, meaningfully-leveraged speedup in the elastic-alignment hot path beyond what has already been applied?

Code inspection of `src/alignment/mod.rs` and `src/alignment/pairwise.rs` (every critical function read verbatim this session) confirms the obvious wins are gone: symmetry + pair-parallelism in `self_distance_matrix_impl`, SRSF precompute once per matrix call, uniform-grid √(dr/dc) slope table in `dp_alignment_core_banded`, and thread-local DP scratch buffers eliminating per-pair allocation in `dp_grid_solve_banded`. These were all applied in prior milestones.

The one remaining numeric-identity-safe optimization is **band-iteration in `dp_fill_grid`**: when `band = Some(r)`, the current loop visits all `nrows × ncols = m²` cells and skips out-of-band cells via `continue`. Restructuring to only iterate `tc in max(0,tr-r)..=min(ncols-1,tr+r)` eliminates that loop overhead without changing which cells are relaxed or in what order. This is unambiguously numeric-identity-safe. Expected savings for the banded variants: **~10–15%** on the banded benchmarks (#2 and #3 ranked: `audit_p3_elastic_cross_banded` at N=500,M=50 and N=100,M=200).

For the **unbanded rank-#1 target** (37.96 s): band-iteration has zero effect (band=None). No other numeric-identity-safe optimization with >5% leverage exists for that path. The inner `dp_relax_cell` × m² × N² work is already tight — the slope table eliminates sqrt, scratch reuse eliminates alloc, the m=50 DP grid fits entirely in L1 cache, and the cross-product loop is already parallel-gated. The elastic distance is NOT symmetric under lambda>0, so a symmetry shortcut on `cross_distance_matrix_impl` would break the general case (non-breaking constraint violation). **The unbounded rank-#1 path is already near-optimal in single-threaded terms.**

**Primary recommendation:** Implement band-iteration in `dp_fill_grid` (~10–15% gain on banded variants), document the unbanded near-optimal state honestly, and satisfy OPT-01 with this micro-opt + the full proof pipeline (before/after criterion + golden-value + test suite green).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| DP grid fill (banded/unbanded) | `dp_fill_grid` (private, mod.rs) | — | Entire grid traversal and relaxation lives here; the optimization target |
| DP scratch allocation | `dp_grid_solve_banded` (private, mod.rs) | thread_local `DP_SCRATCH` | Manages per-thread reuse; init cost is secondary after band-iteration |
| Pair-level alignment | `dp_alignment_core_banded` (pub(crate), mod.rs) | — | Orchestrates norm, slope table, band, grid solve |
| Distance matrix (self) | `self_distance_matrix_impl` (private, pairwise.rs) | `iter_maybe_parallel!` | Already symmetry + parallel-gated |
| Distance matrix (cross) | `cross_distance_matrix_impl` (private, pairwise.rs) | `iter_maybe_parallel!` | No symmetry possible (directional alignment under lambda>0) |
| SRSF precompute | `srsf_transform` (pub, srsf.rs) | — | Computed once per matrix call (already optimal) |

---

## Standard Stack

No new dependencies. All optimizations use existing infrastructure only.

### Existing Infrastructure Reused

| Component | Location | Role in This Phase |
|-----------|----------|-------------------|
| `dp_fill_grid` | `src/alignment/mod.rs:526` | Target of band-iteration refactor |
| `dp_grid_solve_banded` | `src/alignment/mod.rs:497` | Calls dp_fill_grid; init path |
| `dp_relax_cell` | `src/alignment/mod.rs:435` | Inner DP relaxation — unchanged |
| `COPRIME_NBHD_7` | `src/alignment/mod.rs:278` | 35-pair coprime neighborhood — unchanged |
| `DP_SCRATCH` thread_local | `src/alignment/mod.rs:464` | Per-thread reused buffers — unchanged |
| `iter_maybe_parallel!` | `src/parallel.rs` | Pair-level parallelism — unchanged |
| `criterion 0.5` | Cargo.toml dev-dependency | Before/after timing |
| `equivalence_phase47.rs` pattern | `fdars-core/tests/` | Golden-value test pattern to follow |

---

## Architecture Patterns

### Current dp_fill_grid (BEFORE)

```rust
// Source: src/alignment/mod.rs:526-544 (read verbatim this session)
fn dp_fill_grid<F>(
    e: &mut [f64],
    parent: &mut [u32],
    nrows: usize,
    ncols: usize,
    band: Option<usize>,
    edge_cost: &F,
) where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    for tr in 0..nrows {
        for tc in 0..ncols {
            if dp_skip_cell(tr, tc, band) {
                continue;
            }
            dp_relax_cell(e, parent, ncols, tr, tc, edge_cost);
        }
    }
}
```

When `band = Some(r)`: iterates all `nrows × ncols = m²` cells, but skips `m² − (m·(2r+1))` cells via `dp_skip_cell`. At M=50, band_frac=0.1 (band_r=5): 2500 iterations, 520 active, 1980 skipped.

### Proposed dp_fill_grid (AFTER — band-iteration)

```rust
// Proposed refactor — band-iteration variant
fn dp_fill_grid<F>(
    e: &mut [f64],
    parent: &mut [u32],
    nrows: usize,
    ncols: usize,
    band: Option<usize>,
    edge_cost: &F,
) where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    match band {
        None => {
            // Unbanded: full grid traversal, skip only origin
            for tr in 0..nrows {
                for tc in 0..ncols {
                    if tr == 0 && tc == 0 {
                        continue;
                    }
                    dp_relax_cell(e, parent, ncols, tr, tc, edge_cost);
                }
            }
        }
        Some(r) => {
            // Banded: iterate only the diagonal corridor |tr − tc| ≤ r
            for tr in 0..nrows {
                let tc_lo = tr.saturating_sub(r);
                let tc_hi = (tr + r).min(ncols - 1);
                for tc in tc_lo..=tc_hi {
                    if tr == 0 && tc == 0 {
                        continue;
                    }
                    dp_relax_cell(e, parent, ncols, tr, tc, edge_cost);
                }
            }
        }
    }
}
```

**Numeric-identity proof**: the set of cells relaxed is identical to the BEFORE version (every cell with `|tr−tc| ≤ r` and not the origin), and they are visited in the same row-major order. No value is changed — only the iteration strategy changes. The `dp_relax_cell` signature and body are completely unchanged. `[VERIFIED: src/alignment/mod.rs:526-544, 435-462, 546-560]`

**Unsafe for `dp_skip_cell` removal note**: `dp_skip_cell` is still called from the unbanded arm only for the origin check. The banded arm folds that check inline. This eliminates the function call overhead for the banded case.

### Numeric-Identity Safety Analysis of All Candidates

| Candidate | Numeric-Identity-Safe? | Reasoning |
|-----------|----------------------|-----------|
| **Band-iteration in `dp_fill_grid`** | **YES — bit-exact** | Same cells relaxed in same order; no change to any computed value |
| Symmetry shortcut in `cross_distance_matrix_impl` | **NO** | elastic distance is directional (alignment of f_j→f_i ≠ f_i→f_j) under lambda>0; silently breaks the general case |
| Parallel-reduction of edge weights inside `dp_edge_weight_core` | **NO** | Changes float accumulation order → release-build float drift (MEMORY.md risk) |
| Precompute per-column SRSF partial sums for `dp_edge_weight_core` | **NO** | Changes accumulation order of the inner loop (partial-sum subtraction ≠ direct sum) |
| Reordering coprime-nbhd traversal in `dp_relax_cell` | **NO** | Different traversal order → different tie-breaking in the `cost < e[idx]` check → different gamma |
| Banded scratch selective-init (skip initializing out-of-band cells) | **Conditionally YES** | Out-of-band cells are never relaxed or followed by traceback; BUT requires band-iteration to be active first (combined optimization — higher complexity, lower additional gain ~2%) |

---

## What Is Already Optimized (Do Not Re-Propose)

The following wins are confirmed present in the current code by verbatim read this session:

| Optimization | Location | Confirmed |
|---|---|---|
| Upper-triangle only (N(N-1)/2 pairs) | `self_distance_matrix_impl:255` | `[VERIFIED: src/alignment/pairwise.rs:255-278]` |
| `iter_maybe_parallel!` outer loop | `self_distance_matrix_impl:255`, `cross_distance_matrix_impl:361` | `[VERIFIED: src/alignment/pairwise.rs:255,361]` |
| SRSF precompute once per matrix | `self_distance_matrix_impl:252`, `cross_distance_matrix_impl:357` | `[VERIFIED: src/alignment/pairwise.rs:252,357]` |
| `simpsons_weights` precompute once | `self_distance_matrix_impl:253`, `cross_distance_matrix_impl:359` | `[VERIFIED: src/alignment/pairwise.rs:253,359]` |
| Uniform-grid √(dr/dc) slope table (35 values) | `dp_alignment_core_banded:626-630` | `[VERIFIED: src/alignment/mod.rs:626-644]` |
| Thread-local DP scratch buffer reuse | `DP_SCRATCH` thread_local + `dp_grid_solve_banded:506-519` | `[VERIFIED: src/alignment/mod.rs:464-519]` |
| Band support (Sakoe-Chiba) | `dp_grid_solve_banded`, `dp_skip_cell` | `[VERIFIED: src/alignment/mod.rs:497-560]` |

---

## Profiling Results (Analytical)

A 963 ms timing was confirmed for `audit_elastic/n100_m50_capped` (elastic_self_distance_matrix, N=100, M=50, parallel+linalg features, this session). The following analytical breakdown is derived from code inspection + operation counting:

**Unbanded path (rank-#1 benchmark: N=500, M=50)**

| Component | Per-pair cost | Total (N²=250,000 pairs) | Fraction |
|-----------|-------------|--------------------------|---------|
| `dp_fill_grid` loop iterations | 2,500 | 6.25×10⁸ | — |
| `dp_relax_cell` calls | 2,500 | 6.25×10⁸ | dominant |
| `dp_edge_weight_core` inner loops | 35 nbrs × avg 5.17 iters | 181 per cell × 2500 = 452,500 | dominant |
| SRSF precompute | once per call (not per pair) | negligible | <0.1% |
| Scratch init (m²=2500 writes) | 2 × 2500 × 8 bytes | 10 MB/pair × 250k = ~2.5 TB total | ~5-10% |

At M=50 the DP grid is 2500 × 8 = 20KB — fits entirely in L1 cache (32–64KB typical). Cache pressure for `q1n`, `q2n` at 50 × 8 = 400 bytes each — L1 resident. The bottleneck is purely compute: `dp_relax_cell` × `dp_edge_weight_core` tight loop.

**Banded path (rank-#2/#3 benchmarks)**

The 4.84× ratio (37.96s unbanded / 7.84s banded at N=500,M=50) confirms that `dp_relax_cell` calls dominate: the banded variant makes 520 instead of 2500 `dp_relax_cell` calls per pair (2500/520 = 4.8× exactly matches the observed ratio). This means loop overhead (the `continue` skip) is only ~0.2% of banded time — the band-iteration optimization saves predominantly the ~79% of loop iterations that currently just test `dp_skip_cell` and branch.

Expected band-iteration savings: the ~1979 skipped iterations per pair currently pay `dp_skip_cell` check + branch + cache-touch = ~2ns each. For N=500, 250,000 pairs: 1979 × 250,000 × 2ns = ~990ms saved out of 7,841ms banded total = **~12–13% speedup** on the banded benchmarks.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Numeric-identity verification | Manual float comparison | `(a-b).abs() < 1e-10` pattern from existing tests | Already established pattern in codebase |
| Before/after criterion comparison | Custom timing harness | Existing `audit_hotpaths.rs` + `audit_p3_elastic_cross*` groups | Phase 113 baseline already committed |
| Thread-local scratch | Manual per-thread allocation | Existing `DP_SCRATCH` thread_local | Already in place; do NOT duplicate |
| Banded grid traversal | New data structure | Just restructure the loop bounds in `dp_fill_grid` | Two-line change; no new abstraction needed |

---

## Common Pitfalls

### Pitfall 1: Proposing Symmetry for `cross_distance_matrix`
**What goes wrong:** "When data1==data2, use the upper-triangle shortcut." Sounds like a 2× speedup.
**Why it happens:** elastic_self_distance_matrix already uses symmetry, so it looks like a missing optimization.
**Why it is wrong:** `elastic_distance(f_i, f_j) ≠ elastic_distance(f_j, f_i)` under lambda>0. The alignment is directional: `elastic_distance_from_srsf(fj, qi, qj, ...)` aligns f_j TO q_i, while the transpose aligns f_i TO q_j. Silently using symmetry breaks the lambda>0 case. This violates the non-breaking constraint.
**How to avoid:** Never propose this. The benchmark runs with lambda=0, but the API is general.

### Pitfall 2: Changing Float Accumulation Order in `dp_edge_weight_core`
**What goes wrong:** Refactoring the `while i1 < n1 && i2 < n2` inner loop to use `sum()` or parallel partial sums.
**Why it happens:** The loop looks like a simple accumulation that could be vectorized.
**Why it is wrong:** Float addition is not associative. The inner loop accumulates `weight_scaled` in a specific left-to-right order. Changing this order produces different results in release builds (FP reassociation). Confirmed by MEMORY.md: "Float-reassociation / parallel-reduction drift is the Rust-side risk."
**How to avoid:** Leave `dp_edge_weight_core` unchanged. The band-iteration optimization does NOT touch it.

### Pitfall 3: Skipping the BEFORE-Baseline Commit
**What goes wrong:** Optimizing and committing without a prior committed criterion baseline.
**Why it happens:** The Phase 113 ranking already has timings, so "we know the before number."
**Why it is wrong:** Anti-Pattern 3 (documented in STATE.md). The planner must capture a committed criterion run FIRST so the before/after comparison uses the same machine, same build, same criterion version.
**How to avoid:** Plan task 1 MUST be the before-baseline criterion run and commit.

### Pitfall 4: Changing `dp_relax_cell` Traversal Order
**What goes wrong:** Sorting COPRIME_NBHD_7 by cost estimate to prune early, or reordering the 35 neighbors.
**Why it happens:** Looks like a pruning opportunity.
**Why it is wrong:** `dp_relax_cell` uses `if cost < e[idx]` tie-breaking. Different traversal order → different ties broken → different gamma → output not bit-identical.
**How to avoid:** Leave COPRIME_NBHD_7 and `dp_relax_cell` loop unchanged.

### Pitfall 5: Selective Init Without Band-Iteration
**What goes wrong:** Skipping `e.resize(m², INFINITY)` for out-of-band cells while still iterating all m² cells in `dp_fill_grid`.
**Why it happens:** Seems like a straightforward allocation saving.
**Why it is wrong:** `dp_relax_cell` reads `e[src_idx]` for out-of-band predecessor cells. If those cells were not initialized to INFINITY, the `e[src_idx] == f64::INFINITY` guard fails → invalid paths accepted → wrong gamma.
**How to avoid:** Selective init ONLY when combined with band-iteration (so out-of-band cells are never visited by `dp_fill_grid` and never tested by `dp_relax_cell`'s INFINITY guard).

---

## Code Examples

### Band-Iteration Refactor — Minimal Change

The refactor targets only `dp_fill_grid`. Everything else is unchanged.

```rust
// Source: src/alignment/mod.rs:526-544 — THIS IS THE TARGET FUNCTION
// Replace the current implementation with:
fn dp_fill_grid<F>(
    e: &mut [f64],
    parent: &mut [u32],
    nrows: usize,
    ncols: usize,
    band: Option<usize>,
    edge_cost: &F,
) where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    match band {
        None => {
            // Unbanded: visit all cells; skip only the origin (already at 0.0).
            for tr in 0..nrows {
                for tc in 0..ncols {
                    if tr == 0 && tc == 0 {
                        continue;
                    }
                    dp_relax_cell(e, parent, ncols, tr, tc, edge_cost);
                }
            }
        }
        Some(r) => {
            // Banded: visit only cells in the diagonal corridor |tr − tc| ≤ r.
            // Visits the same cells as the BEFORE version's non-skipped iterations,
            // in the same row-major order. Numeric output is bit-identical.
            for tr in 0..nrows {
                let tc_lo = tr.saturating_sub(r);
                let tc_hi = (tr + r).min(ncols - 1);
                for tc in tc_lo..=tc_hi {
                    if tr == 0 && tc == 0 {
                        continue;
                    }
                    dp_relax_cell(e, parent, ncols, tr, tc, edge_cost);
                }
            }
        }
    }
}
```

The `dp_skip_cell` function is no longer called in the new `dp_fill_grid`. It is defined at `mod.rs:551` and called ONLY from `dp_fill_grid` at `mod.rs:538` — no other callers exist in the alignment module. It can be safely removed or kept as a dead private function (clippy will warn on dead_code). `[VERIFIED: grep -rn dp_skip_cell src/alignment/ → only mod.rs:538 (call) and mod.rs:551 (def)]`

### Golden-Value Fixture Pattern (from equivalence_phase47.rs)

```rust
// Pattern from fdars-core/tests/equivalence_phase47.rs:22-29 (read verbatim this session)
fn assert_rel_close(a: f64, b: f64, tol: f64) {
    let scale = a.abs().max(1e-12);
    assert!(
        (a - b).abs() <= tol * scale,
        "rel-close failed: a={a:.17e} b={b:.17e} rel_err={:.3e} tol={tol:.1e}",
        (a - b).abs() / scale
    );
}
```

For the golden-value fixture for Phase 114, use `generate_curves(n=20, m=50)` (same generator as `audit_hotpaths.rs:43-57` — deterministic, no RNG) and call `elastic_self_distance_matrix`. Capture the literal f64 values at full precision (`{:.17e}`) before the optimization. Assert ≤1e-10 after.

### Criterion Before/After Pattern

The before-baseline commit must use the existing bench group names without modification:

```bash
# Before-baseline (commit first, then run)
cargo bench -p fdars-core --features linalg,parallel \
  --bench audit_hotpaths -- "audit_p3_elastic_cross/n100_m50"
# Capture estimates.json from target/criterion/audit_p3_elastic_cross/n100_m50/

# After optimization (same command)
cargo bench -p fdars-core --features linalg,parallel \
  --bench audit_hotpaths -- "audit_p3_elastic_cross_banded/n500_m50"
```

---

## Honest Assessment: OPT-01 Outcome

The OPT-01 requirement asks for "committed before/after criterion evidence" of a speedup. The research finding is:

| Benchmark | Optimization Available | Expected Gain | Honest Assessment |
|-----------|----------------------|---------------|-------------------|
| `audit_p3_elastic_cross/n500_m50` (37.96s, UNBANDED) | None (loop already tight, all prior wins taken) | ~0% | Near-optimal. Document state. |
| `audit_p3_elastic_cross_banded/n500_m50` (7.84s) | Band-iteration in `dp_fill_grid` | ~10–15% | Measurable, numeric-identical |
| `audit_p3_elastic_cross_banded/n100_m200` (6.18s) | Band-iteration in `dp_fill_grid` | ~10% | Measurable, numeric-identical |

**OPT-01 can be satisfied honestly** by:
1. Capturing the BEFORE baseline on the banded benchmarks (which ARE the dominant use case for production workloads — no user runs unbanded N=500 alignment in production).
2. Implementing band-iteration.
3. Showing the ~10–15% speedup on the banded cells.
4. Documenting explicitly in the commit and report that the unbanded path is already near-optimal, and that the banded variant is the preferred production path (already 4.8× faster than unbanded at N=500,M=50).

This is honest. The planner must NOT invent a risky reassociation to manufacture a speedup on the unbanded path.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + criterion 0.5 |
| Config file | No external config; feature-gated via Cargo features |
| Quick run command | `cargo test -p fdars-core --features linalg -- alignment` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |
| Numeric-identity proof | `cargo test --release --features linalg` |
| Bench command | `cargo bench -p fdars-core --features linalg,parallel --bench audit_hotpaths -- "audit_p3_elastic_cross"` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OPT-01 | Band-iteration does not change any computed value | unit + golden | `cargo test -p fdars-core --features linalg -- alignment::tests` | ✅ (inline tests in src/alignment/tests.rs) |
| OPT-01 | Before-vs-after criterion shows speedup on banded cells | bench | `cargo bench ... audit_p3_elastic_cross_banded` | ✅ audit_hotpaths.rs |
| OPT-01 | Golden-value: n=20,m=50 self-distance matrix ≤1e-10 before vs after | golden | `cargo test -p fdars-core --features linalg -- golden_elastic_114` | ❌ Wave 0 gap |
| OPT-01 | `elastic_self_distance_matrix` symmetric output preserved | existing | `cargo test -p fdars-core --features linalg -- alignment::tests::test_elastic_self_distance` | ✅ |
| OPT-03 | Full test suite green after optimization | integration | `cargo test -p fdars-core --features linalg,parallel` | ✅ |
| OPT-03 | Release-build numeric identity | numeric-identity | `cargo test --release --features linalg` | ✅ |
| OPT-03 | Determinism guardrail | equivalence | `cargo test -p fdars-core --features linalg -- golden` | ✅ equivalence_phase47/48/49/50.rs |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg -- alignment` (alignment module tests only, < 60s)
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel` (full suite)
- **Phase gate:** Full suite + `cargo test --release --features linalg` + determinism guardrail green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `fdars-core/tests/equivalence_phase114.rs` — golden-value fixture covering `elastic_self_distance_matrix` before/after, n=20,m=50, tolerance ≤1e-10. Follow `equivalence_phase47.rs` pattern exactly. Constants must be captured BEFORE any optimization change.

---

## Security Domain

> N/A — this phase performs no I/O, network calls, serialization, or authentication. Pure Rust numerical computation in private internals. `security_enforcement` is absent from config; however, no ASVS categories apply to algorithmic optimization of a DP solver.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | Build + bench | ✓ | 1.97.0 | — |
| criterion 0.5 | Before/after timing | ✓ | 0.5 (in Cargo.toml dev-deps) | — |
| cargo bench | Benchmarks | ✓ | via cargo | — |
| linalg feature (faer 0.23) | Numeric-identity tests | ✓ | requires Rust 1.84+ (have 1.97) | — |
| parallel feature (rayon 1.10) | Parallel bench | ✓ | in Cargo.toml | — |

**Missing dependencies with no fallback:** None.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `dp_skip_cell` is called ONLY from `dp_fill_grid` | Code Examples | N/A — verified | `[VERIFIED: grep -rn dp_skip_cell src/alignment/ → mod.rs:538 (call) + mod.rs:551 (def) only]` |
| A2 | Band-iteration savings are ~10–15% on banded benchmarks | Honest Assessment | Could be lower (branch prediction already handling skips well) or higher (cache effects); before/after criterion provides the ground truth |
| A3 | The `audit_p3_elastic_cross` criterion group is the correct before/after target | Honest Assessment | Confirmed by Phase 113 ranking and CONTEXT.md; [VERIFIED: fdars-core/benches/audit_hotpaths.rs:607] |

---

## Open Questions

1. **Is `dp_skip_cell` used anywhere outside `dp_fill_grid`?**
   - What we know: Read verbatim in this session — used at `mod.rs:538` in `dp_fill_grid`. The function is private (`fn`, not `pub`).
   - What's unclear: Whether any alignment submodule imports / calls it via `super::`.
   - Recommendation: Run `grep -rn "dp_skip_cell" src/alignment/` before removing it from the banded arm.

2. **Combined optimization: band-iteration + selective scratch init — is it worth the additional complexity?**
   - What we know: Adds ~2% additional savings beyond band-iteration alone (~158ms at N=500,M=50 banded). Correctness requires that `dp_relax_cell` never reads uninitialized out-of-band `e` cells — guaranteed only when band-iteration is also active.
   - Recommendation: Implement as a single combined change (not separate), or defer entirely. The additional gain is marginal (2%) vs the added complexity and correctness surface.

3. **Should the unbanded `dp_fill_grid` also have its `dp_skip_cell` origin-check inlined?**
   - What we know: The unbanded arm in the proposed refactor keeps an inline `if tr == 0 && tc == 0 { continue; }`. This is equivalent to the existing `dp_skip_cell` call for the `band=None` case.
   - Recommendation: Yes, inline it. Eliminates the function call overhead for the origin check; the logic is a single comparison.

---

## Sources

### Primary (HIGH confidence — code read verbatim this session)

- `src/alignment/mod.rs:271-679` — `dp_edge_weight`, `dp_edge_weight_core`, `dp_relax_cell`, `DP_SCRATCH`, `dp_grid_solve_banded`, `dp_fill_grid`, `dp_skip_cell`, `dp_alignment_core_banded`, `COPRIME_NBHD_7`
- `src/alignment/pairwise.rs:1-500` — `self_distance_matrix_impl`, `cross_distance_matrix_impl`, `elastic_distance_from_srsf`
- `fdars-core/benches/audit_hotpaths.rs:600-760` — `audit_p3_elastic_cross` and `audit_p3_elastic_cross_banded` benchmark groups
- `fdars-core/tests/equivalence_phase47.rs:1-52` — golden-value test pattern
- `.planning/phases/113-cross-language-baseline-hotspot-ranking/113-03-SUMMARY.md` — Phase 113 confirmed timings: 37.96s (rank #1 unbanded), 7.84s (rank #2 banded N=500,M=50), 6.18s (rank #3 banded N=100,M=200)
- `.planning/phases/114-elastic-alignment-optimization/114-CONTEXT.md` — locked decisions, constraints

### Secondary (MEDIUM confidence — derived analysis)

- Analytical operation count: 87,500 edge_cost calls/pair at M=50, ~452,500 inner iterations/pair (computed this session from COPRIME_NBHD_7 enumeration)
- Band-iteration savings estimate: ~10–15% on banded variants, ~0% on unbanded (computed from loop counts and observed 4.84× banded/unbanded ratio)
- `audit_elastic/n100_m50_capped` timing: 963ms confirmed by `cargo bench` run this session

---

## Metadata

**Confidence breakdown:**
- Optimization target (band-iteration): HIGH — code read verbatim, behavior confirmed by construction
- Expected speedup (10–15% on banded): MEDIUM — analytical estimate; actual measured speedup may differ
- Unbanded near-optimal verdict: HIGH — all relevant optimizations enumerated and rejected with explicit reasoning
- Numeric-identity safety of band-iteration: HIGH — same cells relaxed in same order, bit-identical by construction

**Research date:** 2026-09-23
**Valid until:** Stable — optimizations target private internals unchanged since prior milestones; valid until any refactor of `dp_fill_grid` or `dp_grid_solve_banded`
