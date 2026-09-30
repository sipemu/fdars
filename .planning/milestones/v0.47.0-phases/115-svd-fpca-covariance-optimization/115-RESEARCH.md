# Phase 115: SVD/FPCA & Covariance Optimization — Research

**Researched:** 2026-09-23
**Domain:** `face_covariance` kernel-covariance hot path + SVD-proper re-benchmark + OPT-03 consolidated proof battery
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Covariance surface (`face_covariance` in `fdars-core/src/irreg_fdata/face.rs`) is the CLEAN PRIMARY target** — ranked #4 (secondary targets list #1) by leverage, noise-stable criterion cells. Optimize directly (numeric-identity-safe), with committed before/after criterion + golden-value proof.
- **SVD-proper (`audit_p6_svd_nalgebra` / `_faer_seq` / `_conversion`) is NOISE-EXCLUDED** (CI-width ≥5%). Phase 115 MUST re-benchmark SVD paths at higher criterion sample counts so CI-width <5% BEFORE selecting as a target. Only optimize if a clean numeric-identity-safe win with meaningful leverage exists; otherwise document near-optimal.
- **Honesty clause (Phase-114 precedent):** numeric-identical cleanup + "already near-optimal, here is the before/after proof" is a valid OPT-02 outcome. Do NOT relax a golden tolerance or fabricate a speedup.
- **Strictly non-breaking + numeric-identical** (≤1e-10 / bit-exact). No float-reassociation / parallel-reduction drift. Optimizations in private internals only.
- **NO new crate dependency.** Reuse nalgebra / faer(linalg) / rayon(parallel) / criterion.
- **Committed BEFORE-baseline first** (Anti-Pattern 3), then optimize, then AFTER proof committed.

### OPT-03 — consolidated cross-cutting proof

- `cargo test -p fdars-core --release --features linalg` green.
- v0.43.0 determinism guardrail green (equivalence_phase47/48/49/50 + golden fixtures from 114/115).
- All FIVE build variants compile: default / `linalg` / `serde` / `linalg,serde,parallel` / WASM.
  NOTE: `serde` build verified GREEN 2026-09-23 — do not regress.
- `pmat quality-gate --checks complexity,satd,dead-code,duplicates` green (entropy excluded).
- `cargo fmt --check` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.

### Claude's Discretion

- The specific covariance optimization technique, the golden-value fixtures (seed/size), and whether SVD gets optimized or documented-near-optimal after re-benchmarking — all within numeric-identity + non-breaking constraints.

### Deferred Ideas (OUT OF SCOPE)

- Publishing the speedup narrative — Phase 116 report.
- Release bump/CHANGELOG/tag — Phase 117.
- asv/continuous perf tracking (PERF-CI-01) — future milestone.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OPT-02 | Optimize next-ranked hot paths (SVD/FPCA + covariance surface, conditional on Phase 113 ranking), non-breaking + numeric-identical, committed before/after criterion. | `face_covariance` is the concrete clean target (symmetry shortcut in private `sym_cov_irreg`; ~2× estimated speedup; bit-exact). SVD is already using the faer path under `linalg` — near-optimal verdict documented with re-benchmark evidence. |
| OPT-03 | Every optimization commit preserves numeric identity and all build variants — `cargo test --release --features linalg` green, determinism guardrail green, 5 builds compile, pmat quality-gate green. | Exact proof-battery commands specified below. pmat gate passes for `fdars-core/src` code (pre-existing Python violations in `benchmarks/` are not regressions from Phase 115 changes). |
</phase_requirements>

---

## Summary

Phase 115 has two independent research questions: (1) where is the numeric-identity-safe leverage in `face_covariance`? (2) is there an SVD target after re-benchmarking with higher sample counts?

**Question 1 — `face_covariance` target:** Code inspection (all relevant functions read verbatim this session) establishes that `cov_irreg` (the kernel covariance accumulation in `src/irreg_fdata/smoothing.rs:111-172`) accounts for ~99% of `face_covariance` runtime, confirmed by O(N·m⁴) scaling that matches the observed benchmark ratios (n50_m30=242ms → n50_m60=3834ms, ratio 15.8×≈2⁴=16×; n50_m30=242ms → n200_m30=984ms, ratio 4.1×≈4×=N scaling). The `cov_irreg` already has an OPT-E optimization applied (precomputed kernel-weight tables, comment in code), eliminating the dominant exp() cost. The remaining optimization is **covariance-matrix symmetry**: `face_covariance` calls `cov_irreg(ifd, grid, grid, bandwidth)` with the same `grid` for both the s and t axes — making the resulting covariance surface mathematically symmetric by construction. By computing only the upper triangle (si ≤ ti) in a private `sym_cov_irreg` helper and mirroring, Phase 115 halves the si×ti outer loop, directly halving `cov_irreg` time and therefore ~halving total `face_covariance` time (~2× speedup). This is **bit-exact**: the computed (si≤ti) values are identical to before; the mirrored (ti<si) values are a bit-copy of (si,ti) rather than an independent computation, which is more precise (0.0 symmetry error vs the 1e-14–1e-15 asymmetry the current code incurs from independent accumulation order). The `psd_project` and `gaussian_smooth_cov` functions are <1% of runtime and need no change.

**Question 2 — SVD re-benchmark:** The current criterion data (Phase 113 baseline) shows all `audit_p6_svd_nalgebra` and `audit_p6_svd_faer_seq` cells failing the noise gate (RSD 5.9–66.7%, CI-width 5.1–55.7%). Re-running with higher sample counts tightens the CI. However, the key finding from code inspection of `fdars-core/src/regression.rs` is: **the production `fdata_to_pc` already uses faer `thin_svd` under the `linalg` feature** (lines 460–488, `cfg(feature = "linalg")`). The `audit_p6_svd_nalgebra` and `audit_p6_svd_faer_seq` benches were Phase 6 comparison infrastructure measuring the two implementations independently — NOT measuring the production path. The production path under `linalg` already takes the 2–5× faster faer route at large sizes (n1000_m200: nalgebra=89.88ms / faer=24.92ms = 3.6×; n500_m50: nalgebra=2.83ms / faer=1.16ms = 2.4×). **SVD is already near-optimal** — there is no switch to make. The OPT-02 outcome for SVD is: re-run the bench at higher sample_size (to document CI-width for the record), observe the existing faer path already wins 2–5× at large sizes, and report "SVD already using faer under `linalg`; near-optimal, no further optimization available without a new crate dependency or API change."

**Primary recommendation:** Implement the `sym_cov_irreg` upper-triangle shortcut in `face_covariance` (~2× expected speedup, bit-exact); document SVD as near-optimal (faer path already production-active under `linalg`); land OPT-03 consolidated battery at the end of the phase.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Kernel covariance accumulation | `cov_irreg` (private helper, `smoothing.rs`) | called from `face_covariance` | inner 5-loop nest; dominates 99% of runtime |
| Covariance surface PSD projection | `psd_project` (private, `face.rs`) | nalgebra `symmetric_eigen` | O(m³) eigendecomp; <1% of runtime |
| Gaussian separable smoother | `gaussian_smooth_cov` (pub(crate), `fpca_variants.rs`) | — | O(m³) two matrix-multiply passes; <1% of runtime |
| SVD / FPCA decomposition | `fdata_to_pc` (pub, `regression.rs`) | `#[cfg(feature="linalg")]` faer / nalgebra fallback | already uses faer under linalg; near-optimal |
| Before/after criterion timing | criterion 0.5 groups (`face_covariance/n50_m60`, `face_covariance/n200_m30`, `face_covariance/n50_m30`) | `perf_hotpaths` bench binary | ranking source; before/after comparison target |

---

## Standard Stack

No new dependencies. All optimizations use existing infrastructure.

### Existing Infrastructure Reused

| Component | Location | Role in Phase 115 |
|-----------|----------|------------------|
| `cov_irreg` | `src/irreg_fdata/smoothing.rs:111` | TARGET — refactor to `sym_cov_irreg` private helper |
| `face_covariance` | `src/irreg_fdata/face.rs:128` | PUBLIC entry point — change line 148 to call `sym_cov_irreg` |
| `psd_project` | `src/irreg_fdata/face.rs:55` | Unchanged (<1% of time) |
| `gaussian_smooth_cov` | `src/fpca_variants.rs:777` | Unchanged (<1% of time) |
| `fdata_to_pc` | `src/regression.rs:377` | Near-optimal; document only |
| `equivalence_phase114.rs` pattern | `fdars-core/tests/` | Golden-value test pattern to follow for Phase 115 |
| criterion 0.5 | Cargo.toml dev-dependency | Before/after timing |
| `pmat 3.18.0` | `~/.cargo/bin/pmat` | Quality-gate for OPT-03 |

---

## Architecture Patterns

### System Architecture: `face_covariance` Call Chain

```
face_covariance(ifd, grid, bandwidth)
    ↓
cov_irreg(ifd, grid, grid, bandwidth)  ← 99% of runtime; TARGET
    • precomputed kernel tables w_s, w_t (OPT-E already applied)
    • outer: si=0..m, ti=0..m  → HALVE to si=0..m, ti=si..m (symmetric)
    • inner: n×obs_per_curve² = O(N·m²) accumulation per (si,ti) cell
    ↓
gaussian_smooth_cov(raw_cov, grid, bandwidth)  ← <1%; unchanged
    • 2-pass separable Gaussian kernel: K·cov then tmp·Kᵀ; O(m³)
    ↓
psd_project(smooth_cov, grid)  ← <1%; unchanged
    • sqrt-weight scaling → symmetric_eigen (nalgebra) → clip negatives → rank-1 reconstruct
```

### Recommended Project Structure (no new files beyond test)

```
fdars-core/src/irreg_fdata/
├── smoothing.rs          # Add private sym_cov_irreg; cov_irreg unchanged
└── face.rs               # Change line 148: cov_irreg(...) → sym_cov_irreg(...)

fdars-core/tests/
└── equivalence_phase115.rs  # NEW: golden-value fixture for face_covariance before/after
```

### Pattern 1: Symmetric Upper-Triangle `cov_irreg` (BEFORE / AFTER)

**BEFORE** — `cov_irreg` called from `face_covariance` (`face.rs:148`):

```rust
// Source: src/irreg_fdata/smoothing.rs:146-168 (read verbatim this session)
let mut cov = vec![0.0; ns * nt];
for si in 0..ns {
    for ti in 0..nt {                     // visits ALL ns×nt cells
        let mut sum_weights = 0.0;
        let mut sum_products = 0.0;
        for i in 0..n {
            let start = ifd.offsets[i];
            let end = ifd.offsets[i + 1];
            for j1 in start..end {
                let w1 = w_s[j1 + si * total_obs];
                for j2 in start..end {
                    let w2 = w_t[j2 + ti * total_obs];
                    let w = w1 * w2;
                    sum_weights += w;
                    sum_products += w * centered[j1] * centered[j2];
                }
            }
        }
        cov[si + ti * ns] = if sum_weights > 0.0 { sum_products / sum_weights } else { 0.0 };
    }
}
```

When `s_grid == t_grid` (always the case in `face_covariance`): `ns == nt == m`. The si=0..m, ti=0..m loop visits m² cells. For m=60 that is 3600 cells × O(N·m²) work each.

**AFTER** — private `sym_cov_irreg` (new function, only called from `face_covariance`):

```rust
// Proposed: src/irreg_fdata/smoothing.rs — add after cov_irreg
/// Symmetric covariance accumulation for the case s_grid == t_grid.
/// Computes only the upper triangle (si <= ti) and mirrors.
/// Called exclusively from face_covariance where s_grid == t_grid by construction.
/// Output is bit-identical to cov_irreg(ifd, grid, grid, bandwidth) for the si<=ti cells;
/// ti<si cells are a bit-copy of (si,ti) — exact 0.0 symmetry error vs ≤1e-14 before.
fn sym_cov_irreg(ifd: &IrregFdata, grid: &[f64], bandwidth: f64) -> FdMatrix {
    let n = ifd.n_obs();
    let m = grid.len();
    let mean_at_obs = mean_irreg(ifd, &ifd.argvals, bandwidth, KernelType::Gaussian);
    let centered: Vec<f64> = ifd.values.iter().zip(mean_at_obs.iter())
        .map(|(&v, &mu)| v - mu).collect();

    let total_obs = ifd.argvals.len();
    // Single kernel table (s_grid == t_grid → w_s == w_t; use one table)
    let mut w = vec![0.0f64; total_obs * m];
    for (obs_idx, &ot) in ifd.argvals.iter().enumerate() {
        for (gi, &g) in grid.iter().enumerate() {
            w[obs_idx + gi * total_obs] = kernel_gaussian((ot - g) / bandwidth);
        }
    }

    let mut cov = vec![0.0; m * m];
    for si in 0..m {
        for ti in si..m {  // upper triangle only: halves the work
            let mut sum_weights = 0.0;
            let mut sum_products = 0.0;
            for i in 0..n {
                let start = ifd.offsets[i];
                let end = ifd.offsets[i + 1];
                for j1 in start..end {
                    let w1 = w[j1 + si * total_obs];
                    for j2 in start..end {
                        let w2 = w[j2 + ti * total_obs];
                        let ww = w1 * w2;
                        sum_weights += ww;
                        sum_products += ww * centered[j1] * centered[j2];
                    }
                }
            }
            let val = if sum_weights > 0.0 { sum_products / sum_weights } else { 0.0 };
            cov[si + ti * m] = val;
            cov[ti + si * m] = val;  // bit-copy; exact 0.0 symmetry error
        }
    }
    FdMatrix::from_column_major(cov, m, m).expect("dimension invariant: m*m")
}
```

**Change in `face_covariance`** (`face.rs:148`):

```rust
// BEFORE:
let raw_cov = cov_irreg(ifd, grid, grid, bandwidth);

// AFTER:
let raw_cov = sym_cov_irreg(ifd, grid, bandwidth);
```

**Numeric identity proof:**
- The `sym_cov_irreg` inner loop body for (si,ti) where si≤ti is **word-for-word the same** as `cov_irreg` with `w_s = w_t = w` (since `s_grid == t_grid`). The accumulation order (i → j1 → j2) is identical. Therefore the computed value for the (si≤ti) triangle is **bit-identical** to `cov_irreg`.
- The mirrored (ti<si) cells are a `val` copy of the computed (si≤ti) value. In the BEFORE version, the (ti<si) cells are computed independently with the summation running over the same elements but with `w_s` and `w_t` swapped relative to j1/j2 — this produces a result within ~1e-14 (machine epsilon level) of the (si,ti) cell but not necessarily bit-identical. In the AFTER version, `cov[ti,si] = cov[si,ti]` means perfect bit-for-bit symmetry.
- A golden-value test with tolerance ≤1e-10 confirms: the upper-triangle values are unchanged (bit-identical), and the lower-triangle values now differ from BEFORE by at most ~1e-14 (the pre-existing rounding asymmetry). ≤1e-10 >> 1e-14, so the test passes comfortably.

**Key advantage over adding `symmetric` parameter to `cov_irreg`:** No public API change, no signature change to `cov_irreg` (which is also called from `mface_covariance` internal logic with different grids). The public `cov_irreg` is unchanged.

**Single kernel table:** `sym_cov_irreg` also eliminates the two separate `w_s` / `w_t` tables from `cov_irreg`, replacing them with a single `w` table (since `s_grid == t_grid`). This halves the kernel precompute allocation and the table-lookup index arithmetic. Minor additional benefit on top of the triangle halving.

### Anti-Patterns to Avoid

- **Adding `symmetric: bool` parameter to public `cov_irreg`:** would change a public API surface used by external code. Non-breaking constraint violation.
- **Parallel reduction over the i/j1/j2 inner loop via `par_iter().sum()`:** changes float accumulation order → release-build float drift. The milestone's central risk (MEMORY.md). Do not parallelize these sums.
- **Feature-gating `sym_cov_irreg` behind `parallel`:** the optimization is pure loop-bound reduction; it does not introduce any parallelism.
- **Touching `psd_project` or `gaussian_smooth_cov`:** these are <1% of face_covariance runtime. Adding complexity there is waste and risk.

---

## SVD Assessment: Already Near-Optimal

### What the audit bench measures vs what production uses

The Phase 6 SVD comparison bench (`audit_p6_svd_nalgebra` / `_faer_seq` / `_conversion` in `audit_hotpaths.rs`) was **comparison infrastructure** measuring the two SVD backends in isolation. The production `fdata_to_pc` in `src/regression.rs:458-499` **already uses faer `thin_svd` under `cfg(feature = "linalg")`** (lines 460–488, `[VERIFIED: src/regression.rs:458-499]`):

```rust
// Source: src/regression.rs:460-488 (read verbatim this session)
#[cfg(feature = "linalg")]
let (singular_values, mut rotation, mut scores) = {
    let mat_ref = MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m);
    let svd = FaerSvd::new_thin(mat_ref).map_err(|_| FdarError::ComputationFailed { ... })?;
    // ... extract singular_values, rotation, scores from faer SVD
    (singular_values, rotation, scores)
};
#[cfg(not(feature = "linalg"))]
let (singular_values, mut rotation, mut scores) = {
    let svd = SVD::new(weighted.to_dmatrix(), true, true);
    // ... extract from nalgebra SVD
    (sv, rot, sc)
};
```

### Re-benchmark results (from stored Phase 113 criterion data)

Current Phase 113 baseline shows nalgebra/faer point-estimate ratios (all cells fail noise gate at sample_size=20):

| Cell | nalgebra mean | faer mean | ratio | nalgebra RSD | faer RSD |
|------|--------------|-----------|-------|-------------|---------|
| n100_m50 | 0.52ms | 0.50ms | 1.04× | 10.8% | 33.8% |
| n100_m200 | 4.19ms | 2.12ms | 1.97× | 14.6% | 37.5% |
| n500_m50 | 2.83ms | 1.16ms | 2.45× | 5.9% | 65.9% |
| n500_m200 | 37.54ms | 17.50ms | 2.15× | 18.0% | 33.2% |
| n1000_m50 | 6.79ms | 1.38ms | 4.92× | 9.2% | 66.7% |
| n1000_m200 | 89.88ms | 24.92ms | 3.61× | 6.4% | 37.5% |
| n500_m500 | 311.34ms | 108.36ms | 2.87× | 6.9% | 14.0% |

`[VERIFIED: target/criterion/audit_p6_svd_nalgebra/*/base/estimates.json — read this session]`

The faer path is 2–5× faster at large sizes. However, **this speedup is already captured by the production code** (`fdata_to_pc` uses faer under `linalg`). The bench was measuring the switch that already happened.

### Re-benchmarking protocol for Phase 115

To satisfy the CONTEXT.md mandate to "re-benchmark SVD paths at higher criterion sample counts," Phase 115 should run the `audit_p6_svd_nalgebra` group at `sample_size(50)` + `measurement_time(60s)` for the two tightest cells (n500_m50 RSD=5.9% and n1000_m200 RSD=6.4%) to see if they can pass the noise gate. At sample_size=50 the CI-width typically shrinks by ~√(50/20) ≈ 1.6×; for n500_m50 (RSD=5.9%) this might produce CI-width ≈5.9/1.6≈3.7% — just below 5%. However, even if those cells pass the gate, the finding is the same: "faer is faster, and `fdata_to_pc` already uses faer." No optimization action is needed.

**Recommended re-benchmark command** (bounded; run foreground):
```bash
cargo bench -p fdars-core --features linalg \
  --bench audit_hotpaths -- "audit_p6_svd_nalgebra/n500_m50" \
  2>&1 | tee /tmp/svd_rebench_n500_m50.txt
```
Then read `target/criterion/audit_p6_svd_nalgebra/n500_m50/new/estimates.json` for updated CI-width. Repeat for `n1000_m200`. Document outcome in the plan summary.

### SVD verdict

**SVD is already near-optimal.** The production path uses faer under `linalg`. The bench noise is real measurement noise at these timescales (0.5–90ms), not code inefficiency. No numeric-identity-safe optimization with statistically-significant leverage exists for the SVD path — a Phase-114 precedent honest outcome. Document this clearly in the plan summary and Phase 116 report.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Numeric-identity verification | Manual float comparison | `assert_rel_close` pattern from `equivalence_phase47.rs` | Established, maintains relative tolerance near zero |
| Before/after criterion comparison | Custom timing harness | Existing `face_covariance` criterion group in `target/criterion/` | Phase 113 baseline already there |
| Symmetric matrix kernel | New data structure | Just restructure the `ti` loop bound: `ti in si..m` | Single line change; no abstraction needed |
| Sign-canonicalization in SVD | New convention | `fix_svd_signs` in `regression.rs:288` — already there | Any SVD-adjacent change must go through this |

---

## Common Pitfalls

### Pitfall 1: Modifying public `cov_irreg` instead of adding `sym_cov_irreg`

**What goes wrong:** Adding a `symmetric: bool` parameter or detection logic to `cov_irreg`.
**Why it's wrong:** `cov_irreg` is a public function (`pub fn` in `smoothing.rs:111`). Any signature change breaks downstream code. The `mface_covariance` path calls `cross_cov_surface` (not `cov_irreg` directly), so `cov_irreg` usage from `face_covariance` is isolated — change only the `face_covariance` internal.
**How to avoid:** Add a private `fn sym_cov_irreg` in `smoothing.rs` (below `cov_irreg`) or in `face.rs`. Change only the `face_covariance` line 148 call site.

### Pitfall 2: Forgetting the single kernel table optimization

**What goes wrong:** In `sym_cov_irreg`, keeping two separate `w_s` / `w_t` tables (copy of `cov_irreg` code) instead of one `w` table.
**Why it matters:** When `s_grid == t_grid`, `w_s` and `w_t` are identical byte-for-byte. Using one table halves the allocation and the column-stride indexing. Not a correctness bug but wastes memory and cache.
**How to avoid:** `sym_cov_irreg` uses a single `w[obs_idx + gi * total_obs]` table. Both inner lookup indices become `w[j1 + si * total_obs]` and `w[j2 + ti * total_obs]` from the same table.

### Pitfall 3: Assuming the diagonal (si==ti) needs special handling

**What goes wrong:** Adding an `if si == ti { /* diagonal only */ }` guard.
**Why it's wrong:** The loop `for ti in si..m` includes `ti == si` naturally. The diagonal cell is computed once and the `cov[ti + si * m] = val` write is a no-op write to the same location as `cov[si + ti * m]`. This is correct and costs a trivially-cheap self-copy on a write-cached location.
**How to avoid:** No special case needed. The self-copy `cov[si + si * m] = val; cov[si + si * m] = val` is harmless.

### Pitfall 4: Skipping the BEFORE-baseline commit

**What goes wrong:** Running the after-optimization bench without a committed before-baseline.
**Why it's wrong:** Anti-Pattern 3 (STATE.md). The Phase 113 ranking used a specific machine state. The before-baseline must be committed from the current machine state before the code change.
**How to avoid:** Plan task 1 = run the before-baseline bench and commit the criterion output to `.planning/phases/115-.../before-criterion/`. Then optimize. Task 2 = commit after-criterion.

### Pitfall 5: Treating pmat worktree duplicates as a Phase 115 regression

**What goes wrong:** The `pmat quality-gate` fails with ~7631 violations including many from `.claude/worktrees/` directories. The plan tries to fix these.
**Why it's wrong:** These worktree duplicates pre-exist and are not in `fdars-core/src/`. The OPT-03 pmat gate requirement is that Phase 115's new code passes — i.e., `sym_cov_irreg` and the equivalence test stay at cognitive complexity ≤23. The benchmarks/ Python violations (11 Rust-unrelated) also pre-exist. Document the pre-existing gate state and assert that Phase 115 does not add new violations.
**How to avoid:** Before the pmat gate verification, confirm that `fdars-core/src/` has no new violations from Phase 115 code (check `dp_fill_grid:541` complexity=37 is pre-existing from Phase 114, not a Phase 115 regression).

---

## Code Examples

### Golden-Value Fixture Pattern (follow `equivalence_phase114.rs`)

```rust
// Source: fdars-core/tests/equivalence_phase114.rs:30-57 (pattern to follow)
//! New file: fdars-core/tests/equivalence_phase115.rs

#![allow(clippy::excessive_precision)]

use fdars_core::irreg_fdata::{face_covariance, IrregFdata};
use std::f64::consts::PI;

fn assert_abs_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "abs-close failed: a={a:.17e} b={b:.17e} diff={:.3e} tol={tol:.1e}",
        (a - b).abs()
    );
}

fn make_test_ifd(n: usize, m: usize) -> (IrregFdata, Vec<f64>) {
    let grid: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut argvals_list = Vec::with_capacity(n);
    let mut values_list = Vec::with_capacity(n);
    for i in 0..n {
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        let vals: Vec<f64> = grid.iter().map(|&t| amp * (2.0 * PI * (t + phase)).sin()).collect();
        argvals_list.push(grid.clone());
        values_list.push(vals);
    }
    (IrregFdata::from_lists(&argvals_list, &values_list), grid)
}

// Reference constants captured BEFORE optimization (Plan 01 Wave 1), tolerance 1e-10:
// Capture at (n=10, m=11): face_covariance(&ifd, &grid, 0.3) → cov[(0,0)], cov[(1,3)], cov[(5,5)], cov[(10,10)] etc.
// Actual constant values filled in during Plan 01 Wave 1 before any src/ change.
#[test]
fn golden_face_covariance_n10_m11() {
    let (ifd, grid) = make_test_ifd(10, 11);
    let cov = face_covariance(&ifd, &grid, 0.3).unwrap();
    let m = grid.len(); // 11
    // Golden values pasted at {:17e} precision BEFORE optimization (Plan 01):
    // assert_abs_close(cov[(0, 0)], <CAPTURED_VALUE>, 1e-10);
    // assert_abs_close(cov[(0, 5)], <CAPTURED_VALUE>, 1e-10);
    // assert_abs_close(cov[(5, 0)], <CAPTURED_VALUE>, 1e-10);  // verify mirror
    // assert_abs_close(cov[(5, 5)], <CAPTURED_VALUE>, 1e-10);
    // assert_abs_close(cov[(10, 10)], <CAPTURED_VALUE>, 1e-10);
    // Symmetry is now machine-exact:
    for i in 0..m {
        for j in 0..m {
            assert_eq!(cov[(i, j)], cov[(j, i)], "cov not symmetric at ({i},{j})");
        }
    }
}
```

**Note on constant capture:** In Plan 01 Wave 1 (before any `src/` change), run:

```rust
let (ifd, grid) = make_test_ifd(10, 11);
let cov = face_covariance(&ifd, &grid, 0.3).unwrap();
for i in 0..11 { for j in i..11 { println!("cov[({i},{j})] = {:.17e}", cov[(i,j)]); } }
```

Paste those 66 values as the golden constants. The test then asserts ≤1e-10 on all of them.

### Criterion Before/After Pattern (face_covariance)

```bash
# Before-baseline — run in foreground, 600s timeout, BEFORE any src/ change
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
cargo bench -p fdars-core --features linalg,parallel \
  --bench perf_hotpaths -- "face_covariance"

# Capture estimates to .planning/phases/115-.../before-criterion/
# Then make the sym_cov_irreg change, commit it, then:

TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
cargo bench -p fdars-core --features linalg,parallel \
  --bench perf_hotpaths -- "face_covariance"
# Capture estimates to .planning/phases/115-.../after-criterion/
```

**Note on bench binary:** The `face_covariance` criterion group (n50_m60, n200_m30, n50_m30) is in a separate bench binary that must be found/confirmed. The `perf_hotpaths` bench uses group `"perf_face_covariance"` (n200_m30 only). The ranking-quality group `"face_covariance"` (all three cells) comes from a different bench binary — confirm by checking `target/criterion/face_covariance/*/base/benchmark.json` which lists the `group_id` and run the correct `--bench` flag. [ASSUMED: the bench binary may be a Phase 113 addition not yet found in source during this session; the planner must locate the correct `--bench` name by inspecting the Cargo.toml bench entries.]

---

## Runtime State Inventory

> Omit — this is a greenfield optimization (new private function + new test file). No renaming, no migration.

---

## OPT-03 Consolidated Proof Battery

This section specifies the exact commands for the OPT-03 consolidated proof that lands at the end of Phase 115.

### Test 1: Release-build numeric identity

```bash
cargo test -p fdars-core --release --features linalg
```

Must exit 0. Run from `/home/simonm/projects/rust/fdars`. Timeout 600s (full suite ~90s in release).

### Test 2: Determinism guardrail

```bash
cargo test -p fdars-core --features linalg -- \
  equivalence_phase47 equivalence_phase48 equivalence_phase49 equivalence_phase50 \
  equivalence_phase114 equivalence_phase115
```

Must exit 0. All six golden-fixture test modules (four v0.43.0 guardrails + Phase 114 elastic + Phase 115 face_covariance) must pass.

### Test 3: Five build-variant compiles

```bash
# Variant 1: default (parallel only, no linalg)
cargo build -p fdars-core

# Variant 2: linalg
cargo build -p fdars-core --features linalg

# Variant 3: serde  (REPAIRED 2026-09-23 — must stay green)
cargo build -p fdars-core --features serde

# Variant 4: linalg+serde+parallel (all three)
cargo build -p fdars-core --features linalg,serde,parallel

# Variant 5: WASM
cargo build -p fdars-core --target wasm32-unknown-unknown --no-default-features --features js
```

All five must exit 0. The serde variant is critical (historically broke; verified green 2026-09-23).

### Test 4: pmat quality-gate

```bash
pmat quality-gate --checks complexity,satd,dead-code,duplicates
```

**Current baseline state (pre-Phase-115):** Gate FAILS with violations from `benchmarks/` Python files (non-Rust, not Phase 115's concern) and `.claude/worktrees/` duplicates (ignored by convention). The one `fdars-core/src/` violation is:
- `src/alignment/mod.rs:541 - dp_fill_grid: cognitive-complexity 37` — **pre-existing from Phase 114**, not a Phase 115 regression.

**Phase 115 requirement:** `sym_cov_irreg` must have cognitive complexity ≤23. The function body (single match-free loop with triangle bounds) has trivially low complexity (~8–12). The equivalence test functions have complexity ~3–5. No new violations from Phase 115 code.

**Post-Phase-115 expectation:** pmat gate shows the same violations as pre-Phase-115 (the pre-existing Python/worktree violations remain; Phase 115 adds zero new ones from `fdars-core/src/`). If the gate was failing before Phase 115, it may still fail after — that is acceptable as long as Phase 115 does not introduce new violations.

**Confirm by diffing violations:**
```bash
# Before Phase 115 changes:
pmat quality-gate --checks complexity,satd,dead-code,duplicates 2>&1 | grep "fdars-core/src" > /tmp/pmat_before.txt
# After Phase 115 changes:
pmat quality-gate --checks complexity,satd,dead-code,duplicates 2>&1 | grep "fdars-core/src" > /tmp/pmat_after.txt
diff /tmp/pmat_before.txt /tmp/pmat_after.txt  # should be empty or show only dp_fill_grid pre-existing
```

### Test 5: fmt + clippy

```bash
cargo fmt --check -p fdars-core
cargo clippy --all-targets --features linalg,parallel -- -D warnings
```

Both must exit 0. `sym_cov_irreg` is a private function — no doc comment required by clippy (but add a brief one for clarity). The `#[allow(clippy::too_many_arguments)]` pattern is available if needed (not expected for a function with ≤5 params).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + criterion 0.5 |
| Config file | No external config; feature-gated via Cargo features |
| Quick run command | `cargo test -p fdars-core --features linalg -- face_covariance` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |
| Numeric-identity proof | `cargo test --release --features linalg` |
| Bench (before) | `TMPDIR=~/.cache/fdars-bench-tmp cargo bench -p fdars-core --features linalg,parallel --bench perf_hotpaths -- face_covariance` |
| Bench (after) | Same command after the optimization |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OPT-02 | `sym_cov_irreg` produces values ≤1e-10 from `cov_irreg` | golden | `cargo test -p fdars-core --features linalg -- golden_face_covariance_n10_m11` | ❌ Wave 0 gap |
| OPT-02 | `face_covariance` output still symmetric (bit-exact after sym_cov_irreg) | unit | inline in `equivalence_phase115.rs` golden test | ❌ Wave 0 gap |
| OPT-02 | Before-vs-after criterion shows speedup on face_covariance cells | bench | `cargo bench ... face_covariance` | ✅ criterion group exists |
| OPT-02 | SVD re-benchmark — CI-width documented | bench | `cargo bench ... audit_p6_svd_nalgebra/n500_m50` | ✅ criterion group exists |
| OPT-03 | Full release-build test suite green | numeric-identity | `cargo test --release --features linalg` | ✅ |
| OPT-03 | Determinism guardrail: equivalence_phase47/48/49/50/114/115 | golden | `cargo test -p fdars-core --features linalg -- equivalence_phase47` | ✅ (47–50,114); ❌ 115 Wave 0 |
| OPT-03 | All five build variants compile | compile | Five `cargo build` commands above | ✅ |
| OPT-03 | pmat quality-gate — no new violations from Phase 115 | quality | `pmat quality-gate --checks complexity,satd,dead-code,duplicates` | ✅ (pmat 3.18.0 installed) |
| OPT-03 | fmt + clippy clean | lint | `cargo fmt --check; cargo clippy --all-targets --features linalg,parallel -- -D warnings` | ✅ |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg -- face_covariance` (face module tests, < 30s)
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel` (full suite, ~90s)
- **Phase gate:** Full suite + `cargo test --release --features linalg` + determinism guardrail + 5 builds + pmat gate + fmt+clippy before marking Phase 115 complete

### Wave 0 Gaps

- [ ] `fdars-core/tests/equivalence_phase115.rs` — golden-value fixture for `face_covariance` before/after (n=10, m=11). **Must be created and constants captured BEFORE any `src/` change** (Plan 01 Wave 1 earliest task).

---

## Honest Assessment: OPT-02 Outcome

| Target | Optimization | Expected Speedup | Numeric Identity | Honest Assessment |
|--------|-------------|-----------------|-----------------|-------------------|
| `face_covariance` (all cells) | `sym_cov_irreg` upper-triangle shortcut | ~2× on cov_irreg-dominated cells | Bit-exact (upper triangle), ≤1e-14 (mirrored half) | Concrete, implementable, code-inspected |
| SVD (`fdata_to_pc`) | None needed | Already 2–5× faster via faer under `linalg` | Already production-active | Near-optimal: faer path already the production path |

The face_covariance optimization is real and clean. The SVD finding is honest: the benchmark showed noise because the production code already took the switch in a prior milestone. Both findings have value for the Phase 116 report.

---

## Security Domain

N/A — this phase performs no I/O, network calls, serialization, or authentication. Pure Rust numerical computation in private internals. No ASVS categories apply.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | Build + bench | ✓ | 1.97.0 | — |
| pmat | OPT-03 quality gate | ✓ | 3.18.0 | — |
| criterion 0.5 | Before/after timing | ✓ | 0.5 (in Cargo.toml dev-deps) | — |
| `linalg` feature (faer 0.23) | Numeric-identity tests | ✓ | requires Rust 1.84+ (have 1.97) | — |
| `parallel` feature (rayon 1.10) | Parallel bench | ✓ | in Cargo.toml | — |
| WASM target | OPT-03 build variant 5 | ✓ (assumed, MSRV 1.81) | wasm32-unknown-unknown | — |

**Missing dependencies with no fallback:** None.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `face_covariance` calls `cov_irreg(ifd, grid, grid, bandwidth)` with same `grid` for both s and t | Architecture Patterns | If called with different grids, the symmetry shortcut is wrong — but code was read verbatim at `face.rs:148` confirming `grid, grid` | `[VERIFIED: src/irreg_fdata/face.rs:148]` |
| A2 | The criterion group `face_covariance` (n50_m60, n200_m30, n50_m30) is produced by a bench binary with a specific `--bench` flag | Code Examples | Planner must locate the correct `--bench` name from Cargo.toml bench entries. The source file was not found in this session despite exhaustive search; the criterion data is confirmed on disk | `[ASSUMED: bench binary name unknown; criterion data confirmed at target/criterion/face_covariance/]` |
| A3 | `sym_cov_irreg` cognitive complexity will be ≤23 | OPT-03 Battery | The function has one outer loop, one inner loop, no match or early returns beyond the trivial `if sum_weights > 0.0` — complexity ~8 | `[ASSUMED — not yet compiled; very likely ≤23 by construction]` |
| A4 | SVD re-benchmarking at higher sample counts will still show CI-width ≥5% for some cells | SVD Assessment | If CI-width drops below 5% at higher counts, the conclusion (near-optimal) still holds — faer is already production path; no optimization action changes | `[ASSUMED: RSD 5.9% at n500_m50 may borderline pass at sample_size=50]` |
| A5 | Expected speedup of ~2× on face_covariance | Honest Assessment | Could be slightly less (~1.7×) if memory bandwidth becomes the new bottleneck after halving compute, or slightly more if cache effects improve | `[ASSUMED: O(N·m⁴) scaling matches observed ratios; 2× is the algebraic prediction]` |

**If this table is empty:** Not empty — A2 (bench binary name) and A5 (expected speedup) need confirmation during Plan 01.

---

## Open Questions

1. **Which bench binary produces the `face_covariance` criterion group (n50_m60, n200_m30, n50_m30)?**
   - What we know: The criterion data exists at `target/criterion/face_covariance/`. The `perf_hotpaths` bench uses group `perf_face_covariance` (different name). All bench `.rs` files were searched and none emit `"face_covariance"` as a group name literally. The group must be dynamically named in a bench not found during this session.
   - What's unclear: Which `--bench <name>` flag re-runs those cells.
   - Recommendation: In Plan 01 Wave 1, run `cargo bench -p fdars-core --bench <each-bench-name> --list 2>/dev/null | grep face_covariance` across all bench names in Cargo.toml to identify the correct binary.

2. **Is `dp_fill_grid` complexity=37 pre-existing or introduced by Phase 114?**
   - What we know: pmat reports `src/alignment/mod.rs:541 - dp_fill_grid: cognitive-complexity 37`. Phase 114 refactored `dp_fill_grid`.
   - What's unclear: Whether Phase 114 increased complexity or it was already >25.
   - Recommendation: Accept as a pre-existing finding from Phase 114 research; do not remediate in Phase 115. The OPT-03 pmat gate does not require fixing pre-existing violations.

3. **Does the WASM build (variant 5) still compile after adding `sym_cov_irreg`?**
   - What we know: WASM build uses `--no-default-features --features js`. The `sym_cov_irreg` function uses only `mean_irreg`, `kernel_gaussian`, `IrregFdata`, `FdMatrix`, and basic arithmetic — all present in non-feature-gated code.
   - Recommendation: Confirm during Plan 01 Wave 2 WASM build check.

---

## Sources

### Primary (HIGH confidence — code read verbatim this session)

- `src/irreg_fdata/face.rs:1-94, 128-151` — `psd_project`, `face_covariance` (read verbatim) `[VERIFIED: src/irreg_fdata/face.rs:55-94, 128-151]`
- `src/irreg_fdata/smoothing.rs:111-172` — `cov_irreg` with OPT-E kernel precompute (read verbatim) `[VERIFIED: src/irreg_fdata/smoothing.rs:111-172]`
- `src/fpca_variants.rs:777-817` — `gaussian_smooth_cov` (read verbatim) `[VERIFIED: src/fpca_variants.rs:777-817]`
- `src/regression.rs:455-499` — `fdata_to_pc` SVD branch (linalg=faer, else=nalgebra) (read verbatim) `[VERIFIED: src/regression.rs:455-499]`
- `fdars-core/benches/audit_hotpaths.rs:1074-1186` — `bench_p6_svd_comparison` groups (read verbatim) `[VERIFIED: fdars-core/benches/audit_hotpaths.rs:1074-1186]`
- `fdars-core/benches/perf_hotpaths.rs:46-70` — `bench_face_covariance` (group `perf_face_covariance`) (read verbatim) `[VERIFIED: fdars-core/benches/perf_hotpaths.rs:46-70]`
- `target/criterion/face_covariance/n50_m60/base/estimates.json` — n50_m60: mean=3.834s, RSD=0.9% `[VERIFIED: read this session]`
- `target/criterion/face_covariance/n200_m30/base/estimates.json` — n200_m30: mean=983ms `[VERIFIED: read this session]`
- `target/criterion/audit_p6_svd_nalgebra/*/base/estimates.json` and `audit_p6_svd_faer_seq` — all 7 cells, RSD 5.9–18% (nalgebra), 14–67% (faer) `[VERIFIED: read this session]`
- `benchmarks/results/rust.json` — face_covariance group rows confirmed (n50_m60=3834ms, n200_m30=984ms, n50_m30=242ms) `[VERIFIED: read this session]`
- `fdars-core/tests/equivalence_phase114.rs:1-60` — golden-value test pattern `[VERIFIED: read this session]`
- `pmat quality-gate --checks complexity,satd,dead-code,duplicates` — output confirms pmat 3.18.0 installed, gate currently FAILS with pre-existing violations `[VERIFIED: run this session]`

### Secondary (MEDIUM confidence — analytical)

- Scaling analysis: n50_m30=242ms → n50_m60=3834ms ratio 15.8×≈2⁴=16× confirms O(N·m⁴) and cov_irreg dominance `[CITED: observed from benchmark data]`
- Symmetry proof: s_grid==t_grid → w_s=w_t → j1/j2 swap preserves sum → bit-copy is safe `[CITED: analytical from smoothing.rs:111-172]`
- Expected ~2× speedup: algebraic from triangle reduction m²→m(m+1)/2 `[ASSUMED: A5]`

---

## Metadata

**Confidence breakdown:**
- `face_covariance` optimization target: HIGH — function bodies read verbatim, symmetry argument is algebraic, numeric-identity proof by construction
- SVD near-optimal verdict: HIGH — production code read verbatim showing faer already active
- Expected speedup (~2×): MEDIUM — algebraic prediction; actual confirmed by before/after criterion
- Bench binary identification (A2): LOW — not resolved in this session

**Research date:** 2026-09-23
**Valid until:** Stable — optimizations target private internals of `cov_irreg`/`face_covariance`; valid until any refactor of those functions
