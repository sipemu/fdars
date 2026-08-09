# Phase 6: Conditional SVD Library Comparison — Research

**Researched:** 2026-08-08
**Domain:** Rust linear algebra — faer 0.23 SVD API vs nalgebra 0.33 SVD, criterion benchmark design for library comparison
**Confidence:** HIGH (faer API verified from vendored source; Phase 4 evidence verified from on-disk artifacts)

---

## Summary

Phase 4 already resolved the go/no-go decision: the comparison **IS warranted** (GO). SVD compute
accounts for ~99.8–99.9% of `fdata_to_pc_1d` wall-clock at every grid cell; the `to_dmatrix()`
copy is only 0.14–0.17% of wall-clock. Both SC1 conditions are met
[VERIFIED: .planning/phases/04-fpca-svd-allocation-audit/04-VERIFICATION.md:28-29].

This phase is therefore entirely about **how to execute the comparison**, not whether to. The work is:
a criterion bench function `bench_p6_svd_comparison` in the existing `audit_hotpaths.rs` measuring
both nalgebra and faer SVD at fdars' real FPCA workload sizes (N∈{100,500,1000}×M∈{50,200} and
optionally M=500 as a crossover probe), plus a faer adoption note for the backlog item, both
appended to `AUDIT-REPORT.md` as the Phase 6 section.

No code changes to `fdars-core/src`. The bench function is throwaway comparison infrastructure —
audit-only per the CLAUDE.md scope constraint.

**Primary recommendation:** Add `bench_p6_svd_comparison` to `audit_hotpaths.rs` measuring three
quantities per cell: (1) nalgebra SVD time, (2) faer thin_svd time, (3) FdMatrix→faer::MatRef
conversion time separately. Use `set_global_parallelism(Par::Seq)` before the faer call to match
nalgebra's always-sequential behavior, then `set_global_parallelism(Par::Rayon(0))` for a parallel
faer variant. Run with `--features linalg` (faer is already present as a dependency).

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-06 | A conditional nalgebra-vs-faer SVD comparison at fdars' real problem sizes, performed only if benchmarks show SVD to be a significant share of FPCA runtime (else recorded as "not warranted, with evidence") | Phase 4 GO verdict: SVD ~99.8–99.9% of wall-clock, copy ~0.14–0.17%. Comparison IS warranted. Research below covers faer API, bench design, numerical equivalence, adoption risk assessment, and artifact naming. |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SVD bench comparison | Bench/audit harness | — | The comparison lives in `audit_hotpaths.rs` (bench infra), not in `fdars-core/src`. No production tier is modified. |
| Conversion cost measurement | Bench/audit harness | — | FdMatrix→MatRef conversion is measured as an isolated sub-bench within the same bench function. |
| Numerical equivalence check | Test harness | — | A short standalone Rust function (or separate test in the bench binary) that asserts sorted singular values match within tolerance. Not a production code path. |
| Adoption note / backlog item | Report artifact | — | Written to `AUDIT-REPORT.md`, consumed by Phase 9. |

---

## Standard Stack

### Core (all already present — NO new dependencies to add)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faer | 0.23.2 (vendored) | SVD comparison target | Already a dep behind `linalg` feature [VERIFIED: fdars-core/Cargo.toml:43] |
| nalgebra | 0.33 | SVD baseline | Currently used in `fdata_to_pc_1d` [VERIFIED: fdars-core/src/regression.rs:298] |
| criterion | 0.5 | Wall-clock microbenchmarking | Already in dev-dependencies [VERIFIED: fdars-core/Cargo.toml:53] |

No new `Cargo.toml` additions are required. The `linalg` feature already pulls in faer 0.23.2.
The bench just needs to import faer types alongside nalgebra.

### Installation

```bash
# No installation needed — faer is already vendored under the `linalg` feature.
# Run all Phase 6 benches with:
export TMPDIR=/home/simonm/.cache/fdars-bench-tmp
cargo bench --features linalg -p fdars-core --bench audit_hotpaths -- audit_p6
```

### Version Verification

```
faer = "0.23.2"  [VERIFIED: Cargo.lock — name = "faer", version = "0.23.2"]
nalgebra = "0.33" [VERIFIED: fdars-core/Cargo.toml:38]
```

---

## Package Legitimacy Audit

No new packages are introduced by this phase. Both faer and nalgebra are already vendored
and verified in the project's existing lock file.

| Package | Registry | Verdict | Disposition |
|---------|----------|---------|-------------|
| faer 0.23.2 | crates.io | OK (already vendored) | Approved — existing dependency |
| nalgebra 0.33 | crates.io | OK (already vendored) | Approved — existing dependency |

---

## Architecture Patterns

### System Architecture Diagram

```
FdMatrix (column-major Vec<f64>)
        │
        ├── [nalgebra path] ──► weighted.to_dmatrix() ──► SVD::new(dmatrix, true, true) ──► (U, S, Vt)
        │                       DMatrix::from_column_slice                                    nalgebra types
        │
        └── [faer path] ──────► MatRef::from_column_major_slice(&weighted.data, n, m)
                                 ↑ zero-copy view — NO allocation                    │
                                 │                                                   ▼
                                 └──────────────────────────────► mat_ref.thin_svd() ──► Svd<f64>
                                                                                          .U(), .V(), .S()

Bench isolates three quantities:
  ┌─ b.iter(|| nalgebra_svd(black_box(weighted_dmatrix.clone())) )   ← DMatrix clone + SVD
  ├─ b.iter(|| faer_svd(black_box(mat_ref)) )                         ← zero-copy view + SVD
  └─ b.iter(|| MatRef::from_column_major_slice(&data, n, m) )         ← conversion cost alone
```

### faer SVD API (verified from vendored source)

**Key types** [VERIFIED: faer-0.23.2/src/linalg/solvers.rs:1117-1183]:

```rust
// Full SVD — U is m×m, V is n×n
pub fn Svd::new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Result<Self, SvdError>

// Thin SVD — U is m×min(m,n), V is n×min(m,n)  ← USE THIS for FPCA (matches nalgebra's true,true)
pub fn Svd::new_thin<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Result<Self, SvdError>

// Accessors
pub fn U(&self) -> MatRef<'_, T>         // left singular vectors
pub fn V(&self) -> MatRef<'_, T>         // right singular vectors
pub fn S(&self) -> DiagRef<'_, T>        // singular values (nonnegative, nonincreasing order)
```

Also callable as methods on `MatRef` / `Mat` [VERIFIED: faer-0.23.2/src/linalg/solvers.rs:254-268]:

```rust
mat_ref.thin_svd() -> Result<Svd<f64>, SvdError>   // preferred — thin, matches nalgebra thin
mat_ref.svd()      -> Result<Svd<f64>, SvdError>   // full
mat_ref.singular_values() -> Result<Vec<f64>, SvdError>  // values only, no U/V
```

**Constructing `MatRef` from fdars column-major FdMatrix** (zero-copy, no allocation)
[VERIFIED: faer-0.23.2/src/mat/matref.rs:113-117]:

```rust
// data slice must be column-major: element (i,j) at index i + j*nrows
// FdMatrix.data is already column-major — exact match, zero-copy
let mat_ref: MatRef<f64> = MatRef::from_column_major_slice(
    &weighted.data,   // FdMatrix.data field (Vec<f64>)
    n,                // nrows (usize, accepted as Rows: Shape)
    m,                // ncols (usize)
);
```

No allocation. This is a **view**, not a copy. The equivalent of `nalgebra::DMatrix::from_column_slice`
but without allocating — because `MatRef` borrows the existing buffer.

**Extracting singular values as a slice** [VERIFIED: faer-0.23.2/src/col/colref.rs:549]:

```rust
let svd = mat_ref.thin_svd().unwrap();
let s_col = svd.S().column_vector();  // DiagRef → ColRef
let s_slice: &[f64] = s_col.as_slice();  // &[f64] in nonincreasing order
```

### Parallelism control for faer SVD

faer SVD uses `get_global_parallelism()` internally [VERIFIED: faer-0.23.2/src/linalg/solvers.rs:1132].
Its rayon feature is enabled in the vendored version (Cargo.lock includes `rayon` in faer's deps
[VERIFIED: Cargo.lock, name="faer" dependencies list]).

For a **fair sequential comparison** with nalgebra (which is always sequential):

```rust
use faer::{set_global_parallelism, Par};

// Before nalgebra bench group — no action needed (nalgebra is always sequential)
// Before faer SEQUENTIAL bench group:
set_global_parallelism(Par::Seq);
// Before faer PARALLEL bench group (optional exploration cell):
set_global_parallelism(Par::rayon(0));  // 0 = use rayon default thread count
```

> **IMPORTANT:** `set_global_parallelism` must be called OUTSIDE `b.iter()` — it sets a global
> atomic. Call it once per bench group, before `group.bench_function(...)`.

### Bench design — `bench_p6_svd_comparison`

The bench function follows the `bench_p4_fpca` pattern exactly, with three sub-groups:

**Sub-group A — nalgebra SVD (baseline)**
- Build `DMatrix<f64>` input via `weighted.to_dmatrix()` OUTSIDE `b.iter()`
- Inside `b.iter()`: clone the DMatrix + call `SVD::new(dmatrix.clone(), true, true)`
- This measures: DMatrix clone overhead + SVD compute (same as the real `fdata_to_pc_1d` path)
- Alternatively: build a fresh DMatrix each iteration via `iter_batched` — matches real usage

**Sub-group B — faer thin_svd (sequential, fair comparison)**
- Build `FdMatrix` weighted input OUTSIDE `b.iter()`
- Before bench group: `set_global_parallelism(Par::Seq)`
- Inside `b.iter()`: `MatRef::from_column_major_slice(&black_box(&weighted.data), n, m).thin_svd()`
- This measures: MatRef construction (zero-copy) + faer SVD sequential

**Sub-group C — conversion cost alone (attribution)**
- Inside `b.iter()`: `MatRef::from_column_major_slice(&black_box(&weighted.data), n, m)` + `black_box(r)`
- This isolates the FdMatrix→MatRef view construction cost (expected: nanoseconds — it's a pointer + 2 ints)
- For comparison, DMatrix::from_column_slice cost can be measured separately

**Workload sizes (from Phase 1 workload matrix, D-07)**:

| Cell | Expected nalgebra time | Expected faer time | Sample tier |
|------|----------------------|-------------------|-------------|
| N=100, M=50 | ~213 µs | unknown | sample_size(20), mtime(20s) |
| N=100, M=200 | ~1.69 ms | unknown | sample_size(20), mtime(20s) |
| N=500, M=50 | ~1.22 ms | unknown | sample_size(20), mtime(20s) |
| N=500, M=200 | ~16 ms | unknown | sample_size(20), mtime(20s) |
| N=1000, M=50 | ~3.17 ms | unknown | sample_size(20), mtime(20s) |
| N=1000, M=200 | ~38 ms | unknown | sample_size(10), mtime(20s) |
| N=500, M=500 | ~?? (O(m³) × 2.5) | unknown | sample_size(10), mtime(30s) — CROSSOVER PROBE |

The M=500 cell is added as a **crossover probe**: faer may win increasingly as M grows (O(m³) SVD
cost). Expected nalgebra time at M=500: ~100–200 ms/iter extrapolating from M=200 (~16 ms) by O(m³)
= (500/200)³ = ~15.6× → ~250 ms. This is where faer's algorithmic improvements (BLAS-3 SVD kernel,
better cache utilization) should show the largest absolute speedup.

### Numerical equivalence check

nalgebra and faer may differ in:
1. Sign of singular vectors (arbitrary per column — not a correctness issue)
2. Ordering (both output nonincreasing order — should match)
3. Magnitude (should match to numerical precision)

**Safe invariant to compare:** singular values only (sign-invariant, order matches).

```rust
// In a standalone verification function (not in the bench loop):
fn assert_svd_equiv(n: usize, m: usize, tol: f64) {
    let (data, argvals) = generate_curves(n, m);
    // Build weighted input same as fdata_to_pc_1d does
    let weights = simpsons_weights(&argvals);
    let sqrt_w: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut weighted = data.clone();
    for i in 0..n { for j in 0..m { weighted[(i,j)] *= sqrt_w[j]; } }

    // nalgebra path
    let na_svd = SVD::new(weighted.to_dmatrix(), true, true);
    let na_s = na_svd.singular_values;  // DVector<f64>, nonincreasing

    // faer path
    let mat_ref = MatRef::from_column_major_slice(&weighted.data, n, m);
    let fa_svd = mat_ref.thin_svd().unwrap();
    let fa_s = fa_svd.S().column_vector().as_slice().to_vec();

    let k = na_s.len().min(fa_s.len());
    for i in 0..k {
        let rel_err = (na_s[i] - fa_s[i]).abs() / na_s[i].max(1e-12);
        assert!(rel_err < tol, "singular value {i}: nalgebra={}, faer={}, rel_err={}", na_s[i], fa_s[i], rel_err);
    }
}
```

Recommended tolerance: `1e-10` (both use f64 with LAPACK-grade accuracy).

> NOTE: This verification function should be written as a one-off helper called from outside
> `b.iter()` in the bench function (or as a separate `#[test]` in the bench binary guarded by
> `#[cfg(test)]`), not as a production-code assertion.

### Recommended project structure additions

```
fdars-core/
└── benches/
    └── audit_hotpaths.rs   ← add bench_p6_svd_comparison here (already registered file)
.planning/
└── research/
    └── bench/
        ├── p6_svd_nalgebra_linalg_run1.txt
        ├── p6_svd_nalgebra_linalg_run2.txt
        ├── p6_svd_faer_seq_linalg_run1.txt
        ├── p6_svd_faer_seq_linalg_run2.txt
        └── p6_svd_conversion_linalg_run1.txt
```

### Anti-Patterns to Avoid

- **Comparing with faer parallel vs nalgebra sequential:** faer's SVD uses `get_global_parallelism()`; if the global is set to `Par::Rayon(...)` during the nalgebra group, the comparison is unfair. Always set `Par::Seq` before the primary comparison group.
- **Including DMatrix clone in the faer timing but not the nalgebra timing (or vice versa):** The nalgebra path requires a DMatrix input; build it OUTSIDE `b.iter()` and clone inside. The faer path takes a `MatRef` view — build the `FdMatrix` OUTSIDE and create the view inside. This is the fairest pairing.
- **Forgetting to register the new bench group in `criterion_group!`:** The existing `criterion_group!(benches, ...)` macro at line 1043 of `audit_hotpaths.rs` must have `bench_p6_svd_comparison` added to it [VERIFIED: fdars-core/benches/audit_hotpaths.rs:1043-1052].
- **Using `--release` flag with `cargo bench`:** Criterion bench profile is already opt-3; `--release` is rejected by cargo (confirmed in Phase 5 decisions). Confirm via `/release/deps/` binary path in criterion output [ASSUMED from Phase 5 pattern].
- **Conflating faer's thin SVD with full SVD:** `Svd::new_thin` produces U as m×min(m,n) and V as n×min(m,n). `Svd::new` (full) produces U as m×m and V as n×n. For FPCA where N >> M, thin = m×M and full = n×M — these differ significantly. Use `thin_svd()` to match nalgebra's `SVD::new(mat, true, true)` (the `true, true` flags are `compute_u=true, compute_v=true` with thin decomposition) [VERIFIED: fdars-core/src/regression.rs:298].

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fair SVD timing | Manual time::Instant measurement | criterion `b.iter()` | Criterion handles warm-up, outlier detection, statistical stabilization |
| Parallelism fairness | Thread::spawn / custom pool | `faer::set_global_parallelism(Par::Seq)` | faer reads the global at SVD call time — this is the documented API [VERIFIED: faer-0.23.2/src/lib.rs:1163-1169] |
| Singular values comparison | Custom sort/sign normalization | Compare only sorted magnitudes | Sign ambiguity makes vector comparison unreliable; values are order-invariant |
| Matrix input generation | Random data in `b.iter()` | `generate_curves()` OUTSIDE `b.iter()` | Input allocation must not be in the measured window (PITFALLS.md Pitfall 3) |

---

## Common Pitfalls

### Pitfall A: faer SVD uses rayon by default — naive comparison is unfair

**What goes wrong:** If `set_global_parallelism` is not called, faer's SVD uses `get_global_parallelism()` which defaults to the full rayon pool. The comparison then shows faer winning by an unfair margin because it is parallel while nalgebra is sequential.

**Why it happens:** faer's `Svd::new_imp` calls `get_global_parallelism()` at line 1132 of solvers.rs [VERIFIED: faer-0.23.2/src/linalg/solvers.rs:1132]. The global defaults to rayon if no explicit setting is made.

**How to avoid:** Call `faer::set_global_parallelism(Par::Seq)` once before the sequential faer bench group. Report both sequential-faer and parallel-faer timings as separate rows in the results table, clearly labeled.

**Warning signs:** faer shows >10× speedup over nalgebra at small M sizes where the algorithm should not be that much faster.

### Pitfall B: DMatrix construction asymmetry

**What goes wrong:** nalgebra path measures `SVD::new(weighted.to_dmatrix(), true, true)` where `to_dmatrix()` allocates. faer path uses a zero-copy `MatRef::from_column_major_slice`. If both are measured inside `b.iter()`, the nalgebra timing includes one allocation and the faer timing does not — a spurious advantage for faer.

**How to avoid:** Build a pre-allocated `DMatrix` OUTSIDE `b.iter()` for the nalgebra path and clone it inside `b.iter()`. This makes the "DMatrix clone cost" explicit. The faer view creation inside `b.iter()` is valid because it is a zero-cost abstraction (just a pointer + dims — confirmed from matref.rs:116: `MatRef::from_raw_parts(slice.as_ptr(), nrows, ncols, 1, nrows)`). Measure conversion separately in Sub-group C.

### Pitfall C: Sign-flipped singular vectors invalidate comparisons

**What goes wrong:** Code tries to compare left/right singular vectors between nalgebra and faer and finds large differences. This is not a bug — sign conventions for singular vectors are arbitrary per column.

**How to avoid:** Compare ONLY singular values (the `S` diagonal). Both libraries return them in nonincreasing order, so element-wise comparison is valid.

### Pitfall D: /tmp exhaustion causes bench bus-error

**What goes wrong:** `/tmp` tmpfs is at ~94% capacity (MEMORY.md). Running `cargo bench` triggers the doctest linker and causes a bus error unrelated to the benchmark code.

**How to avoid:** Always run `export TMPDIR=/home/simonm/.cache/fdars-bench-tmp` before `cargo bench` (MEMORY.md documented exception). This is the same mitigation used in Phases 3–5.

### Pitfall E: MSRV tension — faer requires Rust 1.84

**What goes wrong:** Running `cargo bench --features linalg` on a Rust < 1.84 compiler fails to compile because faer 0.23 requires rust-version = "1.84.0" [VERIFIED: faer-0.23.2/Cargo.toml]. fdars' own MSRV is 1.81.

**How to avoid:** This is already handled by the `linalg` feature being non-default. The dev environment has Rust 1.97.0 (CLAUDE.md), so it compiles fine. No action needed for running the bench. The adoption note for the backlog must mention this MSRV tension: shipping faer SVD in the default path would raise fdars' effective MSRV to 1.84 (CRAN Windows compatibility concern).

### Pitfall F: M=500 cell may be slow — needs sample_size reduction

**What goes wrong:** At N=1000, M=500, nalgebra SVD is O(m³) × 15× relative to M=200 → expected ~600 ms/iter. With default sample_size=100 this takes ~60s just for the bench warmup.

**How to avoid:** For M=500 cells use `sample_size(10)` and `measurement_time(30s)` — same tier as the slowest Phase 4 cells. Document timing estimate in the function docstring.

---

## Code Examples

### Pattern 1: bench_p6_svd_comparison skeleton

```rust
// Source: mirrors bench_p4_fpca in audit_hotpaths.rs (lines 760–840)
// Feature gate: only compiles under --features linalg (faer is linalg-gated)
#[cfg(feature = "linalg")]
fn bench_p6_svd_comparison(c: &mut Criterion) {
    use faer::{set_global_parallelism, Par, MatRef};
    use nalgebra::linalg::SVD;  // already imported at top of file via fdars_core::regression internals

    // --- sub-group A: nalgebra SVD (baseline) ---
    {
        let mut group = c.benchmark_group("audit_p6_svd_nalgebra");
        group.sample_size(20);
        group.measurement_time(std::time::Duration::from_secs(20));
        group.warm_up_time(std::time::Duration::from_secs(5));

        for (n, m, s_size) in [(100,50,20),(100,200,20),(500,50,20),(500,200,20),(1000,50,20),(1000,200,10)] {
            group.sample_size(s_size);
            let (data, argvals) = generate_weighted_input(n, m);  // see helper below
            let dmatrix = data.to_dmatrix();  // pre-build DMatrix OUTSIDE b.iter()
            group.bench_function(format!("n{n}_m{m}"), |b| {
                b.iter(|| {
                    // Clone DMatrix inside iter (matches real fdata_to_pc_1d allocation pattern)
                    let dm = black_box(dmatrix.clone());
                    black_box(SVD::new(dm, true, true))
                })
            });
        }
        group.finish();
    }

    // --- sub-group B: faer thin_svd SEQUENTIAL (fair comparison) ---
    {
        set_global_parallelism(Par::Seq);  // OUTSIDE b.iter()
        let mut group = c.benchmark_group("audit_p6_svd_faer_seq");
        group.sample_size(20);
        group.measurement_time(std::time::Duration::from_secs(20));
        group.warm_up_time(std::time::Duration::from_secs(5));

        for (n, m, s_size) in [(100,50,20),(100,200,20),(500,50,20),(500,200,20),(1000,50,20),(1000,200,10)] {
            group.sample_size(s_size);
            let (weighted, _) = generate_weighted_input(n, m);  // FdMatrix, column-major
            group.bench_function(format!("n{n}_m{m}"), |b| {
                b.iter(|| {
                    // Zero-copy view — no allocation; MatRef borrows weighted.data
                    let mat_ref = MatRef::<f64>::from_column_major_slice(
                        black_box(&weighted.data), n, m);
                    black_box(mat_ref.thin_svd())
                })
            });
        }
        group.finish();
    }

    // --- sub-group C: conversion cost alone (attribution) ---
    {
        let mut group = c.benchmark_group("audit_p6_svd_conversion");
        // ... measure MatRef::from_column_major_slice + black_box at each cell
        group.finish();
    }
}
```

Note: `generate_weighted_input(n, m)` is a new helper (not in existing `generate_curves`) that
applies the Simpson weights scaling matching `fdata_to_pc_1d` lines 287–295. This must be built
outside `b.iter()` for both sub-groups.

### Pattern 2: faer singular values extraction

```rust
// Source: faer-0.23.2/src/linalg/solvers.rs:1170-1182 and col/colref.rs:549
let svd = mat_ref.thin_svd().unwrap();
let s_col_ref = svd.S().column_vector();
let singular_values: &[f64] = s_col_ref.as_slice();
// singular_values is a borrowed slice in nonincreasing order, same as nalgebra's output
```

### Pattern 3: Artifact naming convention (D-06)

Following the `p<phase>_<target>_<features>_run<N>.txt` convention established in Phase 1:

```
p6_svd_nalgebra_linalg_run1.txt     # nalgebra baseline, run 1 of 2
p6_svd_nalgebra_linalg_run2.txt     # nalgebra baseline, run 2 of 2 (variance check)
p6_svd_faer_seq_linalg_run1.txt     # faer sequential, run 1
p6_svd_faer_seq_linalg_run2.txt     # faer sequential, run 2
p6_svd_faer_par_linalg_run1.txt     # faer parallel (optional exploration), run 1
p6_svd_conversion_linalg_run1.txt   # conversion cost alone, run 1
```

All saved under `.planning/research/bench/`. AUDIT-REPORT Phase 6 section links to each artifact.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| nalgebra SVD (only option) | faer thin_svd available (behind `linalg` feature) | faer 0.18 consolidation | faer's SVD uses BLAS-3 kernels and cache-optimized routines; expected 3–10× faster at square/large matrices |
| Full SVD (all components) | Truncated/thin SVD possible | faer 0.18+ | For FPCA where ncomp « M, thin SVD computes only min(N,M) components — same as nalgebra's `true, true` flags |
| faer separate sub-crates (pre-0.18) | Single `faer` crate | faer 0.18 | Import `use faer::{MatRef, Svd, set_global_parallelism, Par}` from one crate |

**Deprecated/outdated:**
- The old `faer-svd`, `faer-core` sub-crates (pre-0.18): replaced by the unified `faer` crate. Do not reference these.
- `faer::mat::MatRef::from_col_major_slice` (if used in older docs): the current API is `MatRef::from_column_major_slice` [VERIFIED: faer-0.23.2/src/mat/matref.rs:113].

---

## faer Adoption Note (for SC3 backlog item)

This section feeds the "maintenance-burden / stability risk" assessment required by PERF-06 SC3.

### Summary assessment

| Factor | Assessment | Confidence |
|--------|------------|------------|
| Performance vs nalgebra SVD | Expected 3–10× faster at M ≥ 200 [ASSUMED — from STACK.md §"nalgebra vs faer Performance Reasoning" and faer benchmark site; exact numbers will be confirmed by Phase 6 bench run] | MEDIUM |
| API stability | SVD API (thin_svd, from_column_major_slice, U/V/S accessors) has been stable since 0.18; breaking changes in other decompositions (LBLT rename in 0.22, constructor simplification in 0.19) but SVD API unaffected [VERIFIED: changelog + source inspection] | MEDIUM |
| Breaking change frequency | Moderate in 0.18–0.22 (≈ every 2–4 versions); 0.23.x additions only (generalized eigendecomposition, matrix-free SVD) | MEDIUM |
| Maintainer / bus factor | Single maintainer: sarah quinones. Active development. Repository on Codeberg [ASSUMED: from web search; project moved from GitHub to Codeberg] | LOW |
| MSRV tension | faer 0.23 requires Rust 1.84 [VERIFIED: faer-0.23.2/Cargo.toml: rust-version = "1.84.0"]. fdars MSRV is 1.81 [VERIFIED: fdars-core/Cargo.toml:15]. faer is already behind `linalg` feature (non-default), which is already documented as requiring Rust 1.84+ [VERIFIED: fdars-core/Cargo.toml:18-25 comments]. Shipping faer SVD would not change the effective MSRV for CRAN users (who already use default-features=false). |
| Integration cost | faer is already a dependency — zero new Cargo.toml work. The conversion is MatRef::from_column_major_slice (zero-copy, 1 line). The SVD call is mat_ref.thin_svd(). The output extraction (S, U, V) requires 2–3 lines each. Total code delta: ~20 lines in `fdata_to_pc_1d` | HIGH |
| Test/correctness risk | Singular values equivalent within 1e-10 [ASSUMED — standard expectation for LAPACK-grade backends; must be verified by the numerical equivalence check in the bench]. Sign-flip of singular vectors must be normalized if `FpcaResult.rotation` is compared across library boundaries. | MEDIUM |

### Integration ROI verdict (for backlog item)

The integration burden is LOW (faer already vendored, 1 line conversion, stable API), and the
potential speedup is MEDIUM-HIGH (3–10× for M ≥ 200 based on STACK.md reference data). The MSRV
tension is already accepted (linalg feature already has the 1.84 constraint). The main risk is the
single-maintainer bus factor and the historically moderate API churn (which has not touched the SVD
API since 0.18).

**Backlog phrasing (GSD-ready, for Phase 9):**
"Swap nalgebra SVD in `fdata_to_pc_1d` (regression.rs:298) for faer `thin_svd` behind the existing
`linalg` feature. Use `MatRef::from_column_major_slice` (zero-copy) instead of `to_dmatrix()`. Add
numerical equivalence test. Expected: 3–10× speedup for M ≥ 200; ~0 MSRV impact (linalg already
requires 1.84). Integration cost: S effort (~1 week). Severity: P2 (meaningful speedup for FPCA-
heavy workflows; not a blocker). Evidence: Phase 6 bench artifacts (links to p6_svd_*.txt)."

---

## Runtime State Inventory

Not applicable — this is a greenfield audit bench phase. No rename, refactor, or migration of any
stored state is involved.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | faer linalg feature | ✓ | 1.97.0 (CLAUDE.md) | — |
| faer 0.23.2 | bench SVD calls | ✓ | 0.23.2 (vendored, Cargo.lock) | — |
| criterion 0.5 | bench framework | ✓ | 0.5 (dev-dep, Cargo.toml:53) | — |
| TMPDIR workaround | avoid /tmp exhaustion | ✓ | `mkdir -p /home/simonm/.cache/fdars-bench-tmp` | None — required |

**Missing dependencies with no fallback:** None.

**TMPDIR:** Must be set before every `cargo bench` invocation (MEMORY.md documented exception):
```bash
export TMPDIR=/home/simonm/.cache/fdars-bench-tmp
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | criterion 0.5 (bench), Rust built-in `#[test]` (equivalence check) |
| Config file | `fdars-core/Cargo.toml` — `[[bench]] name = "audit_hotpaths" harness = false` [VERIFIED: Cargo.toml:95-96] |
| Quick run command | `export TMPDIR=/home/simonm/.cache/fdars-bench-tmp && cargo bench --features linalg -p fdars-core --bench audit_hotpaths -- audit_p6` |
| Full suite command | Same — all p6 bench groups match `audit_p6` prefix |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PERF-06 | nalgebra-vs-faer bench table at real FPCA sizes exists | bench run | `cargo bench --features linalg -- audit_p6_svd_nalgebra audit_p6_svd_faer_seq` | ❌ Wave 0 — add to audit_hotpaths.rs |
| PERF-06 | Artifacts saved under .planning/research/bench/ | file check | `ls .planning/research/bench/p6_svd_*.txt` | ❌ Wave 0 — produced by bench run |
| PERF-06 | Numerical equivalence — faer and nalgebra singular values agree within 1e-10 | test (in-bench helper or separate #[test]) | `cargo test --features linalg -p fdars-core -- svd_equivalence` | ❌ Wave 0 — add helper |
| PERF-06 | Phase 6 section in AUDIT-REPORT.md with table, adoption note, and backlog item | doc check | manual review | ❌ Wave 0 — written after bench run |

### Sampling Rate

- **Per task commit:** `cargo bench --features linalg -p fdars-core --bench audit_hotpaths -- audit_p6_svd_nalgebra n100_m50`  (single cell smoke-check: ~5s)
- **Per wave merge:** Full `audit_p6` group run (all sizes, both libraries): ~5–10 min
- **Phase gate:** All 6 primary cells benchmarked twice (run1+run2) with ≤5% variance before writing AUDIT-REPORT section

### Wave 0 Gaps

- [ ] `bench_p6_svd_comparison` function in `fdars-core/benches/audit_hotpaths.rs` — covers PERF-06 bench
- [ ] `generate_weighted_input(n, m)` helper in `audit_hotpaths.rs` — needed to build the Simpson-weighted matrix matching `fdata_to_pc_1d` lines 287–295
- [ ] Register `bench_p6_svd_comparison` in `criterion_group!(benches, ...)` at `audit_hotpaths.rs` line 1043 [VERIFIED: audit_hotpaths.rs:1043-1052]
- [ ] Numerical equivalence check function (in-bench or separate `#[test]`)
- [ ] Phase 6 section in `.planning/research/AUDIT-REPORT.md` (written after bench run)

---

## Security Domain

This phase introduces no network calls, user input handling, or authentication. Security domain
is not applicable to a benchmarking/audit phase.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | faer is 3–10× faster than nalgebra for SVD at M ≥ 200 (from STACK.md "nalgebra vs faer Performance Reasoning" table showing 8–13× for large square matrices) | faer Adoption Note | If faer is slower at fdars' rectangular sizes (N >> M, e.g. N=1000×M=200), the backlog item severity drops to P3. The bench run resolves this. |
| A2 | faer SVD API (thin_svd, from_column_major_slice, U/V/S accessors) has been stable since 0.18 and will not break in a 0.24 upgrade | faer Adoption Note | If API breaks on next upgrade, integration cost increases from S to M effort. Partially mitigated by the `linalg` feature pinning faer = "0.23" in Cargo.toml. |
| A3 | Single maintainer (sarah quinones) status from web search | faer Adoption Note | If the project gains additional maintainers or becomes dormant, the bus factor assessment changes. |
| A4 | The `--release` flag is rejected by `cargo bench` (from Phase 5 decision) | Pitfall E | If this behavior changed, the bench runs in debug mode and numbers are invalid. Always verify `/release/deps/` in criterion output. |
| A5 | Numerical equivalence of faer and nalgebra singular values to 1e-10 | Code Examples / Validation | If the backends disagree at larger tolerances, `fdata_to_pc_1d` results would change on adoption — a correctness risk requiring regression testing of existing behavior. |

---

## Open Questions

1. **Does faer's thin_svd actually match nalgebra's `SVD::new(mat, true, true)` output dimensions?**
   - What we know: nalgebra `true, true` means `compute_u=true, compute_v=true` and returns a thin decomposition at N >> M. faer `thin_svd` returns U as m×min(m,n), V as n×min(m,n) [VERIFIED: faer-0.23.2/src/linalg/solvers.rs:1137-1138].
   - What's unclear: whether "thin" means exactly the same thing in both APIs at all (N, M) combinations.
   - Recommendation: verify by asserting `svd.U().nrows() == n && svd.U().ncols() == m` (for N > M) in the equivalence check.

2. **Does `set_global_parallelism(Par::Seq)` persist across criterion iterations?**
   - What we know: `set_global_parallelism` sets a global atomic [VERIFIED: faer-0.23.2/src/lib.rs:1163-1169].
   - What's unclear: whether rayon resets this between bench iterations.
   - Recommendation: call `set_global_parallelism(Par::Seq)` once per bench group (outside `b.iter()`), not inside the iter loop. If criterion runs multiple warmup+measurement cycles, the global persists correctly.

3. **What is the crossover point where faer wins even with the `Par::Seq` constraint?**
   - What we know: at large square matrices (1024×1024) faer wins ~8× over nalgebra (STACK.md). At fdars' rectangular sizes (N=1000, M=200), the matrix is tall-and-thin — faer's algorithmic advantage may be different.
   - What's unclear: exact crossover M value.
   - Recommendation: the M=500 optional cell provides the crossover probe data. If faer wins by > 2× at M=200, M=500 is confirmatory; if faer loses at M=50 but wins at M=200, the crossover is between M=50 and M=200.

---

## Sources

### Primary (HIGH confidence — verified from on-disk sources this session)

- `faer-0.23.2/src/linalg/solvers.rs:1117-1195` — Svd::new, Svd::new_thin, U/V/S accessor signatures
- `faer-0.23.2/src/mat/matref.rs:85-128` — MatRef::from_column_major_slice signature and semantics
- `faer-0.23.2/src/col/colref.rs:297,549` — ColRef::iter and ColRef::as_slice
- `faer-0.23.2/src/lib.rs:1163-1187` — set_global_parallelism, get_global_parallelism, Par enum
- `faer-0.23.2/Cargo.toml` — rust-version = "1.84.0", maintainer, features
- `fdars-core/Cargo.toml:43,53,95-96` — faer dep, criterion dep, audit_hotpaths bench entry
- `fdars-core/src/regression.rs:249-320` — fdata_to_pc_1d SVD call at :298
- `fdars-core/benches/audit_hotpaths.rs:1-57, 760-840, 1043-1052` — generate_curves, bench_p4_fpca pattern, criterion_group registration
- `.planning/research/AUDIT-REPORT.md:571-609` — Phase 4 go/no-go decision, SVD share 99.8–99.9%, copy-share 0.14–0.17%
- `.planning/phases/04-fpca-svd-allocation-audit/04-VERIFICATION.md:28-29` — Phase 4 SC1/SC2 verification
- `Cargo.lock` — faer version 0.23.2, faer depends on rayon (parallel feature active)

### Secondary (MEDIUM confidence — docs.rs fetch this session)

- docs.rs/faer/0.23.2/faer/linalg/solvers/struct.Svd.html — confirmed Svd::new, new_thin, U/V/S API signatures (consistent with source read)
- docs.rs/faer/0.23.2 crate root — confirmed Mat::thin_svd(), Mat::svd() as method forms
- github.com/sarah-quinones/faer-rs CHANGELOG.md — version history 0.18–0.23, breaking change pattern

### Tertiary (LOW confidence — training/search only, marked ASSUMED)

- STACK.md §"nalgebra vs faer Performance Reasoning" — 8–13× faer speedup at large squares (from STACK.md, which cites "faer benchmark data via web search, MEDIUM confidence")
- faer maintainer bus factor and Codeberg repository location — from web search

---

## Metadata

**Confidence breakdown:**
- faer SVD API: HIGH — verified from vendored source files this session
- Phase 4 go/no-go trigger: HIGH — verified from on-disk artifacts and VERIFICATION.md
- Bench design patterns: HIGH — verified from audit_hotpaths.rs and PITFALLS.md source
- faer speedup vs nalgebra: MEDIUM — from STACK.md (which itself rates this MEDIUM)
- faer maintenance/bus-factor: LOW — from web search only

**Research date:** 2026-08-08
**Valid until:** 2026-09-07 (30 days — faer API is stable in 0.23.x; workload matrix sizes are locked by D-07)
