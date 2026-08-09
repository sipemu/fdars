# Phase 3: Elastic Alignment Hot Path - Pattern Map

**Mapped:** 2026-08-07
**Files analyzed:** 2 (1 modified bench source + 1 modified Cargo.toml — Cargo.toml already registered, likely no change)
**Analogs found:** 1 / 1 (exact — same file, same criterion structure)

This is an **analysis phase**: no `fdars-core/src` algorithm changes. The only code added is new criterion bench cases inside the existing Phase-1 audit harness. Every pattern the planner needs already exists in that harness — this map extracts the exact excerpts to mirror plus the exact target-function signatures the new benches call.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/benches/audit_hotpaths.rs` (MODIFY — extend) | benchmark harness | batch / measurement | itself: `bench_elastic_sentinel`, `bench_matrix_sentinel` (same file) | exact |
| `fdars-core/Cargo.toml` `[[bench]]` (already present) | config | build-registration | existing `audit_hotpaths` entry (lines 90–92) | exact (no change needed) |

**Note:** `audit_hotpaths` is already registered with `harness = false` (Cargo.toml:90–92). If the planner chooses to extend the existing file (the CONTEXT.md-recommended path), **no Cargo.toml edit is required**. A new bench file would require a new `[[bench]]` block mirroring lines 90–92.

## Pattern Assignments

### `fdars-core/benches/audit_hotpaths.rs` (benchmark harness, batch/measurement)

**Analog:** the same file's existing sentinel functions. Three internal analogs cover everything the phase adds:
- `bench_elastic_sentinel` (lines 132–152) → template for `elastic_self_distance_matrix` + `_banded` + `elastic_cross_distance_matrix` + `_banded` cells.
- `bench_matrix_sentinel` (lines 103–125) → template for `karcher_mean` + `karcher_mean_banded` cells (correct arg order + `black_box` wrapping already shown).
- `generate_curves` (lines 34–48) → the seeded/deterministic N×M column-major generator to reuse verbatim.

**Imports pattern** (lines 20–28) — extend the `alignment` import to pull in the banded twins + cross-distance:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::alignment::{elastic_self_distance_matrix, karcher_mean};
// ... other module imports ...
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;
```
Extend the alignment line to (all publicly re-exported from `fdars_core::alignment` — confirmed `alignment/mod.rs:66` and `:75-80`):
```rust
use fdars_core::alignment::{
    elastic_cross_distance_matrix, elastic_cross_distance_matrix_banded,
    elastic_self_distance_matrix, elastic_self_distance_matrix_banded,
    karcher_mean, karcher_mean_banded,
};
```

**Seeded N×M generator — reuse verbatim** (lines 34–48). Deterministic (no RNG), column-major `data[i + j * n]` per the `FdMatrix` contract; already returns `(FdMatrix, argvals)`:
```rust
/// Column-major layout: element (i, j) at index `i + j * n`.
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        for j in 0..m {
            let t = argvals[j];
            data[i + j * n] = amp * (2.0 * PI * (t + phase)).sin();
        }
    }
    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, argvals)
}
```
The N×N cross-distance cell (D-02) uses this same generator once and passes the result as **both** `data1` and `data2` (`&data, &data`) — no second generator needed.

**Core bench-cell pattern — build input OUTSIDE `b.iter()`, `black_box` inputs AND outputs** (from `bench_elastic_sentinel`, lines 132–152). Note the `sample_size` / `measurement_time` / `warm_up_time` tuning block and the CAP comment style:
```rust
fn bench_elastic_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_elastic");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    let (data, argvals) = generate_curves(100, 50);          // OUTSIDE b.iter()
    group.bench_function("n100_m50_capped", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix(          // black_box on OUTPUT
                black_box(&data),                            // black_box on INPUTS
                black_box(&argvals),
                black_box(0.0),
            ))
        })
    });
    group.finish();
}
```
Mirror this for each new cell. Distance-matrix signatures (from `pairwise.rs`):
- `elastic_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix` (`pairwise.rs:194`)
- `elastic_self_distance_matrix_banded(data, argvals, lambda, band_frac) -> FdMatrix` (`pairwise.rs:205`)
- `elastic_cross_distance_matrix(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix` (`pairwise.rs:266`)
- `elastic_cross_distance_matrix_banded(data1, data2, argvals, lambda, band_frac) -> FdMatrix` (`pairwise.rs:278`)

**karcher_mean cell pattern — exact arg order + `black_box` wrapping** (from `bench_matrix_sentinel`, lines 112–122). This is the load-bearing signature reference — `karcher_mean` takes `(data, argvals, max_iter, tol, lambda)` with `max_iter: usize`, `tol: f64`, `lambda: f64`:
```rust
let (data, argvals) = generate_curves(100, 50);
group.bench_function("n100_m50", |b| {
    b.iter(|| {
        black_box(karcher_mean(
            black_box(&data),
            black_box(&argvals),
            black_box(10usize),   // max_iter — Phase 3 locks 20usize (D-06)
            black_box(1e-3),      // tol     — Phase 3 locks 1e-4 (D-06)
            black_box(0.0),       // lambda  — Phase 3 locks 0.0  (D-06)
        ))
    })
});
```
Signatures (from `karcher.rs`):
- `karcher_mean(data: &FdMatrix, argvals: &[f64], max_iter: usize, tol: f64, lambda: f64) -> KarcherMeanResult` (`karcher.rs:293`) — **defaults `band_frac = 0.0` → unbanded full DP** (`karcher.rs:300` calls `karcher_mean_impl(.., 0.0)`). This is the D-05 / Anti-Pattern 2 evidence to record.
- `karcher_mean_banded(data, argvals, max_iter, tol, lambda, band_frac) -> KarcherMeanResult` (`karcher.rs:312`) — appends `band_frac: f64` as the 6th arg.

**Registration pattern** (lines 260–270) — add each new group fn to the `criterion_group!` list:
```rust
criterion_group!(
    benches,
    bench_fpca_sentinel,
    bench_matrix_sentinel,
    bench_elastic_sentinel,
    // ... add new p3 elastic-sweep group fns here ...
);
criterion_main!(benches);
```

---

## Shared Patterns

### band_frac → band_radius semantics (governs all `_banded` cells)
**Source:** `alignment/mod.rs:533` `band_radius(band_frac: f64, m: usize) -> Option<usize>`
**Apply to:** every `_banded` bench cell (D-03 fixes `band_frac = 0.1`)
```rust
pub(super) fn band_radius(band_frac: f64, m: usize) -> Option<usize> {
    if band_frac > 0.0 && band_frac < 1.0 {
        Some(((band_frac * m as f64).ceil() as usize).max(1))
    } else {
        None   // band_frac ≤ 0 or ≥ 1 → unbounded (unbanded) path
    }
}
```
Implication the report must state: `band_frac = 0.1` at `M = 200` → radius `ceil(0.1 * 200) = 20` points → ≈10× theoretical DP reduction (m/band = 200/20). `karcher_mean`'s default `band_frac = 0.0` returns `None` → full unbanded DP (Anti-Pattern 2). The bench does not call `band_radius` directly; it passes `0.1` (or `0.0`) into the public `_banded`/plain functions, which call `band_radius` internally.

### Criterion cell discipline (Phase-1 D-02, inherited)
**Source:** every `bench_*_sentinel` fn in `audit_hotpaths.rs`
**Apply to:** all new p3 cells
- Build the `(data, argvals)` input **outside** `b.iter()` (comment `// Build input OUTSIDE b.iter()` — lines 86, 110, 139) so the allocator is not measured.
- Wrap **both** inputs and the return value in `black_box(...)` (Pitfall 3 / D-02).
- Per-cell `sample_size` / `measurement_time` / `warm_up_time` tuning is Claude's discretion (CONTEXT D-42): the borderline N=500×M=200 elastic cell warrants `measurement_time = 60s` (workload matrix); small cells may keep the sentinel defaults (`sample_size(20)`, `measurement_time(20s)`, `warm_up_time(5s)`). Document whatever is applied in a `///` doc comment on the group fn, matching the existing sentinel doc-comment style (e.g. lines 127–131).

### Feature-set + reproducibility tagging (Phase-1 D-06, inherited — not code)
**Source:** module-doc header lines 1–18 + `.planning/phases/01-.../01-CONTEXT.md`
**Apply to:** the AUDIT-REPORT section and `bench/` artifacts, not the source
- Run `release + --features linalg,parallel` (D-01); `iter_maybe_parallel!` in the elastic inner N-loop (`pairwise.rs:227`) means `parallel` is genuinely exercised.
- Raw output → `.planning/research/bench/p3_<target>_<features>_run<N>.txt` (Phase-1 naming).
- Tag each finding with feature set + toolchain version; two-run variance within ±5% (>10% = LOW CONFIDENCE).

## No Analog Found

None. Every new bench cell has an exact in-file analog (`bench_elastic_sentinel` / `bench_matrix_sentinel`), and every target function (plain + banded) is already publicly re-exported from `fdars_core::alignment`. No RESEARCH.md fallback patterns are required.

## Metadata

**Analog search scope:** `fdars-core/benches/audit_hotpaths.rs`, `fdars-core/Cargo.toml`, `fdars-core/src/alignment/{karcher.rs, pairwise.rs, mod.rs}`
**Files scanned:** 5
**Pattern extraction date:** 2026-08-07
