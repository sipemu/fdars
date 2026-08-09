---
phase: 03-elastic-alignment-hot-path
reviewed: 2026-08-08T00:00:00Z
depth: standard
files_reviewed: 1
files_reviewed_list:
  - fdars-core/benches/audit_hotpaths.rs
findings:
  critical: 2
  warning: 3
  info: 1
  total: 6
status: issues-found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-08-08
**Depth:** standard
**Files Reviewed:** 1 (fdars-core/benches/audit_hotpaths.rs)
**Status:** issues-found

## Summary

Reviewed the Phase-3 benchmark harness additions: six new criterion group functions
(`bench_p3_karcher`, `bench_p3_karcher_banded`, `bench_p3_elastic_self`,
`bench_p3_elastic_self_banded`, `bench_p3_elastic_cross`,
`bench_p3_elastic_cross_banded`) plus the pre-existing seven sentinel functions.
All verified against the actual public function signatures in `karcher.rs` and
`pairwise.rs`.

The Phase-1 sentinel functions are mostly correct. The Phase-3 additions have two
critical measurement-correctness defects: missing `black_box` on outputs in two
legacy sentinels, and stale criterion group settings silently inherited by later
cells within a single group function.

---

## Critical Issues

### CR-01: `bench_cv_sentinel` — `fclassif_cv` return not wrapped in `black_box`

**File:** `fdars-core/benches/audit_hotpaths.rs:194-205`

**Issue:** The entire `b.iter` closure returns the `Result<ClassifCvResult, FdarError>`
from `fclassif_cv` without passing it through `black_box`. Criterion's `b.iter`
does NOT automatically `black_box` its closure's return value in criterion 0.5 —
only `b.iter_with_large_drop` or explicit wrapping does. The optimizer therefore
has permission to observe that the closure result is dropped unused, and in a
release build with LTO may eliminate branches or intermediate allocations inside
the call. The sentinel is intended to baseline the CV-loop hot path; dead-code
elimination renders the timing unrepresentative.

**Fix:**
```rust
group.bench_function("n100_m50_capped", |b| {
    b.iter(|| {
        black_box(fclassif_cv(   // <-- wrap the entire call in black_box
            black_box(&data),
            black_box(argvals.as_slice()),
            black_box(y.as_slice()),
            black_box(None),
            black_box("lda"),
            black_box(5usize),
            black_box(5usize),
            black_box(42u64),
        ))
    })
});
```

---

### CR-02: `bench_streaming_sentinel` — `fm.depth_batch` return not wrapped in `black_box`

**File:** `fdars-core/benches/audit_hotpaths.rs:224-229`

**Issue:** `fm.depth_batch(black_box(&data))` returns a `Vec<f64>` that is
immediately dropped without being passed through `black_box`. Like CR-01, this
gives the optimizer freedom to eliminate the construction of the returned vector,
and potentially the depth computation itself, in a release + LTO build. The
streaming depth sentinel is supposed to measure `O(n·m)` build + `O(n·m)` query;
without `black_box` on the output the query result may be optimized away.

Additionally, the intermediate `let fm = StreamingFraimanMuniz::new(state, true)`
is also not black-boxed — `fm` is only used to call `depth_batch`, so if the
result of `depth_batch` is eliminated the optimizer may collapse the entire
`SortedReferenceState::from_reference` + `new` chain as well.

**Fix:**
```rust
group.bench_function("n500_m200", |b| {
    b.iter(|| {
        let state = SortedReferenceState::from_reference(black_box(&data));
        let fm = StreamingFraimanMuniz::new(state, true);
        black_box(fm.depth_batch(black_box(&data)))  // <-- wrap return
    })
});
```

---

## Warnings

### WR-01: `bench_p3_karcher` — `n500_m50` and `n500_m200` cells silently inherit stale criterion settings

**File:** `fdars-core/benches/audit_hotpaths.rs:316-342`

**Issue:** In criterion 0.5, `group.sample_size(N)` and `group.measurement_time(D)`
apply to all subsequent `bench_function` calls within the same group until
overridden. The `n100_m200` cell sets `sample_size(10)` and
`measurement_time(60s)` at lines 300-301. The `n500_m50` cell at line 317 and the
`n500_m200` cell at line 330 have no fresh `group.sample_size` / `group.measurement_time`
calls — they inherit the `n100_m200` values silently.

For `n500_m200` the comment says "borderline cell (workload-matrix 60s)" so the
inherited 60s is intentional. However the `n500_m50` cell, which the comment
describes as "~1.6-4s/iter", receives `measurement_time(60s)` even though 20-30s
would saturate 10 samples. More importantly, there is no explicit documentation in
the code that the settings are inherited, making the behavior appear ambiguous to
future readers and making it impossible to distinguish intentional inheritance from
an omitted call.

The same pattern occurs in `bench_p3_karcher_banded` (lines 398-428),
`bench_p3_elastic_self` (lines 477-499), `bench_p3_elastic_self_banded`
(lines 548-573), `bench_p3_elastic_cross` (lines 629-653), and
`bench_p3_elastic_cross_banded` (lines 705-731): in each group, cells after
the second lose explicit timing declarations.

For `bench_p3_elastic_self`, `n500_m50` is described as "~27 s/iter" so 60s is
arguably appropriate, but for karcher `n500_m50` at "~1.6-4s/iter" the inherited
60s wastes wall-clock time when the benchmarks are re-run.

**Fix:** Add an explicit comment at each cell that intentionally inherits settings,
or add explicit `group.sample_size` / `group.measurement_time` calls. At minimum,
mark the n500_m50 karcher cell:
```rust
// --- n500_m50: ~1.6-4s/iter — inherits n100_m200 settings (10 samples, 60s) ---
// (Oversized measurement_time acceptable for consistency across cells.)
let (data500_50, argvals50b) = generate_curves(500, 50);
```

---

### WR-02: `bench_p3_karcher_banded` — `n100_m50` doc says "~4x vs unbanded" but observed data shows 4-6x at M=200, not M=50

**File:** `fdars-core/benches/audit_hotpaths.rs:362`

**Issue:** The doc comment for `bench_p3_karcher_banded`'s `n100_m50` cell states
"banding reduces cost ~4x vs unbanded". However, the `band_radius` implementation
for `band_frac=0.1, M=50` computes `ceil(0.1 × 50) = 5` points. The theoretical
reduction at M=50 is `m/band = 50/5 = 10×`, not 4×. The 4× figure in the Plan-02
SUMMARY.md refers to the _observed_ reduction at the _large_ N=500×M=200 karcher
cells under high OS jitter, not the M=50 cell. Quoting that number in a code
comment about M=50 is misleading to future maintainers trying to cross-check the
banded results.

The comment also does not acknowledge that the measured karcher cells were all
flagged LOW CONFIDENCE due to OS scheduler jitter, making the specific "~4x"
figure particularly unreliable as a doc-comment claim.

**Fix:** Replace the specific ratio with a range or remove it:
```rust
// --- n100_m50: banding constrains each alignment to band_radius(0.1,50)=5 pts ---
// (Theoretical ~10× DP reduction; observed reduction across cells: 4–6× under jitter)
```

---

### WR-03: `bench_p3_elastic_cross` — doc says "approximately 2× the cost of self-distance" but this is not a sound general claim

**File:** `fdars-core/benches/audit_hotpaths.rs:585-587`

**Issue:** The doc comment states "Cross-distance visits all N×N pairs (not just
upper-triangular), so it is approximately 2× the cost of self-distance at the same
N×M." This claim is misleading as measurement-infrastructure documentation because:

1. `elastic_self_distance_matrix` uses `iter_maybe_parallel!` over the upper
   triangular only (N×(N-1)/2 pairs), then fills both halves. The actual pairwise
   alignment cost for cross at N×N is therefore `N²` pairs vs `N²/2` unique pairs
   for self — so cross visits exactly 2× as many alignments, making the "2×" claim
   correct in pair count.
2. However the empirical results in the SUMMARY show cross at N=100×M=200 took
   27.8s vs self at 17.6s — a ratio of ~1.58×, not ~2×. The comment will be read
   against those artifacts and creates a discrepancy.

The discrepancy arises because `elastic_self_distance_matrix` exploits triangle
symmetry (N(N-1)/2 alignments) while cross uses N² alignments, but the proportions
of overhead (SRSF precomputation, memory allocation) differ between the two. The
"2×" claim is directionally correct but quantitatively wrong relative to empirical
findings already recorded in the phase artifacts.

**Fix:**
```rust
// Cross-distance visits all N×N pairs (vs N(N-1)/2 for self-distance), so it is
// theoretically ~2× more alignment work; empirically ~1.5–2× observed at these sizes.
```

---

## Info

### IN-01: `generate_smoothing_data` — unused `enumerate()` in `y` construction

**File:** `fdars-core/benches/audit_hotpaths.rs:64-71`

**Issue:** The closure computing `y` uses `.enumerate()` at line 67 to capture
both the index `i` and the value `xi`:
```rust
.enumerate()
.map(|(i, &xi)| {
    let noise = ((i as f64 * 17.3 + 0.5).sin()) * 0.3;
    (2.0 * PI * xi).sin() + ...
```
Since `x` is a uniform grid (`x[i] = i / (n-1)`), the index `i` and the value
`xi` encode identical information — `i = xi * (n-1)`. Using `i` for noise and `xi`
for the signal is not wrong but makes the function unnecessarily convoluted. The
noise term could equivalently be written as `(xi * (n-1) as f64 * 17.3 + 0.5).sin()`,
eliminating the `enumerate` and making the intent clearer.

This does not affect measurement correctness (the smoothing sentinel is a legacy
Phase-1 function, not a Phase-3 addition) but adds unnecessary complexity for a
benchmark helper.

**Fix:**
```rust
let y: Vec<f64> = x
    .iter()
    .map(|&xi| {
        let noise = (xi * (x.len() - 1) as f64 * 17.3 + 0.5).sin() * 0.3;
        (2.0 * PI * xi).sin() + 0.5 * (4.0 * PI * xi).cos() + noise
    })
    .collect();
```

---

_Reviewed: 2026-08-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
