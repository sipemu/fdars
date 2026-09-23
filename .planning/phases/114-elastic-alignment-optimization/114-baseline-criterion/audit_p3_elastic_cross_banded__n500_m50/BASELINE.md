# BEFORE-Baseline: audit_p3_elastic_cross_banded/n500_m50

Captured in Plan 01 (Wave 1) on UN-OPTIMIZED code, before any `fdars-core/src` change.
This is the immovable "before" number for the OPT-01 non-overlapping-CI check in Plan 02.

## Bench Command

```
cargo bench -p fdars-core --features linalg,parallel --bench audit_hotpaths -- "audit_p3_elastic_cross_banded/n500_m50"
```

## Criterion Result

| Metric | Value |
|--------|-------|
| **mean (point_estimate)** | **7,310,073,752.4 ns (≈ 7.310 s)** |
| CI lower bound (95%) | 7,072,703,467.4 ns (≈ 7.073 s) |
| CI upper bound (95%) | 7,552,415,932.8 ns (≈ 7.552 s) |
| CI width | ±239 ms (±3.3%) |
| median (point_estimate) | 7,176,158,116.5 ns (≈ 7.176 s) |
| std_dev | 406,000,046.0 ns (≈ 406 ms) |

## Provenance

| Field | Value |
|-------|-------|
| git SHA | `8c0b1d7d6b28db0f2888d1b9f00890d40f0c4161` |
| Feature set | `linalg,parallel` |
| Criterion samples | 10 (measurement_time=60s) |
| Criterion warn | Unable to complete 10 samples in 60s — estimated 66.3s (normal for this cell) |

## Machine

| Field | Value |
|-------|-------|
| OS | Linux simon-xps159530 6.18.45-1-MANJARO #1 SMP PREEMPT_DYNAMIC x86_64 |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900H |

## Plan 02 Usage

Plan 02 must run the SAME bench command on the OPTIMIZED code and compare the after CI
against this before CI. The OPT-01 success criterion is **non-overlapping CIs** (after
upper bound < before lower bound). With this before CI: [7.073 s, 7.552 s], Plan 02's
after upper bound must be < 7.073 s to declare a statistically significant speedup.

The raw `estimates.json` in this directory is the machine-readable record; BASELINE.md
is the human-readable summary for plan review.
