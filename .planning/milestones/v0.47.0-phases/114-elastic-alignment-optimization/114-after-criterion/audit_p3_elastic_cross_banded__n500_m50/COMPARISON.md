# AFTER criterion: audit_p3_elastic_cross_banded/n500_m50

Plan 02 AFTER measurement — `dp_fill_grid` band-iteration refactor (OPT-01).
Compare against the Plan 01 BEFORE-baseline from un-optimized code at SHA 8c0b1d7d.

## Bench Command

```
cargo bench -p fdars-core --features linalg,parallel --bench audit_hotpaths -- "audit_p3_elastic_cross_banded/n500_m50"
```

## Before vs After

| Metric          | BEFORE (Plan 01, SHA 8c0b1d7d)         | AFTER (Plan 02, refactor)               |
|-----------------|----------------------------------------|-----------------------------------------|
| mean            | 7,310,073,752 ns (7.310 s)             | 7,163,865,688 ns (7.164 s)              |
| CI lower (95%)  | 7,072,703,467 ns (7.073 s)             | 7,025,024,834 ns (7.025 s)              |
| CI upper (95%)  | 7,552,415,933 ns (7.552 s)             | 7,291,782,154 ns (7.292 s)              |
| CI width        | ±239 ms (±3.3%)                        | ±133 ms (±1.9%)                         |
| median          | 7,176,158,117 ns (7.176 s)             | 7,227,562,656 ns (7.228 s)              |
| std_dev         | 406,000,046 ns (406 ms)                | 226,712,890 ns (227 ms)                 |

## Speedup Calculation

| Quantity          | Value                              |
|-------------------|------------------------------------|
| Mean delta        | -146,208,064 ns (-146 ms)          |
| Percent change    | **-2.00%** (criterion's own estimate) |
| Criterion p-value | p = 0.33 > 0.05 (no change)        |

## Overlap Verdict: CIs OVERLAP — speedup is within measurement noise

The AFTER CI upper bound is **7,292 ms**, and the BEFORE CI lower bound is **7,073 ms**.
Since AFTER-upper (7,292 ms) > BEFORE-lower (7,073 ms), the confidence intervals overlap.

**The measured -2.0% mean improvement is within noise at this bench cell.**

This is an honest, expected result consistent with the RESEARCH analysis. The explanation:
the banded arm (`audit_p3_elastic_cross_banded/n500_m50`) with `n=500, m=50, band_frac=0.1`
has band radius `r = ceil(0.1 * 50) = 5`, producing a corridor of `2r+1 = 11` cells per
row. At `m=50`, the old code scanned all 50 cells per row and skipped ~78% via
`dp_skip_cell`. The skip-check itself (`abs_diff + comparison`) is cheap compared to the
actual `dp_relax_cell` work (7 neighbors, COPRIME_NBHD_7 iteration, f64 comparisons). At
this cell size (`m=50`), the loop-overhead saving from eliminating ~39 skip-checks per row
is subpercent relative to 11 actual relaxations — hidden in criterion's measurement noise.

For larger `m` (e.g., `m=200`, corridor still 11 cells but savings ~174 skip-checks/row vs
11 relaxations), the algorithmic O(n·r) vs O(n·m) benefit would be more visible. This cell
is not the worst-case for the banded arm; it was chosen as the benchmark cell in Phase 113's
audit and is representative of realistic production usage.

## Why This Change Is Still Correct and Worth Keeping

1. **Numeric identity proven**: golden fixture (debug + release) passes unchanged at <=1e-10.
2. **Algorithmic correctness**: iterates exactly the right cells in the right order.
3. **Code clarity**: the band iteration is now O(n·r) by construction (not by skip-logic);
   the intent is explicit and the dead code (`dp_skip_cell`) is removed.
4. **Foundation for future work**: a larger-m bench cell or a selective-scratch-init
   (Pitfall 5 already documented) could reveal the full speedup on production-scale inputs.

## The Unbanded Rank-#1 Path

The unbanded arm (`audit_p3_elastic_cross_banded` ranking #1 in Phase 113) is
**intentionally not touched**. The RESEARCH analysis confirmed the unbanded arm is already
near-optimal: it iterates all `nrows * ncols` cells with no per-cell branch overhead (the
old `dp_skip_cell` origin check was a single `tr==0 && tc==0` which the CPU branch predictor
handled well), and `dp_relax_cell` dominates. Adding the `match band { None => ..., Some(_)
=> ... }` structure adds one branch per call to `dp_fill_grid` — negligible — but no
further per-row overhead for the None arm.

## Machine

| Field | Value |
|-------|-------|
| OS | Linux simon-xps159530 6.18.45-1-MANJARO #1 SMP PREEMPT_DYNAMIC x86_64 |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900H |
| Git SHA (AFTER) | 30d2c9a2 |
| Feature set | linalg,parallel |
| Criterion samples | 10 |
| Criterion warning | Unable to complete 10 samples in 60s — estimated 71.3s (normal) |
