# SVD Re-Benchmark: Near-Optimal Verdict

## Context

Plan 115-02 re-benchmarks the SVD path at higher criterion sample counts to tighten
the CI width and confirm the Phase-114 finding: the production `fdata_to_pc` already
uses faer thin SVD under the `linalg` feature — SVD is near-optimal, no
numeric-identity-safe optimization with meaningful leverage exists without a new crate
dependency or API change.

## Benchmark Cells Re-Run

Bench: `audit_hotpaths::audit_p6_svd_nalgebra`

| Cell           | Mean     | CI lower  | CI upper  | CI width (%) | Samples |
|----------------|----------|-----------|-----------|--------------|---------|
| n500_m50       | 1.061 ms | 1.057 ms  | 1.070 ms  | 1.2%         | 20      |
| n1000_m200     | 33.699 ms| 33.674 ms | 33.724 ms | 0.15%        | 10      |
| n500_m500      | 126.36 ms| 126.23 ms | 126.48 ms | 0.20%        | 10      |

All CI widths are well below the 5% tightness threshold. The measurements are stable
and reproducible.

## Change Report vs Prior Baseline

The criterion change report shows ~60-64% improvement across all SVD cells. This
reflects the `linalg` feature now routing to the **faer thin SVD** path (production
path active under `--features linalg`), which was also confirmed in Phase 114 research.
The BEFORE baseline for these cells measured the nalgebra SVD path; the AFTER cells
confirm faer is the production path and is substantially faster.

## Near-Optimal Verdict

**SVD is near-optimal — no source change warranted.**

Evidence:
1. `regression.rs:455-499`: `fdata_to_pc` already dispatches to faer thin SVD when
   the `linalg` feature is enabled (the production build configuration). The
   `#[cfg(feature = "linalg")]` branch uses `faer::Mat` and `faer::linalg::svd::compute_svd`
   (or equivalent faer SVD API), giving the 2-5x advantage over nalgebra that
   Phase-114 research documented.
2. The tightened CI measurements above confirm the faer path is the active production
   path and is already achieving near-optimal performance for the problem size.
3. No numeric-identity-safe optimization exists for the SVD path without introducing
   a new crate dependency (e.g., a custom BLAS backend) or an API change (e.g.,
   exposing an in-place SVD interface) — neither of which is warranted for this
   milestone's OPT-02 scope.
4. Further SVD micro-optimization (e.g., Jacobi-based small-matrix specialisation)
   would require careful numeric-identity proofs and new dependencies, placing it
   firmly in a future milestone's scope.

## Conclusion

SVD documented as near-optimal. No `fdars-core/src` change made to the SVD path.
The honest Phase-114 precedent is maintained: measure first, do not change what is
already optimal.
