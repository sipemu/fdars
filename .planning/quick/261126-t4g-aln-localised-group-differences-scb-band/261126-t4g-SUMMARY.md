---
status: complete
---
# Quick 261126-t4g SUMMARY — ALN localised group differences (GH #76)

## Commits
- 222ac2e2 feat(quick-261126-t4g): localise amplitude/phase group differences (closes #76)

## Public API added (additive)
- `fdars_core::regions` (new): `DifferenceRegion`, `difference_regions`; `significant_regions`,
  `significant_regions_from_se`, `SignificantRegion`, `SignificanceDirection` moved here and
  re-exported unchanged from `explain` (now also serde-gated derives). Root: `DifferenceRegion`,
  `difference_regions`.
- `inference::scb_two_sample_band` -> `ScbTwoSampleBand { estimate, lower, upper, half_width,
  critical_value, confidence, statistic, rejected, regions }`.
- `inference::itp_pointwise_pvalues(&ItpResult, argvals, ItpDomainMapping::{Dominant, Support})`,
  `inference::itp_two_pop_regions(a, b, argvals, &ItpRegionConfig) -> ItpRegionsResult`.
- `alignment::elastic_local_difference_test(data, argvals, labels, &LocalDifferenceConfig)
  -> LocalDifferenceResult { amplitude, phase, phase_data, alignment, group_labels, group_sizes }`;
  `LocalTestMethod::{Itp (default), Scb}`, `PhaseRepresentation::{Displacement (default), Psi}`,
  `LocalDifferenceTest::timing -> Timing::{ALater, AEarlier}`.

## ITP domain mapping (documented)
fdatest reports per-component `corrected.pval`; `plot.ITP2` shades for each significant
component a rectangle of width (b−a)/(p−1) at the k-th of p evenly spaced positions (nearest
component). fdars's B-spline basis has extended knots (component centres at (k−2)h, two outer
functions vanish on the grid), so `Dominant` uses the nearest-centre principle on the actual
basis (argmax_k B_k(t)); `Support` is the conservative max over supported components.
Fourier ITP cannot be localised in time (InvalidParameter).

## Validation
- R fdatest 2.1.1 (installed into ~/.cache/R-lib): `golden_itp2bspline.R` replicates
  ITP2bspline's permutation loop (set.seed(7), nknots=5, B=40, p=7) to export T_coeff; fdars's
  rank transform + interval matrix + pval_correct reproduce `pval.matrix` (49 entries) and
  `corrected.pval` = [0.75, 0.5, 0.1, 0.025, 0, 0, 0.225] within 1e-12.
- SCB: `scb_two_sample_test` bit-identical to a verbatim legacy copy (4 data sets × 2 multipliers);
  band statistic/decision == test; band == `mean_scb` of the paired difference; bump localised.
- Synthetic (seeded, 20 curves/group, m=60, ITP, n_perm=999): amplitude bump on the 2nd peak ->
  amplitude region covering 0.68 (A<B), no phase region; localised delay of the 2nd peak (δ=0.1)
  -> phase region covering 0.68 with Timing::AEarlier, no amplitude region. SCB option and ψ
  representation also localise; alignment invariant to label permutation.
- Null family-wise rate (any region flagged), α=0.05, 200 seeded reps (#[ignore] test):
  30/group: ITP 0/200 amplitude, 0/200 phase; SCB 6/200, 5/200. 10/group: ITP 1/200, 2/200;
  SCB 35/200, 38/200 (Degras multiplier bootstrap is liberal for small n — documented on
  `LocalTestMethod::Scb`). Short 20-rep null test runs in the normal suite.

## Deviations / intentional numeric change
- Bug fix in `itp_two_pop`: the permutation null relabelled only group B (group A kept its
  observed rows), destroying power. Now both groups are relabelled (fdatest `coeff[sample(n),]`).
- `itp_two_pop` excludes the B-spline components whose basis vanishes on the grid from the
  closure (reported raw/adjusted 1.0); they diluted every interval. `itp_one_pop` / `itp_flm`
  unchanged (same vanishing-component issue exists there; follow-up candidate).
- fdars ITP is markedly more conservative than fdatest on strong localised effects because
  fdatest's `n_ge/B` gives raw p = 0 and Fisher −2·log 0 = ∞ auto-rejects every interval
  containing that component; fdars keeps valid (n_ge+1)/(B+1). Documented on `itp_two_pop`.
  Tests therefore use effects spanning several basis supports and ≥20–25 curves per group.
- ITP localisation resolution = basis support width (amplitude test allows 0.1 slack).
- SCB in the elastic test uses `scb_two_sample_band`'s paired construction (first min(n_A,n_B)
  curves per group); grid points where the tested quantity is constant across curves
  (warp end points) are never flagged.
- Single code commit (lib.rs / mod.rs re-exports span all three parts).

## Gates
- cargo fmt; lib tests (alignment::local_difference, inference::, regions, explain, tolerance,
  phase_amplitude): 408 passed, 1 ignored; same filters with `--no-default-features --features
  linalg`: 400 passed; doc tests for the new items: 6 passed; long null study passed (--ignored).
- clippy `--all-targets --features linalg,parallel,serde -D warnings` and `linalg,parallel`: clean.
- `RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --features linalg,parallel,serde`: clean.
- `pmat quality-gate --checks complexity,satd,dead-code,duplicates` (3.45.1): 0 violations.
