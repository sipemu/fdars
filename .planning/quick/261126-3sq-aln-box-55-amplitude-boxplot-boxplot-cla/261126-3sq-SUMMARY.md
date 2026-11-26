---
quick_id: 261126-3sq
status: complete
issue: 55
commits:
  - 84d513ce feat(quick-261126-3sq): elastic amplitude/phase boxplots with classify, elastic DP-means (closes #55)
---
# Summary — ALN-BOX (#55)

## Public API added (all additive)
- `alignment::elastic_boxplot` (new): `ElasticBoxplotConfig {alpha=0.05, k_a=1.0}`,
  `BoxplotRegion {Central50, CentralAlpha, WithinWhiskers, Outlier}` (+`level()` = fdasrsf 1..4),
  `BoxplotSide {Q1, Q3}`, `BoxplotClassification`, `AmplitudeBoxplot` + `classify_srsfs`,
  `ElasticPhaseBoxplot` + `classify_warps`, `amplitude_boxplot_from_aligned`,
  `elastic_phase_boxplot`, `elastic_boxplots` → `ElasticBoxplots` + `classify(new_data)`
  (aligns new functions to the Karcher-median template, then classifies amplitude + phase),
  `ElasticBoxplotClassification`.
- `PhaseBoxplot` (MBD): new fields `fence_lower`/`fence_upper` + `classify_warps`
  (Outlier / Central50 / WithinWhiskers; training warps flagged exactly as `outlier_indices`).
  Existing outputs unchanged (fences were already computed; now also stored).
- `alignment::elastic_dpmeans(data, argvals, &DpMeansConfig)` → `DpMeansResult`
  (`DpMeansConfig {penalty: Option<f64>=None → seeded k-means++ heuristic, lambda=0, max_iter=20, tol=0.01, align=true, seed=42}`).
- Re-exported from `alignment`, crate root, and prelude (result types).

## Validation
- Golden (fdasrsf 2.7.5, `golden_boxplots.py` / `.out`): ampbox/phbox on a mocked fdawarp
  with analytic inputs (no DP): per-curve distances, r50, ralpha, whisker distance,
  Q1/Q3/Q1a/Q3a indices, outliers, min/max extreme indices, phase median warp; classify
  distances/levels/sides for 4 new functions covering levels 1–4. Tolerance 1e-12 (amp), 1e-10 (phase).
- Golden (fdasrsf 2.7.5 `dpmeans_align`, `golden_dpmeans.py` / `.out`): K and labels match
  exactly for λ ∈ {0, 0.05, 0.2, 1, 5, 100}; objectives within a documented 10% (fdasrsf SRSF
  via spline derivative vs fdars finite differences; observed 1–6%).
- Invariants: training classification == fit (distances, outliers, ceil(n/2) in 50% region),
  nested r50 ≤ ralpha ≤/whisker, quartiles within r50 and on opposite sides, median classifies
  as deepest/Central50, Weiszfeld stationarity of the phase median, whisker monotone in k_a,
  DP-means: K=1 for huge λ, K=n for λ=0, K non-increasing in λ, objective == independent
  recomputation (no-align single pass), monotone warps, seeded determinism; input validation.

## Gates
- `cargo fmt --all` ✓
- `cargo test -p fdars-core --lib -- elastic_boxplot dpmeans phase_boxplot`: 31 passed
- `cargo test -p fdars-core --doc -- elastic_boxplot dpmeans phase_boxplot`: 6 passed
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean
- `cargo check -p fdars-core --no-default-features --lib`: clean
- pmat cognitive max: elastic_boxplot 15, dpmeans 14, phase_boxplot 17 (≤ 23)
- No serde derives added (siblings have none), so no serde check.

## Deviations
- Phase median Weiszfeld: a warp coinciding with the current median (distance 0) is skipped
  in that step instead of fdasrsf's division by zero (NaN). Zero-norm quartile directions get
  angle 0 instead of NaN. Neither occurs on generic data / goldens.
- Phase box plot numerics deliberately follow fdasrsf (numpy gradient, cumulative trapezoid,
  atan2 log map) rather than fdars' 5-point gradient/Simpson helpers, to match goldens.
- `ElasticBoxplots` uses fdars `karcher_median` (Weiszfeld) as the median alignment; it is
  not numerically identical to fdasrsf `srsf_align(method="median")`, so end-to-end
  `elastic_boxplots` is tested by invariants, not goldens.
- DP-means k-means++ heuristic uses `StdRng` (seeded), so the automatically chosen λ differs
  from fdasrsf's numpy RNG draw.
- Single code commit (mod.rs/lib.rs re-exports are shared by both features).

## Remaining
None for #55.
