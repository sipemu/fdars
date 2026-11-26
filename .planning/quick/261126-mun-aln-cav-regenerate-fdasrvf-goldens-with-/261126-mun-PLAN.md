---
quick_id: 261126-mun
issue: 74
---
# Plan: regenerate fdasrvf golden values with >= 2.5.1 (ALN-CAV, GH #74)

**Goal:** every golden test generated with R fdasrvf 2.5.0 is re-run against CRAN fdasrvf 2.6.0
(`~/.cache/R-lib`), literals/comments updated, numeric changes explained, tolerances tightened
where agreement improved (never loosened).

**Inventory (grep `2.5.0` / fdasrvf in tests + `.planning/quick/*/golden*.R`):**
- #54 `curve_shape/downstream_tests.rs`: `golden_depth_fdasrvf` (curve_depth), `golden_boxplot_fdasrvf_curvebox_data`.
- #56 `srsf_smoothing.rs`: `box_filter_matches_fdasrvf_smooth_data`, `smoothing_spline_denoises_like_fdasrvf`
  (2.5.0 reproduced the 2.5.1 source by hand; now call real `f_to_srvf(smooth = TRUE)`).
- #48 used fdasrsf 2.7.5 (Python, meets the >= 2.7.4 pin) — no fdasrvf goldens, nothing to regenerate.
- #55 boxplots/dpmeans: fdasrsf 2.7.5 (Python) — not affected. #71 already on 2.6.0.

**Tasks:** copy scripts as `*_260.R` (prepend `.libPaths`), run, diff vs 2.5.0 output; measure fdars
residuals; update literals, comments, bounds; add a pointwise smoothed-SRSF comparison; CHANGELOG;
gates (fmt, targeted lib/doc tests, clippy --all-targets).
