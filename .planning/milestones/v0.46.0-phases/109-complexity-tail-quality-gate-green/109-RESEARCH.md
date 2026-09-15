---
phase: "109"
slug: "complexity-tail-quality-gate-green"
status: complete
confidence: HIGH
---

# Phase 109 Research — Complexity Tail

> Compiled inline by the orchestrator from live `pmat quality-gate --checks complexity`
> (pmat 3.18.0). The complete, authoritative violation list is committed at
> `109-complexity-violations.txt` (97 lines, one per violation).

## The target

`pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation`
must report **0**. SATD/dead-code/duplicates are already 0 (Phase 108 + baseline);
this phase closes the **97 complexity violations across 56 files**.

**Threshold (STRICTER than `pmat analyze complexity`):** the gate flags
**cognitive > 23** ("recommended") and > 25 ("maximum"); both count. Cyclomatic
uses its own gate threshold. **Internal target: cognitive ≤ 23, cyclomatic ≤ 30
for every function AND every newly-extracted helper.** (pmat counts repeated
structural patterns inside helpers too — collapse a helper's own repetition if it
self-flags; use the proven `_validate_dims` + `_validate_config` split.)

## Scope by file (violation counts)

Full list in `109-complexity-violations.txt`. Distribution:
- 5: `peer.rs`, `clustering_advanced.rs`
- 4: `helpers.rs`, `coclustering.rs`
- 3: `scalar_on_function/multi.rs`, `scalar_on_function/additive.rs`, `fpca_variants.rs`, `fof_regression.rs`, `famm.rs`, `alignment/nd.rs`, `alignment/clustering.rs`
- 2: `smooth_basis.rs`, `metric/soft_dtw.rs`, `gmm/subspace.rs`, `fts/spectral.rs`, `depth/dispatch.rs`, `density_fda.rs`, `conformal/cv.rs`, `boosting_regression/gamlss.rs`, `boosting_regression/boost_fosr.rs`, `alignment/generative.rs`, `alignment/fpns.rs`
- 1: ~35 more files (simulation, shapelet/*, scalar_on_function/fregre_lm, regression, pda, pace_fpca [pace_fpca_validate_config, the 106 residual], outliers, optimal_design, kshape, kernel_kmeans, irreg_fdata/face, inference/flm, inference/anova, frechet/regression, fem_smoothing, explain_generic/*, alignment/bayesian/closed, etc.)

**2 residual from prior phases (must finish to cog ≤23):** `funfem_cluster`
(clustering_advanced.rs, cog 25) and `pace_fpca_validate_config` (pace_fpca.rs, cog 24).

## Score distribution (difficulty)

cog 24–35: the majority (~55 functions) — usually ONE seam extraction (validation
block or one inner loop) drops them under 23. cog 38–74: ~15 heavier functions
(e.g. hierarchical_from_distances cog 53, plus several 50–74) — need multi-seam
decomposition like the 106 hotspots.

## Numeric / RNG hazard map (verify per-function; preserve exactly)

Same discipline as 106/107. Watch especially:
- **Seeded RNG draw order:** clustering functions (`alignment/clustering.rs`
  kmeans_pp_init/kmedoids/hierarchical, `kernel_kmeans.rs`, `kshape.rs`,
  `coclustering.rs`, `gmm/subspace.rs`), `simulation.rs`, any bootstrap/CV
  (`conformal/cv.rs`) — pass `&mut StdRng` through the same call sequence; no
  reorder.
- **Sequential recurrences / iterative EM:** `gmm/subspace.rs`, `famm.rs`,
  boosting (`gamlss.rs`, `boost_fosr.rs`), `frechet/regression.rs`.
- **FP-accumulation order:** every extracted loop body must be verbatim; no
  reordering of sums/folds (esp. `helpers.rs`, `metric/soft_dtw.rs`,
  `covariance`-style code).
- **Alignment/elastic** (`alignment/nd.rs`, `generative.rs`, `fpns.rs`, `closed.rs`,
  `bayesian.rs`): dynamic-programming / gradient loops — preserve order.

## Recommended plan grouping

~12–16 plans, grouped by module/file proximity, **zero file-overlap** between
plans so they're independent. Each plan ~3–6 functions across a few adjacent
files. Put the heaviest concentrations (peer.rs, clustering_advanced.rs,
coclustering.rs, helpers.rs, alignment/clustering.rs) in their own plans. First
plan = a small TRACER (one file, moderate scores) proving the loop, then the rest.
Avoid one-giant-plan (executor-stall risk) and 97-thin-plans.

## Validation Architecture

Behavior-preservation proven by the existing suite staying green (numeric-identical);
complexity-drop proven by `pmat analyze complexity` per touched file AND the final
`pmat quality-gate --checks complexity,satd,dead-code,duplicates` → 0.

## Verify commands
- Per-file: `pmat analyze complexity` (confirm the function + its helpers ≤23/≤30).
- Phase gate: `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` → 0.
- Suite: `cargo test -p fdars-core --features linalg,parallel`,
  `cargo clippy --all-targets --features linalg,parallel -- -D warnings`,
  `cargo fmt --check`.
