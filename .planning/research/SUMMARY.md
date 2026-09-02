# Research Summary: Shapelet Transform & Classification (v0.33.0)

**Project:** fdars-core (Rust functional-data-analysis library)
**Milestone:** v0.33.0 — Shapelet Transform & Bundled Classifier (GAP-02)
**Researched:** 2026-09-02
**Confidence:** HIGH

---

## Executive Summary

v0.33.0 implements discovery-based shapelet transform with a bundled classifier — a proven time-series feature engineering technique that converts functional curves into discriminative distance-feature matrices suitable for standard classifiers. The approach is non-breaking (new `src/shapelet/` submodule only), requires **zero new external dependencies**, builds entirely on existing fdars primitives, and follows a strict 4-phase implementation sequence (compile-time dependency chain).

**Core recommendation:** Implement phases 57–60 sequentially in order; no reordering or parallelization is possible. The algorithm is well-established (Ye & Keogh 2009; Hills/Lines 2014); the primary risk is per-window z-normalization correctness — this must be verified early with offset/scale-invariance tests.

**Estimated effort:** L (Low) for a mature codebase, reflecting that classification and distance infrastructure already exist. The novelty is discovery machinery and transform logic (~800 LOC across four files), not building new classification from scratch.

---

## Key Findings

### Recommended Stack

**No new dependencies required.** All deliverables build on existing fdars stack: Rust 1.81+ (MSRV), nalgebra/rayon/rand (existing), FdMatrix, classification module. Z-normalization, sliding-window distance, candidate generation, and information-gain scoring are all pure `f64` arithmetic over slices. MSRV remains 1.81; implementation is data-efficient.

### Expected Features

**Table stakes (must ship v0.33.0):**
- Per-window z-normalization helper
- Sliding-window minimum z-normalized Euclidean distance (sdist core primitive)
- Candidate generation (exhaustive + contracted modes)
- Information gain quality measure with optimal split threshold search
- Self-similarity pruning
- ShapeletTransformFit result struct for out-of-sample use
- shapelet_transform_fit + shapelet_transform APIs
- ShapeletTransformClassifier end-to-end pipeline
- ShapeletConfig struct with standard fdars conventions
- Result<T, FdarError> error handling throughout

**Differentiators (v0.33.x, not blockers):**
- Early-abandon optimization (2–8× speedup; critical for large datasets)
- F-statistic quality measure (better for class imbalance)
- Rayon parallelism over candidates
- Serde persistence
- Configurable inner classifier
- predict_proba posteriors

**Out of scope (deferred):**
- Learning-shapelets (requires AD; GAP-08)
- GPU acceleration
- Multivariate shapelet transform
- DTW-based distance

### Architecture Approach

**New `src/shapelet/` submodule with four strictly-ordered files:**
1. **distance.rs** — z_normalize_window, shapelet_distance, Shapelet struct
2. **discovery.rs** — ShapeletConfig, candidate enumeration, IG/F-stat scoring, self-similarity pruning
3. **transform.rs** — ShapeletTransformFit, fit/predict paths
4. **classifier.rs** — bundled STC, calls fclassif_lda_fit

**Non-breaking:** Zero changes to existing modules. Shapelets are a feature-engineering consumer of classification, not an extension. FdMatrix is primary data carrier; column-major layout respected throughout.

**Compile-time dependency chain (cannot reorder):**
Phase 57 → Phase 58 → Phase 59 → Phase 60. Each phase must complete before the next begins.

### Critical Pitfalls (Top 5)

1. **Per-window z-normalization (Phase 57 — CRITICAL CORRECTNESS)**
   - Risk: Series-level normalization destroys shift/scale invariance
   - Mitigation: z_normalize_window operates on window slices only; offset/scale-invariance tests verify
   - Verification: shapelet_distance(s, x) == shapelet_distance(s, x+constant) == shapelet_distance(s, x*scale)

2. **Division by near-zero standard deviation (Phase 57 — CRITICAL STABILITY)**
   - Risk: Constant windows cause division by ~0, producing NaN/Inf
   - Mitigation: Clamp std to 1e-10 minimum
   - Verification: Constant-window and near-constant tests must return finite results

3. **Combinatorial blowup of candidates (Phase 58 — CRITICAL TRACTABILITY)**
   - Risk: Naive enumeration is O(n·m²/2); intractable for real datasets
   - Mitigation: max_candidates parameter (default 10,000) caps search via random sampling
   - Verification: n=100, m=200, max_candidates=1000 returns in <10 seconds

4. **Shapelet distance is minimum, not mean (Phase 57 — CRITICAL SEMANTICS)**
   - Risk: Mean distance makes score depend on frequency, not presence
   - Mitigation: Use strict min operator; known-motif recovery test
   - Verification: Shapelet matching once in noise achieves sdist ≈ 0

5. **Self-similarity pruning omission (Phase 58 — CRITICAL FEATURE QUALITY)**
   - Risk: Top-K filled with shifted variants of same subsequence
   - Mitigation: Greedy selection; skip overlapping candidates from same series
   - Verification: max(correlation between columns) < 0.95; distinct source series check

---

## Implications for Roadmap

### Phase 57: Shapelet Distance Core
**Rationale:** Atomic primitive; all downstream depends on it. Establish correctness of z-norm and distance before discovery.

### Phase 58: Discovery & Ranking
**Rationale:** Builds on Phase 57. Introduces search strategy, quality scoring, and selection logic.

### Phase 59: Shapelet Transform
**Rationale:** Takes discovered shapelets; produces n×K feature matrix for training and out-of-sample.

### Phase 60: Bundled ShapeletTransformClassifier
**Rationale:** End-to-end pipeline: discover → transform → classify. User-facing API.

**Phase Ordering:** Strict compile-time dependency chain; no reordering or parallelization possible. Grouping reflects algorithm structure (Distance → Discovery → Transform → Classifier).

### OPEN DECISIONS FOR ROADMAPPER/PLANNER

1. **Bundled classifier default** — kNN vs. LDA (Phase 60)
   - Recommendation: kNN (canonical from literature; avoids FPCA-on-features weirdness)
   - Action: Phase 60 planner picks k=1 or k=3; expose config enum for alternatives

2. **Quality measure** — InfoGain-only vs. InfoGain + F-statistic (Phase 58)
   - Recommendation: Ship InfoGain as default; expose ShapeletConfig.scorer enum for F-statistic
   - Action: Phase 58 planner implements InfoGain fully; F-stat is optional follow-on

3. **Default configuration values** (Phase 58/60)
   - min_len=3, max_len=m/2 (runtime resolve), n_shapelets=min(10*sqrt(n),1000), max_candidates=10_000, similarity_threshold=0.95, seed=42

4. **Early-abandon API design** (Phase 57)
   - Recommendation: Explicit best_so_far: Option<f64> parameter (testable, composable)
   - Action: Phase 57 planner includes in signature

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new deps; all primitives verified in codebase. Non-invasive reuse. |
| Features | HIGH | Algorithms peer-reviewed; spec mathematically precise. |
| Architecture | HIGH | Submodule mirrors existing patterns. Four-phase chain is rigid. Non-breaking. |
| Pitfalls | HIGH | 13 pitfalls sourced from literature + fdars constraints. Each has mitigation + verification hook. |
| **Overall** | **HIGH** | Comprehensive, data-driven, grounded in peer-reviewed sources. Algorithm well-established. |

### Gaps to Address

1. **F-statistic adaptation:** Phase 58 planner must verify integrated_f_statistic (full curves) adapts to 2-group scalar case or needs new implementation.

2. **Contracted search determinism on distributed machines:** Current seeding assumes single-process. Document assumption; flag for future distributed work.

3. **Multivariate scope:** v0.33.0 is 1D curves only. Explicitly document; defer multivariate design to v0.34+.

4. **Elastic distance choice:** Implementation uses z-normalized Euclidean. Document choice; note elastic alignment as preprocessing alternative.

---

## Sources

**Primary (HIGH confidence):**
- STACK.md — Dependency audit (direct codebase read)
- ARCHITECTURE.md — Integration design (direct codebase read)
- PITFALLS.md — 13 pitfalls from Hills et al. 2014, Bagnall et al. 2017, source code

**Reference (MEDIUM–HIGH):**
- sktime ShapeletTransformClassifier + RandomShapeletTransform (v0.30.0+)
- pyts 0.13.x ShapeletTransform

**Literature (HIGH):**
- Ye & Keogh (2009). KDD 2009. Original shapelet distance definition.
- Hills, Lines, Baranauskas, Mapp, Bagnall (2014). DMKD 28(4). Discovery, IG, self-similarity, time contract.
- Lines & Bagnall (2012). IDEAL 2012. Alternative quality measures.
- Bagnall et al. (2017). DMKD 31(3). Early-abandon importance; speed/accuracy trade-offs.

---

*Research synthesis completed: 2026-09-02*
*Ready for roadmap generation: YES*
