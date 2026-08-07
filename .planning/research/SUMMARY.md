# fdars AUDIT Milestone — Research Summary

**Project:** fdars (Functional Data Analysis for Rust)  
**Milestone:** AUDIT (v0.14.0)  
**Domain:** Performance auditing + feature-parity gap analysis (Rust library vs Python reference)  
**Researched:** 2026-08-07  
**Confidence:** HIGH (direct codebase analysis + official documentation)

---

## Executive Summary

The fdars AUDIT milestone has two intertwined deliverables: (1) a performance audit characterizing hot paths across the codebase with concrete measurements and (2) a feature-parity gap analysis against scikit-fda 0.10.1 to prioritize capability gaps by user value. The audit must be conducted with disciplined methodology to avoid common pitfalls (debug-mode benchmarking, API-name counting, missing accuracy parity checks) that can invalidate findings.

**Recommended approach:** Conduct the audit in phases that separate concerns: static analysis of hot paths first (which functions scale badly and why), then benchmark confirmation (measure wall-clock and instruction counts with proper feature-flag control), then allocation profiling (isolate copy vs compute overhead), and finally gap analysis with a capability taxonomy (not name counting). This ordering minimizes wasted work and ensures early findings are validated before complex measurements begin.

**Key risks and mitigations:** The codebase has documented performance anti-patterns (FdMatrix→DMatrix round-trip copies, O(N²·M²) elastic alignment without auto-banding, sequential CV folds), but naïve benchmarking in debug mode or without large-input variants can hide these. The gap analysis risks inflating backlog with out-of-scope features (plotting, sklearn API) unless a design-goal filter is applied first. The recovery cost for these mistakes is MEDIUM-to-HIGH (re-running benchmarks, re-triaging gaps), so prevention is critical.

---

## Key Findings

### Recommended Audit Stack

The performance audit toolkit is mature and well-understood. All components are already compatible with fdars' ecosystem (Rust 1.81+, Linux dev environment).

**Core measurement tools (must use):**
- **criterion 0.5.1** — Already in repo; statistical wall-clock benchmarking with throughput curves. Add large-input variants and proper black_box wrapping.
- **cargo-flamegraph 0.6.13** — Zero-code-change CPU sampling profiler. Fastest way to identify top 3 hot functions.
- **iai-callgrind 0.16.1** — Deterministic instruction-count CI regression tracking for top 5 hot functions.
- **dhat-rs 0.3.3** — In-process heap allocation profiling. Essential for confirming FdMatrix→DMatrix copy cost.

**Measurement discipline:**
- Always run `cargo bench --features linalg`. Also run with `RAYON_NUM_THREADS=1` to isolate parallelism benefit.
- Benchmark with large inputs (n=200–1000, m=100–500). Small inputs mask O(n²) and O(m³) scaling.
- Run each benchmark twice, confirm within ±5% variance. Use `--save-baseline` for before/after comparison.

### Expected Gaps (Feature Parity Analysis)

**Highest-impact gaps (expected by users migrating from scikit-fda):**

1. **Smoothing module** — Nadaraya-Watson, basis smoothing with CV. Single highest-impact missing area.
2. **Public Lp norm / distance + pairwise matrix** — Required by kNN, k-means, agglomerative clustering.
3. **Functional k-means** — Most-used clustering; fdars has only GMM.
4. **FPLS + FPLSRegression** — Companion to FPCA; expected by R/Python users.
5. **Shift + landmark registration** — Simpler alternatives to elastic; users want cheaper method first.
6. **kNN + Kernel regression** — Basic baselines expected alongside linear model.
7. **Statistical inference (ANOVA, Hotelling T²)** — No hypothesis testing at all.

**Out-of-scope (deliberately not ported):**
- Plotting / visualization — fdars is numeric only
- sklearn pipeline API — Rust trait composition is more idiomatic
- DataFrame IO, pandas integration

### Architecture: Performance Hot-Spot Map

**Tier 1: Elastic alignment (O(N²·M²) or O(N²·M·r) banded)**
- `src/alignment/pairwise.rs` + `karcher_mean()`: Pairwise DP warping
- Anti-pattern: `karcher_mean()` defaults to `band = None`, full O(M²) per pair. For M=200, N=100, K=30: 120 billion ops without band; 7× reduction with 15% band.

**Tier 2: FPCA / SVD round-trip copies (O(N·M) allocation per call)**
- `src/regression.rs:fdata_to_pc_1d()`: Calls `centered.clone()` then `to_dmatrix()` — two N×M copies before SVD.
- `src/elastic_fpca.rs`: Pattern repeated 7 times. At N=500, M=200: 800 KB per call; 5.6 MB for one elastic-FPCA.

**Tier 3: Parallelism gaps (easy wins)**
- `src/classification/cv.rs:fclassif_cv()`: Fold loop sequential; folds independent.
- `src/elastic_fpca.rs`: Inner O(N) loops sequential.
- `src/streaming_depth/`: Batch queries sequential but independent.

### Critical Pitfalls to Avoid

1. **Debug mode benchmarking** — 5–50× slower; hidden in binary path. First step: confirm `/release/`.
2. **Missing large-input variants** — n ≤ 20 masks O(n²) scaling. Add variants at n=200–1000.
3. **API-name counting** — Gap count inflates 2–3×. Build capability matrix, not name list.
4. **Ignoring allocation vs compute** — SVD slow because copy dominates. Run dhat on hot paths.
5. **Feature-flag confusion** — Ridge benchmark unmarked for `--features linalg` gives wrong conclusion.
6. **Linker flakiness** — Bus errors in test harness are infrastructure, not code failures.
7. **Vague backlog items** — "Improve X" is not actionable. Every item needs function, cost, cause, fix, severity, effort.
8. **Treating scikit-fda as gospel** — "scikit-fda has X" ≠ "fdars must have X". Apply design-goal filter.

---

## Implications for Roadmap

### Phase 1: Static Hot-Path Analysis
**Rationale:** Define priority list with zero cost.  
**Delivers:** O(N²)/O(M²) operations list, allocation hotspots, parallelism gaps.  
**Research phase?** No.

### Phase 2: Benchmark Confirmation (Large Inputs + Feature Matrix)
**Rationale:** Validate hot paths are hot at production scale.  
**Delivers:** Criterion results with feature column, flamegraphs, large-input variants, baseline.  
**Discipline:** `cargo bench --features linalg`, `RAYON_NUM_THREADS=1`, black_box, ±5% variance.  
**Research phase?** No.

### Phase 3: Allocation Audit (dhat-rs Profiling)
**Rationale:** Separate allocation from compute cost for top-3 paths.  
**Delivers:** dhat profiles, allocation hotspot ranking, allocation % of wall-clock.  
**Research phase?** No.

### Phase 4: nalgebra vs faer Comparison (Optional)
**Rationale:** If Phase 2 shows SVD > 30% and Phase 3 shows copy not bottleneck, benchmark faer.  
**Delivers:** Faer speedup, conversion cost, crossover point, integration ROI.  
**Scope:** Only if triggered by Phase 2. Else skip.  
**Research phase?** No.

### Phase 5: Gap Analysis (Feature-Parity + Accuracy Verification)
**Rationale:** Map scikit-fda 0.10.1 against fdars capabilities with design-goal filter.  
**Delivers:** Capability matrix, relevance filter applied, accuracy verification, complexity tags.  
**Research phase?** Optional — check if `.planning/codebase/STRUCTURE.md` exists; if not, 2–3 day research phase needed.

### Phase 6: Prioritized Backlog (Final Report)
**Rationale:** Convert findings to GSD-ready backlog, ranked by value.  
**Delivers:** Top-10 perf optimizations + top-10 gap items, all with function/cost/cause/fix/severity/effort, evidence artifacts.  
**Value framework:** P1 (blocks real use), P2 (impairs workload), P3 (nice to have). Effort: S/M/L. Rank by `value / sqrt(effort)`.  
**Research phase?** No.

### Phase Ordering Rationale

1. Static analysis first (costs nothing; defines priority)
2. Benchmark second (validates static; measurement discipline critical)
3. Allocation audit third (only profiles candidates; avoids profiling everything)
4. faer comparison optional (only if triggered; no speculative optimization)
5. Gap analysis after perf understood (roadmapper needs context on perf urgency)
6. Backlog last (aggregates findings; no rework if earlier phases refined priority)

### Research Flags

- **Phase 1–4, 6:** No research phase.
- **Phase 5:** Conditional research phase if API reconciliation incomplete. Check `.planning/codebase/STRUCTURE.md` first.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Tools verified against official docs (docs.rs). Criterion 0.5, iai-callgrind 0.16.1, dhat 0.3.3 are stable. Known linker issue documented. |
| Architecture | HIGH | Hot-spot map from direct fdars source analysis. Anti-patterns verified in code. Complexity follows Golub/Van Loan, Srivastava/Klassen, López-Pintado. |
| Features | MEDIUM | scikit-fda v0.10.1 API verified against official docs. fdars coverage from codebase map. Gap interpretation depends on "equivalent" definition — apply capability filter carefully. |
| Pitfalls | HIGH | Derived from Rust benchmarking best practices (Criterion book, Rust Performance Book) + fdars-specific issues (criterion linker, feature-flag matrix). Pitfalls 1,2,5,9,18 in existing CONCERNS.md. |

**Overall:** HIGH — Well-grounded in verified sources (official docs, code analysis, published papers).

### Gaps to Address

1. **Numerical accuracy parity** — Audit measures performance/counts features but cannot verify numerical accuracy. CONCERNS.md flags B-spline CV, elastic alignment as fragile. Phase 5 must include spot-check against scikit-fda on reference datasets.

2. **Real-world workload characterization** — Assumes n=200–1000, m=100–500. Not validated against actual fdars users. If production workloads differ, priority may shift. Phase 2 documents assumptions; post-audit collect telemetry.

3. **faer stability** — Recommend faer 0.23 as SVD alternative, but faer younger than nalgebra. Adoption risk not quantified. Phase 4 (if triggered) must include maintenance-burden assessment.

4. **Rayon overhead** — Audit measures parallelism benefit via RAYON_NUM_THREADS sweep but not overhead (pool spinup, work-stealing). Phase 2 must include threshold analysis: at what n does overhead get paid back?

---

## Sources

### Primary (HIGH confidence)
- criterion 0.5.1 docs (docs.rs, June 2026)
- iai-callgrind 0.16.1 docs (docs.rs, July 2025)
- dhat-rs 0.3.3 docs (docs.rs, June 2026)
- flamegraph 0.6.13 (crates.io, June 2026)
- Rust Performance Book (nnethercote.github.io/perf-book)
- Direct fdars-core source analysis (2026-08-07)
- Golub & Van Loan, "Matrix Computations" (4th ed.)
- Srivastava & Klassen, "Functional and Shape Data Analysis"

### Secondary (MEDIUM confidence)
- scikit-fda 0.10.1 API reference (fda.readthedocs.io)
- scikit-fda GitHub releases
- Gendignoux 2024, "Optimizing Rayon workloads"
- López-Pintado & Romo (2009), Band Depth complexity
- Sakoe & Chiba (1978), DP band constraint theory

### Tertiary (VALIDATION NEEDED)
- Real-world fdars usage patterns (n/m ranges)
- faer 0.23 stability / maintenance
- Criterion 0.5 linker issue frequency (environment-specific)

---

## Ready for Roadmap

Audit research complete. Four detailed research documents (STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md) provide foundations; this SUMMARY distills key findings and phase structure.

**Roadmapper should:**
1. Use Phase 1–6 as suggested decomposition (tailor per priorities)
2. Flag Phase 5 for conditional research phase if codebase mapping incomplete
3. Apply design-goal filter (out-of-scope: plotting, sklearn API) during gap prioritization
4. Use value-based ranking framework (value / sqrt(effort)) for backlog ordering

**Expected audit timeline:** 4–6 weeks (static analysis + benchmarking + profiling + gap analysis, sequential or parallel per team capacity).

---

*Research completed: 2026-08-07*  
*Synthesized by: GSD Research Synthesizer*  
*Ready for roadmap: YES*
