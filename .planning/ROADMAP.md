# Roadmap: fdars v0.47.0 — Performance: Cross-Language Benchmarks & Optimization Pass

**Milestone:** v0.47.0
**Created:** 2026-09-19
**Granularity:** fine
**Phase numbering:** continues from v0.46.0 (…111) → **Phase 112+** (no reset)

## Overview

A **performance milestone** (not a feature milestone). The deliverable is *evidence* — head-to-head timings of `fdars-core` vs Python **scikit-fda** and the R FDA ecosystem — plus provably-improved implementations of the worst bottlenecks. It is the runtime complement to the v0.14.0 capability audit.

The approach is **measure-first**: stand up an external benchmark harness and establish a fair cross-language baseline (with a mandatory numeric-agreement gate) *before* optimizing anything, then attack the ranked hot paths — elastic alignment O(N²·M²) is the confirmed anchor — proving every optimization numeric-identical via the existing 2942-test suite + before/after criterion, then re-run the harness to publish the gains.

**Hard constraints (honored in every phase):**

- **No new `fdars-core` crate dependency** — the harness is external (workspace-root `benchmarks/`, NOT a Cargo member); optimizations reuse existing infra (`nalgebra`, `faer` behind `linalg`, `rayon` behind `parallel`, criterion 0.5).
- **Strictly non-breaking + numeric-identical** — no public API/signature/behavioral change; every optimization proven behavior-identical by the full test suite + before/after criterion + golden-value checks (≤1e-10 / bit-exact). Protects R + WASM bindings + 28 examples.
- **Measure-first** — a committed before-baseline precedes any optimization.
- **Fairness controls are table stakes** — thread-pinning (`*_NUM_THREADS=1`) + a numeric-agreement gate before any timing is recorded; full version/BLAS/CPU/env capture.
- **REL-01 is terminal** — the `git tag v0.47.0` → crates.io publish is the DEFERRED operator step; never tag/publish inside a phase.

## Phases

- [x] **Phase 112: Benchmark Harness Setup** - External `benchmarks/` harness driving Rust/Python/R on matched inputs, with fairness + numeric-agreement gates (completed 2026-09-22)
- [x] **Phase 113: Cross-Language Baseline & Hotspot Ranking** - Fair baseline across all three ecosystems, merged into `comparison.csv`, plus an evidence-ranked hotspot list (completed 2026-09-23)
- [x] **Phase 114: Elastic Alignment Optimization** - Optimize the anchor O(N²·M²) hot path in `src/alignment/`, numeric-identical, with before/after criterion evidence (completed 2026-09-23)
- [x] **Phase 115: SVD/FPCA & Covariance Optimization** - Optimize the next-ranked hot paths + the consolidated build-variant / numeric-identity / pmat proof (completed 2026-09-23)
- [x] **Phase 116: Cross-Language Re-Run & Published Report** - Re-run the full harness post-optimization; publish the before/after perf report + reproduction instructions (completed 2026-09-25)
- [ ] **Phase 117: Release Preparation & Verification** - Bump 0.46.0 → 0.47.0, CHANGELOG, all whole-crate gates + pmat green (terminal)

## Phase Details

### Phase 112: Benchmark Harness Setup

**Goal**: A reproducible external harness exists that times fdars, scikit-fda, and the R FDA ecosystem on identical inputs, with fairness controls and a numeric-agreement gate wired in before the first measurement.
**Depends on**: Nothing (first phase of this milestone; independent of `fdars-core` internals)
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05
**Success Criteria** (what must be TRUE):

  1. Running the orchestration script generates matched curve-set files from a single seeded generator and every ecosystem runner (Rust/Python/R) consumes the identical shared-format (CSV-default) data — verifiable by a spot-check that the three runners load bit-identical inputs.
  2. The Python runner (scikit-fda 0.10.1, pinned via `requirements.lock`) and the R runner (`fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`fdasrvf`, pinned via `renv.lock`) each execute the overlapping operation set (FPCA, basis smoothing, functional depth, functional boxplot, FPCR regression, elastic alignment/Karcher mean) and emit structured timing JSON.
  3. fdars timings are collected from the existing criterion 0.5 harness by parsing `target/criterion/*/estimates.json` into the shared comparison format — with no new dev-dependency and no `cargo-criterion` binary.
  4. A baseline run pins `OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1` and captures full version + BLAS-backend + CPU + thread-env metadata into `benchmarks/results/env.json`.
  5. A numeric-agreement gate verifies each library computes the same result (sign-aligned FPCA correlation > 0.99; 1e-3 depth/regression, 1e-2 elastic warping) and aborts/flags before any timing is recorded on mismatch.

**Plans**: 3/3 plans executed

- [x] 112-01-PLAN.md — FPCA end-to-end tracer: seeded generator + all-three-runner FPCA slice + criterion collect + agreement gate + env.json (BENCH-01..05 thin path)
- [x] 112-02-PLAN.md — Expand Python + R runners to the full overlapping op set + add functional_boxplot criterion bench (BENCH-02)
- [x] 112-03-PLAN.md — Expand criterion collector + agreement gate to all ops with per-op tolerances + finalize env capture (BENCH-03/04/05)

### Phase 113: Cross-Language Baseline & Hotspot Ranking

**Goal**: A fair, honest cross-language baseline is committed and the optimization targets are ranked by evidence — so Phases 114–115 attack the highest-leverage work first.
**Depends on**: Phase 112 (needs the harness + numeric-agreement gate + env capture to produce a valid baseline)
**Requirements**: BASE-01, BASE-02
**Success Criteria** (what must be TRUE):

  1. A cross-language baseline is established across all three ecosystems over an N×M input grid (≥3 sizes, elastic target at N≥100) and merged into a committed `comparison.csv`.
  2. An evidence-ranked hotspot list exists combining the cross-language results with the criterion suite, ranked by both absolute time and relative-to-baseline leverage.
  3. The ranking is gated on measurement noise (RSD < 5% / CI < ±5%) so no spurious speedup target is chosen; the elastic-alignment anchor is confirmed and the secondary targets (SVD/FPCA, covariance) are ordered.

**Plans**: 2/3 plans executed

- [x] 113-01-PLAN.md — Recalibrate agreement gate (depth/functional_boxplot/fpcr → Pearson corr) + fpca_native_r R branch + populate missing criterion output (tracer slice)
- [x] 113-02-PLAN.md — N×M grid baseline run (gate-before-timing) + assemble committed comparison.csv + re-capture env.json
- [x] 113-03-PLAN.md — Noise-gated leverage ranking + committed hotspot_ranking.md (elastic anchor confirmed, secondary targets ordered) + human fair/honest sign-off

### Phase 114: Elastic Alignment Optimization

**Goal**: The confirmed #1 bottleneck — elastic alignment O(N²·M²) in `src/alignment/` — is measurably faster while producing numerically identical output.
**Depends on**: Phase 113 (ranking confirms the anchor + provides the committed before-baseline reference)
**Requirements**: OPT-01
**Success Criteria** (what must be TRUE):

  1. The elastic-alignment hot path in `src/alignment/` is optimized with committed before/after criterion evidence showing a statistically significant speedup on the target cells.
  2. Output is numeric-identical — the full test suite stays green and a golden-value assertion confirms bit-identical (or ≤1e-10) results before/after.
  3. The change is strictly non-breaking: no public signature change; the optimization lives in internals that compile under the R binding's default-only feature set (or is gated behind `linalg`/`parallel`).

**Plans**: 2/2 plans executed

- [x] 114-01-PLAN.md — Tracer: commit BEFORE-baseline criterion (`audit_p3_elastic_cross_banded/n500_m50`) + golden-value fixture (`equivalence_phase114.rs`, n=20,m=50, ≤1e-10) locking current elastic self-distance output
- [x] 114-02-PLAN.md — `dp_fill_grid` band-iteration refactor (numeric-identical) + prove (golden + release + determinism guardrail + five build variants) + committed AFTER criterion with honest before/after verdict

### Phase 115: SVD/FPCA & Covariance Optimization

**Goal**: The next-ranked hot paths (SVD/FPCA + covariance surface) are optimized with the same numeric-identity proof, and the milestone's cross-cutting build-variant / determinism / pmat guarantees are consolidated and proven green across all optimization work.
**Depends on**: Phase 113 (ranking selects the targets); Phase 114 (shares the measure→optimize→prove pipeline and the accumulated non-breaking proof)
**Requirements**: OPT-02, OPT-03
**Success Criteria** (what must be TRUE):

  1. The next-ranked hot paths (SVD/FPCA and covariance surface, conditional on the Phase 113 ranking) are optimized with committed before/after criterion evidence.
  2. Every optimization commit preserves numeric identity — `cargo test --release --features linalg` is green and the determinism guardrail is green.
  3. All five build variants (default / `linalg` / `serde` / `linalg,serde,parallel` / WASM) compile after the optimizations, and the pmat quality-gate stays green (no new complexity/entropy/SATD/defect violations from decomposition helpers).

**Plans**: 3/3 plans executed

- [x] 115-01-PLAN.md — Capture committed BEFORE-baseline criterion (perf_face_covariance/n200_m30) + golden-value face_covariance fixture (tracer, un-optimized code)
- [x] 115-02-PLAN.md — sym_cov_irreg upper-triangle+mirror optimization (numeric-identical) + AFTER criterion (~2x) + SVD re-benchmark/near-optimal verdict
- [x] 115-03-PLAN.md — OPT-03 consolidated proof battery (release test + determinism guardrail + 5 builds + pmat gate + fmt/clippy), committed

### Phase 116: Cross-Language Re-Run & Published Report

**Goal**: The optimization gains are shown to be visible across all three ecosystems and turned into the milestone's proof artifact — a published head-to-head perf report a reader can reproduce.
**Depends on**: Phases 114 and 115 (report the post-optimization numbers)
**Requirements**: RPT-01
**Success Criteria** (what must be TRUE):

  1. The full cross-language harness is re-run post-optimization and `comparison.csv` is updated with after-optimization numbers.
  2. A published perf report exists with head-to-head before/after tables, N×M scaling analysis, and full methodology disclosure (likely a README speed story).
  3. The report includes reproduction instructions a reader can follow end-to-end to regenerate the numbers from scratch.

**Plans**: 7/7 plans executed (6 waves)
**Wave 1**

- [x] 116-01-PLAN.md — Tracer: FPCA end-to-end at N=500,M=200 (grid-aligned crosslang_grid bench → scikit-fda/R → gate → CSV → generated headline row) + freeze Phase 113 baseline/env + Phase 114/115 A/B evidence + results/ ignore carve-out

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 116-02-PLAN.md — crosslang_grid bench expanded to all six ops (Python/R-matched params, shared CSVs) + OP_MAP labels + functional_boxplot noise-settings bump
- [x] 116-04-PLAN.md — render_report.py: 13 generated blocks (head-to-head, A/B, before/after incl. fdars-internal, not-rerun, N×M scaling vs theory, methodology) + --check/--check-prose/--noise-report/--ab-check (+ tests)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 116-03-PLAN.md — run_all.py full-run plumbing: 67-chunk resumable budgeted scheduler + run log, per-cell timeouts, prebuild, footprint, single `--full` command (+ tests)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 116-05-PLAN.md — Controlled thread-pinned A/B re-confirmation of the face_covariance headline (pre-OPT-02 vs HEAD worktrees, noise-gated, bounded bumps) [INLINE]

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 116-06-PLAN.md — Full 3-ecosystem re-run (fdars 67 chunks + 6-cell scikit-fda/R grid, gate before timing), headline noise remediation, commit after-data comparison.csv [INLINE, ~3-4.5 h]

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 116-07-PLAN.md — Publish documentation/PERFORMANCE.md + README speed story + benchmarks/README + multimethod patch; clean-clone --check

### Phase 117: Release Preparation & Verification

**Goal**: The crate is release-ready at 0.47.0 with every whole-crate gate green; the tag/publish is left as the deferred operator step.
**Depends on**: All prior phases (bump/CHANGELOG/gates are the final proof)
**Requirements**: REL-01
**Success Criteria** (what must be TRUE):

  1. The crate version is bumped 0.46.0 → 0.47.0 and the CHANGELOG is written at both root and crate level.
  2. All whole-crate gates are green — `cargo fmt --check`, `clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build, 28 examples + doctests, `cargo package` — plus the pmat quality-gate.
  3. No tag or publish is performed inside the phase; the `git tag v0.47.0` → crates.io publish is documented as the deferred operator step.

**Plans**: 1/4 plans executed (4 waves)
**Wave 1**

- [x] 117-01-PLAN.md — Tracer: 0.46.0 → 0.47.0 bump + root/crate CHANGELOG `[0.47.0]` (pinned 1.19× only) + install-snippet version bumps + keep pmat caches out of the crate package

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 117-02-PLAN.md — Clear the live pmat violation (`dp_fill_grid` cognitive complexity 37): blocking user decision (numeric-identical refactor vs recorded deviation) [INLINE, checkpoint]

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 117-03-PLAN.md — Cargo gates: fmt, clippy --all-targets, serde build, 28 examples, full test suite (lib / integration / doc, golden-flake isolation) → gate log [INLINE]

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 117-04-PLAN.md — `cargo package` (clean list, crosslang_grid from tarball via fallback), pmat gate, harness pytest + render checks, no-tag assertions, documented operator step, REL-01 verdict [INLINE]

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 112. Benchmark Harness Setup | 3/3 | Complete    | 2026-09-22 |
| 113. Cross-Language Baseline & Hotspot Ranking | 3/3 | Complete    | 2026-09-23 |
| 114. Elastic Alignment Optimization | 2/2 | Complete    | 2026-09-23 |
| 115. SVD/FPCA & Covariance Optimization | 3/3 | Complete    | 2026-09-23 |
| 116. Cross-Language Re-Run & Published Report | 7/7 | Complete    | 2026-09-25 |
| 117. Release Preparation & Verification | 1/4 | In Progress|  |

## Coverage

**v0.47.0 requirements:** 12 total → 12 mapped (100%), 0 orphaned, 0 duplicated.

| Requirement | Phase |
|-------------|-------|
| BENCH-01 | Phase 112 |
| BENCH-02 | Phase 112 |
| BENCH-03 | Phase 112 |
| BENCH-04 | Phase 112 |
| BENCH-05 | Phase 112 |
| BASE-01 | Phase 113 |
| BASE-02 | Phase 113 |
| OPT-01 | Phase 114 |
| OPT-02 | Phase 115 |
| OPT-03 | Phase 115 |
| RPT-01 | Phase 116 |
| REL-01 | Phase 117 |

## Execution Order

Strictly sequential by dependency: **112 → 113 → 114 → 115 → 116 → 117.**

- Harness (112) + baseline (113) must precede optimization (114–115) — you cannot rank or prove speedups without the measurement infrastructure and a committed baseline.
- The two optimization phases both draw on the Phase 113 ranking; elastic alignment (114) goes first (most evidence, clean cross-language equivalent), SVD/FPCA + covariance (115) follows and carries the consolidated build-variant/determinism/pmat proof (OPT-03).
- Report (116) after all optimization; release (117) terminal.

---
*Roadmap created: 2026-09-19*
