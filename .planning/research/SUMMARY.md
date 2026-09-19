# Project Research Summary

**Project:** fdars (crate `fdars-core`) — v0.47.0 Performance milestone
**Domain:** Cross-language numerical/FDA performance benchmarking + behavior-preserving Rust optimization
**Researched:** 2026-09-19
**Confidence:** HIGH

## Executive Summary

v0.47.0 is a **performance milestone** that pairs a cross-language benchmark harness (Rust `fdars-core` vs Python **scikit-fda** vs the R FDA ecosystem) with a strictly behavior-preserving **optimization pass** on the hot paths the harness surfaces. Unlike the prior feature milestones, the deliverable is evidence (head-to-head timings) plus provably-improved implementations — not new functionality. It is the runtime complement to the v0.14.0 audit, which did *capability/API* comparison only and explicitly avoided runtime benchmarking.

The recommended approach is **measure-first**: stand up the external harness and establish a fair cross-language baseline (with a mandatory numeric-agreement gate proving each library computes the same thing) before optimizing anything, then attack the ranked hot paths — elastic alignment O(N²·M²) is the confirmed anchor (37.82 s at N=500,M=50 in the v0.14.0 audit), with SVD/FPCA and the covariance surface as secondary targets. Every optimization is proven numeric-identical by the existing 2942-test suite + before/after criterion, mirroring the v0.30.0 measure→optimize→prove pipeline.

The dominant risk is **methodological fairness**: BLAS/OpenMP thread-count mismatches (NumPy/R use all cores; nalgebra uses one), Python cold-start/JIT overhead, version drift, and algorithm-variant divergence can all produce misleading speedups. Mitigation is baked in from Phase 1: pin threads to 1 on all sides, pin + capture every library version + BLAS backend, and gate every timing on a sign-aligned numeric-agreement check. The Rust-side risk is silent numeric drift (float reassociation, parallel-reduction order) — guarded by bit-identity golden tests run in `--release` after every optimization commit.

## Key Findings

### Recommended Stack

Almost the entire harness lives **outside** `fdars-core/Cargo.toml` (hard constraint: no new crate dependency). Rust reuses the existing **criterion 0.5** harness (results parsed from `target/criterion/*/estimates.json`, `mean.point_estimate` in ns — no new dep, no `cargo-criterion` binary needed). Python and R runners are separate processes driven by a stdlib-only orchestration script.

**Core technologies:**
- **criterion 0.5** (existing dev-dep): Rust timings + before/after cells — parse `estimates.json` directly.
- **scikit-fda 0.10.1 + pytest-benchmark 5.3.0** (Python, pinned via `requirements.lock`/uv): Python timings with structured `--benchmark-json` output. scikit-fda 0.10.1 already verified in an existing venv.
- **R `bench` 1.1.4** (preferred over `microbenchmark` — also tracks memory/GC), packages **fda, fda.usc, refund, fdapace, roahd** + **fdasrvf** (elastic alignment reference), pinned via `renv.lock`; export via `jsonlite`.
- **Thread pinning** (fairness, mandatory): `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE` + `RAYON_NUM_THREADS=1` for the baseline run; a separate parallel run optional.
- **Profiling without new deps:** `cargo-flamegraph` (external `cargo install`) + system `perf`; **dhat is forbidden** (would add a `[dev-dependencies]` entry).

### Expected Features

**Must have (table stakes):**
- Reproducible harness driving all three ecosystems on **matched input data** (single source of truth: generate synthetic curve sets once to a shared format both Python/R/Rust read — CSV is the safe cross-language default; `.npy` if `RcppCNPy` is available for R).
- **Numeric-agreement gate** before any timing (sign-aligned correlation > 0.99 for FPCA eigenfunctions; tolerance 1e-2 for elastic warping (non-unique up to reparameterization), 1e-3 for depth scores/regression coefficients).
- **Thread-count parity** + full **version/environment capture** into `benchmarks/results/env.json`.
- Ranked hotspot list combining cross-language results + the existing criterion suite.

**Should have (competitive):**
- Before/after speedup tables with confidence intervals (CI < ±5%) + N×M scaling analysis.
- Published Markdown perf report / README speed story with full methodology disclosure.

**Defer:**
- asv/airspeed-velocity-style continuous perf tracking (single-milestone comparison doesn't need it).
- Multi-machine / CI perf gating (out of scope this milestone).

### Architecture Approach

The harness is a workspace-root `benchmarks/` directory (NOT a Cargo member → zero crate coupling), plus one orchestration script under `scripts/`. A shared dataset generator emits numbered files (`shared_N{n}_M{m}.csv`) from a seeded RNG matching the formula already in `alignment_benchmarks.rs`; each ecosystem runner consumes the identical files; a collector merges criterion JSON + pytest-benchmark JSON + R `bench` JSON into one `comparison.csv`. The optimization pass plugs into the established v0.30.0 measure→optimize→prove pipeline (before-baseline committed → optimize → after-cells → `cargo test` green → golden ≤1e-10 numeric identity → optional dhat-free allocation check).

**Major components:**
1. `benchmarks/shared/generate.py` — single-source-of-truth matched input data (seeded).
2. Ecosystem runners — `run_rust` (existing criterion benches), `run.py` (pytest-benchmark/scikit-fda), `run.R` (bench/R packages).
3. `collect.py` + orchestration shell script — merge into `comparison.csv` + `env.json`; numeric-agreement checker.

### Critical Pitfalls

1. **BLAS/OpenMP thread-count mismatch** (highest credibility risk) — NumPy/R auto-use all cores, nalgebra uses one → pin `*_NUM_THREADS=1` everywhere for the baseline; record + verify linear scaling.
2. **Missing numeric-agreement gate** (critical) — a timing comparison is meaningless unless the libraries provably compute the same thing; implement `verify_agreement()` (sign-aligned correlation + tolerance) and abort timing if it fails.
3. **Float reassociation / parallel-reduction drift** — refactoring accumulation loops or introducing `par_iter().sum()` silently changes results in release builds; keep accumulation order, use `par_iter().map(..).collect::<Vec<_>>().iter().sum()`, run `cargo test --release --features linalg` + the determinism guardrail after every optimization commit.
4. **Cold-start / warm-up + tiny-input noise** — discard Python import + first-call; require RSD < 5% per data point and ≥3 input sizes (elastic target at N ≥ 100); measure compute, not setup/IO.
5. **Feature-build regression** — re-verify all build variants (default, `linalg`, `serde`, `linalg,serde,parallel`, WASM) after each optimization (serde broke once historically); keep the pmat quality-gate green.

## Implications for Roadmap

Suggested structure: **6 phases (112–117)**, measure-first. Phase numbering continues from v0.46.0 (…111).

### Phase 112: Benchmark Harness Setup
**Rationale:** Infrastructure has zero dependency on fdars internals and must precede any timing; reproducibility (env + thread pinning + numeric-agreement gate) must exist before the first measurement.
**Delivers:** `benchmarks/` scaffold, shared seeded data generator, Python (pytest-benchmark/scikit-fda) + R (bench) runners, criterion JSON integration, collector + orchestration script, pinned `requirements.lock`/`renv.lock`, `env.json` capture, numeric-agreement checker.
**Addresses:** matched-input harness, thread parity, version capture (table stakes).
**Avoids:** thread-count mismatch, cold-start, IO overhead, version drift, missing agreement gate.

### Phase 113: Cross-Language Baseline & Hotspot Ranking
**Rationale:** Evidence before optimization — establish the fair baseline across all three ecosystems and rank targets by leverage.
**Delivers:** extended criterion N×M cells, full harness run, `comparison.csv`, ranked hotspot list (absolute + relative-to-baseline).
**Uses:** criterion 0.5, the Phase 112 harness.
**Avoids:** tiny-input noise, spurious speedup claims (RSD/CI gates).

### Phase 114: Elastic Alignment Optimization (anchor)
**Rationale:** Confirmed #1 bottleneck O(N²·M²); most evidence; clean cross-language equivalent (fdars vs scikit-fda `FisherRaoElasticRegistration` vs R `fdasrvf::time_warping`).
**Delivers:** internal optimization in `src/alignment/` (e.g. Sakoe-Chiba banding / DP-loop restructuring), before/after criterion, numeric-identity proof.
**Implements:** measure→optimize→prove pipeline.
**Avoids:** float reassociation, parallel-reduction drift, feature-build regression.

### Phase 115: SVD/FPCA & Covariance Optimization (conditional on 113 ranking)
**Rationale:** FPCA/SVD ~99.8% of FPCA wall-clock; faer path 1.8–4.1× faster (already behind `linalg`); covariance surface a known secondary target.
**Delivers:** targeted optimizations + same proof pipeline.

### Phase 116: Cross-Language Re-Run & Published Report
**Rationale:** Re-run the full harness after optimization to show the gains are visible across all three ecosystems; turn measurement into the milestone's proof artifact.
**Delivers:** updated `comparison.csv`, Markdown speed story with before/after tables + scaling analysis, reproduction instructions, likely a README section.

### Phase 117: Release Preparation & Verification (terminal)
**Rationale:** Bump/CHANGELOG/gates are the final proof.
**Delivers:** version bump 0.46.0 → 0.47.0, CHANGELOG (root + crate), all whole-crate gates green (fmt, clippy `--all-targets`, full test, serde build, 28 examples + doctests, `cargo package`) **plus** the pmat quality-gate. Tag/publish is the deferred operator step.

### Phase Ordering Rationale
- Harness + baseline (112–113) must precede optimization (114–115) — you cannot rank or prove speedups without the measurement infrastructure and a committed baseline.
- Optimization phases are independent of each other but both depend on the 113 ranking; elastic alignment first (most evidence), SVD/FPCA+covariance conditional on ranking.
- Report (116) after all optimization; release (117) terminal.

### Research Flags
Phases likely needing a brief planning spike:
- **Phase 112:** numeric-agreement checker — exact `fdars fdata_to_pc_1d` sign-canonicalization convention + scikit-fda/R elastic cost-function equivalence need code inspection; tolerance thresholds validated during planning.
- **Phase 113:** criterion noise analysis — RSD < 5% gate + confidence-interval interpretation.

Phases with standard patterns (skip research):
- **Phase 114–115:** established v0.30.0 optimize pipeline + v0.43.0 determinism guardrail.
- **Phase 116–117:** established report + v0.44.0/45.0/46.0 release-gate patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | criterion 0.5 existing; scikit-fda 0.10.1 runtime-verified in a local venv; pytest-benchmark 5.3.0 + R bench 1.1.4 versions confirmed; no new Cargo deps required. |
| Features | MEDIUM | 13 overlapping operations mapped from readthedocs + CRAN; numeric-agreement protocol designed but needs implementation validation in Phase 112; algorithm-variant risk real but mitigable via explicit parameter documentation. |
| Architecture | HIGH | External-harness pattern proven in v0.14.0; criterion integration + before/after pipeline + golden-output tests all precedented (v0.30.0, v0.43.0, v0.46.0). |
| Pitfalls | HIGH | 8 cross-language + 7 optimization pitfalls documented with pre-built, phase-tagged mitigations, cross-checked against fdars history (MEMORY.md, v0.43.0 determinism). |

**Overall confidence:** HIGH

### Gaps to Address
- Exact sign-canonicalization convention of `fdars fdata_to_pc_1d` — inspect code during Phase 112 before implementing the agreement checker.
- scikit-fda vs fdars elastic cost-function equivalence — verify on a reference dataset during Phase 112.
- R `roahd` random-projection-depth default projection count — pin to match fdars before timing.
- R `.npy` reading (`RcppCNPy` availability) — default to CSV if absent.
- Exact pinned R package versions — capture via a `renv::snapshot()` setup run.

## Sources

### Primary (HIGH confidence)
- `.planning/research/STACK.md`, `FEATURES.md`, `ARCHITECTURE.md`, `PITFALLS.md` (this milestone's committed research).
- scikit-fda 0.10.1 official readthedocs API + local verified venv.
- criterion 0.5 source / existing `fdars-core/benches/` (bench structure read directly).
- fdars v0.14.0 `AUDIT-REPORT.md` (elastic-alignment + SVD/FPCA bench evidence with specific numbers).

### Secondary (MEDIUM confidence)
- CRAN Task View + package papers for fda/fda.usc/refund/fdapace/roahd/fdasrvf operation mapping.
- OpenBLAS/Intel MKL threading docs (thread-pinning mechanics); pytest-benchmark/R bench/uv/renv docs.

### Tertiary (LOW confidence)
- FDA-literature-extrapolated input-size grids (no fdars-specific published guidance); numeric-agreement tolerances (from MFPCA defaults + sign-flip convention).

---
*Research completed: 2026-09-19*
*Ready for roadmap: yes*
