# Phase 113: Cross-Language Baseline & Hotspot Ranking - Context

**Gathered:** 2026-09-22
**Status:** Ready for planning
**Mode:** Smart discuss — 2 consequential methodology decisions taken with the user; remaining grey areas auto-accepted per milestone research.

<domain>
## Phase Boundary

Produce a **fair, honest cross-language baseline** (fdars vs scikit-fda vs R FDA ecosystem) over an
N×M input grid, committed as `comparison.csv`, and an **evidence-ranked hotspot list** (absolute +
relative-to-baseline leverage) gated on measurement noise — confirming the elastic-alignment anchor
and ordering the secondary targets (SVD/FPCA, covariance) for Phases 114–115. Uses the Phase-112
harness. Requirements: BASE-01, BASE-02.

Boundary: this phase MEASURES and RANKS. It does NOT optimize any `fdars-core` code (that is
114–115). The only `fdars-core`-adjacent change permitted is criterion bench additions/adjustments
needed to make an op collectable (external-agnostic, no src/API/dependency change), consistent with
Phase 112's precedent.

</domain>

<decisions>
## Implementation Decisions

### Agreement-gate metric calibration (USER DECISION 2026-09-22) — carried from Phase 112
- For **depth, functional_boxplot, and FPCR**, switch the gate to a **sign/scale-invariant Pearson (or rank) correlation ≥ 0.99**, the same family FPCA already uses. Rationale: Phase 112 empirically showed these three agree up to a library normalization/scale convention (MBD ~0.65% abs difference; FPCR scale differs but Pearson ≈ 0.96) — correlation is the honest agreement test; abs-diff was the wrong instrument. This keeps all six ops in the committed baseline.
- FPCA (corr > 0.99), basis smoothing (abs < 1e-3), and elastic warping (L-inf < 1e-2) are UNCHANGED — they already pass.
- The gate remains **fail-closed** and runs **before** any timing is recorded (BENCH-05 invariant preserved). Update `benchmarks/gate.py` per-op dispatch + its unit tests accordingly.

### R FPCA fairness — time both, disclose (USER DECISION 2026-09-22) — carried from Phase 112
- Time **both**: (a) the Simpson's-weighted-SVD version that matches fdars numerically (feeds the agreement gate, apples-to-apples), and (b) **native `fda::pca.fd`** separately as the "real R ecosystem" number.
- Disclose the distinction explicitly in the baseline + the eventual report (Phase 116): the SVD-matched number is the numerically-comparable one; `pca.fd` is what an R user actually runs (different B-spline-coefficient-space algorithm). Only the SVD-matched version is gated for numeric agreement; `pca.fd` is reported as an ecosystem-realistic timing with a footnote.

### N×M input grid (auto-accepted — research defaults)
- **≥3 sizes**; elastic-alignment target at **N ≥ 100** (per BASE-01 + the v0.14.0 audit's 37.82 s @ N=500,M=50 anchor evidence). Recommended grid: N ∈ {100, 300, 500} × M ∈ {50, 200} (Claude's discretion on exact points, keeping elastic at N≥100 and at least one large point to expose the O(N²·M²) scaling). Use the seeded single-source generator so all three ecosystems get identical inputs.

### Measurement-noise gating (auto-accepted — research defaults)
- Rank only on measurements with **RSD < 5%** (criterion) / **CI width < ±5%**. Any op/size whose noise exceeds the gate is flagged and NOT used to pick an optimization target (Anti-Pattern: no spurious speedup target). Rely on the existing criterion 0.5 confidence intervals from `estimates.json`; increase criterion sample size / warm-up only where a target is too noisy.
- Fairness controls from Phase 112 stay in force: `*_NUM_THREADS=1` everywhere, `env.json` captured for the committed baseline run.

### Hotspot leverage ranking (auto-accepted — research defaults)
- Rank by BOTH **absolute time** (biggest wall-clock cost) AND **relative-to-baseline leverage** (how far fdars is behind the fastest ecosystem for that op) — an op where fdars is both slow in absolute terms and slower than scikit-fda/R is the highest-leverage target. Output a committed ranked list (CSV/markdown) that explicitly **confirms the elastic-alignment anchor** and **orders SVD/FPCA + covariance** as the secondary targets for 114–115.

### Claude's Discretion
- Exact grid points, criterion sample-size bumps, the leverage score formula (e.g. absolute_ns × (fdars_ns / min_other_ns)), and the ranked-list file format — Claude's discretion within the locked constraints above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase-112 harness at `benchmarks/`: `generate_data.py`, `python_runner.py`, `r_runner.R` (six ops), `collect_rust.py` (criterion parse, `OP_MAP`), `gate.py` (per-op agreement, needs the correlation-metric update above), `run_all.py` (orchestrator, gate-before-timing), `capture_env.py`, `verify_inputs.py`, plus `benchmarks/fpca_dump/` (standalone fdars-loadings dumper). Post-code-review fixes: R timing now in ns, all-ops path emits real per-op timings, gate fails closed.
- Existing criterion 0.5 benches in `fdars-core/benches/` incl. `bench_functional_boxplot` (added Phase 112), `bench_fregre_lm`, elastic/depth/regression groups — timing source via `target/criterion/*/new/estimates.json`.

### Established Patterns
- Measure-first + committed before-baseline (Anti-Pattern: never skip the before-baseline commit). No new crate dependency; external harness only.

### Integration Points
- `benchmarks/results/comparison.csv` (new committed artifact) + a committed ranked hotspot list; both consumed by Phases 114–116.
- `benchmarks/r_runner.R` gains a native `fda::pca.fd` timing branch (in addition to the SVD-matched FPCA branch).

</code_context>

<specifics>
## Specific Ideas

- The committed baseline is the reference Phases 114–115 optimize against and Phase 116 re-runs — it must be reproducible (env.json + seeds + lockfiles).
- Confirm the elastic anchor with actual N≥100 numbers, not just the v0.14.0 audit recollection.
- Record which ops/sizes were noise-gated out of the ranking (transparency).

</specifics>

<deferred>
## Deferred Ideas

- asv/airspeed-velocity continuous perf tracking (PERF-CI-01) — future milestone.
- Any actual `fdars-core` optimization — Phases 114–115, not here.
- The Phase-116 published report narrative — 116, not here (113 only commits the raw baseline + ranking).

</deferred>
