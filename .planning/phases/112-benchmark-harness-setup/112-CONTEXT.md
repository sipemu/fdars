# Phase 112: Benchmark Harness Setup - Context

**Gathered:** 2026-09-20
**Status:** Ready for planning
**Mode:** Smart discuss — infrastructure phase (decisions pre-locked in ROADMAP/REQUIREMENTS/STATE research)

<domain>
## Phase Boundary

Stand up a **reproducible external benchmark harness** that times fdars, scikit-fda, and the
R FDA ecosystem on identical inputs — with fairness controls and a numeric-agreement gate wired
in **before** the first measurement. First phase of v0.47.0; independent of `fdars-core`
internals. Delivers infrastructure only (generator + three runners + timing collection + env
capture + agreement gate) — it does NOT produce the baseline comparison numbers (that is Phase
113) and does NOT touch `fdars-core` source.

Requirements: BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05.

</domain>

<decisions>
## Implementation Decisions

Decisions are pre-locked by the milestone research (STATE.md "Decisions"/"Blockers", REQUIREMENTS.md).
No user grey areas remain open — this is tooling setup against an already-specified contract.

### Harness Location & Build Isolation
- The harness lives in a **workspace-root `benchmarks/` directory that is NOT a Cargo workspace member** — zero new `fdars-core` (or workspace) crate dependency.
- Python deps pinned in `benchmarks/requirements.lock` (scikit-fda 0.10.1); R deps pinned in `benchmarks/renv.lock` (`fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`fdasrvf`).
- fdars timings come from the **existing criterion 0.5 harness** — parse `target/criterion/*/estimates.json` (`mean.point_estimate`, ns). No new dev-dependency, no `cargo-criterion` binary, no `dhat`.

### Shared Data Format & Generator
- A **single seeded generator** (single source of truth) emits matched curve-set files; **CSV is the default shared format** consumed bit-identically by all three runners (Rust/Python/R).
- `.npy` only if `RcppCNPy` is reliably available on the R side — otherwise default to CSV (research flag: default CSV).
- Spot-check verifies the three runners load bit-identical inputs.

### Overlapping Operation Set
- FPCA, basis smoothing, functional depth, functional boxplot, FPCR regression, elastic alignment / Karcher mean — the ops with a clean cross-language equivalent in all three ecosystems.
- Each runner emits **structured timing JSON** in a shared comparison schema.

### Fairness Controls & Env Capture
- Pin `OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1` on ALL sides (NumPy/R auto-use all cores; nalgebra uses one — the top credibility risk).
- Capture full version + BLAS-backend + CPU + thread-env metadata into `benchmarks/results/env.json`.

### Numeric-Agreement Gate (before any timing)
- Verify each library computes the same result **before** recording any timing; abort/flag on mismatch (Anti-Pattern: a timing comparison is meaningless unless the libraries provably compute the same thing).
- Tolerances: **sign-aligned FPCA correlation > 0.99**; **1e-3** depth/regression; **1e-2** elastic warping.
- FPCA needs sign-canonicalization — align to fdars `fdata_to_pc_1d` convention (research flag: verify exact convention during planning by code inspection).

### Claude's Discretion
- Orchestration script language/shape (shell vs Python driver), exact JSON schema field names, directory sub-layout under `benchmarks/`, curve-set generator RNG choice — all at Claude's discretion, consistent with the locked contract above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing criterion 0.5 benchmarks under `fdars-core/benches/` (8 benchmarks) — timing source for the fdars side via `target/criterion/*/estimates.json`.
- `fdata_to_pc_1d` (`src/regression.rs`) — FPCA reference + sign-canonicalization convention the gate must match.
- Overlapping-op fdars entry points: FPCA (`regression.rs`), depth (`src/depth/`), elastic alignment / Karcher mean (`src/alignment/`), FPCR (`src/scalar_on_function/`).

### Established Patterns
- No new crate dependency (standing project convention); external tooling stays out of the Cargo graph.
- Determinism / seeded reproducibility already a project value (per-thread RNG seeding pattern).

### Integration Points
- New top-level `benchmarks/` dir (sibling to `fdars-core/`), explicitly excluded from the Cargo workspace.
- Consumes fdars build artifacts (`target/criterion/...`) — no source changes to `fdars-core`.

</code_context>

<specifics>
## Specific Ideas

- scikit-fda 0.10.1 exact pin; R packages captured via `renv::snapshot()`.
- scikit-fda vs fdars elastic cost-function equivalence must be verified on a reference dataset in this phase (research flag).
- R `roahd` random-projection-depth default projection count must be pinned to match fdars.

</specifics>

<deferred>
## Deferred Ideas

- asv/airspeed-velocity continuous perf tracking + CI perf-regression gating (PERF-CI-01) — deferred to a future milestone; a single-milestone head-to-head does not need it.
- Producing the actual baseline comparison numbers + hotspot ranking — that is Phase 113, not here.

</deferred>
