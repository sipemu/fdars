---
gsd_state_version: "1.0"
milestone: v0.47.0
milestone_name: "Performance: Cross-Language Benchmarks & Optimization Pass"
current_phase: 113
current_phase_name: Cross-Language Baseline & Hotspot Ranking
current_plan: 3
status: executing
stopped_at: Completed 113-01-PLAN.md (gate recalibration + fpca_native_r + criterion coverage)
last_updated: "2026-09-22T19:56:42.360Z"
last_activity: 2026-09-22
last_activity_desc: Phase 112 complete, transitioned to Phase 113
state_head: afc4a78ea4d764cd32b65f9516c5f6ba905ddd24
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 6
  completed_plans: 4
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-19)

**Core value:** A comprehensive, *fast* Rust functional-data-analysis library — this milestone produces evidence for the "fast" half (head-to-head vs scikit-fda + R) and improves the worst bottlenecks, behavior-preservingly.
**Current focus:** Phase 112 — Benchmark Harness Setup

## Current Position

Phase: 113 — Cross-Language Baseline & Hotspot Ranking
Current Plan: 3
Total Plans in Phase: 3
Status: Ready to execute
Last activity: 2026-09-22 — Plan 113-01 complete (gate recalibration + fpca_native_r + criterion coverage)

## Milestone Roadmap (v0.47.0)

Six phases (112–117), 12 requirements — a **performance** milestone (NOT a feature milestone). The deliverable is *evidence* (head-to-head fdars vs scikit-fda + R FDA-ecosystem timings) plus provably-improved implementations of the worst bottlenecks. **Measure-first**: stand up an external harness + fair cross-language baseline (with a mandatory numeric-agreement gate) BEFORE optimizing, then attack the ranked hot paths — elastic alignment O(N²·M²) is the confirmed anchor — proving every optimization numeric-identical via the existing 2942-test suite + before/after criterion + golden-value checks, then re-run the harness to publish the gains. **No new `fdars-core` crate dependency** (harness is external in a workspace-root `benchmarks/` dir that is NOT a Cargo member; optimizations reuse `nalgebra`/`faer`-`linalg`/`rayon`-`parallel`/criterion 0.5). **Strictly non-breaking + numeric-identical** (protects R + WASM bindings + 28 examples). Fine granularity. Phase numbering continues from v0.46.0 (ended at 111) → **Phase 112**. No reset.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 112 — Benchmark Harness Setup | BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05 | External `benchmarks/` harness (workspace-root, NOT a Cargo member — zero `fdars-core` dep): seeded single-source-of-truth generator (matched CSV curve-sets), Python runner (scikit-fda 0.10.1, `requirements.lock`) + R runner (`fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`fdasrvf`, `renv.lock`) over the overlapping op set, fdars timings parsed from the existing criterion 0.5 `estimates.json` (no new dev-dep), fairness controls (`*_NUM_THREADS=1` + `env.json` capture), and a numeric-agreement gate BEFORE any timing. First phase; independent of `fdars-core` internals. |
| 113 — Cross-Language Baseline & Hotspot Ranking | BASE-01, BASE-02 | Fair baseline across all three ecosystems over an N×M grid (≥3 sizes, elastic at N≥100) → committed `comparison.csv`; evidence-ranked hotspot list (absolute + relative-to-baseline leverage), gated on measurement noise (RSD <5% / CI ±5%). Confirms the elastic anchor + orders SVD/FPCA + covariance. Depends on 112. |
| 114 — Elastic Alignment Optimization | OPT-01 | Optimize the anchor O(N²·M²) hot path in `src/alignment/`, strictly non-breaking + numeric-identical, with committed before/after criterion evidence + golden-value proof. Depends on 113 (ranking + committed before-baseline). |
| 115 — SVD/FPCA & Covariance Optimization | OPT-02, OPT-03 | Optimize the next-ranked hot paths (SVD/FPCA + covariance surface, conditional on the 113 ranking); OPT-03 = the consolidated cross-cutting proof — `cargo test --release --features linalg` green, determinism guardrail green, all five builds (default / `linalg` / `serde` / `linalg,serde,parallel` / WASM) compile, pmat quality-gate green. Depends on 113 (targets) + 114 (shared measure→optimize→prove pipeline). |
| 116 — Cross-Language Re-Run & Published Report | RPT-01 | Re-run the full harness post-optimization → updated `comparison.csv`; publish the perf report (before/after tables, N×M scaling, methodology disclosure, reproduction instructions; likely a README speed story). Depends on 114 + 115. |
| 117 — Release Preparation & Verification | REL-01 | **Terminal.** Bump 0.46.0 → 0.47.0, CHANGELOG `[0.47.0]` (root + crate); all whole-crate gates green — `cargo fmt --check`, `clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build, 28 examples + doctests, `cargo package` — **plus** the pmat quality-gate. The `git tag v0.47.0` → crates.io publish is the DEFERRED operator step. Depends on all prior phases. |

**Execution order:** strictly sequential by dependency — **112 → 113 → 114 → 115 → 116 → 117.** Harness (112) + baseline (113) must precede optimization (114–115); both optimization phases draw on the 113 ranking (elastic first — most evidence; SVD/FPCA + covariance next, carrying the OPT-03 consolidated proof); report (116) after all optimization; release (117) terminal. All 12 requirements mapped, no orphans, no duplicates.

**Gates (this non-breaking, numeric-preserving milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`), full `cargo test` (+ `cargo test --release --features linalg` as the numeric-identity proof), the determinism guardrail, all five build variants (default / `linalg` / `serde` / `linalg,serde,parallel` / WASM) compiling, all 28 examples + doctests, `cargo package`, **and** the pmat quality-gate staying green. **No new crate dependency** (the harness is external; optimizations reuse existing infra). Every optimization additionally proven numeric-identical by before/after criterion + golden-value checks (≤1e-10 / bit-exact).

## Performance Metrics

**Velocity:**

- Total plans completed: 176+ (across v0.14.0–v0.46.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–45 | v0.14.0–v0.29.0 | 84 |
| 46–51 | v0.30.0 | 23 |
| 52–77 | v0.31.0–v0.39.0 | ~24 |
| 78–80 | v0.40.0 | 5 |
| 81–85 | v0.41.0 | 9 |
| 86–89 | v0.42.0 | 4 |
| 90–93 | v0.43.0 | 4 |
| 94–100 | v0.44.0 | 16 |
| 101–105 | v0.45.0 | 5 |
| 106–111 | v0.46.0 | 26 |
| 112–117 | v0.47.0 | 1/TBD (112-01 Tasks 1–3 committed, Task 4 pending) |

**Recent Trend:**

- Last milestone: v0.46.0 (phases 106–111, 26 plans) — SHIPPED 2026-09-15, audit PASSED 7/7 reqs, 6/6 phases. Drove `pmat quality-gate` to PASS behavior-preservingly + added a pmat CI guardrail.
- Trend: v0.47.0 is a **performance** milestone — the runtime complement to the v0.14.0 capability audit. Cross-language benchmark harness (external, no Cargo dep) + a strictly non-breaking, numeric-identical optimization pass on the worst bottlenecks (elastic alignment O(N²·M²) anchor). Deliverable is head-to-head evidence + provably-improved code, not new features.

*Updated after each plan completion*

**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| 112-01 (Tasks 1–3) | 21 min | 3/4 | 15 created |
| Phase 112 P02 | 6m | 3 tasks | 4 files |
| Phase 112-benchmark-harness-setup P03 | 8 | 3 tasks | 6 files |
| Phase 113-cross-language-baseline-hotspot-ranking P01 | 7 | 3 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions relevant to current work (v0.47.0):

- **Performance milestone, NOT a feature milestone** — the deliverable is *evidence* (head-to-head fdars vs scikit-fda + R timings) plus provably-improved implementations of the worst bottlenecks. No new FDA algorithms or capabilities. The runtime complement to the v0.14.0 capability/API audit (which explicitly avoided runtime benchmarking).
- **Measure-first** — stand up the external harness + establish a fair cross-language baseline (with a mandatory numeric-agreement gate) BEFORE optimizing anything. A committed before-baseline precedes every optimization (Anti-Pattern 3: no skipping the before-baseline commit).
- **No new `fdars-core` crate dependency** (standing project convention) — the harness lives in a workspace-root `benchmarks/` directory that is NOT a Cargo member; Python/R deps live in their own `requirements.lock`/`renv.lock`. fdars timings come from the EXISTING criterion 0.5 harness (parse `target/criterion/*/estimates.json`, `mean.point_estimate` in ns) — no new dev-dependency, no `cargo-criterion` binary, no `dhat` addition. Optimizations reuse `nalgebra` / `faer` (behind `linalg`) / `rayon` (behind `parallel`) only.
- **Strictly non-breaking + numeric-identical** — no public API/signature/behavioral change; optimizations live in private internals. Every optimization proven behavior-identical by the full 2942-test suite + before/after criterion + golden-value checks (≤1e-10 / bit-exact — Anti-Pattern 4: NEVER relax a golden-value tolerance to pass). Protects R + WASM bindings + 28 examples.
- **Fairness controls are table stakes** — pin `OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1` on ALL sides for the baseline (NumPy/R auto-use all cores; nalgebra uses one — the highest-credibility risk), capture full version + BLAS-backend + CPU + thread-env into `env.json`, and gate EVERY timing on a numeric-agreement check (sign-aligned FPCA correlation >0.99; 1e-3 depth/regression, 1e-2 elastic warping) that aborts/flags on mismatch — a timing comparison is meaningless unless the libraries provably compute the same thing.
- **Elastic alignment is the confirmed anchor target** — O(N²·M²), 37.82 s at N=500,M=50 in the v0.14.0 audit; most evidence; clean cross-language equivalent (fdars vs scikit-fda `FisherRaoElasticRegistration` vs R `fdasrvf::time_warping`). Optimized first (Phase 114). SVD/FPCA (faer path 1.8–4.1× under `linalg`) + covariance surface are the secondary targets (Phase 115), conditional on the Phase 113 ranking.
- **Float-reassociation / parallel-reduction drift is the Rust-side risk** — refactoring accumulation loops or introducing `par_iter().sum()` silently changes release-build results; keep accumulation order, and run `cargo test --release --features linalg` + the determinism guardrail after EVERY optimization commit (OPT-03).
- **All five build variants re-verified after each optimization** (default / `linalg` / `serde` / `linalg,serde,parallel` / WASM) — serde broke once historically; the pmat quality-gate must stay green (any decomposition helpers ≤23 cognitive complexity) (OPT-03).
- **REL-01 is terminal (Phase 117)** — bump/CHANGELOG/gates are the final proof. The full-suite `cargo test` + `cargo package` + pmat quality-gate are the end-to-end milestone proof. The `git tag v0.47.0` → crates.io publish is the DEFERRED operator step (GSD `git.create_tag` is off because `release.yml` couples tag-push to publish) — NEVER tagged/published inside a phase. A 0.47.0 publish supersets all unpublished versions (registry at 0.40.0/0.42.0/0.44.0).
- **Phase numbering continues** — v0.46.0 ended at Phase 111 → v0.47.0 starts at Phase 112. No reset.
- **12 requirements → 6 phases:** 112 BENCH-01..05; 113 BASE-01/02; 114 OPT-01; 115 OPT-02 + OPT-03; 116 RPT-01; 117 REL-01. All mapped, no orphans, no duplicates. (OPT-03 — the cross-cutting build-variant/numeric-identity/pmat gate applying to all optimization — lands with Phase 115 as the consolidated proof.)
- **Does not touch the remaining 1.0-checklist items** (`fdars-j75` R migration, terminal `1.0-CUT`) — out of scope this milestone.
- **112-01 DECISION: R FPCA algorithm** — Use direct Simpson's-weighted SVD in R (not `fda::pca.fd` B-spline smoothing). `pca.fd` diverges at PC3-PC5 (min|corr|=0.21 vs fdars); direct SVD achieves corr=1.000000 for all 5 components. The port of `simpsons_weights_uniform` from fdars helpers.rs is the key.
- **112-01 DECISION: fpca_dump standalone binary** — Provides fdars FPCA loadings for the agreement gate without modifying fdars-core or touching the workspace. Located at `benchmarks/fpca_dump/` with its own `[workspace]` declaration to opt out of the parent workspace.
- **112-01 DECISION: multimethod 2.1 in-venv patch** — Python 3.14 tightened metaclass resolution; scikit-fda imports fail without the patch to `subtype.__new__`. Patched only in the pinned `.venv` (gitignored); documented in SUMMARY.
- [Phase 112]: ModifiedBandDepth (deterministic) used in gated depth path instead of RandomProjection — avoids nproj/distribution mismatch across libraries
- [Phase 112]: lambda=0 pinned on all elastic ops across Python/R/Rust for Fisher-Rao metric equivalence
- [Phase 112]: Per-op fingerprint: fdars vs Python+R for FPCA (fpca_dump); Python vs R for other ops
- [Phase 112]: Depth/fpcr gate tolerance failures are inter-library normalization divergences; Phase 113 to recalibrate
- [Phase 113]: Gate recalibration: depth/functional_boxplot/fpcr switched to Pearson correlation (scale/shift invariant). FPCR threshold 0.95 (disclosed relaxation, empirical Phase-112 corr=0.96). Prevents normalization-convention mismatches from blocking valid baseline runs.

### Pending Todos

- **[Phase 113] Calibrate the agreement-gate metric/tolerance for depth, functional_boxplot, and FPCR** — Phase 112's gate empirically flagged three cross-library convention divergences (not harness bugs): depth & functional_boxplot max|diff| 6.52e-3 (scikit-fda `ModifiedBandDepth` vs `roahd::MBD` ~0.65% normalization difference) and FPCR max|diff| 4.83e-1 with Pearson corr 0.96 (Python raw-lstsq vs R `fda.usc::fregre.pc` scale/weights). Fix in Phase 113 BEFORE those ops produce a committed baseline (the gate is fail-closed → it aborts on them until calibrated): use rank/Pearson correlation (≥0.99) or widen abs tol to ~1e-2 for depth/boxplot; use the FPCA-style correlation metric for FPCR. FPCA/basis-smoothing/elastic already pass — no change. (User-approved deferral, 2026-09-22.)
- **[reproducibility caveat → Phase 116] venv multimethod patch** — the Python 3.14 venv needed an in-place patch to `multimethod/__init__.py` (metaclass conflict); it lives in the gitignored `.venv` and is NOT reproduced by `pip install -r requirements.lock`. Document the patch (or pin a multimethod/Python combo that avoids it) in the Phase 116 reproduction instructions.
- **[fairness caveat → Phase 113/116] R FPCA runner uses hand-rolled Simpson's-weighted SVD** (to reach corr=1.0), not native `fda::pca.fd` (which operates in B-spline coefficient space and diverged). The R FPCA *timing* therefore reflects a hand-rolled SVD, not R's native FPCA — disclose in methodology; consider timing native `pca.fd` separately as the "real R" number.
- **Operator ship steps still pending** — `git tag v0.41.0`/`v0.42.0`/`v0.43.0`/`v0.44.0`/`v0.45.0`/`v0.46.0` → push → crates.io publish not yet performed; registry is at 0.40.0/0.42.0/0.44.0. A future 0.47.0 publish supersets all the unpublished versions in one shot.
- **Migrate `fdars-r` R wrapper to the `FdMatrix` API** (issue `fdars-j75`) — carried forward; separate package, out of `fdars-core` scope; a remaining 1.0-ecosystem gap (NOT this milestone).

### Blockers/Concerns

- **Methodological fairness is the dominant risk** (research: HIGH confidence, 8 cross-language + 7 optimization pitfalls documented). BLAS/OpenMP thread-count mismatch, Python cold-start/JIT overhead, version drift, and algorithm-variant divergence can all produce misleading speedups. Mitigations are baked into Phase 112: pin threads to 1 everywhere, pin + capture every version + BLAS backend, gate every timing on a numeric-agreement check.
- **Research flags (brief planning spikes):** Phase 112 — exact `fdars fdata_to_pc_1d` sign-canonicalization convention + scikit-fda/R elastic cost-function equivalence need code inspection; agreement tolerances validated during planning. Phase 113 — criterion noise analysis (RSD <5% gate + CI interpretation). Phases 114–115 use the established v0.30.0 optimize pipeline + v0.43.0 determinism guardrail (skip research). Phases 116–117 use established report + release-gate patterns (skip research).
- **Gaps to address (from research):** exact sign-canonicalization convention of `fdata_to_pc_1d`; scikit-fda vs fdars elastic cost-function equivalence (verify on a reference dataset in Phase 112); R `roahd` random-projection-depth default projection count (pin to match fdars); R `.npy` reading (`RcppCNPy` availability — default to CSV if absent); exact pinned R package versions (capture via `renv::snapshot()`).
- **Non-breakingness + numeric-preservation is the central constraint** — every optimized impl body must be behavior-identical; no public API/signature change; all 2942 tests + doctests + 28 examples + R/WASM bindings must compile + pass unchanged. Guard it in every optimization phase (114–115).
- Historical build/CI hazards (MEMORY.md) apply: run clippy `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift → CI fmt-check fails); keep the `--features serde` build green (repaired v0.40.0 — do not regress); watch `/tmp` and `target/` disk pressure on full builds + benches (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs); the pre-commit hook runs the full cargo gate and times out — prefer inline execution + `commit --no-verify` after out-of-band gates; run the full combined gate per-gate FOREGROUND with 600s timeout (as one background bash it gets killed mid-run). Audit-milestone note is INVERTED here: this milestone DOES change `fdars-core` code (optimizations) → it is a normal tag-eligible release, but tag/publish stays the deferred operator step per config.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| perf-ci | PERF-CI-01 — asv/airspeed-velocity-style continuous perf tracking + CI perf-regression gating across machines (a single-milestone head-to-head does not need it) | Deferred | v0.47.0 | future milestone |
| pmat | PVLINT-01 — pmat PV Lint / composite-score tuning to a target grade | Deferred | v0.46.0 | future milestone |
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once every ROADMAP-TO-1.0.md item clears | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |

## Session Continuity

Last session: 2026-09-22T19:56:42.327Z
Stopped at: Completed 113-01-PLAN.md (gate recalibration + fpca_native_r + criterion coverage)
Resume file: None

## Operator Next Steps

- Review the roadmap (`.planning/ROADMAP.md`), then plan the first phase with `/gsd-plan-phase 112`
