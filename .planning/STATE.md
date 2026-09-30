---
gsd_state_version: "1.0"
milestone: v0.47.0
milestone_name: "Performance: Cross-Language Benchmarks & Optimization Pass"
status: Awaiting next milestone
stopped_at: Phase 117 complete — all phases complete
last_updated: "2026-09-30T06:53:31.999Z"
last_activity: 2026-09-30
last_activity_desc: Milestone v0.47.0 completed and archived
state_head: df9f4083b0e63420fd8185be42595e6e7a35bbbc
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 22
  completed_plans: 22
  percent: 100
current_phase: 117
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-25)

**Core value:** A comprehensive, *fast* Rust functional-data-analysis library — this milestone produces evidence for the "fast" half (head-to-head vs scikit-fda + R) and improves the worst bottlenecks, behavior-preservingly.
**Current focus:** Phase 117 — Release Preparation & Verification

## Current Position

Phase: Milestone v0.47.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-09-30 — Milestone v0.47.0 completed and archived

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
| Phase 113 P02 | 9min | 2 tasks | 3 files |
| Phase 114-elastic-alignment-optimization P01 | 161 | 2 tasks | 4 files |
| Phase 114-elastic-alignment-optimization P02 | 591 | 2 tasks | 4 files |
| Phase 115 P01 | 75 min | 2 tasks | 4 files |
| Phase 115-svd-fpca-covariance-optimization P02 | 439 | 2 tasks | 8 files |
| Phase 115-svd-fpca-covariance-optimization P03 | 10min | 1 tasks | 1 files |
| Phase 116 P01 | 20min | 2 tasks | 16 files |
| Phase 116 P02 | 15min | 2 tasks | 4 files |
| Phase 116 P04 | 10min | 2 tasks | 2 files |
| Phase 116 P03 | 28min | 2 tasks | 2 files |
| Phase 116 P07 | 55min | 2 tasks | 4 files |
| Phase 117 P01 | 4min | 2 tasks | 8 files |

## Accumulated Context

### Decisions

- **[Phase 117 — REL-01 outcome, 2026-09-25]** fdars-core 0.47.0 release-ready: 16/16 gates PASS (117-GATE-LOG.md). Release-time pmat violation (dp_fill_grid cog 37, from Phase 114) cleared by a user-approved numeric-identical refactor (`dp_band_cols`). No tag / no publish — operator step documented.
- **[Phase 116 — RPT-01 outcome, 2026-09-25]** Published `documentation/PERFORMANCE.md` + README speed story from a full thread-pinned re-run (fdars faster than scikit-fda and R on all six overlapping ops at N=500,M=200; agreement gate passed on all 6 cells). The controlled pinned A/B re-measured the face_covariance optimization at **1.19×** (Phase 115's unpinned/noisy 3.45× not reproduced single-threaded) — the report headlines 1.19×. Disclosed: background desktop load during part of the fdars run; 7 heavy single-threaded elastic chunks > 540 s ("not run (timeout)"). Code review converged clean after 1 fix pass (CR-01 fail-open gate path in `run_all.py::step_run_gate` fixed).
- **[Phase 114 — OPT-01 outcome, user-approved 2026-09-23]** The elastic-alignment anchor is **already near-optimal** — symmetry (upper-triangle), feature-gated parallelism, one-time SRSF precompute, uniform-grid DP √(dr/dc) slope table, and thread-local scratch reuse were all pre-existing. The one numeric-identity-SAFE change found (band-iteration in `dp_fill_grid`, proven ≤1e-10 in debug+release, all 5 builds compile) is a legitimate cleanup but its speedup at n500,m50 is **within noise** (7.310s→7.164s, CIs overlap, ~2%). User accepted this as the honest OPT-01 outcome: OPT-01 satisfied by the numeric-identical optimization + committed before/after criterion evidence + the documented near-optimal finding (this is a performance-AUDIT milestone; "already near-optimal, here is the proof" is a valid result). Phase 116 reports it honestly. Real gains are likelier in Phase 115 (covariance/SVD).

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
- [Phase 113]: FPCR gate failure at n=100,m=50 (|corr|=0.9495 < 0.95): plan-mandated HARD STOP; comparison.csv not written; threshold not relaxed (Plan 02 halted)
- [Phase 113]: FPCR made compute-but-not-block (user-approved): removed from _GATED_OPS, disclosed in comparison.csv notes; five ops still hard-gated. Grid cells 1-2 passed with FPCR disclosed.
- [Phase 113]: Golden fixture constants captured at generate_curves(20,50) from un-optimized code in Wave 1 — provides immovable numeric-identity reference for Plan 02 OPT-01 verification
- [Phase 113]: BEFORE-baseline mean 7.310s CI [7.073s, 7.552s] — Plan 02 must achieve after-upper < 7.073s for non-overlapping-CI OPT-01 success
- [Phase 113]: dp_fill_grid banded arm now iterates diagonal corridor directly (O(n*r)); dp_skip_cell removed (dead code); numeric identity proven by golden fixture in debug+release at 1e-10; AFTER criterion within noise at n500_m50 (-2%, CIs overlap) — reported honestly per D-honesty-clause; unbanded arm intentionally untouched (near-optimal per RESEARCH)
- [Phase 115]: Committed BEFORE-baseline on perf_face_covariance/n200_m30 at SHA 328215e7 (268 ms mean); Plan 02 speedup check uses this as the reference.
- [Phase 115]: Golden fixture uses separate consts for (0,5) and (5,0) mirror cells (pre-opt differ by ~8e-18); assert_abs_close at 1e-10 ensures green on both pre- and post-opt code.
- [Phase 115]: sym_cov_irreg: single kernel table + upper-triangle loop + bit-copy mirror achieves ~3.45x speedup with bit-identical upper-triangle values (OPT-02)
- [Phase 115]: SVD documented near-optimal: faer thin SVD already production path under linalg feature (regression.rs:455-499)
- [Phase 115]: OPT-03 consolidated proof: 114+115 pass proven numeric-identical, non-breaking (5 builds), deterministic (6 guardrail modules), pmat-clean
- [Phase 116]: 116-01: crosslang bench group named crosslang_fpca (not fpca); render_report.py matches fdars rows by criterion source string, not op label
- [Phase 116]: 116-01: gitignore switched results/ -> results/* + explicit negations; a bare directory pattern cannot be selectively re-included
- [Phase 116]: 116-02: crosslang_grid.rs expanded to all six overlapping ops (basis_smoothing/depth/functional_boxplot/fpcr/elastic added alongside fpca); collect_rust.py OP_MAP labels them with shared op names (TDD RED/GREEN); functional_boxplot criterion settings raised (sample_size 50, measurement_time 20s) per D-10 noise finding
- [Phase 116]: 116-04: render_report.py's analysis core + all 13 emitters implemented and fully unit-tested (67 tests); THEORETICAL_COMPLEXITY grounded in actual fdars-core source (depth/functional_boxplot are O(N*M*log N) via StreamingMbd's sorted-reference rank query, not naive O(N^2*M); elastic is O(M^2), N-independent); before-after tables exclude new crosslang_* rows and _THREAD_SCALING_GROUPS; real ab-check shows Phase 115 face_covariance before-cell is FAIL-NOISY under the strict gate (RSD ~5.98%), so headline_eligible is honestly False until a cleaner phase116_ A/B entry lands
- [Phase 116]: 116-03: 67-chunk resumable scheduler + grid-cell hardening + --full reproduction command implemented; a first-executed chunk's timeout is final (loop continues, inferred-timeout skips cost no budget) while a later chunk's timeout is non-final interrupted (call stops); rerun_log.json not yet created (116-06 does the real re-run)
- [Phase 116]: Phase 116: headlined face_covariance at the pinned, noise-clean 1.19x (phase116_face_covariance_n200_m30), not the larger but unpinned/noisy 3.45x Phase 115 figure; before/after-vs-Phase-113-baseline tables are documented as a traceability record, not an optimization measurement, since the baseline mixed live re-runs with unpinned mixed-vintage criterion output.
- [Phase 117]: REL-01 metadata: fdars-core bumped 0.46.0 -> 0.47.0; root+crate CHANGELOG [0.47.0] entries cite only the pinned 1.19x face_covariance figure (never the unpinned 3.45x); README/docs install snippets refreshed to 0.47; fdars-core/.pmat*/ caches gitignored to keep them out of the crate tarball.

### Pending Todos

- **[backlog, pre-existing — found in Phase 115 review]** In `fdars-core/src/irreg_fdata/face.rs`, a `#[must_use]` + doc block (~lines 182/241-263) is orphaned above the private helper `fill_diag_blocks` due to an intervening doc comment, leaving `pub fn mface_covariance` (line 330) without docs/`#[must_use]`. NOT introduced by Phase 115 (115 only swapped the cov_irreg call site at face.rs:149). Cosmetic; fix by moving the doc block + attribute above `mface_covariance`.

- **[Phase 115] Re-benchmark SVD-proper paths before targeting them (OPT-02)** — Phase 113's ranking noise-excluded ALL `audit_p6_svd_*` cells (CI-width ≥5%, some RSD ≥5%). Phase 115 MUST re-run those criterion benches at higher sample counts (or longer measurement time) so CI-width <5% before selecting an SVD path as an optimization target. The covariance surface (`face_covariance`) is the clean, noise-passing OPT-02 target (optimize directly); FPCA has noise-stable cells + cross-language comparators. (Operator-approved 2026-09-23; see benchmarks/results/hotspot_ranking.md "Phase 115 action".)
- **[Phase 113 — RESOLVED] agreement-gate calibration** — done in Phase 113: depth/functional_boxplot → Pearson corr ≥0.99; FPCR → disclosed non-gated (corr ~0.95→0.86); FPCA → subspace-agreement, hard-gated on fdars-vs-R (1.0) with fdars-vs-Python disclosed (trapezoidal vs Simpson's quadrature); Karcher-mean elastic excluded (noise-gated), pairwise elastic is the anchor. All disclosed in comparison.csv + hotspot_ranking.md for the Phase 116 report.
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
- FPCR gate failure at n=100,m=50: |Pearson corr|=0.9495 < threshold 0.95. Python (scikit-fda) vs R (fda.usc) FPCR fitted values diverge more at n=100 than n=60. Cannot proceed with grid baseline run without threshold relaxation or algorithmic alignment. Options: (A) use n<100 for FPCR cells, (B) exclude FPCR from gate for n>=100 with disclosure, (C) investigate ncomp alignment between libraries.
- NEW (Plan 02 Task 3): FPCA HARD-gate fdars-vs-python min|corr|=0.9874 < 0.99 at cell n=300,m=50 — PC3/PC4 eigenvalue near-degeneracy (subspace agrees 0.999, individual axes 0.988). NOT the FPCR issue; FPCA must stay hard-gated (coordinator). Grid aborted at cell 3; comparison.csv not written. Needs decision: gate FPCA on subspace/principal-angles (recommended), or reduce ncomp, or disclose fdars-vs-python FPCA.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| perf-ci | PERF-CI-01 — asv/airspeed-velocity-style continuous perf tracking + CI perf-regression gating across machines (a single-milestone head-to-head does not need it) | Deferred | v0.47.0 | future milestone |
| pmat | PVLINT-01 — pmat PV Lint / composite-score tuning to a target grade | Deferred | v0.46.0 | future milestone |
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once every ROADMAP-TO-1.0.md item clears | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |

## Session Continuity

Last session: 2026-09-25T20:52:35.703Z
Stopped at: Phase 117 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
