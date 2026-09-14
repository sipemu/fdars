---
gsd_state_version: "1.0"
milestone: v0.46.0
milestone_name: Quality Hardening — pmat Complexity Pass & CI Guardrail
current_phase: 109
current_phase_name: Complexity Tail & Quality-Gate Green
current_plan: 3
status: executing
stopped_at: Completed 109-01-PLAN.md (TRACER)
last_updated: "2026-09-14T12:25:02.924Z"
last_activity: 2026-09-14
last_activity_desc: Phase 109 execution resumed (wave continue)
state_head: 402d4cd3bd82c850a2f3bc77c1805ce16bba0f14
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 26
  completed_plans: 13
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-13)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps on the road to a deliberate 1.0 cut.
**Current focus:** Phase 109 — Complexity Tail & Quality-Gate Green

## Current Position

Phase: 109 (Complexity Tail & Quality-Gate Green) — EXECUTING
Current Plan: 3
Total Plans in Phase: 4
Status: Ready to execute
Last activity: 2026-09-14 — Phase 109 execution resumed (wave continue)

## Milestone Roadmap (v0.46.0)

Six phases (106–111), 7 requirements — a **quality/refactor** milestone (NOT a feature milestone). The deliverable is `pmat quality-gate` PASSING on `fdars-core` (120 complexity + 11 entropy + 2 SATD violations → 0, plus the 1 critical `.unwrap()`) plus a pmat CI guardrail. **Strictly non-breaking + numeric-preserving** — every refactor is pure internal helper extraction; behavior and numeric output stay identical, proven by the existing 2942 lib tests + 215 doctests + 28 examples staying green. No public API/signature changes, `fdars-core` only (protects R + WASM bindings). **No new crate dependency** — pmat is a dev/CI tool (CI adds a pmat install step). Fine granularity. Phase numbering continues from v0.45.0 (ended at 105) → **Phase 106**. No reset.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 106 — Top-Level Complexity Hotspots | QUAL-01 | Decompose the 8 worst functions into helpers under pmat thresholds: `funfem_cluster` (cyc 59), `elastic_net_cd` (cog 54), `wnet_cv_lambda` (cog 45), `wnet` (cog 31), `boost_fofr` (cyc 39), `pace_fpca` (cyc 37), `bayesian_fosr` (cyc 36), `kcfc_cluster` (cyc 28). Behavior/numeric-identical (full test suite green). Independent of 107/108. |
| 107 — SPM Complexity Cluster | QUAL-02 | Decompose the dense `src/spm/` cluster (`mfpca`, `spm_phase1_iterative`, `spm_mewma_monitor`, `select_ncomp`, `conditional_expectation`/partial, `spm_monitor_from_fields`/phase, `profile_phase1`, `spe_multivariate`/stats) under pmat thresholds. Behavior/numeric-identical. Independent of 106/108. |
| 108 — Entropy Extraction & SATD/Unwrap Cleanup | QUAL-04, QUAL-05 | Extract the 11 repeated `DataTransformation` patterns into shared helpers (entropy → 0); reword the 2 SATD doc-comment false-positives (`depth/dispatch.rs:281`, `metric/lp.rs:175`); replace the critical `.unwrap()` at `depth/rpd.rs:229` with `?`/`.expect()`. Independent of the complexity refactors — may run early/in parallel. |
| 109 — Complexity Tail & Quality-Gate Green | QUAL-03 | Clear all remaining complexity violations so `pmat analyze complexity` reports 0 and `pmat quality-gate` PASSES (0 complexity + 0 entropy + 0 SATD + 0 defect). Depends on 106 + 107 + 108 (the count is only verifiable once the hotspot/SPM/entropy/SATD/defect work has landed). |
| 110 — pmat CI Guardrail | CI-01 | Add a GitHub Actions job that installs pmat + runs `pmat quality-gate` against `fdars-core`, failing on reintroduced violations (mirrors the v0.43.0 `determinism-guardrail`); documented for local reproduction. Depends on 109 (gate must be green before enforced). |
| 111 — Release Preparation & Verification | REL-01 | **Terminal.** Bump 0.45.0 → 0.46.0, CHANGELOG `[0.46.0]` (root + crate); all whole-crate gates green — `cargo fmt --check`, `clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build guard, 28 examples + doctests, `cargo package` — **plus** `pmat quality-gate` passing. The `git tag v0.46.0` → crates.io publish is the DEFERRED operator step. Depends on all prior phases. |

**Execution order:** (106, 107, 108) → 109 → 110 → 111. Phases 106, 107, and 108 are independent and may run in parallel (or any order). Phase 109 depends on all three (verifiable count); 110 depends on 109 (gate green before enforcement); 111 (release) is terminal. All 7 requirements mapped, no orphans, no duplicates (QUAL-04 + QUAL-05 share Phase 108 as grouped independent cleanup).

**Gates (this non-breaking, numeric-preserving milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`), full `cargo test` (the non-regression + numeric-preservation proof), a `--features serde` build guard, all 28 examples + doctests, `cargo package`, **and** `pmat quality-gate` passing (the milestone's defining exit criterion). **No new crate dependency** (pmat is a dev/CI tool, invoked externally + installed in CI).

## Performance Metrics

**Velocity:**

- Total plans completed: 150+ (across v0.14.0–v0.45.0)
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
| 106–111 | v0.46.0 | 0/TBD (planned) |

**Recent Trend:**

- Last milestone: v0.45.0 (phases 101–105, 5 plans) — audit 5/5, integration INTEGRATED, release-ready. Cleared the **algorithm (SDTW-O1)** blocker on the 1.0 checklist.
- Trend: v0.46.0 is a **quality/refactor** milestone — an extra quality cycle before the remaining 1.0 items (`fdars-j75`, `1.0-CUT`). Drives `pmat quality-gate` to PASS behavior-preservingly + adds a pmat CI guardrail. Non-breaking, numeric-preserving, no new crate dependency.

*Updated after each plan completion*

**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| (none yet — v0.46.0) | — | — | — |
| Phase 106 P01 | 7 | 3 tasks | 1 files |
| Phase 106-top-level-complexity-hotspots P02 | 15 | 2 tasks | 1 files |
| Phase 106 P03 | 21 | 2 tasks | 2 files |
| Phase 106 P04 | 18 | 2 tasks | 1 files |
| Phase 107 P01 | 408 | 2 tasks | 2 files |
| Phase 107 P02 | 360 | 3 tasks | 3 files |
| Phase 107-spm-complexity-cluster P03 | 25 | 3 tasks | 4 files |
| Phase 107 P04 | 1800 | 3 tasks | 3 files |
| Phase 108 P01 | 15 | 2 tasks | 3 files |
| Phase 108 P02 | 35 | 3 tasks | 11 files |
| Phase 108 P03 | 12 | 2 tasks | 5 files |
| Phase 109 P01 | 7 | 3 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions relevant to current work (v0.46.0):

- **GATE REDEFINED (2026-09-14, operator decision).** The milestone quality-gate is now `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` — the stochastic **entropy** analyzer is EXCLUDED. Rationale: pmat 3.18.0's entropy sampler is non-deterministic (2/5/3 violations across identical runs; residual = widespread doctest `map/collect` idioms) → a stable 0 is impractical and would make the Phase-110 CI gate flaky. The 11→~3 entropy reduction done in Phase 108 is kept as a genuine improvement; entropy is documented as untracked/stochastic, same treatment as the deferred PVLINT-01. **QUAL-04** acceptance = entropy materially reduced + excluded from the deterministic gate (NOT stable-0).
- **COMPLEXITY BAR = cog ≤23 (gate "recommended"), not ≤25.** `pmat quality-gate` flags cognitive >23; that is stricter than `pmat analyze complexity`'s >25. ~97 functions crate-wide still exceed cog 23 (incl. 2 residual from 106/107: `funfem_cluster` cog 25, `pace_fpca_validate_config` cog 24). **Phase 109 = full grind of all ~97 to cog ≤23 / cyc ≤30** (large phase — bigger than 106+107 combined). All future complexity work targets cog ≤23.
- **Quality/refactor milestone, NOT a feature milestone** — the deliverable is the curated `pmat quality-gate` PASSING on `fdars-core` (0 complexity + 0 SATD + 0 defect + 0 dead-code + 0 duplicates; entropy excluded per above) plus a pmat CI guardrail. Measured by **pmat 3.18.0**. No new algorithms or capabilities.
- **Strictly non-breaking + numeric-preserving** — every refactor is pure internal decomposition (extract private helpers); behavior and numeric output stay identical (the established crate norm — private impl bodies stay behavior-identical). No public API/signature/visibility changes. Complexity must be reduced by helper extraction, never by changing signatures (protects callers/bindings/examples).
- **`fdars-core` only** — protects R + WASM bindings; refactors must be internal and not alter the public surface bindings depend on.
- **No new *crate* dependency** — pmat is a dev/CI tool invoked externally, never added to `Cargo.toml` (standing project convention). CI needs a pmat install step (Phase 110).
- **Phase 109 depends on 106 + 107 + 108** — the "0 complexity findings" / quality-gate PASS count is only verifiable once the hotspot, SPM, entropy, SATD, and defect work has landed. Phases 106/107/108 are mutually independent and may run in parallel.
- **CI guardrail lands AFTER quality-gate is green (Phase 110 after 109)** — enforcing the gate before the tree is clean would fail CI on introduction. Mirrors the v0.43.0 `determinism-guardrail` pattern.
- **QUAL-04 + QUAL-05 grouped into Phase 108** — entropy extraction (test-helper + a few src `DataTransformation` patterns), SATD doc-comment rewording, and the single `.unwrap()` fix are all independent of the complexity refactors and small; grouped as one cleanup phase (fine granularity, but no thin single-item phases).
- **The unwrap fix (`depth/rpd.rs:229`) is behavior-preserving** — replace with `?` or `.expect()` with a descriptive message; the surrounding path must keep returning the same result on valid input (the crate norm is `Result<T, FdarError>`, no panics on validated input).
- **The 2 SATD findings are doc-comment false-positives** — keywords ("Defect:"/"Performance:") appear in `///` text; reword the prose so pmat's keyword matcher stops flagging them (no code change).
- **Phase numbering continues** — v0.45.0 ended at Phase 105 → v0.46.0 starts at Phase 106. No reset.
- **7 requirements → 6 phases:** 106 QUAL-01; 107 QUAL-02; 108 QUAL-04 + QUAL-05; 109 QUAL-03; 110 CI-01; 111 REL-01. All mapped, no orphans, no duplicates.
- **REL-01 lands last (Phase 111)** — bump/CHANGELOG/gates are the final proof; the full-suite `cargo test` + `pmat quality-gate` are the end-to-end milestone proof. The `git tag v0.46.0` → crates.io publish is the DEFERRED operator step (GSD `git.create_tag` is off because `release.yml` couples tag-push to publish) — never tagged/published inside a phase.
- [Phase 106]: elastic_net_cd needed 4 private helpers (not 3 per RESEARCH.md): sweep fn itself scored cog 34 after extraction; added elastic_net_cd_update (single-column update, #[inline]) to bring sweep to cog 8
- [Phase 106]: Split-validate pattern established: when extracted validate fn exceeds cog 25, split along natural dimension boundary (data shape guards vs. config param guards)
- [Phase 106]: funfem_cluster and kcfc_cluster decomposed into 11 private helpers; both below pmat cognitive 25 / cyclomatic 30 thresholds; RNG draw order preserved for seeded numeric identity
- [Phase 106]: boost_fofr: 3-way validate split (predictors + dims + config) needed because dims alone scored cog 26 after extraction; extra level brings dims to cog 15
- [Phase 106]: bayesian_fosr_precompute returns xtx (not xbar) to avoid redundant compute_xtx in parent; RNG draw order preserved via unchanged outer iter loop with per-kk gibbs_step calls
- [Phase 106]: [Phase 106-04]: pace_fpca BLUP closure extracted as named fn with all captures as explicit Send parameters (11 params); split-validate pattern applied for validators (dims + config)
- [Phase 107]: Split-validate applied to amewma_validate (7 guards): split into amewma_validate_lambda_range + amewma_validate_params to keep each helper below cog 25
- [Phase 107]: iterative_check_removal_budget returns bool covering both max_removal_fraction and n_after<4 guards; called before iterative_update_state to replicate original break ordering
- [Phase 107]: mfpca_build_stacked takes &mut [f64] for scales — propagates scales[p]=1.0 mutation back to caller (clippy ptr_arg fix)
- [Phase 107]: 107-03: extract validate-and-call helpers in ewma/mewma/stats/phase — mewma_validate returns ncomp, spe_mv_compute_var returns Vec per-obs, monitor_fields_validate consolidates 5 guards (cog 15, no split-validate needed)
- [Phase 107]: cond_exp_build_m_matrix returns (m_matrix, diag_max) tuple where diag_max is from ORIGINAL pre-regularization matrix to preserve Tikhonov retry sequence
- [Phase 107]: profile_validate_windows split-validate keeps profile_validate under threshold (7 guards total)
- [Phase 108]: pmat SATD semantic detection: rewording requires removing the CONCEPT not just the keyword token — lp.rs needed two iterations to clear Performance category
- [Phase 108]: depth/rpd.rs unwrap fix: used .expect() not ? to keep build_augmented return type as FdMatrix and avoid cascading signature changes to the public API
- [Phase 108]: Detrend uses uniform_grid_range_div (div-before-mul); seasonal uses uniform_grid_range (mul-before-div) — FP forms verified non-identical, separate helpers required
- [Phase 108]: uniform_step_grid(n, step) added for i*step patterns (covariance.rs E-09); not consolidatable with uniform_grid_range (FP-distinct)
- [Phase 108]: subtract_fitted chosen over compute_residuals in scalar_on_shape.rs — 3-arg version already exists at line 636; naming collision forced distinct name
- [Phase 108]: pmat entropy counts structurally-identical lines inside a helper: node_bbox needed single-fold accumulation to clear fem_smoothing flag; sin_sin_surface test helper also required
- [Phase 108]: helpers::extract_curves equivalence verified by source inspection (FdMatrix::rows() == (0..nrows).map(|i| row(i)).collect()): reused in soft_dtw without adding new extract_rows helper
- [Phase 109]: 109-01 TRACER: two-seam decomposition for cog 28-32; single-seam for cog 24-25; inline on hot-path helpers; clippy ptr_arg auto-fixed

### Pending Todos

- **Operator ship steps still pending** — `git tag v0.41.0`/`v0.42.0`/`v0.43.0`/`v0.44.0`/`v0.45.0` → push → crates.io publish not yet performed; registry is at 0.40.0/0.42.0/0.44.0. A future 0.46.0 publish supersets all the unpublished versions in one shot.
- **Migrate `fdars-r` R wrapper to the `FdMatrix` API** (issue `fdars-j75`) — carried forward; separate package, out of `fdars-core` scope; a remaining 1.0-ecosystem gap (NOT this milestone).

### Blockers/Concerns

- **No research/SUMMARY.md relevant to this milestone** — intentional: internal quality/refactor milestone measured by pmat 3.18.0, not a new-feature or ecosystem-parity audit. The reference baseline is pmat's thresholds. REQUIREMENTS.md is authoritative. Non-blocking for the roadmap.
- **Non-breakingness + numeric-preservation is the central constraint** — every refactored impl body must be behavior-identical; no public API/signature/visibility change; all 2942 tests + 215 doctests + 28 examples + R/WASM bindings must compile + pass unchanged. Guard it in every complexity/cleanup phase (106–109).
- **pmat must be available for local verification** — every complexity/cleanup phase needs `pmat analyze complexity`/`entropy`/`satd`/`defects` and `pmat quality-gate` runnable to confirm the count drop; ensure pmat 3.18.0 is installed in the execution environment.
- Historical build/CI hazards (MEMORY.md) apply: run clippy `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift → CI fmt-check fails); keep the `--features serde` build green (repaired v0.40.0 — do not regress; NOTE: serde build has a known pre-existing break history — re-verify); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs); the pre-commit hook runs the full cargo gate and times out — prefer inline execution + `commit --no-verify` after out-of-band gates; run the full combined gate per-gate FOREGROUND with 600s timeout (as one background bash it gets killed mid-run).

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| pmat | PVLINT-01 — pmat PV Lint / composite-score tuning to a target grade (this milestone targets the falsifiable quality-gate PASS, not the opinionated composite) | Deferred | v0.46.0 | future milestone |
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once every ROADMAP-TO-1.0.md item clears | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |

## Session Continuity

Last session: 2026-09-14T12:25:02.890Z
Stopped at: Completed 109-01-PLAN.md (TRACER)
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 106` (or run the autonomous flow). Phases 106/107/108 are independent and may be planned/executed in parallel.
