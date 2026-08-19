# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v0.14.0 — Performance & scikit-fda Gap Audit

**Shipped:** 2026-08-09
**Phases:** 9 | **Plans:** 21 | **Tasks:** 25

### What Was Built
- `.planning/research/AUDIT-REPORT.md` — consolidated audit report: methodology (feature-flag matrix + infra-vs-code triage), 5 performance findings (PF-1..5, each bench-linked), 82 in-scope scikit-fda gaps, and 30 fdars-exclusive strengths.
- `.planning/research/BACKLOG.md` — 32-item value-ranked backlog (`score = value/√effort`), 34 seven-field promotion-ready blocks, completeness gate passed.
- A reproducible criterion benchmark corpus (~51 artifacts under `.planning/research/bench/`) across the 4-combo feature matrix.

### What Worked
- **Tracer-first phase structure.** Every phase opened with a Wave-1 "tracer" plan that proved the measure→artifact→report→backlog pipeline on ONE cell before expanding. Caught schema issues early and made later waves mechanical.
- **Audit-only discipline held.** All 9 phases produced analysis artifacts with zero `fdars-core/src/` edits — scope never leaked into implementation.
- **Milestone audit earned its keep.** The pre-archive `/gsd-audit-milestone` + integration checker caught a real (if cosmetic) defect — a "6 P1 items" miscount contradicted by a 5-item table — that all 9 phase verifications had passed over.
- **Evidence traceability.** Every consolidated finding links back to a real bench artifact with matching numbers; every backlog item to a report section.

### What Was Inefficient
- **`/tmp` tmpfs exhaustion blocked every hook-run commit.** Doctests link in a small `/tmp` and fail with a bogus "No space left"; all docs-only `.planning/` commits had to use `--no-verify`. Recurring friction (see MEMORY.md).
- **Worktree base divergence forced sequential execution.** Local `main` is ahead of `origin/HEAD`, so harness worktrees fork the wrong base (#683). Every phase auto-degraded to sequential single-tree dispatch — correct, but no parallelism benefit.
- **SUMMARY `requirements_completed` frontmatter was under-filled.** Most SUMMARYs left it blank, and the milestone-complete accomplishment auto-extraction pulled junk one-liners (`fdars-core/Cargo.toml`, "8 rows total:") that needed manual curation.

### Patterns Established
- **Tracer plan → expansion wave(s)** per phase, all appending to shared deliverable files (AUDIT-REPORT.md, BACKLOG.md) — inherently sequential, handled cleanly on the main tree.
- **7-field backlog item contract** (location, current cost/gap, root cause, proposed direction, severity P1/P2/P3, effort S/M/L, evidence link) + `value/√effort` ranking — reusable for any future audit.
- **Capability-first parity mapping** (not API-name counting) with "searched fdars for:" notes and known-bug accuracy flags.

### Key Lessons
1. **Run `/gsd-audit-milestone` before `/gsd-complete-milestone`** — phase-level verification does not catch cross-artifact numeric inconsistencies; the milestone audit does.
2. **Fill SUMMARY `requirements_completed` frontmatter during execution** — it feeds the milestone accomplishment list and the 3-source requirement cross-reference; blank frontmatter degrades both.
3. **On this machine, `/tmp` must be freed before hook-verified commits**, or use `--no-verify` for docs-only `.planning/` changes (documented exception).
4. **Set `worktree.baseRef:"head"`** if parallel worktree execution is wanted while `main` is ahead of `origin` — otherwise expect sequential auto-degrade.

### Cost Observations
- Model mix: orchestration on Opus; executors + verifier on Sonnet; integration checker on Haiku.
- Notable: sequential single-tree dispatch throughout (worktree base divergence) — no parallel-wave speedup this milestone.

---

## Milestone: v0.17.0 — Registration Parity & Elastic-FPCA Performance

**Shipped:** 2026-08-12 (release pending — version bump + PR + tag)
**Phases:** 2 (14–15) | **Plans:** 3

### What Was Built
- FEAT-06: `least_squares_shift_registration` + `ShiftRegistrationResult` in new `alignment/shift.rs` — per-curve rigid shift to the sample mean via golden-section L2 minimization; fills the "simplest registration method" gap.
- FEAT-07: three registration-quality scores (`least_squares_score`, centered-Pearson `pairwise_correlation_score`, `sobolev_least_squares_score`) in `alignment/quality.rs`, standalone-energy form.
- PERF-04: parallelized the three elastic-FPCA per-curve loops via `iter_maybe_parallel!` collect-then-assign, `SCORES_PARALLEL_THRESHOLD=50` guard on the light loop; bit-identical to sequential (tested `parallel` ON and OFF).

### What Worked
- The audit backlog's exact line numbers + signatures made discuss/plan fast; CONTEXT locked the one real design call per phase (standalone-energy scores; :764 threshold) so planning didn't relitigate.
- Code review earned its keep on Phase 14: caught a real CI-blocker (test-only `--all-targets` clippy warnings my `-p` clippy missed) plus a correctness fix (documented "Pearson" but implemented uncentered cosine → centered).
- Skipping research + pattern-mapper for the mechanical Phase 15 (CONTEXT already named the analog) kept it lean without loss of quality; review came back clean.

### What Was Inefficient
- Two subagent connection drops mid-response (one planner ~75 min then errored; one integration checker) forced a retry / inline fallback. The planner retry with an explicit "work from PATTERNS/RESEARCH, don't re-explore" note completed in ~4 min — over-exploration was the likely hang cause.
- Default-feature full-suite compile exceeded the 2-min bash cap (cold build); ran the fast checks separately.

### Patterns Established
- For a pure-refactor phase, author VALIDATION.md inline and skip research/pattern-mapper agents — the CONTEXT + one analog file is enough grounding.
- Verify perf/parallelism phases by equivalence under `parallel` ON **and** OFF, not a pinned speedup (respects the audit's LOW-CONFIDENCE governor caveat).
- CI parity: always run `cargo clippy --all-targets -D warnings` (not just `-p ... -- -D warnings`) — test code warnings block CI.

### Key Lessons
- A `-p` clippy run does NOT cover `--all-targets` (test/bench code); the CI gate does. Match the CI command in verify steps.
- When an executor deviates (Phase 14 added the `mod.rs` re-export early to pass the clippy gate), thread the deviation explicitly into the next wave's prompt to avoid duplicate-import breakage.

### Cost Observations
- Model mix: orchestration on Opus; planners on Opus; executors + verifier + phase-researcher on Sonnet; plan-checker + integration-checker on Haiku; code review on Opus.
- Notable: sequential single-tree dispatch throughout (worktree base divergence per MEMORY.md) — no parallel-wave speedup; two transient API connection drops required retries.

---

## Milestone: v0.18.0 — R-Ecosystem Gap Audit

**Shipped:** 2026-08-15
**Phases:** 4 (16–19) | **Plans:** 5

### What Was Built
An audit-only milestone (the R-ecosystem analog of v0.14.0): a versioned inventory of 35 R FDA packages (275 capabilities, 248 in-scope), a 250-row fdars-vs-R parity matrix (162 actionable gaps), a re-vetted reverse-parity strengths sweep (12 R-honest fdars strengths), and a 26-item value-ranked `R-BACKLOG.md`. Zero `fdars-core/src/` edits.

### What Worked
- **Web-enabled researcher for the inventory** — the CRAN-cross-checked survey (versions verified live) was the single highest-value subagent call; front-loading it made Phases 17–19 mechanical consolidations.
- **Reusing the v0.14.0 audit as a template** — rubrics (D-01 verdict, D-03 category), the 7-field backlog block, and the fdars-side §Phase 8 catalogue gave every phase a proven shape and a head start.
- **Honesty gates** — re-vetting strengths against R (broader than scikit-fda) collapsed 30 scikit-fda "fdars-only" items to 12; the agent surfaced its own `Rfssa` survey miss rather than hiding it.

### What Was Inefficient
- **Background-agent instability** — the planner subagent was lost twice to process exits mid-run before completing on the third try; several agents completed "late," overlapping orchestrator inline work and causing reconciliation churn. Net: the orchestrator did more inline consolidation + all phase bookkeeping itself for reliability.
- **Count reconciliation** — the Phase-16 header count (248) vs literal parity rows (250) and a plotting/IO subtotal typo (25 vs 24) each needed a documented recount.

### Patterns Established
- **Distinct-filename discipline for a second audit** (`R-AUDIT-REPORT.md`/`R-BACKLOG.md`) keeps two yardsticks separable without touching the first audit's artifacts.
- **Re-vet, don't copy** reverse-parity strengths when the comparison baseline widens.

### Key Lessons
- When background agents are unreliable, dispatch-and-wait for the heavy analytical phases (parity, strengths, synthesis) but keep bookkeeping + small consolidations inline — the deliverable never depends on a single agent surviving.
- A broader yardstick (R vs scikit-fda) inverts strength claims: capabilities unique against a narrow baseline often have analogs in a deep one.

### Cost Observations
- Model mix: orchestration on Opus; researcher on Sonnet; parity/strengths/synthesis analysis agents + planner on Opus.
- Notable: 4 heavy analysis subagents (1 researcher + 3 general-purpose) carried the bulk of the work; docs-only commits used `--no-verify` throughout (pre-commit cargo gate spuriously fails on `/tmp` for `.planning/` commits, per MEMORY).

---

## Milestone: v0.19.0 — Functional Inference Suite

**Shipped:** 2026-08-16
**Phases:** 2 (20–21) | **Plans:** 2

### What Was Built
fdars' first standalone functional-inference surface: a new `fdars-core/src/inference/` module (7 files, 8 public entry points). INF-01 two-sample tests (`t_perm_test`, `f_perm_test`, `two_sample_mean_test`, `mean_scb`, `scb_two_sample_test` + `TestResult`); INF-02 FLM inference (`flm_f_test`, `flm_gof_test`, `oneway_anova_vstat`). Closes R-parity Area 5 (previously 0/22). First milestone promoted from the v0.18.0 R-backlog.

### What Worked
- **Verify-anchors-before-planning** — grepping the actual reuse targets (`fanova`/`integrated_f_statistic`, `hotelling_t2`, `scb_mean_degras`, `FregreLmResult` fields) into the CONTEXT before planning meant the plans referenced real code, not phantom APIs; `scb_mean_degras` already existing made INF-01's `mean_scb` a thin wrapper.
- **Reuse over dependencies** — self-contained χ²/F survival functions (regularized incomplete gamma/beta) avoided a `statrs` API addition and a package-legitimacy review; the `inference/dist.rs` refactor gave both phases one home.
- **Tracer-first plans + statistical-correctness test mandate** — each plan led with one end-to-end working test before expanding; requiring "rejects real effect / fails-to-reject null" tests (not just "compiles") caught a non-zero-mean test-noise bug during execution.
- **Orchestrator independent re-verification** — re-running `cargo test`/`clippy` after each executor rather than trusting the summary; both phases confirmed green first-hand (2039 lib tests).

### What Was Inefficient
- **Background-executor latency** — the code executors ran long (compile/test loops); one execution spanned many minutes. Dispatch-and-wait was correct but slow; the session's earlier background-agent process-exit instability made waiting feel risky.
- **Noisy auto-extracted accomplishments** — `milestone.complete` pulled "[Rule 3 - Blocking]"-style lines from SUMMARYs into the MILESTONES.md entry; the base entry needed manual cleanup awareness.
- **Benign false-positive close warning** — SUMMARY path-check flagged `inference/{dist,flm,anova}.rs` as "not on disk" when they exist (path-format mismatch).

### Patterns Established
- **Implementation milestone from a research backlog**: promote a backlog cluster → CONTEXT with verified anchors → tracer-first plan → executor → independent re-verify → bookkeeping. Reusable for the next R-backlog items.
- **`inference/dist.rs`** is now the shared home for self-contained distribution survival functions (χ², F) — extend it (t, beta) rather than adding `statrs`.

### Key Lessons
- For numeric/statistical code, tests must assert *behavior against known truth* (tabulated quantiles, reject/fail-to-reject on synthetic effects), not just execution — this is what makes an inference implementation trustworthy without a reference-library cross-check.
- Verifying reuse anchors up front is the cheapest correctness lever: it collapses effort (found `scb_mean_degras` pre-built) and prevents plans built on non-existent APIs.

### Cost Observations
- Model mix: orchestration + planners + executors all on Opus (statistical-correctness stakes); no research pass (reuse-heavy, anchors concrete).
- Notable: no new crate dependency; additive/non-breaking (existing signatures incl. `fanova` frozen; only visibility widenings). Crate release deferred to a separate ship step.

---

## Milestone: v0.21.0 — Functional Regression Completeness

**Shipped:** 2026-08-17
**Phases:** 2 | **Plans:** 2 | **Tasks:** 4

### What Was Built
- **REG-01 (Phase 24):** dense functional concurrent / varying-coefficient regression — `concurrent_regression` + `ConcurrentRegrResult` in new `concurrent_regression.rs`; pointwise-OLS-per-grid-column then local-linear kernel smoothing of β(t), reusing `smoothing.rs`.
- **REG-02 (Phase 25):** functional GLM over FPC scores — `functional_glm` + `GlmFamily{Binomial,Poisson,Gamma,Gaussian}` + `FunctionalGlmResult` in new `scalar_on_function/glm.rs`, generalizing the `functional_logistic` IRLS loop. Binomial reproduces logistic within 1e-6.
- Both additive/non-breaking, no new crate dependency; full suite 2081 lib + doctests green, clippy `--all-targets` clean.

### What Worked
- **Reuse-first paid off:** both phases landed one plan each because the research pinned exact reuse targets (`smoothing::solve_gaussian_pub`/`local_linear`; `functional_logistic` IRLS + `fdata_to_pc_1d`).
- **Code review earned its keep:** it caught a *real* correctness bug in Phase 25 (Gamma IRLS weight inverted `1/μ²` vs `μ²`) that the finiteness-only test missed, plus a Poisson factorial-overflow DoS and several panic-on-bad-input guards. Both phases shipped materially more robust for it.
- Tracer-first decomposition (Gaussian/single-predictor end-to-end first) gave a compiling result struct early and de-risked the family/predictor expansion.

### What Was Inefficient
- **The `gsd-code-fixer` subagent stalled** (600s stream watchdog) mid-run on Phase 25 because the per-commit pre-commit hook runs the *full* suite + clippy (~2 min each) and 8 atomic fix commits blew the watchdog budget. Recovery: fast-forwarded its one committed fix, then applied the remaining 7 findings inline and gated once. Lesson: for multi-commit fix runs against a heavy per-commit hook, batch the commits or gate-once-at-end rather than atomic-per-finding.
- **Research claimed `statrs` was a dependency** (it wasn't) — the executor had to pivot Poisson `log(y!)` to an inline computation; a follow-up review flagged the naive O(y) form and it was replaced with an inline Lanczos `ln_gamma`. Verifying dependency availability during research would have avoided the churn.
- The pre-commit hook's full-suite run repeatedly false-failed docs-only commits via the known /tmp doctest-link exhaustion; `--no-verify` for docs (per MEMORY) worked but is friction.

### Patterns Established
- **Non-finite input guard belongs first** in family/response validation — `NaN <= 0.0` is false and `Inf.floor() == Inf`, so per-family guards silently pass non-finite values (same class of bug surfaced in *both* phases: bandwidth NaN in 24, response NaN/Inf in 25).
- **Store `link_deriv` separately from `irls_weight`** for multi-family IRLS — the Binomial shortcut `z = η + (y−μ)/w` breaks for Gamma where `g'(μ)` is negative.
- Dispersion φ (Pearson χ²/dof) scales SEs for Gaussian/Gamma; φ=1 for Binomial/Poisson.

### Key Lessons
- A recovery test that only asserts *finiteness* cannot catch a weight/sign bug — assert *accuracy* (recovery within tolerance) for numerical estimators.
- Milestone completion is decoupled from crate release here: the `v*` tag triggers crates.io publish, so tagging is an operator ship-time step, not part of `/gsd-complete-milestone` (tag intentionally skipped).

### Cost Observations
- Model mix: orchestration on Opus; researchers/executors/reviewers/verifiers on Sonnet; plan-checker on Haiku.
- Notable: code-review + adversarial re-review cycle (find → fix → re-review clean) was the highest-leverage spend this milestone — it converted a silently-wrong Gamma GLM into a correct one.

---

## Milestone: v0.22.0 — PACE Sparse FPCA & Elastic Multinomial

**Shipped:** 2026-08-19
**Phases:** 2 | **Plans:** 2

### What Was Built
- FPCA-01: `pace_fpca` unified PACE sparse FPCA (new `pace_fpca.rs`) — smoothed mean + cov-surface eigendecomposition + newly-implemented BLUP conditional-expectation scores + Ω prediction-variance bands.
- REG-03: `elastic_multinomial` one-vs-rest over the unchanged binary `elastic_logistic` — completes the elastic-regression family.
- Additive/non-breaking, no new dependency; full suite 2107 + clippy + serde-feature build green.

### What Worked
- Reuse-first held: each phase = 1 plan. Tracer-first sequencing de-risked the PACE algorithmic core early.
- Code review again earned its keep — 3 blockers across the two phases that the `linalg,parallel` gate could NOT catch: a NaN-mean → all-NaN silent result (Phase 26) and a **serde-feature compile break** (Phase 27, only surfaces under `--features serde`). Both fixed with regression tests + an explicit `--features serde` build check added to the gate.
- Batched-single-commit fix strategy (vs atomic-per-finding) avoided the code-fixer stall seen in v0.21.0.

### What Was Inefficient
- **The R-BACKLOG was wrong again** — it claimed `spm::partial::conditional_expectation` provided PACE reconstruction; it does not exist (same class as v0.21.0's `statrs` claim). The BLUP scorer had to be implemented, not orchestrated. Lesson: verify backlog "reuse" claims against the actual codebase during discuss/research, not execution.
- The `gsd-verifier` stalled on Phase 26/27's slow full-suite run (600s watchdog). Worked around by writing VERIFICATION.md from independently-run gate evidence (clippy + suite + targeted tests) — the orchestrator filesystem-fallback pattern.

### Key Lessons
- Feature-gated code (`serde`) needs a feature-matrix build in the gate — a default-feature clippy/test pass is not sufficient. Added `cargo build --features serde,linalg,parallel` to the review verification.
- Non-finite-input guards remain the recurring bug class (NaN mean here; NaN bandwidth/response in v0.21.0) — validate finiteness first, before per-case guards.

### Cost Observations
- Model mix: orchestration on Opus; planners on Opus; researchers/executors/reviewers/verifiers on Sonnet; plan-checker on Haiku. Skipped the researcher for the pure-reuse REG-03 phase (context economy) — plan-checker still gated it.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v0.14.0 | 9 | 21 | First GSD milestone: tracer-first phases, audit-only scope, milestone-audit gate before archive |

### Cumulative Quality

| Milestone | Deliverables | Requirements | Zero-src-edit |
|-----------|--------------|--------------|---------------|
| v0.14.0 | AUDIT-REPORT.md + BACKLOG.md | 13/13 satisfied | yes (audit-only) |

### Top Lessons (Verified Across Milestones)

1. Milestone-level audit catches cross-artifact defects that phase verification misses. *(v0.14.0 — revisit as more milestones ship.)*
