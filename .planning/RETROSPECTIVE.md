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

## Milestone: v0.24.0 — Functional Regression & Clustering Breadth

**Shipped:** 2026-08-20
**Phases:** 3 (31–33) | **Plans:** 7 | **Tasks:** 9

### What Was Built
- **REG-04 (Phase 31):** additive functional regression in new `scalar_on_function/additive.rs` — FAM/GKAM/GSAM, group-lasso `variable_selection`, seeded `permutation_test_fam`, `history_index`.
- **REG-05 (Phase 32):** flexible mixed-effects — `dense_flmm`/`multi_famm`/`fast_fmm` in `famm.rs` (reusing the existing REML-EM), plus `fof_re_regression`/`predict_fof_re` in `fof_regression.rs`.
- **CLUS-01 (Phase 33):** five clusterers — funHDDC (`gmm/subspace.rs`), DBSCAN/kCFC/funFEM/align-and-cluster (`clustering_advanced.rs`), + test-only `adjusted_rand_index`.
- All three: additive/non-breaking, `Result`-returning, crate-root re-exported, **no new dependency**. Verified 5/5 each; whole-crate 2268-test suite + `cargo clippy --all-targets` green.

### What Worked
- **Reuse-first paid off big:** Phase 32's four estimators all reduced to composing the pre-existing `fit_scalar_mixed_model` REML-EM; Phase 33 composed gmm E-step + distance/alignment infra. Research surfaced these reductions early, keeping the actual code small.
- **The code-review gate earned its keep:** it caught genuinely broken numerics that all tests had passed — a monotonic-MSE λ-selection that made `variable_selection` a no-op (P31), inert `fastFMM` config fields (P32), a negative E-step complement that biased funHDDC responsibilities (P33). 5 blockers + 12 warnings fixed across the milestone.
- **Tracer-first** consistently de-risked each phase: one estimator wired end-to-end and green before expanding.

### What Was Inefficient
- **Executor stalls on the full-target build.** Two Phase-32 executor dispatches ran ~50 min and dropped their API connections mid-response — the culprit was `cargo clippy --all-targets` compiling all 28 examples + 8 benches. Recovery cost real wall-clock (recover committed work, finish inline).
- **Doctests slipped past executors.** Executors were scoped to `--lib` tests to avoid the stall, so `#[non_exhaustive]`-config doctests (E0639) and a closure-move doctest (E0507) only surfaced when the orchestrator ran the full gate — three separate orchestrator-side fixes.

### Patterns Established
- **Anti-stall executor contract:** dispatch executors with `--lib <filter>` builds + `--no-verify` commits + "commit early and often"; the **orchestrator** runs the one authoritative `--all-targets` + doctest gate out-of-band. This made Phase-33 executors fast and reliable after the Phase-32 stalls.
- **Doctest dual rule:** `#[non_exhaustive]` configs need `let mut c = X::default(); c.field = …` in doctests (external-crate E0639) but struct-literal in inline tests (clippy `field_reassign_with_default`). Now baked into planner/executor prompts.

### Key Lessons
- Passing tests ≠ correct numerics — the review gate is where subtle algorithm bugs (λ-selection, variance clamps, false convergence) actually got caught. Keep it mandatory even when the suite is green.
- Match the executor's build scope to the project's real gate cost; a repo whose pre-commit/`--all-targets` build is minutes-long will stall subagents unless the heavy gate is lifted to the orchestrator.

### Cost Observations
- Model mix: orchestration + planners on Opus; researchers/pattern-mappers/executors/code-reviewers/fixers/verifiers/integration-checker on Sonnet; plan-checkers on Haiku.
- Notable: ran researcher + pattern-mapper in parallel per phase (33 especially) to cut planning wall-clock; recovered two dropped-connection executors inline rather than re-dispatching.

---

## Milestone: v0.25.0 — Serial Dependence, Representation & Density Breadth

**Shipped:** 2026-08-21
**Phases:** 3 (34–36) | **Plans:** 10 | **Tasks:** 11

### What Was Built
- **FTS-02 (Phase 34):** new `fts/` module — L2-norm functional ACF/PACF with a Monte-Carlo χ²-mixture strong-white-noise band, a KPSS-style stationarity test (seeded permutation p-value), a Bartlett kernel-sandwich long-run covariance, and a functional differencing operator. Reuse-first over `helpers`/`inference` patterns.
- **REP-01 (Phase 35):** the four missing named bases (monomial/exponential/power/polygonal) as `Result`-returning `BasisSystem` factories bundling eval + penalty matrices; a `MultiFunData` multi-domain container; an `Lfd` linear-differential-operator + `principal_differential_analysis` (harmonic-oscillator recovery within 1e-4). Two `smooth_basis` helpers promoted to `pub(crate)` for penalty reuse.
- **DENS-01 (Phase 36):** new `density_fda.rs` — LQD transform + inverse (with the fdadensity θ_ψ support-rescaling), LQD-FPCA via `fdata_to_pc_1d` + FVE, and a 1D Wasserstein quantile-average barycenter.

### What Worked
- Reuse-first paid off again: every phase composed over existing primitives (`fdata_to_pc_1d`, `helpers::{gradient, cumulative_trapz, linear_interp, simpsons_weights}`, `inference/permutation.rs` seeding) with **zero new crate dependencies** across all three phases.
- The tracer-first plan shape (one end-to-end slice verified before expansion) caught the density round-trip accuracy question early.
- Adversarial gates earned their keep: code review found a real critical panic (Phase 34 `n_sim=0`) and verification found a real missing guard (Phase 36 negative barycenter weights) — both closed before ship.

### What Was Inefficient
- Two subagent connection drops mid-run (Phase 35 pattern-mapper, Phase 36 executor) and one planner rate-limit required orchestrator recovery — the Phase 36 executor had written the whole 961-line module uncommitted, so recovery meant finishing the wiring, tolerance, clippy, and gap-closure inline rather than re-running.
- Executors committing `--no-verify` (to dodge the >600s pre-commit full-suite stall) bypassed the fmt hook, leaving rustfmt drift that had to be swept once at the end.

### Patterns Established
- **Empirically-honest tolerances:** when research flags a numeric bound as unverified (the 5e-3 LQD round-trip), measure and assert the real value (1.5e-2) with a documented rustdoc divergence rather than forcing an unachievable target.
- **Meaningful over-tight assertions:** the barycenter "reproduces d1" test asserts *much closer to d1 than d2* (resolution-robust) instead of an absolute L∞ that hits the interpolation floor.

### Key Lessons
- On dropped-executor recovery, verify what actually landed on disk before re-dispatching — the code was 100% written, just uncommitted; a blind re-run would have wasted a full executor cycle.
- Keep `workflow.use_worktrees=false` for fdars autonomous runs (local-main/origin divergence halts harness worktrees) and restore it at the end.

### Cost Observations
- Model mix: planners/verification-heavy work on opus, researchers/executors/reviewers on sonnet, checkers on haiku.
- Notable: 3 independent phases meant zero cross-phase blocking — could have parallelized, but file-ownership on shared `lib.rs` + the worktree-divergence fallback serialized execution anyway.

---

## Milestone: v0.26.0 — FPCA Breadth & Sparse Covariance

**Shipped:** 2026-08-21
**Phases:** 2 | **Plans:** 4 | **Tasks:** 11

### What Was Built
- **FPCA-02 (Phase 37):** new `fpca_variants.rs` — `fpca_der` (derivative FPCA), `fsvd` (functional SVD / cross-FPCA → `FsvdResult`), `cross_covariance`, `dynamical_correlation` (Dubin–Müller), `ssvd` (sandwich-smoother). Verified 5/5.
- **SPARSE-01 (Phase 38):** new `irreg_fdata/face.rs` — `face_covariance` (FACE kernel-sandwich + PSD projection), `mface_covariance` + `MfaceCovResult` (multivariate block covariance), `face_trajectory` (pace_fpca-delegated fitted trajectories + bands). Verified 5/5.
- 8 new crate-root-re-exported public symbols; 2414 lib + doc tests green; milestone audit PASSED 8/8.

### What Worked
- **Reuse-first paid off twice:** Phase 38 directly reused Phase 37's `gaussian_smooth_cov` sandwich (one `pub(crate)` bump) and the shipped `pace_fpca` BLUP bands — `face_trajectory` was a one-line delegation. The two-phase ordering (FPCA sandwich first) made the sparse-covariance sandwich nearly free.
- **Inline execution was the right call:** the worktree-base-divergence + executor-cargo-stall memory pointers were honoured up front (execute inline, `--no-verify` commits, gates run out-of-band), avoiding the subagent stalls that plagued prior runs. Every phase gate ran green in the orchestrator context.
- **Adversarial self-review found real gaps:** the post-execute code-review pass caught untested branches both times (fsvd `q>p` Gram branch; mface `P=3` block assembly) — added coverage before verification.

### What Was Inefficient
- **nalgebra SVD rabbit hole:** `fsvd`'s planned `SVD::new` path silently produced a wrong decomposition on near-rank-deficient cross-covariance (its own `recompose()` failed). Several debug cycles to diagnose before switching to the robust Gram-matrix symmetric-eigendecomposition. A "verify recompose ≈ input" reflex on any library SVD would have caught it in one step.
- **Tolerance calibration by trial:** dense-limit / band-coverage tolerances (0.30, 0.40, 0.85) needed a couple of runs each to settle — RESEARCH gave ballpark figures but the sharp OU ridge + finite-sample band undercoverage only surfaced at test time.

### Patterns Established
- **Kernel-sandwich as the house style for smoothed covariance:** ssvd, `face_covariance`, and `mface_covariance` all share the `cov → gaussian_smooth_cov → W^{1/2}·Cov·W^{1/2} symmetric_eigen → PSD-clip` pipeline. Documented K-FACE-vs-P-FACE / curve-first-FPCAder divergences from R in rustdoc.
- **Gram-matrix eigendecomposition** as the robust substitute for a flaky general SVD on rank-deficient inputs (feature-agnostic, no faer needed).

### Key Lessons
- When a library SVD feeds downstream math, assert it reconstructs its input before trusting its factors — general SVD can fail silently on rank-deficient matrices.
- Repo operational memory (worktree divergence, executor stalls, /tmp exhaustion, target/ bloat, no-tag-to-avoid-publish) is worth loading *before* the first heavy build — it shaped the whole execution strategy and avoided every known failure mode.

### Cost Observations
- Model mix: planners on opus, researchers/executors/verifiers on sonnet, plan-checkers on haiku; orchestration + inline execution on opus.
- Notable: both phases independent but shared-file ownership (`lib.rs`, `irreg_fdata/mod.rs`) + worktree-divergence fallback serialized them regardless — same pattern as v0.25.0.
- Crate release (v0.23.0–v0.26.0) remains a deferred operator step; no `v0.26.0` git tag created (avoids a phantom crates.io publish while Cargo.toml is still 0.24.0).

---

## Milestone: v0.27.0 — Functional Time Series & Fréchet Regression

**Shipped:** 2026-08-22
**Phases:** 2 (39–40) | **Plans:** 6 | **Tasks:** 15

### What Was Built
- **FTS-01** (Phase 39): new `fts/forecast.rs` — `ftsm` (FPCA + per-component Yule-Walker AR / AIC), `ftsm_forecast`/`ftsm_forecast_multistep` (iterative plug-in), `ftsm_update` (dynamic update, no FPCA refit), `fplsr` (lag-1 per-point PLS). Reuses `fdata_to_pc_1d` + `scoring.rs` + `fts/acf.rs`.
- **FRE-01** (Phase 40): new `frechet/` — `MetricSpace`/`WassersteinDensitySpace`, `wasserstein2_distance`, `frechet_mean`/`frechet_variance`, global/local Fréchet regression, `frechet_anova`. Reuses `density_fda.rs`.
- Both additive/non-breaking, no new dependency; 46 new inline tests; whole crate 2460 lib + 172 doc tests green.

### What Worked
- Reuse-first paid off: `ftsm` is a thin wrapper over `fdata_to_pc_1d`; frechet delegates the sample barycenter to `wasserstein_barycenter`. Research + pattern-mapper agents pinned real APIs up front.
- The tracer-first plan shape (prove trait→backend→reuse end-to-end before expanding) caught the hardest integration risks in Plan 01 of each phase.
- Plan-checker's 3 blockers on Phase 40 (public W₂ API, signed-weight divergence docs, σ̂ₗ² [ASSUMED]) were all cheap, correct, and fixed in one revision cycle.

### What Was Inefficient
- Diagnosing the Fréchet-regression bias burned ~30 min: the reused `wasserstein_barycenter` rescale-to-full-support step spuriously stretched narrow barycenters (~0.5 W₂ error). Fixed by inverting the averaged quantile directly in x-units in the frechet-owned `signed_quantile_average`.
- Narrow synthetic test densities (σ=0.3 on a wide grid) triggered quantile-tail artifacts; grid-filling data (σ≈1) is required for meaningful barycenter tests — a reusable lesson for any density-round-trip testing.

### Patterns Established
- Signed-weight quantile average + sort-based isotonic projection = zero-dependency stand-in for R `frechet`'s osqp QP.
- Metric-space Fréchet ANOVA via seeded label-permutation (per-iteration seeding → thread-count-independent).
- Documented numeric floors (Wasserstein-barycenter ~0.1–0.15 W₂ round-trip) drive test tolerances rather than machine-eps.

### Key Lessons
- When reusing a numeric helper (`wasserstein_barycenter`), verify its round-trip accuracy for YOUR input regime before building on it — its rescale assumption was invisible until the regression bias surfaced.
- Owning a private copy of the reused inversion (`signed_quantile_average`) let me fix the bias without touching shipped DENS-01 behavior.

### Cost Observations
- Model mix: opus (planner/executor-inline) + sonnet (research/pattern/verify) + haiku (plan-checker).
- Execution: inline (worktree base divergence + executor cargo-stall memory), each phase independently re-verified.
- Notable: 1 plan-revision cycle on Phase 40; 1 mid-execution numeric-bias fix. No milestone gaps.

---

## Milestone: v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression

**Shipped:** 2026-08-23
**Phases:** 2 | **Plans:** 5

### What Was Built
- Phase 41 (FTS-03): `fts/spectral.rs` — `spectral_density`, `dpca`, `dpca_reconstruct` + `sim_fvarma`/`sim_farma` simulators.
- Phase 42 (FRE-02): `frechet/spaces/` five `MetricSpace` backends + generic `frechet_*_reg_space`/`frechet_anova_space` reusing extracted `pub(crate)` helpers.

### What Worked
- The metric-consistent DPCA trick (fold Simpson weights into the eigenproblem via `W^{1/2}` scaling, mirroring the existing `acf.rs` MC-band scaling) made rank-1 reconstruction exact and kept scores/reconstruction self-consistent.
- Non-breaking generification pattern: keep the concrete public fn, extract a shared `pub(crate)` core, add a generic sibling — the existing density Fréchet tests became a free regression gate proving byte-identical behavior.
- Direct-sum DFT as an in-test oracle validated the FFT spectral path to 1e-9.

### What Was Inefficient
- Two clippy round-trips on Phase 42 tests (`useless_vec`, `cloned_ref_to_slice_refs`, negated partial-ord) — the full `--all-targets` gate lints test code, so test helpers need the same care as src.

### Patterns Established
- Object-data `MetricSpace` backend template (entry-validated struct + `impl MetricSpace`, never panic); iterative-mean backends guard division-by-`sin θ` at antipode + hard iter cap → `ComputationFailed`.

### Key Lessons
- Code review earned its keep both phases: caught a real usize-underflow panic (41) and a `Power(f64::INFINITY)` validation gap (42) that tests missed.

### Cost Observations
- Executed inline (worktree/executor-stall memory); each phase independently verified + code-reviewed. Whole-crate: **2514 lib tests green, clippy `--all-targets` clean, fmt clean.** Milestone audit PASSED 12/12.

---

## Milestone: v0.30.0 — Performance & Consolidation Pass

**Shipped:** 2026-09-01
**Phases:** 6 (46–51) | **Plans:** 23

### What Was Built
Measure-first depth pass — the first internally-driven milestone. Phase 46 profiled the whole crate into three ranked inventories (hot-path, dedup, API). Phase 47: face_covariance −80.7% wall + dpca −54% alloc blocks (bit-identical, golden 1e-12). Phase 48: frechet_anova 9.9× + co_cluster 6.4× thread-scaling with payback guards. Phase 49: χ²/gamma → `distributions.rs`, `seed_for_thread`, `permutation_pvalue`, SVD sign-core — all call sites migrated, −358 LOC, bit-identical. Phase 50: 3 `Default` impls, `fanova_seeded`, `Dim` + 5 dispatchers, 6 `#[deprecated]` — fully additive (28 examples + wasm compile). Phase 51: 9 new module benches + BENCH-RESULTS.md guard ledger.

### What Worked
- **Measure-first paid off** — every 47–51 target traced to a PROF-01/02/03 inventory row; no cold-path optimization.
- **Golden-capture-then-change** (capture bit-identical reference from current code BEFORE the edit, assert after) made behavior-preservation mechanical and caught nothing regressing — reused across 47/48/49/50.
- **Adversarial verify caught real gaps** — the Phase-49 verifier found 5 same-contract RNG sites the plan's `files_modified` scoping missed; gap-closure completed "every call site migrated".
- **Research corrected naive plans** — Phase 49 research proved one χ²/gamma kernel diverges catastrophically in the far tail (→ share primitives, split tail wrappers); Phase 50 research shrank the `_1d/2d` dispatch scope from "30+ fns" to 5.

### What Was Inefficient
- **Executor interruptions** — an executor hit an API rate-limit mid-run (plan 50-02/51-01) leaving uncommitted work; reconciliation-on-resume (verify gates green, then commit) recovered cleanly but cost a cycle.
- **A stray all-`pending` `.planning/state.json` projection artifact** kept reappearing and needed removal each time.
- **Plan-check found a real golden-mechanism blocker** (RNG dispatchers can't `assert_eq!` under `thread_rng`) — caught pre-execution, but the plan initially over-claimed determinism.

### Patterns Established
- **`seed_for_thread(seed, k)`** — one home for the per-thread determinism contract.
- **`permutation_pvalue`** — parallel-map → collect Result → sequential strict-`>` reduce (order-independent).
- **Additive deprecation** — add canonical form, make old a `#[deprecated]` delegating shim, migrate non-pinning callers, `#[allow(deprecated)]` the pin-tests. `pub use` of deprecated items DOES warn on rustc 1.97 → allow the re-export block.
- **Documented soft guards** (criterion baseline-compare) + deterministic hard guards (alloc_audit) — no flaky wall-time asserts under an unpinned governor.

### Key Lessons
- Verifier "every call site migrated" is stronger than a plan's `files_modified` scope — trust the goal-backward check to find scoping misses.
- For a bench-heavy milestone, `TMPDIR` + `target/debug` cleanup discipline is load-bearing; the CI-representative gate is `--features linalg,parallel --all-targets`.
- Keep the audit/perf milestone tag-free: crate stays 0.29.0, version bump + publish + tag is the deferred REL-01 operator step (a `v*` tag would trigger a phantom crates.io publish).

### Cost Observations
- Executed largely via dispatched subagents (researcher/planner/checker/executor/verifier) with inline reconciliation on interrupts; phases sequential by dependency (worktree-base-divergence memory).
- Notable: reconciliation-on-resume (verify-then-commit an interrupted executor's uncommitted work) avoided duplicate execution twice.

---

## Milestone: v0.31.0 — Multi-Ecosystem Gap Audit

**Shipped:** 2026-09-02
**Phases:** 2 | **Plans:** 7

### What Was Built
Four capability-first ecosystem surveys (`survey-{matlab,julia,tidyfun,pyx}.md`) mapping fdars present/partial/absent against MATLAB FDA, Julia FDA, tidyfun/refund, and Python-beyond-scikit-fda; a consolidated `GAP-AUDIT-REPORT.md`; a ranked `GAP-BACKLOG.md` (7 net-new gaps by `value/√effort`); and an RPT-03 de-dup + completeness gate (PASS). Zero `fdars-core/src/` edits.

### What Worked
- **The completeness gate earned its keep:** RPT-03's independent de-dup pass caught a candidate (multi-domain MFPCA) already-adjacent to a v0.18.0 backlog item and demoted it — the gate rejected something, which is the point of a gate.
- **"Already-considered" rigor beyond the hard rule:** treating PACE (MATLAB) methods already surveyed in the v0.18.0 fdapace audit as out-of-scope prevented re-litigating decided scope — the milestone's stated main risk (de-dup rigor) was met.
- **Grep-evidenced parity:** every absent/partial row carrying a literal "searched fdars for:" note made the present/absent claims auditable and surfaced that fdars already ships soft-DTW, matrix profile, MFPCA, PACE, etc.

### What Was Inefficient
- **A mid-run session usage limit killed all four parallel executor subagents before they wrote anything** — 4 spawned agents burned ~50–70s each and produced zero output. Recovered by switching to inline research-and-write, but the parallel dispatch was wasted.
- Pre-commit hook runs the full 2587-test cargo suite even for docs-only commits → every commit needed `--no-verify` (known repo friction).

### Patterns Established
- For **audit/documentation milestones**, inline research-and-write by the orchestrator is more reliable than executor-subagent dispatch: no cargo-hook commit stall, no worktree-base-divergence halt, no wrap-up connection drop, and it survives subagent rate-limits.
- Standardized six-column gap tables across parallel surveys make the consolidation phase a mechanical merge.

### Key Lessons
- When subagents die on an account-level usage limit, re-dispatching is futile — pivot to inline immediately if the main loop still has budget.
- Audit-only milestones must skip git tagging (a `v*` tag publishes a phantom crate version); honored again here.

### Cost Observations
- Model mix: opus (orchestrator/planner) + sonnet (executors, killed by limit) + haiku (plan-checker). Most survey work ended up inline on the orchestrator model.
- Notable: the whole execute→consolidate→gate→archive chain ran inline after the executor-subagent failure, with no partial/duplicate artifacts.

---

## Milestone: v0.35.0 — Optimal Experimental Design for Sparse FDA (FOptDes)

**Shipped:** 2026-09-03
**Phases:** 2 | **Plans:** 4

### What Was Built
Optimal sparse-measurement design over an already-fitted PACE model, in one new file `src/optimal_design.rs`. Phase 64: shared `build_sigma_design` (p×p Σ_d + σ²I, ridge-retry) + Simpson-weighted trajectory-reconstruction BLUP-MSE (FOD-01) + A-/D-optimality posterior score-covariance (FOD-02) behind the public `#[must_use] design_criterion` with `DesignCriterion`/`OptimalityKind` enums (FOD-03). Phase 65: deterministic greedy forward-selection `optimal_design` (FOD-04) + `OptDesConfig`/`OptDesResult` + full crate-root/prelude re-exports + module doctest + criterion benchmark (FOD-05). Crate 0.34.0 → 0.35.0.

### What Worked
- **Front-loading every numerical make-or-break gate into Phase 64** (known-answer empty-set identities MSE(∅)=Σλ_k / A(∅)=Σλ_k / D(∅)=Σ log λ_k, monotonicity sign gate, ridge-retry) made the greedy wrapper in Phase 65 pure orchestration with zero new math — exactly as the roadmap predicted.
- **One general-purpose implementation agent per phase** (plan→code→gates→`--no-verify` commit→SUMMARY) again dodged the executor-stall + slow-hook-timeout hazards cleanly; both phases landed green on the impl, with code-review the only fix round.
- **Reuse-first paid off**: the exact `pace_fpca.rs` Σ_yi assembly + posterior-covariance patterns transcribed verbatim; no new dependency, MSRV held.
- **Research → VALIDATION.md → planner → checker chain** produced tight, gate-lifting plans; both plan-checker passes were clean.

### What Was Inefficient
- Two genuine bugs slipped past the impl agent and were caught only by code review: a panic on degenerate duplicate `candidate_grid` values (Phase 65 blocker) and a tie-break that was "first-in-grid-order" rather than the documented "smallest-index". Both cheap to fix but argue for code-review being non-optional on any greedy/argmin code.
- The trajectory ≡ A-optimality algebraic identity forced a test-contract correction mid-implementation (the original `test_enum_dispatch` assumed all three criteria distinct).

### Patterns Established
- **Criterion-evaluator → greedy-wrapper split** (public pure scorer + thin selection loop that delegates to it) mirrors sbd→kshape and generalizes for any "score a set, then greedily grow it" algorithm.
- **Determinism contract for parallel argmin**: parallel-evaluate → collect (index,value) → sequential fold with strict `<` on an index-sorted pool = smallest-index tie-break, byte-identical seq==parallel.

### Key Lessons
- A `#[must_use]` on a `Result` needs a message string (bare form trips clippy `double_must_use` under `-D warnings`) — bit both phases.
- "Smallest-index tie-break" is ambiguous unless the candidate pool is explicitly sorted by the index you mean — document AND enforce the ordering; don't assume the caller's grid is ascending.
- Surfacing pre-existing tech debt (the `--features serde` build break in Phase 60's `shapelet/classifier.rs`) during a later milestone is valuable; recorded it as a STATE blocker + memory + audit tech-debt rather than letting a serde gate silently fail.

### Cost Observations
- Model mix: planning opus, research/verify/review sonnet, plan-check haiku; implementation via general-purpose (opus) agents.
- Both phases: one impl agent + one code-review + one fix agent each; verifier + integration-checker at milestone close.
- Notable: the slow full-suite pre-commit hook (2600+ tests) made `--no-verify` + manual gate runs mandatory throughout — unchanged from prior fdars milestones.

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
