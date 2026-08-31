# Phase 49: Code Consolidation / Dedup - Context

**Gathered:** 2026-08-31
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — behavior-preserving refactor driven by the PROF-02 inventory

<domain>
## Phase Boundary

Factor the four PROF-02-ranked duplicated numerical/statistical-test machineries into shared
`pub(crate)` helpers and migrate every call site — reducing surface area and drift risk while
leaving **all observable behavior unchanged** (numeric outputs bit-identical or provably-equivalent
within documented tolerance). Requirements: CONS-01 (numerical machinery) + CONS-02 (statistical-test
scaffolding). Consumes `.planning/phases/46-whole-crate-profiling-measurement/PROF-02-dedup-inventory.md`.

Out of scope: the "Already Consolidated" items (Simpson/quadrature `simpsons_weights`, Cholesky
`linalg.rs`, FPCA scoring `fdata_to_pc_1d`) — PROF-02 confirmed no local reimplementations in the 9
new subsystems. No public API/signature changes (that is Phase 50). No new crate dependency.

</domain>

<decisions>
## Implementation Decisions

### Consolidation Scope (operator-confirmed)
- **All 4 PROF-02 targets are in scope** (both CONS-01 and CONS-02 requirements met this phase):
  - **#1 χ²/F survival + regularized incomplete gamma** (CONS-01, HIGH leverage — 2 independent
    hand-rolled gamma kernels, drift-prone): `inference/dist.rs` + `spm/chi_squared.rs`.
  - **#4 SVD sign-fix** (CONS-01, correctness-critical): promote `fix_svd_signs` to `pub(crate)`,
    migrate the inline mirror in `pace_fpca.rs:219`.
  - **#2 Permutation-test loops** (CONS-02): shared `permutation_pvalue` helper unifying the
    shuffle→recompute→count→`(1+count)/(1+n_perm)` scaffold.
  - **#3 Per-thread seeded RNG** (CONS-02): shared `seed_for_thread(seed, k)` centralizing the
    `StdRng::seed_from_u64(seed + k)` determinism contract.

### χ²/gamma helper location (operator-confirmed)
- **New module `src/distributions.rs`** hosts `reg_gamma_p`, `reg_gamma_q`, `chi2_sf`, `chi2_cdf`,
  `chi2_quantile` (all `pub(crate)`). Keeps `helpers.rs` focused on quadrature/RNG.

### Behavior-preservation gates (Claude's Discretion on exact mechanics)
- Capture golden/equivalence references from the CURRENT (pre-refactor) code BEFORE migrating each
  call site; assert bit-identical (or documented tolerance) after. Mirror the Phase 47/48
  golden-equivalence-then-migrate pattern (`tests/equivalence_phase49.rs`).
- Full suite green under BOTH feature configs (`--features linalg,parallel` AND
  `--no-default-features --features linalg`) + `clippy --all-targets` clean at every commit.
- `cargo fmt -p fdars-core` + `git commit --no-verify` per commit; prefix cargo with
  `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`.

### CRITICAL constraints (behavior-preservation linchpins — must be honored by the plan)
- **`t_perm_test` / `f_perm_test` (`inference/permutation.rs`) use a SINGLE shared *advancing* `StdRng`
  across permutations** — NOT per-perm reseeding. Forcing them into a per-perm-reseed
  `permutation_pvalue` helper WOULD change their returned p-values (a numeric-output change). The
  shared helper must either (a) support both seeding modes (shared-advancing vs per-perm-reseed), or
  (b) explicitly exclude `t_perm/f_perm` and consolidate only the per-perm-reseed sites. Phase 48
  already flagged `t_perm/f_perm` as deferred *from parallelization* for exactly this reason.
- **`frechet_anova` (`frechet/anova.rs`) was parallelized in Phase 48** with per-perm reseed
  (`seed.wrapping_add(perm)`) behind `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200`. If its loop is
  migrated to the shared helper, the Phase-48 goldens (`tests/equivalence_phase48.rs`,
  `golden_frechet_anova_*`) must stay bit-identical AND the payback-threshold/parallel path preserved.
- **`explain/importance.rs` + `explain_generic/importance.rs`**: Phase 48 explicitly deferred
  `explain/importance` parallelization to "Phase 49 CONS-02 folds it into the already-parallel generic
  path". The permutation-loop consolidation is the vehicle for that fold-in.
- **Seeded-RNG consolidation must preserve the exact offset formula** `seed + k as u64` (determinism
  contract) — `seed_for_thread` returns `StdRng::seed_from_u64(seed + k as u64)`, nothing else.

</decisions>

<code_context>
## Existing Code Insights

### PROF-02 anchors (from the inventory)
- χ²/gamma: `inference/dist.rs:99` (`chi_square_sf`, SF-oriented, `gamma_p_series`/`gamma_q_cf`) vs
  `spm/chi_squared.rs:164` (`chi2_cdf`, CDF-oriented, `regularized_gamma_p`); `chi2_quantile` at
  `chi_squared.rs:189` (Wilson-Hilferty + Newton). `chi_square_sf_df` (real df, Satterthwaite) at
  `dist.rs:118`.
- Permutation loops: `inference/permutation.rs:175,238` (seq), `frechet/anova.rs:171` (seq → now
  parallel via Phase 48), `explain/importance.rs:131,221` (seq), `function_on_scalar.rs:831,847`
  (par), `famm.rs:861` (par), `explain_generic/importance.rs:68` (par).
- Seeded RNG: 10 thread-offset sites (`gmm/em.rs`, `clustering.rs`, `coclustering.rs`, `alignment/*`,
  `explain/*`, `scalar_on_function/bootstrap.rs`, …).
- SVD sign-fix: canonical `regression.rs:180` (`fix_svd_signs`, called `:381,:991`); inline mirror
  `pace_fpca.rs:219`.

### Established patterns to reuse
- Feature-gated parallelism via `parallel.rs` macros (`iter_maybe_parallel!` etc.).
- Golden-equivalence-then-migrate (Phase 47 `equivalence_phase47.rs`, Phase 48 `equivalence_phase48.rs`).
- `pub(crate)` shared helpers in `helpers.rs` / `linalg.rs` are the canonical precedent.

### Proposed signatures (PROF-02)
- `pub(crate) fn reg_gamma_p(a: f64, x: f64) -> f64`, `reg_gamma_q(a, x) -> f64`,
  `chi2_sf(x, df) -> f64`, `chi2_cdf(x, df) -> f64`, `chi2_quantile(p, k: usize) -> f64` (distributions.rs)
- `pub(crate) fn permutation_pvalue<F: Fn(&[usize]) -> f64 + Sync>(observed, n, n_perm, seed, stat) -> f64`
  (feature-gated parallel) — subject to the t_perm/f_perm seeding-mode constraint above.
- `pub(crate) fn seed_for_thread(seed: u64, k: usize) -> StdRng` (helpers.rs)
- promote `fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)` to `pub(crate)`.

</code_context>

<specifics>
## Specific Ideas

- New module `src/distributions.rs` for the χ²/gamma family (operator choice over extending helpers.rs).
- Author `tests/equivalence_phase49.rs` capturing pre-refactor references for each migrated call site
  (χ² SF/CDF/quantile values, permutation p-values, RNG-stream determinism, SVD signs) and asserting
  bit-identity after migration.

</specifics>

<deferred>
## Deferred Ideas

- Already-consolidated machinery (Simpson/quadrature, Cholesky, FPCA scoring) — confirmed no local
  reimplementations; explicitly out of scope.
- Public API/signature unification and deprecations — that is Phase 50 (API-01/02/03), not this phase.
- If the shared `permutation_pvalue` helper cannot cleanly host `t_perm/f_perm`'s shared-advancing-RNG
  mode without risk, leaving those two sites un-migrated (with a documented rationale) is acceptable —
  behavior-preservation outranks call-site count.

</deferred>
