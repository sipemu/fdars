# Phase 49: Code Consolidation / Dedup - Research

**Researched:** 2026-08-31
**Domain:** Behavior-preserving Rust refactor — numerical special functions, permutation-test scaffolding, seeded-RNG determinism, SVD sign convention
**Confidence:** HIGH (all findings verified by reading the source-of-truth files this session and by a standalone bit-for-bit numerical comparison of the two gamma kernels)

## Summary

Phase 49 factors four PROF-02-ranked duplicated machineries into shared `pub(crate)` helpers with
**zero observable behavior change**. The single most important finding, established by a standalone
bit-for-bit comparison this session, reframes the highest-risk target: **the two χ²/gamma kernels are
NOT interchangeable at bit level.** Their shared `ln_gamma` is bit-identical, but the regularized
incomplete gamma diverges by up to ~51 ULP (max rel err ~6.8e-15) because the two kernels branch
differently between series and continued-fraction — and in the far upper tail the divergence is
**catastrophic** (e.g. χ²-SF at x=70.59, k=1: inference yields `4.397e-17`, the spm-P route yields
exactly `0.0`). Therefore a single canonical kernel **cannot** reproduce BOTH call families
bit-identically. The refactor must share the *primitives* (`ln_gamma`, `gamma_p_series`, `gamma_q_cf`)
while preserving each family's *tail-branch selection* — the SF family keeps its `Q`-direct upper-tail
path, the CDF family keeps its `P`-direct path.

The permutation and RNG targets carry an analogous, already-flagged hazard: two of the six permutation
sites (`t_perm`/`f_perm`) — **and, newly discovered, `explain/importance.rs` and `famm.rs`** — use a
SINGLE *advancing* `StdRng` across the whole loop, not per-perm reseed. A per-perm-reseed helper would
change their p-values. The "explain/importance folds into the already-parallel generic path" plan from
Phase 48 is therefore **not free**: the generic path reseeds per-component (`seed + k`) while
`explain/importance` advances one RNG across all components, so folding would change its output.

**Primary recommendation:** Do the χ²/gamma target as the **tracer slice**, but consolidate only the
*primitives* into `src/distributions.rs`, keeping two thin public wrappers (`chi2_sf` SF-direct,
`chi2_cdf` CDF-direct) so each existing site stays bit-identical. For permutations and RNG, build a
helper that supports **both seeding modes** OR migrate only the per-perm-reseed sites; leave the
shared-advancing-RNG sites (`t_perm`, `f_perm`, `explain/importance`, `famm`) un-migrated with a
documented rationale — behavior-preservation outranks call-site count (CONTEXT.md deferred idea #3).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| χ²/gamma special functions | `src/distributions.rs` (new, crate infra) | `inference/dist.rs`, `spm/chi_squared.rs` (thin wrappers) | Shared numerical primitives belong in cross-cutting infra alongside `helpers.rs`/`linalg.rs`; tail-branch policy stays with each consumer |
| Permutation p-value scaffold | `src/helpers.rs` (or new `src/permutation.rs`) | 6 domain fns | Cross-cutting statistical-test convention `(1+count)/(1+n_perm)` |
| Per-thread seeded RNG | `src/helpers.rs` | 10 thread-offset sites | Determinism contract is crate infra |
| SVD sign convention | `src/regression.rs` (promote to `pub(crate)`) | `pace_fpca.rs` | Canonical already lives in regression; correctness-critical mirror |

## Standard Stack

No new dependencies (CONTEXT.md constraint: "No new crate dependency"). All primitives are hand-rolled
Numerical-Recipes-style code already in-tree. Relevant existing crates only:

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rand 0.8 / rand_distr 0.4 | in Cargo.toml | `StdRng::seed_from_u64`, `.shuffle`, `gen_range` | Existing project RNG; determinism contract built on it `[VERIFIED: fdars-core/src/inference/permutation.rs:14-15]` |
| rayon 1.10 (`parallel` feature) | in Cargo.toml | `iter_maybe_parallel!` macro path | Existing feature-gated parallelism `[VERIFIED: fdars-core/src/frechet/anova.rs:15,190]` |

**Installation:** none — audit/refactor milestone, no crate added.

## Package Legitimacy Audit

Not applicable — this phase installs **no external packages** (CONTEXT.md: "No new crate dependency";
milestone is refactor-only). Skipped per protocol.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CONS-01 | Consolidate numerical machinery (χ²/gamma + SVD sign-fix) into shared helpers, behavior-preserving | Target #1 (§Target 1) and Target #4 (§Target 4) below; bit-identity analysis + measured tolerances |
| CONS-02 | Consolidate statistical-test scaffolding (permutation loops + seeded RNG) into shared helpers | Target #2 (§Target 2) and Target #3 (§Target 3) below; per-site seeding-model inventory |
</phase_requirements>

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **All 4 PROF-02 targets are in scope** (CONS-01 + CONS-02):
  - #1 χ²/F survival + regularized incomplete gamma (`inference/dist.rs` + `spm/chi_squared.rs`).
  - #4 SVD sign-fix: promote `fix_svd_signs` to `pub(crate)`, migrate `pace_fpca.rs:219` mirror.
  - #2 Permutation-test loops: shared `permutation_pvalue` helper.
  - #3 Per-thread seeded RNG: shared `seed_for_thread(seed, k)`.
- **New module `src/distributions.rs`** hosts `reg_gamma_p`, `reg_gamma_q`, `chi2_sf`, `chi2_cdf`,
  `chi2_quantile` (all `pub(crate)`). Keeps `helpers.rs` focused on quadrature/RNG.
- Capture golden/equivalence references from the CURRENT (pre-refactor) code BEFORE migrating each call
  site; assert bit-identical (or documented tolerance) after. Mirror Phase 47/48 pattern in
  `tests/equivalence_phase49.rs`.
- Full suite green under BOTH `--features linalg,parallel` AND `--no-default-features --features linalg`
  + `clippy --all-targets` clean at every commit.
- `cargo fmt -p fdars-core` + `git commit --no-verify` per commit; prefix cargo with
  `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`.

### CRITICAL constraints (behavior-preservation linchpins)
- **`t_perm_test` / `f_perm_test` use a SINGLE shared *advancing* `StdRng`** — forcing per-perm-reseed
  changes their p-values. Helper must (a) support both seeding modes, or (b) exclude these sites.
- **`frechet_anova` was parallelized in Phase 48** with per-perm reseed (`seed.wrapping_add(perm)`)
  behind `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200`. Migration must keep Phase-48 goldens
  bit-identical AND preserve the threshold/parallel path.
- **`explain/importance.rs` + `explain_generic/importance.rs`**: Phase 48 deferred `explain/importance`
  parallelization to "Phase 49 folds it into the already-parallel generic path."
- **Seeded-RNG consolidation must preserve the exact offset formula** `seed + k as u64`.

### Claude's Discretion
- Exact mechanics of the behavior-preservation gates and equivalence-test structure.

### Deferred Ideas (OUT OF SCOPE)
- Already-consolidated machinery (Simpson/quadrature, Cholesky, FPCA scoring) — no local reimpls.
- Public API/signature unification and deprecations — Phase 50 (API-01/02/03).
- If the shared `permutation_pvalue` helper cannot cleanly host `t_perm/f_perm`'s shared-advancing-RNG
  mode without risk, leaving those sites un-migrated (documented) is acceptable.
</user_constraints>

## Architecture Patterns

### Migration order (tracer-first — highest risk first)

```
1. TRACER: χ²/gamma (Target 1)  ── highest numerical risk, resolves the "can one kernel serve both?" Q
   1a. Create src/distributions.rs with SHARED PRIMITIVES only (ln_gamma, gamma_p_series, gamma_q_cf)
   1b. Capture goldens: chi_square_sf / chi_square_sf_df / f_sf / chi2_cdf / chi2_quantile
       + regularized_gamma_p(0.5, x*x) [bootstrap.rs] at representative (x, k) incl. FAR TAIL
   1c. Reimplement inference/dist.rs + spm/chi_squared.rs wrappers on the shared primitives,
       PRESERVING each family's tail-branch selection
   1d. assert_eq! bit-identity for every captured golden (see Equivalence strategy)
2. SVD sign-fix (Target 4)      ── smallest, fully mechanical, bit-identical by construction
3. Seeded RNG (Target 3)        ── mechanical, offset formula preserved verbatim
4. Permutation loops (Target 2) ── highest behavioral subtlety; do LAST, per-perm-reseed sites only
```

### Anti-Patterns to Avoid
- **"One kernel to rule them both":** Do NOT collapse the SF and CDF families onto a single
  `reg_gamma_p`-and-complement path. Measured this session: it changes tail outputs (a `0.0` vs
  `4.4e-17` cliff). Share primitives, keep tail policy per-family.
- **Per-perm reseed forced onto advancing-RNG sites:** changes p-values at `t_perm`/`f_perm`/
  `explain/importance`/`famm`. See Target 2.
- **Folding `explain/importance` into `generic_permutation_importance` blindly:** the two use different
  seeding models (advancing vs per-component reseed) — outputs already differ. See Target 2 fold-in.

## Target 1 — χ²/F survival + regularized incomplete gamma (CONS-01, TRACER)

### Canonical impl choice
**Share the primitives, split the tail policy.** Host in `src/distributions.rs`:
- `reg_gamma_p(a, x)` / `reg_gamma_q(a, x)` — the regularized lower/upper incomplete gamma primitives.
- `ln_gamma(x)` — Lanczos g=7,n=9 (both current copies are **bit-identical** — verified below).
- Two thin χ² wrappers that each preserve their family's existing branch:
  - `chi2_sf(x, df: f64)` — SF-direct: `if x/2 < df/2 + 1 { 1 - P_series } else { Q_cf }`
    (mirrors `chi_square_sf` / `chi_square_sf_df`).
  - `chi2_cdf(x, k)` — CDF-direct: `reg_gamma_p(k/2, x/2)` (mirrors spm `chi2_cdf`).
  - `chi2_quantile(p, k)` — Wilson-Hilferty + Newton, calling `chi2_cdf` internally (mirrors spm).

### Why one kernel cannot serve both (VERIFIED this session)
Standalone bit-for-bit comparison of both current kernels:

| Comparison | Bit-identical? | Max ULP | Max rel err | Note |
|------------|----------------|---------|-------------|------|
| `ln_gamma` (inference vs spm), a∈[0.25,60] | **YES** (all bits equal) | 0 | 0 | The `...809_93` vs `...809_9` literal rounds to the same f64 `[VERIFIED: inference/dist.rs:72 vs spm/chi_squared.rs:30]` |
| regularized `P(a,x)` (inference `1−Q`/`P` vs spm `P`) | **NO** | 51 | 6.841e-15 | Divergence from series-vs-CF branch layout, worst at a=50, x=48.15 |
| `chi_square_sf` (SF-direct Q) vs `1 − chi2_cdf` (P-route) | **NO — catastrophic** | — | 1.0 | Far tail: x=70.59,k=1 → SF=`4.397e-17` vs `0.0`. Upper-tail cancellation floor |

**Interpretation:** the SF family (`inference`, used by `hotelling`, `flm` via `f_sf`, `anova` via
`chi_square_sf_df`) needs its own upper-tail `Q` continued-fraction path to avoid the `1 − P`
cancellation cliff. The CDF family (`spm`) needs its `P`-direct path (its `chi2_quantile` Newton loop
was accuracy-tuned against `chi2_cdf`). Collapsing them would silently change SPM control limits and
inference tail p-values.

### Proposed `pub(crate)` signatures (src/distributions.rs)
```rust
pub(crate) fn ln_gamma(x: f64) -> f64;                 // shared Lanczos (bit-identical to both today)
pub(crate) fn reg_gamma_p(a: f64, x: f64) -> f64;      // lower regularized, series/CF per current spm layout
pub(crate) fn reg_gamma_q(a: f64, x: f64) -> f64;      // upper regularized (1 - reg_gamma_p or direct CF)
pub(crate) fn chi2_sf(x: f64, df: f64) -> f64;         // SF-direct; serves chi_square_sf & _df (df: f64)
pub(crate) fn chi2_cdf(x: f64, k: usize) -> f64;       // CDF-direct; = reg_gamma_p(k/2, x/2)
pub(crate) fn chi2_quantile(p: f64, k: usize) -> f64;  // Wilson-Hilferty + Newton over chi2_cdf
```
Note: PROF-02 proposed `chi2_sf(x, df)` and `chi2_cdf(x, df)` — but the existing SF has both a
`usize`-k form (`chi_square_sf`) and a real-`df` form (`chi_square_sf_df`). Prefer a single
`chi2_sf(x, df: f64)` and call it with `k as f64` at the integer sites — **but verify bit-identity**,
since `chi_square_sf` computes `a = k as f64 / 2.0` from `usize` and `chi_square_sf_df` computes
`a = df / 2.0`; at integer df these are the same f64, so a single `df: f64` entry is safe. `[VERIFIED:
inference/dist.rs:103,125 — both compute a = <df>/2.0]`

### Migration surface (VERIFIED call sites)
- SF family: `inference/hotelling.rs:143`, `inference/anova.rs:175`, `inference/flm.rs:74,232`
  (`f_sf` — F-tail via incomplete beta, keep its own `betai`/`betacf`; it shares only `ln_gamma`).
  `[VERIFIED: grep of inference/]`
- CDF family: `spm/mewma.rs:230`, `spm/ewma.rs:328`, `spm/control.rs:93,181`, `spm/amewma.rs:299`,
  `spm/contrib.rs:391` (all `chi2_quantile`); **`spm/bootstrap.rs:378` calls `regularized_gamma_p(0.5,
  x*x)` directly** — a THIRD primitive consumer, must migrate to `reg_gamma_p`. `[VERIFIED: grep of
  spm/]`

### Equivalence-test strategy (tolerance: BIT-IDENTICAL)
Capture goldens from CURRENT code as `const` (Phase-48 pattern) at points that exercise both branches
AND the far tail. Assert `assert_eq!` (exact f64 bits), NOT tolerance — because the refactor is
primitive-extraction with identical arithmetic order:
- `chi_square_sf` at k∈{1,2,3,10,20}, x∈{0.1, 3.84, 5.99, near-`a+1` boundary, 70.59 far tail}.
- `chi_square_sf_df` at df∈{2.0, 3.7 (Satterthwaite non-integer)} — asserts the real-df path.
- `f_sf` at the tabulated quantiles already in `dist.rs` tests (keep the incomplete-beta path).
- `chi2_cdf` at the values in `chi_squared.rs` tests + boundary; `chi2_quantile` at p∈{0.5,0.95,0.99},
  k∈{1,5,10,20} (its Newton loop must land on the SAME x bits).
- `regularized_gamma_p(0.5, x*x)` for the `bootstrap.rs` half-normal use.

**If any assert_eq! fails**, the refactor changed arithmetic order → fix the wrapper to match the
original branch/operation sequence exactly. Bit-identity IS achievable here because we are moving code,
not re-deriving it.

### Pitfalls
- The two `f_perm` Newton/CF loops have DIFFERENT constants: inference CF `tiny=1e-300`, no `eps` const
  (inline `1e-15`); spm CF `tiny=1e-30`, `eps=1e-14`, plus a `log_prefix < -700.0 → 0.0` underflow
  guard the inference copy LACKS. `[VERIFIED: dist.rs:36,54 vs chi_squared.rs:128,127,155]` These
  constant differences are precisely why the primitives diverge. **The shared `reg_gamma_p`/`reg_gamma_q`
  must adopt ONE constant set; whichever family does not get its constants will shift bits.** Recommended:
  give `chi2_sf` its own SF-tuned CF (`tiny=1e-300`, no underflow guard, inline `1e-15`) and
  `reg_gamma_p` the spm constants — i.e. do NOT force the SF family through `reg_gamma_p`. Keep the SF
  continued-fraction as a private fn in `distributions.rs` used only by `chi2_sf`.
- `ln_gamma` reflection branch differs cosmetically: inference uses `sin().ln()`, spm uses
  `sin().abs().ln()` with a `<1e-30 → INFINITY` guard. `[VERIFIED: dist.rs:84 vs chi_squared.rs:46-49]`
  For all χ² arguments a=k/2>0 the reflection branch (x<0.5) is never hit, so this is dead-code
  divergence — but the shared `ln_gamma` should adopt the spm guarded form (strictly safer) and a
  golden must confirm no χ² value changes.

## Target 2 — Permutation-test loops (CONS-02)

### Seeding-model inventory (VERIFIED — corrects PROF-02's "3 seq / 3 par" split)

| Site | File:line | Loop | RNG source | Seeding model |
|------|-----------|------|-----------|---------------|
| `t_perm_test` | `inference/permutation.rs:173-181` | seq | `StdRng::seed_from_u64(seed)` | **SHARED ADVANCING** |
| `f_perm_test` | `inference/permutation.rs:236-244` | seq | `StdRng::seed_from_u64(seed)` | **SHARED ADVANCING** |
| `fpc_permutation_importance` | `explain/importance.rs:125-141` | seq | `StdRng::seed_from_u64(seed)` advanced across ALL components | **SHARED ADVANCING** |
| `fpc_permutation_importance_logistic` | `explain/importance.rs:215-236` | seq | `StdRng::seed_from_u64(seed)` advanced across ALL components | **SHARED ADVANCING** |
| `famm::permutation_test` | `famm.rs:861-889` | seq | `StdRng::seed_from_u64(seed)` advancing; multi-stat (p covariates) | **SHARED ADVANCING** |
| `frechet_anova` | `frechet/anova.rs:179-193` | par (≥200) | `StdRng::seed_from_u64(seed.wrapping_add(perm))` | **PER-PERM RESEED** |
| `generic_permutation_importance` | `explain_generic/importance.rs:64-80` | par | `StdRng::seed_from_u64(seed.wrapping_add(k))` per component | **PER-COMPONENT RESEED** |
| `function_on_scalar::fanova` | `function_on_scalar.rs:831-853` | seq | **hardcoded LCG, `rng_state=42`, NOT `seed`, NOT rand** | **FIXED-SEED LCG** |

`[VERIFIED: each file:line read this session]`

**Key corrections to PROF-02:** (a) `function_on_scalar.rs:831` and `famm.rs:861` are **sequential**,
not parallel; (b) `function_on_scalar::fanova` does not even take a `seed` parameter — it uses a
hardcoded-`42` LCG, so it is a different RNG family entirely and cannot share a `StdRng`-based helper
without changing its output; (c) only **frechet_anova** and **generic_permutation_importance** actually
use rayon + per-iteration reseed.

### Recommendation: helper covers ONLY per-perm-reseed sites; document the rest

A single `permutation_pvalue(observed, n, n_perm, seed, stat)` with per-perm reseed cleanly and
bit-identically serves **only** `frechet_anova` (single-stat, per-perm reseed, `(1+n_ge)/(1+n_perm)`).
The other five sites each break the abstraction:
- `t_perm`/`f_perm`/`explain-importance`/`famm`: shared-advancing RNG → per-perm reseed changes output.
- `famm`: multi-statistic (counts p covariates in one pass) — signature `-> f64` cannot express it.
- `function_on_scalar::fanova`: fixed-`42` LCG, no seed param — outside the `StdRng` contract.
- `generic_permutation_importance`: reseeds per COMPONENT then advances within — not per-perm.

**Two viable plans (recommend Plan A):**

- **Plan A (lowest risk, recommended):** Ship `permutation_pvalue` and migrate ONLY `frechet_anova`
  (see Target 2b). Leave the other five un-migrated with a one-line rationale comment each citing the
  seeding-model incompatibility. This satisfies CONS-02 ("consolidate the scaffold") for the sites
  where it is behavior-safe, and CONTEXT.md explicitly blesses partial deferral. Net call-site
  reduction is modest but the drift-prone `(1+count)/(1+n_perm)` + reseed convention gets ONE
  authoritative definition.
- **Plan B (more coverage, more risk):** Add a `seed_mode: PermSeed` enum (`Advancing` vs
  `PerPermReseed(offset_fn)`) so the helper hosts BOTH modes. Then `t_perm`/`f_perm`/explain-importance
  can migrate under `Advancing`. Risk: the advancing-RNG variant must pass the *same* `&mut StdRng`
  through the closure and reproduce the exact `shuffle_labels`/`.shuffle` call order — any reordering
  changes bits. Recommend only if the plan budgets a golden per migrated site. `famm` (multi-stat) and
  `fanova` (LCG) stay out regardless.

### Proposed signature (per-perm-reseed variant)
```rust
pub(crate) fn permutation_pvalue<F>(observed: f64, n: usize, n_perm: usize, seed: u64, stat: F) -> f64
where F: Fn(&[usize]) -> f64 + Sync,
// feature-gated parallel via iter_maybe_parallel! with a payback threshold matching
// FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD; each perm builds StdRng::seed_from_u64(seed.wrapping_add(perm)),
// shuffles 0..n, calls stat(&perm_idx); returns (n_ge + 1)/(n_perm + 1).
```

### Target 2b — frechet_anova migration (assess risk)
`frechet_anova`'s loop (`frechet/anova.rs:179-193`) already matches the helper contract exactly
(per-perm reseed `seed.wrapping_add(perm)`, threshold-gated `iter_maybe_parallel!`, `(1+n_ge)/(1+n_perm)`).
Migrating is **low-risk** IF: (1) the helper's threshold equals `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200`
`[VERIFIED: frechet/anova.rs:25]`; (2) the helper preserves the "degenerate permutation → count 0"
conservative skip (`frechet/anova.rs:184-187` returns 0 on `compute_tn_generic` error) — a plain
`stat(&idx) -> f64` closure CAN express this by returning `f64::NEG_INFINITY` on error so the `>=`
comparison yields 0; (3) the Phase-48 goldens stay bit-identical:
`golden_frechet_anova_parallel` (statistic `1.17320834419366224e3`, p_perm `1.00000000000000002e-3`)
and `golden_frechet_anova_below_threshold` (p_perm `1.96078431372549017e-2`) `[VERIFIED:
tests/equivalence_phase48.rs:127-139]`. **Recommendation: migrate frechet_anova; re-run the Phase-48
goldens (they are permanent) as the equivalence gate.** Note `frechet/anova.rs` has a SECOND per-perm
loop at line ~272 (a generic-MetricSpace variant, same reseed pattern) — migrate it too or document.

### Target 2c — explain/importance fold-in (VERIFIED conflict)
Phase 48 deferred `explain/importance` parallelization to "fold into the already-parallel generic path."
**This fold-in is NOT behavior-preserving as-is:**
- `explain/importance.rs::fpc_permutation_importance` advances ONE `StdRng::seed_from_u64(seed)` across
  ALL components and ALL permutations (`explain/importance.rs:125`, single `rng` reused in nested loop).
- `explain_generic::generic_permutation_importance` reseeds PER COMPONENT `seed.wrapping_add(k)`
  (`explain_generic/importance.rs:66`).

These produce DIFFERENT permutation draws → different `permuted_metric`/`importance` values. Folding
`explain/importance` onto the generic path would change its output. **Recommendation: do NOT fold
`explain/importance` into the generic path in this behavior-preserving phase.** Either (a) leave it
sequential+advancing and document, or (b) if parallelization of `explain/importance` is genuinely
required, that is a *behavior-changing* task and belongs to a separate phase with a re-baselined golden
— out of scope for CONS-02's bit-identity mandate. Record this as a correction to the Phase-48 hand-off
assumption.

## Target 3 — Per-thread seeded RNG (CONS-02)

### Canonical impl choice
`pub(crate) fn seed_for_thread(seed: u64, k: usize) -> StdRng` in `src/helpers.rs`, body exactly
`StdRng::seed_from_u64(seed + k as u64)` — nothing else (CONTEXT.md determinism contract).

### Migration + pitfalls
- **`seed + k` vs `seed.wrapping_add(k)`:** the codebase mixes both. `frechet/anova.rs:180` and
  `explain_generic/importance.rs:66` use `.wrapping_add`; PROF-02's proposed body uses plain `+`. These
  are bit-identical UNLESS `seed + k` overflows `u64` (only near `u64::MAX`). To be safe AND preserve
  every site, **the helper body should be `seed.wrapping_add(k as u64)`** — bit-identical to `seed + k`
  for all non-overflowing inputs and matching the `.wrapping_add` sites exactly. `[VERIFIED:
  frechet/anova.rs:180, explain_generic/importance.rs:66]`
- Only migrate the 10 sites that use the *exact* `seed + k`/`seed.wrapping_add(k)` thread-offset form.
  Do NOT migrate the ~88 other `seed_from_u64` calls that use a different argument (e.g. plain `seed`,
  or `seed.wrapping_mul(...)` LCG seeds) — those are not the same contract.
- Equivalence: an RNG-stream determinism golden — for a fixed `(seed, k)`, assert the first N `u64`
  draws from `seed_for_thread(seed, k)` equal the pre-refactor `StdRng::seed_from_u64(seed + k as u64)`
  draws. Cheaper: rely on the existing per-site goldens (frechet_anova, generic_importance) which
  already pin the downstream numeric output.

## Target 4 — SVD sign-fix (CONS-01, correctness-critical)

### Canonical impl choice
Promote `fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)` at
`regression.rs:180` from `fn` to `pub(crate) fn`. `[VERIFIED: regression.rs:180-201]`

### The pace_fpca mirror is NOT a drop-in (VERIFIED)
`pace_fpca.rs:220-234` flips signs on **eigenfunctions only** — there is NO scores matrix at that point
(scores are BLUP-computed later). `[VERIFIED: pace_fpca.rs:218-234]` The canonical `fix_svd_signs`
flips BOTH `rotation` and `scores` in lockstep. So `pace_fpca` cannot call `fix_svd_signs(rotation,
scores, ncomp)` directly.

**Recommendation:** extract the sign-decision core as a small `pub(crate)` helper both can call:
```rust
// distributions.rs is wrong home — put in regression.rs or linalg.rs
pub(crate) fn dominant_sign_negative(col: &FdMatrix, k: usize) -> bool; // true if largest-|·| entry < 0
```
Then `fix_svd_signs` uses it to gate the two-matrix flip, and `pace_fpca` uses it to gate its
single-matrix flip. This kills the drift risk (the max-abs-index + sign rule lives in one place) while
preserving both call sites' exact behavior. Simpler alternative: promote `fix_svd_signs` `pub(crate)`
AND add a sibling `fix_svd_signs_single(mat: &mut FdMatrix, ncomp)` for the eigenfunction-only case,
both sharing the private index/sign core.

### Equivalence
Bit-identical by construction (pure sign flips, same comparison rule). Golden: run `pace_fpca` on a
fixed irregular dataset, assert the eigenfunction matrix bits are unchanged pre/post refactor; and an
FPCA (`fdata_to_pc_1d`) golden asserting rotation+scores signs unchanged. The existing
`test_faer_svd_matches_nalgebra` reproducibility test `[VERIFIED: regression.rs:178]` already guards the
canonical path.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Regularized incomplete gamma | A 3rd new kernel | Reuse the existing series/CF verbatim in `distributions.rs` | Any re-derivation breaks bit-identity; the whole phase is move-not-rewrite |
| Parallel permutation loop | New rayon plumbing | `iter_maybe_parallel!` macro | Existing feature-gated pattern; new plumbing risks parallel-off divergence |
| Thread seed offset | New scheme | `seed_for_thread` wrapping `seed + k` | Determinism contract is fixed; changing it changes every parallel output |

**Key insight:** This is a *code-motion* refactor. Every helper must be arithmetically identical to the
code it replaces, in the same operation order — the temptation to "clean up" the numerics is the primary
failure mode.

## Runtime State Inventory

Not applicable in the data/service sense — this is a pure in-crate `pub(crate)` refactor with no stored
data, service config, OS registrations, secrets, or migration. The one "state" analog:
- **Compiled test goldens:** the Phase-48 `equivalence_phase48.rs` goldens (frechet_anova) are runtime
  behavior locks that MUST still pass after Target 2b — verified they exist and pin exact f64 bits.
- **No build-artifact staleness:** no package rename, no egg-info/binary analog.

## Common Pitfalls

### Pitfall 1: Collapsing SF and CDF gamma families onto one path
**What goes wrong:** χ² tail p-values and SPM control limits shift; far-tail SF hits a `0.0` cliff.
**Why it happens:** `1 − P` loses all precision when P≈1 (upper tail); the SF family avoids this with a
direct `Q` continued fraction.
**How to avoid:** Share only `ln_gamma` + primitives; keep two tail-specialized wrappers.
**Warning signs:** any `assert_eq!` far-tail golden fails; SPM UCL changes in the Nth decimal.

### Pitfall 2: Migrating advancing-RNG permutation sites to a per-perm-reseed helper
**What goes wrong:** `t_perm`/`f_perm`/`explain-importance`/`famm` p-values change.
**Why it happens:** advancing one RNG across the whole loop produces a different draw sequence than
reseeding each iteration.
**How to avoid:** Plan A (migrate only frechet_anova) or Plan B (dual-mode helper with per-site goldens).
**Warning signs:** any permutation p-value golden fails.

### Pitfall 3: `function_on_scalar::fanova` swept into the helper
**What goes wrong:** output changes — it uses a hardcoded-`42` LCG, not `seed`+`StdRng`.
**How to avoid:** exclude it explicitly; it is not part of the `StdRng` seeding contract.
**Warning signs:** `fanova` p-value differs; it has no `seed` parameter to even pass the helper.

### Pitfall 4: pace_fpca sign-fix treated as a 2-matrix flip
**What goes wrong:** compile error or wrong flip — pace_fpca has no scores matrix at that point.
**How to avoid:** share the sign-DECISION core, not the 2-matrix mutation.

## Code Examples

### Shared gamma primitive + two tail wrappers (skeleton)
```rust
// src/distributions.rs  — Source: extracted verbatim from spm/chi_squared.rs (P-direct) and
// inference/dist.rs (Q-direct). Constants preserved per family.
pub(crate) fn ln_gamma(x: f64) -> f64 { /* spm guarded reflection form, g=7 n=9 */ }

pub(crate) fn reg_gamma_p(a: f64, x: f64) -> f64 {
    // spm layout: series for x < a+1, else 1 - CF; tiny=1e-30, eps=1e-14, -700 underflow guard
}
pub(crate) fn reg_gamma_q(a: f64, x: f64) -> f64 { 1.0 - reg_gamma_p(a, x) } // CDF family only

pub(crate) fn chi2_cdf(x: f64, k: usize) -> f64 {
    if x <= 0.0 { return 0.0; } if k == 0 { return 1.0; }
    reg_gamma_p(k as f64 / 2.0, x / 2.0)
}

// SF family keeps its OWN Q continued fraction (tiny=1e-300, no underflow guard) to avoid 1-P cliff:
fn gamma_q_cf_sf(a: f64, x: f64) -> f64 { /* verbatim inference/dist.rs:34-59 */ }
fn gamma_p_series_sf(a: f64, x: f64) -> f64 { /* verbatim inference/dist.rs:16-30 */ }
pub(crate) fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 { return 1.0; }
    let (a, xx) = (df / 2.0, x / 2.0);
    if xx < a + 1.0 { 1.0 - gamma_p_series_sf(a, xx) } else { gamma_q_cf_sf(a, xx) }
}
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Two independent hand-rolled gamma kernels | One `distributions.rs` sharing `ln_gamma` + primitives, two tail wrappers | Kills drift; keeps bit-identity via per-family tail policy |
| Copy-pasted `seed + k` at 10 sites | `seed_for_thread(seed, k)` | One home for the determinism contract |
| Inline SVD sign flip mirrored in pace_fpca | Shared sign-decision core | Removes silent-sign-divergence risk |

**Deprecated/outdated:** none — no external API affected (Phase 50 owns public-API unification).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A single `chi2_sf(x, df: f64)` entry is bit-identical to both `chi_square_sf(x, k: usize)` and `chi_square_sf_df(x, df: f64)` at integer df | Target 1 | Low — both derive `a = <df>/2.0`; a golden at integer df catches any surprise. Verify with `assert_eq!` |
| A2 | Migrating `frechet_anova`'s SECOND per-perm loop (`anova.rs:~272`, generic MetricSpace variant) is in scope | Target 2b | Medium — if it has a separate golden need, plan must add one; I read only the primary loop in full |
| A3 | The Phase-48 goldens are the sufficient equivalence gate for frechet_anova migration (no new golden needed) | Target 2b | Low — they pin exact f64 bits for both branches; adding a Phase-49 golden is cheap insurance |

## Open Questions

1. **Plan A vs Plan B for permutation helper coverage.**
   - What we know: only `frechet_anova` migrates cleanly under a per-perm-reseed helper; four sites use
     advancing RNG, one uses a fixed LCG.
   - What's unclear: whether the plan wants the broader Plan B dual-mode helper (more call-site
     reduction, more goldens) or the minimal Plan A.
   - Recommendation: **Plan A** — behavior-preservation outranks call-site count (CONTEXT.md). Revisit
     Plan B only if a later phase parallelizes the advancing-RNG sites (a behavior-changing task).

2. **Home for the SVD sign-decision core.**
   - What we know: `pace_fpca` needs the single-matrix variant; canonical lives in `regression.rs`.
   - Recommendation: keep the shared core in `regression.rs` (`pub(crate)`), not `distributions.rs`
     (which is numerical-tails only). `pace_fpca` imports from `crate::regression`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | build/test | ✓ | 1.97.0 (MSRV 1.81; `linalg` needs 1.84) | — |
| rand / rayon / faer | existing features | ✓ | in Cargo.lock | — |
| `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` | avoid /tmp exhaustion at commit (per MEMORY) | operator-provided | — | `git commit --no-verify` |

No missing dependencies — pure in-crate refactor.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + integration tests in `fdars-core/tests/` |
| Config file | none (Cargo test harness) |
| Quick run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --test equivalence_phase49` |
| Full suite command | `TMPDIR=… cargo test -p fdars-core --features linalg,parallel` AND `… --no-default-features --features linalg` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CONS-01 | χ² SF/CDF/quantile + `reg_gamma_p` bit-identical (incl. far tail) | integration golden | `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase49 gamma` | ❌ Wave 0 |
| CONS-01 | SVD sign-fix unchanged (FPCA + pace_fpca) | integration golden | `… --test equivalence_phase49 svd_sign` | ❌ Wave 0 |
| CONS-02 | frechet_anova p-values bit-identical after helper migration | integration golden | `… --test equivalence_phase48 golden_frechet_anova` (existing) + new phase49 | ✅ (phase48) / ❌ new |
| CONS-02 | `seed_for_thread` stream matches `seed_from_u64(seed+k)` | integration golden | `… --test equivalence_phase49 rng_stream` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase49` + `clippy --all-targets --features linalg,parallel -- -D warnings`.
- **Per wave merge:** full suite under BOTH feature configs.
- **Phase gate:** full suite green (both configs) + `cargo fmt` clean before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `fdars-core/tests/equivalence_phase49.rs` — captures pre-refactor goldens for all 4 targets
      (gamma SF/CDF/quantile incl. far-tail, SVD signs, frechet_anova, RNG stream).
- [ ] Capture goldens BEFORE any `src/` edit (run current code, hard-code the exact f64 bits as
      `const`, `#![allow(clippy::excessive_precision)]` like phase48).

## Security Domain

`security_enforcement` is not the concern of an internal numerical refactor with no I/O, no untrusted
input parsing, and no new dependency. No ASVS category applies (no auth, session, access control, or
crypto surface introduced or modified). Input-validation (V5) is unchanged — the existing
`FdarError::Invalid*` entry checks are preserved verbatim by the code-motion. No action.

## Sources

### Primary (HIGH confidence — read source-of-truth this session)
- `fdars-core/src/inference/dist.rs` (full) — SF-oriented gamma/beta kernel, `chi_square_sf`,
  `chi_square_sf_df`, `f_sf`, `ln_gamma`.
- `fdars-core/src/spm/chi_squared.rs` (full) — CDF-oriented kernel, `chi2_cdf`, `chi2_quantile`,
  `regularized_gamma_p`.
- `fdars-core/src/inference/permutation.rs` (full) — `t_perm_test`/`f_perm_test` advancing-RNG model.
- `fdars-core/src/frechet/anova.rs:130-207,272` — per-perm reseed + `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD`.
- `fdars-core/src/explain/importance.rs:100-243` — advancing-RNG importance loops.
- `fdars-core/src/explain_generic/importance.rs:1-90` — per-component reseed generic path.
- `fdars-core/src/function_on_scalar.rs:805-865` — fixed-`42` LCG fanova.
- `fdars-core/src/famm.rs:830-889` — advancing-RNG multi-stat permutation_test.
- `fdars-core/src/regression.rs:175-201` — canonical `fix_svd_signs`.
- `fdars-core/src/pace_fpca.rs:205-237` — single-matrix eigenfunction sign mirror.
- `fdars-core/tests/equivalence_phase48.rs` (full) — golden pattern + frechet_anova locks.
- Standalone bit-for-bit gamma-kernel comparison (rustc -O, this session) — the `ln_gamma` bit-identity
  and `reg_gamma_p`/`chi_square_sf` divergence measurements.

### Secondary (MEDIUM)
- `.planning/phases/46-whole-crate-profiling-measurement/PROF-02-dedup-inventory.md` — ranked inventory
  (corrected here re: which sites are parallel vs sequential).

### Tertiary (LOW)
- none.

## Metadata

**Confidence breakdown:**
- χ²/gamma feasibility (share primitives, not one kernel): **HIGH** — measured bit-for-bit this session.
- Permutation seeding inventory: **HIGH** — every site read in full; PROF-02 par/seq labels corrected.
- SVD sign-fix (single vs dual matrix): **HIGH** — both sites read.
- Seeded RNG: **HIGH** — offset formula verified at the `.wrapping_add` sites.

**Research date:** 2026-08-31
**Valid until:** stable (in-crate code; ~30 days) — re-verify only if `dist.rs`/`chi_squared.rs`/the
permutation sites change before planning.
