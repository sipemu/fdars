---
phase: 49-code-consolidation-dedup
plan: 04
subsystem: permutation-test
tags: [refactor, dedup, consolidation, permutation, frechet-anova, golden-equivalence, CONS-02, plan-A]
requires:
  - phase: 48
    provides: frechet_anova per-perm reseed + FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200 + permanent Phase-48 goldens
  - phase: 49
    plan: 03
    provides: pub(crate) helpers::seed_for_thread(seed, k) determinism contract + equivalence_phase49.rs harness
provides:
  - src/permutation_test.rs — the single authoritative pub(crate) permutation_pvalue scaffold (per-perm reseed, threshold-gated parallel, (1+n_ge)/(1+n_perm))
  - frechet_anova PRIMARY per-perm loop migrated onto permutation_pvalue (bit-identical, both configs)
  - tests/equivalence_phase49.rs — new phase49 frechet golden (parallel 999 + sequential 50) as in-phase insurance
  - CONS-02 permutation consolidation (Plan A): 1 site migrated, 5 documented-and-excluded, 1 second loop document-and-skipped
affects: []
actuals:
  tokens: 44000
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - "draw-application contract: helper shuffles a position vector 0..n once per perm; caller GATHERS its own per-position data (group_labels[perm_idx[i]]) — one Fisher–Yates on a length-n slice under the same per-perm seed ⇒ identical position-permutation ⇒ bit-identical perm_labels"
    - "behavior-preservation-outranks-callsite-count (Plan A): migrate ONLY the site whose loop matches the contract; document-and-exclude advancing-RNG / LCG / multi-statistic sites"
    - "golden-equivalence capture-then-migrate (assert_eq! bit-identical, NOT tolerance) under BOTH feature configs; permanent Phase-48 goldens are the backstop"
key-files:
  created:
    - fdars-core/src/permutation_test.rs
  modified:
    - fdars-core/src/lib.rs
    - fdars-core/src/frechet/anova.rs
    - fdars-core/tests/equivalence_phase49.rs
    - fdars-core/src/inference/permutation.rs
    - fdars-core/src/explain/importance.rs
    - fdars-core/src/famm.rs
    - fdars-core/src/function_on_scalar.rs
key-decisions:
  - "Plan A (RESEARCH, operator-locked): ONE authoritative pub(crate) permutation_pvalue; migrate ONLY frechet_anova's primary loop (the single site whose per-perm-reseeded-StdRng + threshold-gated-parallel + (1+n_ge)/(1+n_perm) loop already matches the contract). Behavior-preservation outranks call-site count."
  - "Draw-application contract makes bit-identity PROVABLE, not incidental: the helper shuffles a 0..n position vector once with the given rng; the frechet closure builds perm_labels by GATHERING group_labels[perm_idx[i]] rather than re-shuffling. Same per-perm seed + one Fisher–Yates on a length-n slice ⇒ identical position-permutation ⇒ bit-identical perm_labels. Phase-48 goldens are the backstop."
  - "Second frechet loop (frechet_anova_space, generic MetricSpace, ~line 272) is SEQUENTIAL with NO dedicated golden → DOCUMENT-AND-SKIP. Migrating it would introduce parallelism with no equivalence backstop, violating the behavior-preservation mandate."
  - "Degenerate-permutation skip expressed by stat returning f64::NEG_INFINITY (never >= observed), reproducing frechet's old conservative count-0 branch exactly."
  - "Module named permutation_test (deliberately distinct from the existing inference::permutation submodule)."
  - "Threshold is a caller-owned param (frechet passes FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200) so the payback point stays with the caller; parallel path preserved."
  - "The 5 incompatible sites (t_perm_test, f_perm_test, explain/importance x2, famm) use a single ADVANCING StdRng (per-perm reseed would change p-values); function_on_scalar::fanova uses a hardcoded-42 LCG with no seed param (outside the StdRng contract); famm is additionally multi-statistic (per-covariate) — the -> f64 scaffold cannot express it. Each carries a one-line rationale comment."
  - "Phase-48 hand-off 'fold explain/importance into the already-parallel generic path' is behavior-CHANGING (generic path reseeds per-component seed+k; importance advances one RNG) → recorded as a DEFERRAL at both importance sites, NOT implemented."
requirements-completed: [CONS-02 (permutation-scaffold target)]
coverage:
  - id: P1
    description: "permutation_pvalue is the one authoritative permutation scaffold; frechet_anova primary loop migrated bit-identically"
    requirement: CONS-02
    verification:
      - kind: integration
        ref: "equivalence_phase48::golden_frechet_anova_parallel + _below_threshold AND equivalence_phase49::frechet_anova_permutation_scaffold_bit_identical pass under --features linalg,parallel AND --no-default-features --features linalg"
        status: pass
    human_judgment: false
  - id: P2
    description: "FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200 parallel path preserved; parallel .sum() bit-identical to sequential and to parallel-OFF"
    requirement: CONS-02
    verification:
      - kind: unit
        ref: "permutation_test::tests::parallel_and_sequential_branches_agree; golden n_perm=999 (parallel) + n_perm=50 (sequential) both bit-identical"
        status: pass
    human_judgment: false
  - id: P3
    description: "5 advancing-RNG/LCG sites documented-and-excluded; second frechet loop document-and-skipped; explain/importance fold-in recorded as behavior-changing deferral"
    requirement: CONS-02
    verification:
      - kind: integration
        ref: "comment-only changes; full suite 2586 lib + integration green under BOTH configs; clippy --all-targets clean"
        status: pass
    human_judgment: false
  - id: P4
    description: "Wave-4 gate: full suite green both configs + clippy --all-targets clean + fmt clean; no public signature change; no new dependency"
    requirement: CONS-02
    verification:
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean; cargo fmt --check => clean"
        status: pass
      - kind: integration
        ref: "cargo test both feature configs => all green (2586 lib each)"
        status: pass
    human_judgment: false
status: complete
---

# Phase 49 Plan 04: Permutation-Test Consolidation (CONS-02, Plan A) Summary

Consolidated the permutation-test scaffold (CONS-02) via RESEARCH **Plan A** (lowest risk,
operator-locked): shipped ONE authoritative `pub(crate) fn permutation_pvalue` (per-perm reseed,
threshold-gated parallel, `(1+n_ge)/(1+n_perm)`) and migrated **only** `frechet_anova`'s primary
per-perm loop — the single site whose loop already matches the contract exactly. The five
advancing-RNG / LCG / multi-statistic sites and `frechet_anova`'s second (generic-MetricSpace) loop
are documented-and-excluded, because **behavior-preservation outranks call-site count**: a migration
whose bit-identity we cannot prove with a golden would violate the phase's mandate.

## What shipped

- **`src/permutation_test.rs`** (new, `pub(crate)`): `permutation_pvalue<F>(observed, n, n_perm,
  seed, threshold, stat) where F: Fn(&[usize]) -> f64 + Sync`. Per-perm reseed via
  `helpers::seed_for_thread` (from plan 49-03), shuffles a `0..n` position vector once, hands the
  closure `&perm_idx`, counts `stat(&perm_idx) >= observed`, dispatches parallel via
  `iter_maybe_parallel!` when `n_perm >= threshold` (caller-owned) else sequential, returns
  `(n_ge+1)/(n_perm+1)`. Degenerate perms are expressed by `stat` returning `f64::NEG_INFINITY`.
  Three unit tests (parallel==sequential, NEG_INFINITY skips, exact rational).
- **`frechet/anova.rs`**: the PRIMARY loop now calls `permutation_pvalue`; the closure builds
  `perm_labels` by **gathering** `group_labels[perm_idx[i]]` (the draw-application contract) and
  returns `NEG_INFINITY` on the `compute_tn_generic` error path. `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200`
  is passed to the helper (parallel path preserved). The SECOND loop (`frechet_anova_space`) is
  document-and-skipped with a rationale comment. Removed now-unused `iter_maybe_parallel!` /
  `ParallelIterator` imports.
- **`tests/equivalence_phase49.rs`**: a new `frechet` golden
  (`frechet_anova_permutation_scaffold_bit_identical`) mirroring the Phase-48
  `two_group_densities` fixture, asserting the SAME statistic + permutation p-value bits for both the
  n_perm=999 (parallel) and n_perm=50 (sequential) cases — captured against CURRENT frechet_anova
  before the Task-2 migration (cheap in-phase insurance per RESEARCH A3).
- **`lib.rs`**: `pub(crate) mod permutation_test;` (name deliberately distinct from the existing
  `inference/permutation.rs` submodule).
- **Rationale comments** (no logic change) at the 5 un-migrated sites +
  `frechet_anova_space`'s second loop.

## Sites: migrated vs documented-excluded

| Site | Disposition | Reason |
|------|-------------|--------|
| `frechet/anova.rs` PRIMARY loop | **MIGRATED** | Per-perm reseeded StdRng, threshold-gated parallel, `(1+n_ge)/(1+n_perm)` — matches contract; Phase-48 goldens back it |
| `frechet/anova.rs` second loop (`frechet_anova_space`) | document-and-skip | Sequential generic-MetricSpace variant, NO dedicated golden → migrating adds unbacked parallelism |
| `inference/permutation.rs` `t_perm_test` | documented-excluded | Single ADVANCING StdRng → per-perm reseed changes p-value |
| `inference/permutation.rs` `f_perm_test` | documented-excluded | Single ADVANCING StdRng → per-perm reseed changes p-value |
| `explain/importance.rs` `fpc_permutation_importance` | documented-excluded | Advancing StdRng across ALL components; + fold-in deferral |
| `explain/importance.rs` `..._logistic` | documented-excluded | Advancing StdRng across ALL components; + fold-in deferral |
| `famm.rs` `permutation_test` | documented-excluded | Advancing StdRng AND multi-statistic (per-covariate) — `-> f64` cannot express it |
| `function_on_scalar.rs` `fanova` | documented-excluded | Hardcoded-42 LCG, no `seed` param — outside the StdRng contract |

The Phase-48 hand-off "fold explain/importance into the already-parallel generic path" is recorded
at both importance sites as a **behavior-CHANGING deferral** (generic path reseeds per-component
`seed+k`; importance advances one RNG) — NOT implemented here.

## Commit count

3 atomic commits (one per task):
- `0386e0fd` feat(49-04): add pub(crate) permutation_pvalue scaffold + frechet golden (CONS-02)
- `70787605` refactor(49-04): migrate frechet_anova primary loop onto permutation_pvalue (CONS-02)
- `8ed80439` docs(49-04): document the 5 un-migrated permutation sites + fold-in deferral (CONS-02)

## Golden results (both feature configs)

| Golden | `--features linalg,parallel` | `--no-default-features --features linalg` |
|--------|------------------------------|-------------------------------------------|
| `equivalence_phase48::golden_frechet_anova_parallel` (n_perm=999) | pass — stat `1.17320834419366224e3`, p `1.00000000000000002e-3` | pass (bit-identical) |
| `equivalence_phase48::golden_frechet_anova_below_threshold` (n_perm=50) | pass — p `1.96078431372549017e-2` | pass (bit-identical) |
| `equivalence_phase49::frechet_anova_permutation_scaffold_bit_identical` (999 + 50) | pass (bit-identical) | pass (bit-identical) |
| Full suite (lib) | 2586 pass, 0 fail | 2586 pass, 0 fail |
| `clippy --all-targets -D warnings` | clean | — |
| `cargo fmt --check` | clean | — |

The `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200` parallel path is preserved: the n_perm=999 golden
exercises it, the n_perm=50 golden exercises the below-threshold sequential path, and both are
bit-identical to the pre-migration values.

## Deviations from Plan

None material — the plan's Plan-A design (migrate one site, document the rest) was followed exactly.

Minor enabling adjustment, within the plan's discretion on mechanics:
- **[Rule 3 - Blocking] Task-1 dead_code.** `permutation_pvalue` had no consumer at the end of
  Task 1, so `clippy -D warnings` failed on `dead_code`. Added a temporary `#[allow(dead_code)]`
  (with a NOTE pointing to Task 2) and removed it in Task 2 once frechet_anova wired it up. No effect
  on shipped code.

## Phase-level completion

Phase 49 (Code Consolidation / Dedup) is fully covered: **CONS-01** (χ²/gamma — plan 01; SVD signs —
plan 02) + **CONS-02** (seeded-RNG `seed_for_thread` — plan 03; permutation scaffold — plan 04).

## Known Stubs

None. `permutation_pvalue` is fully implemented and consumed by `frechet_anova`; the un-migrated
sites retain their real inline implementations (comments only).

## Threat Flags

None. Internal permutation-scaffold refactor — no new network endpoint, auth path, file access, or
schema at a trust boundary. T-49-06 (frechet p-value tampering) and T-49-07 (accidental migration of
advancing-RNG/LCG sites) were mitigated exactly as the register prescribed: permanent Phase-48
goldens + new phase49 frechet golden `assert_eq!` bit-identical under both configs; only frechet_anova
migrated, the 5 incompatible sites documented-and-excluded (not touched).

## Self-Check: PASSED

- `fdars-core/src/permutation_test.rs` — FOUND
- `fdars-core/tests/equivalence_phase49.rs` (frechet golden) — FOUND
- Commit `0386e0fd` — FOUND
- Commit `70787605` — FOUND
- Commit `8ed80439` — FOUND
