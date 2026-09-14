---
phase: 107-spm-complexity-cluster
verified: 2026-09-14T12:00:00Z
status: passed
score: 4/4 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/107-spm-complexity-cluster/107-01-PLAN.md
  - .planning/phases/107-spm-complexity-cluster/107-01-SUMMARY.md
  - .planning/phases/107-spm-complexity-cluster/107-02-PLAN.md
  - .planning/phases/107-spm-complexity-cluster/107-02-SUMMARY.md
  - .planning/phases/107-spm-complexity-cluster/107-03-PLAN.md
  - .planning/phases/107-spm-complexity-cluster/107-03-SUMMARY.md
  - .planning/phases/107-spm-complexity-cluster/107-04-PLAN.md
  - .planning/phases/107-spm-complexity-cluster/107-04-SUMMARY.md
  - fdars-core/src/spm/amewma.rs
  - fdars-core/src/spm/contrib.rs
  - fdars-core/src/spm/ewma.rs
  - fdars-core/src/spm/frcc.rs
  - fdars-core/src/spm/iterative.rs
  - fdars-core/src/spm/mewma.rs
  - fdars-core/src/spm/mfpca.rs
  - fdars-core/src/spm/ncomp.rs
  - fdars-core/src/spm/partial.rs
  - fdars-core/src/spm/phase.rs
  - fdars-core/src/spm/profile.rs
  - fdars-core/src/spm/stats.rs
covered_digest: "v1:sha256:0d55331c20d3f3a995acbc18cd522f34166c7e5a828b5f5e8058c139eee6e951"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 107: SPM Complexity Cluster Verification Report

**Phase Goal:** Decompose the concentrated complexity cluster inside `fdars-core/src/spm/` into smaller private helper functions so each falls under pmat's cognitive/cyclomatic thresholds — behavior and numeric output unchanged.
**Verified:** 2026-09-14T12:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | None of the 13 target SPM functions appear as ERROR-level complexity violations under pmat 3.18.0 (cog >25 or cyc >30) | VERIFIED | pmat run on all 10 affected files: `spm_amewma_monitor` cog 11, `spm_phase1_iterative` cog 12, `mfpca` cog 0, `select_ncomp` cog 14, `t2_contributions_mfpca` cog 13, `spe_contributions` cog 6, `spm_ewma_monitor` cog 1, `spm_mewma_monitor` cog 7, `spe_multivariate` cog 6, `spm_monitor_from_fields` cog 0, `conditional_expectation` cog 3, `frcc_phase1` cog 3, `profile_phase1` cog 3. No target function exceeds threshold. `pmat analyze complexity --path fdars-core/src/spm/` reports 0 ERRORs, 8 Warnings only (pre-existing warning-level functions explicitly deferred to Phase 109). |
| 2 | No newly-extracted private helper function itself exceeds pmat thresholds (cog >25 or cyc >30) | VERIFIED | All helpers confirmed below threshold by direct pmat output: highest helper scores are `amewma_validate_lambda_range` cog 15, `monitor_fields_validate` cog 15, `frcc_validate` cog 16, `cond_exp_solve_with_retry` cog 13, `mewma_validate` cog 13, `profile_validate` cog 13 — all safely below the cog 25 error threshold. |
| 3 | Full test suite (2942 lib + 156 SPM module tests) passes with behavior and numeric output identical | VERIFIED | `cargo test -p fdars-core --features linalg,parallel --lib -- spm::` → 156 passed, 0 failed. `cargo test -p fdars-core --features linalg,parallel --lib` → 2942 passed, 0 failed. Both confirmed by direct foreground run. |
| 4 | No public API change: all 13 target functions retain original signatures and visibility; all newly-extracted helpers are private `fn` | VERIFIED | grep over all 12 modified spm files: only pre-existing `pub fn` declarations present (the 13 original targets plus pre-existing public functions). Zero new `pub fn` or `pub(crate) fn` introduced. `git diff fca1db5a..556e0fa6 --name-only` confirms only `fdars-core/src/spm/` and `.planning/` files modified — no Cargo.toml changes, no new crate dependency. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Deferred Items

`solve_spd` (partial.rs, cog 21) and `rolling_betas` (profile.rs, cog 21) are WARNING-level functions intentionally left untouched — explicitly deferred to Phase 109 (QUAL-03 tail pass) per PLAN.md scope note. These are not failures for Phase 107.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/spm/amewma.rs` | Private helpers added; spm_amewma_monitor cog 52→11 | VERIFIED | 6 functions total: spm_amewma_monitor (cog 11), amewma_validate (cog 3), amewma_validate_lambda_range (cog 15), amewma_validate_params (cog 10), amewma_project_standardize (cog 12), amewma_ewma_step (cog 1) |
| `fdars-core/src/spm/iterative.rs` | Private helpers added; spm_phase1_iterative cog 42→12 | VERIFIED | 5 functions total: spm_phase1_iterative (cog 12), iterative_validate (cog 11), iterative_identify_flagged (cog 9), iterative_check_removal_budget (cog 3), iterative_update_state (cog 0) |
| `fdars-core/src/spm/mfpca.rs` | Private helpers added; mfpca cog 41→0 | VERIFIED | 5 functions total: mfpca (cog 0), mfpca_validate (cog 12), mfpca_compute_means_scales (cog 5), mfpca_build_stacked (cog 12), mfpca_extract_components (cog 12). WR-01 fix (rename _svd→svd) applied in commit 556e0fa6. |
| `fdars-core/src/spm/ncomp.rs` | Private helpers added; select_ncomp cog 32→14 | VERIFIED | 6 functions total: select_ncomp (cog 14), ncomp_validate_eigenvalues (cog 11), ncomp_validate_threshold (cog 4), plus 3 pre-existing arm helpers all below threshold |
| `fdars-core/src/spm/contrib.rs` | Private helpers added; t2_contributions_mfpca cog 30→13, spe_contributions cog 27→6 | VERIFIED | 10 functions total; both targets well below threshold. WR-03 fix (t2_mfpca_validate returns only p) applied in commit 556e0fa6. |
| `fdars-core/src/spm/ewma.rs` | Private helpers added; spm_ewma_monitor cog 27→1 | VERIFIED | 6 functions total: spm_ewma_monitor (cog 1), ewma_validate (cog 6), ewma_truncate_scores (cog 6), ewma_compute_t2_exact (cog 3), ewma_compute_spe (cog 9) |
| `fdars-core/src/spm/mewma.rs` | Private helpers added; spm_mewma_monitor cog 25→7 | VERIFIED | 3 functions total: spm_mewma_monitor (cog 7), mewma_validate (cog 13), mewma_compute_exact_t2 (cog 3) |
| `fdars-core/src/spm/stats.rs` | Private helpers added; spe_multivariate cog 27→6 | VERIFIED | 7 functions total: spe_multivariate (cog 6), spe_mv_validate (cog 4), spe_mv_var_check (cog 8), spe_mv_compute_var (cog 3) |
| `fdars-core/src/spm/phase.rs` | Private helpers added; spm_monitor_from_fields cog 27→0 | VERIFIED | 12 functions total: spm_monitor_from_fields (cog 0), monitor_fields_validate (cog 15), monitor_fields_project (cog 6), monitor_fields_recon (cog 6). Pre-existing mf_spm_phase1 at cog 23 is a warning-level function not in the 13 targets. |
| `fdars-core/src/spm/partial.rs` | Private helpers added; conditional_expectation cog 29→3 | VERIFIED | 9 functions total: conditional_expectation (cog 3), cond_exp_sigma2 (cog 3), cond_exp_build_m_matrix (cog 10), cond_exp_solve_with_retry (cog 13). Pre-existing solve_spd (cog 21) and spm_monitor_partial (cog 20) are warning-level, out of scope. |
| `fdars-core/src/spm/frcc.rs` | Private helpers added; frcc_phase1 cog 29→3 | VERIFIED | 6 functions total: frcc_phase1 (cog 3), frcc_validate (cog 16), frcc_validate_split (cog 6), frcc_compute_r_squared (cog 4) |
| `fdars-core/src/spm/profile.rs` | Private helpers added; profile_phase1 cog 26→3 | VERIFIED | 7 functions total: profile_phase1 (cog 3), profile_validate (cog 13), profile_validate_windows (cog 6), profile_lag1_autocorr (cog 4), profile_effective_n (cog 1). Pre-existing rolling_betas (cog 21) is warning-level, out of scope. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `spm_amewma_monitor` | `amewma_ewma_step` | call inside outer for-i loop | WIRED | Outer recurrence loop intact; ewma_step receives lambda_cur, not lambda_prev; AMEWMA sequential recurrence preserved |
| `mfpca` | `mfpca_build_stacked` | call with `&mut scales` | WIRED | scales passed as &mut [f64]; scales[p]=1.0 mutation propagates correctly to MfpcaResult.scales |
| `conditional_expectation` | `cond_exp_solve_with_retry` | call with diag_max from original matrix | WIRED | diag_max captured from pre-regularization M matrix; Tikhonov retry sequence byte-identical |
| `frcc_phase1` | `frcc_validate` | call before split_indices | WIRED | fosr_lambda guard moved into frcc_validate pre-split; all guards consolidated before split call |
| `profile_phase1` | `profile_validate_windows` | called from profile_validate | WIRED | Split-validate applied; profile_validate (cog 13) delegates window guards to profile_validate_windows (cog 6) |

### pmat Complexity Before/After Summary

| Function | File | Before cog | After cog | Before cyc | After cyc | Status |
|----------|------|-----------|----------|-----------|----------|--------|
| `spm_amewma_monitor` | amewma.rs | 52 | 11 | 25 | 7 | CLEARED |
| `spm_phase1_iterative` | iterative.rs | 42 | 12 | 16 | 5 | CLEARED |
| `mfpca` | mfpca.rs | 41 | 0 | 21 | 1 | CLEARED |
| `select_ncomp` | ncomp.rs | 32 | 14 | 10 | 5 | CLEARED |
| `t2_contributions_mfpca` | contrib.rs | 30 | 13 | 12 | 5 | CLEARED |
| `spe_contributions` | contrib.rs | 27 | 6 | 11 | 4 | CLEARED |
| `spm_ewma_monitor` | ewma.rs | 27 | 1 | (n/a) | 2 | CLEARED |
| `spm_mewma_monitor` | mewma.rs | 25 | 7 | (n/a) | 5 | CLEARED |
| `spe_multivariate` | stats.rs | 27 | 6 | (n/a) | 4 | CLEARED |
| `spm_monitor_from_fields` | phase.rs | 27 | 0 | (n/a) | 1 | CLEARED |
| `conditional_expectation` | partial.rs | 29 | 3 | 16 | 3 | CLEARED |
| `frcc_phase1` | frcc.rs | 29 | 3 | 13 | 2 | CLEARED |
| `profile_phase1` | profile.rs | 26 | 3 | 12 | 2 | CLEARED |

All 13 target functions cleared. No newly-extracted helper exceeds cog 25 or cyc 30.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SPM module tests (156 tests) | `cargo test -p fdars-core --features linalg,parallel --lib -- spm::` | 156 passed, 0 failed | PASS |
| Full lib suite (2942 tests) | `cargo test -p fdars-core --features linalg,parallel --lib` | 2942 passed, 0 failed | PASS |
| clippy all-targets | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | Finished with 0 warnings | PASS |
| fmt check | `cargo fmt -p fdars-core --check` | Clean (no output) | PASS |
| pmat spm/ errors | `pmat analyze complexity --path fdars-core/src/spm/` | 0 ERRORs, 8 Warnings (pre-existing warning-level fns, deferred) | PASS |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| QUAL-02 | 107-01, 107-02, 107-03, 107-04 | SPM complexity cluster refactored under pmat thresholds, behavior/numeric-identical | SATISFIED | All 13 target functions cleared by pmat; 2942 lib tests green; no public API change. REQUIREMENTS.md traceability row for QUAL-02 maps to Phase 107 and marks status Complete. |

QUAL-02 as defined in REQUIREMENTS.md: "The SPM complexity cluster — `mfpca`, `spm_phase1_iterative`, `spm_mewma_monitor`, `select_ncomp`, `conditional_expectation` (partial), `spm_monitor_from_fields` (phase), `profile_phase1`, `spe_contributions` (stats) — is refactored under pmat thresholds, behavior/numeric-identical." All named functions cleared. The 5 additional functions targeted by the phase plans (`spm_amewma_monitor`, `t2_contributions_mfpca`, `spe_multivariate`, `spm_ewma_monitor`, `frcc_phase1`) were additional scope beyond the QUAL-02 text — also cleared.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | No TBD/FIXME/XXX markers found in any modified spm file | — | — |

### Code Review Cross-Check (107-REVIEW.md)

The review identified 0 critical findings and 3 warnings. Cross-check against current codebase:

- **WR-01** (_svd param named with underscore but actively used — mfpca.rs): FIXED in commit 556e0fa6. Parameter renamed to `svd`.
- **WR-02** (t2_pc_significance ncomp==0 guard — contrib.rs): Pre-existing issue, not a refactor regression; behavior-changing fix deferred to a future milestone as documented in 107-REVIEW.md. Acceptable deferral.
- **WR-03** (t2_contributions_mfpca discards validated ncomp — contrib.rs): FIXED in commit 556e0fa6. t2_mfpca_validate now returns only `p`.

All refactor-introduced review findings (WR-01, WR-03) resolved before verification. One pre-existing issue (WR-02) deferred — acceptable for a behavior-preserving refactor phase.

### Scope Discipline Verification

- Only `fdars-core/src/spm/` source files and `.planning/` documentation files modified across all commits (fca1db5a..556e0fa6).
- `fdars-core/Cargo.toml` unchanged — no new crate dependency introduced.
- All 13 target functions retain their original `pub fn` signatures exactly.
- Zero new `pub fn` or `pub(crate) fn` declarations introduced by the phase.

---

## Verdict

**PASSED.** All 13 SPM error-level complexity functions cleared under pmat 3.18.0 thresholds (cognitive ≤25, cyclomatic ≤30). All new private helper functions are themselves below threshold. The full 2942-test suite is green with numerically identical output. No public API surface changed. No new crate dependency. Cargo fmt, clippy, and pmat report clean. QUAL-02 is SATISFIED.

The two deferred out-of-scope warning-level functions (`solve_spd` cog 21, `rolling_betas` cog 21) and the pre-existing WR-02 defensive-guard gap are acknowledged and intentionally deferred to Phase 109 and a future milestone respectively — not blockers for Phase 107's stated goal.

---

_Verified: 2026-09-14T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
