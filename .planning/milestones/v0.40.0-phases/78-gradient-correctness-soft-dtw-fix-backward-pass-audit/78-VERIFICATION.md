---
phase: 78-gradient-correctness-soft-dtw-fix-backward-pass-audit
verified: 2026-09-07T00:00:00Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
verified_by: orchestrator-inline (gsd-verifier subagent dropped at wrap-up before writing; re-verified inline per repo's connection-drop pattern)
---

# Phase 78: Gradient Correctness — soft_dtw Fix & Backward-Pass Audit Verification Report

**Phase Goal:** Every hand-written backward/gradient pass in the crate produces a correct (non-zero, boundary-seeded) gradient — starting with the concrete `soft_dtw_backward` endpoint-seed fix and extending to a full audited sweep of the sibling gradient passes.
**Verified:** 2026-09-07
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth (from success criteria) | Status | Evidence |
|---|-------|--------|----------|
| 1 | `soft_dtw_backward` preserves the `E[n][m]=1.0` endpoint seed (non-zero E on non-identical input) | ✓ VERIFIED | Guard `if i == n && j == m { continue; }` at `fdars-core/src/metric/soft_dtw.rs:278`; test `soft_dtw_backward_nonzero_and_matches_oracle` asserts non-zero E |
| 2 | New regression test asserts non-zero gradient cross-checked vs `corrected_oracle_gradient` AND the v0.39.0 `Dual` path within ~1e-6 | ✓ VERIFIED | `soft_dtw_backward_nonzero_and_matches_oracle` passes: shipped grad == oracle == Dual, rel err ≤ 1e-6 |
| 3 | `soft_dtw_barycenter` on non-identical curves converges to a barycenter measurably different from the pointwise mean | ✓ VERIFIED | `test_soft_dtw_barycenter_moves_from_mean` (L2 > 0.1) + `_shifted` (converged, mean ≈ 1.0) pass |
| 4 | Existing `test_soft_dtw_barycenter_*` tests tightened so they cannot pass on an all-zero gradient | ✓ VERIFIED | `_identical` (converged + bounded + close-to-series), `_shifted` (converged + mean + L2>0.1 + bounded), `_vs_tslearn` (converged + amplitude band + L2>0.05) — all assert real behavior |
| 5 | CORR-02: all 9 named modules audited with clean/fixed disposition, traceable | ✓ VERIFIED | `78-AUDIT.md` 9-row disposition table: `metric/soft_dtw` = fixed (CORR-01), other 8 = clean with per-module rationale |
| 6 | Behavior-preserving except the intended soft_dtw correction | ✓ VERIFIED | Only `metric/soft_dtw.rs` gradient/optimizer changed; full suite green elsewhere |
| 7 | Barycenter-divergence exposed by the fix (WR-03) resolved in-scope + backlogged | ✓ VERIFIED | Inverse-curvature step (commit c1179749) converges & stays bounded; SDTW-O1 logged in STATE.md Deferred Items |
| 8 | Whole-crate gates green | ✓ VERIFIED | See Gates below |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/metric/soft_dtw.rs` | endpoint-skip guard + stable barycenter step | ✓ EXISTS + SUBSTANTIVE | Guard at :278; `soft_dtw_accumulate_gradient_and_weight` + inverse-curvature `update_barycenter` |
| `fdars-core/src/metric/tests.rs` | tightened + new barycenter tests | ✓ EXISTS + SUBSTANTIVE | `_identical`/`_shifted`/`_moves_from_mean` assert convergence/boundedness/movement |
| `fdars-core/tests/validate_against_r.rs` | tightened integration test | ✓ EXISTS + SUBSTANTIVE | `_vs_tslearn` restored to max_iter=50, asserts converged + amplitude band + L2 movement |
| `78-AUDIT.md` | CORR-02 disposition table | ✓ EXISTS + SUBSTANTIVE | 9 rows, all traceable; follow-on WR-03 finding recorded |

**Artifacts:** 4/4 verified

## Gates

Re-run by the orchestrator on the final tree (post-remediation):

- `cargo fmt --manifest-path fdars-core/Cargo.toml -- --check` — **clean**
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — **clean** (no warnings)
- `cargo test -p fdars-core --features linalg,parallel` — **2861 lib tests + all integration binaries pass, 0 failed**

## Notes

- Requirements CORR-01 and CORR-02 both fully covered (Plan 01, Plan 02).
- Regression-catching proven by revert-test: the barycenter/oracle tests FAIL when the endpoint guard is removed and PASS with it applied.
- The CORR-01 gradient fix exposed a latent `soft_dtw_barycenter` divergence (fixed lr=1/n). Per the locked CONTEXT decision + user direction, a minimal in-scope step-size safeguard (soft-DBA inverse-curvature update) was applied so the public function converges instead of diverging; a proper global optimizer (L-BFGS / multi-restart) is deferred to backlog item SDTW-O1.
- Code review (78-REVIEW.md): 0 critical, 3 warnings — all resolved (status: resolved).
