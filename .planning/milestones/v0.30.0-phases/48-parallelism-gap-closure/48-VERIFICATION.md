---
phase: 48-parallelism-gap-closure
verified: 2026-08-31T00:00:00Z
status: passed
human_sign_off: "2026-08-31 — operator accepted the thread-scaling magnitude (frechet_anova 9.9x, co_cluster 6.4x under powersave governor). The 6-10x is an order of magnitude above plausible governor/scheduler noise; direction unambiguous, criterion p<0.05. Bit-identity, determinism, payback guards, and no-signature-change all machine-verified."
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Compare PERF-PARALLEL-RESULTS.md 1-thread (RAYON_NUM_THREADS=1) vs 20-thread medians at the large cells (frechet_anova n24_m81_nperm999; co_cluster n200_m50_ninit8). Confirm the 1-vs-N-thread ratio is a genuine speedup, not measurement noise."
    expected: "N-thread median materially below 1-thread median at the large cell (recorded 9.9x frechet_anova, 6.4x co_cluster). Direction should hold; absolute ratios may shift under a performance governor."
    why_human: "Thread-scaling medians were captured under an unpinned `powersave` CPU governor (cpupower/sudo unavailable), so absolute numbers are LOW-CONFIDENCE. Whether the measured ~6-10x is a real speedup vs governor/scheduler noise is a runtime performance judgement grep cannot make. Planner explicitly deferred this to human verification (48-VALIDATION.md Manual-Only Verifications)."
---

# Phase 48: Parallelism-Gap Closure Verification Report

**Phase Goal:** A user with the `parallel` feature enabled gets multi-threaded speedups on the newer subsystems that previously ran only sequentially, with no small-input regression and bit-equivalent results versus the sequential path.
**Verified:** 2026-08-31
**Status:** passed (operator signed off the deferred thread-scaling-magnitude judgement 2026-08-31)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | `frechet_anova` permutation p-value is bit-identical parallel-ON vs parallel-OFF for the same seed (PERF-03) | ✓ VERIFIED | Ran `equivalence_phase48` under both `--features linalg,parallel` and `--no-default-features --features linalg` — `golden_frechet_anova_parallel` asserts `p_value_permutation == 1.00000000000000002e-3` and passes in both configs (identical `assert_eq!` literals) |
| 2 | `frechet_anova` parallel output bit-identical to pre-parallel sequential; determinism via per-perm reseed independent of thread count (PERF-03) | ✓ VERIFIED | `src/frechet/anova.rs:180` `StdRng::seed_from_u64(seed.wrapping_add(perm as u64))` inside `count_ge` closure; `iter_maybe_parallel!(0..n_perm).map(count_ge).sum()` (line 190) is an order-independent integer sum → thread-count-independent |
| 3 | Payback-threshold outer-if (`FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD`) routes small n_perm to sequential; both branches same golden (PERF-03) | ✓ VERIFIED | `anova.rs:25` const=200; `anova.rs:189` outer-if `if n_perm >= THRESHOLD { par } else { seq }`; `golden_frechet_anova_below_threshold` passes both configs |
| 4 | `co_cluster` n_init loop parallelized (parallel map → `collect::<Result<Vec>>` → sequential strict-`>` reduce, lowest-index tie-break); bit-identical parallel vs seq and ON vs OFF (PERF-03) | ✓ VERIFIED | `coclustering.rs:977-994`: `iter_maybe_parallel!(0..n_init).map(run_init).collect::<Result<Vec<_>,_>>()?` then `results.into_iter().reduce(|acc,r| if r.log_likelihood > acc.log_likelihood { r } else { acc })`; `golden_co_cluster_parallel` (log_likelihood + row/col labels) passes both configs |
| 5 | Per-init seeding `config.seed.wrapping_add(init*1000)` order-independent; result independent of thread count (PERF-03) | ✓ VERIFIED | `coclustering.rs:945` `let seed = config.seed.wrapping_add(init as u64 * 1000)` inside `run_init` |
| 6 | Error behavior preserved: first init error in index order propagates; `co_cluster` signature byte-identical (PERF-03) | ✓ VERIFIED | `collect::<Result<Vec,_>>()?` returns first Err in iteration order = the error the sequential `?` short-circuited on (documented at coclustering.rs:973-976); `git diff 76708e0d..HEAD` shows no `pub fn co_cluster` signature line changed |
| 7 | Both parallelized public signatures byte-identical; no new crate dependency (PERF-03) | ✓ VERIFIED | Signature diff empty for both `frechet_anova` and `co_cluster`; Cargo.toml diff since phase start adds only `[[bench]] name = "perf_parallelism"` — no dependency line |
| 8 | Deferred targets documented with rationale, not silently dropped; VALIDATION signed off (PERF-03) | ✓ VERIFIED | PERF-PARALLEL-RESULTS.md documents t_perm/f_perm (changes p-values), `frechet_anova_space<S>` (needs `S: Sync` signature widening — confirmed still sequential at anova.rs:270-280), explain/importance (low fan-out, folds into Phase 49); 48-VALIDATION.md `nyquist_compliant: true`, sign-off checklist all `[x]` |
| — | Thread-scaling: parallel path faster on large inputs (criterion evidence) (PERF-03, SC1) | ⚠️ HUMAN | frechet_anova 9.9x, co_cluster 6.4x recorded — but under `powersave` governor (LOW-CONFIDENCE absolute numbers). Direction present + wired; "genuine speedup vs noise" routed to human (see Human Verification) |

**Score:** 8/8 must-have truths verified. The thread-scaling *speedup magnitude* (ROADMAP SC1) is present and measured but its authoritativeness is a human judgement (governor caveat) — routed to human verification, not counted as a gap.

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `fdars-core/src/frechet/anova.rs` | frechet_anova parallel + threshold const | ✓ VERIFIED | const at :25, `iter_maybe_parallel!` par branch at :190, seq branch at :192; wired |
| `fdars-core/src/coclustering.rs` | co_cluster parallel n_init map + seq reduce + threshold const | ✓ VERIFIED | const at :55, par map/collect at :977-983, seq reduce at :988-994 |
| `fdars-core/tests/equivalence_phase48.rs` | 5 goldens (2 frechet, 2 co_cluster, 1 fixture) | ✓ VERIFIED | 5 `#[test]`; all pass under BOTH feature configs (ran both) |
| `fdars-core/benches/perf_parallelism.rs` | permanent thread-scaling bench | ✓ VERIFIED | `bench_frechet_anova` + `bench_co_cluster`, `criterion_group!`, calls real `frechet_anova`/`co_cluster` |
| `fdars-core/Cargo.toml` `[[bench]] perf_parallelism` | bench registered | ✓ VERIFIED | :103-104 `name = "perf_parallelism"`, `harness = false` |
| `PERF-PARALLEL-RESULTS.md` | thread-scaling + thresholds + deferrals | ✓ VERIFIED | 9.9x/6.4x tables, threshold rationale, 3 documented deferrals, governor caveat |
| `48-VALIDATION.md` | signed off | ✓ VERIFIED | `nyquist_compliant: true`, all sign-off boxes `[x]`, approval stamped |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| per-perm reseed | order-independent parallel sum | `iter_maybe_parallel!(0..n_perm).map(count_ge).sum()` | WIRED | anova.rs:180+190; integer sum commutative → bit-identical |
| `iter_maybe_parallel!` OFF | sequential compile | golden holds under both configs | WIRED | Confirmed by running goldens under `--no-default-features --features linalg` |
| per-init seed + seq strict-> reduce | identical best pick + tie-break | `collect::<Result<Vec>>` then `reduce(strict >)` | WIRED | coclustering.rs:945,988 |
| `collect::<Result<Vec>>()` | first-Err-in-order = sequential `?` | error behavior preserved | WIRED | coclustering.rs:980; documented invariant |
| perf_parallelism bench | Phase 51 BENCH-02 regression guard | permanent registration | WIRED | Cargo.toml [[bench]] |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| Goldens bit-identical parallel-ON | `cargo test ... --features linalg,parallel --test equivalence_phase48` | 5 passed; 0 failed | ✓ PASS |
| Goldens bit-identical parallel-OFF | `cargo test ... --no-default-features --features linalg --test equivalence_phase48` | 5 passed; 0 failed | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| PERF-03 | 48-01/02/03 | Close parallelism gaps in newer subsystems with feature-gated rayon, equivalence-tested vs sequential, payback-threshold guard | ✓ SATISFIED (thread-scaling magnitude → human) | frechet_anova + co_cluster parallelized, bit-identical under both configs (SC2), payback guards (SC3), RNG determinism + no new dep (SC4). SC1 speedup direction present; magnitude authoritativeness deferred to human (governor caveat) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| — | — | No debt markers (TBD/FIXME/XXX) in any modified src/test/bench file | ℹ️ Info | Clean |

### Human Verification Required

**1. Thread-scaling speedup is genuine (not measurement noise)**

**Test:** Compare `PERF-PARALLEL-RESULTS.md` 1-thread (`RAYON_NUM_THREADS=1`) vs 20-thread medians at the large cells — `perf_parallelism_frechet_anova/n24_m81_nperm999` and `perf_parallelism_co_cluster/n200_m50_ninit8`. Ideally re-run under a `performance` CPU governor.
**Expected:** N-thread median materially below 1-thread median (recorded 9.9x frechet_anova, 6.4x co_cluster). Direction should hold; absolute ratios may shift under a pinned governor.
**Why human:** Medians were captured under an unpinned `powersave` governor (cpupower/sudo unavailable), making absolute numbers LOW-CONFIDENCE. Whether the ratio is a real speedup vs scheduler/governor noise is a runtime performance judgement grep cannot make. The planner explicitly routed this to human verification in 48-VALIDATION.md.

### Gaps Summary

No gaps. All eight plan-level must-have truths are verified in the codebase: both target loops (`frechet_anova`, `co_cluster`) are parallelized via `iter_maybe_parallel!` with per-iteration seeding, bit-identical output was confirmed by running the 5 equivalence goldens under BOTH the parallel-ON and parallel-OFF feature configs, payback-threshold outer-if guards are present with below-threshold goldens, public signatures are byte-identical, no crate dependency was added, and the three unsafe/marginal targets are documented+deferred rather than dropped.

The only open item is ROADMAP Success Criterion 1's thread-scaling *speedup magnitude*: the code is present, wired, and benchmarked (9.9x / 6.4x), but the measurement ran under a `powersave` governor, so confirming the speedup is genuine (not noise) is a human judgement the planner deliberately deferred. This makes the phase `human_needed` rather than `passed` — the implementation is complete; only the performance-magnitude confirmation awaits a human.

---

_Verified: 2026-08-31_
_Verifier: Claude (gsd-verifier)_
