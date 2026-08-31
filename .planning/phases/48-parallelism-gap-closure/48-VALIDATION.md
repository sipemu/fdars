---
phase: 48
slug: parallelism-gap-closure
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-31
---

# Phase 48 — Validation Strategy

> Behavior-preserving parallelism phase. Each parallelized loop must produce **bit-identical** output
> to the sequential path (feature `parallel` ON vs OFF), proven by a committed golden test run under
> both feature configs. A payback-threshold outer-if guards small inputs. Per-iteration RNG seeding
> (`seed + k`) is the determinism linchpin. Suite green (both feature configs) at every commit.

**Scope (bit-identical-safe only, per 48-RESEARCH.md):** `frechet_anova` (already per-perm seeded)
and `co_cluster` n_init (already per-init seeded). **DEFERRED** (would change numeric output — out of
scope for a behavior-preserving milestone): `t_perm_test`/`f_perm_test` (single shared advancing RNG —
parallelizing needs a per-perm reseed that changes the returned p-values) and `explain/importance`
(low fan-out; Phase 49 CONS-02 folds it into the already-parallel generic path).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `#[test]` + criterion 0.5 (`harness = false`) |
| **Quick run (parallel ON)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Parallel-OFF run** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --no-default-features --features linalg` |
| **Full gate** | `cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test -p fdars-core --features linalg,parallel` |

---

## Sampling Rate

- **After every task commit:** existing suite green (parallel ON).
- **After every wave:** full suite ON + **parallel-OFF suite** + clippy `--all-targets`.
- **Before verify:** both feature configs green; ≥1 thread-scaling cell shows speedup at large input.

---

## Per-Task Verification Map

| Req | Behavior | Test Type | Command | Status |
|-----|----------|-----------|---------|--------|
| PERF-03 | `frechet_anova` p-value bit-identical parallel vs sequential AND ON vs OFF | golden equivalence | `cargo test -- golden_frechet_anova_parallel` under both feature configs | ✅ |
| PERF-03 | `co_cluster` labels + log-likelihood bit-identical parallel vs sequential | golden equivalence | `cargo test -- golden_co_cluster_parallel` under both configs | ✅ |
| PERF-03 | `frechet_anova` thread-scaling ≥2× at n_perm=999 (20 vs 1 thread) | criterion | `RAYON_NUM_THREADS=1` vs `=20 cargo bench --bench perf_parallelism -- frechet_anova` → **9.9×** | ✅ |
| PERF-03 | `co_cluster` thread-scaling speedup at n_init≥ threshold | criterion | `RAYON_NUM_THREADS=1` vs `=20 ... -- co_cluster` → **6.4×** | ✅ |
| PERF-03 | Payback threshold guards small inputs (below-threshold uses sequential) | unit | `*_below_threshold` goldens pass; consts documented (frechet 200, co_cluster 3) | ✅ |
| PERF-03 | Existing determinism/seed tests still pass (no output change) | unit | `cargo test -- anova_permutation_is_seed_reproducible co_cluster` | ✅ |
| all | Existing suite green — BOTH feature configs | integration | `cargo test --features linalg,parallel` AND `--no-default-features --features linalg` → 2583 lib tests each | ✅ |

---

## Wave 0 Requirements

- [x] `tests/equivalence_phase48.rs` — golden tests for `frechet_anova` + `co_cluster` capturing the CURRENT (pre-parallel) seeded output; pass under both feature configs after parallelization.
- [x] `benches/perf_parallelism.rs` + `[[bench]] name = "perf_parallelism"` (PERMANENT — Phase 51 BENCH-02) — thread-scaling cells (frechet_anova + co_cluster).
- [x] Free disk before bench builds: `rm -rf target/debug/{incremental,examples}`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Instructions |
|----------|-------------|------------|--------------|
| Thread-scaling "speedup is real" judgement | PERF-03 | Governor `powersave` LOW-CONFIDENCE; a human confirms the 1-vs-N-thread ratio is a genuine speedup, not noise | Compare PERF-PARALLEL-RESULTS.md 1-thread vs 20-thread medians at the large cell |

---

## Validation Sign-Off

- [x] `frechet_anova` + `co_cluster` golden tests pass under BOTH `--features parallel` and parallel-off (bit-identical) — 5/5 equivalence tests each config
- [x] Payback-threshold consts documented; below-threshold path is sequential (`FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200`, `CO_CLUSTER_INIT_PARALLEL_THRESHOLD=3`)
- [x] ≥1 thread-scaling criterion cell shows speedup at large input — frechet_anova 9.9×, co_cluster 6.4×
- [x] Per-iteration `seed + k` determinism preserved (existing seed-reproducibility tests green, values UNCHANGED)
- [x] Deferred targets (t_perm_test/f_perm_test, frechet_anova_space, explain/importance) documented with rationale in PERF-PARALLEL-RESULTS.md
- [x] Full suite green (both feature configs) + clippy `--all-targets` clean; no public signature change (`frechet_anova`/`co_cluster` byte-identical; `frechet_anova_space` `S: Sync` widening explicitly deferred)
- [x] `nyquist_compliant: true` set once all above hold

**Approval:** ✅ validated 2026-08-31 — PERF-03 met (2 safe parallelizations with payback guards + bit-identical output under both configs; unsafe/marginal targets documented+deferred). Governor `powersave` → absolute medians LOW-CONFIDENCE, but the 6–10× thread-scaling direction is unambiguous.
