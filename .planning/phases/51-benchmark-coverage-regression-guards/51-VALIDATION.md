---
phase: 51
slug: benchmark-coverage-regression-guards
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-01
---

# Phase 51 — Validation Strategy

> MEASUREMENT-ONLY phase. Every change is a NEW file (`benches/<module>_benchmarks.rs`, `BENCH-RESULTS.md`)
> plus one `[[bench]]` line per bench in `fdars-core/Cargo.toml`. NO `src/` behavior change, NO new crate
> dependency (criterion 0.5 + dhat 0.3 are pre-existing dev-deps). The regression guards are DOCUMENTED SOFT
> guards (criterion measures; `cargo bench` + baseline-compare catches regressions — they do NOT fail CI) plus
> the UNCHANGED deterministic hard-assert alloc guards (`tests/alloc_audit_dpca.rs`, `tests/alloc_audit_fpca.rs`,
> feature `dhat-heap`). NO wall-time `assert!` tests (governor-sensitive — locked out of scope). The primary
> gate is `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (it LINTS bench code; a plain
> `-p … -D warnings` false-greens per MEMORY ci-clippy-all-targets-gate).

**Scope (BENCH-01 + BENCH-02):**
- **BENCH-01:** 9 new `[[bench]]` files for the 9 unbenchmarked modules — `inference::t_perm_test` (51-01 TRACER),
  `fts::ftsm`, `frechet::frechet_global_reg`, `boosting_regression::boost_fosr`, `coclustering::co_cluster_select`
  (51-02), `fem_smoothing::fem_smooth_gcv`, `density_fda::lqd_fpca`, `fpca_variants::fpca_der`,
  `irreg_fdata::mface_covariance` (51-03). Each covers the module's SIBLING entry — never duplicating a cell
  already in perf_hotpaths (dpca/face_covariance/fem_smooth) or perf_parallelism (frechet_anova/co_cluster).
- **BENCH-02:** `BENCH-RESULTS.md` (51-04) consolidating the Phase 47 (dpca −54% blocks, face_covariance −80.7%)
  + Phase 48 (frechet_anova 9.9×, co_cluster 6.4×) before/after as documented soft guards + the 9 new-module first
  baselines + the governor caveat. Confirms perf_hotpaths/perf_parallelism stay registered PERMANENT and the two
  alloc_audit guards stay intact.

**Deliberately excluded (documented, not dropped):** wall-time `assert!` regression tests (governor-sensitive/flaky);
any behavior-changing `src/` optimization (measurement-only); CI wiring to auto-run benches on every PR (benches
stay `cargo bench`-on-demand).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | criterion 0.5 (benches, `harness = false`) + Rust `#[test]` (existing alloc_audit guards) |
| **Quick run (compile all benches)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --benches --features linalg,parallel` |
| **Primary gate (LINTS bench code)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` |
| **Run one bench once (baseline capture)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench -p fdars-core --features linalg,parallel --bench <name> -- --quick` |
| **Alloc hard-guards (unchanged)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features dhat-heap,linalg --test alloc_audit_dpca --test alloc_audit_fpca` |
| **Disk-safety retry** | on "linking with cc failed" (NOT a code bug): `rm -rf target/debug/{incremental,examples}` then retry (MEMORY target-dir-fills-home-partition) |
| **Commit hygiene** | `cargo fmt` per commit + `git commit --no-verify` (slow hook stalls the executor watchdog; fmt run manually to avoid the drift MEMORY noverify-commits-leave-fmt-drift warns of) |

---

## Sampling Rate

- **After every task commit:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (primary gate) + `cargo bench --bench <name> -- --quick` (the new cell runs once) + `cargo fmt`.
- **After every wave:** `cargo build -p fdars-core --benches --features linalg,parallel` (whole bench suite still compiles — catches any Cargo.toml serialization mistake across waves).
- **Before verify (phase gate):** all 9 module benches compile + lint clean + run; both PERMANENT perf benches still registered; both alloc_audit tests still pass; BENCH-RESULTS.md landed; `cargo fmt` sweep clean; no `src/` change; no new dependency; no `v*` tag pushed.

---

## Per-Requirement Verification Map

| Req | Behavior | Test Type | Command | Plan | Status |
|-----|----------|-----------|---------|------|--------|
| BENCH-01 | `inference_benchmarks` (t_perm_test) compiles + registers + runs — TRACER proves the full add-bench pipeline | smoke (compile+run) | `cargo build --benches …` + `cargo bench --bench inference_benchmarks -- --quick` | 51-01 | ⬜ |
| BENCH-01 | `fts_benchmarks` (ftsm), `frechet_benchmarks` (frechet_global_reg), `boosting_regression_benchmarks` (boost_fosr), `coclustering_benchmarks` (co_cluster_select) compile + register + run | smoke | `cargo bench --bench <each> -- --quick` | 51-02 | ⬜ |
| BENCH-01 | `fem_smoothing_benchmarks` (fem_smooth_gcv), `density_fda_benchmarks` (lqd_fpca), `fpca_variants_benchmarks` (fpca_der), `face_benchmarks` (mface_covariance) compile + register + run | smoke | `cargo bench --bench <each> -- --quick` | 51-03 | ⬜ |
| BENCH-01 | all 9 module benches lint clean under the primary gate (bench code linted) | lint | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | 51-01/02/03 | ⬜ |
| BENCH-01 | no new dependency; no `src/` behavior change | static | `git diff --stat` touches only `benches/`, `Cargo.toml` [[bench]] blocks, and `.planning/` (no `src/`, no `[dependencies]`/`[dev-dependencies]` add) | 51-01/02/03 | ⬜ |
| BENCH-02 | `perf_hotpaths` + `perf_parallelism` remain `[[bench]]`-registered PERMANENT | smoke | `cargo build --benches …` (both compile) + grep Cargo.toml:98-104 | 51-04 | ⬜ |
| BENCH-02 | deterministic alloc guards intact | integration | `cargo test --features dhat-heap,linalg --test alloc_audit_dpca --test alloc_audit_fpca` | 51-04 | ⬜ |
| BENCH-02 | `BENCH-RESULTS.md` consolidates Phase 47/48 before/after + 9 new baselines + governor caveat | doc review | grep BENCH-RESULTS.md for `powersave`, `alloc_audit`, `perf_hotpaths`, `−54%`/`−80.7%`, `9.9×`/`6.4×` | 51-04 | ⬜ |

---

## Wave 0 Requirements

- [ ] `benches/inference_benchmarks.rs` (t_perm_test) — **TRACER**, created by 51-01; copies `generate_curves` verbatim (NO RNG).
- [ ] `benches/fts_benchmarks.rs` (ftsm), `benches/frechet_benchmarks.rs` (frechet_global_reg — CONCRETE, not `_space`), `benches/boosting_regression_benchmarks.rs` (boost_fosr, BoostingConfig::default), `benches/coclustering_benchmarks.rs` (co_cluster_select, TINY 2×2 grid) — created by 51-02.
- [ ] `benches/fem_smoothing_benchmarks.rs` (fem_smooth_gcv, SMALLER 256-node mesh), `benches/density_fda_benchmarks.rs` (lqd_fpca), `benches/fpca_variants_benchmarks.rs` (fpca_der), `benches/face_benchmarks.rs` (irreg_fdata::mface_covariance) — created by 51-03.
- [ ] 9 `[[bench]] name=… harness=false` entries in `fdars-core/Cargo.toml` — added serialized across waves 1→2→3 (Cargo.toml is a shared file; each plan appends its own, adjacent to the PERMANENT blocks, never modifying them).
- [ ] `.planning/phases/51-…/BENCH-RESULTS.md` — created by 51-04.
- [ ] Framework install: NONE — criterion + dhat pre-installed dev-deps.

---

## Guard Ledger (BENCH-02 — soft vs hard)

| Guard | Kind | Fails CI? | Cell | Owner |
|-------|------|-----------|------|-------|
| `perf_hotpaths` (dpca/face_covariance/fem_smooth) | criterion soft (documented baseline) | NO — `cargo bench` + baseline compare only | perf_dpca/n200_m50, perf_face_covariance/n200_m30, perf_fem_smooth/nodes576 | Cargo.toml:98-104 (PERMANENT, unchanged) |
| `perf_parallelism` (frechet_anova/co_cluster) | criterion soft (thread-scaling) | NO | perf_parallelism_frechet_anova/n24_m81_nperm999, perf_parallelism_co_cluster/n200_m50_ninit8 | Cargo.toml:98-104 (PERMANENT, unchanged) |
| `alloc_audit_dpca` | dhat HARD assert | YES (deterministic) | dpca alloc-block count | tests/alloc_audit_dpca.rs (unchanged) |
| `alloc_audit_fpca` | dhat HARD assert | YES (deterministic) | fsvd/ssvd alloc-block counts | tests/alloc_audit_fpca.rs (unchanged) |
| 9 new module benches (BENCH-01) | criterion soft (first baseline) | NO | one representative sibling cell per module | benches/<module>_benchmarks.rs (new) |
| wall-time `assert!` regression tests | — | — | EXCLUDED — governor-sensitive/flaky (locked out of scope) | — |

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Instructions |
|----------|-------------|------------|--------------|
| Absolute wall-time bench medians are comparable across runs | BENCH-02 | CPU governor `powersave` is unpinned (`cpupower` pin needs sudo — unavailable) → wall-time cells are LOW-CONFIDENCE | Treat the documented medians/ratios as indicative, not authoritative; the thread-scaling DIRECTION (N-thread < 1-thread) and the allocation-block counts (dhat, governor-independent) are the HIGH-confidence signals. Re-run under a `performance` governor when a pinned environment is available to supersede the LOW-CONFIDENCE absolute medians. Recorded verbatim in BENCH-RESULTS.md. |

---

## Validation Sign-Off

- [ ] BENCH-01: all 9 module benches (`inference`, `fts`, `frechet`, `boosting_regression`, `coclustering`, `fem_smoothing`, `density_fda`, `fpca_variants`, `face`) compile via `cargo build --benches --features linalg,parallel`, lint clean under `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and each runs once via `--bench <name> -- --quick`
- [ ] BENCH-01: each new cell covers the module's SIBLING entry — none duplicates dpca/face_covariance/fem_smooth (perf_hotpaths) or frechet_anova/co_cluster (perf_parallelism)
- [ ] BENCH-01: every generator is deterministic (NO `rand` import in any new bench); data built OUTSIDE `b.iter()`; `black_box` on inputs and results
- [ ] BENCH-02: `perf_hotpaths` + `perf_parallelism` still `[[bench]]`-registered PERMANENT (Cargo.toml:98-104 unchanged); `alloc_audit_dpca`/`alloc_audit_fpca` still pass under `--features dhat-heap,linalg`
- [ ] BENCH-02: `BENCH-RESULTS.md` consolidates Phase 47 (−54% dpca blocks, −80.7% face_covariance) + Phase 48 (9.9× frechet_anova, 6.4× co_cluster) before/after, the 9 new-module first baselines, and the governor/environment caveat verbatim
- [ ] No `src/` behavior change; no new crate dependency; no `v*` tag pushed (audit milestone — MEMORY audit-milestone-no-git-tag)
- [ ] `cargo fmt` clean (whole-crate sweep at phase end); every commit `--no-verify`
- [ ] `nyquist_compliant: true` set once all above hold

**Approval:** pending
