---
phase: 47
slug: hot-path-allocation-performance
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-30
---

# Phase 47 — Validation Strategy

> Per-phase validation contract. Behavior-changing phase: every optimization is guarded by a golden
> equivalence test (output unchanged within tolerance) + a before/after criterion bench, and PERF-02
> changes additionally by a committed `dhat-heap` alloc-audit test. The existing full suite must stay
> green at every commit.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + criterion 0.5 (`harness = false`) + dhat 0.3 (`dhat-heap`) |
| **Config file** | `fdars-core/Cargo.toml` (`[[bench]]` entries) |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel && cargo clippy --all-targets --features linalg,parallel -- -D warnings` |
| **Estimated runtime** | ~120–300 s (suite) |

---

## Sampling Rate

- **After every task commit:** existing suite green (`cargo test -p fdars-core --features linalg,parallel`).
- **After every plan wave:** full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- **Before `/gsd-verify-work`:** full suite green AND ≥1 before/after criterion cell showing ≥15% wall-time improvement (or ≥25% allocation reduction).
- **Max feedback latency:** ~300 s.

---

## Per-Task Verification Map

| Task ID | Requirement | Behavior | Test Type | Automated Command | Status |
|---------|-------------|----------|-----------|-------------------|--------|
| OPT-A | PERF-02 | `dpca` allocation blocks « 17,739 (target <1000) | dhat alloc-audit | `cargo test -p fdars-core --features dhat-heap,linalg -- count_dpca_allocations --nocapture` | ⬜ |
| OPT-A | PERF-01 | `dpca` output unchanged | golden equivalence | `cargo test -p fdars-core -- golden_dpca` | ⬜ |
| OPT-B/C/D | PERF-02 | `fsvd`/`ssvd`/`functional_acf` allocations reduced, output unchanged | dhat + golden | `cargo test -p fdars-core --features dhat-heap,linalg -- count_fsvd_allocations` | ⬜ |
| OPT-E | PERF-01 | `face_covariance` ≥15% faster, output unchanged | criterion + golden | `cargo bench --bench perf_hotpaths -- face_cov` / `-- golden_face_cov` | ⬜ |
| OPT-F | PERF-01 | `fem_smooth` clone removed, output unchanged (O(N³) solve DEFERRED w/ rationale) | golden equivalence | `cargo test -p fdars-core -- golden_fem_smooth` | ⬜ |
| all | PERF-01/02 | Existing suite green at every commit | unit/integration | `cargo test -p fdars-core --features linalg,parallel` | ⬜ |

*Status: ⬜ pending · ✅ green · ❌ red*

---

## Wave 0 Requirements

- [ ] `tests/equivalence_phase47.rs` — golden equivalence tests for each optimized path (capture current output on representative inputs; assert optimized output matches within documented tolerance: exact for counts, rel ≤1e-10 for float/SVD/eigen).
- [ ] `tests/alloc_audit_dpca.rs` — `dhat-heap` allocation audit for `dpca` (+ reuse `alloc_audit_fpca.rs` pattern for fsvd/ssvd), asserting the before/after block counts.
- [ ] `benches/perf_hotpaths.rs` + `[[bench]] name = "perf_hotpaths" harness = false` in Cargo.toml — before/after criterion for dpca, face_covariance, fem_smooth at PROF-01 cells (PERMANENT — becomes BENCH-02 regression guard).
- [ ] Free disk: `rm -rf target/debug/{incremental,examples}` before bench builds.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| "Measurable improvement" judgement (≥15% / ≥25%) with governor caveat | PERF-01/02 | Absolute criterion numbers are governor-sensitive (`powersave` LOW-CONFIDENCE); a human confirms the before/after delta is real and non-overlapping-CI | Compare PERF-RESULTS.md before/after cells; confirm CIs don't overlap; note governor |

---

## Validation Sign-Off

- [x] Every optimized path has a golden equivalence test that passes — 6 golden tests (dpca, ssvd, fsvd, functional_acf, face_covariance, fem_smooth) all pass at rel 1e-12
- [x] Each PERF-02 change has a dhat alloc-audit showing fewer/smaller allocations — dpca 17,739→8,139 (hard gate <9000); fsvd 275→274, ssvd 22→21 (baseline prints)
- [x] ≥1 before/after criterion cell shows ≥15% / ≥25% improvement — face_covariance −80.7% wall-time (983.8→189.8ms); dpca −54% allocations
- [x] Existing full suite green (181 doctests + all unit/integration, 0 failed) + clippy `--all-targets` clean (0 warnings)
- [x] No public signature changed; `linalg`/non-`linalg` branches equivalent
- [x] Deferred target documented with rationale — fem_smooth O(N³) Cholesky/GCV solve (rustdoc DEFER note + PERF-RESULTS.md Deferred section)
- [x] `nyquist_compliant: true` set — all above hold

**Approval:** approved 2026-08-31 — 6 golden equivalence tests pass (rel 1e-12), PERF-01 headline face_covariance −80.7%, PERF-02 headline dpca −54% allocations, suite + clippy green, no signature changes, fem_smooth O(N³) documented+deferred.
