---
phase: 49
slug: code-consolidation-dedup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-31
---

# Phase 49 — Validation Strategy

> Behavior-preserving consolidation phase. Each duplicated machinery is factored into shared
> `pub(crate)` helpers and every call site migrated, with **bit-identical** output proven by a golden
> captured from the CURRENT (pre-refactor) code and asserted (`assert_eq!`) after migration — under
> BOTH feature configs (`parallel` ON vs OFF). No public signature change; no new crate dependency.

**Scope (4 PROF-02 targets):** χ²/gamma → `src/distributions.rs` (CONS-01), SVD sign-decision core
(CONS-01), `seed_for_thread` (CONS-02), `permutation_pvalue` + `frechet_anova` primary loop (CONS-02).
**Deliberately excluded (documented, not dropped):** the χ²/gamma tail kernels stay *split* (SF-direct
vs P-direct — one kernel diverges catastrophically in the far tail, RESEARCH-verified); permutation
sites on a single advancing RNG (`t_perm`/`f_perm`/`explain-importance`×2/`famm`) and the hardcoded-LCG
`function_on_scalar::fanova` are NOT migrated (migrating changes their draws → changes output); the
Phase-48 "fold explain/importance into the generic path" idea is a **behavior-changing** deferral, not
implemented here; `frechet_anova`'s second (generic-MetricSpace) loop is document-and-skipped (no golden).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `#[test]` + integration tests in `fdars-core/tests/` |
| **Quick run (parallel ON)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --test equivalence_phase49` |
| **Parallel-OFF run** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --no-default-features --features linalg --test equivalence_phase49` |
| **Full gate** | `cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test -p fdars-core --features linalg,parallel` (+ parallel-OFF suite) |

---

## Sampling Rate

- **After every task commit:** `equivalence_phase49` filtered golden green (parallel ON) + `clippy --all-targets`.
- **After every wave:** full suite ON + **parallel-OFF suite** + clippy `--all-targets`.
- **Before verify:** both feature configs green; `cargo fmt` clean; all migrated call sites bit-identical.

---

## Per-Task Verification Map

| Req | Behavior | Test Type | Command | Status |
|-----|----------|-----------|---------|--------|
| CONS-01 | χ² SF/CDF/quantile + `reg_gamma_p` bit-identical (incl. far tail x=70.59,k=1) after distributions.rs code-motion | integration golden | `cargo test … --test equivalence_phase49 gamma` under both configs | ⬜ |
| CONS-01 | SVD sign-fix unchanged (FPCA `fix_svd_signs` + `pace_fpca` single-matrix flip via shared sign-decision core) | integration golden | `… --test equivalence_phase49 svd_sign` under both configs | ⬜ |
| CONS-02 | `frechet_anova` primary-loop p-values bit-identical after `permutation_pvalue` migration; threshold=200 path preserved | integration golden | `… --test equivalence_phase48 golden_frechet_anova` (existing) + `… --test equivalence_phase49 frechet` | ⬜ |
| CONS-02 | `seed_for_thread(seed,k)` stream == `StdRng::seed_from_u64(seed + k)` at every migrated thread-offset site | integration golden | `… --test equivalence_phase49 rng_stream` | ⬜ |
| all | Excluded sites documented (rationale comment at each) — not silently dropped | static | grep the rationale comments at the 5 un-migrated permutation sites | ⬜ |
| all | Existing suite green — BOTH feature configs; no public signature change; no new dep | integration | `cargo test --features linalg,parallel` AND `--no-default-features --features linalg` | ⬜ |

---

## Wave 0 Requirements

- [ ] `fdars-core/tests/equivalence_phase49.rs` (created by plan 49-01, appended by 49-02/03/04) — captures pre-refactor goldens for all 4 targets: gamma SF/CDF/quantile incl. far-tail, SVD signs, `frechet_anova` p-values, RNG stream.
- [ ] Capture goldens BEFORE any `src/` edit (run current code, hard-code the exact f64 bits as `const`, `#![allow(clippy::excessive_precision)]` as in `equivalence_phase48.rs`).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Instructions |
|----------|-------------|------------|--------------|
| (none) | — | This phase is fully machine-verifiable — bit-identical golden equivalence + both-config suite + clippy leave no runtime-judgement gap. | — |

---

## Validation Sign-Off

- [ ] χ²/gamma goldens (SF/CDF/quantile + `reg_gamma_p`, incl. far tail) pass bit-identically under BOTH configs after `distributions.rs` code-motion
- [ ] SVD sign-fix goldens pass bit-identically (FPCA + pace_fpca) via the shared sign-decision core
- [ ] `frechet_anova` primary-loop migrated; Phase-48 + phase49 frechet goldens bit-identical both configs; threshold=200 path preserved; second loop document-and-skipped
- [ ] `seed_for_thread` stream bit-identical to `seed_from_u64(seed+k)` at every migrated site; offset formula unchanged
- [ ] All excluded permutation sites carry a one-line rationale comment (not silently dropped); explain/importance fold-in recorded as a behavior-changing deferral
- [ ] Full suite green (both feature configs) + clippy `--all-targets` clean; no public signature change; no new dependency
- [ ] `nyquist_compliant: true` set once all above hold

**Approval:** pending
