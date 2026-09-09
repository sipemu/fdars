# Phase 91 — Suite-Wide Robustness Audit (ROBUST-01, ROBUST-02)

**Audited:** 2026-09-09
**Method:** Reuses the Phase 90 technique — run the full suite under each supported feature configuration and under a deliberately-unsupported no-linalg config as a fast leak-detector, plus a thread-count / repeated-run determinism sweep, then classify every candidate assertion by fragility class and disposition. Evidence is empirical (real full-suite runs), not inferred.

**Headline finding:** ZERO additional fragile tests require fixing. The backend-fragility class is fully covered (Phase 90 guards + pre-existing `#[cfg(feature="linalg")]` guards), RNG/thread-order is robust (bit-identical under a thread-count sweep), and the environment/disk hypothesis is disproven. The whole suite is green under both CI-supported configs.

---

## Config matrix (ROBUST-01 evidence)

| Config | Command | Result |
|--------|---------|--------|
| CI config 1 (linalg + serde + parallel) | `cargo test -p fdars-core --features linalg,parallel,serde` | exit 0 — **3676 passed, 0 failed**; 3 goldens ok |
| CI config 2 (no-default + linalg) | `cargo test -p fdars-core --no-default-features --features linalg` | exit 0 — **3672 passed, 0 failed, 5 ignored** (unrelated doctests); 3 goldens ok |
| Leak-detector (default features, NO linalg → nalgebra backend) | `cargo test -p fdars-core --features parallel` | exit 0 — **3660 passed, 0 failed, 8 ignored** (the 3 Phase-90 goldens + 5 unrelated doctests) |

The two CI configs of record are from `.github/workflows/rust-ci.yml` (`cargo test --features linalg,parallel,serde` and `cargo test --no-default-features --features linalg`). The leak-detector deliberately omits `linalg` to route `fdata_to_pc` through the nalgebra SVD backend — the exact condition that exposed the Phase 90 flake. It surfaces **no** unguarded backend-dependent failure: everything is either backend-independent (passes) or already guarded (ignored/compiled-out).

## Thread-count & repeated-run sweep (determinism, ROBUST-02)

On the serde CI config (`--features linalg,parallel,serde`), machine `nproc=20`:

| Thread count | Command | Result |
|--------------|---------|--------|
| RAYON_NUM_THREADS=1 | `RAYON_NUM_THREADS=1 cargo test -p fdars-core --features linalg,parallel,serde` | 3676 passed, 0 failed, goldens ok |
| RAYON_NUM_THREADS=2 | `RAYON_NUM_THREADS=2 cargo test -p fdars-core --features linalg,parallel,serde` | 3676 passed, 0 failed, goldens ok |
| default (20) | `cargo test -p fdars-core --features linalg,parallel,serde` | 3676 passed, 0 failed, goldens ok |

**Passed counts are bit-identical (3676) across thread counts** ⇒ the per-thread reseed contract `seed_for_thread(seed, k) = StdRng::seed_from_u64(seed.wrapping_add(k as u64))` genuinely makes output thread-count-independent. Combined with Phase 90's 10× consecutive `--features linalg,parallel` green runs and 3 consecutive serde-config runs, there is no observed intermittency.

## Fragility-class inventory

### Class 1 — Feature-backend-dependent goldens
`assert_eq!` bit-identity on outputs routed through the feature-gated faer/nalgebra SVD branch in `fdata_to_pc`.

| Test / file | Disposition | Rationale | Reproduction |
|-------------|-------------|-----------|--------------|
| `golden_co_cluster_parallel`, `golden_co_cluster_below_threshold` (`equivalence_phase48.rs`) | covered-by-guard | Phase 90 `#[cfg_attr(not(feature = "linalg"), ignore)]`; ignored off-linalg, pass under both linalg configs | `cargo test -p fdars-core --features parallel --test equivalence_phase48 -- golden_co_cluster` → `2 ignored` |
| `svd_sign_fpca_two_matrix_bit_identical` (`equivalence_phase49.rs`) | covered-by-guard | Phase 90 ignore-guard | `cargo test -p fdars-core --features parallel --test equivalence_phase49 -- svd_sign_fpca_two_matrix_bit_identical` → `1 ignored` |
| `svd_equivalence.rs` (2 asserts) | covered-by-guard | Pre-existing `#[cfg(feature="linalg")]` — compiled out off-linalg | present only under `--features linalg` |
| `validate_against_r.rs` (linalg-specific asserts) | covered-by-guard | Pre-existing `#[cfg(feature="linalg")]` guard | present only under `--features linalg` |
| `svd_sign_pace_eigenfunctions_single_matrix_bit_identical` (`equivalence_phase49.rs`) | deterministic-safe-to-leave (INTENTIONALLY UNGUARDED) | Passes off-linalg (`1 passed`); its single-matrix PACE eigenfunction sign path does not hit the near-zero faer/nalgebra divergence. Per 90-DIAGNOSIS §7 it must NOT be guarded. | `cargo test -p fdars-core --features parallel --test equivalence_phase49 -- svd_sign_pace_eigenfunctions_single_matrix_bit_identical` → `1 passed` |

### Class 2 — RNG / thread-order nondeterminism
`assert_eq!` bit-identity on outputs of parallel/seeded code (co_cluster n_init map, permutation p-values, parallel reductions).

| Candidate | Disposition | Rationale |
|-----------|-------------|-----------|
| All parallel/seeded goldens (`equivalence_phase48.rs` frechet_anova, permutation goldens; co_cluster reduce) | deterministic-safe-to-leave | The thread-count sweep above shows bit-identical passed counts at RAYON_NUM_THREADS 1/2/20; the per-thread reseed contract makes output thread-count-independent. No intermittency across 10× (P90) + repeated runs. |
| χ²/gamma distribution goldens (`equivalence_phase49.rs`, ~104 asserts) | deterministic-safe-to-leave | Pure arithmetic (no RNG, no SVD backend) — bit-identical by construction; all pass under every config including no-linalg. |
| R-reference value goldens (`validate_against_r.rs`, ~95 asserts) | deterministic-safe-to-leave / covered-by-guard | Backend-independent numerical references; pass off-linalg where not linalg-gated. |

### Class 3 — Environment / BLAS-thread / disk dependence
Assertions hypothesized to vary with `RAYON_NUM_THREADS`, BLAS/faer runtime detection, or disk/timing.

| Candidate | Disposition | Rationale |
|-----------|-------------|-----------|
| (the golden bit-identity asserts, re. env) | deterministic-safe-to-leave | Deterministic across every run in this audit (thread sweep, repeated runs, both CI configs). No env-pinning or serialization warranted — no proven race. |
| `alloc_audit_fpca.rs`, `alloc_audit_dpca.rs` | deterministic-safe-to-leave | Allocation audits, no value/bit-identity assertions to flake. |

## Environment/disk hypothesis

The Phase 90 "intermittent flake / disk pressure" hypothesis is **overturned** (see 90-DIAGNOSIS §4): the golden flake was a deterministic feature-configuration artifact (linalg off → nalgebra backend), not environmental. This audit finds no residual environment/disk/thread-timing dependence anywhere in the suite: every assertion is deterministic within a fixed feature set, and thread-count-independent. No serialization guard, no `RAYON_NUM_THREADS` pin, and no new dependency are warranted.

## Findings (ROBUST-01)

**Zero additional fragile tests require fixing.** Reasons per class:
- **Backend-dependent goldens:** all covered — Phase 90 guarded the 3 that leaked; `svd_equivalence.rs` and `validate_against_r.rs` were already `#[cfg(feature="linalg")]`-guarded; the no-linalg leak-detector surfaces no further failure.
- **RNG/thread-order:** robust — bit-identical passed counts under a 1/2/20 thread sweep; the seeded per-thread reseed contract holds.
- **Environment/disk:** disproven hypothesis — deterministic across all runs.

The one deliberate non-action is the PACE sibling `svd_sign_pace_eigenfunctions_single_matrix_bit_identical`, recorded as intentionally-unguarded (guarding it would be over-guarding a sound test).

## Disposition summary

| Fragility class | Candidates | covered-by-guard | deterministic-safe-to-leave | fixed (this phase) |
|-----------------|-----------|------------------|-----------------------------|--------------------|
| Feature-backend-dependent goldens | 6 groups | 4 (P90 ×3 + pre-existing files ×N) | 1 (PACE, intentional) | 0 |
| RNG / thread-order nondeterminism | 3 groups | — | 3 | 0 |
| Environment / BLAS / disk | 2 groups | — | 2 | 0 |
| **Total** | — | — | — | **0** |

## Conditional-fix branch outcome (ROBUST-02, Task 3)

**No-op — zero code changes.** The leak-detector found no NEW unguarded fragile test, so the Phase-90-style conditional fix was not triggered. This is a legitimate completed outcome: all backend-dependent tests are already covered by Phase-90 + pre-existing guards; RNG/thread-order is robust; env/disk is disproven. No `assert_eq!` was relaxed to a tolerance (none needed — no FP nondeterminism proven with a supported config), and no new crate dependency was added (`Cargo.toml` unchanged).
