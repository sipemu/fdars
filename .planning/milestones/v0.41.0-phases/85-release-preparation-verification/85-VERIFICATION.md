---
schema: verification
phase: 85
requirements: [REL-01]
status: passed
score: 1/1
verified: 2026-09-07
verifier: orchestrator (with user decision on the co_cluster caveat)
caveat: co_cluster golden flake (pre-existing, load-dependent, passes isolated) — logged STAB-03
---

# Phase 85 Verification — Release Preparation & Verification

REL-01: crate bumped to 0.41.0, CHANGELOG documents the breaking changes, docs refreshed, release-readiness verified. Operator performs the final tag/publish.

## Criterion — version bumped 0.40.0 → 0.41.0: PASS
`fdars-core/Cargo.toml` `version = "0.41.0"` (confirmed). README dep lines + `documentation/{GETTING-STARTED,ARCHITECTURE,DEVELOPMENT}.md` version strings bumped to 0.41. Historical `[0.40.0]` CHANGELOG / illustrative refs correctly left; MSRV 1.81 unchanged. Commit `db1b2b40`.

## Criterion — CHANGELOG [0.41.0] with breaking changes: PASS
`## [0.41.0] - 2026-09-07` present above `[0.40.0]`, breaking-framed, grouped Removed/Changed/Added covering API-01 (6 removed forms), API-02 (2 seals), API-03 (12 non_exhaustive), API-04 (renames + deriv/lp grid-enum collapses), Added (STABILITY.md, ROADMAP-TO-1.0.md, DerivDomain/DerivResult/LpDomain), with migration notes; preamble corrected (no longer claims uniformly additive). Commit `faa91e9d`.

## Criterion — docs refreshed: PASS
README + guides version-consistent; new stability docs cross-linked. No code-snippet references to removed/renamed symbols (Phases 81–83 already migrated examples/doctests — proven by `cargo build --examples` + doctests green).

## Criterion — whole-crate release-readiness gates green: PASS (with documented caveat)
- `cargo fmt --check` — PASS
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — PASS (zero warnings)
- `cargo build --features serde` — PASS
- `cargo build --examples` — PASS (28/28)
- doctests — PASS; lib tests — `2850 passed; 0 failed`; all integration tests pass EXCEPT the co_cluster golden pair
- `cargo package -p fdars-core` — PASS (sanity; no upload)

**Caveat (user-accepted 2026-09-07):** `golden_co_cluster_parallel` / `golden_co_cluster_below_threshold` fail under a full parallel/disk-pressured `cargo test` but PASS 5/0 in isolation (single-threaded, orchestrator-confirmed). Pre-existing thread-count-dependent nondeterminism in co_cluster's parallel restart selection vs a bit-exact golden; co_cluster was NOT touched this milestone (last changed Phase 48). Already logged in `documentation/ROADMAP-TO-1.0.md` (STAB-03) as a pre-1.0 gap. A clean full-suite rerun was not achievable on this host (`/home` ~17G free vs a ~66G full debug test tree — environment limit, not code). Per user decision, REL-01 is recorded PASSED with this caveat; the operator re-runs the full suite on a disk-healthy machine before the final `git tag v0.41.0` → publish.

## Boundary honored
NO `git tag v0.41.0` created; NO real `cargo publish` run (only `cargo package`). The tag/publish is the operator's final manual step.

## Verdict
Phase goal ACHIEVED. status: passed (1/1, documented caveat). Milestone v0.41.0 code + docs are release-ready pending the operator's final full-suite check + tag/publish.
