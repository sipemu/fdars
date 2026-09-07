# Phase 85: Release Preparation & Verification - Context

**Gathered:** 2026-09-07
**Status:** Ready for planning
**Mode:** Scope-bound (release prep; content determined by Phases 81–84 outcomes)

<domain>
## Phase Boundary

Prepare `fdars-core` for a 0.41.0 release and VERIFY release-readiness: bump the version, write the CHANGELOG `[0.41.0]` entry documenting all breaking changes, refresh version references in docs, and confirm the whole-crate gates + serde build + all 28 examples/doctests are green. **This phase PREPARES + VERIFIES only.** The `git tag v0.41.0` push → crates.io publish (release.yml) is the FINAL OPERATOR-DRIVEN step — this phase must NOT create the tag or publish. Ships as 0.41.0 (NOT 1.0).

</domain>

<decisions>
## Implementation Decisions

### Version bump (0.40.0 → 0.41.0)
- `fdars-core/Cargo.toml:3` `version = "0.40.0"` → `"0.41.0"`.
- README.md dependency lines: `fdars-core = "0.40"` (line ~44) and `fdars-core = { version = "0.40", ... }` (line ~84) → `"0.41"`.
- Grep the tree for any other `0.40` version references in tracked docs/examples and bump consistently (do NOT touch CHANGELOG's historical `[0.40.0]` heading or Cargo.lock's transitive pins beyond fdars-core's own).

### CHANGELOG `[0.41.0]` entry (breaking changes explicitly called out)
Add a new `## [0.41.0] - 2026-09-07` section ABOVE `[0.40.0]`. This is the FIRST breaking release after the additive run — lead with a clear "**BREAKING**" framing. Document, grouped by keepachangelog sections:
- **Removed** (API-01): the 6 deprecated forms `mean_2d`, `fanova`, `random_tukey_2d`, `random_projection_2d`, `fraiman_muniz_2d`, `modal_2d` + their crate-root/prelude re-exports. Migration: use `mean(…, Dim::Two)`, `fanova_seeded(…, 42)`, and `<depth>(…, Dim::Two)`.
- **Changed** — sealing (API-02): `sort_nan_safe`, `solve_gaussian_pub` are now `pub(crate)` (were unintentionally `pub`).
- **Changed** — `#[non_exhaustive]` (API-03): added to 10 public enums (`PeerPenalty`, `LambdaChoice`, `LambdaMethod`, `DesignCriterion`, `OptimalityKind`, `ExtrapolationPolicy`, `ImputationMethod`, `SelectionCriterion`, `BasisType`, `BasisCriterion`) + 2 result structs (`OptimBandwidthResult`, `KnnCvResult`) — downstream exhaustive `match`/struct-literal on these now needs a wildcard/`..`.
- **Changed** — naming (API-04): renamed `funhddC_cluster`→`fun_hddc_cluster`, `FosrResult2d`→`Fosr2dResult`, `GmmResult`→`GmmFitResult`; collapsed `deriv_1d`/`deriv_2d`→`deriv(…, DerivDomain)`→`DerivResult`, and `lp_self_1d`/`lp_cross_1d`/`lp_self_2d`/`lp_cross_2d`→`lp_self`/`lp_cross(…, LpDomain)`. Migration notes for each.
- **Added**: `documentation/STABILITY.md` (semver + MSRV policy), `documentation/ROADMAP-TO-1.0.md` (1.0 gap checklist); new public enums `DerivDomain`, `DerivResult`, `LpDomain`.
- **Note**: no numeric/behavioral change — breaking is API shape only; MSRV unchanged (1.81 crate / 1.84 linalg).
- Update the CHANGELOG preamble if it claims the span is "additive and non-breaking" — 0.41.0 is explicitly breaking.

### Docs refresh
- Ensure README + guides don't reference the removed/renamed symbols in code snippets (spot-check; the compile-time proof is doctests/examples already migrated in Phases 81–83). Bump version strings. Optionally link STABILITY.md / ROADMAP-TO-1.0.md from README's docs list.

### Verification (release-readiness — the load-bearing part of this phase)
Whole-crate gates must ALL be green:
- `cargo fmt --check`
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- `cargo test` (lib + doctests + integration)
- `cargo build --features serde`
- `cargo build --examples` (all 28)
- `cargo package` dry-run sanity (optional but good release hygiene: `cargo package -p fdars-core --no-verify` or `cargo publish --dry-run` — confirms the crate is packageable; do NOT actually publish).

**KNOWN FLAKE (see MEMORY cocluster-svdsign-golden-flake-fullrun):** under a full parallel `cargo test`, `golden_co_cluster_parallel`, `golden_co_cluster_below_threshold` (equivalence_phase48) and `svd_sign_fpca_two_matrix_bit_identical` (equivalence_phase49) may fail (env/BLAS/disk-pressure dependent) yet PASS per-binary in isolation. If they fail in the full run, re-run each binary alone (`cargo test --test equivalence_phase48`, `--test equivalence_phase49`) to confirm green — that counts as the suite passing for release purposes. They are NOT a v0.41.0 regression (already logged in STAB-03 / ROADMAP-TO-1.0.md). Free disk first if needed (`rm -rf target/debug/{incremental,examples}`; `/home` was ~94%).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- CHANGELOG.md already follows keepachangelog + SemVer; `[0.40.0] - 2026-09-07` is the current top entry.
- All breaking changes are already IN the code (Phases 81–83) and proven by green gates + 28 examples/doctests — Phase 85 documents + re-verifies them, it does not re-implement.

### Established Patterns
- Release mechanics (MEMORY v0370-shipped): bump `fdars-core/Cargo.toml` version → commit → `git tag vX.Y.Z` → push tag → release.yml runs cargo publish. **Phase 85 stops BEFORE the tag** (operator step).
- `docs/` is gitignored; tracked docs are in `documentation/`.

### Integration Points
- Bumping version updates `Cargo.lock` for the `fdars-core` entry — run a build/`cargo update -p fdars-core --precise 0.41.0` is not needed; a normal build refreshes the lock. Commit the lock change.

</code_context>

<specifics>
## Specific Ideas

- Commit hazard (MEMORY): pre-commit hook full-test-suite times out at 30s → `cargo fmt` then `git commit --no-verify`, run gates out-of-band.
- Do NOT create a git tag or run `cargo publish` — explicitly out of scope; leave that for the operator. The phase's "done" is: version bumped, CHANGELOG written, docs consistent, all gates verified green (flake handled per note).

</specifics>

<deferred>
## Deferred Ideas

- The actual `git tag v0.41.0` + crates.io publish — operator's final manual step after this phase verifies readiness.
- All STAB-03 checklist items (future 1.0 milestone).

</deferred>
