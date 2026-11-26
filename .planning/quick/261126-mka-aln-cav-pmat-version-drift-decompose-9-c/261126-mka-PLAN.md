---
quick_id: 261126-mka
issue: 75
type: quality / CI tooling
---

# Quick 261126-mka: pmat version drift (#75)

## Goal
Local pmat 3.45.1 flags 9 cognitive-complexity violations that CI's pinned
pmat 3.18.0 passes. Make local and CI agree: decompose the flagged functions
behaviour-preservingly to cognitive complexity <= 23, then bump the CI pin to
3.45.1 if the full curated gate passes locally.

## Flagged (pmat 3.45.1, `pmat quality-gate --checks complexity,satd,dead-code,duplicates` from fdars-core/)
| file | fn | cog |
|---|---|---|
| classification/fit.rs | `predict_from_scores` | 26 |
| frechet/spaces/correlation.rs | `weighted_frechet_mean` | 28 |
| frechet/spaces/spherical.rs | `weighted_frechet_mean` | 37 |
| metric/soft_dtw.rs (test) | `corrected_oracle_gradient` | 31 |
| pda.rs | `Lfd::apply` | 29 |
| spm/mfpca.rs | `MfpcaResult::project` | 30 |
| streaming_depth/bd.rs | `bd_one_inner`, `bd_one_from_row` | 25, 25 |
| streaming_depth/rolling.rs | `RollingReference::push` | 54 |
satd / dead-code / duplicates: 0 violations locally before changes.

## Validation strategy (record first, then refactor)
1. Add one bit-identity pin test per function (FNV-1a over `f64::to_bits` of
   all outputs on deterministic LCG inputs; hand-built parameter structs where
   possible to avoid SVD dependence), run on the UNMODIFIED code, embed the
   recorded fingerprints, commit before any refactor.
2. Refactor; pins + existing module tests must pass unchanged.

## Tasks
1. Pins (commit 1).
2. Decompose each function into private helpers (same float-op order).
3. Re-run pmat full curated gate; fmt, clippy --all-targets, module tests.
4. Bump CI pin 3.18.0 -> 3.45.1 iff gate passes; replace the no-op
   `--fail-on-violation` flag as required by 3.45.
