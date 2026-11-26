---
quick_id: 261126-mun
issue: 74
status: complete
commits:
  - 2cf3defc test(quick-261126-mun): regenerate fdasrvf goldens with CRAN fdasrvf 2.6.0 (closes #74)
---
# Summary: fdasrvf goldens regenerated with 2.6.0 (ALN-CAV, GH #74)

Tests/docs only; no library code or numeric change. No public API added.

## What was regenerated (scripts + outputs in this directory, `*_260.R` / `*_260.out`)
Note: plain `R_LIBS_USER=~/.cache/R-lib Rscript` still loads 2.5.0 (the user's R_LIBS_USER
env points at `~/R/library`); the scripts prepend `.libPaths(c("~/.cache/R-lib", ...))`.

| Golden | 2.5.0 -> 2.6.0 | fdars residual | Bound (old -> new) |
|---|---|---|---|
| `curvebox_data` distances (#54) | bit-identical | 9.5e-15 | 1e-12 -> 5e-14 |
| `curve_depth` amplitude, T=480 (#54) | moved <= 3.5e-4 | 2.07e-3 (was 2.09e-3) | 3e-3 -> 2.5e-3 |
| `curve_depth` phase, T=480 (#54) | moved up to 1.2e-2 (curve 5 0.676 -> 0.664) | 1.01e-2 (was 2.2e-2) | ranking only -> 1.5e-2 + ranking |
| `smooth.data` (#56) | bit-identical | < 1e-14 | unchanged (1e-14) |
| `f_to_srvf(smooth=TRUE)` (#56) | real fn == hand-written 2.5.0 stand-in (0 ulp) | L2 dist 2.7e-3 (max 9.6e-3) | new: < 4e-3 |

Rankings of all six curves (amplitude and phase) still agree exactly.

**Why depth changed:** the R sources of `curve_depth`, `curve_boxplot`, `curvebox_data` are
identical between 2.5.0 and 2.6.0 (deparsed and diffed); the change comes from 2.5.1's DP
(`DPQ`: 3x finer cost grid + linear `q2` interpolation; `DPQ2`: linear SRVF integration,
`nbhd_dim` 7 -> 10) and `gam_to_psi` now renormalising `psi` with an `atan2` angle — the old
test comment attributed the phase gap to exactly that missing renormalisation, which is why
phase agreement halved. fdasrvf 2.6.0's own T = 60 -> 480 drift is 1.2e-2 (amp) / 1.6e-2 (phase).

## Gates
- `cargo fmt --all` clean (an unrelated in-flight orchestrator edit to `metric/dtw_lb/tests.rs`
  showed up in the tree mid-run; it was committed by its owner as 77ba1f50, not included here).
- `cargo test -p fdars-core --lib srsf_smoothing` 6/6, `curve_shape::downstream_tests` 23 total ok.
- Doc tests `srsf_smoothing` 2/2, `curve_shape` 24/24.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.

## Not done / notes
- #48 (fdasrsf 2.7.5, satisfies >= 2.7.4) and #55 (fdasrsf 2.7.5) have no fdasrvf goldens.
- The legacy `validation/R/*` suite (fdasrvf alignment/Karcher/tolerance/elastic FPCA/regression,
  5–20 % relative tolerances, generating version not recorded) predates the backlog and was not
  in #74's scope; regenerating its expected JSON with 2.6.0 would be a separate task.
