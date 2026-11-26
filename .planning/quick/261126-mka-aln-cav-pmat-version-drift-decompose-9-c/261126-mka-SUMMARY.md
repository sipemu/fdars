---
quick_id: 261126-mka
issue: 75
status: complete
commits:
  - cf230efe test(quick-261126-mka): record bit-identity pins on pre-refactor code (refs #75)
  - e61448c1 refactor(quick-261126-mka): decompose pmat-3.45-flagged functions + bump CI pmat pin to 3.45.1 (closes #75)
---

# Quick 261126-mka: pmat version drift (#75) — SUMMARY

## Outcome
- Local pmat 3.45.1 full curated gate (`pmat quality-gate --checks complexity,satd,dead-code,duplicates`,
  run from `fdars-core/`): **9 violations → 0, PASSED (exit 0)**. Before: 9 complexity, 0 satd,
  0 dead-code, 0 duplicates — so nothing other than the 9 complexity findings differed between
  local 3.45.1 and CI 3.18.0.
- CI pin bumped 3.18.0 → 3.45.1 (`.github/workflows/rust-ci.yml`, cache key, install step).
  `--fail-on-violation` dropped: 3.45 documents it as a no-op since 3.32 (blocking violations exit 1
  by default; verified locally: rc=1 before refactor, rc=0 after). The installed-version check now
  compares `pmat --version | head -n1`, because 3.45 prints extra build-metadata lines (the old full
  string comparison would never match → reinstall every CI run).

## Refactors (private helpers only; public API unchanged)
| function | decomposition |
|---|---|
| `ClassifFit::predict_from_scores` | `discriminant_predict(g, score_fn)` (LDA+QDA softmax/argmax) + `knn_predict_one` |
| `CorrelationMatrixSpace::weighted_frechet_mean` | shared `spaces::check_mean_inputs` + `renormalize_unit_diagonal` |
| `SphericalSpace::weighted_frechet_mean` | `check_mean_inputs` + `extrinsic_init` + `weighted_log_sum` + `normalize_or_err` |
| `corrected_oracle_gradient` (test) | `oracle_forward` + `oracle_backward` + `oracle_term` |
| `Lfd::apply` | `validate_apply` + `apply_curve` |
| `MfpcaResult::project` | `validate_project` (same error order) |
| `StreamingBd::bd_one_inner` / `bd_one_from_row` | both delegate to `bd_one_by(x_at)` + `inside_band` (also removes the duplicated body) |
| `RollingReference::push` | free fn `remove_sorted_value` |
Same floating-point operation order everywhere.

## Validation
- 8 pin tests (FNV-1a over `f64::to_bits` of all outputs, helper `test_helpers::bits_fingerprint`,
  inputs from integer-LCG `test_helpers::lcg_values`) written and run on the UNMODIFIED code, values
  embedded, committed (cf230efe) BEFORE any refactor; all pass unchanged after:
  `predict_from_scores_bit_identity_pin` (6 method/branch combos, hand-built parameters — no SVD
  dependence), `correlation_mean_bit_identity_pin`, `spherical_karcher_bit_identity_pin`,
  `corrected_oracle_gradient_bit_identity_pin`, `lfd_apply_bit_identity_pin` (orders 1–3,
  constant/grid coefs, uniform/non-uniform grids), `mfpca_project_bit_identity_pin` (hand-built
  result incl. sub-threshold scale + error-order checks), `streaming_bd_bit_identity_pin`
  (tie-heavy data, depth_one + depth_batch), `rolling_push_bit_identity_pin` (duplicates and ±0.0
  through eviction).
- Caveat: pins involving `ln/exp/acos/sin/cos` are recorded on x86_64 Linux glibc; the CI runner is
  the same arch/OS, but a different libm could in principle change last bits.

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib -- classification frechet soft_dtw pda spm streaming_depth explain_generic`: 499 passed.
- `cargo test -p fdars-core --doc -- pda mfpca streaming_depth frechet classification`: 5 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean
  (pin commit cf230efe alone tripped `items_after_test_module` in test_helpers.rs; fixed in e61448c1).
- pmat 3.45.1 curated gate: PASSED, 0 violations.

## Deviations / notes
- STATE.md not edited (orchestrator override); worktree `cav-75`, not pushed.
- Cannot verify the CI `cargo install pmat --version 3.45.1 --locked` step locally beyond the crate
  existing on crates.io (latest is 3.46.0; local 3.45.1 was built from the crates.io source archive).
