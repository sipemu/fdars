---
phase: 55
title: Gram-Matrix Export
status: passed
commit: 30431aa2
---

# Phase 55 Verification — Gram-Matrix Export (GAK-05/06)

Each ROADMAP "Phase 55" success criterion → PASS/FAIL with evidence.

## Criterion 1 — Train Gram (n_train × n_train, symmetric, PSD, unit diagonal) carrying training self-kernels
**PASS.** `gak_gram_train` returns `GakGramTrain { gram, log_self, sigma, .. }`. `gram` is n×n, symmetric by assignment (shares `build_train_gram` with `gak_gram_matrix`), unit diagonal, PSD.
Evidence: `test_gram_train_shape_psd` — asserts `shape == (n,n)`, bit-exact symmetry, `|diag−1|<1e-12`, `min_eig ≥ −1e-8`, and `log_self().len() == n`.

## Criterion 2 — Predict Gram n_test × n_train (asserted, not transposed), cross-normalized against STORED training self-kernels, same σ
**PASS.** `gak_gram_predict` returns n_test × n_train; entry `(t,j) = exp(loggak(test_t,train_j,σ) − 0.5·(loggak(test_t,test_t,σ) + train.log_self[j]))` using `train.sigma` + `train.log_self`.
Evidence: `test_gram_predict_shape` — asserts exactly `(3,5)` with n_test=3 ≠ n_train=5 (a transpose would fail). `test_gram_predict_reproduces_train` — predict(train,train_data) ≈ train.gram within 1e-12, proving stored-diagonal cross-normalization is correct.

## Criterion 3 — Split enforced: prediction reuses identical σ + stored training self-kernels (not test-only), every entry ∈ [0,1]
**PASS.** σ and training diagonals are read from the `GakGramTrain` result, never recomputed from the test set; normalization uses `train.log_self[j]`.
Evidence: `test_gram_predict_sigma_consistency` — fit σ=3.7; test set's own median σ differs by >1; predict entry matches a manual `loggak(...,3.7)` recompute (would differ if test-σ were used). `test_gram_predict_normalized` — every entry ∈ [0,1], identical curve → ≈1.0 in its column.

## Criterion 4 — Rustdoc example demonstrates end-to-end handoff; Gram stays O(n²), diagonals computed once (no 2× recomputation)
**PASS.** Doctests on `gak_gram_train` and `gak_gram_predict` show train Gram + n_test×n_train predict Gram → commented `SVC(kernel='precomputed')` handoff (no Python dependency). `build_train_gram` computes the O(n) diagonal once and reuses it in normalization; predict computes test self-kernels once (O(n_test)).
Evidence: `cargo test --doc gak` (4 passed, 2 new); code path in `build_train_gram` (single `diag_log` pass) and `gak_gram_predict` (single `test_self` pass).

## Gate Results
- lib tests: 17 passed / 0 failed (`--features linalg`, and default features).
- doctests: 4 passed / 0 failed.
- fmt: clean. clippy `--all-targets --features linalg,parallel -D warnings`: clean.

**All 4 criteria PASS → status: passed.**
