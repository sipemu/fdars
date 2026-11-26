---
quick_id: 261126-vzq
status: complete
issues: [77, 78, 79]
commits: [03ae1e6f, 181cf3c7, 3ade8a99]
---
# Summary — alignment bug batch (#77, #78, #79)

Each issue: failing regression test first (observed red), then root-cause fix (green).

## #77 transfer_alignment warp order — 03ae1e6f
- Root cause: total warp built as `compose_warps(bridge, within)` = γ_b∘γ_i; correct is γ_i∘γ_b
  (f_i∘γ_i ≈ μ_T, μ_T∘γ_b ≈ μ_S).
- Fix: swap the order (`transfer.rs`), doc states the composition.
- Test `transfer_two_peak_non_commuting_warps_recover_source_shape`: two-peak template, targets
  μ₂∘(t+a t(1−t)), a=±0.5. Red: sup error 0.136 vs source template; green: < 0.1 (≈0.05–0.07).
  Also checks `aligned_data[i] == target_i∘gammas[i]`.
- Behaviour change: gammas/aligned_data/distances for non-commuting warps (CHANGELOG Fixed).

## #78 landmark_register zero warps — 181cf3c7
- Root cause: early-return path for inconsistent input returned `FdMatrix::zeros(n, m)` warps
  (same in `detect_and_register`).
- Fix: `identity_warps` helper — identity on `[argvals[0], argvals[last]]` (= argvals when
  consistent). Signature kept (Result would be breaking; issue lists identity as acceptable).
- Test `test_gh78_inconsistent_input_returns_identity_warps` (issue repro + wrong-length argvals +
  detect_and_register).

## #79 TSRVF ladders — 3ade8a99
- Root cause: the ladder constructions themselves were wrong (Schild returned log_to(exp_from v);
  pole returned −log_to(exp_from(−v))). The transported vector v = −log_qi(μ) was correct: its
  exact PT along qi→μ is log_μ(qi), the LogMap vector. (Issue's suggested "transport another
  vector" diagnosis is not needed.)
- Fix: standard one-rung Schild's and pole ladders (Lorenzi & Pennec 2014).
- Tests: pole ladder vs closed-form sphere PT (rel err < 1e-8; red: 0.87); Schild's rel err
  < 0.05 and quadratic shrinkage of relative error (red: 0.88); TSRVF methods agree with LogMap
  to < 1e-6 (red: Schild deviation 1.0 = 2×).
- Behaviour change: ladder tangent vectors (CHANGELOG Fixed). LogMap unchanged.

## Gates
- cargo fmt --all: clean
- clippy -p fdars-core --all-targets --features linalg,parallel,serde -D warnings: clean
- cargo test --lib alignment (654 ok), landmark (39 ok); --no-default-features --features linalg
  alignment::tests/transfer/landmark (182 ok)
- rustdoc -Dwarnings: clean; pmat quality-gate: 0 violations

## Deviations / remaining
- #78: no fallible `Result` variant added (would be breaking / new API); candidate for 1.0.
