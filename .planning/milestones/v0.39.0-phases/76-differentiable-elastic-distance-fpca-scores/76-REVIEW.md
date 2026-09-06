---
phase: 76-differentiable-elastic-distance-fpca-scores
reviewed: 2026-09-06T00:00:00Z
depth: deep
files_reviewed: 6
files_reviewed_list:
  - fdars-core/src/metric/soft_dtw.rs
  - fdars-core/src/alignment/differentiable.rs
  - fdars-core/src/alignment/mod.rs
  - fdars-core/src/metric/mod.rs
  - fdars-core/src/regression.rs
  - fdars-core/src/lib.rs
findings:
  critical: 0
  high: 0
  medium: 0
  low: 2
  total: 2
status: findings
---

# Phase 76: Code Review Report

**Reviewed:** 2026-09-06
**Depth:** deep (cross-file AD chain-rule verification + independent numeric reproduction)
**Files Reviewed:** 6
**Status:** findings (2 low-severity, non-blocking)

## Summary

Phase 76 adds three generic-over-`Scalar` (forward-mode AD) operations alongside their
existing f64 counterparts: `soft_dtw_distance_generic`, `amplitude_distance_at_warp_generic`
(fixed-warp SRSF amplitude distance), and `project_scores_generic` / `FpcaResult::project_generic`
(FPCA score projection). Correctness feeds Phase 77 + downstream ML, so the review focused on
gradient correctness, f64 parity, and the two executor claims.

**Verdict: the implementation is correct and additive.** All AD chain-rule derivations were
verified by hand and independently reproduced numerically. The full lib suite (2853 tests),
clippy `--all-targets`, and `cargo fmt --check` are all green. No existing public f64 signature
was changed, and no new crate dependency was added. The two low-severity findings are documented
domain edge cases (SRSF/L2 `sqrt` singularities), already covered by the phase's threat model and
by callers-own-range-checking documentation — they are not defects introduced by this work.

### Verdict on CRITICAL claim #1 — pre-existing oracle bug: CONFIRMED REAL and CONFIRMED PRE-EXISTING

The shipped `soft_dtw_backward` (`metric/soft_dtw.rs:262-306`) has a genuine bug. The reverse
double-loop iterates `i` from `n`→`1` and `j` from `m`→`1`. On the **first** iteration `(i=n, j=m)`
the endpoint seed `e[n][m] = 1.0` (set at line 267) is immediately overwritten: all three
neighbour contributions are gated off at the endpoint (`a`: `i < n` false; `b`: `j < m` false;
`c`: both false), so `e[i][j] = a + b + c = 0.0` (line 302) zeroes the seed. With `E` all-zero,
`soft_dtw_accumulate_gradient` returns an all-zero gradient.

- **Independently reproduced** (scratch reimplementation, x=[0.1,0.4,0.9,1.2,0.7],
  y=[0.2,0.3,1.0,1.1,0.6], gamma=1.0): shipped oracle = `[0,0,0,0,0]`; the endpoint-preserving
  ("corrected") oracle = `[-0.4515, -0.2678, 0.3106, 0.8372, -0.1703]`, which matches an
  independent central FD (h=1e-6) to `3.4e-10`. So the corrected reference is the true gradient
  and the shipped one is genuinely all-zeros.
- **Genuinely pre-existing:** the identical `e[i][j] = a + b + c` with `e[n][m] = 1.0` and no
  endpoint-skip guard is present at the phase parent `315e7a1f` and much earlier at commit
  `6bd5c4ce` (the metric.rs split). Phase 76 did **not** touch `soft_dtw_backward` — it is
  additive-only. The `git log -S 'e[n][m] = 1.0'` match on `cc8a44fc` is a false positive caused
  solely by the *test's* `corrected_oracle_gradient` adding a second `e[n][m] = 1.0` occurrence.
- **Masking mechanism confirmed:** `soft_dtw_barycenter` descends on this ~zero gradient, so the
  barycenter update is a near no-op and the pre-existing barycenter tests still pass — the bug is
  latent. This is correctly logged as a backlog item (per the SC #1 test doc-comment) rather than
  fixed in-scope. See WR/IN findings for the recommendation to file it.

Because the shipped oracle is all-zeros, SC #1 (Dual vs oracle ≤1e-9) could NOT be honestly met
against the *shipped* oracle. The executor's resolution — validate the Dual gradient against a
**corrected** in-test oracle (endpoint preserved) that is itself independently confirmed by central
FD — is legitimate and non-circular (the corrected oracle is a hand-written analytic
forward/backward DP recursion, a fundamentally different algorithm from forward-mode operator
overloading). SC #1 is honestly met against a correct reference.

### Verdict on CRITICAL claim #2 — is the new Dual soft-DTW gradient genuinely CORRECT? YES, CONFIRMED CORRECT (non-circular)

- The Dual gradient equals the corrected analytic oracle to ≤1e-9 (`dual_gradient_vs_oracle`) and
  central FD to ≤1e-6 across gamma ∈ {0.5, 1.0, 2.0} for all 5 coordinates (`dual_gradient_vs_fd`).
- The FD arm is a true `(f(x+h)−f(x−h))/2h` computed via `soft_dtw_distance_generic::<f64>` — it
  is NOT the Dual result re-derived. The corrected oracle is likewise an independent DP recursion.
  Both independent references agree with each other (and with my own separate FD reproduction to
  3.4e-10), so the check is not "zeros compared to zeros" and not circular.
- Test data is spanning/non-degenerate (5-point series exercising multiple DP paths, gamma
  non-trivial), so the gradient is genuinely non-zero and the test is not rigged.

**The Dual soft-DTW gradient is correct.**

---

## Claim-by-claim evidence (claims #3–#5)

### Claim #3 — `inf*0 = NaN` tangent fix: CORRECT, f64 value path unchanged

`softmin3_generic` (`metric/soft_dtw.rs:73-115`) adds a per-term guard:
`term(v) = if v >= S::infinity() { S::zero() } else { S::exp((v - min_val) * neg_inv_gamma) }`.

- **Primal/f64 parity holds bit-for-bit.** The substitution only fires when `v` is exactly `+inf`.
  For that case the original f64 code computed `exp((inf - finite)*(-1/gamma)) = exp(-inf) = 0.0`;
  the new code returns `S::zero() = 0.0`. Identical. For finite `v` the else-branch is the exact
  original expression. When all three are `+inf`, the earlier `min_val >= S::infinity()` guard
  (line 90) returns before `term` is reached. (`gamma > 0` is required, matching the existing
  contract.) The `f64_parity` test asserts full `assert_eq!` bit-identity across
  gamma ∈ {0.1, 1.0, 10.0} plus a 20-point series — green.
- **Tangent fix is genuinely needed and correct.** Under `Dual`, `(inf_sentinel - min_val)` has
  value=+inf, tangent=0; `* neg_inv_gamma` (a constant) applies the product rule
  `0*(-1/γ) + inf*0 = NaN`, and `exp` then propagates NaN. A `+inf` sentinel carries no gradient,
  so substituting exact `S::zero()` (value 0, tangent 0) is the mathematically correct tangent, and
  it removes the NaN. Verified by inspection of `Dual::mul` (`autodiff.rs:235-245`).

### Claim #4 — SRSF amplitude-at-warp correctness (`alignment/differentiable.rs`): CORRECT

- `generic_linear_interp` (33-55): `curve[j]*(1-alpha) + curve[j+1]*alpha`, alpha an f64 constant
  → linear in curve values, gradient flows correctly. Endpoint clamping and
  `partition_point(...).saturating_sub(1).min(m-2)` bracket are sound; `dt <= 0` guard prevents
  divide-by-zero on non-increasing grids.
- `generic_srsf_central_diff` (64-87): central diff interior + one-sided boundaries; `m==1` guard
  returns `S::zero()` for the derivative and avoids `curve[1]` OOB access. **Chain rule verified
  by hand:** for `q(v) = signum(v)·sqrt(|v|)`, the composite `signum(d)*sqrt(abs(d))` under `Dual`
  yields tangent `t/(2·sqrt(|v|))` (signum tangent 0; abs tangent `t·sign(v)`; sqrt divides by
  `2·sqrt(|v|)`; product rule cancels `sign(v)²=1`). This equals the true analytic
  `dq/dv = 1/(2·sqrt(|v|))` for both v>0 and v<0. Correct.
- `generic_l2_srsf_distance` (93-100): `sqrt(Σ (q1-q2)² w)` — standard weighted-L2; gradient flows
  through q2. Correct.
- `ẋ = 0` domain issue: at a derivative zero, `abs`'s at-zero tangent is 0 but `sqrt(0)` divides by
  `2·0` → NaN tangent. This is documented in the module header and threat model (T-76-01, accepted;
  callers own range checking), and the test grid `[0.1, 0.9]` deliberately avoids the sin/cos
  derivative zeros at t=0.25/0.75. See LOW-01.
- SC #4 (Dual vs FD ≤1e-5) and SC #7 (f64 parity ≤1e-10) are green; both use spanning curves with
  nonzero derivatives on the grid.

### Claim #5 — additive / non-breaking: CONFIRMED

- `soft_dtw_distance` signature unchanged (`x:&[f64], y:&[f64], gamma:f64) -> f64`); body delegates
  to `soft_dtw_distance_inner::<f64>` with `assert_eq!` bit-parity gating.
- `elastic_distance`, `amplitude_distance`, `phase_distance_pair` (`alignment/pairwise.rs`) and
  `FpcaResult::project` (`regression.rs:106`) are byte-for-byte unchanged (only additive methods
  inserted after `project`).
- No `Cargo.toml` change in either commit; `grep 'num[-_]traits' Cargo.toml` = 0. MSRV preserved.
- Re-exports are minimal per plan: `soft_dtw_distance_generic` + `amplitude_distance_at_warp_generic`
  at module level only; `project_scores_generic` at crate root. No duplicate/conflicting exports.
- Gates: full lib suite 2853/2853 green; `clippy --all-targets --features linalg,parallel -D warnings`
  zero warnings; `cargo fmt --check` no drift.

### DIF-03 note — non-associative multiply is handled correctly

`project` computes `((data-mean)*rotation)*weights` (left-assoc) while `project_scores_generic`
computes `centered * (rotation*weights)`. f64 multiply is non-associative, so results may differ by
~1 ULP. The parity test correctly uses `≤1e-12` (NOT `assert_eq!` bit-identity) for this path,
which is the right tolerance choice. The `rotation[(j,k)]*weights[j]` fold also makes the analytic
gradient constant exact, so the ≤1e-12 analytic check (`rotation*weights`) is legitimately the
strongest possible check for this linear map — and it is non-circular.

---

## Low

### LOW-01: SRSF `sqrt(|f'|)` yields a NaN tangent at derivative zeros (documented, accepted)

**File:** `fdars-core/src/alignment/differentiable.rs:84`
**Issue:** `S::signum(deriv) * S::sqrt(S::abs(deriv))` produces a NaN tangent when `deriv == 0`
(`sqrt(0)` tangent = `0 / (2*0)`). This is an inherent SRSF domain singularity, is documented in
the module header (lines 17-23) and function docs, is covered by threat model T-76-01 (accepted),
and tests avoid it. Not introduced by this work; identical spirit to the existing f64 SRSF code.
**Fix:** None required for this phase. If a future ergonomic API (Phase 77) exposes this to less
careful callers, consider an optional epsilon-guard, e.g. treat `|deriv| < eps` as a zero-tangent
term, or document a required minimum-derivative precondition at the public boundary.

### LOW-02: `generic_l2_srsf_distance` has a NaN tangent when the distance is exactly zero

**File:** `fdars-core/src/alignment/differentiable.rs:99` (and `:84` transitively)
**Issue:** `S::sqrt(dist_sq)` yields a NaN tangent if `dist_sq == 0` (i.e. `curve2` aligned SRSF
exactly equals `q1_ref`). This is the standard non-differentiability of Euclidean norm at the
origin and only occurs at the exact minimizer, which is an unusual gradient-evaluation point. Same
class as LOW-01 (documented sqrt-singularity behavior). Not a blocker.
**Fix:** None required. If exposed publicly in Phase 77, document that the gradient is undefined at
distance 0, or optionally return a zero gradient there.

## Info

### IN-01: File the pre-existing `soft_dtw_backward` endpoint bug as a tracked backlog item

**File:** `fdars-core/src/metric/soft_dtw.rs:262-306` (bug site); `:454-521` (test doc-comment noting it)
**Issue:** The confirmed pre-existing all-zero-gradient bug in `soft_dtw_backward` is noted only in
a test doc-comment and (per that comment) the DIF-02 SUMMARY. It silently degrades
`soft_dtw_barycenter` to a near no-op (the barycenter barely moves from its mean initialization).
**Fix:** Add `if i == n && j == m { continue; }` at the top of the inner loop in
`soft_dtw_backward` (mirroring the corrected in-test oracle) under a dedicated bug-fix phase, with a
regression test asserting `soft_dtw_barycenter` measurably improves the soft-DTW objective vs the
mean init. Confirm this is filed in the backlog (not only in a test comment), since the SUMMARY is
archived-per-phase and easy to lose.

---

_Reviewed: 2026-09-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
