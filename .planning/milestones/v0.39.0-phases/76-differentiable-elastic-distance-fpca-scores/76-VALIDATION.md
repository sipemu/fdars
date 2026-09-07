---
phase: "76"
slug: "differentiable-elastic-distance-fpca-scores"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: validated
nyquist_compliant: true
wave_0_complete: false
created: "2026-09-06"
---

# Phase 76 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (inline `#[cfg(test)] mod tests`) |
| **Config file** | none |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel differentiable soft_dtw fpca` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–180 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run the quick command for the touched op.
- **After every plan wave:** Run the full suite.
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean + `cargo fmt --check` clean.
- **Max feedback latency:** ~180 seconds.

---

## Per-Task Verification Map

| Task | Plan | Req | Test Type | Automated Command | Status |
|------|------|-----|-----------|-------------------|--------|
| soft-DTW generic + oracle grad | DIF-02 | DIF-02 | unit | `cargo test ... soft_dtw` | ⬜ pending |
| amplitude-at-warp generic | DIF-02 | DIF-02 | unit | `cargo test ... differentiable` | ⬜ pending |
| FPCA score projection generic | DIF-03 | DIF-03 | unit | `cargo test ... fpca` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red*

### Validation architecture (from 76-RESEARCH.md § Validation Architecture)

**DIF-02 — elastic distance:**
1. **Oracle:** `soft_dtw_distance_generic<Dual>` seeded per input → gradient vector; compare to the existing hand-written `soft_dtw_accumulate_gradient` oracle (tight tolerance, e.g. ≤1e-9).
2. **Finite-difference:** dual gradient vs central FD `(f(x+h)−f(x−h))/2h`, h≈1e-6, ≤1e-6.
3. **f64 parity:** `soft_dtw_distance` (existing) vs the f64 instantiation of the generic core — ideally bit-identical, else ≤1e-12; same for `amplitude_distance_at_warp`.

**DIF-03 — FPCA scores:**
1. **Analytic gradient:** score_k = Σ_j (curve[j] − mean[j])·rotation[j,k]·weights[j]; ∂score_k/∂curve[j] = rotation[j,k]·weights[j] — assert the `Dual` gradient equals this closed form to ≤1e-12 (stronger than FD).
2. **Finite-difference:** cross-check ≤1e-6.
3. **f64 parity:** generic projection at f64 reproduces `FpcaResult::project` scores within tolerance.

**Test-curve construction (project-memory pitfall — avoid low-rank/degenerate data):**
- Use random combinations of `sin(πt)`, `cos(2πt)`, `sin(3πt)` with distinct coefficients (spanning, full-rank).
- SRSF/amplitude tests must avoid zero-derivative points (do not sample where the curve derivative is 0 → SRSF `sqrt(|ẋ|)` non-smooth).

---

## Wave 0 Requirements

- [ ] Inline `#[cfg(test)] mod tests` in the new/modified modules — oracle + FD + f64-parity tests per op.
- [ ] No framework install (Rust built-in harness).

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
