# Phase 78: Gradient Correctness — soft_dtw Fix & Backward-Pass Audit - Context

**Gathered:** 2026-09-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Every hand-written backward/gradient pass in the crate produces a correct (non-zero, boundary-seeded) gradient. Concretely: (1) fix the acknowledged `soft_dtw_backward` endpoint-seed bug so it stops zeroing the whole gradient, with a regression test proving a non-zero gradient AND real barycenter movement; (2) tighten the existing `test_soft_dtw_barycenter_*` tests so they cannot pass on an all-zero gradient; (3) audit the ~10 sibling hand-written backward/gradient passes named in CORR-02, dispositioning each "clean" or "fixed". Behavior-preserving everywhere except the intended `soft_dtw` gradient correction.

This is NOT a redesign of the barycenter optimizer or any other algorithm — scope is the endpoint-seed defect + the audit sweep.

</domain>

<decisions>
## Implementation Decisions

### Fix Scope & Audit Rigor
- **CORR-02 sweep findings:** Fix small, localized, behavior-preserving bugs in-phase (analogous to the endpoint-seed one-liner). Defer anything requiring a redesign to the backlog with a logged item — do not expand this phase into an algorithm rewrite.
- **Audit artifact:** Record the CORR-02 clean/fixed disposition table in a dedicated `78-AUDIT.md` in the phase directory, referenced from the phase SUMMARY, so the audit trail survives milestone archival.
- **soft_dtw test assertions (SC #1):** Assert all three of — (a) non-zero backward `E` / accumulated gradient on non-identical input; (b) `soft_dtw_barycenter` on non-identical curves lands measurably far (clear L2 margin) from the pointwise mean; (c) the shipped gradient matches the `corrected_oracle_gradient` / v0.39.0 `Dual` path within a tight relative tolerance (~1e-6).
- **`corrected_oracle_gradient` helper:** Keep it as the independent SC#1 cross-check reference. Once the shipped code is fixed (shipped == oracle), update its doc comment so it no longer describes a live production bug.

### Claude's Discretion
- Exact tolerance/margin constants, test data (non-identical curve construction), and per-sibling audit rationale wording are at Claude's discretion, guided by the reference oracle and existing test conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/metric/soft_dtw.rs` — `soft_dtw_forward` (R table), `soft_dtw_backward` (E matrix, the buggy pass at ~line 262), `soft_dtw_accumulate_gradient`, `soft_dtw_barycenter`, and the `corrected_oracle_gradient` test helper (~line 467) that already prototypes the fix.
- The v0.39.0 `Dual` forward-mode AD path (`soft_dtw_distance_generic` with `Dual::seed`) — independent gradient reference already wired into the test module.

### Established Patterns
- The bug: the reverse double-loop in `soft_dtw_backward` visits the endpoint `(n, m)` and overwrites the `E[n][m] = 1.0` seed with `a + b + c = 0` (all neighbour contributions gated off at the endpoint), zeroing the entire backward pass. The shipped exponent formula is algebraically equal to the oracle's (`r[i][j] - r[i+1][j] + r[i+1][j]` collapses to `r[i][j]`), so the ONLY correction needed is skipping the endpoint write: `if i == n && j == m { continue; }`.
- Conditional serde derive pattern, column-major `FdMatrix`, `Result<T, FdarError>` returns — all carried conventions.

### Integration Points
- CORR-02 sweep targets: `alignment/differentiable`, `autodiff`, `boosting_regression/gamlss`, `elastic_regression/logistic`, `explain_generic/counterfactual`, `regression`, `seasonal/mod`, `smooth_basis`, `metric/soft_dtw`.
- Whole-crate gates: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`.

</code_context>

<specifics>
## Specific Ideas

- The fix is already prototyped and validated in `soft_dtw.rs`'s `corrected_oracle_gradient` test — porting that endpoint-skip into the shipped `soft_dtw_backward` is the concrete CORR-01 change.
- Historical build hazards apply (MEMORY.md): clippy with `--all-targets --features linalg,parallel`, `cargo fmt` per commit, watch `/tmp`/`target/` disk pressure, prefer inline execution + `commit --no-verify` after out-of-band gates if long builds stall subagents.

</specifics>

<deferred>
## Deferred Ideas

- Any CORR-02 sibling bug that would require an algorithm redesign (not a localized boundary-seed fix) → logged to backlog, not fixed in this phase.

</deferred>
