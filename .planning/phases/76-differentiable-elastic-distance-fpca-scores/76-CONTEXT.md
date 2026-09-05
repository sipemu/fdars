# Phase 76: Differentiable Elastic Distance & FPCA Scores - Context

**Gathered:** 2026-09-06
**Status:** Ready for planning
**Mode:** Auto-generated (internal numeric-API phase — no user-facing product grey areas; making existing f64 ops generic-over-`Scalar`)

<domain>
## Phase Boundary

Make TWO scoped FDA operations generic over the Phase 75 `Scalar` substrate so that, instantiated at `Dual`, they yield exact forward-mode gradients w.r.t. a curve's input values, and instantiated at `f64` they reproduce the existing numerics:

1. **DIF-02 — elastic distance:** the soft-DTW / amplitude+phase elastic distance path.
2. **DIF-03 — FPCA score projection:** projecting a curve onto trained FPCA loadings to get FPC scores.

This phase consumes Phase 75's `autodiff::{Scalar, Dual}` substrate (committed, all gates green). It does NOT add the public gradient API / composition example / prelude surfacing — that is Phase 77 (DIF-04). It only makes these two ops generic + validates gradients.

</domain>

<decisions>
## Implementation Decisions

### Locked at milestone questioning (REQUIREMENTS.md "Milestone Design Decisions")
- **Generic-over-`Scalar`, gradients compose.** Write the op body once over `S: Scalar`; instantiate at `f64` and `Dual`.
- **Forward-mode only.**
- **Additive/non-breaking.** The existing f64 functions (`soft_dtw_distance`, `elastic_distance`, `amplitude_distance`, `phase_distance_pair`, and the FPCA scoring path in `regression.rs`) MUST stay byte-for-byte unchanged in signature and behavior. Add the generic version ALONGSIDE (e.g. a `*_generic<S: Scalar>` companion, or a private generic core that the existing f64 fn delegates to — delegation is allowed ONLY if the f64 output is provably identical within 1e-12 and no public signature changes). Protects R + WASM bindings + 28 examples.
- **No new crate dependency.**

### Claude's Discretion (resolve at plan/research time)
- **Container for generic curves:** the target f64 fns already take `&[f64]` slices — so the generic versions take `&[S]` slices (NOT a generic `FdMatrix<S>`; `FdMatrix` stays `Vec<f64>`, unchanged). `argvals`, `gamma`, `lambda`, FPCA `rotation`/`mean`/integration `weights` stay `f64` — only the curve whose gradient we want goes generic (`&[S]`). Lift f64 constants into the scalar via `S::from_f64(...)`.
- **Which distance(s) to genericize (DIF-02):** at minimum `soft_dtw_distance` (the pilot with a known hand-written gradient oracle) AND the amplitude/elastic distance (`elastic_distance` / `amplitude_distance` in `alignment/pairwise.rs`, SRSF-based). Research must confirm the SRSF `q`-transform and the DP/warping internals are differentiable under `Dual` (SRSF uses `sqrt(|ẋ|)·sign` — differentiable except at ẋ=0; the value-only `PartialOrd` from Phase 75 handles the DP min-branching). If closed-curve / nd / banded variants are hard, they are OUT OF SCOPE for this phase — the scalar + amplitude/phase pair on the 1-D open path is the DIF-02 target.
- **FPCA score entry (DIF-03):** the projection that maps a (centered) curve onto trained loadings → scores. Research must pin the exact function (`regression.rs` FPCA scoring — `fdata_to_pc_1d` computes the FPCA; the *projection of a curve onto existing loadings* is the differentiable target). Rotation/mean/weights stay f64; input curve `&[S]`; output scores `Vec<S>`. The op uses only `sub + mul + AddAssign + zero` (per Phase-75 research) — the simplest generic path.
- **Gradient direction:** w.r.t. the input curve's sample values (a `&[S]` of length m). Forward-mode: seed one input's tangent at a time (or provide a helper that builds the length-m gradient by m seeded evaluations — a full Jacobian helper is Phase 77; here a single directional/seeded-gradient that the tests drive is sufficient).

</decisions>

<code_context>
## Existing Code Insights

### Generic targets (all already take `&[f64]` slices — clean to genericize)
- `fdars-core/src/metric/soft_dtw.rs:49` — `soft_dtw_distance(x: &[f64], y: &[f64], gamma: f64) -> f64`. **The pilot** — it has a hand-written gradient elsewhere in the module to validate the `Dual` gradient against (SC #1).
- `fdars-core/src/alignment/pairwise.rs:103` — `elastic_distance(f1, f2, argvals, lambda)`; `:384` — `amplitude_distance(...)`; `:389` — `phase_distance_pair(...)`. SRSF-based amplitude/phase elastic distances.
- `fdars-core/src/warping.rs` — SRSF `q`-transform + `phase_distance`; the SRSF conversion is the differentiable core the amplitude distance builds on.
- FPCA scoring: `fdars-core/src/regression.rs` (`fdata_to_pc_1d`, `FpcaResult` — mean/rotation/weights). The projection-onto-loadings is the DIF-03 target.

### Substrate (Phase 75, committed)
- `fdars-core/src/autodiff.rs` — `Scalar` trait (±,×,÷,neg,sqrt,exp,ln,sin,cos,powf,abs,signum,zero,one,from_f64,infinity, value-only PartialEq/PartialOrd, AddAssign/SubAssign/MulAssign), concrete `Dual{value,tangent}`, `seed`/`constant`/`extract`/`diff`. `impl Scalar for f64` is a zero-cost passthrough.

### Established patterns
- Column-major `FdMatrix` = `Vec<f64>` (`src/matrix.rs`) — do NOT genericize it. Generic entry points take `&[S]` slices.
- `helpers::simpsons_weights` (f64 integration weights — stay f64).
- All public fns return `Result<T, FdarError>` where validation applies; pure numeric kernels may return the value directly (match the existing fn's shape).

### Integration points
- New generic fns registered in `src/lib.rs` as needed; the FULL crate-root + prelude surface is Phase 77. Keep re-exports minimal here.

</code_context>

<specifics>
## Specific Ideas

The make-or-break gates (SC-driven, known-answer style):
- **DIF-02 SC #1:** `Dual` gradient of the elastic distance == central finite differences (≤1e-6) AND == the existing hand-written `soft_dtw` gradient (tight tolerance) at test points.
- **DIF-02 SC #2:** f64 instantiation reproduces `elastic_distance`/`amplitude_distance` to ≤1e-12 (ideally bit-identical if delegating).
- **DIF-03 SC #3:** `Dual` gradient of FPC scores == central finite differences (≤1e-6).
- **DIF-03 SC #4:** f64 instantiation reproduces existing FPCA scores within tolerance.
- **SC #5:** both generic paths live ALONGSIDE the existing f64 fns; no new dependency; whole crate green.

Watch SRSF non-smoothness at ẋ=0 (document/avoid in test data, as Phase 75 did for sqrt(0)). Use spanning, well-conditioned test curves (project memory: β-recovery / gradient tests silently pass on low-rank data — use non-degenerate curves).

</specifics>

<deferred>
## Deferred Ideas

- Closed-curve / n-D / banded elastic-distance generic variants — out of scope; 1-D open path only this phase.
- Full length-m Jacobian / ergonomic public gradient API + composition example — Phase 77 (DIF-04).
- Reverse-mode — DIF-F1.
- Making the existing f64 public signatures themselves generic (vs additive-alongside) — DIF-F3.

</deferred>
