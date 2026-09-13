# Requirements: fdars — v0.45.0 Soft-DTW Barycenter — Global Optimizer (SDTW-O1)

**Defined:** 2026-09-13
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps on the road to a deliberate 1.0 cut.

Milestone goal: replace the single-init majorization-minimization (MM / soft-DBA) descent in
`soft_dtw_barycenter` with a proper global optimizer — an in-crate L-BFGS local optimizer wrapped
in a deterministically-seeded multi-restart loop — closing the last in-core algorithm-quality item
(`SDTW-O1`) on `documentation/ROADMAP-TO-1.0.md`. Strictly additive/non-breaking; no new crate
dependency; ships as 0.45.0 (still 0.x).

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Optimizer

- [x] **SDTW-01**: `soft_dtw_barycenter` minimizes the soft-DTW barycenter objective with an
  in-crate L-BFGS optimizer (limited-memory two-loop recursion + backtracking/Armijo line
  search) driven by the exact soft-DTW gradient the crate already computes, replacing the
  fixed-step MM descent. No new crate dependency; compiles and runs in the default feature set
  and on `wasm32` (no `linalg`/`argmin` gating).
- [x] **SDTW-02**: A multi-restart wrapper runs the local optimizer from multiple initializations
  (pointwise mean + deterministically-seeded perturbations, seeded via the project's
  `StdRng::seed_from_u64(seed + k)` convention) and returns the barycenter with the lowest total
  soft-DTW cost across restarts; results are reproducible for a fixed seed.

### API

- [x] **SDTW-03**: A `SoftDtwBarycenterConfig` builder-convention config struct (fields incl.
  `n_restarts`, `max_iter`, `tol`, `gamma`, `seed`) plus a new config-taking entry point expose
  the new optimizer; the existing `soft_dtw_barycenter(data, gamma, max_iter, tol)` signature is
  preserved as a thin, behavior-compatible wrapper so no existing call site, prelude re-export,
  R/WASM binding, or example changes. New public items are additively re-exported (crate root +
  prelude) with a running doctest.

### Quality Evidence

- [ ] **SDTW-04**: A test (and/or example) demonstrates the new optimizer reaches a strictly lower
  total soft-DTW barycenter cost than the MM baseline on a case where MM stalls in a local
  optimum, and that the legacy-wrapper path preserves f64 behavior (converges to a bounded
  barycenter within the data's convex hull; no divergence).

### Release

- [ ] **REL-01**: Bump `fdars-core` 0.44.0 → 0.45.0 with a `[0.45.0]` entry in both CHANGELOGs
  (root + crate), check off `SDTW-O1` on `documentation/ROADMAP-TO-1.0.md`, and pass all
  whole-crate gates green: `cargo fmt --check`, `cargo clippy --all-targets --features
  linalg,parallel -- -D warnings`, full `cargo test`, a `--features serde` build guard, all 28
  examples + doctests, and `cargo package`. The `git tag v0.45.0` → crates.io publish is the
  DEFERRED operator step (GSD `git.create_tag` stays off — the tag auto-publishes via
  `release.yml`).

## Future Requirements

Deferred beyond this milestone. Tracked but not in the current roadmap.

### Ecosystem / Release

- **fdars-j75**: Migrate the external `fdars-r` R wrapper to the `FdMatrix` API. Separate package,
  out of `fdars-core` scope; a remaining 1.0-ecosystem gap on `ROADMAP-TO-1.0.md`.
- **1.0-CUT**: Bump to `1.0.0` and declare the public API stable under `STABILITY.md`. Terminal
  item — must be last, once `SDTW-O1` (this milestone) and `fdars-j75` clear.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| A new crate dependency (`argmin`, external autodiff, etc.) for the optimizer | `argmin` is only reachable transitively via `anofox-regression` behind the optional `linalg` feature; barycenter must keep working in default + WASM builds. Standing project convention: no new crate dependency (v0.44.0 hand-wrote the reverse-mode tape). L-BFGS is hand-rolled in-crate. |
| Differentiating the warp-searched `elastic_distance` (discrete DP argmin) | A discrete DP argmin has no meaningful gradient; deferred at v0.39.0 (DIF-02) and unchanged. Only the smooth soft-DTW objective is optimized here. |
| Changing the soft-DTW distance/gradient definitions or the `Var`/`Dual` autodiff core | The gradient is correct as of v0.40.0 (CORR-01); this milestone changes only the *optimizer* consuming it, not the objective or its derivatives. |
| Breaking the existing `soft_dtw_barycenter(data, gamma, max_iter, tol)` signature | Non-breaking is the central constraint — protects R + WASM bindings + 28 examples; the config API is additive alongside the preserved wrapper. |
| Parallelizing restarts as a hard requirement | Determinism + correctness are the goal; restart parallelism (feature-gated via `rayon`) is an optional implementation detail, not a scoped requirement. |

## Traceability

Which phases cover which requirements. Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SDTW-01 | Phase 101 | Complete |
| SDTW-02 | Phase 102 | Complete |
| SDTW-03 | Phase 103 | Complete |
| SDTW-04 | Phase 104 | Pending |
| REL-01  | Phase 105 | Pending |

**Coverage:**

- v1 requirements: 5 total
- Mapped to phases: 5 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-13*
*Last updated: 2026-09-13 after roadmap creation (5 phases, 101–105)*
</content>
