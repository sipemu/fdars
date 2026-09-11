<!-- generated-by: gsd-doc-writer -->
# Roadmap to 1.0

This document enumerates everything that remains before `fdars-core` can be cut to a stable
`1.0.0`. It is a **gap checklist only** — it does not execute any item. Each entry is scoped for
a future 1.0 milestone; the governing stability contract is
[STABILITY.md](./STABILITY.md).

The items below were deferred out of the v0.41.0 API-stabilization milestone (see the Phase 81
audit inventory approval gate) or carried forward from the pre-existing backlog. Each item lists
a one-line description and the milestone/scope where it belongs.

---

## API surface — deferred audit items

**Update (v0.42.0):** all API-section items below (`AUD-09`/`AUD-12`/`AUD-13`/`AUD-19`–`AUD-23`) are **cleared** — sealed `wire`, `#[non_exhaustive]` configs, and the full `Dim`-dispatch naming unification shipped in v0.42.0.

**Update (v0.43.0):** the **Quality** blocker (the `golden`-test flake under `## Test & quality debt`) is **cleared** — root-caused as a deterministic faer-vs-nalgebra SVD backend divergence (not an environmental flake) and fixed with a test-side `cfg`-guard, audited suite-wide, and locked by a CI determinism guardrail (Phases 90–92, v0.43.0).

These are the naming/visibility items surfaced by the Phase 81 API audit and **deferred** (not
forced into Phases 82/83) to the 1.0 gap.

- [x] **`AUD-09` + `AUD-13` — the `wire` module.** `pub mod wire` is a public-but-unwired
  JS/R interchange seam (24 public types, 0 re-exports). Before 1.0, decide to either **wire it
  up** as a supported interchange API or **seal it** `pub(crate)`. `AUD-13` (`#[non_exhaustive]`
  on the `wire` layer structs) is moot while `wire` stays public and is resolved by whichever
  path is chosen. *Scope: 1.0 milestone (API).*
- [x] **`AUD-12` — config-struct `#[non_exhaustive]` + construction path.** Add
  `#[non_exhaustive]` to the ~22 public config structs, **paired with** a builder or
  `Default` + `..Default::default()` construction escape hatch so sealing them does not break
  external `Config { .. }` literals. This is feature work, not mechanical cleanup. *Scope: 1.0
  milestone (API).*
- [x] **`AUD-19` — `geometric_median` naming.** Consolidate the lone `_1d`/`_2d` suffix forms
  onto a `Dim`-dispatched signature. *Scope: 1.0 milestone (optional naming).*
- [x] **`AUD-20` — `hausdorff` naming.** Same `Dim`-dispatch consolidation for the
  `hausdorff_*` family. *Scope: 1.0 milestone (optional naming).*
- [x] **`AUD-21` — `functional_spatial` naming.** Consolidate
  `functional_spatial_*` / `kernel_functional_spatial_*` onto `Dim` dispatch. *Scope: 1.0
  milestone (optional naming).*
- [x] **`AUD-22` — the large lone-`_1d`/`_2d` suffix batch.** The biggest and highest-risk
  naming edit: the remaining lone-suffix functions across the crate **and all 28 examples**.
  *Scope: 1.0 milestone (optional naming — sequence carefully).*
- [x] **`AUD-23` — `LpeerResult` → `LocalPeerResult`.** Rename the result type for clarity and
  to match the `PeerResult` sibling. *Scope: 1.0 milestone (optional naming).*

---

## Test &amp; quality debt

- [x] **`golden`-test flake (co_cluster / svd_sign).** `golden_co_cluster_parallel`,
  `golden_co_cluster_below_threshold` (`equivalence_phase48`), and
  `svd_sign_fpca_two_matrix_bit_identical` (`equivalence_phase49`). **Root-caused in v0.43.0
  (Phase 90):** NOT an environmental flake — a *deterministic* feature-configuration artifact.
  The goldens were captured under the `faer` SVD backend (`--features linalg`); a build without
  `linalg` routes `fdata_to_pc` through the `nalgebra` backend, which diverges the FPCA rotation
  categorically (the "intermittent" symptom was just whether `--features linalg` was present).
  **Fixed** with a least-invasive test-side `#[cfg_attr(not(feature = "linalg"), ignore)]` guard
  on the three tests — NOT a tolerance comparison, NOT serialization, no `src/` change, no new
  dependency. A suite-wide robustness audit (Phase 91) found zero further fragile tests, and a CI
  `determinism-guardrail` job (Phase 92) locks the green baseline against regression.
  *Cleared: v0.43.0, Phases 90–92.*

---

## Pre-existing backlog

- [ ] **`SDTW-O1` — `soft_dtw_barycenter` optimizer replacement.** Replace the current
  MM-step barycenter descent with a proper global optimizer. **Note:** the `soft_dtw_backward`
  zero-gradient bug itself was **already FIXED in v0.40.0 (CORR-01)** — this remaining item is
  the optimizer-quality improvement, not the gradient correctness fix. *Scope: 1.0 milestone
  (algorithm quality) or a dedicated backlog milestone.*
- [x] **`DIF-F1` — reverse-mode / VJP.** Add reverse-mode (vector-Jacobian-product)
  differentiation to the differentiable core. *Scope: differentiable-core expansion.* — **Done v0.44.0** (in-crate Wengert-list tape + `Var`/`vjp`).
- [x] **`DIF-F2` — broaden the differentiable subset.** Extend the set of algorithms that are
  differentiable. *Scope: differentiable-core expansion.* — **Done v0.44.0** (basis eval, FPCR prediction, roughness penalty, functional depth — all generic over `Scalar`, FD-checked).
- [x] **`DIF-F3` — generic `f64` hot-path signatures.** Generalize hot-path signatures over a
  scalar type so autodiff types flow through. *Scope: differentiable-core expansion.* — **Done v0.44.0** (`l2_distance`/`trapz`/`inner_product`/`inner_product_l2` generic over `T: Scalar`, non-breaking).

---

## Ecosystem

- [ ] **`fdars-j75` — `fdars-r` `FdMatrix` migration.** Migrate the external R wrapper
  (`fdars-r`) to the current `FdMatrix` API. This is a **separate package**, out of
  `fdars-core` scope, but it is a 1.0-ecosystem gap that should be closed alongside the core
  1.0 cut. *Scope: ecosystem (external package).*

---

## The 1.0 cut

- [ ] **1.0-CUT — bump to `1.0.0` and declare the API stable.** Once every item above clears,
  bump the crate version to `1.0.0` and declare the public API stable under the post-1.0
  guarantees defined in [STABILITY.md](./STABILITY.md). This is the terminal item and must be
  done last. *Scope: 1.0 milestone (release).*

---

## See Also

- [STABILITY.md](./STABILITY.md) — the semver / API-stability policy and MSRV policy that the
  `1.0.0` cut brings fully into force.
- `CHANGELOG.md` (repository root) — running record of releases and breaking changes.

> This checklist enumerates deferred work only; it deliberately excludes items already resolved
> (for example, the serde/`ClassifFit` build breakage repaired in v0.40.0 is **green** and is
> not an open gap).
