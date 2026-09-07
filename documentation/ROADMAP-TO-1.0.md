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

These are the naming/visibility items surfaced by the Phase 81 API audit and **deferred** (not
forced into Phases 82/83) to the 1.0 gap.

- [ ] **`AUD-09` + `AUD-13` — the `wire` module.** `pub mod wire` is a public-but-unwired
  JS/R interchange seam (24 public types, 0 re-exports). Before 1.0, decide to either **wire it
  up** as a supported interchange API or **seal it** `pub(crate)`. `AUD-13` (`#[non_exhaustive]`
  on the `wire` layer structs) is moot while `wire` stays public and is resolved by whichever
  path is chosen. *Scope: 1.0 milestone (API).*
- [ ] **`AUD-12` — config-struct `#[non_exhaustive]` + construction path.** Add
  `#[non_exhaustive]` to the ~22 public config structs, **paired with** a builder or
  `Default` + `..Default::default()` construction escape hatch so sealing them does not break
  external `Config { .. }` literals. This is feature work, not mechanical cleanup. *Scope: 1.0
  milestone (API).*
- [ ] **`AUD-19` — `geometric_median` naming.** Consolidate the lone `_1d`/`_2d` suffix forms
  onto a `Dim`-dispatched signature. *Scope: 1.0 milestone (optional naming).*
- [ ] **`AUD-20` — `hausdorff` naming.** Same `Dim`-dispatch consolidation for the
  `hausdorff_*` family. *Scope: 1.0 milestone (optional naming).*
- [ ] **`AUD-21` — `functional_spatial` naming.** Consolidate
  `functional_spatial_*` / `kernel_functional_spatial_*` onto `Dim` dispatch. *Scope: 1.0
  milestone (optional naming).*
- [ ] **`AUD-22` — the large lone-`_1d`/`_2d` suffix batch.** The biggest and highest-risk
  naming edit: the remaining lone-suffix functions across the crate **and all 28 examples**.
  *Scope: 1.0 milestone (optional naming — sequence carefully).*
- [ ] **`AUD-23` — `LpeerResult` → `LocalPeerResult`.** Rename the result type for clarity and
  to match the `PeerResult` sibling. *Scope: 1.0 milestone (optional naming).*

---

## Test &amp; quality debt

- [ ] **`golden`-test flake (co_cluster / svd_sign).** `golden_co_cluster_parallel`,
  `golden_co_cluster_below_threshold` (`equivalence_phase48`), and
  `svd_sign_fpca_two_matrix_bit_identical` (`equivalence_phase49`) **pass per-binary but flake
  under full parallel `cargo test`** (env/BLAS/disk-pressure dependent). Make them deterministic
  before 1.0 — either switch from bit-identity to a tolerance comparison, or serialize the
  affected tests. *Scope: 1.0 milestone (quality).*

---

## Pre-existing backlog

- [ ] **`SDTW-O1` — `soft_dtw_barycenter` optimizer replacement.** Replace the current
  MM-step barycenter descent with a proper global optimizer. **Note:** the `soft_dtw_backward`
  zero-gradient bug itself was **already FIXED in v0.40.0 (CORR-01)** — this remaining item is
  the optimizer-quality improvement, not the gradient correctness fix. *Scope: 1.0 milestone
  (algorithm quality) or a dedicated backlog milestone.*
- [ ] **`DIF-F1` — reverse-mode / VJP.** Add reverse-mode (vector-Jacobian-product)
  differentiation to the differentiable core. *Scope: differentiable-core expansion.*
- [ ] **`DIF-F2` — broaden the differentiable subset.** Extend the set of algorithms that are
  differentiable. *Scope: differentiable-core expansion.*
- [ ] **`DIF-F3` — generic `f64` hot-path signatures.** Generalize hot-path signatures over a
  scalar type so autodiff types flow through. *Scope: differentiable-core expansion.*

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
