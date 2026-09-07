<!-- generated-by: gsd-doc-writer -->
# Stability &amp; Semantic Versioning Policy

This document defines the API-stability contract and the Minimum Supported Rust Version
(MSRV) policy for `fdars-core`. It states what "stable" means for the crate, how breaking
changes are staged and released, and which Rust toolchains are supported.

`fdars-core` is currently a **0.x** crate. Under [Semantic Versioning](https://semver.org),
the 0.x series does not yet guarantee a stable public API — this policy describes the rules we
follow now (pre-1.0) and the stronger guarantees that take effect at the 1.0 cut. The concrete
work that remains before that cut is tracked in [ROADMAP-TO-1.0.md](./ROADMAP-TO-1.0.md).

---

## What "Stable" Means

The **public API** governed by semantic versioning is the set of items an external crate can
name and use when depending on `fdars-core`:

- **Re-exported items** — everything surfaced through `src/lib.rs` root re-exports and the
  `fdars_core::prelude` convenience module.
- **Public function signatures** — the name, parameter types/order, and return type of any
  `pub fn` reachable from the crate root or a public module.
- **Public types, enums, and fields** — `pub struct`/`pub enum` definitions, their public
  fields, their public variants, and their derived trait implementations.

The following are **explicitly outside** the stability guarantee:

- **`pub(crate)` items** — internal helpers such as `linalg`, `distributions`,
  `permutation_test`, and `test_helpers`. These are implementation details and may change at
  any time without a semver event.
- **The `wire` module** — `pub mod wire` is a deliberate but currently *unwired* interchange
  seam (intended for JS/R data exchange). It is public today but carries **no stability
  guarantee pre-1.0**. Before the 1.0 cut it will be either wired up as a supported API or
  sealed to `pub(crate)`; see `AUD-09`/`AUD-13` in
  [ROADMAP-TO-1.0.md](./ROADMAP-TO-1.0.md).

If an item is reachable from outside the crate but is not intended to be a supported API, it
does not become part of the stable surface merely by being reachable — the lists above are the
authoritative definition.

---

## Deprecation Process

Breaking changes are **staged, not abrupt**. When a public item must change or be removed:

1. **Deprecate first.** The old form is annotated `#[deprecated(since = "X.Y.Z", note = "…")]`
   with a note pointing at the replacement. It keeps working and forwards to the new form.
2. **Provide the replacement in the same release.** Callers can migrate immediately; the
   deprecated form emits a compiler warning but does not break builds.
3. **Remove later.** The deprecated form is deleted in a subsequent release, at which point the
   removal is the breaking change.

This is exactly the pattern used for the `Dim`-based consolidation (e.g. `mean_2d` →
`mean(…, Dim::Two)`), where each `_2d` form was marked `#[deprecated(since = "0.30.0", …)]`
before removal.

### Breaking changes under 0.x

While the crate is on 0.x, **breaking changes ship in MINOR version bumps** (`0.40.x` →
`0.41.0`), per the SemVer 0.x rule that the minor position acts as the "major" position before
1.0. The v0.41.0 milestone is the first breaking release after a long additive run: it removed
long-deprecated forms and tightened the public surface. Each such change is called out in
`CHANGELOG.md`.

---

## Post-1.0 Breaking-Change Policy

Once the crate reaches `1.0.0` (see [ROADMAP-TO-1.0.md](./ROADMAP-TO-1.0.md)), the following
constitute **breaking changes** and require a **major** version bump:

- **Signature changes** — altering a public function's parameters, their order/types, or its
  return type.
- **Visibility changes** — removing a public item or reducing its visibility (`pub` →
  `pub(crate)`), or removing a re-export.
- **Enum-variant changes** — removing or renaming a public variant, or (for enums *not* marked
  `#[non_exhaustive]`) adding a variant.
- **Struct-field changes** — removing, renaming, or changing the type of a public field, or
  (for structs *not* marked `#[non_exhaustive]`) adding a field.

### The role of `#[non_exhaustive]`

As of Phase 82, `#[non_exhaustive]` is applied across the crate's public enums and result
structs (baseline ~330 sites, plus the enums and result structs closed in API-03). This
attribute is what lets us **add** variants and fields *without* a breaking change: downstream
code cannot exhaustively match a `#[non_exhaustive]` enum or construct a `#[non_exhaustive]`
struct with a field literal, so future additions remain backward-compatible. Additive growth of
result types and enums is therefore a **minor** version event, not a major one.

Config structs are a known exception today: several `pub`-field config structs are *not* yet
`#[non_exhaustive]`, because sealing them would break external `Config { .. }` literals without a
builder/`Default` construction path. Closing that gap is scoped as `AUD-12` in
[ROADMAP-TO-1.0.md](./ROADMAP-TO-1.0.md).

### MSRV bumps

Raising the MSRV is treated as a **minor-version event** and is called out in `CHANGELOG.md`.
See [MSRV Policy](#msrv-policy) below.

---

## Stability Surface Conventions

The following crate-wide conventions are part of the stable surface and callers may rely on
them:

- **Derived traits.** All public types derive `Debug`, `Clone`, and `PartialEq` (consistent
  across 97+ types). Removing one of these derives from a public type is a breaking change.
- **`#[must_use]`.** Expensive computations (74+ functions) and their result types are marked
  `#[must_use]`; results are meant to be consumed, and this annotation is considered part of the
  documented contract.
- **Column-major `FdMatrix` invariant.** All functional data is stored column-major: element
  `(row, col)` lives at index `row + col * nrows`; rows are observations/curves and columns are
  evaluation points. Public APIs that accept or return `FdMatrix` rely on this layout, and it
  will not change without a major version bump post-1.0.
- **`Result`-based error flow.** Public functions return `Result<T, FdarError>` rather than
  panicking on invalid input. The `FdarError` variant set (`InvalidDimension`,
  `InvalidParameter`, `ComputationFailed`, `InvalidEnumValue`) is `#[non_exhaustive]`, so new
  variants may be added in a minor release.

---

## MSRV Policy

`fdars-core` maintains a **two-tier** Minimum Supported Rust Version:

| Tier | MSRV | Where it applies | Why |
|------|------|------------------|-----|
| **Crate (default features)** | **1.81** | The crate as built with `default = ["parallel"]` and all non-`linalg` features | Set for **CRAN Windows** compatibility — the R toolchain on CRAN's Windows builders uses Rust 1.81.0, and the `fdars-r` package builds `fdars-core` with `default-features = false`. |
| **`linalg` feature** | **1.84** | Only when the optional `linalg` feature is enabled | The `linalg` feature pulls in `faer 0.23+` (Cholesky, ridge regression via `anofox-regression`), and `faer 0.23` requires Rust **1.84.0**. `linalg` is therefore *not* in the default feature set. |

### Consistency with `Cargo.toml`

The crate MSRV documented here — **1.81** — matches `fdars-core/Cargo.toml`
(`rust-version = "1.81"`) exactly. That field is the single source of truth for the base MSRV
and this document must not drift from it.

The `linalg`-feature requirement of **1.84** is a *feature-conditional* MSRV. It is enforced by
the `faer 0.23` dependency (which is optional and behind the `linalg` feature) rather than by a
second `rust-version` field, so it is surfaced here and in the crate's inline documentation
rather than in `Cargo.toml`.

### Bump policy

- Raising either tier's MSRV is a **minor-version event** and is announced in `CHANGELOG.md`.
- The base crate MSRV (1.81) is held deliberately low for CRAN compatibility; it will not be
  raised casually.
- Because `linalg` is opt-in and non-default, downstream users who need the base 1.81 MSRV
  (including WASM and CRAN builds) are unaffected by the 1.84 `linalg` requirement.

---

## See Also

- [ROADMAP-TO-1.0.md](./ROADMAP-TO-1.0.md) — the concrete gap checklist that must clear before
  the `1.0.0` cut declares this policy's stronger post-1.0 guarantees in force.
- [DEVELOPMENT.md](./DEVELOPMENT.md) — prerequisites, toolchain matrix, and CI pipeline.
- [ARCHITECTURE.md](./ARCHITECTURE.md) — the module layout and public-surface structure that
  this policy governs.
- `CHANGELOG.md` (repository root) — the running record of releases, breaking changes, and MSRV
  bumps.
