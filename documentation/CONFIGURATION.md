<!-- generated-by: gsd-doc-writer -->
# Configuration

`fdars-core` is a pure-Rust library. It has **no runtime configuration** — no environment
variables, no `.env` files, no config files, and no configuration structs that survive outside
a single process invocation. All configuration is compile-time, expressed entirely through
**Cargo feature flags** and the toolchain version that satisfies the crate's MSRV constraints.

---

## No Runtime Environment Variables

The crate does not read `process::env` or any external config at runtime. The only secrets
used in this repository are CI secrets (`CODECOV_TOKEN`, `CARGO_REGISTRY_TOKEN`), which live
in GitHub Actions and are never present in the library binary.

---

## Cargo Feature Flags

Defined in `fdars-core/Cargo.toml` under `[features]`:

| Feature | Default | Enables | MSRV | WASM-compatible |
|---------|---------|---------|------|-----------------|
| `parallel` | **yes** | `rayon` 1.10 — multi-threaded iterators via five `maybe_parallel!` macros | 1.81 | yes |
| `linalg` | no | `faer` 0.23 + `anofox-regression` 0.4 — Cholesky, ridge regression (`ridge_regression_fit`) | **1.84** | **no** |
| `serde` | no | `serde` 1 (derive) + `serde_json` 1 — `Serialize`/`Deserialize` on 157+ types | 1.81 | yes |
| `js` | no | `getrandom` 0.2 with `js` feature — secure RNG seeding in WASM via JavaScript | 1.81 | yes (WASM only) |
| `dhat-heap` | no | Allocation-profiling gate for dev integration tests; activates `#[global_allocator]` in test harness | 1.81 | no |

### Default feature set

```toml
default = ["parallel"]
```

A plain `cargo build -p fdars-core` (or `cargo add fdars-core`) enables rayon parallelism and
nothing else. The R package (`fdars-r`) always builds with `default-features = false` to avoid
pulling in rayon on CRAN Windows.

---

## Feature Details

### `parallel` — rayon-based multi-threading

- **Dependency added:** `rayon = "1.10"` (optional)
- **Effect:** Five macros in `src/parallel.rs` (`iter_maybe_parallel!`, `slice_maybe_parallel!`,
  `slice_maybe_parallel_mut!`, `maybe_par_chunks_mut!`, `maybe_par_chunks_mut_enumerate!`) compile
  to rayon parallel iterators. Without the feature, they compile to sequential iterators.
- **Affected modules:** `alignment`, `basis`, `classification`, `clustering`, `concurrent_regression`,
  `detrend`, `distance`, `famm`, `function_on_scalar`, `metric`, `outliers`, `seasonal`, `spm`,
  `streaming_depth`, `tolerance`, `utility`.
- **Thread pool:** Default rayon global thread pool (one thread per logical CPU). No project-level
  override is provided; callers may set the pool size via `rayon::ThreadPoolBuilder` before calling
  any fdars function.
- **Seeding in parallel loops:** Per-thread RNG is seeded as `StdRng::seed_from_u64(seed + k as u64)`
  for reproducibility across thread counts.

```bash
# Enable parallelism (also the default):
cargo build -p fdars-core --features parallel

# Disable parallelism (sequential mode — matches R package build):
cargo build -p fdars-core --no-default-features
```

### `linalg` — advanced linear algebra (Cholesky, ridge regression)

- **Dependencies added:** `faer = "0.23"` (optional), `anofox-regression = "0.4"` (optional)
- **MSRV raised to 1.84:** `faer` 0.23 requires Rust 1.84.0. The global MSRV in `Cargo.toml` is
  1.81 for CRAN Windows compatibility; `linalg` is intentionally excluded from the default set.
- **Public API gated:** `ridge_regression_fit` + `RidgeResult` (re-exported from `src/lib.rs` at
  line 585 under `#[cfg(feature = "linalg")]`). Basic OLS and Cholesky are always available via
  `src/linalg.rs` without this feature.
- **Not WASM-compatible:** `faer` uses native CPU intrinsics and cannot compile for `wasm32-unknown-unknown`.

```bash
# Enable linalg (requires Rust >= 1.84):
cargo build -p fdars-core --features linalg,parallel

# Combined with serde:
cargo build -p fdars-core --features linalg,parallel,serde

# Run tests with linalg (CI default):
cargo test -p fdars-core --features linalg,parallel,serde
```

### `serde` — serialization support

- **Dependencies added:** `serde = { version = "1", features = ["derive"] }` (optional),
  `serde_json = "1"` (optional)
- **Effect:** `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
  is applied to 157+ types across the codebase, including `FdMatrix`, `FpcaResult`, `SpmChart`,
  config structs (`GmmClusterConfig`, `StlConfig`, `ConformalConfig`, `ElasticConfig`, etc.),
  and result structs (`FregreLmResult`, `FunctionalLogisticResult`, `ClassifFit`, `RidgeResult`,
  and many others).
- **`ExplainLayer.extra`:** Gains a `serde_json::Value` field for untyped JSON payloads when
  the feature is enabled.
- **Use case:** Persistent FDA pipelines that need to serialize fitted models to JSON or any
  serde-compatible format (MessagePack, CBOR, etc.).
- Note: `serde` and `serde_json` are always available in `[dev-dependencies]`, so tests can
  exercise serialization without enabling the feature in the library build.

```bash
cargo build -p fdars-core --features serde
cargo build -p fdars-core --features linalg,parallel,serde
```

### `js` — WASM RNG seeding via JavaScript

- **Dependency:** `getrandom = { version = "0.2", features = ["js"] }` (made optional via the
  `js` feature; already added unconditionally for `wasm32` targets via the
  `[target.'cfg(target_arch = "wasm32")'.dependencies]` table).
- **Effect:** Enables `getrandom/js` so that secure random number seeding works inside a browser
  or Node.js WASM runtime. Without `js`, `getrandom` will fail at runtime in a WASM context.
- **When to use:** Any WASM build that uses randomized algorithms (GMM, bootstrap, elastic alignment,
  random-projection depth, etc.).

```bash
# Standard WASM build without secure RNG:
cargo build -p fdars-core --target wasm32-unknown-unknown --no-default-features

# WASM build with JS-backed secure RNG:
cargo build -p fdars-core --target wasm32-unknown-unknown --no-default-features --features js
```

### `dhat-heap` — allocation profiling (dev-only)

- **Dependency:** `dhat = "0.3"` in `[dev-dependencies]`
- **Effect:** Activates `#[global_allocator]` in two integration test files
  (`tests/alloc_audit_fpca.rs`, `tests/alloc_audit_dpca.rs`). Enables heap profiling of FPCA
  and DPCA allocation paths.
- **Warning:** Never enable in release builds or standard CI. Running multiple dhat tests in
  parallel panics with "A profiler already exists" because only one `#[global_allocator]` may
  be active at a time. The CI workflow explicitly excludes this feature from all standard jobs.
- **Manual usage:**

```bash
# Requires a writable TMPDIR (see project memory note on /tmp exhaustion):
TMPDIR=~/.cache/fdars-bench-tmp cargo test -p fdars-core \
  --features dhat-heap,linalg -- --test-threads=1
```

---

## Build Configurations Reference

| Use case | Command |
|----------|---------|
| Default (parallel, no linalg) | `cargo build -p fdars-core` |
| Sequential / R-package mode | `cargo build -p fdars-core --no-default-features` |
| Full featured (parallel + linalg + serde) | `cargo build -p fdars-core --features linalg,parallel,serde` |
| WASM — minimal | `cargo build -p fdars-core --target wasm32-unknown-unknown --no-default-features` |
| WASM — with JS RNG | `cargo build -p fdars-core --target wasm32-unknown-unknown --no-default-features --features js` |
| Docs (all features rendered) | `cargo doc -p fdars-core --no-deps --features linalg,parallel,serde` |
| Clippy (CI standard) | `cargo clippy -p fdars-core --all-targets --features linalg,parallel,serde -- -D warnings` |
| Tests (CI standard) | `cargo test -p fdars-core --features linalg,parallel,serde` |
| Tests (sequential CI) | `cargo test -p fdars-core --no-default-features --features linalg` |
| Allocation profiling | `cargo test -p fdars-core --features dhat-heap,linalg -- --test-threads=1` |

---

## Toolchain Requirements

| Requirement | Value | Source |
|-------------|-------|--------|
| MSRV (all features except `linalg`) | `1.81.0` | `fdars-core/Cargo.toml` `rust-version` |
| MSRV with `linalg` feature | `1.84.0` | `faer` 0.23 constraint, documented in `[features]` comment |
| Development runtime | `1.97.0` (stable) | Project context |
| WASM target | `wasm32-unknown-unknown` | `fdars-core/Cargo.toml` target table |
| CRAN Windows constraint | 1.81.0 — reason `linalg` is not in default | `Cargo.toml` comment line 14 |

No `rust-toolchain.toml` file is present in the repository; toolchain selection is left to the
calling environment.

---

## CI and Coverage Configuration

### Continuous Integration (`.github/workflows/rust-ci.yml`)

The `rust-ci.yml` workflow runs on push/PR to `main` for paths under `fdars-core/` and on
GitHub release publication.

| Job | Rust toolchains | Feature set | Notes |
|-----|----------------|-------------|-------|
| `test` | stable, beta, nightly | `linalg,parallel,serde` | Primary matrix |
| `test` (sequential) | stable, beta, nightly | `--no-default-features --features linalg` | Verifies non-parallel path |
| `clippy` | stable | `linalg,parallel,serde` | `--all-targets -D warnings` |
| `fmt` | stable | (all) | `cargo fmt --check` |
| `docs` | stable | `linalg,parallel,serde` | `RUSTDOCFLAGS=-Dwarnings` |
| `wasm` | stable | `--no-default-features` then `--features js` | Target: `wasm32-unknown-unknown` |
| `coverage` | stable | `linalg,parallel,serde` | `cargo-llvm-cov` → Codecov |
| `publish` | stable | (default) | Runs only on GitHub release events, after all other jobs pass |

Features excluded from all CI jobs: `dhat-heap` (parallel-profiler panic), `js` (WASM-only,
covered by dedicated `wasm` job).

### Publishing (`.github/workflows/release.yml`)

A separate `release.yml` workflow publishes to crates.io on any `v*` tag push using
`CARGO_REGISTRY_TOKEN`. It does **not** run tests — it relies on CI having already passed on
the commit being tagged. Audit-only milestones that make no code changes must not create a
`v*` tag, as doing so triggers an unnecessary publish.

### Coverage (`codecov.yml`)

Coverage is collected via `cargo-llvm-cov` and uploaded to Codecov with the `rust` flag.

| Target | Threshold |
|--------|-----------|
| Project (overall) | 70% |
| Patch (new code per PR) | 50% |

The `r` flag is also configured (`carryforward: true`) for the separate `fdars-r` R bindings
package, which is not part of this workspace.
<!-- VERIFY: Codecov dashboard URL and team/org project link -->
