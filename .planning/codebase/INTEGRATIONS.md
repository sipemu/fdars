# External Integrations

**Analysis Date:** 2026-08-07

## APIs & External Services

**Package Registry:**
- Crates.io - Published library registry
  - SDK/Client: Cargo (built-in)
  - Auth: `CARGO_REGISTRY_TOKEN` (GitHub Actions secret)

**Documentation & Versioning:**
- GitHub (sipemu/fdars) - Source repository
  - VCS: Git
  - Registry metadata: homepage, documentation, repository URLs in `fdars-core/Cargo.toml`
  - docs.rs - Hosted Rust documentation
  - Release integration: Automatic publish via GitHub Actions on release event

**Code Coverage:**
- Codecov.io - Coverage tracking and reporting
  - CI Integration: `.github/workflows/rust-ci.yml` uploads lcov.info
  - Configuration: `codecov.yml` (70% project target, 50% patch threshold)
  - Auth: `CODECOV_TOKEN` (GitHub Actions secret)

**CI/CD Pipeline:**
- GitHub Actions - Automated testing and publishing
  - Workflows: `.github/workflows/rust-ci.yml`, `.github/workflows/release.yml`
  - No external build infrastructure dependency
  - Test matrix: stable, beta, nightly Rust toolchains

## Data Storage

**Databases:**
- None — pure computational library with no persistent data backend

**File Storage:**
- Local filesystem only — data passed via `FdMatrix` (in-memory Vec<f64>)
- No cloud storage integrations (AWS S3, Google Cloud Storage, etc.)

**Caching:**
- In-memory only via Rust data structures
- No Redis, memcached, or other caching layer

## Authentication & Identity

**Auth Provider:**
- None required for library users
- Internal: GitHub Actions uses token-based authentication for crates.io publishing and Codecov uploads

**Secrets Management:**
- GitHub Actions Secrets:
  - `CARGO_REGISTRY_TOKEN` - Publish to crates.io
  - `CODECOV_TOKEN` - Upload coverage reports
- No `.env` file or local secret management

## Monitoring & Observability

**Error Tracking:**
- None — All errors returned as `Result<T, FdarError>` (see `src/error.rs`)
- No external error reporting service (Sentry, etc.)

**Logs:**
- Standard Rust logging via `log` crate (0.4) — optional transitive dependency
- No centralized logging backend (ELK stack, CloudWatch, Datadog, etc.)
- GitHub Actions CI logs available in repository Actions tab

**Performance Profiling:**
- Criterion benchmarks generate HTML reports in `target/criterion/`
- No external APM (Application Performance Monitoring) integration
- Manual profiling via `cargo bench` locally

## CI/CD & Deployment

**Hosting:**
- Crates.io (Rust package registry)
- docs.rs (auto-generated documentation)
- GitHub (source control and releases)

**CI Pipeline:**
- GitHub Actions (`.github/workflows/rust-ci.yml`)
  - Test stage: `cargo test --all-features`, `cargo test --no-default-features --features linalg`
  - Lint stage: `cargo clippy` with custom allow-list
  - Format stage: `cargo fmt --check`
  - Documentation: `cargo doc --no-deps --all-features`
  - WASM compilation: `cargo build --target wasm32-unknown-unknown`
  - Coverage: `cargo llvm-cov --all-features --lcov`
- Publish stage: Triggered on GitHub release event, runs `cargo publish --token $CARGO_REGISTRY_TOKEN`
- No manual deployment required — Cargo publishes directly to crates.io

**Deployment Gates:**
- All CI stages (test, clippy, fmt, docs, wasm) must pass before publish
- Coverage and codecov status checks configured but not blocking

## Environment Configuration

**Required env vars:**
- `CARGO_REGISTRY_TOKEN` - Publish credentials (GitHub Actions only, not needed locally)
- `CODECOV_TOKEN` - Coverage uploader token (GitHub Actions only)
- No application-level env vars — all configuration is compile-time features or function parameters

**Secrets location:**
- GitHub Actions Secrets (via `.github/workflows/`)
- No local .env files or credential stores in repository

## Webhooks & Callbacks

**Incoming:**
- Codecov webhook posts coverage reports back to GitHub PR checks
- GitHub release webhook triggers publish workflow

**Outgoing:**
- crates.io API endpoint: `https://crates.io/api/v1/crates/new` (via `cargo publish`)
- Codecov API endpoint: `https://codecov.io/` (via codecov/codecov-action@v4)
- docs.rs: Automatically builds docs after crates.io publish (no explicit webhook)

## Cross-Crate Dependencies

**R Package Binding:**
- `fdars-r` (separate GitHub repository: sipemu/fdars-r)
  - Uses `fdars-core` as a dependency via `Cargo.toml`
  - Builds Rust bindings for R via extendr/rextendr
  - Not included in this workspace

**No Direct Integration with:**
- Databases (SQL, NoSQL)
- Web frameworks (actix, axum, rocket)
- Async runtimes (tokio, async-std)
- HTTP clients (reqwest, hyper)
- Message queues (kafka, rabbitmq)
- Cloud SDKs (AWS, GCP, Azure)
- ML frameworks (TensorFlow, PyTorch)

---

*Integration audit: 2026-08-07*
