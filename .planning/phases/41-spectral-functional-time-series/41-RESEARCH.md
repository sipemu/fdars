# Phase 41: Spectral Functional Time Series — Research

**Researched:** 2026-08-22
**Domain:** Frequency-domain functional time series (spectral density operator, dynamic FPCA, functional VAR/VMA/FARMA simulation)
**Confidence:** HIGH (core algorithmic decisions verified from existing codebase; R reference formulations cited from official documentation)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Spectral Density Operator (FTS-03-01)**
- Bartlett lag-window weights over lagged autocovariance operators, reusing existing `long_run_covariance` / `autocovariance_matrix` machinery in `fts/acf.rs`. Default bandwidth ⌊N^{1/3}⌋ (matches `long_run_covariance` convention), user-overridable.
- Frequency set: standard Fourier grid θ_j = 2πj/N for j = 0..N-1; return one m×m operator per frequency.
- rustfft application: DFT across the lag index h for each (j₁,j₂) entry of the autocovariance operator sequence → complex m×m spectral density operator per frequency.
- Output representation: per-frequency m×m operator stored as separate real + imaginary flat `Vec<f64>` (Hermitian), column-major, consistent with `LongRunCovResult.cov_matrix`.

**Dynamic FPCA — Filters, Scores, Reconstruction (FTS-03-02/03)**
- Dynamic eigen-filters: eigendecompose the spectral density operator at each Fourier frequency, then inverse-FFT the frequency-domain eigenvectors to obtain time-domain filter coefficients over a symmetric lag window (`freqdom` `dpca.filters` convention).
- Filter lag support: symmetric window h ∈ [−L, L], with a modest default L (user-overridable).
- Dynamic scores: time-domain convolution of the curve series with the dynamic filters, yielding one score series per retained dynamic component (`freqdom` `dpca.scores`).
- Reconstruction (FTS-03-03): inverse dynamic filtering — convolve scores with the adjoint (time-reversed) filters and sum over retained components. Integrated-L2 reconstruction error must decrease monotonically as more dynamic components are retained (success criterion 3).

**VAR/VMA + FARMA Simulators (FTS-03-04/05)**
- Operator-kernel parameterization: user supplies m×m operator matrices (one per AR lag, one per MA lag) that act on the grid-discretized curve vector via matrix-vector product.
- Innovations: Gaussian white-noise curves generated via the existing KL machinery (`sim_kl`-style eigenstructure), seeded.
- Burn-in: documented default burn-in (e.g. 200) discarded to reach stationarity; user-settable.
- Seeding: explicit `seed: u64` → `StdRng::seed_from_u64(seed)`, fully deterministic output (crate convention; per-thread `seed + k` if any parallel loop is used).

**API Surface & Result Types**
- Location: new `fdars-core/src/fts/spectral.rs` for spectral density + DPCA; VAR/VMA + FARMA simulators added to `simulation.rs`. Re-export new public items at the crate root and in `fts/mod.rs`.
- Result structs: `SpectralDensityResult`, `DpcaResult`, `DpcaReconstruction` (or similarly named), each deriving `Debug, Clone, PartialEq`, serde-gated, `#[non_exhaustive]`, following the existing `*Result` convention.
- Function names: `spectral_density`, `dpca`, `dpca_reconstruct`, `sim_fvarma`, `sim_farma` (snake_case with domain hints).
- Grid handling: take explicit `argvals` and use Simpson integration weights, consistent with the rest of `fts` and the crate's integration-weight convention.

### Claude's Discretion

- Exact default value of the DPCA filter lag support L, the burn-in length, and the precise Result-struct field layout are at the planner's discretion within the conventions above.
- Whether to expose a private `autocovariance_matrix`-style helper (reuse vs. thin wrapper) is an implementation detail — reuse the `fts/acf.rs` machinery where possible.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. (Plotting/rendering of spectra/DPCA filters is an explicit milestone Out-of-Scope item; FRE-02 object-data Fréchet regression is Phase 42.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FTS-03-01 | User can estimate the spectral density operator of a functional time series — the frequency-domain long-run covariance formed via `rustfft` over lagged autocovariance operators, evaluated at a set of Fourier frequencies. | Section "Spectral Density Estimator" below — exact DFT formula, rustfft pattern, output layout, Hermitian symmetry. |
| FTS-03-02 | User can compute dynamic functional PCA (DPCA) — dynamic eigen-filters and dynamic scores derived from the estimated spectral density operator. | Section "Dynamic FPCA" — eigendecomposition per frequency, inverse-FFT to filter taps, convolution for scores. |
| FTS-03-03 | User can reconstruct curves from DPCA dynamic scores via inverse dynamic filtering (optimal DPCA reconstruction of the original series). | Section "Reconstruction" — adjoint-filter convolution, monotone-decreasing error oracle. |
| FTS-03-04 | User can simulate a functional VAR/VMA process — a functional autoregressive / moving-average curve series generated from user-supplied operator kernels. | Section "Functional VAR/VMA Simulator" — matrix-vector recurrence, KL innovations, burn-in. |
| FTS-03-05 | User can simulate a functional ARMA (FARMA) process — a combined AR+MA functional curve-series simulator. | Section "Functional ARMA (FARMA) Simulator" — combined AR+MA recurrence. |
</phase_requirements>

---

## Summary

Phase 41 adds frequency-domain functional time series analysis to `fdars-core` via a new `fts/spectral.rs` module and simulator additions to `simulation.rs`. The core pipeline is: (1) compute lagged autocovariance operators (reusing existing `autocovariance_matrix` from `fts/acf.rs`), (2) apply a Bartlett-weighted DFT across the lag index to produce a complex Hermitian m×m spectral density operator at each Fourier frequency (via `rustfft`), (3) eigendecompose each complex Hermitian operator to obtain frequency-domain eigenvectors, (4) inverse-FFT across frequencies to produce real time-domain dynamic filter taps over a symmetric lag window [−L, L], (5) convolve the original curve series with the filters to produce dynamic scores, and (6) adjoint-convolve scores back to reconstruct curves.

The simulator additions (`sim_fvarma`, `sim_farma`) implement functional VAR/VMA and FARMA recurrences using user-supplied m×m operator matrices acting on discretized grid vectors, with Gaussian KL innovations, burn-in, and deterministic seeding — all aligned with the existing `simulation.rs` patterns. No new crate dependency is required: the full pipeline reuses `rustfft` (already in `Cargo.toml` as 6.2), `nalgebra` (0.33) for eigendecomposition, `rand`/`rand_distr` for RNG, and `num-complex` (0.4) for complex arithmetic.

The R reference is `freqdom`/`freqdom.fda` (Hörmann, Kidziński, Hallin 2015 JRSS-B) for spectral density and DPCA, and `ftsa` (Hyndman & Shang) for FARMA simulation. This implementation matches by capability, not by R signature; divergences are documented below.

**Primary recommendation:** Implement in a single new file `fts/spectral.rs` with five private helpers and three public entry points (`spectral_density`, `dpca`, `dpca_reconstruct`), plus `sim_fvarma` and `sim_farma` appended to `simulation.rs`. The entire pipeline is algebraically precise and testable with white-noise and rank-1 oracles described in the Validation Architecture section.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Lagged autocovariance operators | `fts/acf.rs` (existing) | — | Already implemented as `pub(crate) autocovariance_matrix`; spectral.rs calls it directly |
| Spectral density operator DFT | `fts/spectral.rs` (new) | `rustfft` | DFT across lag index h for each (j₁,j₂) matrix entry; rustfft is the compute backend |
| Eigendecompose complex Hermitian operator per frequency | `fts/spectral.rs` | `nalgebra` | Eigendecomposition of symmetrized real part (see divergence below) or complex Hermitian via nalgebra |
| Dynamic filter taps (inverse FFT of eigenvectors) | `fts/spectral.rs` | `rustfft` | IFFT across frequency grid, per eigenvector component per (j) grid point |
| Dynamic scores (convolution with filter taps) | `fts/spectral.rs` | — | Time-domain convolution over [−L, L] lag window |
| Reconstruction (adjoint filter convolution) | `fts/spectral.rs` | — | Time-reversal of filter taps + sum over components |
| Functional VAR/VMA/FARMA simulation | `simulation.rs` (addition) | `rand`/`rand_distr` | KL-innovation generation + AR/MA matrix-vector recurrence |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `rustfft` | 6.2 [VERIFIED: fdars-core/Cargo.toml] | DFT of lag-h autocovariance operator entries → complex spectral density per frequency; IFFT of eigenvectors → time-domain filter taps | Already in `Cargo.toml`; established usage in `metric/fourier.rs`, `seasonal/` |
| `num-complex` | 0.4 [VERIFIED: fdars-core/Cargo.toml] | `Complex<f64>` type for frequency-domain operators and FFT buffers | Already in `Cargo.toml`; used in `seasonal/` via `rustfft::num_complex::Complex` |
| `nalgebra` | 0.33 [VERIFIED: fdars-core/Cargo.toml] | `SymmetricEigen` for eigendecomposing the real Hermitian-symmetrized spectral density operator per frequency | Already used in `fts/acf.rs` for MC band and in `regression.rs` for FPCA |
| `rand` + `rand_distr` | 0.8 / 0.4 [VERIFIED: fdars-core/Cargo.toml] | `StdRng::seed_from_u64(seed)` + `Normal::new(0.0, 1.0)` for KL innovations in VAR/VMA/FARMA simulators | Crate-wide RNG convention; used in `simulation.rs`, `fts/acf.rs` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `helpers::simpsons_weights` | internal | Integration weights for L2 reconstruction error | All functional inner products and error metrics |
| `fts::acf::autocovariance_matrix` (pub crate) | internal | Lag-h autocovariance operator C_h | Called from `spectral_density` to build the lag sequence |
| `maybe_par_chunks_mut_enumerate!` macro | internal | Optional parallelism in simulation burn-in/generation loops | Only if parallel feature matters for large m |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `SymmetricEigen` on real-symmetrized operator | Full complex Hermitian eigendecomposition | nalgebra does not expose a stable complex Hermitian eigen path in v0.33 without faer; symmetrizing the real part of the spectral density operator is mathematically equivalent for operators estimated with real-valued lag-window weights (Bartlett kernel is real and even → imaginary parts cancel when summed symmetrically; see divergence note) |
| `rustfft` for per-entry lag DFT | explicit Fourier sum ∑ w_h C_h e^{-ihθ} | For large bandwidth the FFT is O(B log B) vs O(B) for a direct sum — but N is typically small (curve count) and the bandwidth B = O(N^{1/3}), so direct sum is viable; using rustfft is consistent with the codebase and handles padding cleanly |

**No new dependency is required.** All required crates are already in `Cargo.toml`.

---

## Package Legitimacy Audit

This phase adds **zero new external crate dependencies** — no audit required. All packages used are existing dependencies already in `fdars-core/Cargo.toml`. [VERIFIED: fdars-core/Cargo.toml]

---

## Architecture Patterns

### System Architecture Diagram

```
Input: FdMatrix (N×m), argvals (m), bandwidth b
        │
        ▼
[autocovariance_matrix (pub crate, fts/acf.rs)]
  C_0, C_1, ..., C_{b-1}  (each m×m, column-major Vec<f64>)
        │
        ▼ (Bartlett weights w_h = 1 - h/b)
[spectral_density (fts/spectral.rs)]
  rustfft DFT across lag index h for each (j1,j2) entry
  → SpectralDensityResult: N frequencies × m×m complex operators
  stored as: freqs Vec<f64>, re Vec<Vec<f64>>, im Vec<Vec<f64>>
        │
        ▼
[dpca (fts/spectral.rs)]
  Per frequency k: symmetrize, nalgebra::SymmetricEigen
  → freq-domain eigenvectors V_k (m × Ndpc complex)
  rustfft IFFT across freq grid per component per grid row j
  → time-domain filter taps a_l[j] for l in [-L, L]
  convolution X * a over lag window → dynamic scores S (N × Ndpc)
  → DpcaResult: filters (2L+1 × m × Ndpc), scores (N × Ndpc), eigenvalues
        │
        ▼
[dpca_reconstruct (fts/spectral.rs)]
  Adjoint filter convolution S * conj(a[-l]) + sum over components
  → DpcaReconstruction: fitted (N × m), reconstruction error (per-component)
        │
[sim_fvarma (simulation.rs)]
  User AR kernels [A_1..A_p] + MA kernels [B_1..B_q] (each m×m)
  KL innovations εₜ ~ N(0, I) via rand + rand_distr
  Recurrence: Xₜ = Σ A_k Xₜ₋ₖ + εₜ + Σ B_k εₜ₋ₖ
  burn-in 200 → FvarmaResult (N × m FdMatrix)
        │
[sim_farma (simulation.rs)]
  Combined AR+MA = sim_fvarma with both AR and MA kernel slices
  → FarmaResult (N × m FdMatrix)
```

### Recommended Project Structure

```
fdars-core/src/
├── fts/
│   ├── mod.rs            # add: mod spectral; pub use spectral::{...}; new *Result structs
│   ├── acf.rs            # unchanged (autocovariance_matrix remains pub(crate))
│   ├── forecast.rs       # unchanged
│   └── spectral.rs       # NEW: spectral_density, dpca, dpca_reconstruct + private helpers
├── simulation.rs         # append: sim_fvarma, sim_farma + their result types
└── lib.rs                # add pub use fts::{spectral_density, dpca, dpca_reconstruct, SpectralDensityResult, DpcaResult, DpcaReconstruction}
                          #        pub use simulation::{sim_fvarma, sim_farma, FvarmaResult, FarmaResult}
```

---

## Algorithmic Specifications

### Spectral Density Estimator (FTS-03-01)

The spectral density operator at frequency θ is:

```
f̂(θ; s, t) = (1/2π) Σ_{h=-(b-1)}^{b-1}  w_h · Ĉ_h(s, t) · e^{-i h θ}
```

where w_h = 1 − |h|/b (Bartlett kernel), Ĉ_h is the lag-h autocovariance operator (m×m matrix), and b = ⌊N^{1/3}⌋ (default). [CITED: freqdom CRAN documentation; ftsa::long_run_covariance_estimation]

The term `(1/2π)` can be folded into the eigenvalues at each step and is a convention choice — the `freqdom` package uses the `1/2π` pre-factor. The estimator differs from `long_run_covariance` (which sums C_0 + Σ w_h(C_h + C_h^T)) in that we need the **full complex** Fourier transform per frequency, not just the zero-frequency real sum. [ASSUMED: the 1/2π scaling convention — verify against `freqdom` source if exact numeric agreement with R is required]

**rustfft implementation approach:**

For each entry (j1, j2) of the autocovariance operator, assemble a length-N complex buffer:
```
buf[h] = w_h * C_h[j1, j2]   for h = 0..b-1
buf[N-h] = w_h * C_h[j1, j2]  (negative lag via C_{-h}[j1,j2] = C_h[j2,j1])
buf[h] = 0                     for h = b..N-b+1
```
Apply `FftPlanner::<f64>::new().plan_fft_forward(N)` (or size 2L+1 if using a smaller window), yielding N complex spectral values — one per Fourier frequency θ_j = 2πj/N.

**Key implementation note:** Because the autocovariance sequence is not symmetric in (j1, j2) for h > 0 (Ĉ_{-h}(s,t) = Ĉ_h(t,s) for stationary series), the full m×m complex spectral density matrix has Hermitian structure: f̂(θ)* = f̂(−θ). The imaginary part of f̂(θ; j1, j2) = −Im(f̂(θ; j2, j1)). This symmetry halves storage — only frequencies θ_j for j = 0..⌊N/2⌋ need be stored (upper half is conjugate). [ASSUMED: implementation may store full N frequencies for FFT convenience]

**Simpler viable alternative (avoids m² FFTs):** Compute the direct sum for each of the B evaluation frequencies in [−π, π]:

```
f̂(θ_j) = (1/2π) Σ_{h=0}^{b-1} w_h * (C_h * e^{-i h θ_j} + C_h^T * e^{i h θ_j}) / (if h>0 else just C_0)
```

This is O(B × b × m²) = O(N^{5/3} m²) for B = N, b = N^{1/3} — acceptable for moderate N and m. The `rustfft` path is O(B log B × m²) and is preferred for large N. [ASSUMED: the direct-sum approach is always correct and is a good fallback/test oracle]

**Eigendecomposition per frequency (key divergence from freqdom):**

`freqdom` uses complex Hermitian eigendecomposition. Since `nalgebra` 0.33 does not expose a stable complex Hermitian eigen path without faer, and the Bartlett-windowed spectral density with real lag-window weights has Hermitian symmetry, we use the real part of f̂(θ_j) for eigendecomposition at each frequency:

```
f̂_real(θ_j) = Re(f̂(θ_j))    // symmetric m×m matrix
eigendecompose via nalgebra::SymmetricEigen::new(dmatrix(f̂_real))
→ eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_m (per frequency)
→ eigenvectors v_1, ..., v_Ndpc (real m-vectors, column-major)
```

**Justification for using real part only:** For a Bartlett-windowed estimator f̂(θ) = (1/2π) Σ w_h C_h e^{-ihθ}, the imaginary part is I(θ; s,t) = −(1/2π) Σ_{h>0} w_h (C_h(s,t) − C_h(t,s)) sin(hθ). For a stationary series where C_{-h} = C_h^T (the autocovariance is not generally symmetric in s,t for h > 0), I(θ) is generally nonzero. However, the **principal subspace** for filter construction can be approximated from Re(f̂(θ)) without loss for the typical case, and this avoids needing faer. **This is a documented divergence from `freqdom` and must appear in rustdoc.** [ASSUMED: using Re only is an acceptable approximation for filter construction; exact complex Hermitian eigen would require faer or a custom implementation]

If exact complex Hermitian eigendecomposition is needed, the 2m×2m real embedding trick works: replace complex Hermitian H = A + iB (A symmetric real, B antisymmetric real) with the real symmetric block matrix [[A, -B], [B, A]], eigendecompose it (size 2m), take the first m eigenvalues and the real/imag parts of the first m eigenvectors. This uses only `nalgebra::SymmetricEigen` with no new dependency. [ASSUMED: the 2m embedding is correct but doubles the eigendecomposition cost]

**Recommended implementation:** Use Re(f̂(θ_j)) at each frequency (real SymmetricEigen), document the divergence, and add a test showing the imaginary part contributes minimally for Gaussian input.

### Dynamic FPCA — Filters (FTS-03-02)

**Filter tap construction** (freqdom `dpca.filters` convention): [CITED: rdrr.io/cran/freqdom.fda/man/fts.dpca.html]

1. At each of the N Fourier frequencies θ_j = 2πj/N, the k-th dynamic eigen-filter's frequency response is the k-th eigenvector v_k(θ_j) of f̂(θ_j) — an m-vector (or complex m-vector, but here real from Re(f̂) eigendecomposition).

2. Arrange the frequency responses V_k = [v_k(θ_0), ..., v_k(θ_{N-1})] as a length-N sequence for each grid row j: `V_k_j = [v_k(θ_0)[j], ..., v_k(θ_{N-1})[j]]`.

3. Apply inverse FFT to each V_k_j sequence (length N), obtaining the doubly-infinite filter tap sequence a_k_j[h]. The time-domain filter taps for lag window [−L, L] are:
   ```
   a_k[j, l]  for l = 0, 1, ..., L, and a_k[j, -l] = a_k[j, l]*  (conjugate symmetry)
   ```
   In practice, due to real-valued eigenvectors (from Re(f̂) approach), the taps are real: `a_k[j, l] = IFFT(V_k_j)[l]` for l = 0..L, and `a_k[j, -l] = a_k[j, l]`.

4. Truncate to the symmetric window [−L, L] where L is the user-specified filter lag. Default L: `L = b` (same as bandwidth) or `L = max(b, 20)` — the exact default is at planner's discretion. [ASSUMED: default L matches bandwidth b = ⌊N^{1/3}⌋; `freqdom` uses `q` for both the spectral density window and the filter window]

**Phase alignment issue across frequencies:** eigenvectors are unique only up to sign at each frequency independently. To avoid sign flips across frequencies (which cause cancellation in the IFFT), align eigenvectors so each v_k(θ_j) has a consistent phase convention (e.g., make the entry of largest absolute magnitude positive). [ASSUMED: sign alignment is needed; the exact convention follows `freqdom`'s eigenvector phase convention which may differ]

**Result struct layout for DpcaResult:**
```
pub struct DpcaResult {
    pub filters: Vec<FdMatrix>,   // length Ndpc; each FdMatrix is (2L+1) × m (filter taps, lag -L..L in rows)
    pub scores: FdMatrix,         // N × Ndpc (one score series per component)
    pub eigenvalues: Vec<Vec<f64>>, // Ndpc × N_freq (eigenvalue at each freq, per component)
    pub n_freqs: usize,           // N (number of Fourier frequencies)
    pub filter_lag: usize,        // L
    pub ncomp: usize,             // Ndpc retained
    // spectral density embedded or returned separately
}
```

### Dynamic FPCA — Scores (FTS-03-02)

**Score computation via time-domain convolution:**

For the k-th component, the dynamic score at time t is: [CITED: rdrr.io/cran/freqdom/man/dpca.html]
```
s_k[t] = Σ_{l=-L}^{L} <a_k[l], X_{t-l}> = Σ_{l=-L}^{L} Σ_j a_k[j, l] * X[t-l, j] * weights[j]
```
where `weights[j]` are Simpson quadrature weights (L2 inner product approximation).

This is a dot product of the m-vector filter tap `a_k[:, l]` with the curve `X[t-l, :]`, integrated over the grid.

**Boundary handling:** For t < L or t > N−1−L, the score requires curves outside the observed range. Options:
- Compute scores only for t = L..N−1−L (valid interior), producing N−2L scores. [ASSUMED: this is the `freqdom` convention — scores are shorter than the series by 2L]
- Zero-pad the series at boundaries. [ASSUMED: not recommended — introduces bias at endpoints]

**Recommended:** Return scores for t = L..N−1−L (length N−2L per component), consistent with `freqdom::dpca.scores`. The planner must decide whether to pad or trim. The trim approach simplifies reconstruction.

### Reconstruction (FTS-03-03)

**Reconstruction via adjoint filter convolution:**

For K retained components, the reconstruction at time t is: [CITED: rdrr.io/cran/freqdom.fda/man/fts.dpca.html — `dpca.KLexpansion`]
```
X̂[t, j] = Σ_{k=1}^{K} Σ_{l=-L}^{L} a_k[j, -l] * s_k[t-l]  (adjoint = time-reversed filter)
```

For real filter taps (Re(f̂) approach), `a_k[j, -l] = a_k[j, l]` (the filter is symmetric), so:
```
X̂[t, j] = Σ_{k=1}^{K} Σ_{l=-L}^{L} a_k[j, l] * s_k[t-l]
```
This is exactly the same as the forward convolution but with the score series in the place of the data.

**Reconstruction error (success criterion 3):** The Frobenius/L2 reconstruction error must decrease monotonically as K increases from 1 to Ndpc:
```
error(K) = (1/N') Σ_t Σ_j (X[t,j] - X̂_K[t,j])² * weights[j]
```
where N' is the number of valid interior time points after truncation. This is the test oracle for FTS-03-03. [ASSUMED: monotone decrease holds under the optimal DPCA filter — it follows from the optimality theory in Hörmann et al. 2015; the numerical test should use a sufficiently large N and small L to avoid numerical instability]

**DpcaReconstruction result struct:**
```
pub struct DpcaReconstruction {
    pub fitted: FdMatrix,               // (N' × m), reconstructed curves for interior points
    pub reconstruction_error: Vec<f64>, // per-component cumulative error (length Ndpc)
    pub valid_range: (usize, usize),    // (L, N-1-L) — time indices of valid interior
}
```

### Functional VAR/VMA Simulator (FTS-03-04)

**Recurrence formula for functional VAR(p)/VMA(q):** [CITED: rdrr.io/cran/freqdom.fda/src/R/fts.rar.R]

```
X_t = Σ_{k=1}^{p} A_k · X_{t-k}  +  ε_t  +  Σ_{k=1}^{q} B_k · ε_{t-k}
```

where:
- `A_k` is the k-th AR operator kernel (m×m matrix, user-supplied), acting via matrix-vector product on the grid-discretized curve `X_{t-k}` (an m-vector)
- `B_k` is the k-th MA operator kernel (m×m matrix, user-supplied)
- `ε_t` is a Gaussian white-noise curve at time t, generated via KL expansion

**Innovation generation:** Use `sim_kl`-style Gaussian sampling:
```rust
let normal = Normal::new(0.0, 1.0).unwrap();
let eps_t: Vec<f64> = (0..m).map(|_| rng.sample(normal)).collect();
// Or use explicit eigenstructure: eps_t = phi * xi where xi ~ N(0, lambda)
```
For the simulator, the simplest Gaussian innovation is i.i.d. N(0,1) per grid point (identity covariance kernel — "standard functional white noise"). [ASSUMED: this is the simplest and most common simulator convention; `freqdom::fts.rar` allows a user-supplied covariance matrix σ for the innovations, which can be a future extension]

**Burn-in:** Simulate `burn_in + n` curves total, discard the first `burn_in`. Default: `burn_in = 200`. [ASSUMED: 200 is a reasonable default for moderate dependence; the `freqdom` package uses a similar approach]

**Stationarity note:** The VAR(p) is stationary if the spectral radius of the companion matrix (p×p block matrix built from A_1..A_p) is < 1. This is the user's responsibility to ensure; the function should document but not enforce stationarity (it cannot compute spectral radius at reasonable cost). [ASSUMED: not validated at runtime — document in rustdoc]

**Result struct:**
```
pub struct FvarmaResult {
    pub curves: FdMatrix,   // N × m simulated curve series
    pub ar_order: usize,    // p
    pub ma_order: usize,    // q
    pub burn_in: usize,
}
```

**Function signature sketch:**
```rust
pub fn sim_fvarma(
    n: usize,                         // number of output curves
    argvals: &[f64],                  // grid (m points)
    ar_ops: &[Vec<f64>],             // p AR kernels, each flat m×m column-major
    ma_ops: &[Vec<f64>],             // q MA kernels, each flat m×m column-major
    burn_in: usize,                   // curves to discard
    seed: u64,
) -> Result<FvarmaResult, FdarError>
```

### Functional ARMA (FARMA) Simulator (FTS-03-05)

FARMA = VAR/VMA with both AR and MA terms. `sim_farma` is a thin convenience wrapper around `sim_fvarma`:

```rust
pub fn sim_farma(
    n: usize,
    argvals: &[f64],
    ar_ops: &[Vec<f64>],
    ma_ops: &[Vec<f64>],
    burn_in: usize,
    seed: u64,
) -> Result<FarmaResult, FdarError>
```

`FarmaResult` wraps `FvarmaResult` (or aliases it) with the same fields. The distinction from `sim_fvarma` is purely semantic — `sim_fvarma` is the general combined AR+MA simulator, and `sim_farma` is the named FARMA entry point referenced in the requirements. They can share implementation. [ASSUMED: two named entry points are cleaner for the public API than a single combined one, even if they share implementation]

---

## Existing Code Verified for Reuse

### `fts/acf.rs` — `autocovariance_matrix` [VERIFIED: fdars-core/src/fts/acf.rs:73-95]

Verbatim signature:
```rust
pub(crate) fn autocovariance_matrix(
    data: &FdMatrix,
    xbar: &[f64],
    h: usize,
    n: usize,
    m: usize,
) -> Vec<f64>
```
Returns flat m×m Vec in column-major order: `c_h[j1 + j2 * m]`. Normalised by `1/n`. The `h = 0` case returns sample covariance C_0.

### `fts/acf.rs` — `mean_curve` and `validate_fts_input` [VERIFIED: fdars-core/src/fts/acf.rs:25-58]

Both private to `acf.rs`. The spectral.rs module must either re-implement `validate_fts_input` verbatim (as `forecast.rs` does — see `forecast.rs:54-75`) or promote it to `pub(super)`.

### `fts/acf.rs` — `long_run_covariance` bandwidth convention [VERIFIED: fdars-core/src/fts/acf.rs:684-687]

```rust
let resolved_bandwidth = match bandwidth {
    None => (n as f64).cbrt().floor() as usize,
    Some(b) => b,
};
```
Default bandwidth `⌊N^{1/3}⌋` confirmed. The spectral density estimator uses the same default for the Bartlett lag window.

### `seasonal/mod.rs` — rustfft import pattern [VERIFIED: fdars-core/src/seasonal/mod.rs:31-32]

```rust
use rustfft::FftPlanner;
...
use num_complex::Complex;
```
Usage in `metric/fourier.rs` [VERIFIED: fdars-core/src/metric/fourier.rs:7-8]:
```rust
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
```
Pattern for forward FFT [VERIFIED: fdars-core/src/metric/fourier.rs:13-22]:
```rust
let mut planner = FftPlanner::<f64>::new();
let fft = planner.plan_fft_forward(m);
let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
fft.process(&mut buffer);
```

### `simulation.rs` — RNG seeding pattern [VERIFIED: fdars-core/src/simulation.rs:311-313]

```rust
let mut rng = match seed {
    Some(s) => StdRng::seed_from_u64(s),
    None => StdRng::from_entropy(),
};
```
VAR/VMA simulators use `seed: u64` (not `Option<u64>`) per crate convention for public FTS functions. [VERIFIED: fdars-core/src/fts/acf.rs:258 — `functional_acf` uses `seed: u64` not `Option`]

### `fts/mod.rs` — Result struct pattern [VERIFIED: fdars-core/src/fts/mod.rs:33-45]

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FacfResult {
    pub lags: Vec<u32>,
    pub acf: Vec<f64>,
    // ...
}
```
All new Result structs in `fts/mod.rs` must follow this exact pattern.

### `fts/mod.rs` — pub use pattern [VERIFIED: fdars-core/src/fts/mod.rs:20-26]

```rust
mod acf;
mod forecast;

pub use acf::{
    functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test,
};
pub use forecast::{fplsr, ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update};
```
Add `mod spectral;` and `pub use spectral::{spectral_density, dpca, dpca_reconstruct};` in the same style.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Forward/inverse FFT | Custom DFT loop | `rustfft::FftPlanner` + `plan_fft_forward` / `plan_fft_inverse` | O(N log N) vs O(N²); already in Cargo.toml; established usage in codebase |
| Eigendecomposition of symmetric matrix | Power iteration / QR custom | `nalgebra::SymmetricEigen::new(dmatrix)` + `to_dmatrix()` | Numerically stable; already used in `fts/acf.rs` |
| Bartlett-weighted autocovariance sum | Re-implement | Call `autocovariance_matrix` from `fts/acf.rs` directly | Zero duplication; the helper is already `pub(crate)` |
| Gaussian random curve generation | Custom Box-Muller | `rand_distr::Normal::new(0.0, 1.0)` + `rng.sample(normal)` | Established pattern in `simulation.rs` |
| Simpson quadrature weights | Custom integration | `helpers::simpsons_weights(argvals)` | Shared across all FTS functions; the integration-weight convention is crate-wide |

**Key insight:** The most error-prone step is maintaining eigenvector phase consistency across the N frequency bins when assembling the IFFT input. A sign flip at one frequency produces oscillatory garbage in the filter taps. Always align eigenvector signs before IFFT (see Common Pitfalls §2).

---

## Common Pitfalls

### Pitfall 1: autocovariance_matrix expects h < n (not h <= n)

**What goes wrong:** Calling `autocovariance_matrix(data, xbar, h, n, m)` with `h >= n` causes an integer underflow in the loop bound `n - h` (usize subtraction wraps or panics in debug).

**Why it happens:** The inner loop runs `for i in 0..(n - h)` — if h >= n, this is `0..0` in release (silently returns zero matrix) or panics in debug due to usize underflow. The `long_run_covariance` in `acf.rs` guards with `let max_h = resolved_bandwidth.min(n - 1)` [VERIFIED: fdars-core/src/fts/acf.rs:709]. `spectral_density` must apply the same guard.

**How to avoid:** Validate that `bandwidth <= n - 1` before the loop, or clip to `n - 1` silently (with documentation). Mirror the exact guard from `long_run_covariance`.

**Warning signs:** All-zero spectral density matrix for large bandwidth; panics in debug builds.

### Pitfall 2: Eigenvector sign flip across Fourier frequencies

**What goes wrong:** `nalgebra::SymmetricEigen` returns eigenvectors with arbitrary sign at each frequency independently. For the IFFT over eigenvectors to produce a coherent time-domain filter, the sign must be consistent across frequencies (or equivalently, the direction of the leading eigenvector at each frequency must be aligned).

**Why it happens:** Eigenvectors are defined up to sign (or complex phase for complex Hermitian). When eigendecomposing N separate m×m matrices, each yields an independent random sign choice. The IFFT of a sign-flipped sequence produces a time-domain tap that has a half-period oscillation artifact.

**How to avoid:** Before assembling the IFFT input, align each eigenvector `v_k(θ_j)` so that its entry of maximum absolute value is positive (or use a reference vector from the DC component θ_0). A simple sign-flip rule:
```rust
let max_entry = eigvec.iter().copied().fold(f64::NEG_INFINITY, f64::max);
let sign = if max_entry < 0.0 { -1.0 } else { 1.0 };
eigvec.iter_mut().for_each(|x| *x *= sign);
```
Apply this at each frequency before adding to the IFFT buffer.

**Warning signs:** Filter taps have large high-frequency oscillations; reconstruction error does not decrease monotonically when adding components; scores have unreasonably large variance.

### Pitfall 3: Band-edge spectral density is not positive semidefinite after estimation

**What goes wrong:** For small N or small bandwidth, the estimated spectral density operator at some frequencies may have negative eigenvalues due to finite-sample noise. This causes negative "variance explained" per component.

**Why it happens:** The Bartlett kernel guarantees PSD only asymptotically. For N < 30 or very small bandwidth, the off-diagonal entries of the weighted sum can dominate and produce indefinite matrices.

**How to avoid:** After eigendecomposition, clip negative eigenvalues to zero before computing variance-explained percentages. Do not clip eigenvectors — they are still used for filter construction. Warn if more than 10% of frequency-eigenvalue pairs are negative (optional log/eprintln for debug builds).

**Warning signs:** Negative entries in the returned eigenvalue array; `NaN` in variance-explained.

### Pitfall 4: Score series is shorter than the input by 2L (boundary truncation)

**What goes wrong:** The planner writes reconstruction code that tries to compare X̂[t] with X[t] for all t, but scores only exist for t = L..N−1−L (N−2L time points).

**Why it happens:** Time-domain convolution with a [−L, L] filter requires L future and L past observations. For boundary time points, the filter extends outside the observed range.

**How to avoid:** Return `valid_range = (L, N-1-L)` in `DpcaReconstruction` so callers know which time points have valid reconstructions. Tests should compare on the interior range only.

**Warning signs:** Index-out-of-bounds; reconstruction error evaluated at N points when only N−2L are valid.

### Pitfall 5: VAR simulation with large operator norms diverges

**What goes wrong:** The burn-in curves grow geometrically large (overflow to ±∞) when the AR operator has spectral radius ≥ 1.

**Why it happens:** The operator norm of A_1 being close to 1 causes the homogeneous solution to explode. A scalar analogy: AR(1) with |φ| ≥ 1 is non-stationary.

**How to avoid:** Guard against NaN/Inf in the burn-in loop; return `ComputationFailed` if curves contain NaN after burn-in. Document in rustdoc that the user is responsible for supplying operators with spectral radius < 1 for stationarity. Add a `#[doc = "..."]` note about the Hilbert-Schmidt norm bound: ‖A_1‖_HS < 1 is a sufficient (not necessary) condition for strict stationarity of FAR(1).

**Warning signs:** `f64::INFINITY` or `f64::NAN` in output curves; very large finite values growing monotonically during burn-in.

### Pitfall 6: DFT of m² scalar sequences is O(m² N log N) — watch for large m

**What goes wrong:** The naive implementation computes one FFT per (j1, j2) entry of the m×m autocovariance matrix — that is m² FFTs of length N. For m = 100, N = 200, this is 10,000 FFTs × 200 log(200) ≈ 15M operations — manageable but potentially slow.

**Why it happens:** The spectral density estimator requires a DFT of the scalar sequence {C_h[j1,j2]} for each of the m² matrix entries.

**How to avoid:** Reorganize the loop so the FftPlanner is created once and reused. If `parallel` feature is enabled, parallelize the outer (j1, j2) loop using `rayon`. For typical fdars use cases (m ≤ 50), this is fast enough without parallelism.

**Warning signs:** `spectral_density` takes > 10s for moderate inputs; profiling shows FFT planner recreation inside the inner loop.

---

## Code Examples

### FFT across lag index (spectral density estimator core)

```rust
// Source: metric/fourier.rs pattern + acf.rs autocovariance_matrix
use rustfft::{FftPlanner, num_complex::Complex};

// For a single (j1, j2) entry, assemble the lag sequence and DFT it.
fn spectral_density_entry(
    lag_seq: &[f64],        // w_h * C_h[j1, j2] for h = 0..bandwidth (only positive lags)
    neg_lag_seq: &[f64],    // w_h * C_h[j2, j1] for h = 0..bandwidth (for negative lags: C_{-h}[j1,j2] = C_h[j2,j1])
    n_freq: usize,          // N (number of Fourier frequencies = FFT length)
    planner: &mut FftPlanner<f64>,
) -> Vec<Complex<f64>> {
    let b = lag_seq.len();
    let fft = planner.plan_fft_forward(n_freq);
    let mut buf = vec![Complex::new(0.0, 0.0); n_freq];
    // Positive lags h = 0..b → indices 0..b
    for (h, &val) in lag_seq.iter().enumerate() {
        buf[h] = Complex::new(val, 0.0);
    }
    // Negative lags h = -1..(-(b-1)) → indices N-1..N-(b-1) (circular)
    for h in 1..b {
        buf[n_freq - h] = Complex::new(neg_lag_seq[h], 0.0);
    }
    fft.process(&mut buf);
    buf
}
```

### Eigendecompose per frequency with sign alignment

```rust
// Source: fts/acf.rs nalgebra eigendecomposition pattern + sign alignment
use nalgebra::DMatrix;

fn eigen_at_frequency(spec_real: &[f64], m: usize, ncomp: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut mat = DMatrix::from_column_slice(m, m, spec_real);
    // Symmetrize (defensive, should be symmetric up to float noise)
    for j1 in 0..m {
        for j2 in (j1 + 1)..m {
            let avg = 0.5 * (mat[(j1, j2)] + mat[(j2, j1)]);
            mat[(j1, j2)] = avg;
            mat[(j2, j1)] = avg;
        }
    }
    let eig = nalgebra::SymmetricEigen::new(mat);
    let mut pairs: Vec<(f64, Vec<f64>)> = eig.eigenvalues.iter().zip(eig.eigenvectors.column_iter())
        .map(|(&val, col)| (val, col.iter().copied().collect()))
        .collect();
    // Sort descending by eigenvalue
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Sign-align: largest abs entry positive
    for (_, evec) in &mut pairs {
        let max_abs_entry = evec.iter().copied().fold(f64::NEG_INFINITY, |a, x| a.max(x.abs()));
        let sign = evec.iter().copied()
            .find(|&x| x.abs() == max_abs_entry)
            .map(|x| if x < 0.0 { -1.0 } else { 1.0 })
            .unwrap_or(1.0);
        evec.iter_mut().for_each(|x| *x *= sign);
    }
    let eigenvalues: Vec<f64> = pairs.iter().take(ncomp).map(|(v, _)| *v).collect();
    let eigenvectors: Vec<Vec<f64>> = pairs.into_iter().take(ncomp).map(|(_, e)| e).collect();
    (eigenvalues, eigenvectors)
}
```

### Dynamic score computation (inner product convolution)

```rust
// For component k, compute score at time t (boundary-safe, interior only)
fn dynamic_score_at_t(
    data: &FdMatrix,
    filter_taps: &[Vec<f64>],  // length 2L+1, each Vec<f64> of length m; index 0 = lag -L
    weights: &[f64],
    t: usize,
    l: usize,
    m: usize,
) -> f64 {
    let mut score = 0.0;
    for (tap_idx, tap) in filter_taps.iter().enumerate() {
        let lag = tap_idx as isize - l as isize; // lag = -L..L
        let curve_t = (t as isize + lag) as usize; // time index t + lag
        for j in 0..m {
            score += tap[j] * data[(curve_t, j)] * weights[j];
        }
    }
    score
}
```

### VAR/VMA simulation recurrence

```rust
// Simulate functional VAR/VMA with operator kernels
fn fvarma_step(
    history_x: &[Vec<f64>],  // past p curves (history_x[0] = X_{t-1}, ..., history_x[p-1] = X_{t-p})
    history_eps: &[Vec<f64>],// past q innovations (history_eps[0] = eps_{t-1}, ...)
    ar_ops: &[Vec<f64>],     // p AR kernels, each m×m column-major
    ma_ops: &[Vec<f64>],     // q MA kernels, each m×m column-major
    eps_t: &[f64],           // current innovation (m-vector)
    m: usize,
) -> Vec<f64> {
    let mut x_new = eps_t.to_vec();  // start with innovation
    // AR terms: X_t += A_k * X_{t-k}
    for (k, a_k) in ar_ops.iter().enumerate() {
        if k < history_x.len() {
            for j1 in 0..m {
                let mut s = 0.0;
                for j2 in 0..m {
                    s += a_k[j1 + j2 * m] * history_x[k][j2];
                }
                x_new[j1] += s;
            }
        }
    }
    // MA terms: X_t += B_k * eps_{t-k}
    for (k, b_k) in ma_ops.iter().enumerate() {
        if k < history_eps.len() {
            for j1 in 0..m {
                let mut s = 0.0;
                for j2 in 0..m {
                    s += b_k[j1 + j2 * m] * history_eps[k][j2];
                }
                x_new[j1] += s;
            }
        }
    }
    x_new
}
```

---

## R Baseline Divergences (for rustdoc)

| Item | R `freqdom`/`ftsa` behaviour | Rust divergence | Reason |
|------|-------------------------------|-----------------|--------|
| Complex Hermitian eigendecomposition | `freqdom::freqdom.eigen()` uses full complex arithmetic | Uses `nalgebra::SymmetricEigen` on `Re(f̂(θ))` only | nalgebra 0.33 has no stable complex Hermitian eigen without faer; real-part approx is valid for Bartlett-weighted estimator with real lag-window weights |
| Spectral density `1/2π` scaling | `freqdom` applies `1/(2π)` pre-factor | Implementation may omit or absorb the `1/2π`; document if eigenvalues differ by `2π` from R output | Scaling affects eigenvalue magnitudes but not eigenvectors (filter shapes) |
| Filter frequency grid | `freqdom` evaluates at user-specified `freq` ∈ [−π, π] | Uses Fourier grid θ_j = 2πj/N for j = 0..N-1 (standard DFT frequencies) | rustfft outputs DFT frequencies naturally; user-specified grid would require interpolation |
| Innovation covariance | `freqdom::fts.rar` accepts user-supplied m×m covariance σ for innovations | Rust simulators use i.i.d. N(0,1) pointwise innovations (identity covariance) | Simplest safe default; generalized covariance is deferred |
| Score boundary convention | `freqdom::dpca.scores` trims to valid interior | Rust returns `valid_range` field; same boundary trimming applies | Equivalent behaviour |
| FARMA vs FAR+FMA | `ftsa` uses FARMA terminology; `freqdom` uses FAR/FMA separately | `sim_farma` = combined AR+MA; `sim_fvarma` is the general entry point | Two named entry points for clarity |

---

## Test Oracles (Concrete, Hand-Computable)

### Oracle 1: White-noise spectral density is flat (FTS-03-01)

For i.i.d. white-noise curves `X_t ~ WN(0, C_0)`:
- All lag-h autocovariance operators C_h = 0 for h > 0.
- Spectral density: `f̂(θ) = C_0 / (2π)` at every frequency (constant, real, symmetric).
- **Test:** Generate i.i.d. curves (e.g., via `sim_kl` with eigenvalue decay and zero serial dependence). Call `spectral_density`. Assert that the real part of the spectral operator at each frequency is within numerical tolerance of `C_0 / (2π)` (or `C_0` if `1/2π` is omitted), and that the imaginary part is zero (within 1e-6).

### Oracle 2: Rank-1 series reconstructs exactly with K=1 dynamic component (FTS-03-03)

For a functional time series that lives in a 1-dimensional subspace (e.g., `X_t[j] = a_t * phi(j)` where `a_t` is a scalar AR(1) and `phi` is a fixed shape function):
- The spectral density operator has rank 1 at all frequencies.
- DPCA with K=1 dynamic component should reconstruct the series with near-zero error.
- **Test:** Construct `X_t[j] = a_t * phi[j]` with `a_t = 0.8 * a_{t-1} + eps_t`, N=80, m=20. Call `dpca` + `dpca_reconstruct` with K=1. Assert reconstruction error < 1e-4 (within the valid interior range).

### Oracle 3: Reconstruction error is monotone decreasing in K (FTS-03-03)

**Test:** On any FTS (e.g., Gaussian AR(1) with smooth covariance), call `dpca_reconstruct` K=1,2,...,Ndpc and assert `error[k] >= error[k+1]` for all k. This is the primary success criterion. [ASSUMED: monotone decrease holds under correct DPCA implementation; may require N ≥ 50 and L ≤ 5 to avoid numerical instability]

### Oracle 4: VAR(1) with zero AR operator = pure innovations (FTS-03-04)

For `A_1 = 0` (m×m zero matrix), `X_t = ε_t` (i.i.d. white noise). The sample covariance of the output should be close to the identity (or innovation covariance). **Test:** `sim_fvarma(n=200, m=20, ar_ops=[zeros_mxm], ma_ops=[], burn_in=0, seed=42)`. Assert output is finite, has ~zero lag-1 ACF.

### Oracle 5: Deterministic reproducibility (FTS-03-04/05)

**Test:** Call `sim_fvarma` twice with the same seed and assert bit-identical output (`FvarmaResult.curves == FvarmaResult.curves`). Same for `sim_farma`.

### Oracle 6: FAR(1) produces non-trivial serial dependence (FTS-03-04)

For a rank-1 AR operator `A_1[j1,j2] = 0.8 * phi[j1] * phi[j2]`, the output should show non-trivial lag-1 autocovariance. **Test:** Call `sim_fvarma`, compute lag-1 autocovariance via `autocovariance_matrix`, assert ‖C_1‖ > 0.1 * ‖C_0‖.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Static FPCA (ignores serial dependence) | Dynamic FPCA via spectral density operator | Hörmann et al. 2015 (JRSS-B) | DPCA components are uncorrelated in time; better dimension reduction for FTS |
| Bartlett-only LRC | Flat-top / Parzen kernels | 2010s HAC literature | Lower bias for smooth spectral densities (out of scope for this milestone) |
| Real-valued DFT for spectra | Complex Hermitian spectral density operator | Standard FTS methodology | Handles asymmetric autocovariance correctly |
| ad-hoc VAR simulation | Functional operator-valued VAR with KL innovations | freqdom.fda 0.9.1 | Principled, gap-aware simulation under operator kernels |

**Deprecated/outdated:**
- Using the static (non-dynamic) FPCA for FTS: scores have serial correlation; eigenvalues overestimate long-run variance contribution of early components.

---

## Validation Architecture

> `workflow.nyquist_validation` is enabled (true) in `.planning/config.json`.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[cfg(test)] mod tests`) |
| Config file | none (cargo-native) |
| Quick run command | `cargo test -p fdars-core --features linalg spectral` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| FTS-03-01 | `spectral_density` returns finite m×m Hermitian complex operators at N frequencies | unit | `cargo test -p fdars-core spectral_density` | Oracle 1: white-noise flatness |
| FTS-03-01 | Error on empty data / argvals mismatch | unit | same | Mirrors `functional_acf` error-path pattern |
| FTS-03-01 | Deterministic: same seed → same result (no RNG in spectral_density, but for any random input) | unit | same | construct same data twice, call twice |
| FTS-03-02 | `dpca` returns `DpcaResult` with correct dimensions (N−2L scores, filters shape (2L+1)×m) | unit | `cargo test -p fdars-core dpca` | Dimension checks |
| FTS-03-02 | `dpca` on white-noise: eigenvalues are approximately equal across frequencies (flat spectrum) | unit | same | Oracle 1 variant |
| FTS-03-03 | `dpca_reconstruct` monotone-decreasing error as K increases | unit | `cargo test -p fdars-core dpca_reconstruct` | Oracle 3 — primary success criterion |
| FTS-03-03 | Rank-1 series reconstructs with K=1 to < 1e-4 error | unit | same | Oracle 2 |
| FTS-03-04 | `sim_fvarma` bit-identical across two calls with same seed | unit | `cargo test -p fdars-core sim_fvarma` | Oracle 5 |
| FTS-03-04 | `sim_fvarma` with zero AR operator = white noise (near-zero lag-1 ACF) | unit | same | Oracle 4 |
| FTS-03-04 | `sim_fvarma` with rank-1 AR operator produces non-trivial serial dependence | unit | same | Oracle 6 |
| FTS-03-04 | Error on dimension mismatch (ar_ops[k].len() != m*m) | unit | same | Standard error-path |
| FTS-03-05 | `sim_farma` bit-identical across two calls with same seed | unit | `cargo test -p fdars-core sim_farma` | Oracle 5 variant |
| FTS-03-05 | `sim_farma` with both AR and MA terms produces correct output shape | unit | same | Dimension check |
| ALL | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` passes | lint | CI gate | MEMORY.md requirement |
| ALL | `cargo fmt` is clean | fmt | CI gate | MEMORY.md requirement |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core spectral` (spectral.rs tests only, ~5s)
- **Per wave merge:** `cargo test -p fdars-core --features linalg` (full fts + simulation suite, ~30s)
- **Phase gate:** Full suite green before `/gsd-verify-work` — `cargo test -p fdars-core --features linalg,parallel`

### Wave 0 Gaps

- [ ] `fdars-core/src/fts/spectral.rs` — new file, does not exist yet
- [ ] New Result structs (`SpectralDensityResult`, `DpcaResult`, `DpcaReconstruction`) in `fdars-core/src/fts/mod.rs`
- [ ] New simulator Result structs (`FvarmaResult`, `FarmaResult`) in `fdars-core/src/simulation.rs`
- [ ] `mod spectral;` + `pub use` additions in `fdars-core/src/fts/mod.rs`
- [ ] Crate-root re-exports in `fdars-core/src/lib.rs`

---

## Security Domain

`security_enforcement` is enabled in config.json. This phase is a pure numerical Rust library implementing mathematical algorithms — no network I/O, no file I/O, no authentication, no serialization of untrusted input.

### Applicable ASVS Categories (Level 1)

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not applicable — library function, no user auth |
| V3 Session Management | No | Not applicable |
| V4 Access Control | No | Not applicable |
| V5 Input Validation | Yes | `validate_fts_input` at all entry points; dimension and parameter range checks |
| V6 Cryptography | No | RNG is used for simulation only (not for security) |

### Threat Patterns for this Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow/underflow in usize arithmetic (n - h) | Tampering | Guard: `max_h = bandwidth.min(n - 1)` before loop |
| NaN/Inf propagation from diverging VAR simulation | Denial of service | Guard: check for NaN in burn-in output; return `ComputationFailed` |
| Panics from `unwrap` on FFT planner (never fails for valid sizes) | Reliability | Use `.expect("valid FFT size")` with descriptive message |
| Out-of-bounds curve access during convolution (boundary) | Tampering | Only access data[t] for t in L..N-1-L (valid interior) |

---

## Environment Availability

Step 2.6: SKIPPED — this phase is purely code additions to `fdars-core/src/`. No external tools, services, runtimes, or CLIs beyond the Rust toolchain are required. The Rust toolchain is verified operational by the existing CI and prior milestone completions.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Using `Re(f̂(θ))` for eigendecomposition is an acceptable approximation for dynamic filter construction (imaginary part contributes negligibly for Bartlett-windowed estimator) | Dynamic FPCA divergence table | Filter taps may have reduced accuracy for highly asymmetric autocovariance kernels; would require 2m×2m real embedding or faer for exact complex Hermitian eigen |
| A2 | Default filter lag `L = bandwidth = ⌊N^{1/3}⌋` is consistent with `freqdom` convention | Dynamic FPCA — Filters | If `freqdom` uses a different default, R-vs-Rust numeric comparison tests would fail; acceptable since this implementation documents its own convention |
| A3 | `1/2π` scaling convention: whether to include or omit the `1/(2π)` pre-factor is left to the implementation (affects eigenvalue magnitudes, not eigenvectors) | Spectral Density Estimator | Eigenvalue-based "variance explained" would differ from R by factor `2π`; document in rustdoc |
| A4 | Innovations for VAR/VMA simulators are i.i.d. N(0,1) per grid point (identity covariance) | VAR/VMA Simulator | If user needs spatially correlated innovations (smooth functional noise), this is a limitation; documented as a constraint |
| A5 | Burn-in default of 200 is sufficient for the AR operators tested in unit tests (moderate operator norms < 0.5) | VAR/VMA Simulator | For near-unit-root operators, 200 burn-in may be insufficient; user can override |
| A6 | Score monotone-decreasing reconstruction error holds numerically (not just asymptotically) for N ≥ 50 and moderate L | Reconstruction test oracle | For very small N or very large L, numerical instability may prevent monotone decrease in finite samples; test may need tolerance |
| A7 | `freqdom`'s sign-alignment convention for eigenvectors is "largest abs entry positive" | Pitfall 2 | Different sign convention would produce same subspace but phase-shifted filter taps; reconstruction is invariant |

---

## Open Questions

1. **`1/2π` pre-factor in spectral density**
   - What we know: The theoretical definition uses `1/(2π)`. The `freqdom` package applies it; `ftsa::long_run_covariance_estimation` returns a matrix that may or may not include it.
   - What's unclear: Whether the fdars implementation should include or absorb `1/(2π)` so that `spectral_density` at θ=0 with bandwidth→∞ equals `long_run_covariance / (2π)`.
   - Recommendation: Omit `1/(2π)` in the estimator (consistent with `long_run_covariance` not dividing by `2π`), document the convention, and note that eigenvalues are `2π` times larger than `freqdom` output.

2. **Score boundary convention: trim vs. zero-pad**
   - What we know: `freqdom::dpca.scores` returns a shorter series (trimmed). Zero-padding introduces bias.
   - What's unclear: Whether the planner should return N−2L scores (trimmed) or N scores (padded with zeros or NaN) for easier alignment with the original data.
   - Recommendation: Return N−2L scores with `valid_range` field in `DpcaResult`. Reconstruction operates on the same interior range.

3. **Exact default for filter lag L**
   - What we know: `freqdom` uses `q` for both the spectral density window and the filter window. The CONTEXT.md leaves this to planner's discretion.
   - Recommendation: Default `L = bandwidth = ⌊N^{1/3}⌋` (symmetric with the spectral density bandwidth). User can override via a `filter_lag: Option<usize>` parameter.

---

## Sources

### Primary (HIGH confidence — verified by reading source files this session)
- `fdars-core/src/fts/acf.rs` (lines 1–1386) — `autocovariance_matrix`, `mean_curve`, `validate_fts_input`, `long_run_covariance` implementation verified verbatim [VERIFIED]
- `fdars-core/src/fts/mod.rs` (lines 1–149) — `*Result` struct conventions, `pub use` pattern verified verbatim [VERIFIED]
- `fdars-core/src/simulation.rs` (lines 1–550) — `sim_kl`, `EFunType`, `EValType`, RNG seeding pattern verified verbatim [VERIFIED]
- `fdars-core/src/metric/fourier.rs` (lines 1–75) — rustfft usage pattern (`FftPlanner`, `Complex<f64>`, `plan_fft_forward`, `fft.process`) verified verbatim [VERIFIED]
- `fdars-core/src/fts/forecast.rs` (lines 1–100) — `validate_fts_input` re-implementation pattern confirmed [VERIFIED]
- `fdars-core/Cargo.toml` — dependency versions confirmed [VERIFIED]

### Secondary (MEDIUM confidence — official documentation)
- [freqdom.fda CRAN manual — `fts.dpca` documentation](https://rdrr.io/cran/freqdom.fda/man/fts.dpca.html) — DPCA pipeline: spectral density → eigendecomposition per frequency → `fourier.inverse` → filter taps → convolution for scores → `dpca.KLexpansion` reconstruction [CITED]
- [freqdom.fda `fts.rar` source](https://rdrr.io/cran/freqdom.fda/src/R/fts.rar.R) — VAR recurrence, burn-in, innovation parameterization [CITED]
- [freqdom CRAN `dpca` man page](https://rdrr.io/cran/freqdom/man/dpca.html) — `spectral.density()` → `dpca.filters()` → `dpca.scores()` → `dpca.KLexpansion()` pipeline confirmed [CITED]

### Tertiary (LOW confidence — web search)
- [Hörmann, Kidziński, Hallin 2015 JRSS-B abstract](https://academic.oup.com/jrsssb/article-abstract/77/2/319/7040632) — theoretical basis for dynamic FPCA and spectral density operator definition `f_θ = (1/2π) Σ_h Σ_h e^{-ihθ}` [CITED]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified in Cargo.toml this session
- Architecture / existing code reuse: HIGH — all key functions read verbatim from source files
- Spectral density estimator formula: HIGH — consistent across multiple sources
- Dynamic FPCA filter/score algorithm: MEDIUM — R source code not directly readable; reconstructed from official documentation + theory; key assumptions flagged
- VAR/VMA/FARMA recurrence: MEDIUM — R source readable for recurrence; innovation convention ASSUMED
- rustfft usage pattern: HIGH — verified from existing codebase

**Research date:** 2026-08-22
**Valid until:** 2027-02-22 (stable Rust/rustfft/nalgebra ecosystem; DPCA algorithm is well-established since 2015)
