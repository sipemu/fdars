# Example 01: Simulating Functional Data

## What this demonstrates

Functional data is generated synthetically using the Karhunen-Loève (KL) expansion, which represents random functions as linear combinations of orthogonal eigenfunctions weighted by random scores. This example shows how different eigenfunction systems (Fourier, Legendre, Wiener) and eigenvalue decay patterns (linear, exponential, Wiener) produce curves with different smoothness and variability characteristics.

You will also see how to add pointwise or curve-level measurement noise and how `fdars-core` stores functional data in column-major layout.

## API functions used

- `simulation::sim_fundata()` — simulate n curves via KL expansion
- `simulation::eigenfunctions()` — evaluate eigenfunction systems on a grid
- `simulation::eigenvalues()` — generate eigenvalue sequences
- `simulation::add_error_pointwise()` — add independent Gaussian noise at each point
- `simulation::add_error_curve()` — add a single noise term per curve
- `helpers::extract_curves()` — convert column-major matrix to Vec of individual curves

## How to run

```bash
cargo run --example simulation
```

## Expected output

Displays eigenfunction sizes, eigenvalue sequences, data matrix dimensions, curve summaries (min/max/mean), noise MSE comparisons, and column-major indexing examples.

## Key concepts

- **Karhunen-Loève expansion**: X(t) = μ(t) + Σ ξ_k √λ_k φ_k(t), where φ_k are eigenfunctions and ξ_k are random scores
- **Column-major layout**: data[i + j * n] accesses curve i at grid point j
- **Eigenvalue decay**: faster decay → smoother curves; slower decay → rougher curves
