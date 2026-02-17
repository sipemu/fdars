# Example 02: Functional Data Operations

## What this demonstrates

Core operations on functional data: computing the cross-sectional mean, centering (subtracting the mean), numerical derivatives, Lp norms, the geometric median, Simpson's rule integration, and inner products. These are the building blocks for most FDA analyses.

The geometric median is a robust alternative to the pointwise mean — it minimizes the sum of L2 distances to all curves and is less sensitive to outliers.

## API functions used

- `fdata::mean_1d()` — pointwise mean across curves
- `fdata::center_1d()` — subtract the mean from each curve
- `fdata::deriv_1d()` — numerical differentiation (1st, 2nd, etc.)
- `fdata::norm_lp_1d()` — Lp norms (L1, L2, L∞) per curve
- `fdata::geometric_median_1d()` — robust central tendency
- `utility::integrate_simpson()` — numerical integration using Simpson's rule
- `utility::inner_product()` — inner product between two curves
- `utility::inner_product_matrix()` — Gram matrix of all pairwise inner products

## How to run

```bash
cargo run --example functional_operations
```

## Expected output

Mean function values, centering verification, derivative ranges, Lp norm tables, geometric median comparison, integration results, and inner product/Gram matrix values.

## Key concepts

- **Centering**: subtracting the mean so the centered data has zero mean
- **Lp norm**: ||f||_p = (∫|f(t)|^p dt)^(1/p); L∞ is the supremum norm
- **Geometric median**: argmin_μ Σ ||X_i - μ||_2, more robust than pointwise mean
- **Inner product**: <f,g> = ∫ f(t)g(t) dt, computed via Simpson's rule
