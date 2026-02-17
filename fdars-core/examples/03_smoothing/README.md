# Example 03: Kernel Smoothing Methods

## What this demonstrates

Smoothing noisy functional observations using kernel-based nonparametric regression methods. Compares four approaches — Nadaraya-Watson, local linear, local polynomial, and k-NN — at various bandwidth/k values, using MSE against a known true function.

Bandwidth selection is critical: too small leads to overfitting (tracking noise), too large leads to oversmoothing (losing signal).

## API functions used

- `smoothing::nadaraya_watson()` — kernel-weighted local mean
- `smoothing::local_linear()` — local linear regression
- `smoothing::local_polynomial()` — local polynomial regression (any degree)
- `smoothing::knn_smoother()` — k-nearest-neighbor averaging

## How to run

```bash
cargo run --example smoothing
```

## Expected output

MSE tables for each method at different bandwidths/k values, a comparison of all methods at their best settings, and sample smoothed values at selected grid points.

## Key concepts

- **Nadaraya-Watson**: ŷ(x) = Σ K_h(x-x_i)y_i / Σ K_h(x-x_i), simple kernel average
- **Local linear**: fits a line in each kernel window, reducing boundary bias
- **Local polynomial**: generalizes to degree-p polynomials for flexible fitting
- **Bias-variance tradeoff**: small bandwidth = low bias / high variance; large = opposite
