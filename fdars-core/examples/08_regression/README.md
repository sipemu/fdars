# Example 08: Functional PCA and PLS Regression

## What this demonstrates

Dimensionality reduction of functional data using Functional Principal Component Analysis (FPCA) and Partial Least Squares (PLS) regression. FPCA extracts the dominant modes of variation (eigenfunctions) and their scores, enabling reconstruction with controlled approximation error. PLS finds directions in the functional predictor space that are maximally correlated with a scalar response.

## API functions used

- `regression::fdata_to_pc_1d()` — functional PCA (singular values, loadings, scores, mean)
- `regression::fdata_to_pls_1d()` — PLS regression (weights, scores, loadings)
- `utility::integrate_simpson()` — compute scalar response as integral of each curve
- `helpers::extract_curves()` — extract individual curves for response construction

## How to run

```bash
cargo run --example regression
```

## Expected output

Variance explained per PC (individual and cumulative), PC loadings and scores, reconstruction MSE by number of components, PLS weights and scores.

## Key concepts

- **FPCA**: functional analog of PCA; finds orthogonal eigenfunctions maximizing explained variance
- **Scores**: projections of centered curves onto eigenfunctions (finite-dimensional summary)
- **Reconstruction**: X̂(t) = μ(t) + Σ score_k · loading_k(t); more components → better approximation
- **PLS**: finds directions that maximize covariance between predictors and response
