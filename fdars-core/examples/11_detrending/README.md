# Example 11: Detrending and Decomposition

## What this demonstrates

Removing trends from functional data using parametric (linear, polynomial) and nonparametric (LOESS) methods. Also shows automatic method selection, additive seasonal decomposition (trend + seasonal + remainder), and STL (Seasonal and Trend decomposition using LOESS).

The synthetic signal combines a linear trend with a sinusoidal seasonal component, allowing direct comparison of recovered components against known truth.

## API functions used

- `detrend::detrend_linear()` — linear trend removal
- `detrend::detrend_polynomial()` — polynomial trend removal
- `detrend::detrend_loess()` — LOESS-based flexible detrending
- `detrend::auto_detrend()` — automatic method selection
- `detrend::decompose_additive()` — additive decomposition (trend + seasonal + remainder)
- `detrend::stl_decompose()` — STL decomposition

## How to run

```bash
cargo run --example detrending
```

## Expected output

Detrending MSE for each method/setting, auto-selected method, decomposition component MSEs, STL results, and verification that components sum to the original signal.

## Key concepts

- **Linear detrending**: fits y = a + bt and subtracts the trend
- **LOESS detrending**: fits a local polynomial smoother as the trend — captures nonlinear trends
- **Additive decomposition**: y(t) = trend(t) + seasonal(t) + remainder(t)
- **STL**: iterative LOESS-based decomposition, robust to outliers
