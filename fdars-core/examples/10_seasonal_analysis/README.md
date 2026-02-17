# Example 10: Seasonal Analysis

## What this demonstrates

Estimating the period of seasonal signals using multiple methods (FFT, ACF, Autoperiod, SAZED ensemble), detecting peaks, measuring seasonal strength, and identifying changes in seasonality over time. Synthetic signals with known periods enable validation of each method.

SAZED is a parameter-free ensemble method that combines spectral, ACF, and zero-crossing estimates for robust period detection.

## API functions used

- `seasonal::estimate_period_fft()` — FFT-based period estimation
- `seasonal::estimate_period_acf()` — ACF-based period estimation
- `seasonal::autoperiod()` — Autoperiod algorithm
- `seasonal::sazed()` — SAZED ensemble detection
- `seasonal::detect_peaks()` — peak detection with prominence filtering
- `seasonal::seasonal_strength_variance()` — variance-based seasonal strength
- `seasonal::seasonal_strength_spectral()` — spectral seasonal strength
- `seasonal::detect_seasonality_changes()` — change points in seasonality

## How to run

```bash
cargo run --example seasonal_analysis
```

## Expected output

Period estimates from all methods (should be near 2.0), SAZED component breakdown, detected peaks with timing, seasonal strength scores, and detected change points for the signal with changing amplitude.

## Key concepts

- **FFT period**: dominant frequency peak in the power spectrum
- **ACF period**: first significant peak in the autocorrelation function
- **SAZED**: ensemble of 5 components (spectral, ACF peak, ACF average, zero-crossing, spectral diff)
- **Seasonal strength**: how much variance is explained by the seasonal component (0 = none, 1 = all)
