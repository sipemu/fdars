# Example 09: Outlier Detection

## What this demonstrates

Detecting outlying functional observations using the likelihood ratio test (LRT) with bootstrap threshold calibration. The method compares the distribution of curve residuals before and after removing each candidate, flagging curves whose removal significantly changes the fit. Results are confirmed using depth-based measures.

Three types of outliers are injected: magnitude shift, shape anomaly (high-frequency oscillation), and partial shift (affecting only part of the domain).

## API functions used

- `outliers::outliers_threshold_lrt()` — bootstrap calibration of the LRT threshold
- `outliers::detect_outliers_lrt()` — flag outliers exceeding the threshold
- `depth::fraiman_muniz_1d()` — depth-based confirmation of outlier status

## How to run

```bash
cargo run --example outlier_detection
```

## Expected output

Bootstrap threshold, detected outlier indices (with true/false positive classification), depth ranking of all curves, and agreement between LRT and depth methods.

## Key concepts

- **LRT outlier detection**: tests whether each curve's residuals differ significantly from the bulk
- **Bootstrap threshold**: simulates the null distribution to calibrate the significance level
- **Depth confirmation**: outliers should appear as low-depth curves, providing independent validation
- **Trimming**: excludes a fraction of extreme curves when computing the reference model
