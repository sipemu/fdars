# Seasonality Detection Methods: A Comparative Study

## Introduction

This document describes and compares six methods for detecting seasonality in functional time series data. We evaluate each method's performance across different scenarios including varying seasonal strengths, non-linear trends, and different trend types.

## Detection Methods

### 1. AIC Comparison (Fourier vs P-spline)

**Concept**: If data is seasonal, a Fourier basis should fit better than P-splines because Fourier bases naturally capture periodic patterns.

**Mathematical formulation**:

For a curve $y(t)$, we fit two models:

1. **Fourier basis**: $\hat{y}(t) = \sum_{k=0}^{K} a_k \cos(2\pi k t) + b_k \sin(2\pi k t)$

2. **P-spline**: $\hat{y}(t) = \sum_{j=1}^{J} c_j B_j(t)$ with penalty $\lambda \int [\hat{y}''(t)]^2 dt$

We compute AIC for each:
$$\text{AIC} = n \log(\text{RSS}/n) + 2 \cdot \text{edf}$$

where RSS is the residual sum of squares and edf is the effective degrees of freedom.

**Detection rule**: Seasonality detected if $\text{AIC}_{\text{P-spline}} - \text{AIC}_{\text{Fourier}} > 0$

**Interpretation**: When Fourier has lower AIC, the periodic structure is significant enough to justify the global periodic assumption over the local flexibility of splines.

---

### 2. FFT Confidence

**Concept**: Use Fast Fourier Transform to detect dominant frequencies. Strong peaks in the periodogram indicate periodic components.

**Mathematical formulation**:

Given a time series $y_1, y_2, \ldots, y_n$, compute the discrete Fourier transform:
$$Y_k = \sum_{j=1}^{n} y_j e^{-2\pi i (j-1)(k-1)/n}$$

The periodogram (power spectrum) is:
$$P_k = |Y_k|^2$$

**Detection score**:
$$\text{Confidence} = \frac{\max_k P_k}{\text{mean}(P_k)}$$

**Detection rule**: Seasonality detected if Confidence $> 6.0$

**Interpretation**: A high ratio indicates one frequency dominates, suggesting periodicity rather than random noise.

---

### 3. ACF Confidence

**Concept**: Autocorrelation at the seasonal lag should be high for seasonal data.

**Mathematical formulation**:

The autocorrelation function at lag $h$ is:
$$\rho_h = \frac{\sum_{t=1}^{n-h}(y_t - \bar{y})(y_{t+h} - \bar{y})}{\sum_{t=1}^{n}(y_t - \bar{y})^2}$$

For seasonal data with period $p$, we expect $\rho_p$ to be significantly positive.

**Detection score**: Maximum ACF value at estimated period

**Detection rule**: Seasonality detected if ACF confidence $> 0.25$

**Interpretation**: High autocorrelation at the seasonal lag indicates the pattern repeats.

---

### 4. Variance Strength

**Concept**: Decompose variance into seasonal and residual components. High seasonal variance ratio indicates seasonality.

**Mathematical formulation**:

Decompose the series: $y_t = T_t + S_t + R_t$ (trend + seasonal + residual)

The seasonal strength is:
$$\text{SS}_{\text{var}} = 1 - \frac{\text{Var}(R_t)}{\text{Var}(y_t - T_t)}$$

Alternatively:
$$\text{SS}_{\text{var}} = \frac{\text{Var}(S_t)}{\text{Var}(S_t + R_t)}$$

**Detection rule**: Seasonality detected if $\text{SS}_{\text{var}} > 0.2$

**Interpretation**: Values close to 1 mean the seasonal component dominates; values close to 0 mean residual noise dominates.

**Important**: The `period` parameter must be in the same units as `argvals`. For data normalized to [0,1] with 5 annual cycles, use `period = 0.2`.

---

### 5. Spectral Strength

**Concept**: Measure the proportion of spectral power at the seasonal frequency.

**Mathematical formulation**:

Using the periodogram $P_k$, identify the seasonal frequency $f_s = 1/\text{period}$.

$$\text{SS}_{\text{spectral}} = \frac{\sum_{k \in \mathcal{S}} P_k}{\sum_{k} P_k}$$

where $\mathcal{S}$ includes the seasonal frequency and its harmonics.

**Detection rule**: Seasonality detected if $\text{SS}_{\text{spectral}} > 0.3$

**Interpretation**: High values indicate spectral energy is concentrated at seasonal frequencies rather than spread across all frequencies.

---

### 6. Automatic Basis Selection

**Concept**: Let the model selection process decide—if Fourier basis is selected over P-splines, the data is likely seasonal.

**Method**: Uses `select.basis.auto()` with AIC criterion. If the selected basis type is "fourier", seasonality is detected.

**Note**: The internal FFT-based seasonal hint has a threshold that is too low (2.0 instead of ~6.0), causing 100% false positive rate. This needs to be fixed in the Rust implementation.

---

## Experiments

### Experiment 1: Varying Seasonal Strength

**Setup**:
- 11 seasonal strength levels: 0.0, 0.1, ..., 1.0
- 50 curves per strength level
- 5 years of monthly data (60 observations)
- Signal: $y(t) = s \cdot [\sin(2\pi \cdot 5t) + 0.3\cos(4\pi \cdot 5t)] + \epsilon$
- Noise: $\epsilon \sim N(0, 0.3^2)$
- Ground truth: seasonal if $s \geq 0.2$

### Experiment 2: Non-linear Trend

**Setup**:
- 6 seasonal strengths × 6 trend strengths
- Non-linear trend: quadratic + cubic + sigmoid components
- Tests robustness of methods to confounding trends

### Experiment 3: Multiple Trend Types

**Setup**:
- 8 trend types: none, linear, quadratic, cubic, exponential, logarithmic, sigmoid, slow_sine
- 5 seasonal strengths × 4 trend strengths per type
- Tests which trend types cause false positives

---

## Results

### Overall Performance (Experiment 1)

| Method | F1 Score | Precision | Recall | FPR | Specificity |
|--------|----------|-----------|--------|-----|-------------|
| **Variance Strength** | **97.3%** | 98.2% | 96.4% | 2.0% | 92.0% |
| Spectral Strength | 95.3% | 97.4% | 93.3% | 10.0% | 89.0% |
| FFT Confidence | 94.8% | 99.3% | 90.7% | 4.0% | 97.0% |
| AIC Comparison | 91.5% | 94.3% | 88.9% | 18.0% | 76.0% |
| ACF Confidence | 85.4% | 98.3% | 75.6% | 10.0% | 94.0% |
| Basis Auto* | 20.9% | 59.4% | 12.7% | 40.0% | 61.0% |

*Basis Auto has a bug in the internal threshold (needs fix in Rust code)

### Detection Rates by Seasonal Strength

| Strength | AIC | FFT | ACF | Var Str | Spec Str |
|----------|-----|-----|-----|---------|----------|
| 0.0 | 18% | 4% | 10% | 2% | 10% |
| 0.1 | 30% | 2% | 2% | 2% | 12% |
| 0.2 | 56% | 34% | 2% | 60% | 50% |
| 0.3 | 86% | 84% | 34% | 96% | 90% |
| 0.5 | 88% | 100% | 90% | 100% | 100% |
| 1.0 | 96% | 100% | 100% | 100% | 100% |

### Robustness to Trends (Experiment 2)

| Method | F1 (no trend) | F1 (max trend) | F1 Drop |
|--------|---------------|----------------|---------|
| Spectral | 96.3% | 92.5% | 3.9% |
| FFT | 93.7% | 91.8% | 2.0% |
| AIC | 92.2% | 87.0% | 5.7% |
| ACF | 87.4% | 83.5% | 4.5% |

### Problematic Trend Types (Experiment 3)

| Trend Type | FFT FPR | Spectral FPR | Issue |
|------------|---------|--------------|-------|
| slow_sine | **100%** | 0% | FFT detects non-seasonal oscillation |
| quadratic | 10% | 5% | Minor |
| sigmoid | 5% | 5% | Minor |
| linear | 0% | 10% | Handled well |

---

## Interpretation

### Why Variance Strength Performs Best

1. **Direct measurement**: It directly measures the proportion of variance explained by the seasonal component
2. **Robust decomposition**: The STL-like decomposition separates trend from seasonality
3. **Calibrated threshold**: The 0.2 threshold corresponds well to the transition between weak and moderate seasonality

### Why FFT is Vulnerable to slow_sine

FFT detects *any* periodic signal, regardless of period. A slow sine wave (1 cycle over 5 years) appears as a strong peak in the periodogram, indistinguishable from true seasonality. Spectral Strength avoids this by focusing on the *expected* seasonal frequency.

### Why AIC Comparison Has Higher FPR

P-splines with smoothing can sometimes overfit to noise, making Fourier appear relatively better even without true seasonality. The comparison is also sensitive to the range of basis functions tested.

### Why Basis Auto Failed

The internal `detect_seasonality_fft` function uses a threshold of 2.0, but pure noise typically has FFT confidence of 2.5-7.0. This needs to be increased to ~6.0.

---

## Recommendations for Unknown Datasets

### Primary Recommendation: Variance Strength

```r
# Compute seasonal strength with variance method
period_in_argvals_units <- (argvals_range) / expected_cycles_per_series
strength <- seasonal_strength(fd, period = period_in_argvals_units,
                              method = "variance", detrend = "linear")
is_seasonal <- strength > 0.2
```

**Why**: Best F1 score (97.3%), lowest FPR (2%), robust to trends.

### Secondary Check: Spectral Strength

```r
strength <- seasonal_strength(fd, period = period_in_argvals_units,
                              method = "spectral", detrend = "linear")
is_seasonal <- strength > 0.3
```

**Why**: More robust to unknown trend types, especially slow oscillations.

### Ensemble Approach (Most Robust)

```r
# Detect with multiple methods
var_detected <- seasonal_strength(fd, period, method="variance") > 0.2
spec_detected <- seasonal_strength(fd, period, method="spectral") > 0.3
fft_detected <- estimate_period(fd, method="fft")$confidence > 6.0

# Majority vote
is_seasonal <- (var_detected + spec_detected + fft_detected) >= 2
```

### Handling Unknown Period

If the seasonal period is unknown:
```r
# Estimate period first
period_result <- estimate_period(fd, method = "fft", detrend = "linear")
estimated_period <- period_result$period

# Then compute seasonal strength
strength <- seasonal_strength(fd, period = estimated_period, method = "variance")
```

### Critical Considerations

1. **Period units**: Always use argvals units for the period parameter
2. **Detrending**: Use `detrend = "linear"` for most cases
3. **Threshold calibration**: The suggested thresholds assume noise SD ~ 0.3 relative to signal amplitude
4. **Visual verification**: Always plot a sample of curves to verify detection makes sense

---

## Conclusion

For detecting seasonality in functional time series:

1. **Variance Strength** is the most accurate method when the seasonal period is known
2. **Spectral Strength** is most robust to confounding trends and unknown oscillations
3. **FFT Confidence** works well but is vulnerable to slow non-seasonal oscillations
4. **AIC Comparison** provides an interpretable alternative but has higher false positive rates
5. **ACF Confidence** is conservative (low FPR) but misses weak seasonality

The key insight is that simple variance-based decomposition outperforms more complex spectral methods when properly configured with the correct period parameter.
