# Detect Amplitude Modulation in Seasonal Time Series

Detects whether the amplitude of a seasonal pattern changes over time
(amplitude modulation). Uses either Hilbert transform or wavelet
analysis to extract the time-varying amplitude envelope.

## Usage

``` r
detect_amplitude_modulation(
  fdataobj,
  period,
  method = c("hilbert", "wavelet"),
  modulation_threshold = 0.15,
  seasonality_threshold = 0.3
)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Seasonal period in argvals units.

- method:

  Method for amplitude extraction: "hilbert" (default) or "wavelet".

- modulation_threshold:

  Coefficient of variation threshold for detecting modulation. Default:
  0.15 (i.e., CV \> 15% indicates modulation).

- seasonality_threshold:

  Strength threshold for seasonality detection. Default: 0.3.

## Value

A list with components:

- is_seasonal:

  Logical, whether seasonality is detected

- seasonal_strength:

  Overall seasonal strength (spectral method)

- has_modulation:

  Logical, whether amplitude modulation is detected

- modulation_type:

  Character: "stable", "emerging", "fading", "oscillating", or
  "non_seasonal"

- modulation_score:

  Coefficient of variation of amplitude (0 = stable)

- amplitude_trend:

  Trend in amplitude (-1 to 1): negative = fading, positive = emerging

- amplitude_curve:

  Time-varying amplitude (fdata object)

- time_points:

  Time points for the amplitude curve

## Details

The function first checks if seasonality is present using the spectral
method (which is robust to amplitude modulation). If seasonality is
detected, it extracts the amplitude envelope and analyzes its variation.

**Hilbert method**: Uses Hilbert transform to compute the analytic
signal and extract instantaneous amplitude. Fast but assumes narrowband
signals.

**Wavelet method**: Uses Morlet wavelet transform at the target period.
More robust to noise and non-stationarity.

**Modulation types**:

- `stable`: Constant amplitude (modulation_score \< threshold)

- `emerging`: Amplitude increases over time (amplitude_trend \> 0.3)

- `fading`: Amplitude decreases over time (amplitude_trend \< -0.3)

- `oscillating`: Amplitude varies but no clear trend

- `non_seasonal`: No seasonality detected

## Examples

``` r
# Generate data with emerging seasonality
t <- seq(0, 1, length.out = 200)
amplitude <- 0.2 + 0.8 * t  # Amplitude grows over time
X <- matrix(amplitude * sin(2 * pi * t / 0.2), nrow = 1)
fd <- fdata(X, argvals = t)

result <- detect_amplitude_modulation(fd, period = 0.2)
print(result$modulation_type)  # "emerging"
#> [1] "emerging"
print(result$amplitude_trend)  # Positive value
#> [1] 1

# Compare Hilbert vs Wavelet methods
result_hilbert <- detect_amplitude_modulation(fd, period = 0.2, method = "hilbert")
result_wavelet <- detect_amplitude_modulation(fd, period = 0.2, method = "wavelet")
```
