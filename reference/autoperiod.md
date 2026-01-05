# Autoperiod: Hybrid FFT + ACF Period Detection

Implements the Autoperiod algorithm (Vlachos et al. 2005) which combines
FFT-based candidate detection with ACF validation and gradient ascent
refinement for robust period estimation.

## Usage

``` r
autoperiod(
  fdataobj,
  n_candidates = 5,
  gradient_steps = 10,
  detrend_method = c("none", "linear", "auto")
)
```

## Arguments

- fdataobj:

  An fdata object.

- n_candidates:

  Maximum number of FFT peaks to consider as candidates. Default: 5.
  More candidates increases robustness but also computation time.

- gradient_steps:

  Number of gradient ascent steps for period refinement. Default: 10.
  More steps improves precision.

- detrend_method:

  Detrending method to apply before period estimation:

  "none"

  :   No detrending (default)

  "linear"

  :   Remove linear trend

  "auto"

  :   Automatic AIC-based selection of detrending method

## Value

A list of class "autoperiod_result" with components:

- period:

  Best detected period

- confidence:

  Combined confidence (normalized FFT power \* ACF validation)

- fft_power:

  FFT power at the detected period

- acf_validation:

  ACF validation score (0-1)

- n_candidates:

  Number of candidates evaluated

- candidates:

  Data frame of all candidate periods with their scores

## Details

The Autoperiod algorithm works in three stages:

1.  **Candidate Detection**: Finds peaks in the FFT periodogram

2.  **ACF Validation**: Validates each candidate using the
    autocorrelation function. Checks that the ACF shows a peak at the
    candidate period, and applies harmonic analysis to distinguish
    fundamental periods from harmonics.

3.  **Gradient Refinement**: Refines each candidate using gradient
    ascent on the ACF to find the exact period that maximizes the ACF
    peak.

The final period is chosen based on the product of normalized FFT power
and ACF validation score.

## References

Vlachos, M., Yu, P., & Castelli, V. (2005). On periodicity detection and
structural periodic similarity. In Proceedings of the 2005 SIAM
International Conference on Data Mining.

## See also

[`sazed`](https://sipemu.github.io/fdars/reference/sazed.md) for an
ensemble method,
[`estimate.period`](https://sipemu.github.io/fdars/reference/estimate.period.md)
for simpler single-method estimation

## Examples

``` r
# Generate seasonal data with period = 2
t <- seq(0, 20, length.out = 400)
X <- matrix(sin(2 * pi * t / 2) + 0.1 * rnorm(400), nrow = 1)
fd <- fdata(X, argvals = t)

# Detect period using Autoperiod
result <- autoperiod(fd)
print(result)
#> Autoperiod Detection
#> --------------------
#> Period:         2.0050
#> Confidence:     0.9801
#> FFT Power:      0.5041
#> ACF Validation: 1.0000
#> Candidates:     5

# View all candidates
print(result$candidates)
#>      period    fft_power  acf_score combined_score
#> 1 2.0050125 0.5041332413 1.00000000   9.801127e-01
#> 2 0.1230069 0.0002283611 0.55978906   2.485291e-04
#> 3 0.2108212 0.0002246269 0.47722953   2.084106e-04
#> 4 0.5168326 0.0002032098 0.00741635   2.929986e-06
#> 5 0.2194511 0.0001802642 0.47722953   1.672505e-04
```
