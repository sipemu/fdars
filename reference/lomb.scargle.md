# Lomb-Scargle Periodogram

Computes the Lomb-Scargle periodogram for period detection in
unevenly-sampled data. The Lomb-Scargle method is designed for
irregularly spaced observations and reduces to the standard periodogram
for evenly-spaced data.

## Usage

``` r
lomb.scargle(fdataobj, oversampling = 4, nyquist_factor = 1)
```

## Arguments

- fdataobj:

  An fdata object. Can have regular or irregular sampling.

- oversampling:

  Oversampling factor for the frequency grid. Higher values give finer
  frequency resolution. Default: 4.

- nyquist_factor:

  Maximum frequency as a multiple of the pseudo-Nyquist frequency.
  Default: 1.

## Value

A list of class "lomb_scargle_result" with components:

- frequencies:

  Vector of evaluated frequencies

- periods:

  Corresponding periods (1/frequency)

- power:

  Normalized Lomb-Scargle power at each frequency

- peak_period:

  Period with highest power

- peak_frequency:

  Frequency with highest power

- peak_power:

  Maximum power value

- false_alarm_probability:

  False alarm probability at peak

- significance:

  Significance level (1 - FAP)

## Details

The Lomb-Scargle periodogram is particularly useful when:

- Data has gaps or missing observations

- Sampling is not uniform (e.g., astronomical observations)

- Working with irregular functional data

The algorithm follows Scargle (1982) with significance estimation from
Horne & Baliunas (1986). For each test frequency, it computes:

\$\$P(\omega) = \frac{1}{2\sigma^2} \left\[ \frac{(\sum_j (y_j -
\bar{y}) \cos\omega(t_j - \tau))^2}{\sum_j \cos^2\omega(t_j - \tau)} +
\frac{(\sum_j (y_j - \bar{y}) \sin\omega(t_j - \tau))^2}{\sum_j
\sin^2\omega(t_j - \tau)} \right\]\$\$

where \\\tau\\ is a phase shift chosen to make the sine and cosine terms
orthogonal.

## References

Scargle, J.D. (1982). Studies in astronomical time series analysis. II.
Statistical aspects of spectral analysis of unevenly spaced data. The
Astrophysical Journal, 263, 835-853.

Horne, J.H., & Baliunas, S.L. (1986). A prescription for period analysis
of unevenly sampled time series. The Astrophysical Journal, 302,
757-763.

## See also

[`estimate.period`](https://sipemu.github.io/fdars/reference/estimate.period.md),
[`sazed`](https://sipemu.github.io/fdars/reference/sazed.md)

## Examples

``` r
# Regular sampling
t <- seq(0, 10, length.out = 200)
X <- matrix(sin(2 * pi * t / 2), nrow = 1)
fd <- fdata(X, argvals = t)
result <- lomb.scargle(fd)
print(result)
#> Lomb-Scargle Periodogram
#> ------------------------
#> Peak period:    2.0000
#> Peak frequency: 0.5000
#> Peak power:     99.5000
#> FAP:            0.0000e+00
#> Significance:   1.0000
#> 
#> Frequency grid: 395 points (0.1000 to 9.9500)

# Irregular sampling (simulated)
set.seed(42)
t_irreg <- sort(runif(100, 0, 10))
X_irreg <- matrix(sin(2 * pi * t_irreg / 2), nrow = 1)
fd_irreg <- fdata(X_irreg, argvals = t_irreg)
result_irreg <- lomb.scargle(fd_irreg)
```
