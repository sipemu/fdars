# Estimate Instantaneous Period

For signals with time-varying frequency (drifting period), estimates the
instantaneous period at each time point using the Hilbert transform.

## Usage

``` r
instantaneous.period(fdataobj)
```

## Arguments

- fdataobj:

  An fdata object.

## Value

A list with fdata objects:

- period:

  Instantaneous period at each time point

- frequency:

  Instantaneous frequency at each time point

- amplitude:

  Instantaneous amplitude (envelope) at each time point

## Details

The Hilbert transform is used to compute the analytic signal, from which
the instantaneous phase is extracted. The derivative of the phase gives
the instantaneous frequency, and 1/frequency gives the period.

This is particularly useful for signals where the period is not
constant, such as circadian rhythms with drift or frequency-modulated
signals.

## Examples

``` r
# Chirp signal with increasing frequency
t <- seq(0, 10, length.out = 500)
freq <- 0.5 + 0.1 * t  # Frequency increases from 0.5 to 1.5
X <- matrix(sin(2 * pi * cumsum(freq) * diff(c(0, t))), nrow = 1)
fd <- fdata(X, argvals = t)

# Estimate instantaneous period
inst <- instantaneous.period(fd)
# plot(inst$period)  # Shows decreasing period over time
```
