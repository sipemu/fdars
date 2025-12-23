# Generate Brownian Bridge

Simulate sample paths from a Brownian bridge, which is a Brownian motion
conditioned to return to 0 at time 1.

## Usage

``` r
r.bridge(n, t, sigma = 1, seed = NULL)
```

## Arguments

- n:

  Number of sample paths.

- t:

  Evaluation points (should include 0 and 1 for standard bridge).

- sigma:

  Volatility (default 1).

- seed:

  Optional random seed.

## Value

An fdata object containing the simulated paths.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
bb_data <- r.bridge(n = 20, t = t)
plot(bb_data)
```
