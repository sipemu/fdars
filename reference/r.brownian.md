# Generate Brownian Motion

Simulate sample paths from standard Brownian motion (Wiener process).

## Usage

``` r
r.brownian(n, t, sigma = 1, x0 = 0, seed = NULL)
```

## Arguments

- n:

  Number of sample paths.

- t:

  Evaluation points.

- sigma:

  Volatility (standard deviation per unit time, default 1).

- x0:

  Initial value (default 0).

- seed:

  Optional random seed.

## Value

An fdata object containing the simulated paths.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
bm_data <- r.brownian(n = 20, t = t)
plot(bm_data)
```
