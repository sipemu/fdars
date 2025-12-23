# Generate Ornstein-Uhlenbeck Process

Simulate sample paths from an Ornstein-Uhlenbeck process using the
Euler-Maruyama discretization scheme.

## Usage

``` r
r.ou(n, t, mu = 0, theta = 1, sigma = 1, x0 = 0, seed = NULL)
```

## Arguments

- n:

  Number of sample paths to generate.

- t:

  Evaluation points (numeric vector).

- mu:

  Long-term mean (default 0).

- theta:

  Mean reversion rate (default 1).

- sigma:

  Volatility (default 1).

- x0:

  Initial value (default 0).

- seed:

  Optional random seed.

## Value

An fdata object containing the simulated paths.

## Details

The OU process satisfies the SDE: dX(t) = -theta \* X(t) dt + sigma \*
dW(t)

## Examples

``` r
t <- seq(0, 1, length.out = 100)
ou_data <- r.ou(n = 20, t = t, theta = 2, sigma = 1)
plot(ou_data)
```
