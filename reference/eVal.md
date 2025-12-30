# Generate Eigenvalue Sequence

Generates eigenvalue sequences with different decay patterns. These
control the variance contribution of each mode in Karhunen-Loeve
simulation.

## Usage

``` r
eVal(M, type = c("linear", "exponential", "wiener"))
```

## Arguments

- M:

  Number of eigenvalues to generate.

- type:

  Character. Type of eigenvalue decay:

  linear

  :   lambda_k = 1/k for k = 1, ..., M

  exponential

  :   lambda_k = exp(-k) for k = 1, ..., M

  wiener

  :   lambda_k = 1/((k - 0.5)\*pi)^2, the Wiener process eigenvalues

## Value

A numeric vector of length M containing the eigenvalues in decreasing
order.

## Details

The eigenvalues control how much variance each eigenfunction contributes
to the simulated curves:

- linear:

  Slow decay, higher modes contribute more variation. Produces rougher
  curves.

- exponential:

  Fast decay, higher modes contribute very little. Produces smoother
  curves.

- wiener:

  Specific decay matching Brownian motion. Use with Wiener
  eigenfunctions for true Brownian motion simulation.

## See also

[`eFun`](https://sipemu.github.io/fdars/reference/eFun.md),
[`simFunData`](https://sipemu.github.io/fdars/reference/simFunData.md)

## Examples

``` r
# Compare decay patterns
lambda_lin <- eVal(20, "linear")
lambda_exp <- eVal(20, "exponential")
lambda_wie <- eVal(20, "wiener")

plot(1:20, lambda_lin, type = "b", log = "y", ylim = c(1e-10, 1),
     main = "Eigenvalue Decay Patterns", xlab = "k", ylab = expression(lambda[k]))
lines(1:20, lambda_exp, col = "red", type = "b")
lines(1:20, lambda_wie, col = "blue", type = "b")
legend("topright", c("Linear", "Exponential", "Wiener"),
       col = c("black", "red", "blue"), lty = 1, pch = 1)
```
