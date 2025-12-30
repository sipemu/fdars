# Add Measurement Error to Functional Data

Adds independent Gaussian noise to functional data observations.

## Usage

``` r
addError(fdataobj, sd = 0.1, type = c("pointwise", "curve"), seed = NULL)
```

## Arguments

- fdataobj:

  An object of class `fdata`.

- sd:

  Standard deviation of the Gaussian noise.

- type:

  Type of noise:

  pointwise

  :   (Default) Independent noise at each evaluation point. Each
      f_i(t_j) gets independent noise.

  curve

  :   Common noise level per curve. Each curve gets a single random
      value added to all its points.

- seed:

  Optional integer random seed for reproducibility.

## Value

An `fdata` object with added noise.

## See also

[`simFunData`](https://sipemu.github.io/fdars/reference/simFunData.md),
[`sparsify`](https://sipemu.github.io/fdars/reference/sparsify.md)

## Examples

``` r
t <- seq(0, 1, length.out = 100)
fd_clean <- simFunData(n = 20, argvals = t, M = 5, seed = 42)
fd_noisy <- addError(fd_clean, sd = 0.1)

par(mfrow = c(1, 2))
plot(fd_clean, main = "Clean Data")

plot(fd_noisy, main = "With Noise (sd = 0.1)")

par(mfrow = c(1, 1))

# Higher noise level
fd_very_noisy <- addError(fd_clean, sd = 0.5)
plot(fd_very_noisy, main = "High Noise (sd = 0.5)")
```
