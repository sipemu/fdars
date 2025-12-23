# Bootstrap Functional Data

Generate bootstrap samples from functional data. Supports naive
bootstrap (resampling curves with replacement) and smooth bootstrap
(adding noise based on estimated covariance structure).

## Usage

``` r
fdata.bootstrap(
  fdataobj,
  n.boot = 200,
  method = c("naive", "smooth"),
  variance = NULL,
  seed = NULL
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- n.boot:

  Number of bootstrap replications (default 200).

- method:

  Bootstrap method: "naive" for resampling with replacement, "smooth"
  for adding Gaussian noise (default "naive").

- variance:

  For method="smooth", the variance of the added noise. If NULL,
  estimated from the data.

- seed:

  Optional seed for reproducibility.

## Value

A list of class 'fdata.bootstrap' with components:

- boot.samples:

  List of n.boot fdata objects, each a bootstrap sample

- original:

  The original fdata object

- method:

  The bootstrap method used

- n.boot:

  Number of bootstrap replications

## Examples

``` r
# Create functional data
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
fd <- fdata(X, argvals = t)

# Naive bootstrap
boot_naive <- fdata.bootstrap(fd, n.boot = 100, method = "naive")

# Smooth bootstrap
boot_smooth <- fdata.bootstrap(fd, n.boot = 100, method = "smooth")
```
