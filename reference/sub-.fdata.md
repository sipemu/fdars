# Subset method for fdata objects

Subset method for fdata objects

## Usage

``` r
# S3 method for class 'fdata'
x[i, j, drop = FALSE]
```

## Arguments

- x:

  An object of class 'fdata'.

- i:

  Row indices (which curves to keep).

- j:

  Column indices (which time points to keep).

- drop:

  Logical. If TRUE and only one curve selected, return vector.
