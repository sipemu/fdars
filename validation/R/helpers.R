# Shared helpers for validation scripts

library(jsonlite)

# Resolve paths relative to the validation/ directory
validation_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))))
  }
  # Fallback: assume we're sourced from validation/R/
  return(dirname(getwd()))
}

data_dir <- function() file.path(validation_dir(), "data")
expected_dir <- function() {
  d <- file.path(validation_dir(), "expected")
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  d
}

load_data <- function(name) {
  fromJSON(file.path(data_dir(), paste0(name, ".json")))
}

save_expected <- function(obj, name) {
  path <- file.path(expected_dir(), paste0(name, ".json"))
  write_json(obj, path, digits = 17, auto_unbox = TRUE)
  message(sprintf("  wrote %s", path))
}

# Convert column-major flat vector to R matrix (n rows, m cols)
to_matrix <- function(data_vec, n, m) {
  matrix(data_vec, nrow = n, ncol = m, byrow = FALSE)
}

# Convert R matrix to fdata object (fda.usc)
to_fdata <- function(data_vec, n, m, argvals) {
  mat <- to_matrix(data_vec, n, m)
  fda.usc::fdata(mat, argvals = argvals)
}
