#!/usr/bin/env Rscript
# Run all validation scripts comparing fdars vs fda.usc

cat("============================================================\n")
cat("Running fdars vs fda.usc Validation Suite\n")
cat("============================================================\n\n")

# Check if fda.usc is installed
if (!requireNamespace("fda.usc", quietly = TRUE)) {
  stop("fda.usc package is required for validation. Install with: install.packages('fda.usc')")
}

# Check if fdars is installed
if (!requireNamespace("fdars", quietly = TRUE)) {
  stop("fdars package is not installed. Install with: R CMD INSTALL .")
}

# Get the validation directory
# Try different methods to find the script location
validation_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  # Fallback to current working directory based detection
  if (file.exists("validation/validate_fdata.R")) {
    "validation"
  } else if (file.exists("validate_fdata.R")) {
    "."
  } else {
    stop("Cannot find validation scripts. Run from package root directory.")
  }
})

if (is.null(validation_dir) || validation_dir == "" || validation_dir == ".") {
  validation_dir <- "validation"
}

cat("Running fdata validation...\n")
cat("------------------------------------------------------------\n")
source(file.path(validation_dir, "validate_fdata.R"))

cat("\n\nRunning metric validation...\n")
cat("------------------------------------------------------------\n")
source(file.path(validation_dir, "validate_metric.R"))

cat("\n\nRunning depth validation...\n")
cat("------------------------------------------------------------\n")
source(file.path(validation_dir, "validate_depth.R"))

cat("\n\n============================================================\n")
cat("Validation Suite Complete\n")
cat("============================================================\n")
