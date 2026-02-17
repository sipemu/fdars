#!/usr/bin/env Rscript
# Run all validation scripts to generate expected outputs.
# Usage: Rscript validation/R/run_all.R

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  return(getwd())
}

script_dir <- get_script_dir()
message("=== Running all fdars-core validation scripts ===\n")
message(sprintf("Script directory: %s\n", script_dir))

# List all validation scripts (exclude infrastructure scripts)
scripts <- list.files(script_dir, pattern = "^validate_.*\\.R$", full.names = TRUE)

if (length(scripts) == 0) {
  message("No validation scripts found matching 'validate_*.R'")
  quit(status = 0)
}

failures <- character(0)
for (script in sort(scripts)) {
  name <- basename(script)
  message(sprintf("--- Running %s ---", name))
  result <- tryCatch({
    source(script, local = new.env())
    TRUE
  }, error = function(e) {
    message(sprintf("  ERROR: %s", conditionMessage(e)))
    FALSE
  })
  if (!result) failures <- c(failures, name)
  message("")
}

if (length(failures) > 0) {
  message(sprintf("\n=== %d script(s) FAILED: %s ===", length(failures), paste(failures, collapse = ", ")))
  quit(status = 1)
} else {
  message(sprintf("\n=== All %d validation scripts completed successfully ===", length(scripts)))
}
