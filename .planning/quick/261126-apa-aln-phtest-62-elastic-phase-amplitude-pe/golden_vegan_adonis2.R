# Golden reference for fdars-core inference::permanova (GH #62 ALN-PHTEST).
# R vegan 2.7.8, adonis2 (one-way PERMANOVA). Run:
#   Rscript golden_vegan_adonis2.R
.libPaths(c("~/.cache/R-lib", .libPaths()))
suppressMessages(library(vegan))
cat("vegan", as.character(packageVersion("vegan")), "\n")
n <- 15
pts <- function(offset, labs) {
  X <- matrix(0, n, 3)
  for (i in 0:(n - 1)) for (c in 0:2) X[i + 1, c + 1] <- sin(0.37 * (i + 1) * (c + 1) + c)
  X[, 1] <- X[, 1] + offset * labs
  X
}
run <- function(name, D, labs, perms) {
  g <- factor(labs)
  set.seed(1)
  a <- adonis2(D ~ g, data = data.frame(g = g), permutations = perms)
  cat(name, "\n")
  cat(sprintf("  SS_between = %.17g\n  SS_within = %.17g\n  SS_total = %.17g\n",
              a$SumOfSqs[1], a$SumOfSqs[2], a$SumOfSqs[3]))
  cat(sprintf("  R2 = %.17g\n  F = %.17g\n  p(%d perms) = %.6g\n", a$R2[1], a$F[1], perms, a$`Pr(>F)`[1]))
}
# A: 3 balanced groups, Euclidean distance, offset 1.1
labsA <- (0:(n - 1)) %% 3
XA <- pts(1.1, labsA)
run("A_euclid_3groups", dist(XA, method = "euclidean"), labsA, 9999)
# B: 2 unbalanced groups (6/9), Manhattan (non-Euclidean-embeddable in general), offset 0.25
labsB <- c(rep(0, 6), rep(1, 9))
XB <- pts(0.25, labsB)
run("B_manhattan_2groups", dist(XB, method = "manhattan"), labsB, 9999)
# C: 4 unbalanced groups, Euclidean, no offset (H0-like)
labsC <- c(0,1,2,3,0,1,2,3,0,1,2,0,1,0,3)
XC <- pts(0.0, labsC)
run("C_euclid_4groups_null", dist(XC, method = "euclidean"), labsC, 9999)
