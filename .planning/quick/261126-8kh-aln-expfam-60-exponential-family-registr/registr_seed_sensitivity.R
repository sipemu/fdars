# R registr 2.2.1 seed sensitivity of register_fpca (fpca_seed 1988 vs 1, 7, 42) on the golden data.
# Run from this directory: Rscript registr_seed_sensitivity.R
src <- readLines("golden_registr.R")
start <- grep("^# ---------------------------------------------------------------- LCG", src)
end <- grep("^# --------------------------------------------------------- emitters", src)
suppressMessages({.libPaths(c("~/.cache/R-lib", .libPaths())); library(registr)})
options(warn=-1)
eval(parse(text = src[start:(end-1)]))
cmp <- function(Y, fam, npc, ...) {
  a <- register_fpca(Y, family=fam, npc=npc, verbose=0, ...)
  for (s in c(1, 7, 42)) {
    b <- register_fpca(Y, family=fam, npc=npc, verbose=0, fpca_seed=s, ...)
    cat(fam, "seed", s, "max|dbeta|", max(abs(a$hinv_beta-b$hinv_beta)), "rms dt", sqrt(mean((a$Y$t_hat-b$Y$t_hat)^2)),
        "loss", tail(a$convergence$registration_loss,1), tail(b$convergence$registration_loss,1), "\n")
  }
}
cmp(sim(101,15,50,0), "gaussian", 2)
cmp(sim(202,40,60,1), "binomial", 1)
cmp(sim(303,15,50,2), "poisson", 1)
cmp(sim(505,15,50,0,2), "gaussian", 1, incompleteness="trailing", lambda_inc=0.05)
