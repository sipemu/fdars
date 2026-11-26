.libPaths(c("~/.cache/R-lib", .libPaths())); library(elasdics); options(digits=17)
curve <- function(t){ rbind(t*cos(13*t), t*sin(13*t)) }
set.seed(18)
data_curves <- lapply(1:4, function(i){
  m <- sample(10:15, 1)
  delta <- abs(rnorm(m, mean = 1, sd = 0.05))
  t <- cumsum(delta)/sum(delta)
  data.frame(t(curve(t)) + 0.07*t*matrix(cumsum(rnorm(2*length(delta))), ncol = 2))
})
pr <- function(lab, x) { cat(lab, ": ", paste(sprintf("%.17g", x), collapse=", "), "\n", sep="") }
endpoint <- function(em) {
  if (em$type == "smooth") {
    pfun <- function(t) t(elasdics:::make_design(t, knots=em$knots, closed=TRUE, type="smooth") %*% em$coefs)
    K <- em$knots
    sapply(1:2, function(j) sum(sapply(1:(length(K)-1), function(k) integrate(function(u) { v <- pfun(u); v[j,]*sqrt(colSums(v^2)) }, K[k], K[k+1], rel.tol=1e-13, subdivisions=1000)$value)))
  } else {
    colSums(em$coefs * sqrt(rowSums(em$coefs^2)) * diff(em$knots))
  }
}
run <- function(name, curves, knots, type) {
  t0 <- Sys.time()
  em <- compute_elastic_mean(curves, knots = knots, type = type, closed = TRUE)
  cat("=====", name, " time", format(Sys.time()-t0), "\n")
  pr("coef x1", em$coefs[,1]); pr("coef x2", em$coefs[,2])
  pr("endpoint", endpoint(em))
  for (i in seq_along(curves)) { dc <- em$data_curves[[i]]; pr(paste("topt", i), dc$t_optim); pr(paste("t", i), dc$t); pr(paste("dist", i), attr(dc, "dist_to_mean")) }
  em
}
sm <- run("SPIRAL SMOOTH 11", data_curves, seq(0,1,length=11), "smooth")
pm <- run("SPIRAL POLY 15", data_curves, seq(0,1,length=15), "polygon")
# genuinely closed shapes: noisy ellipse-like curves, different start points and point counts
set.seed(7)
shape <- function(s) rbind(cos(2*pi*s) + 0.3*cos(4*pi*s), 0.6*sin(2*pi*s))
closed_curves <- lapply(1:5, function(i) {
  m <- sample(12:18, 1)
  s <- sort(runif(m)); s <- (s + runif(1)) %% 1; s <- sort(s)
  s0 <- runif(1); s <- ((s - s[1]) + s0) %% 1
  ord <- order((s - s0) %% 1); s <- s[ord]
  data.frame(t(shape(s)) + matrix(rnorm(2*m, sd=0.03), ncol=2))
})
for (i in 1:5) { pr(paste("C x1", i), closed_curves[[i]][,1]); pr(paste("C x2", i), closed_curves[[i]][,2]) }
cs <- run("CLOSED SMOOTH 9", closed_curves, seq(0,1,length=9), "smooth")
cp <- run("CLOSED POLY 13", closed_curves, seq(0,1,length=13), "polygon")
