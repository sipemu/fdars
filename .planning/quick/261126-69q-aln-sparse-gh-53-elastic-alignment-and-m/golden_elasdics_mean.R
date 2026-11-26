.libPaths(c("~/.cache/R-lib", .libPaths())); library(elasdics); options(digits=17)
curve <- function(t){ rbind(t*cos(13*t), t*sin(13*t)) }
set.seed(18)
data_curves <- lapply(1:4, function(i){
  m <- sample(10:15, 1)
  delta <- abs(rnorm(m, mean = 1, sd = 0.05))
  t <- cumsum(delta)/sum(delta)
  data.frame(t(curve(t)) + 0.07*t*matrix(cumsum(rnorm(2*length(delta))), ncol = 2))
})
for (i in 1:4) { cat("curve",i," n=",nrow(data_curves[[i]]),"\n"); cat("x1:", sprintf("%.17g", data_curves[[i]][,1]), sep=", "); cat("\n"); cat("x2:", sprintf("%.17g", data_curves[[i]][,2]), sep=", "); cat("\n") }
pm <- compute_elastic_mean(data_curves, knots = seq(0,1,length=15), type = "poly")
cat("POLY coefs x1:", sprintf("%.17g", pm$coefs[,1]), sep=", "); cat("\n")
cat("POLY coefs x2:", sprintf("%.17g", pm$coefs[,2]), sep=", "); cat("\n")
for (i in 1:4) { cat("poly topt",i, sprintf("%.17g", pm$data_curves[[i]]$t_optim), sep=", "); cat("\n"); cat("dist", sprintf("%.17g", attr(pm$data_curves[[i]], "dist_to_mean")), "\n") }
sm <- compute_elastic_mean(data_curves, knots = seq(0,1,length=11))
cat("SMOOTH coefs x1:", sprintf("%.17g", sm$coefs[,1]), sep=", "); cat("\n")
cat("SMOOTH coefs x2:", sprintf("%.17g", sm$coefs[,2]), sep=", "); cat("\n")
for (i in 1:4) { cat("smooth topt",i, sprintf("%.17g", sm$data_curves[[i]]$t_optim), sep=", "); cat("\n"); cat("dist", sprintf("%.17g", attr(sm$data_curves[[i]], "dist_to_mean")), "\n") }

# consistent distances of each curve at returned t_optim to returned mean
for (i in 1:4) {
  dc <- pm$data_curves[[i]]; sd <- get_srv_from_points(dc[, c(1,3,4)])
  d <- elasdics:::compute_distance(data.frame(t=pm$knots[-15], pm$coefs), sd, dc$t_optim, FALSE)
  cat("poly final dist", sprintf("%.17g", d), "\n")
}
pfun <- function(t) t(elasdics:::make_design(t, knots=sm$knots, closed=FALSE, type="smooth") %*% sm$coefs)
K <- sm$knots
pint <- function(f, a, b) { br <- sort(unique(c(a, K[K > a & K < b], b))); if (b <= a) return(0); sum(sapply(1:(length(br)-1), function(j) integrate(f, br[j], br[j+1], rel.tol=1e-13, subdivisions=1000)$value)) }
pn <- pint(function(u) sapply(u, function(v) sum(pfun(v)^2)), 0, 1)
cat("smooth mean sq norm", sprintf("%.17g", pn), "\n")
for (i in 1:4) {
  dc <- sm$data_curves[[i]]; sd <- get_srv_from_points(dc[, c(1,3,4)])
  s <- c(sd$t,1); q <- t(as.matrix(sd[,-1])); tt <- dc$t_optim
  H <- sapply(1:ncol(q), function(k) pint(function(u) sapply(u, function(v) { cp <- sum(q[,k]*pfun(v)); (s[k+1]-s[k])*max(cp,0)^2 }), tt[k], tt[k+1]))
  qn <- sum(q^2 %*% diff(s))
  cat("smooth final dist", sprintf("%.17g", sqrt(pn+qn-2*sum(sqrt(H)))), "\n")
}
