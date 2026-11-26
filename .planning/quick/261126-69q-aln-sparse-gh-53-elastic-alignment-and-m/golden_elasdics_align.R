.libPaths(c("~/.cache/R-lib", .libPaths())); library(elasdics); options(digits=17)
data_curve1 <- data.frame(x1 = c(1, 0.5, -1, -1), x2 = c(1, -0.5, -1, 1))
data_curve2 <- data.frame(x1 = c(0.1,0.7)*sin(1:6), x2 = cos(1:6))
a <- align_curves(data_curve1, data_curve2)
print(a$data_curve2_aligned); print(a$elastic_dist)
d1 <- data_curve1; d1$t <- 0:3/3
a2 <- align_curves(d1, data_curve2); print(a2$data_curve2_aligned$t_optim); print(a2$elastic_dist)
c1 <- data.frame(x1 = sin(0:12/5), x2 = cos(0:12/5))
c2 <- data.frame(x1 = c(1, 0.5, -1, -1), x2 = c(1, -0.5, -1, 1))
ac <- align_curves(c1, c2, closed = TRUE)
print(ac$data_curve1$t); print(ac$data_curve2_aligned); print(ac$elastic_dist)
# Correct (pre-wrap) closed distance, see SUMMARY deviation 1:
c1 <- elasdics:::check_closed(c1); c2 <- elasdics:::check_closed(c2)
s1 <- get_srv_from_points(c1); s2 <- get_srv_from_points(c2)
t <- get_arc_length_param(c2)
loss <- sapply(1:length(s2$t), function(i) elasdics:::compute_distance(s1, s2, t - t[i], TRUE))
init <- t - t[which.min(loss)]
print(elasdics:::find_optimal_t_discrete_closed(c(s1$t,1), t(s1[,-1]), c(s2$t,1), t(s2[,-1]), initial_t=init, eps=0.01))
