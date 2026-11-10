#!/usr/bin/env Rscript
# Spec 10.4 rev 60: only the ten cells with matching flexsurv conventions.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
check_only <- identical(args, "--check-only")
script <- "scripts/fixtures/generate_slice_b.R"
stopifnot(file.exists(script), getRversion() == "4.4.2")
versions <- c(flexsurv = "2.3.2", survival = "3.7.0")
for (p in names(versions)) {
  stopifnot(as.character(utils::packageVersion(p)) == versions[[p]])
}
suppressPackageStartupMessages(library(flexsurv))
options(OutDec = ".", digits = 17, warn = 2)

# A shared, reproducible spread of rows; no patient data or fitted parameters.
seed <- 20261110L
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
cells <- c("untruncated_exact", "untruncated_right", "untruncated_left",
           "untruncated_interval", "left_truncated_exact", "left_truncated_right",
           "right_truncated_exact", "right_truncated_left", "right_truncated_interval",
           "two_sided_exact")
rows <- lapply(cells, function(cell) {
  kind <- sub(".*_", "", cell)
  t <- c(.25, .5, 1, 2, 4) * runif(5, .9, 1.1)
  left <- if (grepl("^(left|two)_", cell)) .4 * t else rep(0, 5)
  right <- if (grepl("^(right|two)_", cell)) 1.4 * t else rep(Inf, 5)
  t2 <- if (kind == "interval") 1.2 * t else rep(NA_real_, 5)
  if (cell %in% c("right_truncated_exact", "two_sided_exact")) right[5] <- t[5]
  if (cell == "right_truncated_interval") t2[5] <- right[5]
  data.frame(cell = cell, kind = kind, time = t, time2 = t2,
             left_trunc = left, right_trunc = right)
})
data <- do.call(rbind, rows)
rownames(data) <- as.character(seq_len(nrow(data)))
stopifnot(nrow(data) == 50L, all(table(data$cell) == 5L),
          all(data$time > data$left_trunc), all(data$time <= data$right_trunc))
for (cell in c("right_truncated_exact", "two_sided_exact")) {
  d <- data[data$cell == cell, ]
  stopifnot(any(d$time == d$right_trunc))
}
d <- data[data$cell == "right_truncated_interval", ]
stopifnot(any(d$time2 == d$right_trunc))
interval <- data$kind == "interval"
finite_window <- is.finite(data$right_trunc)
stopifnot(all(data$time2[interval] <= data$right_trunc[interval]),
          all(data$time2[interval] - data$time[interval] >= 1e-4 * data$time2[interval]),
          all(data$right_trunc[finite_window] - data$left_trunc[finite_window] >=
              1e-4 * data$right_trunc[finite_window]))

parameters <- list(
  weibull = list(shape_one = c(shape = 1, scale = 2),
                 shape_1_5 = c(shape = 1.5, scale = 1.7)),
  lognormal = list(positive_location = c(meanlog = .3, sdlog = .8),
                   negative_location = c(meanlog = -.1, sdlog = 1.1)),
  loglogistic = list(shape_two = c(shape = 2, scale = 1.5),
                     shape_three = c(shape = 3, scale = 2))
)
distributions <- c(weibull = "weibull", lognormal = "lnorm", loglogistic = "llogis")
close <- function(actual, expected) {
  stopifnot(length(actual) == length(expected), all(is.finite(actual)),
            all(is.finite(expected)),
            all(abs(actual - expected) <= 1e-10 + 1e-8 * abs(expected)))
}

# Independent section-2 algebra in R, not the values exported as references.
log_survival <- function(t, family, p) {
  switch(family,
         weibull = pweibull(t, p[1], p[2], lower.tail = FALSE, log.p = TRUE),
         lognormal = plnorm(t, p[1], p[2], lower.tail = FALSE, log.p = TRUE),
         loglogistic = -log1p((t / p[2])^p[1]))
}
log_density <- function(t, family, p) {
  switch(family,
         weibull = dweibull(t, p[1], p[2], log = TRUE),
         lognormal = dlnorm(t, p[1], p[2], log = TRUE),
         loglogistic = log(p[1] / p[2]) + (p[1] - 1) * log(t / p[2]) -
           2 * log1p((t / p[2])^p[1]))
}
log_mass <- function(a, b, family, p) {
  sa <- log_survival(a, family, p)
  sb <- log_survival(b, family, p)
  sa + log(-expm1(sb - sa))
}

families <- list()
for (family in names(parameters)) {
  vectors <- lapply(names(parameters[[family]]), function(label) {
    p <- parameters[[family]][[label]]
    bounds <- unlist(data[c("time", "time2", "left_trunc", "right_trunc")])
    bounds <- bounds[is.finite(bounds)]
    stopifnot(all(-log_survival(bounds, family, p) <= 15))
    log_z <- log_mass(data$left_trunc, data$right_trunc, family, p)
    stopifnot(all(log_z >= log(1e-6)))
    expected <- numeric(nrow(data))
    for (i in seq_len(nrow(data))) {
      d <- data[i, ]
      numerator <- switch(d$kind,
        exact = log_density(d$time, family, p),
        right = log_survival(d$time, family, p),
        left = log_mass(0, d$time, family, p),
        interval = log_mass(d$time, d$time2, family, p))
      if (d$kind != "exact") stopifnot(numerator >= log(1e-6))
      expected[i] <- numerator - log_z[i]
    }
    values <- numeric(nrow(data))
    for (cell in cells) {
      idx <- which(data$cell == cell)
      d <- data[idx, ]
      kind <- d$kind[1]
      d$Y <- if (any(d$left_trunc > 0)) {
        survival::Surv(d$left_trunc, d$time, rep(kind == "exact", nrow(d)))
      } else if (kind == "left") {
        survival::Surv(d$time, d$time, rep(2L, nrow(d)), type = "interval")
      } else if (kind == "interval") {
        survival::Surv(d$time, d$time2, type = "interval2")
      } else {
        survival::Surv(d$time, rep(kind == "exact", nrow(d)))
      }
      call <- list(formula = Y ~ 1, data = d, dist = distributions[[family]],
                   inits = unname(p), fixedpars = TRUE, hessian = FALSE)
      if (all(is.finite(d$right_trunc))) call$rtrunc <- quote(right_trunc)
      fit <- do.call(flexsurv::flexsurvreg, call)
      stopifnot(fit$npars == 0L, length(fit$logliki) == length(idx),
                identical(rownames(fit$data$Y), rownames(d)))
      close(unname(fit$res[names(p), "est"]), unname(p))
      close(sum(fit$logliki), fit$loglik)
      close(fit$logliki, expected[idx])
      values[idx] <- unname(fit$logliki)
    }
    stopifnot(all(is.finite(values)), all(values[data$kind != "exact"] <= 0))
    list(name = label, parameters = as.list(p), loglik = values)
  })
  families[[family]] <- list(flexsurv_dist = distributions[[family]], vectors = vectors)
}
message("Checks passed: 50 shared synthetic rows, 10 cells, 6 fixed vectors (300 values)")
if (check_only) quit(save = "no", status = 0L)

# Base R JSON: 17 significant digits; only unused time2 and infinite U become null.
to_json <- function(x) {
  if (is.list(x)) {
    fields <- vapply(x, to_json, character(1))
    if (is.null(names(x))) return(paste0("[", paste(fields, collapse = ","), "]"))
    keys <- encodeString(names(x), quote = '"')
    return(paste0("{", paste0(keys, ":", fields, collapse = ","), "}"))
  }
  if (is.numeric(x)) {
    stopifnot(!any(is.nan(x)))
    fields <- ifelse(is.na(x) | is.infinite(x), "null", sprintf("%.17g", x))
  } else {
    stopifnot(is.character(x))
    fields <- encodeString(x, quote = '"')
  }
  if (length(fields) == 1L) return(fields)
  paste0("[", paste(fields, collapse = ","), "]")
}
output <- args[[1]]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
# Keep a CSV companion for human inspection, with identical row order/precision.
csv <- as.data.frame(lapply(data, function(x) {
  if (is.numeric(x)) ifelse(is.na(x), "", sprintf("%.17g", x)) else x
}))
write.table(csv, file.path(output, "data.csv"), row.names = FALSE, col.names = TRUE,
            sep = ",", quote = FALSE)
meta <- list(seed = seed, generator = script, R = as.character(getRversion()),
             flexsurv = versions[["flexsurv"]], survival = "3.7-0",
             platform = R.version$platform, generator_md5 = unname(tools::md5sum(script)),
             rng = c("Mersenne-Twister", "Inversion", "Rejection"),
             call = "flexsurvreg(Y ~ 1, data=d, dist=dist, inits=p, fixedpars=TRUE, hessian=FALSE, rtrunc=right_trunc when finite)")
writeLines(to_json(list(meta = meta, data = as.list(data), families = families)),
           file.path(output, "slice_b_likelihood.json"), useBytes = TRUE)
message("Wrote data.csv and slice_b_likelihood.json to ", output)
