#!/usr/bin/env Rscript
# spec 10.4/11.1: synthetic observations, never flexsurv::bc or its row outputs.

# Command-line options and pinned environment ------------------------------

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) <= 1L)
check_only <- identical(args, "--check-only")
output <- "tests/fixtures/bc_like"
if (length(args) && !check_only) {
  if (startsWith(args[[1L]], "--")) {
    stop("Unknown option: ", args[[1L]])
  }
  output <- args[[1L]]
}

script <- "scripts/fixtures/generate_bc_like.R"
versions <- c(flexsurv = "2.3.2", survival = "3.7.0")
stopifnot(file.exists(script), getRversion() == "4.4.2")
for (name in names(versions)) {
  stopifnot(as.character(utils::packageVersion(name)) == versions[[name]])
}
suppressPackageStartupMessages(library(flexsurv))
options(OutDec = ".", digits = 17)

# Parameter conventions and package evaluation -----------------------------

group_sizes <- c(Good = 229L, Medium = 229L, Poor = 228L)
group_names <- names(group_sizes)
cases <- list(
  fitted = c(
    shape = 1.3796517717622814, scale_Good = 11.422861823436245,
    beta_Medium = -0.6135891674792526, beta_Poor = -1.2122137266943085
  ),
  unit_shape = c(shape = 1, scale_Good = 10, beta_Medium = -0.5, beta_Poor = -1),
  alternative = c(shape = 1.5, scale_Good = 12, beta_Medium = -0.6, beta_Poor = -1.2)
)

group_scales <- function(parameters) {
  beta <- c(0, parameters[["beta_Medium"]], parameters[["beta_Poor"]])
  setNames(parameters[["scale_Good"]] * exp(beta), group_names)
}

close <- function(actual, expected) {
  stopifnot(
    length(actual) == length(expected),
    all(is.finite(actual)), all(is.finite(expected)),
    all(abs(actual - expected) <= 1e-10 + 1e-8 * abs(expected))
  )
}

survreg_total <- function(rows, mu, sigma) {
  # spec 4.6: offset fixes each log-scale location; the intercept stays at zero.
  fit <- survival::survreg(
    survival::Surv(time, event) ~ 1 + offset(mu),
    data = rows, dist = "weibull", init = 0, scale = sigma,
    control = survival::survreg.control(maxiter = 0)
  )
  close(unname(coef(fit)), 0)
  close(fit$scale, sigma)
  close(fit$linear.predictors, mu)
  close(fit$y[, "time"], rows$time)
  stopifnot(fit$iter == 0L, all(fit$y[, "status"] == rows$event))
  unname(fit$loglik[[2L]])
}

flex_values <- function(fit, times, groups, type) {
  predictions <- summary(
    fit, newdata = data.frame(group = factor(groups, levels = group_names)),
    t = times, type = type, cross = FALSE, ci = FALSE, tidy = TRUE
  )
  stopifnot(
    nrow(predictions) == length(times), identical(predictions$time, times),
    identical(as.character(predictions$group), as.character(groups))
  )
  predictions$est
}

# Synthetic observations and time-first grid -------------------------------

seed <- 8201L
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
group <- factor(rep(group_names, group_sizes), levels = group_names)
contrasts(group) <- contr.treatment(group_names, base = 1L)
event_time <- rweibull(length(group), cases$fitted[["shape"]], group_scales(cases$fitted)[group])
censor_time <- runif(length(group), min = 0.5, max = 7.5)
data <- data.frame(
  row_id = seq_along(group), time = pmin(event_time, censor_time),
  event = as.integer(event_time <= censor_time), group = group
)
stopifnot(
  nrow(data) == 686L, identical(data$row_id, seq_len(686L)),
  identical(as.integer(table(data$group)), unname(group_sizes)),
  identical(as.character(data$group), rep(group_names, group_sizes)),
  all(is.finite(data$time)), all(data$time > 0), setequal(data$event, c(0L, 1L))
)

eval_times <- seq(0, 7, by = 0.05)
grid_times <- rep(eval_times, length(group_names))
grid_groups <- rep(group_names, each = length(eval_times))
references <- list()

# Fixed-parameter references ----------------------------------------------

for (label in names(cases)) {
  parameters <- cases[[label]]
  scales <- group_scales(parameters)
  mu <- unname(log(scales[data$group]))
  sigma <- 1 / parameters[["shape"]]
  close(exp(mu), unname(scales[data$group]))
  close(mu, model.matrix(~ group, data) %*% c(log(parameters[["scale_Good"]]), parameters[3:4]))

  flex <- flexsurv::flexsurvreg(
    survival::Surv(time, event) ~ group, data = data, dist = "weibull",
    inits = unname(parameters), fixedpars = 1:4, hessian = FALSE
  )
  close(flex$res[, "est"], unname(parameters))
  close(flex$data$Y[, "time"], data$time)
  stopifnot(
    flex$npars == 0L, length(flex$logliki) == nrow(data),
    identical(rownames(flex$data$Y), as.character(data$row_id)),
    all(flex$data$Y[, "status"] == data$event)
  )

  flex_survival <- flex_values(flex, data$time, data$group, "survival")
  flex_log_survival <- -flex_values(flex, data$time, data$group, "cumhaz")
  flex_grid <- matrix(flex_values(flex, grid_times, grid_groups, "survival"), ncol = 3L)

  density <- survival::dsurvreg(data$time, mean = mu, scale = sigma)
  cdf <- survival::psurvreg(data$time, mean = mu, scale = sigma)
  stopifnot(all(density > 0), all(cdf > 0 & cdf < 1))
  log_survival <- log1p(-cdf)
  log_likelihood <- ifelse(data$event == 1L, log(density), log_survival)
  total <- survreg_total(data, mu, sigma)
  row_totals <- vapply(seq_len(nrow(data)), function(i) {
    survreg_total(data[i, , drop = FALSE], mu[i], sigma)
  }, numeric(1))
  survreg_grid <- matrix(1 - survival::psurvreg(
    grid_times, mean = unname(log(scales[grid_groups])), scale = sigma
  ), ncol = 3L)

  close(log_likelihood, row_totals)
  close(sum(log_likelihood), total)
  close(sum(flex$logliki), flex$loglik)
  close(flex$logliki, log_likelihood)
  close(flex$loglik, total)
  close(flex_log_survival, log_survival)
  close(flex_survival, 1 - cdf)
  close(flex_grid, survreg_grid)
  stopifnot(
    identical(dim(flex_grid), c(141L, 3L)), identical(dim(survreg_grid), c(141L, 3L)),
    all(flex_grid[1L, ] == 1), all(survreg_grid[1L, ] == 1)
  )

  references[[label]] <- c(as.list(parameters), list(
    scale_by_group = as.list(scales), mu_by_group = as.list(log(scales)), sigma = sigma,
    flexsurv = list(
      log_likelihood = unname(flex$logliki), total_log_likelihood = flex$loglik,
      log_survival = flex_log_survival, survival = flex_survival, survival_grid = flex_grid
    ),
    survreg = list(
      log_likelihood = log_likelihood, total_log_likelihood = total,
      log_survival = log_survival, survival = 1 - cdf, survival_grid = survreg_grid
    )
  ))
}

stopifnot(identical(names(references), names(cases)))
for (label in names(cases)) {
  stopifnot(identical(unlist(references[[label]][names(cases[[label]])]), cases[[label]]))
}
message("Checks passed: 686 synthetic rows, 3 fixed vectors, 141-by-3 grids, both R paths")
if (check_only) {
  quit(save = "no", status = 0L)
}

# Fixture export (the reviewed A8.1 format, plus time-first matrix rows) -----

to_json <- function(value) {
  if (is.matrix(value)) {
    rows <- lapply(seq_len(nrow(value)), function(i) value[i, ])
    stopifnot(identical(do.call(rbind, rows), value))
    value <- rows
  }
  if (is.list(value)) {
    fields <- vapply(value, to_json, character(1))
    if (is.null(names(value))) {
      return(paste0("[", paste(fields, collapse = ","), "]"))
    }
    keys <- encodeString(names(value), quote = '"')
    return(paste0("{", paste0(keys, ":", fields, collapse = ","), "}"))
  }
  if (is.numeric(value)) {
    stopifnot(length(value) > 0L, all(is.finite(value)))
    encoded <- sprintf("%.17g", value)
    expected <- trimws(formatC(value, digits = 17, format = "g"))
    stopifnot(identical(encoded, expected))
  } else {
    stopifnot(is.character(value))
    encoded <- encodeString(value, quote = '"')
  }
  if (length(encoded) == 1L) {
    return(encoded)
  }
  paste0("[", paste(encoded, collapse = ","), "]")
}

dir.create(output, recursive = TRUE, showWarnings = FALSE)
csv <- file.path(output, "data.csv")
csv_rows <- sprintf("%d,%.17g,%d,%s", data$row_id, data$time, data$event, as.character(data$group))
writeLines(c("row_id,time,event,group", csv_rows), csv, useBytes = TRUE)
expected_data <- as.data.frame(lapply(data, as.character))
expected_data$time <- trimws(formatC(data$time, digits = 17, format = "g"))
stopifnot(identical(read.csv(csv, colClasses = "character"), expected_data))

header <- list(
  schema_version = 1,
  seed = seed,
  rng = list(kind = "Mersenne-Twister", normal_kind = "Inversion", sample_kind = "Rejection"),
  r_version = as.character(getRversion()),
  packages = as.list(versions),
  platform = R.version$platform,
  generating_script = script,
  checksums = list(
    algorithm = "md5", data.csv = unname(tools::md5sum(csv)),
    generator = unname(tools::md5sum(script))
  ),
  units = "years",
  event_coding = list(exact = 1, right = 0),
  row_order = "Good/Medium/Poor blocks",
  simulation = list(parameters = as.list(cases$fitted), censoring = "Uniform(0.5,7.5) independent"),
  group_sizes = as.list(group_sizes),
  counts = list(exact = sum(data$event == 1L), right = sum(data$event == 0L)),
  events_by_group = as.list(setNames(as.integer(tapply(data$event, data$group, sum)), group_names)),
  parameter_convention = "b_g=b_Good*exp(beta_g); mu_g=log(b_g); sigma=1/shape",
  evaluation_paths = list(
    flexsurv = paste0(
      "flexsurvreg(dist=weibull,inits=(shape,b_Good,beta_Medium,beta_Poor),fixedpars=1:4); ",
      "logliki/loglik; summary(cross=FALSE,ci=FALSE,type=survival/cumhaz)"
    ),
    survreg = paste0(
      "dsurvreg/psurvreg(mean=mu,scale=sigma,distribution=weibull); ",
      "survreg(Surv~1+offset(mu),init=0,scale=sigma,maxiter=0)$loglik[2]; ",
      "full-data and singleton totals"
    )
  )
)
grid <- list(time = eval_times, groups = group_names, shape = c(141L, 3L))
payload <- to_json(list(header = header, row_id = data$row_id, grid = grid, cases = references))
json <- file.path(output, "expected.json")
writeLines(payload, json, useBytes = TRUE)
stopifnot(identical(readLines(json), payload))
message("Wrote data.csv and expected.json to ", output)
