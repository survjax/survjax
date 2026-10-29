#!/usr/bin/env Rscript
# spec 10.3/10.4: pinned R references, synthetic rows and fixed parameters only.

# Command-line options -----------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) <= 1L)

check_only <- identical(args, "--check-only")
output <- "tests/fixtures/exponential"
if (length(args) && !check_only) {
  if (startsWith(args[[1L]], "--")) {
    stop("Unknown option: ", args[[1L]])
  }

  output <- args[[1L]]
}

# Reference environment ----------------------------------------------------

script <- "scripts/fixtures/generate_exponential.R"
stopifnot(file.exists(script), getRversion() == "4.4.2")

versions <- c(flexsurv = "2.3.2", survival = "3.7.0")
for (name in names(versions)) {
  # Version checks precede loading, so an unpinned installation cannot emit data.
  stopifnot(as.character(utils::packageVersion(name)) == versions[[name]])
}

suppressPackageStartupMessages(library(flexsurv))
options(OutDec = ".", digits = 17)

# Synthetic observations --------------------------------------------------

seed <- 8101L
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)

n <- 16L
generating_rate <- 0.35
event_time <- rexp(n, rate = generating_rate)
censor_time <- runif(n, min = 0.5, max = 5)

data <- data.frame(
  row_id = seq_len(n),
  time = pmin(event_time, censor_time),
  event = as.integer(event_time <= censor_time)
)

stopifnot(
  identical(data$row_id, seq_len(n)),
  all(is.finite(data$time)),
  all(data$time > 0),
  setequal(data$event, c(0L, 1L))
)

# Fixed-parameter reference checks ----------------------------------------

# These are generator comparisons in R double precision, not Python test tolerances.
close <- function(actual, expected) {
  stopifnot(
    length(actual) == length(expected),
    all(is.finite(actual)),
    all(is.finite(expected)),
    all(abs(actual - expected) <= 1e-10 + 1e-8 * abs(expected))
  )
}

# Installed ?survreg and ?survreg.control: explicit init, no Newton iterations.
evaluate_survreg <- function(rows, mu) {
  fit <- survival::survreg(
    survival::Surv(time, event) ~ 1,
    data = rows,
    dist = "exponential",
    init = mu,
    control = survival::survreg.control(maxiter = 0)
  )

  close(unname(coef(fit)), mu)
  close(fit$scale, 1)
  stopifnot(fit$iter == 0L, nrow(fit$y) == nrow(rows))
  close(fit$y[, "time"], rows$time)
  stopifnot(all(fit$y[, "status"] == rows$event))

  unname(fit$loglik[[2L]])
}

references <- list()
rates <- c(slow = 0.1, generating = generating_rate, fast = 1.2)

for (label in names(rates)) {
  rate <- unname(rates[[label]])
  mu <- -log(rate) # spec 4.6: survreg location is log(mean), rate = exp(-mu).

  # flexsurv likelihood and observed-time predictions
  flex <- flexsurv::flexsurvreg(
    survival::Surv(time, event) ~ 1,
    data = data,
    dist = "exp",
    inits = rate,
    fixedpars = 1,
    hessian = FALSE
  )

  close(flex$res["rate", "est"], rate)
  stopifnot(
    flex$npars == 0L,
    length(flex$logliki) == n,
    identical(rownames(flex$data$Y), as.character(data$row_id))
  )
  close(flex$data$Y[, "time"], data$time)
  stopifnot(all(flex$data$Y[, "status"] == data$event))
  close(sum(flex$logliki), flex$loglik)

  surv <- summary(flex, t = data$time, type = "survival", ci = FALSE)[[1L]]
  cumhaz <- summary(flex, t = data$time, type = "cumhaz", ci = FALSE)[[1L]]

  # Match by time explicitly; never rely on summary's presentation order.
  si <- match(data$time, surv$time)
  hi <- match(data$time, cumhaz$time)
  stopifnot(
    nrow(surv) == n, nrow(cumhaz) == n,
    !anyNA(si), !anyNA(hi),
    !anyDuplicated(si), !anyDuplicated(hi)
  )

  flex_survival <- surv$est[si]
  flex_log_survival <- -cumhaz$est[hi]
  close(exp(flex_log_survival), flex_survival)

  # survreg density/CDF contributions and fixed-parameter totals
  total <- evaluate_survreg(data, mu)
  density <- survival::dsurvreg(
    data$time, mean = mu, scale = 1, distribution = "exponential"
  )
  cdf <- survival::psurvreg(
    data$time, mean = mu, scale = 1, distribution = "exponential"
  )

  stopifnot(
    length(density) == n, length(cdf) == n,
    all(density > 0),
    all(cdf > 0 & cdf < 1),
    all(flex_survival > 0 & flex_survival < 1)
  )

  log_survival <- log1p(-cdf)
  log_likelihood <- ifelse(data$event == 1L, log(density), log_survival)
  row_totals <- vapply(seq_len(n), function(i) {
    evaluate_survreg(data[i, , drop = FALSE], mu)
  }, numeric(1))

  close(log_likelihood, row_totals)
  close(sum(log_likelihood), total)

  # Agreement of the two reference paths
  close(flex$logliki, log_likelihood)
  close(flex$loglik, total)
  close(flex_log_survival, log_survival)
  close(flex_survival, 1 - cdf)

  references[[label]] <- list(
    rate = rate,
    mean = 1 / rate,
    log_rate = log(rate),
    log_mean = mu,
    survreg_scale = 1,
    flexsurv = list(
      log_likelihood = unname(flex$logliki),
      total_log_likelihood = flex$loglik,
      log_survival = flex_log_survival,
      survival = flex_survival
    ),
    survreg = list(
      log_likelihood = log_likelihood,
      total_log_likelihood = total,
      log_survival = log_survival,
      survival = 1 - cdf
    )
  )
}

stopifnot(identical(names(references), names(rates)))
for (label in names(rates)) {
  stopifnot(identical(references[[label]]$rate, unname(rates[[label]])))
}

message("Checks passed: 16 synthetic rows, 3 fixed rates, both R reference paths")
if (check_only) {
  quit(save = "no", status = 0L)
}

# Fixture export ----------------------------------------------------------

# Fixed-schema base-R export; 17 significant digits round-trip R doubles.
to_json <- function(value) {
  if (is.list(value)) {
    fields <- vapply(value, to_json, character(1))
    keys <- encodeString(names(value), quote = '"')
    return(paste0("{", paste0(keys, ":", fields, collapse = ","), "}"))
  }

  if (is.numeric(value)) {
    stopifnot(length(value) > 0L, all(is.finite(value)))
    encoded <- sprintf("%.17g", value)
    # The pinned macOS R decimal reader can lose an ULP; check the text instead.
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
csv_rows <- sprintf("%d,%.17g,%d", data$row_id, data$time, data$event)
writeLines(c("row_id,time,event", csv_rows), csv, useBytes = TRUE)

expected_data <- data.frame(
  row_id = as.character(data$row_id),
  time = trimws(formatC(data$time, digits = 17, format = "g")),
  event = as.character(data$event)
)
stopifnot(identical(read.csv(csv, colClasses = "character"), expected_data))

header <- list(
  schema_version = 1,
  seed = seed,
  rng = list(
    kind = "Mersenne-Twister",
    normal_kind = "Inversion",
    sample_kind = "Rejection"
  ),
  r_version = as.character(getRversion()),
  packages = as.list(versions),
  platform = R.version$platform,
  generating_script = script,
  checksums = list(
    algorithm = "md5",
    data.csv = unname(tools::md5sum(csv)),
    generator = unname(tools::md5sum(script))
  ),
  units = "years",
  event_coding = list(exact = 1, right = 0),
  row_order = "simulation order",
  simulation = list(
    n = n,
    event_rate = generating_rate,
    censoring = "Uniform(0.5,5) independent"
  ),
  counts = list(exact = sum(data$event == 1L), right = sum(data$event == 0L)),
  parameter_convention = "rate=1/mean=exp(-mu); mu=log(mean); sigma=1",
  evaluation_paths = list(
    flexsurv = paste0(
      "flexsurvreg(dist=exp,inits=rate,fixedpars=1); ",
      "logliki/loglik; summary(type=survival/cumhaz,ci=FALSE)"
    ),
    survreg = paste0(
      "dsurvreg/psurvreg(distribution=exponential,mean=mu,scale=1); ",
      "survreg(dist=exponential,init=mu,control=survreg.control(maxiter=0))$loglik[2]; ",
      "full-data and singleton totals"
    )
  )
)

payload <- to_json(list(header = header, row_id = data$row_id, cases = references))
json <- file.path(output, "expected.json")
writeLines(payload, json, useBytes = TRUE)
stopifnot(identical(readLines(json), payload))

message("Wrote data.csv and expected.json to ", output)
