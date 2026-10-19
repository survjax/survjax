#!/usr/bin/env Rscript
# spec 10.3/10.4: pinned R references, synthetic rows and fixed parameters only.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) <= 1L)
check_only <- identical(args, "--check-only")
output <- if (length(args) && !check_only) args[[1L]] else "tests/fixtures/exponential"
script <- "scripts/fixtures/generate_exponential.R"
stopifnot(file.exists(script), getRversion() == "4.4.2")
versions <- c(flexsurv = "2.3.2", survival = "3.7.0")
for (name in names(versions)) {
  # Version checks precede loading, so an unpinned installation cannot emit data.
  stopifnot(as.character(utils::packageVersion(name)) == versions[[name]])
}
suppressPackageStartupMessages(library(flexsurv))
options(OutDec = ".", digits = 17)
seed <- 8101L
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
n <- 16L
event_time <- rexp(n, rate = 0.35)
censor_time <- runif(n, min = 0.5, max = 5)
data <- data.frame(row_id = seq_len(n), time = pmin(event_time, censor_time),
                   event = as.integer(event_time <= censor_time))
stopifnot(identical(data$row_id, seq_len(n)), all(is.finite(data$time)),
          all(data$time > 0), setequal(data$event, c(0L, 1L)))

# These are generator comparisons in R double precision, not Python test tolerances.
close <- function(actual, expected) {
  stopifnot(length(actual) == length(expected), all(is.finite(actual)),
            all(is.finite(expected)),
            all(abs(actual - expected) <= 1e-10 + 1e-8 * abs(expected)))
}
references <- list()
rates <- c(slow = 0.1, generating = 0.35, fast = 1.2)
for (label in names(rates)) {
  rate <- unname(rates[[label]])
  mu <- -log(rate) # spec 4.6: survreg location is log(mean), rate = exp(-mu).
  flex <- flexsurv::flexsurvreg(survival::Surv(time, event) ~ 1,
    data = data, dist = "exp", inits = rate, fixedpars = 1, hessian = FALSE)
  close(flex$res["rate", "est"], rate)
  stopifnot(flex$npars == 0L, length(flex$logliki) == n,
            identical(rownames(flex$data$Y), as.character(data$row_id)))
  close(flex$data$Y[, "time"], data$time)
  stopifnot(all(flex$data$Y[, "status"] == data$event))
  close(sum(flex$logliki), flex$loglik)
  surv <- summary(flex, t = data$time, type = "survival", ci = FALSE)[[1L]]
  cumhaz <- summary(flex, t = data$time, type = "cumhaz", ci = FALSE)[[1L]]
  # Match by time explicitly; never rely on summary's presentation order.
  si <- match(data$time, surv$time)
  hi <- match(data$time, cumhaz$time)
  stopifnot(nrow(surv) == n, nrow(cumhaz) == n, !anyNA(si), !anyNA(hi),
            !anyDuplicated(si), !anyDuplicated(hi))
  flex_survival <- surv$est[si]
  flex_log_survival <- -cumhaz$est[hi]
  close(exp(flex_log_survival), flex_survival)

  # Installed ?survreg and ?survreg.control: explicit init, no Newton iterations.
  evaluate_survreg <- function(rows) {
    fit <- survival::survreg(survival::Surv(time, event) ~ 1,
      data = rows, dist = "exponential", init = mu,
      control = survival::survreg.control(maxiter = 0))
    close(unname(coef(fit)), mu)
    close(fit$scale, 1)
    stopifnot(fit$iter == 0L, nrow(fit$y) == nrow(rows))
    close(fit$y[, "time"], rows$time)
    stopifnot(all(fit$y[, "status"] == rows$event))
    unname(fit$loglik[[2L]])
  }
  total <- evaluate_survreg(data)
  density <- survival::dsurvreg(data$time, mean = mu, scale = 1,
                                distribution = "exponential")
  cdf <- survival::psurvreg(data$time, mean = mu, scale = 1,
                            distribution = "exponential")
  stopifnot(length(density) == n, length(cdf) == n, all(density > 0),
            all(cdf > 0 & cdf < 1), all(flex_survival > 0 & flex_survival < 1))
  log_survival <- log1p(-cdf)
  log_likelihood <- ifelse(data$event == 1L, log(density), log_survival)
  row_totals <- vapply(seq_len(n), function(i) {
    evaluate_survreg(data[i, , drop = FALSE])
  }, numeric(1))
  close(log_likelihood, row_totals)
  close(sum(log_likelihood), total)
  close(flex$logliki, log_likelihood)
  close(flex$loglik, total)
  close(flex_log_survival, log_survival)
  close(flex_survival, 1 - cdf)
  references[[label]] <- list(rate = rate, mu = mu,
    flexsurv = list(log_likelihood = unname(flex$logliki),
      total_log_likelihood = flex$loglik, log_survival = flex_log_survival,
      survival = flex_survival),
    survreg = list(log_likelihood = log_likelihood, total_log_likelihood = total,
      log_survival = log_survival, survival = 1 - cdf))
}
message("Checks passed: 16 synthetic rows, 3 fixed rates, both R reference paths")
if (check_only) quit(save = "no", status = 0L)

# Fixed-schema base-R export; 17 significant digits round-trip R doubles.
number <- function(x) {
  stopifnot(is.numeric(x), length(x) > 0L, all(is.finite(x)))
  sprintf("%.17g", x)
}
vector_json <- function(x) paste0("[", paste(number(x), collapse = ","), "]")
string <- function(x) encodeString(x, quote = '"')
reference_json <- function(x) paste0(
  '{"log_likelihood":', vector_json(x$log_likelihood),
  ',"total_log_likelihood":', number(x$total_log_likelihood),
  ',"log_survival":', vector_json(x$log_survival),
  ',"survival":', vector_json(x$survival), '}')
cases <- vapply(references, function(x) paste0(
  '{"rate":', number(x$rate), ',"mean":', number(1 / x$rate),
  ',"log_rate":', number(log(x$rate)), ',"log_mean":', number(x$mu),
  ',"survreg_scale":1,"flexsurv":', reference_json(x$flexsurv),
  ',"survreg":', reference_json(x$survreg), '}'), character(1))
dir.create(output, recursive = TRUE, showWarnings = FALSE)
csv <- file.path(output, "data.csv")
writeLines(c("row_id,time,event", sprintf("%d,%.17g,%d",
  data$row_id, data$time, data$event)), csv, useBytes = TRUE)
payload <- paste0(
  '{"header":{"schema_version":1,"seed":', seed,
  ',"rng":{"kind":"Mersenne-Twister","normal_kind":"Inversion",',
  '"sample_kind":"Rejection"},"r_version":', string(as.character(getRversion())),
  ',"packages":{"flexsurv":"2.3.2","survival":"3.7.0"},"platform":',
  string(R.version$platform), ',"generating_script":', string(script),
  ',"checksums":{"algorithm":"md5","data.csv":',
  string(unname(tools::md5sum(csv))), ',"generator":',
  string(unname(tools::md5sum(script))), '},"units":"years",',
  '"event_coding":{"exact":1,"right":0},"row_order":"simulation order",',
  '"simulation":{"n":16,"event_rate":0.35,"censoring":"Uniform(0.5,5) independent"},',
  '"counts":{"exact":', sum(data$event == 1L), ',"right":', sum(data$event == 0L),
  '},"parameter_convention":"rate=1/mean=exp(-mu); mu=log(mean); sigma=1",',
  '"evaluation_paths":{"flexsurv":"flexsurvreg(dist=exp,inits=rate,fixedpars=1); ',
  'logliki/loglik; summary(type=survival/cumhaz,ci=FALSE)",',
  '"survreg":"dsurvreg/psurvreg(distribution=exponential,mean=mu,scale=1); ',
  'survreg(dist=exponential,init=mu,control=survreg.control(maxiter=0))$loglik[2]; ',
  'full-data and singleton totals"}},"row_id":', vector_json(data$row_id),
  ',"cases":{', paste0(string(names(cases)), ':', cases, collapse = ","), '}}')
writeLines(payload, file.path(output, "expected.json"), useBytes = TRUE)
message("Wrote data.csv and expected.json to ", output)
