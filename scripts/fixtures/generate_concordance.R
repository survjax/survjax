#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) <= 1L)
check_only <- identical(args, "--check-only")
output <- "tests/fixtures/concordance"
if (length(args) && !check_only) {
  if (startsWith(args[[1L]], "--")) stop("Unknown option: ", args[[1L]])
  output <- args[[1L]]
}
script <- "scripts/fixtures/generate_concordance.R"
versions <- c(survival = "3.7.0", flexsurv = "2.3.2")
stopifnot(file.exists(script), getRversion() == "4.4.2")
for (p in names(versions)) {
  stopifnot(as.character(utils::packageVersion(p)) == versions[[p]])
}
options(OutDec = ".", digits = 17)
seed <- 20261023L
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
# spec 9.2: leading rows guarantee all three time-tie kinds.
time <- c(1, 1, 1, 2, 2, 3, sample(1:20, 94, replace = TRUE))
event <- c(1L, 1L, 0L, 0L, 0L, 1L, rbinom(94, 1, 0.4))
cases <- list(
  list(name = "heavy_ties", time = time, event = event,
       scores = list(sample(-5:5, 100, TRUE), sample(-5:5, 100, TRUE), rep(1, 100))),
  list(name = "untied_high_censoring", time = c(0.25, 1.5, 2.75, 4),
       event = c(0L, 1L, 0L, 0L),
       scores = list(c(4, 3, 2, 1), c(1, 2, 3, 4), rep(1, 4)))
)
for (i in seq_along(cases)) {
  cohort <- cases[[i]]
  stopifnot(all(is.finite(cohort$time)), all(cohort$time > 0))
  cases[[i]]$r <- lapply(cohort$scores, function(score) {
    rows <- data.frame(time = cohort$time, event = cohort$event, score = score)
    fit <- survival::concordance(survival::Surv(time, event) ~ score, data = rows,
                                reverse = TRUE, timewt = "n", timefix = FALSE)
    counts <- unname(fit$count)
    stopifnot(all(is.finite(counts)), all(counts == floor(counts)),
              is.finite(fit$concordance), sum(counts[1:3]) > 0)
    list(concordance = unname(fit$concordance), concordant = counts[1],
         discordant = counts[2], tied_x = counts[3], tied_y = counts[4],
         tied_xy = counts[5])
  })
}
message("Checked 104 synthetic rows, two cohorts, three score draws each")
if (check_only) quit(save = "no", status = 0L)
# Fixed-schema base-R JSON, preserving arrays even for singleton lists.
to_json <- function(x) {
  if (is.list(x)) {
    values <- vapply(x, to_json, character(1))
    if (is.null(names(x))) return(paste0("[", paste(values, collapse = ","), "]"))
    fields <- paste0(encodeString(names(x), quote = '"'), ":", values)
    return(paste0("{", paste(fields, collapse = ","), "}"))
  }
  if (is.numeric(x)) {
    stopifnot(all(is.finite(x)))
    values <- sprintf("%.17g", x)
  } else values <- encodeString(x, quote = '"')
  if (length(values) == 1L) return(values)
  paste0("[", paste(values, collapse = ","), "]")
}
dir.create(output, recursive = TRUE, showWarnings = FALSE)
csv <- file.path(output, "data.csv")
rows <- unlist(lapply(cases, function(x) {
  sprintf("%s,%d,%.17g,%d,%.17g,%.17g,%.17g", x$name, seq_along(x$time),
          x$time, x$event, x$scores[[1]], x$scores[[2]], x$scores[[3]])
}))
writeLines(c("case,row_id,time,event,score1,score2,score3", rows), csv)
meta <- list(seed = seed, generator = script, R = as.character(getRversion()),
             survival = "3.7-0", flexsurv = "2.3.2", platform = R.version$platform,
             rng = "Mersenne-Twister/Inversion/Rejection", units = "years",
             call = paste0('survival::concordance(survival::Surv(time, event) ~ score, ',
                           'data = rows, reverse = TRUE, timewt = "n", timefix = FALSE)'),
             generator_md5 = unname(tools::md5sum(script)),
             csv_md5 = unname(tools::md5sum(csv)))
writeLines(to_json(list(meta = meta, cases = cases)),
           file.path(output, "concordance_synthetic.json"))
message("Wrote CSV and JSON to ", output)
