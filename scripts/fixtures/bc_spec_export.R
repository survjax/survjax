# spec 11.1: the independent tester defines this additional export schema.

bc_spec_fixture <- function(data, references, times, seed, script, versions) {
  case_names <- c(
    fixed_a = "unit_shape",
    fixed_b = "alternative",
    r_optimum = "fitted"
  )
  parameters <- list()
  results <- list()
  curves <- list()

  for (name in names(case_names)) {
    reference <- references[[case_names[[name]]]]
    parameters[[name]] <- list(
      shape = reference$shape,
      good_scale = reference$scale_Good,
      beta_medium = reference$beta_Medium,
      beta_poor = reference$beta_Poor
    )

    results[[name]] <- list()
    for (source in c("flexsurv", "survreg")) {
      values <- reference[[source]]
      stopifnot(
        length(values$log_likelihood) == nrow(data),
        length(values$survival) == nrow(data),
        all(is.finite(values$log_likelihood)),
        all(is.finite(values$survival))
      )

      results[[name]][[source]] <- list(
        loglik = values$log_likelihood,
        survival = values$survival
      )
    }

    curves[[name]] <- reference$flexsurv$survival_grid
    stopifnot(
      identical(dim(curves[[name]]), c(length(times), 3L)),
      all(is.finite(curves[[name]])),
      all(curves[[name]][1L, ] == 1)
    )
  }

  fitted_parameters <- references$fitted[c(
    "shape", "scale_Good", "beta_Medium", "beta_Poor"
  )]
  expected_groups <- rep(c("Good", "Medium", "Poor"), c(229L, 229L, 228L))

  # Check the fixed-vector labels independently of the case-name mapping.
  stopifnot(
    identical(unname(unlist(parameters$fixed_a)), c(1, 10, -0.5, -1)),
    identical(unname(unlist(parameters$fixed_b)), c(1.5, 12, -0.6, -1.2)),
    identical(
      unname(unlist(parameters$r_optimum)),
      unname(unlist(fitted_parameters))
    ),
    identical(data$row_id, seq_len(686L)),
    identical(as.character(data$group), expected_groups),
    all(is.finite(data$time)),
    all(data$time > 0),
    setequal(data$event, c(0L, 1L)),
    identical(names(parameters), c("fixed_a", "fixed_b", "r_optimum"))
  )

  fixture <- list(
    meta = list(
      seed = seed,
      generator = script,
      R = as.character(getRversion()),
      flexsurv = versions[["flexsurv"]],
      survival = "3.7-0"
    ),
    data = list(
      time = data$time,
      event = data$event,
      group = as.character(data$group)
    ),
    parameters = parameters,
    results = results,
    grid = list(time = times, survival = curves)
  )

  # spec 11.1: the assembled export must preserve the checked R values.
  # Independent grid-column orientation is checked in generate_bc_like.R.
  stopifnot(
    identical(fixture$data$time, data$time),
    identical(fixture$data$event, data$event),
    identical(fixture$data$group, as.character(data$group)),
    identical(fixture$grid$time, times)
  )

  for (name in names(case_names)) {
    reference <- references[[case_names[[name]]]]
    reference_parameters <- reference[c(
      "shape", "scale_Good", "beta_Medium", "beta_Poor"
    )]
    stopifnot(
      identical(
        unname(unlist(fixture$parameters[[name]])),
        unname(unlist(reference_parameters))
      ),
      identical(fixture$grid$survival[[name]], reference$flexsurv$survival_grid)
    )

    for (source in c("flexsurv", "survreg")) {
      values <- fixture$results[[name]][[source]]
      stopifnot(
        identical(values$loglik, reference[[source]]$log_likelihood),
        identical(values$survival, reference[[source]]$survival)
      )
    }
  }

  fixture
}
