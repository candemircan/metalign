library(jsonlite)

z_score <- function(x) {
    if (all(is.na(x))) {
        return(x)
    }
    (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}


calculate_model_stats <- function(model, null_model_ll = NULL) {
    ll <- as.numeric(logLik(model))
    k <- attr(logLik(model), "df")
    if (is.null(k)) k <- length(coef(model))

    aic_val <- AIC(model)
    bic_val <- BIC(model)
    nll <- -ll

    if (inherits(model, "mlogit")) {
        # mlogit often fails with standard BIC() (returns NA) because nobs is missing/ambiguous
        # Manual BIC: -2*LL + k*log(n_obs)
        # n_obs is the number of choices made.
        # In our scripts, response is a logical/binary vector indicating choice.
        # sum(response) gives number of choice events (N).
        mf <- model.frame(model)
        y <- model.response(mf)
        # Ensure y is numeric/logical for summation
        n_obs <- sum(as.numeric(y))

        # Recalculate BIC if standard failed
        if (is.na(bic_val) || is.null(bic_val)) {
            bic_val <- -2 * ll + k * log(n_obs)
        }

        # McFadden R2 from summary
        s <- summary(model)
        if (!is.null(s$mfR2)) {
            r2 <- as.numeric(s$mfR2)
        } else if (!is.null(null_model_ll)) {
            r2 <- 1 - (ll / null_model_ll)
        } else {
            r2 <- NA
        }
    } else {
        # For other models (e.g. glmer)
        if (!is.null(null_model_ll)) {
            # Manual McFadden R2: 1 - LL_mod / LL_null
            r2 <- 1 - (ll / null_model_ll)
        } else {
            r2 <- NA
        }
    }

    return(list(
        aic = aic_val,
        bic = bic_val,
        nll = nll,
        r2 = r2,
        k = k,
        loglik = ll,
        log_evidence = -0.5 * bic_val,
        log_joint_map = ll
    ))
}

calculate_log10_bf <- function(bic_h0, bic_h1) {
    delta_bic <- bic_h0 - bic_h1
    log10_bf <- delta_bic / (2 * log(10))
    return(log10_bf)
}

calculate_lrt <- function(m_restricted, m_full) {
    ll_r <- as.numeric(logLik(m_restricted))
    ll_f <- as.numeric(logLik(m_full))

    df_r <- attr(logLik(m_restricted), "df")
    df_f <- attr(logLik(m_full), "df")

    # Fallback if attribute is missing
    if (is.null(df_r)) df_r <- length(coef(m_restricted))
    if (is.null(df_f)) df_f <- length(coef(m_full))

    chisq_stat <- 2 * (ll_f - ll_r)
    df_diff <- df_f - df_r

    p_val <- NA
    if (df_diff > 0) {
        p_val <- pchisq(chisq_stat, df_diff, lower.tail = FALSE)
    }

    return(list(
        chisq = chisq_stat,
        df = df_diff,
        p_value = p_val
    ))
}

calculate_lrt_from_stats <- function(stats_r, stats_f) {
    ll_r <- stats_r$loglik
    ll_f <- stats_f$loglik

    df_r <- stats_r$k
    df_f <- stats_f$k

    chisq_stat <- 2 * (ll_f - ll_r)
    df_diff <- df_f - df_r

    p_val <- NA
    if (df_diff > 0) {
        p_val <- pchisq(chisq_stat, df_diff, lower.tail = FALSE)
    }

    return(list(
        chisq = chisq_stat,
        df = df_diff,
        p_value = p_val
    ))
}

save_results <- function(data, filename) {
    write_json(data, filename, pretty = TRUE, auto_unbox = TRUE)
    cat(sprintf("Results saved to %s\n", filename))
}
