source("bin/utils.R")
library(lme4, quietly = TRUE)
library(tidyverse, quietly = TRUE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript eval_categorylearning.R <experiment_name> <backbone_name>")
}

experiment_name <- args[1]
backbone_name <- args[2]

cat(sprintf("Running eval_categorylearning.R for %s %s\n", experiment_name, backbone_name))

stats_file <- sprintf("data/evals/categorylearning/%s_%s_stats.csv", experiment_name, backbone_name)
if (!file.exists(stats_file)) stop(sprintf("Stats file not found: %s", stats_file))

df <- read.csv(stats_file)
is_main <- experiment_name == "MAIN"

if (!is_main) {
    main_stats_file <- sprintf("data/evals/categorylearning/MAIN_%s_stats.csv", backbone_name)
    if (!file.exists(main_stats_file)) stop("Main model stats file not found for ablation")
    df_main <- read.csv(main_stats_file)

    # Assuming alignment
    df$main_logit_1 <- df_main$metalign_logit_1
}

# Z-score predictors
df$base_logit_1 <- z_score(df$base_logit_1)
if ("metalign_logit_1" %in% names(df)) df$metalign_logit_1 <- z_score(df$metalign_logit_1)
if ("main_logit_1" %in% names(df)) df$main_logit_1 <- z_score(df$main_logit_1)

df$participant <- as.factor(df$participant)

results <- list(
    experiment_name = experiment_name,
    backbone_name = backbone_name
)

# Fit Null model for McFadden R2
cat("Fitting Null Model (for R2)...\n")
m_null <- glmer(choice ~ 1 + (1 | participant), data = df, family = binomial)
null_ll <- as.numeric(logLik(m_null))

if (is_main) {
    cat("Fitting Model 0 (Base only)...\n")
    m0 <- glmer(choice ~ base_logit_1 + (1 + base_logit_1 | participant),
        data = df, family = binomial,
    )
    stats0 <- calculate_model_stats(m0, null_model_ll = null_ll)

    cat("Fitting Model 1 (Base + Metalign)...\n")
    m1 <- glmer(choice ~ base_logit_1 + metalign_logit_1 + (1 + base_logit_1 + metalign_logit_1 | participant),
        data = df, family = binomial,
    )
    stats1 <- calculate_model_stats(m1, null_model_ll = null_ll)

    log10_bf <- calculate_log10_bf(stats0$bic, stats1$bic)
    lrt <- calculate_lrt(m0, m1)
    cat(sprintf("BIC0: %.2f, BIC1: %.2f, Log10 BF: %.2f, LRT p-value: %.3e\n", stats0$bic, stats1$bic, log10_bf, lrt$p_value))

    results$model_0 <- stats0
    results$model_1 <- stats1
    results$comparison_0_vs_1 <- list(
        log10_bayes_factor = log10_bf,
        lrt = lrt
    )
} else {
    cat("Fitting Model 1 (Base + Ablation)...\n")
    m1 <- glmer(choice ~ base_logit_1 + metalign_logit_1 + (1 + base_logit_1 + metalign_logit_1 | participant),
        data = df, family = binomial,
    )
    stats1 <- calculate_model_stats(m1, null_model_ll = null_ll)

    cat("Fitting Model 2 (Base + Ablation + Main)...\n")
    m2 <- glmer(choice ~ base_logit_1 + metalign_logit_1 + main_logit_1 + (1 + base_logit_1 + metalign_logit_1 + main_logit_1 | participant),
        data = df, family = binomial,
    )
    stats2 <- calculate_model_stats(m2, null_model_ll = null_ll)

    log10_bf <- calculate_log10_bf(stats1$bic, stats2$bic)
    lrt <- calculate_lrt(m1, m2)
    cat(sprintf("BIC1: %.2f, BIC2: %.2f, Log10 BF: %.2f, LRT p-value: %.3e\n", stats1$bic, stats2$bic, log10_bf, lrt$p_value))

    results$model_1 <- stats1
    results$model_2 <- stats2
    results$comparison_1_vs_2 <- list(
        log10_bayes_factor = log10_bf,
        lrt = lrt
    )
}

save_results(results, sprintf("data/evals/categorylearning/%s_%s_analysis_r.json", experiment_name, backbone_name))
