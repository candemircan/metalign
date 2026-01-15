source("bin/utils.R")
library(mlogit, quietly = TRUE)
library(tidyverse, quietly = TRUE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript eval_thingso1o.R <experiment_name> <backbone_name>")
}

experiment_name <- args[1]
backbone_name <- args[2]

cat(sprintf("Running eval_thingso1o.R for %s %s\n", experiment_name, backbone_name))

stats_file <- sprintf("data/evals/thingso1o/%s_%s_stats.csv", experiment_name, backbone_name)
if (!file.exists(stats_file)) stop(sprintf("Stats file not found: %s", stats_file))

df <- read.csv(stats_file)
df <- df %>% mutate(trial_id = row_number())

is_main <- experiment_name == "MAIN"

if (!is_main) {
    main_stats_file <- sprintf("data/evals/thingso1o/MAIN_%s_stats.csv", backbone_name)
    if (!file.exists(main_stats_file)) stop("Main model stats file not found for ablation")

    df_main <- read.csv(main_stats_file)
    df$metalign_sim_main_0 <- df_main$metalign_sim_0
    df$metalign_sim_main_1 <- df_main$metalign_sim_1
    df$metalign_sim_main_2 <- df_main$metalign_sim_2
}

cols_to_pivot <- c(
    "base_sim_0", "base_sim_1", "base_sim_2",
    "metalign_sim_0", "metalign_sim_1", "metalign_sim_2"
)
if (!is_main) {
    cols_to_pivot <- c(cols_to_pivot, "metalign_sim_main_0", "metalign_sim_main_1", "metalign_sim_main_2")
}

# Reduce memory usage by selecting only necessary columns
cols_to_keep <- c(
    "subject_id", "trial_id", "y", "metalign_sim_0", "metalign_sim_1", "metalign_sim_2",
    "base_sim_0", "base_sim_1", "base_sim_2"
)
if (!is_main) {
    cols_to_keep <- c(cols_to_keep, "metalign_sim_main_0", "metalign_sim_main_1", "metalign_sim_main_2")
}
df <- df %>% select(all_of(cols_to_keep))
gc()

df_long <- df %>%
    pivot_longer(
        cols = any_of(gsub("_[0-9]$", "", cols_to_pivot) %>% paste0("_", 0:2)),
        names_to = c(".value", "alt"),
        names_pattern = "(.*)_([0-9])$"
    ) %>%
    mutate(
        alt = as.numeric(alt),
        is_choice = (y == alt),
        participant_id = as.factor(subject_id), # Note: thingso1o has subject_id
        trial_id = as.factor(trial_id)
    )

# Remove original df to free memory
rm(df)
gc()

# Explicit dummies (ASCs)
df_long$asc1 <- as.numeric(df_long$alt == 1)
df_long$asc2 <- as.numeric(df_long$alt == 2)

df_long$base_sim <- z_score(df_long$base_sim)
df_long$metalign_sim <- z_score(df_long$metalign_sim)
if (!is_main) {
    df_long$metalign_sim_main <- z_score(df_long$metalign_sim_main)
}

data_m <- mlogit.data(df_long,
    choice = "is_choice",
    shape = "long",
    alt.var = "alt",
    chid.var = "trial_id",
    id.var = "participant_id"
)

results <- list(
    experiment_name = experiment_name,
    backbone_name = backbone_name
)



if (is_main) {
    cat("Fitting Null Model (for R2)...\n")
    m_null <- mlogit(is_choice ~ asc1 + asc2 | 0,
        data = data_m,
        rpar = c(asc1 = "n", asc2 = "n"),
        panel = TRUE,
        iterlim = 2000
    )
    null_ll <- as.numeric(logLik(m_null))
    rm(m_null)
    gc()

    cat("Fitting Model 0 (Base only)...\n")
    m0 <- mlogit(is_choice ~ base_sim + asc1 + asc2 | 0,
        data = data_m,
        rpar = c(base_sim = "n", asc1 = "n", asc2 = "n"),
        panel = TRUE,
        iterlim = 2000
    )

    stats0 <- calculate_model_stats(m0, null_model_ll = null_ll)
    cat(sprintf("Model 0 BIC: %.2f\n", stats0$bic))
    rm(m0)
    gc()

    cat("Fitting Model 1 (Base + Metalign)...\n")
    m1 <- mlogit(is_choice ~ base_sim + metalign_sim + asc1 + asc2 | 0,
        data = data_m,
        rpar = c(base_sim = "n", metalign_sim = "n", asc1 = "n", asc2 = "n"),
        panel = TRUE,
        iterlim = 2000
    )

    stats1 <- calculate_model_stats(m1, null_model_ll = null_ll)
    cat(sprintf("Model 1 BIC: %.2f\n", stats1$bic))
    rm(m1)
    gc()

    log10_bf <- calculate_log10_bf(stats0$bic, stats1$bic)
    lrt <- calculate_lrt_from_stats(stats0, stats1)
    cat(sprintf("Log10 BF (M1/M0): %.2f, LRT p-value: %.3e\n", log10_bf, lrt$p_value))

    results$model_0 <- stats0
    results$model_1 <- stats1
    results$comparison_0_vs_1 <- list(
        log10_bayes_factor = log10_bf,
        lrt = lrt
    )
} else {
    cat("Fitting Null Model (for R2)...\n")
    m_null <- mlogit(is_choice ~ asc1 + asc2 | 0,
        data = data_m,
        rpar = c(asc1 = "n", asc2 = "n"),
        panel = TRUE,
        iterlim = 2000
    )
    null_ll <- as.numeric(logLik(m_null))
    rm(m_null)
    gc()

    cat("Fitting Model 1 (Base + Ablation)...\n")
    m1 <- mlogit(is_choice ~ base_sim + metalign_sim + asc1 + asc2 | 0,
        data = data_m,
        rpar = c(base_sim = "n", metalign_sim = "n", asc1 = "n", asc2 = "n"),
        panel = TRUE,
        iterlim = 2000
    )
    stats1 <- calculate_model_stats(m1, null_model_ll = null_ll)
    rm(m1)
    gc()

    cat("Fitting Model 2 (Base + Ablation + Main)...\n")
    m2 <- mlogit(is_choice ~ base_sim + metalign_sim + metalign_sim_main + asc1 + asc2 | 0,
        data = data_m,
        rpar = c(base_sim = "n", metalign_sim = "n", metalign_sim_main = "n", asc1 = "n", asc2 = "n"),
        panel = TRUE,
        iterlim = 2000
    )
    stats2 <- calculate_model_stats(m2, null_model_ll = null_ll)
    rm(m2)
    gc()

    log10_bf <- calculate_log10_bf(stats1$bic, stats2$bic)
    lrt <- calculate_lrt_from_stats(stats1, stats2)
    cat(sprintf("Log10 BF (M2/M1): %.2f, LRT p-value: %.3e\n", log10_bf, lrt$p_value))

    results$model_1 <- stats1
    results$model_2 <- stats2
    results$comparison_1_vs_2 <- list(
        log10_bayes_factor = log10_bf,
        lrt = lrt
    )
}

save_results(results, sprintf("data/evals/thingso1o/%s_%s_analysis_r.json", experiment_name, backbone_name))
