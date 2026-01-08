using CSV, DataFrames, MixedModels, StatsBase, CategoricalArrays, JSON

if length(ARGS) == 0
    error("Please provide the path to the stats CSV file as an argument.")
end

csv_file = ARGS[1]

data = CSV.read(csv_file, DataFrame)

# Convert categorical variables
data.participant = categorical(data.participant)
data.trial = categorical(data.trial)

# For binary choice, we model the probability of choosing category 1
# The choice variable should be 0 or 1
# Using logits (log-odds) as predictors

# Fit null model with only m1 (base model) logit
formula_null = @formula(choice ~ m1_logit_1 + (1|trial) + (1|participant))

model_null = fit(MixedModel, formula_null, data, Bernoulli(); fast=true)

println("=" ^ 80)
println("NULL MODEL (Base model only)")
println("=" ^ 80)
println(model_null)
println()

# Fit full model with both m1 and m2 (metalign model) logits
formula_full = @formula(choice ~ m1_logit_1 + m2_logit_1 + (1|trial) + (1|participant))

model_full = fit(MixedModel, formula_full, data, Bernoulli(); fast=true)

println("=" ^ 80)
println("FULL MODEL (Base + Metalign)")
println("=" ^ 80)
println(model_full)
println()

# Likelihood ratio test
lrt = MixedModels.likelihoodratiotest(model_null, model_full)
println("=" ^ 80)
println("LIKELIHOOD RATIO TEST")
println("=" ^ 80)
println(lrt)
println()

# Print model comparison metrics
println("=" ^ 80)
println("MODEL COMPARISON")
println("=" ^ 80)
println("Null Model - AIC: ", round(aic(model_null), digits=2), " BIC: ", round(bic(model_null), digits=2))
println("Full Model - AIC: ", round(aic(model_full), digits=2), " BIC: ", round(bic(model_full), digits=2))
println()

# --- Bayesian Evidence Approximation ---
delta_bic = bic(model_null) - bic(model_full)
# BIC-based approximation of Bayes Factor: BF10 ≈ exp(delta_bic / 2)
# With very large delta_bic, we use log10 to avoid overflow
log10_bf = (delta_bic / 2) / log(10)

println("=" ^ 80)
println("BAYESIAN MODEL COMPARISON (BIC Approximation)")
println("=" ^ 80)
println("Delta BIC: ", round(delta_bic, digits=2))
println("Log10(Bayes Factor B10): ~", round(log10_bf, digits=2))
println()

# Extract and save key metrics
results = Dict(
    "null_model" => Dict(
        "aic" => aic(model_null),
        "bic" => bic(model_null),
        "loglikelihood" => loglikelihood(model_null),
        "fixed_effects" => Dict(
            "intercept" => coef(model_null)[1],
            "m1_logit_1" => coef(model_null)[2]
        )
    ),
    "full_model" => Dict(
        "aic" => aic(model_full),
        "bic" => bic(model_full),
        "loglikelihood" => loglikelihood(model_full),
        "fixed_effects" => Dict(
            "intercept" => coef(model_full)[1],
            "m1_logit_1" => coef(model_full)[2],
            "m2_logit_1" => coef(model_full)[3]
        )
    ),
    "likelihood_ratio_test" => Dict(
        "deviance" => lrt.deviance[2],
        "dof" => lrt.dof[2],
        "pvalue" => lrt.pval[2]
    ),
    "model_comparison" => Dict(
        "delta_aic" => aic(model_null) - aic(model_full),
        "delta_bic" => delta_bic,
        "log10_bayes_factor" => log10_bf
    )
)

# Save results to JSON file
output_file = replace(csv_file, "_stats.csv" => "_analysis.json")
open(output_file, "w") do f
    JSON.print(f, results, 4)
end
