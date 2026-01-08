using CSV, DataFrames, MixedModels, StatsBase, CategoricalArrays, JSON

if length(ARGS) == 0
    error("Please provide the path to the stats CSV file as an argument.")
end

csv_file = ARGS[1]

data = CSV.read(csv_file, DataFrame)

# one way to model this dataset is as follows
# 
data_long = DataFrame()
for choice_idx in 0:2
    tmp = select(data, :subject_id, :y)
    tmp.choice_idx = fill(choice_idx, nrow(data))
    tmp.is_choice = Int.(data.y .== choice_idx)
    tmp.m1_sim = data[:, Symbol("m1_sim_$choice_idx")]
    tmp.m2_sim = data[:, Symbol("m2_sim_$choice_idx")]
    tmp.triplet_id = 1:nrow(data)
    append!(data_long, tmp)
end

data_long.subject_id = categorical(data_long.subject_id)
data_long.triplet_id = categorical(data_long.triplet_id)


formula_null = @formula(is_choice ~ m1_sim + (1|triplet_id) + (1|subject_id))

model_null = fit(MixedModel, formula_null, data_long, Bernoulli(); fast=true)

println("=" ^ 80)
println("NULL MODEL (Base model only)")
println("=" ^ 80)
println(model_null)
println()

formula_full = @formula(is_choice ~ m1_sim + m2_sim + (1|triplet_id) + (1|subject_id))

model_full = fit(MixedModel, formula_full, data_long, Bernoulli(); fast=true)

println("=" ^ 80)
println("FULL MODEL (Base + Metalign)")
println("=" ^ 80)
println(model_full)
println()


lrt = MixedModels.likelihoodratiotest(model_null, model_full)
println("=" ^ 80)
println("LIKELIHOOD RATIO TEST")
println("=" ^ 80)
println(lrt)
println()


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
            "m1_sim" => coef(model_null)[2]
        )
    ),
    "full_model" => Dict(
        "aic" => aic(model_full),
        "bic" => bic(model_full),
        "loglikelihood" => loglikelihood(model_full),
        "fixed_effects" => Dict(
            "intercept" => coef(model_full)[1],
            "m1_sim" => coef(model_full)[2],
            "m2_sim" => coef(model_full)[3]
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

println("=" ^ 80)
println("Results saved to: ", output_file)
println("=" ^ 80)