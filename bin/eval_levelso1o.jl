using CSV, DataFrames, MixedModels, StatsBase, CategoricalArrays, JSON

if length(ARGS) < 2
    error("Usage: julia eval_levelso1o.jl <experiment_name> <backbone_name>")
end

experiment_name = ARGS[1]
backbone_name = ARGS[2]
csv_file = "data/evals/levelso1o/$(experiment_name)_$(backbone_name)_stats.csv"

if !isfile(csv_file)
    error("Stats file not found: $csv_file")
end

data = CSV.read(csv_file, DataFrame)

# Reshape from wide to long format
data_long = DataFrame()
for choice_idx in 0:2
    tmp = select(data, :participant_id, :y)
    tmp.choice_idx = fill(choice_idx, nrow(data))
    tmp.is_choice = Int.(data.y .== choice_idx)
    tmp.base_sim = data[:, Symbol("base_sim_$choice_idx")]
    tmp.ablation_sim = data[:, Symbol("metalign_sim_$choice_idx")]
    tmp.triplet_id = 1:nrow(data)
    append!(data_long, tmp)
end

data_long.participant_id = categorical(data_long.participant_id)
data_long.triplet_id = categorical(data_long.triplet_id)

# Determine if this is main or an ablation
is_main = experiment_name == "MAIN"

# If ablation, load the main model data to get full metalign
if !is_main
    main_csv_file = "data/evals/levelso1o/MAIN_$(backbone_name)_stats.csv"
    if !isfile(main_csv_file)
        error("Main model stats file not found: $main_csv_file")
    end
    main_data = CSV.read(main_csv_file, DataFrame)
    
    # Add main metalign similarities to long format data
    main_long = DataFrame()
    for choice_idx in 0:2
        tmp = DataFrame(
            triplet_id = 1:nrow(main_data),
            choice_idx = fill(choice_idx, nrow(main_data)),
            main_sim = main_data[:, Symbol("metalign_sim_$choice_idx")]
        )
        append!(main_long, tmp)
    end
    
    # Merge with ablation data
    data_long = leftjoin(data_long, main_long, on=[:triplet_id, :choice_idx])
end

# Normalize predictors to help convergence
dropmissing!(data_long)
data_long.base_sim = zscore(data_long.base_sim)
data_long.ablation_sim = zscore(data_long.ablation_sim)
if "main_sim" in names(data_long)
    data_long.main_sim = zscore(data_long.main_sim)
end

if is_main
    # Model 0: Base only
    formula_0 = @formula(is_choice ~ base_sim + (1 + base_sim | participant_id))
    model_0 = fit(MixedModel, formula_0, data_long, Bernoulli())
    
    println("=" ^ 80)
    println("MODEL 0: Base only")
    println("=" ^ 80)
    println(model_0)
    println()
    # Main model: only compare M0 vs M1
    formula_1 = @formula(is_choice ~ base_sim + ablation_sim + (1 + base_sim + ablation_sim | participant_id))
    model_1 = fit(MixedModel, formula_1, data_long, Bernoulli())
    
    println("=" ^ 80)
    println("MODEL 1: Base + Full Metalign")
    println("=" ^ 80)
    println(model_1)
    println()
    
    lrt = MixedModels.likelihoodratiotest(model_0, model_1)
    println("=" ^ 80)
    println("LIKELIHOOD RATIO TEST")
    println("=" ^ 80)
    println(lrt)
    println()
    
    println("=" ^ 80)
    println("MODEL COMPARISON")
    println("=" ^ 80)
    println("Model 0 - AIC: ", round(aic(model_0), digits=2), " BIC: ", round(bic(model_0), digits=2))
    println("Model 1 - AIC: ", round(aic(model_1), digits=2), " BIC: ", round(bic(model_1), digits=2))
    println()
    
    delta_bic = bic(model_0) - bic(model_1)
    log10_bf = (delta_bic / 2) / log(10)
    
    println("=" ^ 80)
    println("BAYESIAN MODEL COMPARISON (BIC Approximation)")
    println("=" ^ 80)
    println("Delta BIC: ", round(delta_bic, digits=2))
    println("Log10(Bayes Factor B10): ~", round(log10_bf, digits=2))
    println()
    
    # Save results
    results = Dict(
        "experiment_name" => experiment_name,
        "backbone_name" => backbone_name,
        "is_main_model" => true,
        "model_0" => Dict(
            "aic" => aic(model_0),
            "bic" => bic(model_0),
            "loglikelihood" => loglikelihood(model_0),
            "fixed_effects" => Dict(
                "intercept" => coef(model_0)[1],
                "base_sim" => coef(model_0)[2]
            )
        ),
        "model_1" => Dict(
            "aic" => aic(model_1),
            "bic" => bic(model_1),
            "loglikelihood" => loglikelihood(model_1),
            "fixed_effects" => Dict(
                "intercept" => coef(model_1)[1],
                "base_sim" => coef(model_1)[2],
                "metalign_sim" => coef(model_1)[3]
            )
        ),
        "lrt_0_vs_1" => Dict(
            "deviance" => lrt.deviance[2],
            "dof" => lrt.dof[2],
            "pvalue" => lrt.pval[2]
        ),
        "comparison_0_vs_1" => Dict(
            "delta_aic" => aic(model_0) - aic(model_1),
            "delta_bic" => delta_bic,
            "log10_bayes_factor" => log10_bf
        )
    )
else
    # Ablation model: only compare M1 vs M2
    formula_1 = @formula(is_choice ~ base_sim + ablation_sim + (1 + base_sim + ablation_sim | participant_id))
    model_1 = fit(MixedModel, formula_1, data_long, Bernoulli())
    
    println("=" ^ 80)
    println("MODEL 1: Base + Ablation ($experiment_name)")
    println("=" ^ 80)
    println(model_1)
    println()
    
    formula_2 = @formula(is_choice ~ base_sim + ablation_sim + main_sim + (1 + base_sim + ablation_sim + main_sim | participant_id))
    model_2 = fit(MixedModel, formula_2, data_long, Bernoulli())
    
    println("=" ^ 80)
    println("MODEL 2: Base + Ablation + Full Metalign")
    println("=" ^ 80)
    println(model_2)
    println()
    
    lrt = MixedModels.likelihoodratiotest(model_1, model_2)
    
    println("=" ^ 80)
    println("LIKELIHOOD RATIO TEST: Model 1 vs Model 2")
    println("=" ^ 80)
    println(lrt)
    println()
    
    println("=" ^ 80)
    println("MODEL COMPARISON")
    println("=" ^ 80)
    println("Model 1 - AIC: ", round(aic(model_1), digits=2), " BIC: ", round(bic(model_1), digits=2))
    println("Model 2 - AIC: ", round(aic(model_2), digits=2), " BIC: ", round(bic(model_2), digits=2))
    println()
    
    delta_bic = bic(model_1) - bic(model_2)
    log10_bf = (delta_bic / 2) / log(10)
    
    println("=" ^ 80)
    println("BAYESIAN MODEL COMPARISON (BIC Approximation)")
    println("=" ^ 80)
    println("Delta BIC: ", round(delta_bic, digits=2), " | Log10(BF): ~", round(log10_bf, digits=2))
    println()
    
    # Save results
    results = Dict(
        "experiment_name" => experiment_name,
        "backbone_name" => backbone_name,
        "is_main_model" => false,
        "model_1" => Dict(
            "aic" => aic(model_1),
            "bic" => bic(model_1),
            "loglikelihood" => loglikelihood(model_1),
            "fixed_effects" => Dict(
                "intercept" => coef(model_1)[1],
                "base_sim" => coef(model_1)[2],
                "ablation_sim" => coef(model_1)[3]
            )
        ),
        "model_2" => Dict(
            "aic" => aic(model_2),
            "bic" => bic(model_2),
            "loglikelihood" => loglikelihood(model_2),
            "fixed_effects" => Dict(
                "intercept" => coef(model_2)[1],
                "base_sim" => coef(model_2)[2],
                "ablation_sim" => coef(model_2)[3],
                "main_sim" => coef(model_2)[4]
            )
        ),
        "lrt_1_vs_2" => Dict(
            "deviance" => lrt.deviance[2],
            "dof" => lrt.dof[2],
            "pvalue" => lrt.pval[2]
        ),
        "comparison_1_vs_2" => Dict(
            "delta_aic" => aic(model_1) - aic(model_2),
            "delta_bic" => delta_bic,
            "log10_bayes_factor" => log10_bf
        )
    )
end

# Save results to JSON file
output_file = "data/evals/levelso1o/$(experiment_name)_$(backbone_name)_analysis.json"
open(output_file, "w") do f
    JSON.print(f, results, 4)
end

println("=" ^ 80)
println("Results saved to: ", output_file)
println("=" ^ 80)
