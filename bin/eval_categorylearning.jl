using CSV, DataFrames, MixedModels, StatsBase, CategoricalArrays, JSON

if length(ARGS) < 2
    error("Usage: julia eval_categorylearning.jl <experiment_name> <backbone_name>")
end

experiment_name = ARGS[1]
backbone_name = ARGS[2]
csv_file = "data/evals/categorylearning/$(experiment_name)_$(backbone_name)_stats.csv"

if !isfile(csv_file)
    error("Stats file not found: $csv_file")
end

data = CSV.read(csv_file, DataFrame)
data.participant = categorical(data.participant)
data.trial = categorical(data.trial)

# Determine if this is main or an ablation
is_main = experiment_name == "MAIN"

# If ablation, load the main model data
if !is_main
    main_csv_file = "data/evals/categorylearning/MAIN_$(backbone_name)_stats.csv"
    if !isfile(main_csv_file)
        error("Main model stats file not found: $main_csv_file")
    end
    main_data = CSV.read(main_csv_file, DataFrame)
    
    # Add main model logits
    data.main_logit_0 = main_data.metalign_logit_0
    data.main_logit_1 = main_data.metalign_logit_1
end

# Normalize predictors to help convergence
dropmissing!(data)
data.base_logit_1 = zscore(data.base_logit_1)
if "metalign_logit_1" in names(data)
    data.metalign_logit_1 = zscore(data.metalign_logit_1)
end
if "main_logit_1" in names(data)
    data.main_logit_1 = zscore(data.main_logit_1)
end

if is_main
    # Model 0: Base only
    formula_0 = @formula(choice ~ 1 + base_logit_1 + (1 + base_logit_1 | participant))
    model_0 = fit(MixedModel, formula_0, data, Bernoulli())
    
    println("=" ^ 80)
    println("MODEL 0: Base only")
    println("=" ^ 80)
    println(model_0)
    println()
    # Main model: only compare M0 vs M1
    formula_1 = @formula(choice ~ 1 + base_logit_1 + metalign_logit_1 + (1 + base_logit_1 + metalign_logit_1 | participant))
    model_1 = fit(MixedModel, formula_1, data, Bernoulli())
    
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
                "base_logit_1" => coef(model_0)[2]
            )
        ),
        "model_1" => Dict(
            "aic" => aic(model_1),
            "bic" => bic(model_1),
            "loglikelihood" => loglikelihood(model_1),
            "fixed_effects" => Dict(
                "intercept" => coef(model_1)[1],
                "base_logit_1" => coef(model_1)[2],
                "metalign_logit_1" => coef(model_1)[3]
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
    # Ablation: only compare M1 vs M2
    formula_1 = @formula(choice ~ 1 + base_logit_1 + metalign_logit_1 + (1 + base_logit_1 + metalign_logit_1 | participant))
    model_1 = fit(MixedModel, formula_1, data, Bernoulli())
    
    println("=" ^ 80)
    println("MODEL 1: Base + Ablation ($experiment_name)")
    println("=" ^ 80)
    println(model_1)
    println()
    
    formula_2 = @formula(choice ~ 1 + base_logit_1 + metalign_logit_1 + main_logit_1 + (1 + base_logit_1 + metalign_logit_1 + main_logit_1 | participant))
    model_2 = fit(MixedModel, formula_2, data, Bernoulli())
    
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
                "base_logit_1" => coef(model_1)[2],
                "ablation_logit_1" => coef(model_1)[3]
            )
        ),
        "model_2" => Dict(
            "aic" => aic(model_2),
            "bic" => bic(model_2),
            "loglikelihood" => loglikelihood(model_2),
            "fixed_effects" => Dict(
                "intercept" => coef(model_2)[1],
                "base_logit_1" => coef(model_2)[2],
                "ablation_logit_1" => coef(model_2)[3],
                "main_logit_1" => coef(model_2)[4]
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

output_file = "data/evals/categorylearning/$(experiment_name)_$(backbone_name)_analysis.json"
open(output_file, "w") do f
    JSON.print(f, results, 4)
end

println("=" ^ 80)
println("Results saved to: ", output_file)
println("=" ^ 80)
