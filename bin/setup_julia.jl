using Pkg

Pkg.activate(".")

dependencies = [
    "CSV",
    "DataFrames",
    "MixedModels",
    "StatsBase",
    "CategoricalArrays",
    "GLM",
    "JSON"
]

Pkg.add(dependencies)

Pkg.precompile()

println("Julia environment setup complete!")
