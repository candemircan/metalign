using Pkg

Pkg.activate(".")

dependencies = [
    "CSV",
    "DataFrames",
    "MixedModels",
    "StatsBase",
    "CategoricalArrays",
    "JSON"
]

Pkg.add(dependencies)

Pkg.precompile()

println("Julia environment setup complete!")
