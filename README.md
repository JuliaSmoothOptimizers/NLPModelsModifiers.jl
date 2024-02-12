# NLPModelsModifiers

This package provides optimization models based on [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
The API can be found on their [documentation page](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api).

The models in this package specialize on modifying existing methods.

## How to Cite

 If you use NLPModelsModifiers.jl in your work, please cite using the format given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl/blob/main/CITATION.cff).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4603924.svg)](https://doi.org/10.5281/zenodo.4603924)
[![GitHub release](https://img.shields.io/github/release/JuliaSmoothOptimizers/NLPModelsModifiers.jl.svg)](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/NLPModelsModifiers.jl/stable)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/NLPModelsModifiers.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/NLPModelsModifiers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/NLPModelsModifiers.jl)

![CI](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl/workflows/CI/badge.svg?branch=main)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/NLPModelsModifiers.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/NLPModelsModifiers.jl)

## Models

The following models are implemented.

- [`FeasibilityFormNLS`](@ref): Creates residual variables and constraints, so that the residual is linear.
- [`FeasibilityResidual`](@ref): Creates a nonlinear least squares model from an equality constrained problem in which the residual function is the constraints function.
- [`LBFGSModel`](@ref): Creates a model using a LBFGS approximation to the Hessian using an existing NLPModel.
- [`LSR1Model`](@ref): Creates a model using a LSR1 approximation to the Hessian using an existing NLPModel.
- [`SlackModel`](@ref): Creates an equality constrained problem with bounds on the variables using an existing NLPModel.
- [`SlackNLSModel`](@ref): Creates an equality constrained nonlinear least squares problem with bounds on the variables using an existing NLSModel.

See the [documentation](https://JuliaSmoothOptimizers.github.io/NLPModels.jl/dev/) for details on the models.

## Installation

```julia
pkg> add NLPModelsModifiers
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
