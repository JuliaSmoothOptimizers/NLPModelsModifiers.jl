# [NLPModelsModifiers.jl documentation](@id Home)

This package provides optimization models based on [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
The API can be found on their [documentation page](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api).

The models in this package specialize on modifying existing methods.

## Install

Install NLPModelsModifiers.jl with the following command.
```julia
pkg> add NLPModelsModifiers
```

## Models

The following models are implemented.

- [`FeasibilityFormNLS`](@ref): Creates residual variables and constraints, so that the residual is linear.
- [`FeasibilityResidual`](@ref): Creates a nonlinear least squares model from an equality constrained problem in which the residual function is the constraints function.
- [`LBFGSModel`](@ref): Creates a model using a LBFGS approximation to the Hessian using an existing NLPModel.
- [`LSR1Model`](@ref): Creates a model using a LSR1 approximation to the Hessian using an existing NLPModel.
- [`SlackModel`](@ref): Creates an equality constrained problem with bounds on the variables using an existing NLPModel.
- [`SlackNLSModel`](@ref): Creates an equality constrained nonlinear least squares problem with bounds on the variables using an existing NLSModel.

## License

This content is released under the [MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/) License.

## Contents

```@contents
```
