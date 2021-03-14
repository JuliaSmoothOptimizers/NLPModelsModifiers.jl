# Models

- [`FeasibilityFormNLS`](@ref): Creates residual variables and constraints, so that the residual is linear.
- [`FeasibilityResidual`](@ref): Creates a nonlinear least squares model from an equality constrained problem in which the residual function is the constraints function.
- [`LBFGSModel`](@ref): Creates a model using a LBFGS approximation to the Hessian using an existing NLPModel.
- [`LSR1Model`](@ref): Creates a model using a LSR1 approximation to the Hessian using an existing NLPModel.
- [`SlackModel`](@ref): Creates an equality constrained problem with bounds on the variables using an existing NLPModel.
- [`SlackNLSModel`](@ref): Creates an equality constrained nonlinear least squares problem with bounds on the variables using an existing NLSModel.

```@docs
FeasibilityFormNLS
FeasibilityResidual
LBFGSModel
LSR1Model
SlackModel
SlackNLSModel
```