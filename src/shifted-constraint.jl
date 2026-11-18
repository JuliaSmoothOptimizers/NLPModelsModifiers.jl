export ShiftedConstraintModel, ShiftedConstraintNLSModel

@doc raw"""A model whose constraints are shifted.

Given a model, this type represents a second model in which the equality 
constraints are shifted to 0. More precisely, if the original model has the form

```math
\begin{aligned}
       \min_x \quad & f(x)\\
\mathrm{s.t.} \quad & c(x) = v,\\
                    &  ℓ  ≤  g(x)   ≤  u,
\end{aligned}
```

the new model appears to the user as

```math
\begin{aligned}
       \min_x \quad & f(x)\\
\mathrm{s.t.} \quad & d(x) = 0,\\
                    &  ℓ  ≤  g(x)   ≤  u,
\end{aligned}
```
where ``d(x) = c(x) - v``.
"""
mutable struct ShiftedConstraintModel{
  T,
  S<:AbstractVector{T},
  M<:AbstractNLPModel{T,S},
} <: AbstractNLPModel{T,S}
  meta::NLPModelMeta{T, S}
  model::M

  shift_lin::S
  shift_nln::S
  shift::S
end

@doc raw"""Like `ShiftedConstraintModel`, this model shifts the equality 
constraints of a nonlinear least-squares model to 0.
"""
mutable struct ShiftedConstraintNLSModel{
  T,
  S<:AbstractVector{T},
  M<:AbstractNLSModel{T,S},
} <: AbstractNLSModel{T,S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  model::M

  shift_lin::S
  shift_nln::S
  shift::S
end

const ShiftedConstraintModels = 
  Union{ShiftedConstraintModel, ShiftedConstraintNLSModel}

# retrieve counters from underlying model
@default_counters ShiftedConstraintModels model
@default_nlscounters ShiftedConstraintNLSModel model

nls_meta(nlp::ShiftedConstraintNLSModel) = nlp.nls_meta

get_model(nlp::ShiftedConstraintModels) = nlp.model

NLPModels.show_header(io::IO, nlp::ShiftedConstraintModel) =
  println(io, "ShiftedConstraintModel - Model with constraints shifted to 0")

function Base.show(io::IO, nlp::ShiftedConstraintModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.model.counters)
end

NLPModels.show_header(io::IO, nls::ShiftedConstraintNLSModel) =
  println(io, "ShiftedConstraintNLSModel - Nonlinear least-squares model with constraints shifted to 0")

function Base.show(io::IO, nls::ShiftedConstraintNLSModel)
  show_header(io, nls)
  show(io, nls.meta, nls.nls_meta)
  show(io, nls.model.counters)
end

# ------------------------------------------------------------------------
# NLPModels API
# ------------------------------------------------------------------------

# Update constraint definitions.
for (cons_f!, shift_f, ncon) in (
  (:cons!, :shift, :(nlp.meta.ncon)),
  (:cons_lin!, :shift_lin, :(nlp.meta.nlin)),
  (:cons_nln!, :shift_nln, :(nlp.meta.nnln)),
)
  @eval function NLPModels.$cons_f!(
    nlp::ShiftedConstraintModels,
    x::AbstractVector,
    cx::AbstractVector,
  )
    @lencheck nlp.meta.nvar x
    @lencheck $ncon cx

    $cons_f!(nlp.model, x, cx)
    cx .-= nlp.$shift_f           # Shift the constraints to 0.
    return cx
  end
end

# Default to the model for all other API functions.
for (model, fun) in (
  # objective
  (:ShiftedConstraintModels, :obj),
  (:ShiftedConstraintModels, :grad!),
  # Jacobian products
  (:ShiftedConstraintModels, :jprod_lin!),
  (:ShiftedConstraintModels, :jprod_nln!),
  (:ShiftedConstraintModels, :jtprod_lin!),
  (:ShiftedConstraintModels, :jtprod_nln!),
  # Hessian products
  (:ShiftedConstraintModels, :hprod!),       # with y
  (:ShiftedConstraintModels, :jth_hprod!),
  (:ShiftedConstraintModels, :ghjvprod!),
  # sparse Jacobian
  (:ShiftedConstraintModels, :jac_lin_structure!),
  (:ShiftedConstraintModels, :jac_nln_structure!),
  (:ShiftedConstraintModels, :jac_lin_coord!),
  (:ShiftedConstraintModels, :jac_nln_coord!),
  # sparse Hessian
  (:ShiftedConstraintModels, :hess_structure!),
  (:ShiftedConstraintModels, :hess_coord!),  # with y
  (:ShiftedConstraintModels, :jth_hess_coord!),
  # NLS only
  (:ShiftedConstraintNLSModel, :residual!),
  (:ShiftedConstraintNLSModel, :jprod_residual!),
  (:ShiftedConstraintNLSModel, :jtprod_residual!),
  (:ShiftedConstraintNLSModel, :hprod_residual!),
  (:ShiftedConstraintNLSModel, :jac_structure_residual!),
  (:ShiftedConstraintNLSModel, :jac_coord_residual!),
  (:ShiftedConstraintNLSModel, :hess_structure_residual!),
  (:ShiftedConstraintNLSModel, :hess_coord_residual!),
)
  
  @eval function NLPModels.$fun(nlp::$model, args...; kwargs...)
    return $fun(nlp.model, args...; kwargs...)
  end
end

for (model, fun, nargs) in (
  # objective
  (:ShiftedConstraintNLSModel, :obj, 1),
  (:ShiftedConstraintNLSModel, :grad!, 2),
  (:ShiftedConstraintModels, :objgrad!, 2),
  # Jacobian products
  (:ShiftedConstraintModels, :jprod!, 3),
  (:ShiftedConstraintModels, :jtprod!, 3),
  # Hessian products
  (:ShiftedConstraintModels, :hprod!, 3),       # without y
  # sparse Jacobian
  (:ShiftedConstraintModels, :jac_coord!, 2),
  # sparse Hessian
  (:ShiftedConstraintModels, :hess_coord!, 2),  # without y
)
  args = [Symbol(:arg, k) for k = 1:nargs]
  @eval function NLPModels.$fun(nlp::$model, $([:($arg::AbstractVector) for arg in args]...); kwargs...)
    return $fun(nlp.model, $(args...); kwargs...)
  end
end

NLPModels.jac_structure!(
  nlp::ShiftedConstraintModels,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T <: Integer} = jac_structure!(nlp.model, rows, cols)