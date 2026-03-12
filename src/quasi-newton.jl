export AbstractDiagonalQNModel,
  QuasiNewtonModel,
  LBFGSModel,
  LSR1Model,
  DiagonalPSBModel,
  DiagonalAndreiModel,
  SpectralGradientModel,
  get_model,
  get_op

abstract type QuasiNewtonModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractDiagonalQNModel{T, S} <: QuasiNewtonModel{T, S} end

"""
    get_model(nlp::QuasiNewtonModel)

Return the underlying model of a `QuasiNewtonModel`.
"""
function get_model(nlp::QuasiNewtonModel)
  error("get_model is not implemented for $(typeof(nlp)).")
end

"""
    get_op(nlp::QuasiNewtonModel)

Return the quasi-Newton operator of a `QuasiNewtonModel`.
"""
function get_op(nlp::QuasiNewtonModel)
  error("get_op is not implemented for $(typeof(nlp)).")
end

mutable struct LBFGSModel{
  T,
  S,
  M <: AbstractNLPModel{T, S},
  Meta <: AbstractNLPModelMeta{T, S},
  Op <: LBFGSOperator{T},
} <: QuasiNewtonModel{T, S}
  meta::Meta
  model::M
  op::Op
  v::S  # extra vector to compute and store yₖ = ∇fₖ₊₁ - ∇fₖ
end

get_model(nlp::LBFGSModel) = nlp.model
get_op(nlp::LBFGSModel) = nlp.op
@default_counters LBFGSModel model (neval_hprod,)
NLPModels.neval_hprod(nlp::LBFGSModel) = get_op(nlp).nprod

mutable struct LSR1Model{
  T,
  S,
  M <: AbstractNLPModel{T, S},
  Meta <: AbstractNLPModelMeta{T, S},
  Op <: LSR1Operator{T},
} <: QuasiNewtonModel{T, S}
  meta::Meta
  model::M
  op::Op
  v::S  # extra vector to compute and store yₖ = ∇fₖ₊₁ - ∇fₖ
end

get_model(nlp::LSR1Model) = nlp.model
get_op(nlp::LSR1Model) = nlp.op
@default_counters LSR1Model model (neval_hprod,)
NLPModels.neval_hprod(nlp::LSR1Model) = get_op(nlp).nprod

mutable struct DiagonalQNModel{
  T,
  S,
  M <: AbstractNLPModel{T, S},
  Meta <: AbstractNLPModelMeta{T, S},
  Op <: AbstractDiagonalQuasiNewtonOperator{T},
} <: AbstractDiagonalQNModel{T, S}
  meta::Meta
  model::M
  op::Op
end

get_model(nlp::DiagonalQNModel) = nlp.model
get_op(nlp::DiagonalQNModel) = nlp.op
@default_counters DiagonalQNModel model (neval_hprod,)
NLPModels.neval_hprod(nlp::DiagonalQNModel) = get_op(nlp).nprod

"Construct a `LBFGSModel` from another type of model."
function LBFGSModel(nlp::AbstractNLPModel{T, S}; kwargs...) where {T, S}
  op = LBFGSOperator(T, nlp.meta.nvar; kwargs...)
  v = similar(nlp.meta.x0)
  return LBFGSModel{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op, v)
end

"Construct a `LSR1Model` from another type of nlp."
function LSR1Model(nlp::AbstractNLPModel{T, S}; kwargs...) where {T, S}
  op = LSR1Operator(T, nlp.meta.nvar; kwargs...)
  v = similar(nlp.meta.x0)
  return LSR1Model{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op, v)
end

"""
    DiagonalPSBModel(nlp; d0 = fill!(S(undef, nlp.meta.nvar), 1.0))

Construct a `DiagonalPSBModel` from another type of nlp, in which the Hessian is approximated
via a diagonal PSB quasi-Newton operator.
`d0` is the initial approximation of the diagonal of the Hessian, and by default a vector of ones.
See the
[`DiagonalPSB operator documentation`](https://juliasmoothoptimizers.github.io/LinearOperators.jl/stable/reference/#LinearOperators.DiagonalPSB).
"""
function DiagonalPSBModel(
  nlp::AbstractNLPModel{T, S};
  d0::S = fill!(S(undef, nlp.meta.nvar), one(T)),
) where {T, S}
  op = DiagonalPSB(d0)
  return DiagonalQNModel{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op)
end

"""
    DiagonalAndreiModel(nlp; d0 = fill!(S(undef, nlp.meta.nvar), 1.0))

Construct a `DiagonalAndreiModel` from another type of nlp, in which the Hessian is approximated
via a diagonal Andrei quasi-Newton operator.
`d0` is the initial approximation of the diagonal of the Hessian, and by default a vector of ones.
See the
[`DiagonalAndrei operator documentation`](https://juliasmoothoptimizers.github.io/LinearOperators.jl/stable/reference/#LinearOperators.DiagonalAndrei).
"""
function DiagonalAndreiModel(
  nlp::AbstractNLPModel{T, S};
  d0::S = fill!(S(undef, nlp.meta.nvar), one(T)),
) where {T, S}
  op = DiagonalAndrei(d0)
  return DiagonalQNModel{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op)
end

"""
    SpectralGradientModel(nlp; σ = 1.0)

Construct a `SpectralGradientModel` rhat approximates the Hessian as `σI` from another type of nlp.
The keyword argument `σ` is the initial positive multiple of the identity.
See the
[`SpectralGradient operator documentation`](https://juliasmoothoptimizers.github.io/LinearOperators.jl/stable/reference/#LinearOperators.SpectralGradient)
for more information about the used algorithms.
"""
function SpectralGradientModel(nlp::AbstractNLPModel{T, S}; σ::T = one(T)) where {T, S}
  op = SpectralGradient(σ, nlp.meta.nvar)
  return DiagonalQNModel{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op)
end

NLPModels.show_header(io::IO, nlp::QuasiNewtonModel) =
  println(io, "$(typeof(nlp)) - A QuasiNewtonModel")

function Base.show(io::IO, nlp::QuasiNewtonModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, get_model(nlp).counters)
end

function NLPModels.reset_data!(nlp::QuasiNewtonModel)
  reset!(get_op(nlp))
  return nlp
end

# the following methods are not affected by the Hessian approximation
for meth in (
  :obj,
  :grad,
  :cons,
  :cons_lin,
  :cons_nln,
  :jac_coord,
  :jac_lin_coord,
  :jac_nln_coord,
  :jac,
  :jac_lin,
  :jac_nln,
)
  @eval NLPModels.$meth(nlp::QuasiNewtonModel, x::AbstractVector) = $meth(get_model(nlp), x)
end
for meth in (
  :grad!,
  :cons!,
  :cons_lin!,
  :cons_nln!,
  :jprod,
  :jprod_lin,
  :jprod_nln,
  :jtprod,
  :jtprod_lin,
  :jtprod_nln,
  :objgrad,
  :objgrad!,
  :jac_coord!,
  :jac_lin_coord!,
  :jac_nln_coord!,
)
  @eval NLPModels.$meth(nlp::QuasiNewtonModel, x::AbstractVector, y::AbstractVector) =
    $meth(get_model(nlp), x, y)
end
for meth in (:jprod!, :jprod_lin!, :jprod_nln!, :jtprod!, :jtprod_lin!, :jtprod_nln!)
  @eval NLPModels.$meth(
    nlp::QuasiNewtonModel,
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
  ) = $meth(get_model(nlp), x, y, z)
end
NLPModels.jac_structure!(
  nlp::QuasiNewtonModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = jac_structure!(get_model(nlp), rows, cols)
NLPModels.jac_lin_structure!(
  nlp::QuasiNewtonModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = jac_lin_structure!(get_model(nlp), rows, cols)
NLPModels.jac_nln_structure!(
  nlp::QuasiNewtonModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = jac_nln_structure!(get_model(nlp), rows, cols)

# the following methods are affected by the Hessian approximation
NLPModels.hess_op(nlp::QuasiNewtonModel, x::AbstractVector; kwargs...) = get_op(nlp)
NLPModels.hprod(nlp::QuasiNewtonModel, x::AbstractVector, v::AbstractVector; kwargs...) =
  get_op(nlp) * v

function NLPModels.hprod!(
  nlp::QuasiNewtonModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  kwargs...,
)
  return hprod!(nlp, x, v, Hv; kwargs...)
end
function NLPModels.hprod!(
  nlp::QuasiNewtonModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar Hv x v
  mul!(Hv, get_op(nlp), v)
  return Hv
end

function Base.push!(nlp::QuasiNewtonModel, args...)
  push!(get_op(nlp), args...)
  return nlp
end

# not implemented: hess_structure, hess_coord, hess, ghjvprod
