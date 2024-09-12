export AbstractDiagonalQNModel,
  QuasiNewtonModel,
  LBFGSModel,
  LSR1Model,
  DiagonalPSBModel,
  DiagonalAndreiModel,
  SpectralGradientModel,
  DiagonalBFGSModel

abstract type QuasiNewtonModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractDiagonalQNModel{T, S} <: QuasiNewtonModel{T, S} end

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
end

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
end

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

"Construct a `LBFGSModel` from another type of model."
function LBFGSModel(nlp::AbstractNLPModel{T, S}; kwargs...) where {T, S}
  op = LBFGSOperator(T, nlp.meta.nvar; kwargs...)
  return LBFGSModel{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op)
end

"Construct a `LSR1Model` from another type of nlp."
function LSR1Model(nlp::AbstractNLPModel{T, S}; kwargs...) where {T, S}
  op = LSR1Operator(T, nlp.meta.nvar; kwargs...)
  return LSR1Model{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op)
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

function DiagonalBFGSModel(
  nlp::AbstractNLPModel{T, S};
  d0::S = fill!(S(undef, nlp.meta.nvar), one(T)),
) where {T, S}
  op = DiagonalBFGS(d0)
  return NLPModelsModifiers.DiagonalQNModel{T, S, typeof(nlp), typeof(nlp.meta), typeof(op)}(nlp.meta, nlp, op)
end

NLPModels.show_header(io::IO, nlp::QuasiNewtonModel) =
  println(io, "$(typeof(nlp)) - A QuasiNewtonModel")

function Base.show(io::IO, nlp::QuasiNewtonModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.model.counters)
end

@default_counters QuasiNewtonModel model

function NLPModels.reset_data!(nlp::QuasiNewtonModel)
  reset!(nlp.op)
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
  @eval NLPModels.$meth(nlp::QuasiNewtonModel, x::AbstractVector) = $meth(nlp.model, x)
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
    $meth(nlp.model, x, y)
end
for meth in (:jprod!, :jprod_lin!, :jprod_nln!, :jtprod!, :jtprod_lin!, :jtprod_nln!)
  @eval NLPModels.$meth(
    nlp::QuasiNewtonModel,
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
  ) = $meth(nlp.model, x, y, z)
end
NLPModels.jac_structure!(
  nlp::QuasiNewtonModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = jac_structure!(nlp.model, rows, cols)
NLPModels.jac_lin_structure!(
  nlp::QuasiNewtonModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = jac_lin_structure!(nlp.model, rows, cols)
NLPModels.jac_nln_structure!(
  nlp::QuasiNewtonModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = jac_nln_structure!(nlp.model, rows, cols)

# the following methods are affected by the Hessian approximation
NLPModels.hess_op(nlp::QuasiNewtonModel, x::AbstractVector; kwargs...) = nlp.op
NLPModels.hprod(nlp::QuasiNewtonModel, x::AbstractVector, v::AbstractVector; kwargs...) = nlp.op * v

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
  mul!(Hv, nlp.op, v)
  return Hv
end

NLPModels.neval_hprod(nlp::LBFGSModel) = nlp.op.nprod
NLPModels.neval_hprod(nlp::LSR1Model) = nlp.op.nprod
NLPModels.neval_hprod(nlp::DiagonalQNModel) = nlp.op.nprod

function Base.push!(nlp::QuasiNewtonModel, args...)
  push!(nlp.op, args...)
  return nlp
end

# not implemented: hess_structure, hess_coord, hess, ghjvprod
