export FixedVariableEliminationModel, FixedVariableEliminationNLSModel

mutable struct FixedVariableEliminationModel{
  T,
  S <: AbstractVector{T},
  M <: AbstractNLPModel{T, S},
} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  model::M

  free::Vector{Int}   # indices (in the full model) of the free variables

  x_full::S    # length nvar(nlp); fixed entries permanently hold their bound value
  g_full::S    # scratch buffer, length nvar(nlp)
  v_full::S    # scratch buffer, length nvar(nlp)
end

mutable struct FixedVariableEliminationNLSModel{
  T,
  S <: AbstractVector{T},
  M <: AbstractNLPModel{T, S},
} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  model::M

  free::Vector{Int}   # indices (in the full model) of the free variables

  x_full::S    # length nvar(nlp); fixed entries permanently hold their bound value
  g_full::S    # scratch buffer, length nvar(nlp)
  v_full::S    # scratch buffer, length nvar(nlp)
end

const FixedVariableEliminationModels =
  Union{FixedVariableEliminationModel, FixedVariableEliminationNLSModel}

# retrieve counters from underlying model
@default_counters FixedVariableEliminationModels model
@default_nlscounters FixedVariableEliminationNLSModel model

nls_meta(nlp::FixedVariableEliminationNLSModel) = nlp.nls_meta

# ------------------------------------------------------------------------
# NLPModels API
# ------------------------------------------------------------------------

function NLPModels.obj(nlp::FixedVariableEliminationModels, x::AbstractVector)
  @lencheck nlp.meta.nvar x

  nlp.x_full[nlp.free] .= x           # load buffer with free variables
  return obj(nlp.model, nlp.x_full)
end

function NLPModels.grad!(nlp::FixedVariableEliminationModels, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  grad!(nlp.model, nlp.x_full, nlp.g_full) # apply gradient on full model
  g .= @view nlp.g_full[nlp.free]          # unload gradient of free variables
  return g
end

function NLPModels.objgrad!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  g::AbstractVector,
)
  @lencheck nlp.meta.nvar x g

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  f, _ = objgrad!(nlp.model, nlp.x_full, nlp.g_full) # apply objgrad on full model
  g .= @view nlp.g_full[nlp.free]                    # unload gradient of free variables
  return f, g
end

for (cons_f!, jprod_f!, jtprod_f!, ncon) in (
  (:cons!, :jprod!, :jtprod!, :ncon),
  (:cons_lin!, :jprod_lin!, :jtprod_lin!, :nlin),
  (:cons_nln!, :jprod_nln!, :jtprod_nln!, :nnln),
)
  @eval begin
    function NLPModels.$cons_f!(
      nlp::FixedVariableEliminationModels,
      x::AbstractVector,
      cx::AbstractVector,
    )
      @lencheck nlp.meta.nvar x
      @lencheck nlp.meta.$ncon cx

      nlp.x_full[nlp.free] .= x                # load buffer with free variables
      $cons_f!(nlp.model, nlp.x_full, cx)
      return cx
    end

    function NLPModels.$jprod_f!(
      nlp::FixedVariableEliminationModels,
      x::AbstractVector,
      v::AbstractVector,
      Jv::AbstractVector,
    )
      @lencheck nlp.meta.nvar x v
      @lencheck nlp.meta.$ncon Jv

      nlp.x_full[nlp.free] .= x                # load buffer with free variables
      nlp.v_full .= 0
      nlp.v_full[nlp.free] .= v                # load buffer with free variables
      $jprod_f!(nlp.model, nlp.x_full, nlp.v_full, Jv)
      return Jv                                # J(x) * v = J(X) * [v_free; 0]
    end

    function NLPModels.$jtprod_f!(
      nlp::FixedVariableEliminationModels,
      x::AbstractVector,
      v::AbstractVector,
      Jtv::AbstractVector,
    )
      @lencheck nlp.meta.nvar x Jtv
      @lencheck nlp.meta.$ncon v

      nlp.x_full[nlp.free] .= x                # load buffer with free variables
      $jtprod_f!(nlp.model, nlp.x_full, v, nlp.g_full)
      Jtv .= @view nlp.g_full[nlp.free]        # J(x)' * v = (J(X)' * v)_free
      return Jtv
    end
  end
end

function NLPModels.hprod!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v Hv

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  nlp.v_full .= 0
  nlp.v_full[nlp.free] .= v                # load buffer with free variables
  hprod!(nlp.model, nlp.x_full, nlp.v_full, nlp.g_full; obj_weight = obj_weight)
  Hv .= @view nlp.g_full[nlp.free]         # unload Hessian-vector product of free variables
  return Hv
end

function NLPModels.hprod!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon y

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  nlp.v_full .= 0
  nlp.v_full[nlp.free] .= v                # load buffer with free variables
  hprod!(nlp.model, nlp.x_full, y, nlp.v_full, nlp.g_full; obj_weight = obj_weight)
  Hv .= @view nlp.g_full[nlp.free]         # unload Hessian-vector product of free variables
  return Hv
end
