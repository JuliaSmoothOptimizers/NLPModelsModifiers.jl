export FixedVariableEliminationModel, FixedVariableEliminationNLSModel

@doc raw"""A model whose fixed variables are eliminated.

Given a model, this type represents a second model in which the fixed variables,
i.e., the variables whose lower and upper bounds are equal, are removed from the
unknowns and replaced by their fixed value. More precisely, if the original
model has the form

```math
\begin{aligned}
       \min_X \quad & f(X)\\
\mathrm{s.t.} \quad & c_L ≤ c(X) ≤ c_U,\\
                    &  ℓ  ≤  X   ≤  u,
\end{aligned}
```

Denoting with $$\mathcal{F} = \{i \in \{1, 2, \ldots, n\} : ℓ_i = u_i\}$$ the set of indices of the fixed variables,
the new model appears to the user as

```math
\begin{aligned}
       \min_x \quad & \bar f(x)\\
\mathrm{s.t.} \quad & c_L ≤ \bar c(x) ≤ c_U,\\
                    &  ℓ_{\mathcal{F}^c}  ≤ x ≤  u_{\mathcal{F}^c},
\end{aligned}
```
where $$\mathcal{F}^c = \{1, 2, \ldots, n\} \setminus \mathcal{F}$$ and 
```math
\begin{aligned}
       \bar f(x) &= f(x, l) = f(x, u) = f(X), \\
       \bar c(x) &= c(x, l) = c(x, u) = c(X).
\end{aligned}
```

!!! warning
    It is recommended to formulate NLPModels without fixed variables.
    Users interested in performance should avoid using this model type.
"""
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

@doc raw"""Like `FixedVariableEliminationModel`, this model eliminates the fixed variables
of a nonlinear least-squares model. 
"""
mutable struct FixedVariableEliminationNLSModel{
  T,
  S <: AbstractVector{T},
  M <: AbstractNLSModel{T, S},
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

for (Model, cons_f!, jprod_f!, jtprod_f!, ncon) in (
  (:FixedVariableEliminationModels, :cons!, :jprod!, :jtprod!, :(nlp.meta.ncon)),
  (:FixedVariableEliminationModels, :cons_lin!, :jprod_lin!, :jtprod_lin!, :(nlp.meta.nlin)),
  (:FixedVariableEliminationModels, :cons_nln!, :jprod_nln!, :jtprod_nln!, :(nlp.meta.nnln)),
  (:FixedVariableEliminationNLSModel, :residual!, :jprod_residual!, :jtprod_residual!, :(nlp.nls_meta.nequ),),
)
  @eval begin
    function NLPModels.$cons_f!(
      nlp::$Model,
      x::AbstractVector,
      cx::AbstractVector,
    )
      @lencheck nlp.meta.nvar x
      @lencheck $ncon cx

      nlp.x_full[nlp.free] .= x                # load buffer with free variables
      $cons_f!(nlp.model, nlp.x_full, cx)
      return cx
    end

    function NLPModels.$jprod_f!(
      nlp::$Model,
      x::AbstractVector,
      v::AbstractVector,
      Jv::AbstractVector,
    )
      @lencheck nlp.meta.nvar x v
      @lencheck $ncon Jv

      nlp.x_full[nlp.free] .= x                # load buffer with free variables
      nlp.v_full .= 0
      nlp.v_full[nlp.free] .= v                # load buffer with free variables
      $jprod_f!(nlp.model, nlp.x_full, nlp.v_full, Jv)
      return Jv                                # J(x) * v = J(X) * [v_free; 0]
    end

    function NLPModels.$jtprod_f!(
      nlp::$Model,
      x::AbstractVector,
      v::AbstractVector,
      Jtv::AbstractVector,
    )
      @lencheck nlp.meta.nvar x Jtv
      @lencheck $ncon v

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

function NLPModels.jth_hprod!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
)
  @lencheck nlp.meta.nvar x v Hv
  @rangecheck 1 nlp.meta.ncon j

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  nlp.v_full .= 0
  nlp.v_full[nlp.free] .= v                # load buffer with free variables
  jth_hprod!(nlp.model, nlp.x_full, nlp.v_full, j, nlp.g_full)
  Hv .= @view nlp.g_full[nlp.free]         # ∇²cⱼ(x) * v = (∇²cⱼ(X))[v_free; 0]
  return Hv
end

function NLPModels.ghjvprod!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  nlp.g_full .= 0
  nlp.g_full[nlp.free] .= g                # load buffer with free variables
  nlp.v_full .= 0
  nlp.v_full[nlp.free] .= v                # load buffer with free variables
  ghjvprod!(nlp.model, nlp.x_full, nlp.g_full, nlp.v_full, gHv)
  return gHv                               # (gᵀ ∇²cᵢ(x) v)ᵢ = ([g_free; 0]ᵀ ∇²cᵢ(X) [v_free; 0])ᵢ
end

function NLPModels.hprod_residual!(
  nlp::FixedVariableEliminationNLSModel,
  x::AbstractVector,
  i::Integer,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck nlp.meta.nvar x v Hiv
  @rangecheck 1 nlp.nls_meta.nequ i

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  nlp.v_full .= 0
  nlp.v_full[nlp.free] .= v                # load buffer with free variables
  hprod_residual!(nlp.model, nlp.x_full, i, nlp.v_full, nlp.g_full)
  Hiv .= @view nlp.g_full[nlp.free]        # ∇²Fᵢ(x) * v = (∇²Fᵢ(X))[v_free; 0]
  return Hiv
end
