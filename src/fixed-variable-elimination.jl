export FixedVariableEliminationModel, FixedVariableEliminationNLSModel
export recover_full_solution, recover_full_solution!

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

  # positions, in the coordinates of the full model, of the entries involving fixed variables
  jac_dead::Vector{Int}      # in the Jacobian
  jac_lin_dead::Vector{Int}  # in the linear Jacobian
  jac_nln_dead::Vector{Int}  # in the nonlinear Jacobian
  hess_dead::Vector{Int}     # in the Hessian of the Lagrangian

  x_full::S    # length nvar(nlp); fixed entries permanently hold their bound value
  g_full::S    # scratch buffer, length nvar(nlp)
  v_full::S    # length nvar(nlp); fixed entries permanently hold zero
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

  # positions, in the coordinates of the full model, of the entries involving fixed variables
  jac_dead::Vector{Int}      # in the Jacobian
  jac_lin_dead::Vector{Int}  # in the linear Jacobian
  jac_nln_dead::Vector{Int}  # in the nonlinear Jacobian
  hess_dead::Vector{Int}     # in the Hessian of the Lagrangian
  jac_residual_dead::Vector{Int}  # in the Jacobian of the residual
  hess_residual_dead::Vector{Int} # in the Hessian of the residual

  x_full::S    # length nvar(nlp); fixed entries permanently hold their bound value
  g_full::S    # scratch buffer, length nvar(nlp)
  v_full::S    # length nvar(nlp); fixed entries permanently hold zero
end

const FixedVariableEliminationModels =
  Union{FixedVariableEliminationModel, FixedVariableEliminationNLSModel}

# retrieve counters from underlying model
@default_counters FixedVariableEliminationModels model
@default_nlscounters FixedVariableEliminationNLSModel model

nls_meta(nlp::FixedVariableEliminationNLSModel) = nlp.nls_meta

get_model(nlp::FixedVariableEliminationModels) = nlp.model

NLPModels.show_header(io::IO, nlp::FixedVariableEliminationModel) =
  println(io, "FixedVariableEliminationModel - Model without fixed variables")

function Base.show(io::IO, nlp::FixedVariableEliminationModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.model.counters)
end

NLPModels.show_header(io::IO, nls::FixedVariableEliminationNLSModel) =
  println(io, "FixedVariableEliminationNLSModel - Nonlinear least-squares model without fixed variables")

function Base.show(io::IO, nls::FixedVariableEliminationNLSModel)
  show_header(io, nls)
  show(io, nls.meta, nls.nls_meta)
  show(io, nls.model.counters)
end

"""
    X = recover_full_solution(nlp, x)

Return the vector `X` of the full model corresponding to the vector `x` of the free variables
of `nlp`: the free entries of `X` are those of `x` and the fixed entries hold their fixed value.
If `nlp` is wrapped in other models, it can be retrieved with `find_model`.
"""
recover_full_solution(nlp::FixedVariableEliminationModels, x::AbstractVector) =
  recover_full_solution!(nlp, x, similar(nlp.x_full))

"""
    recover_full_solution!(nlp, x, X)

In-place version of [`recover_full_solution`](@ref).
"""
function recover_full_solution!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  X::AbstractVector,
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.model.meta.nvar X

  X .= nlp.x_full                          # fixed entries hold their fixed value
  X[nlp.free] .= x                         # load free variables
  return X
end

function fixed_variable_elimination_meta(
  meta::AbstractNLPModelMeta{T, S};
  name = meta.name * "-nofixed",
) where {T, S}
  free = setdiff(1:(meta.nvar), meta.ifix)

  if isempty(free)
    throw(error("NLPModels whose variables are all fixed are not supported."))
  end

  return NLPModelMeta{T, S}(
    length(free),
    x0 = meta.x0[free],
    lvar = meta.lvar[free],
    uvar = meta.uvar[free],
    ncon = meta.ncon,
    lcon = meta.lcon,
    ucon = meta.ucon,
    y0 = meta.y0,
    nnzj = meta.nnzj,   # the sparsity structure of the full model is kept,
    nnzh = meta.nnzh,   # the entries involving fixed variables become explicit zeros
    lin = meta.lin,
    lin_nnzj = meta.lin_nnzj,
    nln_nnzj = meta.nln_nnzj,
    name = name,
    minimize = meta.minimize,
    islp = meta.islp,
    sparse_jacobian = meta.sparse_jacobian,
    sparse_hessian = meta.sparse_hessian,
    grad_available = meta.grad_available,
    jac_available = meta.jac_available,
    hess_available = meta.hess_available,
    jprod_available = meta.jprod_available,
    jtprod_available = meta.jtprod_available,
    hprod_available = meta.hprod_available,
  )
end

function fixed_variable_elimination_buffers(meta::AbstractNLPModelMeta{T, S}) where {T, S}
  x_full = copy(meta.x0)
  x_full[meta.ifix] .= @view meta.lvar[meta.ifix]
  g_full = similar(meta.x0)
  v_full = fill!(similar(meta.x0), zero(T))
  return x_full, g_full, v_full
end

"Construct a `FixedVariableEliminationModel` from another type of model."
function FixedVariableEliminationModel(model::AbstractNLPModel; name = model.meta.name * "-nofixed")
  isempty(model.meta.ifix) && return model

  free = setdiff(1:(model.meta.nvar), model.meta.ifix)
  meta = fixed_variable_elimination_meta(model.meta, name = name)
  x_full, g_full, v_full = fixed_variable_elimination_buffers(model.meta)

  # the dead entries are computed at the first call to the sparse API: each list is a distinct [-1]
  fnlp = FixedVariableEliminationModel(meta, model, free, [-1], [-1], [-1], [-1], x_full, g_full, v_full)
  finalizer(nlp -> finalize(nlp.model), fnlp)

  return fnlp
end

"Construct a `FixedVariableEliminationNLSModel` from another type of nonlinear least-squares model."
function FixedVariableEliminationNLSModel(
  model::AbstractNLSModel{T, S};
  name = model.meta.name * "-nofixed",
) where {T, S}
  isempty(model.meta.ifix) && return model

  free = setdiff(1:(model.meta.nvar), model.meta.ifix)
  meta = fixed_variable_elimination_meta(model.meta, name = name)
  nls_meta = NLSMeta{T, S}(
    model.nls_meta.nequ,
    meta.nvar,
    x0 = meta.x0,
    nnzj = model.nls_meta.nnzj,
    nnzh = model.nls_meta.nnzh,
    lin = model.nls_meta.lin,
    jac_residual_available = model.nls_meta.jac_residual_available,
    hess_residual_available = model.nls_meta.hess_residual_available,
    jprod_residual_available = model.nls_meta.jprod_residual_available,
    jtprod_residual_available = model.nls_meta.jtprod_residual_available,
    hprod_residual_available = model.nls_meta.hprod_residual_available,
  )
  x_full, g_full, v_full = fixed_variable_elimination_buffers(model.meta)

  # the dead entries are computed at the first call to the sparse API: each list is a distinct [-1]
  fnls = FixedVariableEliminationNLSModel(
    meta,
    nls_meta,
    model,
    free,
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    x_full,
    g_full,
    v_full,
  )
  finalizer(nls -> finalize(nls.model), fnls)

  return fnls
end


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
  nlp.v_full[nlp.free] .= v                # load buffer with free variables
  hprod_residual!(nlp.model, nlp.x_full, i, nlp.v_full, nlp.g_full)
  Hiv .= @view nlp.g_full[nlp.free]        # ∇²Fᵢ(x) * v = (∇²Fᵢ(X))[v_free; 0]
  return Hiv
end

# ------------------------------------------------------------------------
# Sparse API
# ------------------------------------------------------------------------

# The dead entries are computed at the first call to the sparse API; until then, the list is [-1].
is_uninitialized(dead::Vector{Int}) = length(dead) == 1 && dead[1] == -1

for (Model, structure_f!, nnz, dead) in (
  (:FixedVariableEliminationModels, :jac_structure!, :(nlp.meta.nnzj), :jac_dead),
  (:FixedVariableEliminationModels, :jac_lin_structure!, :(nlp.meta.lin_nnzj), :jac_lin_dead),
  (:FixedVariableEliminationModels, :jac_nln_structure!, :(nlp.meta.nln_nnzj), :jac_nln_dead),
  (:FixedVariableEliminationNLSModel, :jac_structure_residual!, :(nlp.nls_meta.nnzj), :jac_residual_dead,),
)
  @eval function NLPModels.$structure_f!(
    nlp::$Model,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
  ) where {T <: Integer}
    @lencheck $nnz rows cols

    $structure_f!(nlp.model, rows, cols)     # compute the sparsity structure of the full model

    # Kill the column entries that correspond to fixed variables.
    record = is_uninitialized(nlp.$dead)     # first call: record the dead entries
    record && empty!(nlp.$dead)              # remove the [-1] sentinel
    n_free = length(nlp.free)
    for k in eachindex(cols)                 # O(nnzj)
      col_j_free_idx = searchsortedfirst(nlp.free, cols[k])  # O(log(n_free))
      is_col_j_free = nlp.free[min(col_j_free_idx, n_free)] == cols[k]
      if is_col_j_free
        cols[k] = col_j_free_idx             # remap the column index to the free variable space
      else
        rows[k], cols[k] = 1, 1              # Kill the entry: move it to some entry in the free variable space.
        record && push!(nlp.$dead, k)        # Bookkeeping
      end
    end
    return rows, cols
  end
end

for (Model, structure_f!, nnz, dead) in (
  (:FixedVariableEliminationModels, :hess_structure!, :(nlp.meta.nnzh), :hess_dead),
  (:FixedVariableEliminationNLSModel, :hess_structure_residual!, :(nlp.nls_meta.nnzh), :hess_residual_dead,),
)
  @eval function NLPModels.$structure_f!(
    nlp::$Model,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
  )
    @lencheck $nnz rows cols

    $structure_f!(nlp.model, rows, cols) # compute the sparsity structure of the full model

    # Kill the entries that correspond to fixed variables.
    record = is_uninitialized(nlp.$dead)     # first call: record the dead entries
    record && empty!(nlp.$dead)              # remove the [-1] sentinel
    n_free = length(nlp.free)
    for k in eachindex(cols)                 # O(nnzh)

      # Check column
      col_h_free_idx = searchsortedfirst(nlp.free, cols[k])  # O(log(n_free))
      col_is_free_h_idx = nlp.free[min(col_h_free_idx, n_free)] == cols[k]

      # Check row
      row_h_free_idx = searchsortedfirst(nlp.free, rows[k])  # O(log(n_free))
      row_is_free_h_idx = nlp.free[min(row_h_free_idx, n_free)] == rows[k]

      if col_is_free_h_idx && row_is_free_h_idx
        cols[k] = col_h_free_idx         # remap the column index to the free variable space
        rows[k] = row_h_free_idx         # remap the row index to the free variable space
      else
        rows[k], cols[k] = 1, 1          # Kill the entry: move it to some entry in the free variable space.
        record && push!(nlp.$dead, k)    # Bookkeeping
      end
    end
    return rows, cols
  end
end

# Dead entries for the coord functions. If the corresponding structure function has not been
# called yet, call it once on temporary arrays: it records the dead entries.
for (Model, structure_f!, nnz, dead) in (
  (:FixedVariableEliminationModels, :jac_structure!, :(nlp.meta.nnzj), :jac_dead),
  (:FixedVariableEliminationModels, :jac_lin_structure!, :(nlp.meta.lin_nnzj), :jac_lin_dead),
  (:FixedVariableEliminationModels, :jac_nln_structure!, :(nlp.meta.nln_nnzj), :jac_nln_dead),
  (:FixedVariableEliminationModels, :hess_structure!, :(nlp.meta.nnzh), :hess_dead),
  (:FixedVariableEliminationNLSModel, :jac_structure_residual!, :(nlp.nls_meta.nnzj), :jac_residual_dead,),
  (:FixedVariableEliminationNLSModel, :hess_structure_residual!, :(nlp.nls_meta.nnzh), :hess_residual_dead,),
)
  @eval function $(Symbol(dead, :_entries))(nlp::$Model)
    if is_uninitialized(nlp.$dead) # Initialize the dead entries by calling the corresponding structure function on temporary arrays.
      $structure_f!(nlp, Vector{Int}(undef, $nnz), Vector{Int}(undef, $nnz))
    end
    return nlp.$dead
  end
end

for (Model, coord_f!, nnz, dead) in (
  (:FixedVariableEliminationModels, :jac_coord!, :(nlp.meta.nnzj), :jac_dead),
  (:FixedVariableEliminationModels, :jac_lin_coord!, :(nlp.meta.lin_nnzj), :jac_lin_dead),
  (:FixedVariableEliminationModels, :jac_nln_coord!, :(nlp.meta.nln_nnzj), :jac_nln_dead),
  (:FixedVariableEliminationNLSModel, :jac_coord_residual!, :(nlp.nls_meta.nnzj), :jac_residual_dead,),
)
  @eval function NLPModels.$coord_f!(nlp::$Model, x::AbstractVector, vals::AbstractVector)
    @lencheck nlp.meta.nvar x
    @lencheck $nnz vals

    nlp.x_full[nlp.free] .= x                # load buffer with free variables
    $coord_f!(nlp.model, nlp.x_full, vals)
    vals[$(Symbol(dead, :_entries))(nlp)] .= 0  # zero the entries involving fixed variables
    return vals
  end
end

function NLPModels.hess_coord!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  hess_coord!(nlp.model, nlp.x_full, vals; obj_weight = obj_weight)
  vals[hess_dead_entries(nlp)] .= 0        # zero the entries involving fixed variables
  return vals
end

function NLPModels.hess_coord!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  hess_coord!(nlp.model, nlp.x_full, y, vals; obj_weight = obj_weight)
  vals[hess_dead_entries(nlp)] .= 0        # zero the entries involving fixed variables
  return vals
end

function NLPModels.jth_hess_coord!(
  nlp::FixedVariableEliminationModels,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector,
)
  @lencheck nlp.meta.nvar x
  @rangecheck 1 nlp.meta.ncon j
  @lencheck nlp.meta.nnzh vals

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  jth_hess_coord!(nlp.model, nlp.x_full, j, vals)
  vals[hess_dead_entries(nlp)] .= 0        # zero the entries involving fixed variables
  return vals
end

function NLPModels.hess_coord_residual!(
  nlp::FixedVariableEliminationNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.nls_meta.nequ v
  @lencheck nlp.nls_meta.nnzh vals

  nlp.x_full[nlp.free] .= x                # load buffer with free variables
  hess_coord_residual!(nlp.model, nlp.x_full, v, vals)
  vals[hess_residual_dead_entries(nlp)] .= 0 # zero the entries involving fixed variables
  return vals
end
