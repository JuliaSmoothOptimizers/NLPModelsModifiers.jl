"""
    remove_fixed_variables.jl

Standalone adaptation of the "MakeParameter" fixed-variable elimination
logic found in MadNLP.jl's `SparseCallback` (see MadNLP.jl/src/utils.jl).

Instead of wrapping your model inside MadNLP's full callback machinery,
this file provides a single function, `remove_fixed_variables`, that you
can call once before handing your problem to any NLPModels.jl-compatible
solver (e.g. ExactPenalty.jl, Ipopt, etc.):

    julia> nlp = MyNLPModel(...)
    julia> nlp_free = remove_fixed_variables(nlp)
    julia> stats = my_solver(nlp_free)
    julia> x_full = recover_full_solution(nlp_free, stats.solution)

If `nlp` has no fixed variables (i.e. no `i` with `lvar[i] == uvar[i]`),
`remove_fixed_variables` returns `nlp` unchanged, and `recover_full_solution`
is a no-op in that case too, so it is always safe to call both.
"""

export FixedVariableEliminationModel, remove_fixed_variables, recover_full_solution

"""
    FixedVariableEliminationModel(nlp, free, fixed, ...) <: AbstractNLPModel{T,S}

A thin wrapper around `nlp` that hides the variables in `fixed` (those with
`lvar[i] == uvar[i]`) from the optimizer. Fixed variables are kept at their
bound value and are transparently substituted back in whenever the
underlying model `nlp` is evaluated.

Do not construct this directly; use [`remove_fixed_variables`](@ref).
"""
struct FixedVariableEliminationModel{
  T,
  S<:AbstractVector{T},
  M<:AbstractNLPModel{T,S},
  Meta<:AbstractNLPModelMeta{T,S},
} <: AbstractNLPModel{T,S}
  meta::Meta
  counters::Counters
  model::M

  free::Vector{Int}   # indices (in the full model) of the free variables
  fixed::Vector{Int}  # indices (in the full model) of the fixed variables

  ind_jac_free::Vector{Int}  # indices into the *full* Jacobian nnz vector kept after elimination
  ind_hess_free::Vector{Int} # indices into the *full* Hessian nnz vector kept after elimination

  jac_rows::Vector{Int}  # reduced Jacobian sparsity pattern (already remapped to free-variable space)
  jac_cols::Vector{Int}
  hess_rows::Vector{Int} # reduced Hessian sparsity pattern (already remapped to free-variable space)
  hess_cols::Vector{Int}

  x_full::S    # length nvar(nlp); fixed entries permanently hold their bound value
  g_full::S    # scratch buffer, length nvar(nlp)
  v_full::S    # scratch buffer, length nvar(nlp), used for jprod!/hprod!
  jac_buffer::S  # scratch buffer, length nnzj(nlp)
  hess_buffer::S # scratch buffer, length nnzh(nlp)
end

get_model(nlp::FixedVariableEliminationModel) = nlp.model

"""
    remove_fixed_variables(nlp::AbstractNLPModel)

Return a new `AbstractNLPModel` in which every variable `i` with
`lvar[i] == uvar[i]` has been removed from the optimization and is instead
treated as a fixed parameter equal to its bound. The returned model has
`nvar` reduced by the number of fixed variables, and behaves exactly like
`nlp` in every other respect (same objective, same constraints, evaluated
with the fixed variables substituted in).

If `nlp` has no fixed variables, `nlp` itself is returned unchanged (no
wrapping, no overhead).

Use [`recover_full_solution`](@ref) to map a solution of the reduced
problem back to the original variable space.
"""
function remove_fixed_variables(nlp::AbstractNLPModel{T,S}) where {T,S}
  lvar = get_lvar(nlp)
  uvar = get_uvar(nlp)
  isfixed = lvar .== uvar

  nfixed = count(isfixed)
  if nfixed == 0
    return nlp
  end

  n = get_nvar(nlp)
  free = findall(.!isfixed)
  fixed = findall(isfixed)
  nfree = length(free)

  isfree = .!isfixed
  map_full_to_free = fill(-1, n)
  map_full_to_free[free] .= 1:nfree

  # --- Jacobian sparsity ---
  nnzj_full = get_nnzj(nlp.meta)
  jac_rows_full = Vector{Int}(undef, nnzj_full)
  jac_cols_full = Vector{Int}(undef, nnzj_full)
  if nnzj_full > 0
    NLPModels.jac_structure!(nlp, jac_rows_full, jac_cols_full)
  end
  ind_jac_free = findall(@view isfree[jac_cols_full])
  jac_rows = jac_rows_full[ind_jac_free]
  jac_cols = map_full_to_free[jac_cols_full[ind_jac_free]]

  # --- Hessian sparsity (lower triangle assumed, as in NLPModels convention) ---
  nnzh_full = get_nnzh(nlp.meta)
  hess_rows_full = Vector{Int}(undef, nnzh_full)
  hess_cols_full = Vector{Int}(undef, nnzh_full)
  if nnzh_full > 0
    NLPModels.hess_structure!(nlp, hess_rows_full, hess_cols_full)
  end
  ind_hess_free =
    findall(k -> isfree[hess_rows_full[k]] && isfree[hess_cols_full[k]], 1:nnzh_full)
  hess_rows = map_full_to_free[hess_rows_full[ind_hess_free]]
  hess_cols = map_full_to_free[hess_cols_full[ind_hess_free]]

  # --- Buffers ---
  x_full = copy(get_x0(nlp))
  x_full[fixed] .= lvar[fixed]  # fixed variables permanently sit at their bound
  g_full = similar(x_full)
  v_full = similar(x_full)
  jac_buffer = similar(x_full, nnzj_full)
  hess_buffer = similar(x_full, nnzh_full)

  x0 = get_x0(nlp)[free]
  new_lvar = lvar[free]
  new_uvar = uvar[free]

  ncon = get_ncon(nlp)
  y0 = ncon > 0 ? get_y0(nlp) : similar(x0, 0)
  lcon = ncon > 0 ? get_lcon(nlp) : similar(x0, 0)
  ucon = ncon > 0 ? get_ucon(nlp) : similar(x0, 0)

  meta = NLPModelMeta(
    nfree;
    x0 = x0,
    lvar = new_lvar,
    uvar = new_uvar,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = length(ind_jac_free),
    nnzh = length(ind_hess_free),
    lin = nlp.meta.lin,
    minimize = nlp.meta.minimize,
    islp = false,
    name = string(nlp.meta.name, " (", nfixed, " fixed variable(s) removed)"),
  )

  return FixedVariableEliminationModel(
    meta,
    Counters(),
    nlp,
    free,
    fixed,
    ind_jac_free,
    ind_hess_free,
    jac_rows,
    jac_cols,
    hess_rows,
    hess_cols,
    x_full,
    g_full,
    v_full,
    jac_buffer,
    hess_buffer,
  )
end

"""
    recover_full_solution(nlp, x)

Return `x` mapped back to the original variable space, restoring any fixed
variables to their bound. Unwraps through any number of `get_model`-defined
wrappers (e.g. `ShiftedConstraintModel`, `QuasiNewtonModel`) to find the
underlying `FixedVariableEliminationModel`. Returns `x` unchanged if none
is found.
"""
function recover_full_solution(nlp::FixedVariableEliminationModel, x::AbstractVector)
  x_full = copy(nlp.x_full)
  x_full[nlp.free] .= x
  return x_full
end

recover_full_solution(nlp::AbstractNLPModel, x::AbstractVector) =
  applicable(get_model, nlp) ? recover_full_solution(get_model(nlp), x) : x

# ------------------------------------------------------------------------
# NLPModels API
# ------------------------------------------------------------------------

function _update_x_full!(nlp::FixedVariableEliminationModel, x::AbstractVector)
  nlp.x_full[nlp.free] .= x
  return nlp.x_full
end

function NLPModels.obj(nlp::FixedVariableEliminationModel, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_obj)
  x_full = _update_x_full!(nlp, x)
  return NLPModels.obj(nlp.model, x_full)
end

function NLPModels.grad!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  g::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_grad)
  x_full = _update_x_full!(nlp, x)
  NLPModels.grad!(nlp.model, x_full, nlp.g_full)
  g .= @view nlp.g_full[nlp.free]
  return g
end

function NLPModels.cons!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  c::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_cons)
  x_full = _update_x_full!(nlp, x)
  NLPModels.cons!(nlp.model, x_full, c)
  return c
end

function NLPModels.jac_structure!(
  nlp::FixedVariableEliminationModel,
  rows::AbstractVector,
  cols::AbstractVector,
)
  rows .= nlp.jac_rows
  cols .= nlp.jac_cols
  return rows, cols
end

function NLPModels.jac_coord!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jac)
  x_full = _update_x_full!(nlp, x)
  NLPModels.jac_coord!(nlp.model, x_full, nlp.jac_buffer)
  vals .= @view nlp.jac_buffer[nlp.ind_jac_free]
  return vals
end

function NLPModels.jprod!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jprod)
  x_full = _update_x_full!(nlp, x)
  fill!(nlp.v_full, zero(eltype(nlp.v_full)))
  nlp.v_full[nlp.free] .= v
  NLPModels.jprod!(nlp.model, x_full, nlp.v_full, Jv)
  return Jv
end

function NLPModels.jtprod!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jtprod)
  x_full = _update_x_full!(nlp, x)
  NLPModels.jtprod!(nlp.model, x_full, v, nlp.g_full)
  Jtv .= @view nlp.g_full[nlp.free]
  return Jtv
end

function NLPModels.hess_structure!(
  nlp::FixedVariableEliminationModel,
  rows::AbstractVector,
  cols::AbstractVector,
)
  rows .= nlp.hess_rows
  cols .= nlp.hess_cols
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(eltype(x)),
)
  NLPModels.increment!(nlp, :neval_hess)
  x_full = _update_x_full!(nlp, x)
  NLPModels.hess_coord!(nlp.model, x_full, nlp.hess_buffer; obj_weight = obj_weight)
  vals .= @view nlp.hess_buffer[nlp.ind_hess_free]
  return vals
end

function NLPModels.hess_coord!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(eltype(x)),
)
  NLPModels.increment!(nlp, :neval_hess)
  x_full = _update_x_full!(nlp, x)
  NLPModels.hess_coord!(nlp.model, x_full, y, nlp.hess_buffer; obj_weight = obj_weight)
  vals .= @view nlp.hess_buffer[nlp.ind_hess_free]
  return vals
end

function NLPModels.hprod!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(eltype(x)),
)
  NLPModels.increment!(nlp, :neval_hprod)
  x_full = _update_x_full!(nlp, x)
  fill!(nlp.v_full, zero(eltype(nlp.v_full)))
  nlp.v_full[nlp.free] .= v
  Hv_full = nlp.g_full  # reuse scratch buffer
  NLPModels.hprod!(nlp.model, x_full, nlp.v_full, Hv_full; obj_weight = obj_weight)
  Hv .= @view Hv_full[nlp.free]
  return Hv
end

function NLPModels.hprod!(
  nlp::FixedVariableEliminationModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(eltype(x)),
)
  NLPModels.increment!(nlp, :neval_hprod)
  x_full = _update_x_full!(nlp, x)
  fill!(nlp.v_full, zero(eltype(nlp.v_full)))
  nlp.v_full[nlp.free] .= v
  Hv_full = nlp.g_full  # reuse scratch buffer
  NLPModels.hprod!(nlp.model, x_full, y, nlp.v_full, Hv_full; obj_weight = obj_weight)
  Hv .= @view Hv_full[nlp.free]
  return Hv
end