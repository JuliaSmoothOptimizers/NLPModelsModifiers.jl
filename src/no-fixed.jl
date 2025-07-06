export NoFixedModel

mutable struct NoFixedModel{
  T,
  S,
  M <: AbstractNLPModel{T, S},
  Meta <: AbstractNLPModelMeta{T, S},
} <: AbstractNLPModel{T, S}
  ifix_mask::BitVector
  meta::Meta
  model::M
  x_copy::S
  temp::S
  temp2::S
  is_implemented_hessian::Bool
  new_rows::Vector{Int}
  new_cols::Vector{Int}
  mask::Vector{Int}
  temp3::S
  jac_lin_rows::Vector{Int}
  jac_lin_cols::Vector{Int}
  jac_nln_rows::Vector{Int}
  jac_nln_cols::Vector{Int}
  temp4::S
  jac_lin_mask::Vector{Int}
  jac_nln_mask::Vector{Int}
end

function compute_ifix_mask(model::AbstractNLPModel)
    n = get_nvar(model)
    ifix = get_ifix(model)
    mask = falses(n)  # BitVector of length n
    for i in ifix
        mask[i] = true
    end
    return mask
end

function NoFixedModel(model::AbstractNLPModel{T, S}) where {T, S}
  x0_full = get_x0(model)
  lvar_full = get_lvar(model)
  uvar_full = get_uvar(model)
  ifix_mask = compute_ifix_mask(model)
  free_indices = findall(.!ifix_mask)

  nvar = get_nvar(model) - length(model.meta.ifix)
  x0 = similar(x0_full, nvar)
  lvar = similar(get_lvar(model), nvar)
  uvar = similar(get_uvar(model), nvar)
  x_copy = similar(x0_full)

  j = 0
  for i = 1:get_nvar(model)
    if !ifix_mask[i]
      j += 1
      x0[j] = x0_full[i]
      lvar[j] = lvar_full[i]
      uvar[j] = uvar_full[i]
    else
      x_copy[i] = lvar_full[i]
    end
  end
  # Prepare hessian structure evaluation
  new_rows = Vector{Int}(undef, get_nnzh(model))
  new_cols = Vector{Int}(undef, get_nnzh(model))
  mask = Vector{Int}(undef, get_nnzh(model))
  is_implemented_hessian = true
  full_to_red = fill(0, get_nvar(model))
  for (j, i) in enumerate(free_indices)
    full_to_red[i] = j
  end

  count_nnzh = 0
  try
    old_rows, old_cols = hess_structure(model)
    for i=1:get_nnzh(model)
      if !(ifix_mask[old_rows[i]] || ifix_mask[old_cols[i]])
        count_nnzh += 1
        mask[count_nnzh] = i
        new_rows[count_nnzh] = full_to_red[old_rows[i]]
        new_cols[count_nnzh] = full_to_red[old_cols[i]]
      end
    end
  catch
    is_implemented_hessian = false
  end
  resize!(new_rows, count_nnzh)
  resize!(new_cols, count_nnzh)
  resize!(mask, count_nnzh)

  jac_lin_rows = Vector{Int}(undef, get_lin_nnzj(model))
  jac_lin_cols = Vector{Int}(undef, get_lin_nnzj(model))
  jac_lin_mask = Vector{Int}(undef, get_lin_nnzj(model))
  jac_nln_rows = Vector{Int}(undef, get_nln_nnzj(model))
  jac_nln_cols = Vector{Int}(undef, get_nln_nnzj(model))
  jac_nln_mask = Vector{Int}(undef, get_nln_nnzj(model))

  lin_rows, lin_cols = jac_lin_structure(model)
  nln_rows, nln_cols = jac_nln_structure(model)
  count_nnzj = 0
  count_lin_nnzj = 0
  count_nln_nnzj = 0

  for (idx, (r, c)) in enumerate(zip(lin_rows, lin_cols))
      if !ifix_mask[c]
        count_lin_nnzj+=1
        jac_lin_rows[count_lin_nnzj] = r
        jac_lin_cols[count_lin_nnzj] = full_to_red[c]
        jac_lin_mask[count_lin_nnzj] = idx 
      end
  end

  for (idx, (r, c)) in enumerate(zip(nln_rows, nln_cols))
      if !ifix_mask[c]
        count_nln_nnzj+=1
        jac_nln_rows[count_nln_nnzj] = r
        jac_nln_cols[count_nln_nnzj] = full_to_red[c]
        jac_nln_mask[count_nln_nnzj] = idx
      end
  end
  count_nnzj = count_lin_nnzj + count_nln_nnzj

  resize!(jac_lin_rows, count_lin_nnzj)
  resize!(jac_lin_cols, count_lin_nnzj)
  resize!(jac_lin_mask, count_lin_nnzj)
  resize!(jac_nln_rows, count_nln_nnzj)
  resize!(jac_nln_cols, count_nln_nnzj)
  resize!(jac_nln_mask, count_nln_nnzj)

  meta = NLPModelMeta{T, S}(
    nvar;
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    nlvb = model.meta.nlvb,
    nlvo = model.meta.nlvo,
    nlvc = model.meta.nlvc,
    ncon = model.meta.ncon,
    y0 = model.meta.y0,
    lcon = model.meta.lcon,
    ucon = model.meta.ucon,
    nnzo = model.meta.nnzo,
    nnzj = count_nnzj,
    lin_nnzj = count_lin_nnzj,
    nln_nnzj = count_nln_nnzj,
    nnzh = count_nnzh,
    lin = model.meta.lin,
    minimize = model.meta.minimize,
    islp = model.meta.islp,
    name = model.meta.name * " (no fixed variables)",
  )
  return NoFixedModel(ifix_mask, meta, model, x_copy, similar(x_copy), similar(x_copy), is_implemented_hessian, new_rows, new_cols, mask, similar(x_copy, get_nnzh(model)), jac_lin_rows, jac_lin_cols, jac_nln_rows, jac_nln_cols, similar(x_copy, get_nnzj(model)), jac_lin_mask, jac_nln_mask)
end

function transform_x(nlp::NoFixedModel{T, S}, x::S) where {T, S}
  @lencheck nlp.meta.nvar x
  x_full = nlp.x_copy
  fixed_value = get_lvar(nlp.model)
  j = 0
  @inbounds @simd for i in 1:get_nvar(nlp.model)
    if nlp.ifix_mask[i]
      x_full[i] = fixed_value[i]
    else
      j += 1
      x_full[i] = x[j]
    end
  end
  return x_full
end

function transform_v!(nlp::NoFixedModel{T, S}, v::S, v_long::S) where {T, S}
  @lencheck nlp.meta.nvar v
  @lencheck nlp.model.meta.nvar v_long
  j = 0
  @inbounds @simd for i in 1:get_nvar(nlp.model)
    if nlp.ifix_mask[i]
      v_long[i] = zero(T)
        else
            j += 1
            v_long[i] = v[j]
        end
    end
    return v_long
end
function get_from_temp!(nlp::NoFixedModel{T, S}, temp::S, g::S) where {T, S}
    @lencheck nlp.meta.nvar g
    j = 0
    @inbounds @simd for i in 1:get_nvar(nlp.model)
        if !nlp.ifix_mask[i]
            j += 1
            g[j] = temp[i]
        end
    end
    return g
end

NLPModels.show_header(io::IO, nlp::NoFixedModel) =
  println(io, "$(typeof(nlp)) - A NLPModel without fixed variables")

function Base.show(io::IO, nlp::NoFixedModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.model.counters)
end

@default_counters NoFixedModel model

# Rely on default hess and jac functions
for meth in (
  :obj,
  :cons,
  :cons_lin,
  :cons_nln,
)
  @eval NLPModels.$meth(nlp::NoFixedModel, x::AbstractVector; kwargs...) =
    $meth(nlp.model, transform_x(nlp, x); kwargs...)
end
for meth in (
  :cons!,
  :cons_lin!,
  :cons_nln!,
)
  @eval NLPModels.$meth(nlp::NoFixedModel, x::AbstractVector, y::AbstractVector; kwargs...) =
    $meth(nlp.model, transform_x(nlp, x), y; kwargs...)
end

for meth in (
  :jprod,
  :jprod_lin,
  :jprod_nln,
)
  @eval function NLPModels.$meth(
    nlp::NoFixedModel,
    x::AbstractVector,
    v::AbstractVector
  )
  transform_v!(nlp, v, nlp.temp)
  Jv = $meth(nlp.model, transform_x(nlp, x), nlp.temp)
  return Jv
  end
end
for meth in (
  :jprod!,
  :jprod_lin!,
  :jprod_nln!,
)
  @eval function NLPModels.$meth(
    nlp::NoFixedModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
  )
  transform_v!(nlp, v, nlp.temp)
  $meth(nlp.model, transform_x(nlp, x), nlp.temp, Jv)
  return Jv
  end
end
for meth in (
  :jtprod,
  :jtprod_lin,
  :jtprod_nln,
)
  @eval function NLPModels.$meth(
    nlp::NoFixedModel,
    x::AbstractVector,
    v::AbstractVector
  )
  Jtv = similar(nlp.meta.x0)
  nlp.temp = $meth(nlp.model, transform_x(nlp, x), v)
  get_from_temp!(nlp, nlp.temp, Jtv)
  return Jtv
  end
end
for meth in (
  :jtprod!,
  :jtprod_lin!,
  :jtprod_nln!,
)
  @eval function NLPModels.$meth(
    nlp::NoFixedModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
  )
  $meth(nlp.model, transform_x(nlp, x), v, nlp.temp)
  get_from_temp!(nlp, nlp.temp, Jtv)
  return Jtv
  end
end

function NLPModels.grad(nlp::NoFixedModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  g = similar(x)
  grad!(nlp.model, transform_x(nlp, x), nlp.temp)
  get_from_temp!(nlp, nlp.temp, g)
  return g
end
function NLPModels.grad!(nlp::NoFixedModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  grad!(nlp.model, transform_x(nlp, x), nlp.temp)
  get_from_temp!(nlp, nlp.temp, g)
end
function NLPModels.objgrad(nlp::NoFixedModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  g = similar(x)
  fx, nlp.temp = objgrad!(nlp.model, transform_x(nlp, x), nlp.temp)
  get_from_temp!(nlp, nlp.temp, g)
  return (fx, g)
end
function NLPModels.objgrad!(nlp::NoFixedModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  fx, nlp.temp = objgrad!(nlp.model, transform_x(nlp, x), nlp.temp)
  get_from_temp!(nlp, nlp.temp, g)
  return (fx, g)
end
function NLPModels.hprod(
  nlp::NoFixedModel,
  x::AbstractVector,
  v::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x v
  Hv = similar(v)
  transform_v!(nlp, v, nlp.temp2)
  hprod!(nlp.model, transform_x(nlp, x), nlp.temp2, nlp.temp; kwargs...)
  get_from_temp!(nlp, nlp.temp, Hv)
  return Hv
end
function NLPModels.hprod!(
  nlp::NoFixedModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x v
  transform_v!(nlp, v, nlp.temp2)
  hprod!(nlp.model, transform_x(nlp, x), nlp.temp2, nlp.temp; kwargs...)
  get_from_temp!(nlp, nlp.temp, Hv)
end
function NLPModels.hprod(
  nlp::NoFixedModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon y
  Hv = similar(v)
  transform_v!(nlp, v, nlp.temp2)
  hprod!(nlp.model, transform_x(nlp, x), y, nlp.temp2, nlp.temp; kwargs...)
  get_from_temp!(nlp, nlp.temp, Hv)
  return Hv
end
function NLPModels.hprod!(
  nlp::NoFixedModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon y
  transform_v!(nlp, v, nlp.temp2)
  hprod!(nlp.model, transform_x(nlp, x), y, nlp.temp2, nlp.temp; kwargs...)
  get_from_temp!(nlp, nlp.temp, Hv)
end

function NLPModels.ghjvprod!(
  nlp::NoFixedModel,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nlp.meta.nvar x g v
  transform_v!(nlp, g, nlp.temp)
  transform_v!(nlp, v, nlp.temp2)
  ghjvprod!(nlp.model, transform_x(nlp, x), nlp.temp, nlp.temp2, gHv)
  return gHv
end

function NLPModels.jac_lin_structure!(nlp::NoFixedModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    rows .= nlp.jac_lin_rows
    cols .= nlp.jac_lin_cols
    return rows, cols
end
function NLPModels.jac_nln_structure!(nlp::NoFixedModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    rows .= nlp.jac_nln_rows
    cols .= nlp.jac_nln_cols
    return rows, cols
end
function NLPModels.jac_lin_coord!(nlp::NoFixedModel, x::AbstractVector, vals::AbstractVector)
    NLPModels.jac_lin_coord!(nlp.model, transform_x(nlp, x), view(nlp.temp4, 1:(get_lin_nnzj(nlp.model))))
    for (k, i) in enumerate(nlp.jac_lin_mask)
        vals[k] = nlp.temp4[i]
    end
    return vals
end
function NLPModels.jac_nln_coord!(nlp::NoFixedModel, x::AbstractVector, vals::AbstractVector)
    NLPModels.jac_nln_coord!(nlp.model, transform_x(nlp, x), view(nlp.temp4, 1:(get_nln_nnzj(nlp.model))))
    for (k, i) in enumerate(nlp.jac_nln_mask)
        vals[k] = nlp.temp4[i]
    end
    return vals
end

function NLPModels.hess_coord!(
  nlp::NoFixedModel,
  x::AbstractVector,
  vals::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x
  if !nlp.is_implemented_hessian
    throw(ErrorException("Not implemented hess_structure! for original problem $(typeof(nlp.model))"))
  end
  hess_coord!(nlp.model, transform_x(nlp, x), nlp.temp3; kwargs...)
  @inbounds @simd for k in 1:length(nlp.mask)
  vals[k] = nlp.temp3[nlp.mask[k]]
end
  return vals
end
function NLPModels.hess_coord!(
  nlp::NoFixedModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  if !nlp.is_implemented_hessian
    throw(ErrorException("Not implemented hess_structure! for original problem $(typeof(nlp.model))"))
  end
  hess_coord!(nlp.model, transform_x(nlp, x), y, nlp.temp3; kwargs...)
  @inbounds @simd for k in 1:length(nlp.mask)
  vals[k] = nlp.temp3[nlp.mask[k]]
end
  return vals
end
function NLPModels.hess_structure!(
  nlp::NoFixedModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if !nlp.is_implemented_hessian
    throw(ErrorException("Not implemented hess_structure! for original problem $(typeof(nlp.model))"))
  end
  return nlp.new_rows, nlp.new_cols
end

function NLPModels.jth_hess_coord!(
  nlp::NoFixedModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x
  @rangecheck 1 nlp.meta.ncon j
  if !nlp.is_implemented_hessian
    throw(ErrorException("Not implemented hess_structure! for original problem $(typeof(nlp.model))"))
  end
  jth_hess_coord!(nlp.model, transform_x(nlp, x), j, nlp.temp3; kwargs...)
  @inbounds @simd for k in 1:length(nlp.mask)
    vals[k] = nlp.temp3[nlp.mask[k]]
  end
  return vals
end
function NLPModels.jth_hprod!(
  nlp::NoFixedModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector;
  kwargs...,
)
  @lencheck nlp.meta.nvar x v Hv
  @rangecheck 1 nlp.meta.ncon j
  transform_v!(nlp, v, nlp.temp2)
  jth_hprod!(nlp.model, transform_x(nlp, x), nlp.temp2, j, nlp.temp; kwargs...)
  get_from_temp!(nlp, nlp.temp, Hv)
end
