export SlackModel, SlackNLSModel

@doc raw"""A model whose only inequality constraints are bounds.

Given a model, this type represents a second model in which slack variables are
introduced so as to convert linear and nonlinear inequality constraints to
equality constraints and bounds. More precisely, if the original model has the
form

```math
\begin{aligned}
       \min_x \quad & f(x)\\
\mathrm{s.t.} \quad & c_L ≤ c(x) ≤ c_U,\\
                    &  ℓ  ≤  x   ≤  u,
\end{aligned}
```

the new model appears to the user as

```math
\begin{aligned}
       \min_X \quad & f(X)\\
\mathrm{s.t.} \quad & g(X) = 0,\\
                    & L ≤ X ≤ U.
\end{aligned}
```

The unknowns ``X = (x, s)`` contain the original variables and slack variables
``s``. The latter are such that the new model has the general form

```math
\begin{aligned}
       \min_x \quad & f(x)\\
\mathrm{s.t.} \quad & c(x) - s = 0,\\
                    & c_L ≤ s ≤ c_U,\\
                    &  ℓ  ≤ x ≤ u.
\end{aligned}
```

although no slack variables are introduced for equality constraints.

The slack variables are implicitly ordered as `[s(low), s(upp), s(rng)]`, where
`low`, `upp` and `rng` represent the indices of the constraints of the form
``c_L ≤ c(x) < ∞``, ``-∞ < c(x) ≤ c_U`` and
``c_L ≤ c(x) ≤ c_U``, respectively.
"""
mutable struct SlackModel{T, S, M <: AbstractNLPModel{T, S}} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  model::M
end

NLPModels.show_header(io::IO, nlp::SlackModel) =
  println(io, "SlackModel - Model with slack variables")

function Base.show(io::IO, nlp::SlackModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.model.counters)
end

"""Like `SlackModel`, this model converts inequalities into equalities and bounds.
"""
mutable struct SlackNLSModel{T, S, M <: AbstractNLPModel{T, S}} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  model::M
end

NLPModels.show_header(io::IO, nls::SlackNLSModel) =
  println(io, "SlackNLSModel - Nonlinear least-squares model with slack variables")

function Base.show(io::IO, nls::SlackNLSModel)
  show_header(io, nls)
  show(io, nls.meta, nls.nls_meta)
  show(io, nls.model.counters)
end

function slack_meta(meta::NLPModelMeta{T, S}; name = meta.name * "-slack") where {T, S}
  ns = meta.ncon - length(meta.jfix)
  jlow = meta.jlow
  jupp = meta.jupp
  jrng = meta.jrng

  # Don't introduce slacks for equality constraints!
  lvar = [meta.lvar; meta.lcon[[jlow; jupp; jrng]]]  # l ≤ x  and  cₗ ≤ s
  uvar = [meta.uvar; meta.ucon[[jlow; jupp; jrng]]]  # x ≤ u  and  s ≤ cᵤ
  lcon = similar(meta.x0, meta.ncon)
  lcon[setdiff(1:(meta.ncon), meta.jfix)] .= zero(T)
  lcon[meta.jfix] = meta.lcon[meta.jfix]
  ucon = similar(meta.x0, meta.ncon)
  ucon[setdiff(1:(meta.ncon), meta.jfix)] .= zero(T)
  ucon[meta.jfix] = meta.ucon[meta.jfix]

  x0 = similar(meta.x0, meta.nvar + ns)
  x0[1:(meta.nvar)] .= meta.x0
  x0[(meta.nvar):end] .= zero(T)
  return NLPModelMeta(
    meta.nvar + ns,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = meta.ncon,
    lcon = lcon,
    ucon = ucon,
    y0 = meta.y0,
    nnzj = meta.nnzj + ns,
    nnzh = meta.nnzh,
    lin = meta.lin,
    nln = meta.nln,
    name = name,
    minimize = meta.minimize,
  )
end

"Construct a `SlackModel` from another type of model."
function SlackModel(model::AbstractNLPModel; name = model.meta.name * "-slack")
  model.meta.ncon == length(model.meta.jfix) && return model

  meta = slack_meta(model.meta, name = name)

  snlp = SlackModel(meta, model)
  finalizer(nlp -> finalize(nlp.model), snlp)

  return snlp
end

function SlackNLSModel(
  model::AbstractNLSModel{T, S};
  name = model.meta.name * "-slack",
) where {T, S}
  ns = model.meta.ncon - length(model.meta.jfix)
  ns == 0 && return model

  meta = slack_meta(model.meta, name = name)
  x0 = similar(model.meta.x0, model.meta.nvar + ns)
  x0[(model.meta.nvar):end] .= zero(T)
  x0[1:(model.meta.nvar)] .= model.meta.x0
  nls_meta = NLSMeta{T, S}(
    model.nls_meta.nequ,
    model.meta.nvar + ns,
    x0 = x0,
    nnzj = model.nls_meta.nnzj,
    nnzh = model.nls_meta.nnzh,
    lin = model.nls_meta.lin,
    nln = model.nls_meta.nln,
  )

  snls = SlackNLSModel(meta, nls_meta, model)
  finalizer(nls -> finalize(nls.model), snls)

  return snls
end

const SlackModels = Union{SlackModel, SlackNLSModel}

# retrieve counters from underlying model
@default_counters SlackModels model
@default_nlscounters SlackNLSModel model

nls_meta(nlp::SlackNLSModel) = nlp.nls_meta

function NLPModels.obj(nlp::SlackModels, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  # f(X) = f(x)
  return obj(nlp.model, @view x[1:(nlp.model.meta.nvar)])
end

function NLPModels.grad!(nlp::SlackModels, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  # ∇f(X) = [∇f(x) ; 0]
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views grad!(nlp.model, x[1:n], g[1:n])
  g[(n + 1):(n + ns)] .= 0
  return g
end

function NLPModels.objgrad!(nlp::SlackModels, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views f, _ = objgrad!(nlp.model, x[1:n], g[1:n])
  g[(n + 1):(n + ns)] .= 0
  return f, g
end

function NLPModels.cons!(nlp::SlackModels, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  @views begin
    cons!(nlp.model, x[1:n], c)
    c[nlp.model.meta.jlow] -= x[(n + 1):(n + nlow)]
    c[nlp.model.meta.jupp] -= x[(n + nlow + 1):(n + nlow + nupp)]
    c[nlp.model.meta.jrng] -= x[(n + nlow + nupp + 1):(n + nlow + nupp + nrng)]
  end
  return c
end

function NLPModels.jac_structure!(
  nlp::SlackModels,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzj rows cols
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  nnzj = nlp.model.meta.nnzj
  @views jac_structure!(nlp.model, rows[1:nnzj], cols[1:nnzj])
  jlow = nlp.model.meta.jlow
  jupp = nlp.model.meta.jupp
  jrng = nlp.model.meta.jrng
  nj, lj = nnzj, length(jlow)
  rows[(nj + 1):(nj + lj)] .= jlow
  nj, lj = nj + lj, length(jupp)
  rows[(nj + 1):(nj + lj)] .= jupp
  nj, lj = nj + lj, length(jrng)
  rows[(nj + 1):(nj + lj)] .= jrng
  cols[(nnzj + 1):end] .= (n + 1):(nlp.meta.nvar)
  return rows, cols
end

function NLPModels.jac_coord!(nlp::SlackModels, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  n = nlp.model.meta.nvar
  nnzj = nlp.model.meta.nnzj
  @views jac_coord!(nlp.model, x[1:n], vals[1:nnzj])
  vals[(nnzj + 1):(nlp.meta.nnzj)] .= -1
  return vals
end

function NLPModels.jprod!(
  nlp::SlackModels,
  x::AbstractVector,
  v::AbstractVector,
  jv::AbstractVector,
)
  # J(X) V = [J(x)  -I] [vₓ] = J(x) vₓ - vₛ
  #                     [vₛ]
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon jv
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  @views jprod!(nlp.model, x[1:n], v[1:n], jv)
  k = 1
  # use 3 loops to avoid forming [jlow ; jupp ; jrng]
  for j in nlp.model.meta.jlow
    jv[j] -= v[n + k]
    k += 1
  end
  for j in nlp.model.meta.jupp
    jv[j] -= v[n + k]
    k += 1
  end
  for j in nlp.model.meta.jrng
    jv[j] -= v[n + k]
    k += 1
  end
  return jv
end

function NLPModels.jtprod!(
  nlp::SlackModels,
  x::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
)
  # J(X)ᵀ v = [J(x)ᵀ] v = [J(x)ᵀ v]
  #           [ -I  ]     [  -v   ]
  @lencheck nlp.meta.nvar x jtv
  @lencheck nlp.meta.ncon v
  n = nlp.model.meta.nvar
  nlow = length(nlp.model.meta.jlow)
  nupp = length(nlp.model.meta.jupp)
  nrng = length(nlp.model.meta.jrng)
  @views begin
    jtprod!(nlp.model, x[1:n], v, jtv[1:n])
    jtv[(n + 1):(n + nlow)] = -v[nlp.model.meta.jlow]
    jtv[(n + nlow + 1):(n + nlow + nupp)] = -v[nlp.model.meta.jupp]
    jtv[(n + nlow + nupp + 1):(nlp.meta.nvar)] = -v[nlp.model.meta.jrng]
  end
  return jtv
end

function NLPModels.hess_structure!(
  nlp::SlackModels,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzh rows cols
  return hess_structure!(nlp.model, rows, cols)
end

function NLPModels.hess_coord!(
  nlp::SlackModels,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  n = nlp.model.meta.nvar
  return hess_coord!(nlp.model, view(x, 1:n), vals, obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  nlp::SlackModels,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  n = nlp.model.meta.nvar
  return hess_coord!(nlp.model, view(x, 1:n), y, vals, obj_weight = obj_weight)
end

# Kept in case some model implements `hess` but not `hess_coord/structure`
function NLPModels.hess(nlp::SlackModels, x::AbstractVector{T}; kwargs...) where {T}
  @lencheck nlp.meta.nvar x
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  Hx = hess(nlp.model, view(x, 1:n); kwargs...).data
  return Symmetric([Hx spzeros(T, n, ns); spzeros(T, ns, n + ns)], :L)
end

function NLPModels.hess(
  nlp::SlackModels,
  x::AbstractVector{T},
  y::AbstractVector{T};
  kwargs...,
) where {T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  Hx = hess(nlp.model, view(x, 1:n), y; kwargs...).data
  return Symmetric([Hx spzeros(T, n, ns); spzeros(T, ns, n + ns)], :L)
end

function NLPModels.hprod!(
  nlp::SlackModels,
  x::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v hv
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  # using hv[1:n] doesn't seem to work here
  @views hprod!(nlp.model, x[1:n], v[1:n], hv[1:n], obj_weight = obj_weight)
  hv[(n + 1):(nlp.meta.nvar)] .= 0
  return hv
end

function NLPModels.hprod!(
  nlp::SlackModels,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v hv
  @lencheck nlp.meta.ncon y
  n = nlp.model.meta.nvar
  ns = nlp.meta.nvar - n
  # using hv[1:n] doesn't seem to work here
  @views hprod!(nlp.model, x[1:n], y, v[1:n], hv[1:n], obj_weight = obj_weight)
  hv[(n + 1):(nlp.meta.nvar)] .= 0
  return hv
end

function NLPModels.ghjvprod!(
  nlp::SlackModels,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  n = nlp.model.meta.nvar
  return ghjvprod!(nlp.model, view(x, 1:n), view(g, 1:n), view(v, 1:n), gHv)
end

function NLPModels.residual!(nls::SlackNLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  return residual!(nls.model, view(x, 1:(nls.model.meta.nvar)), Fx)
end

function NLPModels.jac_residual(nls::SlackNLSModel, x::AbstractVector{T}) where {T}
  @lencheck nls.meta.nvar x
  n = nls.model.meta.nvar
  ns = nls.meta.nvar - n
  ne = nls.nls_meta.nequ
  Jx = jac_residual(nls.model, @view x[1:n])
  if issparse(Jx)
    return [Jx spzeros(T, ne, ns)]
  else
    return [Jx zeros(T, ne, ns)]
  end
end

function NLPModels.jac_structure_residual!(
  nls::SlackNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.nls_meta.nnzj rows
  @lencheck nls.nls_meta.nnzj cols
  return jac_structure_residual!(nls.model, rows, cols)
end

function NLPModels.jac_coord_residual!(nls::SlackNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  return jac_coord_residual!(nls.model, view(x, 1:(nls.model.meta.nvar)), vals)
end

function NLPModels.jprod_residual!(
  nls::SlackNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nls.meta.nvar x v
  @lencheck nls.nls_meta.nequ Jv
  return jprod_residual!(
    nls.model,
    view(x, 1:(nls.model.meta.nvar)),
    v[1:(nls.model.meta.nvar)],
    Jv,
  )
end

function NLPModels.jtprod_residual!(
  nls::SlackNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nequ v
  n = nls.model.meta.nvar
  ns = nls.meta.nvar - n
  @views jtprod_residual!(nls.model, x[1:n], v, Jtv[1:n])
  Jtv[(n + 1):(n + ns)] .= 0
  return Jtv
end

function NLPModels.jac_op_residual!(
  nls::SlackNLSModel,
  x::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nequ Jv
  prod! = @closure (res, v, α, β) -> begin
    jprod_residual!(nls, x, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_residual!(nls, x, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{eltype(x)}(
    nls_meta(nls).nequ,
    nls_meta(nls).nvar,
    false,
    false,
    prod!,
    ctprod!,
    ctprod!,
  )
end

function NLPModels.hess_residual(
  nls::SlackNLSModel,
  x::AbstractVector{T},
  v::AbstractVector{T},
) where {T}
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  n = nls.model.meta.nvar
  ns = nls.meta.nvar - n
  Hx = hess_residual(nls.model, view(x, 1:n), v)
  if issparse(Hx)
    return [Hx spzeros(T, n, ns); spzeros(T, ns, n + ns)]
  else
    return [Hx zeros(T, n, ns); zeros(T, ns, n + ns)]
  end
end

function NLPModels.hess_structure_residual!(
  nls::SlackNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.nls_meta.nnzh rows cols
  return hess_structure_residual!(nls.model, rows, cols)
end

function NLPModels.hess_coord_residual!(
  nls::SlackNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.nls_meta.nnzh vals
  return hess_coord_residual!(nls.model, view(x, 1:(nls.model.meta.nvar)), v, vals)
end

function NLPModels.jth_hess_residual(nls::SlackNLSModel, x::AbstractVector{T}, i::Int) where {T}
  @lencheck nls.meta.nvar x
  n = nls.model.meta.nvar
  ns = nls.meta.nvar - n
  Hx = jth_hess_residual(nls.model, view(x, 1:n), i)
  if issparse(Hx)
    return [Hx spzeros(T, n, ns); spzeros(T, ns, n + ns)]
  else
    return [Hx zeros(T, n, ns); zeros(T, ns, n + ns)]
  end
end

function NLPModels.hprod_residual!(
  nls::SlackNLSModel,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hv::AbstractVector,
)
  @lencheck nls.meta.nvar x v Hv
  n = nls.model.meta.nvar
  ns = nls.meta.nvar - n
  @views hprod_residual!(nls.model, x[1:n], i, v[1:n], Hv[1:n])
  Hv[(n + 1):(n + ns)] .= 0
  return Hv
end

function NLPModels.hess_op_residual!(
  nls::SlackNLSModel,
  x::AbstractVector,
  i::Int,
  Hiv::AbstractVector,
)
  @lencheck nls.meta.nvar x Hiv
  prod! = @closure (res, v, α, β) -> begin
    hprod_residual!(nls, x, i, v, Hiv)
    if β == 0
      @. res = α * Hiv
    else
      @. res = α * Hiv + β * res
    end
    return res
  end
  return LinearOperator{eltype(x)}(
    nls_meta(nls).nvar,
    nls_meta(nls).nvar,
    true,
    true,
    prod!,
    prod!,
    prod!,
  )
end
