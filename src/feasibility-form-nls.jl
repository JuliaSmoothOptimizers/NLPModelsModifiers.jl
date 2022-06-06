export FeasibilityFormNLS

"""Converts a nonlinear least-squares problem with residual ``F(x)`` to a nonlinear
optimization problem with constraints ``F(x) = r`` and objective ``\\tfrac{1}{2}\\|r\\|^2``.
In other words, converts
```math
\\begin{aligned}
       \\min_x \\quad & \\tfrac{1}{2}\\|F(x)\\|^2 \\\\
\\mathrm{s.t.} \\quad & c_L ≤ c(x) ≤ c_U \\\\
                      &   ℓ ≤   x  ≤ u
\\end{aligned}
```
to
```math
\\begin{aligned}
   \\min_{x,r} \\quad & \\tfrac{1}{2}\\|r\\|^2 \\\\
\\mathrm{s.t.} \\quad & F(x) - r = 0 \\\\
                      & c_L ≤ c(x) ≤ c_U \\\\
                      &   ℓ ≤   x  ≤ u
\\end{aligned}
```
If you rather have the first problem, the `nls` model already works as an NLPModel of
that format.
"""
mutable struct FeasibilityFormNLS{T, S, M <: AbstractNLSModel{T, S}} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  internal::M
  counters::NLSCounters
end

function NLPModels.show_header(io::IO, nls::FeasibilityFormNLS)
  println(io, "FeasibilityFormNLS - Nonlinear least-squares moving the residual to constraints")
end

"""
    FeasibilityFormNLS(nls)

Converts a nonlinear least-squares problem with residual `F(x)` to a nonlinear
optimization problem with constraints `F(x) = r` and objective `¹/₂‖r‖²`.
"""
function FeasibilityFormNLS(
  nls::AbstractNLSModel{T, S};
  name = "$(nls.meta.name)-ffnls",
) where {T, S}
  nequ = nls.nls_meta.nequ
  meta = nls.meta
  nvar = meta.nvar + nequ
  ncon = meta.ncon + nequ
  nnzh = nls.nls_meta.nnzh + nequ + (meta.ncon == 0 ? 0 : meta.nnzh) # Some indexes can be repeated
  x0 = similar(meta.x0, nvar)
  x0[1:(meta.nvar)] .= meta.x0
  x0[(meta.nvar + 1):end] .= zero(T)
  lvar = similar(meta.x0, nvar)
  lvar[1:(meta.nvar)] .= meta.lvar
  lvar[(meta.nvar + 1):end] .= T(-Inf)
  uvar = similar(meta.x0, nvar)
  uvar[1:(meta.nvar)] .= meta.uvar
  uvar[(meta.nvar + 1):end] .= T(Inf)
  lcon = similar(meta.y0, ncon)
  lcon[1:nequ] .= zero(T)
  lcon[(nequ + 1):end] .= meta.lcon
  ucon = similar(meta.y0, ncon)
  ucon[1:nequ] .= zero(T)
  ucon[(nequ + 1):end] .= meta.ucon
  y0 = similar(meta.y0, ncon)
  y0[1:nequ] .= zero(T)
  y0[(nequ + 1):end] .= meta.y0
  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    lcon = lcon,
    ucon = ucon,
    y0 = y0,
    lin = meta.lin .+ nequ, # [nls.nls_meta.lin; meta.lin .+ nequ] linear API for residual not (yet) implemented
    nnzj = meta.nnzj + nls.nls_meta.nnzj + nequ,
    nln_nnzj = meta.nln_nnzj + nls.nls_meta.nnzj + nequ,
    lin_nnzj = meta.lin_nnzj,
    nnzh = nnzh,
    name = name,
  )
  nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x0, nnzj = nequ, nnzh = 0, lin = 1:nequ)

  nlp = FeasibilityFormNLS{T, S, typeof(nls)}(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end

function NLPModels.obj(nlp::FeasibilityFormNLS, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  n = nlp.internal.meta.nvar
  r = @view x[(n + 1):end]
  return dot(r, r) / 2
end

function NLPModels.grad!(nlp::FeasibilityFormNLS, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  g[1:n] .= 0.0
  g[(n + 1):end] .= @view x[(n + 1):end]
  return g
end

function NLPModels.objgrad!(nlp::FeasibilityFormNLS, x::Array{Float64}, g::Array{Float64})
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  r = @view x[(n + 1):end]
  f = dot(r, r) / 2
  g[1:n] .= 0.0
  g[(n + 1):end] .= @view x[(n + 1):end]
  return f, g
end

function NLPModels.cons_nln!(nlp::FeasibilityFormNLS, xr::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.nnln c
  increment!(nlp, :neval_cons_nln)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.nnln, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  r = @view xr[(n + 1):end]
  residual!(nlp.internal, x, @view c[1:ne])
  c[1:ne] .-= r
  if m > 0
    cons_nln!(nlp.internal, x, @view c[(ne + 1):end])
  end
  return c
end

function NLPModels.cons_lin!(nlp::FeasibilityFormNLS, xr::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.nlin c
  increment!(nlp, :neval_cons_lin)
  x = @view xr[1:nlp.internal.meta.nvar]
  return cons_lin!(nlp.internal, x, c)
end

function NLPModels.jac_nln_structure!(
  nlp::FeasibilityFormNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nln_nnzj rows cols
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.nnln, nlp.internal.nls_meta.nequ
  nnzjF = nlp.internal.nls_meta.nnzj
  @views jac_structure_residual!(nlp.internal, rows[1:nnzjF], cols[1:nnzjF])
  if m > 0
    idx = nnzjF .+ (1:nlp.internal.meta.nln_nnzj)
    @views jac_nln_structure!(nlp.internal, rows[idx], cols[idx])
    rows[idx] .+= ne
  end
  rows[(end - ne + 1):end] .= 1:ne
  cols[(end - ne + 1):end] .= n .+ (1:ne)
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  nlp::FeasibilityFormNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.lin_nnzj rows cols
  return jac_lin_structure!(nlp.internal, rows, cols)
end

function NLPModels.jac_nln_coord!(nlp::FeasibilityFormNLS, xr::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.nln_nnzj vals
  increment!(nlp, :neval_jac_nln)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.nnln, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  nnzjF = nlp.internal.nls_meta.nnzj
  nnzjc = m > 0 ? nlp.internal.meta.nln_nnzj : 0
  I = 1:nnzjF
  @views jac_coord_residual!(nlp.internal, x, vals[I])
  if m > 0
    I = (nnzjF + 1):(nnzjF + nnzjc)
    @views jac_nln_coord!(nlp.internal, x, vals[I])
  end
  vals[(nnzjF + nnzjc + 1):(nnzjF + nnzjc + ne)] .= -1
  return vals
end

function NLPModels.jac_lin_coord!(nlp::FeasibilityFormNLS, xr::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.lin_nnzj vals
  increment!(nlp, :neval_jac_lin)
  x = @view xr[1:nlp.internal.meta.nvar]
  return jac_lin_coord!(nlp.internal, x, vals)
end

function NLPModels.jprod_nln!(
  nlp::FeasibilityFormNLS,
  xr::AbstractVector,
  v::AbstractVector,
  jv::AbstractVector,
)
  @lencheck nlp.meta.nvar xr v
  @lencheck nlp.meta.nnln jv
  increment!(nlp, :neval_jprod_nln)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.nnln, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jprod_residual!(nlp.internal, x, v[1:n], jv[1:ne])
  @views jv[1:ne] .-= v[(n + 1):end]
  if m > 0
    @views jprod_nln!(nlp.internal, x, v[1:n], jv[(ne + 1):end])
  end
  return jv
end

function NLPModels.jprod_lin!(
  nlp::FeasibilityFormNLS,
  xr::AbstractVector,
  vr::AbstractVector,
  jv::AbstractVector,
)
  @lencheck nlp.meta.nvar xr vr
  @lencheck nlp.meta.nlin jv
  increment!(nlp, :neval_jprod_lin)
  x = @view xr[1:nlp.internal.meta.nvar]
  v = @view vr[1:nlp.internal.meta.nvar]
  return jprod_lin!(nlp.internal, x, v, jv)
end

function NLPModels.jtprod!(
  nlp::FeasibilityFormNLS,
  xr::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
)
  @lencheck nlp.meta.nvar xr jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jtprod_residual!(nlp.internal, x, v[1:ne], jtv[1:n])
  if m > 0
    @views jtv[1:n] .+= jtprod(nlp.internal, x, v[(ne + 1):end])
  end
  @views jtv[(n + 1):end] .= -v[1:ne]
  return jtv
end

function NLPModels.jtprod_nln!(
  nlp::FeasibilityFormNLS,
  xr::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
)
  @lencheck nlp.meta.nvar xr jtv
  @lencheck nlp.meta.nnln v
  increment!(nlp, :neval_jtprod_nln)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.nnln, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jtprod_residual!(nlp.internal, x, v[1:ne], jtv[1:n])
  if m > 0
    @views jtv[1:n] .+= jtprod_nln(nlp.internal, x, v[(ne + 1):end])
  end
  @views jtv[(n + 1):end] .= -v[1:ne]
  return jtv
end

function NLPModels.jtprod_lin!(
  nlp::FeasibilityFormNLS,
  xr::AbstractVector{T},
  v::AbstractVector,
  jtvr::AbstractVector,
) where {T}
  @lencheck nlp.meta.nvar xr jtvr
  @lencheck nlp.meta.nlin v
  increment!(nlp, :neval_jtprod_lin)
  x = @view xr[1:nlp.internal.meta.nvar]
  jtv = @view jtvr[1:nlp.internal.meta.nvar]
  jtprod_lin!(nlp.internal, x, v, jtv)
  @views jtvr[(nlp.internal.meta.nvar + 1):end] .= zero(T)
  return jtvr
end

function NLPModels.hess_structure!(
  nlp::FeasibilityFormNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzh rows cols
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  nnzhF = nlp.internal.nls_meta.nnzh
  nnzhc = m > 0 ? nlp.internal.meta.nnzh : 0
  I = 1:nnzhF
  @views hess_structure_residual!(nlp.internal, rows[I], cols[I])
  if m > 0
    I = (nnzhF + 1):(nnzhF + nnzhc)
    @views hess_structure!(nlp.internal, rows[I], cols[I])
  end
  I = (nnzhF + nnzhc + 1):(nnzhF + nnzhc + ne)
  rows[I] .= (n + 1):(n + ne)
  cols[I] .= (n + 1):(n + ne)
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::FeasibilityFormNLS,
  xr::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(xr)),
)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  nnzhF = nlp.internal.nls_meta.nnzh
  nnzhc = m > 0 ? nlp.internal.meta.nnzh : 0
  x = @view xr[1:n]
  y1 = @view y[1:ne]
  y2 = @view y[(ne + 1):(ne + m)]
  I = 1:nnzhF
  @views hess_coord_residual!(nlp.internal, x, y1, vals[I])
  if m > 0
    I = (nnzhF + 1):(nnzhF + nnzhc)
    @views hess_coord!(nlp.internal, x, y2, vals[I], obj_weight = 0.0)
  end
  vals[(nnzhF + nnzhc + 1):(nnzhF + nnzhc + ne)] .= obj_weight
  return vals
end

function NLPModels.hprod!(
  nlp::FeasibilityFormNLS,
  xr::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Real = one(eltype(xr)),
)
  @lencheck nlp.meta.nvar xr v hv
  @lencheck nlp.meta.ncon y
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  T = eltype(xr)
  if m > 0
    @views hprod!(nlp.internal, x, y[(ne + 1):end], v[1:n], hv[1:n], obj_weight = zero(T))
  else
    fill!(hv, zero(T))
  end
  for i = 1:ne
    @views hv[1:n] .+= hprod_residual(nlp.internal, x, i, v[1:n]) * y[i]
  end
  @views hv[(n + 1):end] .= obj_weight * v[(n + 1):end]
  return hv
end

function NLPModels.ghjvprod!(
  nlp::FeasibilityFormNLS,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  IF = 1:ne
  Ic = (ne + 1):(ne + m)
  gHv[IF] .= [dot(g[1:n], hprod_residual(nlp.internal, x[1:n], j, v[1:n])) for j in IF]
  if m > 0
    @views ghjvprod!(nlp.internal, x[1:n], g[1:n], v[1:n], gHv[Ic])
  end
  return gHv
end

function NLPModels.residual!(nlp::FeasibilityFormNLS, x::AbstractVector, Fx::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.nls_meta.nequ Fx
  increment!(nlp, :neval_residual)
  n = nlp.internal.meta.nvar
  Fx .= @view x[(n + 1):end]
  return Fx
end

function NLPModels.jac_structure_residual!(
  nlp::FeasibilityFormNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.nls_meta.nnzj rows cols
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  rows .= 1:ne
  cols .= n .+ (1:ne)
  return rows, cols
end

function NLPModels.jac_coord_residual!(
  nlp::FeasibilityFormNLS,
  x::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.nls_meta.nnzj vals
  increment!(nlp, :neval_jac_residual)
  vals[1:(nlp.nls_meta.nnzj)] .= 1
  return vals
end

function NLPModels.jprod_residual!(
  nlp::FeasibilityFormNLS,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.nls_meta.nequ Jv
  increment!(nlp, :neval_jprod_residual)
  n = nlp.internal.meta.nvar
  Jv .= @view v[(n + 1):end]
  return Jv
end

function NLPModels.jtprod_residual!(
  nlp::FeasibilityFormNLS,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.nls_meta.nequ v
  increment!(nlp, :neval_jtprod_residual)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  Jtv[1:n] .= zero(eltype(x))
  Jtv[(n + 1):end] .= v
  return Jtv
end

function NLPModels.hess_structure_residual!(
  nlp::FeasibilityFormNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.nls_meta.nnzh rows cols
  return rows, cols  # Hessian of residual is zero; do not change rows and cols
end

function NLPModels.hess_coord_residual!(
  nlp::FeasibilityFormNLS,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.nls_meta.nequ v
  @lencheck nlp.nls_meta.nnzh vals
  increment!(nlp, :neval_hess_residual)
  return vals
end

function NLPModels.jth_hess_residual(nlp::FeasibilityFormNLS, x::AbstractVector, i::Int)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jhess_residual)
  n = nlp.meta.nvar
  return spzeros(eltype(x), n, n)
end

function NLPModels.hprod_residual!(
  nlp::FeasibilityFormNLS,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck nlp.meta.nvar x v Hiv
  increment!(nlp, :neval_hprod_residual)
  fill!(Hiv, zero(eltype(x)))
  return Hiv
end
