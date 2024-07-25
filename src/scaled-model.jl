export ScaledModel

struct IpoptScaling{T}
  max_gradient::T
end

function _set_constraints_scaling!(cons, Ji, Jj, Jx, max_gradient)
  # Return a vector storing at index i norm(∇cᵢ, Inf)
  for (i, j, x) in zip(Ji, Jj, Jx)
    cons[i] = max(cons[i], abs(x))
  end
  # Compute scaling as min(1, max_gradient / norm(∇cᵢ, Inf) )
  for i in eachindex(cons)
    cons[i] = min(1.0, max_gradient / cons[i])
  end
end

function _set_jacobian_scaling!(Jx, Ji, Jj, cons)
  k = 0
  for (i, j) in zip(Ji, Jj)
    Jx[k += 1] = cons[i]
  end
end

function scale_model!(scaling::IpoptScaling{T}, nlp) where T
  n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
  nnzj = NLPModels.get_nnzj(nlp)
  x0 = NLPModels.get_x0(nlp)
  g = NLPModels.grad(nlp, x0)
  scaling_obj = min(one(T), scaling.max_gradient / norm(g, Inf))
  scaling_cons = similar(x0, m)
  scaling_jac  = similar(x0, nnzj)
  fill!(scaling_cons, zero(T))
  Ji, Jj = NLPModels.jac_structure(nlp)
  NLPModels.jac_coord!(nlp, x0, scaling_jac)
  _set_constraints_scaling!(scaling_cons, Ji, Jj, scaling_jac, scaling.max_gradient)
  _set_jacobian_scaling!(scaling_jac, Ji, Jj, scaling_cons)
  return (scaling_obj, scaling_cons, scaling_jac)
end

@doc raw"""
    ScaledModel

Scale the nonlinear program
```math
\begin{aligned}
       min_x  \quad & f(x)\\
\mathrm{s.t.} \quad &  c♭ ≤ c(x) ≤ c♯ \\
                    & x ≥ 0
\end{aligned}
```
as
```math
\begin{aligned}
       min_x  \quad & σf . f(x)\\
\mathrm{s.t.} \quad &  σc . c♭ ≤ σc . c(x) ≤ σc . c♯ \\
                    & x ≥ 0
\end{aligned}
```
with ``σf`` a scalar defined as
```
σf = min(1, max_gradient / norm(g0, Inf))

```
and ``σc`` a vector whose size is equal to the number of constraints in the model.
For ``i=1, ..., m``,
```
σc[i] = min(1, max_gradient / norm(J0[i, :], Inf))

```

The vector ``g0 = ∇f(x0)`` and the matrix ``J0 = ∇c(x0)`` are resp.
the gradient and the Jacobian evaluated at the initial point ``x0``.

"""
struct ScaledModel{T, S, M} <: NLPModels.AbstractNLPModel{T, S}
  nlp::M
  meta::NLPModels.NLPModelMeta{T, S}
  counters::NLPModels.Counters
  scaling_obj::T
  scaling_cons::S # [size m]
  scaling_jac::S  # [size nnzj]
  buffer_cons::S  # [size m]
end

NLPModels.show_header(io::IO, nlp::ScaledModel) =
  println(io, "ScaledModel - Model with scaled objective and constraints")


function ScaledModel(
    nlp::NLPModels.AbstractNLPModel{T, S};
    scaling=IpoptScaling(T(100)),
) where {T, S}
  n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
  x0 = NLPModels.get_x0(nlp)
  buffer_cons  = S(undef, m)
  scaling_obj, scaling_cons, scaling_jac = scale_model!(scaling, nlp)
  meta = NLPModels.NLPModelMeta(
    n;
    lvar=NLPModels.get_lvar(nlp),
    uvar=NLPModels.get_uvar(nlp),
    x0=NLPModels.get_x0(nlp),
    y0 = NLPModels.get_y0(nlp) .* scaling_cons,
    nnzj=NLPModels.get_nnzj(nlp),
    nnzh=NLPModels.get_nnzh(nlp),
    ncon=m,
    lcon=NLPModels.get_lcon(nlp) .* scaling_cons,
    ucon=NLPModels.get_ucon(nlp) .* scaling_cons,
    minimize=true,
  )

  return ScaledModel(
    nlp,
    meta,
    NLPModels.Counters(),
    scaling_obj,
    scaling_cons,
    scaling_jac,
    buffer_cons,
  )
end

function NLPModels.obj(nlp::ScaledModel{T, S}, x::AbstractVector) where {T, S <: AbstractVector{T}}
  @lencheck nlp.meta.nvar x
  return nlp.scaling_obj * NLPModels.obj(nlp.nlp, x)
end

function NLPModels.cons!(nlp::ScaledModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  NLPModels.cons!(nlp.nlp, x, c)
  c .*= nlp.scaling_cons
  return c
end

function NLPModels.grad!(nlp::ScaledModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  NLPModels.grad!(nlp.nlp, x, g)
  g .*= nlp.scaling_obj
  return g
end

function NLPModels.jprod!(nlp::ScaledModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  NLPModels.jprod!(nlp.nlp, x, v, Jv)
  Jv .*= nlp.scaling_cons
  return Jv
end

function NLPModels.jtprod!(nlp::ScaledModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  v_scaled = nlp.buffer_cons
  v_scaled .= v .* nlp.scaling_cons
  NLPModels.jtprod!(nlp.nlp, x, v_scaled, Jtv)
  return Jtv
end

function NLPModels.jac_structure!(nlp::ScaledModel, jrows::AbstractVector, jcols::AbstractVector)
  NLPModels.jac_structure!(nlp.nlp, jrows, jcols)
  return jrows, jcols
end

function NLPModels.jac_coord!(nlp::ScaledModel, x::AbstractVector, jac::AbstractVector)
  NLPModels.jac_coord!(nlp.nlp, x, jac)
  jac .*= nlp.scaling_jac
  return jac
end

function NLPModels.hess_structure!(nlp::ScaledModel, hrows::AbstractVector, hcols::AbstractVector)
  @lencheck nlp.meta.nnzh hrows hcols
  NLPModels.hess_structure!(nlp.nlp, hrows, hcols)
  return hrows, hcols
end

function NLPModels.hess_coord!(
    nlp::ScaledModel,
    x::AbstractVector,
    vals::AbstractVector;
    obj_weight::Real=one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  σ = obj_weight * nlp.scaling_obj
  NLPModels.hess_coord!(nlp.nlp, x, vals; obj_weight=σ)
  return vals
end

function NLPModels.hess_coord!(
    nlp::ScaledModel,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight::Real=one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  y_scaled = nlp.buffer_cons
  y_scaled .= y .* nlp.scaling_cons
  σ = obj_weight * nlp.scaling_obj
  NLPModels.hess_coord!(nlp.nlp, x, y_scaled, vals; obj_weight=σ)
  return vals
end

function NLPModels.hprod!(
  nlp::ScaledModel,
  x::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v hv
  σ = obj_weight * nlp.scaling_obj
  NLPModels.hprod!(nlp.nlp, x, v, hv; obj_weight = σ)
  return hv
end

function NLPModels.hprod!(
  nlp::ScaledModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v hv
  @lencheck nlp.meta.ncon y
  y_scaled = nlp.buffer_cons
  y_scaled .= y .* nlp.scaling_cons
  σ = obj_weight * nlp.scaling_obj
  NLPModels.hprod!(nlp.nlp, x, y, v, hv; obj_weight = σ)
  return hv
end

