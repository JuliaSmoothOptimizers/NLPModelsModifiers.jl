export ScaledModel

struct ConservativeScaling{T}
  max_gradient::T
end

function _set_constraints_scaling!(scaling, Ji, Jj, Jx, max_gradient)
  # Store norm(∇cᵢ, Inf) at index i of vector scaling
  for (i, j, x) in zip(Ji, Jj, Jx)
    scaling[i] = max(scaling[i], abs(x))
  end
  # Compute scaling as min(1, max_gradient / norm(∇cᵢ, Inf) )
  for i in eachindex(scaling)
    scaling[i] = min(1.0, max_gradient / scaling[i])
  end
end

function _set_jacobian_scaling!(Jx, Ji, Jj, scaling)
  k = 0
  for (i, j) in zip(Ji, Jj)
    Jx[k += 1] = scaling[i]
  end
end

function scale_model!(scaling::ConservativeScaling{T}, nlp) where T
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
       min_x  \quad &  f(x)\\
\mathrm{s.t.} \quad &  c_L ≤ c(x) ≤ c_U,\\
                    &  ℓ ≤ x ≥ u,
\end{aligned}
```
as
```math
\begin{aligned}
       min_x  \quad &  σf . f(x)\\
\mathrm{s.t.} \quad &  σc . c_L ≤ σc . c(x) ≤ σc . c_U, \\
                    &  ℓ ≤ x ≥ u,
\end{aligned}
```
with ``σf`` a positive scalar defined as
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
By default, the threshold parameter `max_gradient` is set to 100.0.

The method has been originally proposed in Ipopt [1].

## Reference

[1] Wächter, A., & Biegler, L. T. (2006).
On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming.
Mathematical programming, 106(1), 25-57.

"""
struct ScaledModel{T, S, M} <: NLPModels.AbstractNLPModel{T, S}
  nlp::M
  meta::NLPModels.NLPModelMeta{T, S}
  counters::NLPModels.Counters
  scaling_obj::T
  scaling_cons::S     # [size m]
  scaling_cons_lin::S # [size nlin]
  scaling_cons_nln::S # [size nnln]
  scaling_jac::S      # [size nnzj]
  scaling_jac_lin::S  # [size lin_nnzj]
  scaling_jac_nln::S  # [size nln_nnzj]
  buffer_cons::S      # [size m]
end

function ScaledModel(
  nlp::NLPModels.AbstractNLPModel{T, S};
  scaling=ConservativeScaling(T(100)),
) where {T, S}
  n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
  x0 = NLPModels.get_x0(nlp)
  buffer_cons  = S(undef, m)

  # Compute scaling for the problem as a whole.
  scaling_obj, scaling_cons, scaling_jac = scale_model!(scaling, nlp)

  # Get scaling for linear and nonlinear constraints.
  scaling_cons_lin = scaling_cons[nlp.meta.lin]
  scaling_cons_nln = scaling_cons[nlp.meta.nln]
  scaling_jac_lin = zeros(T, nlp.meta.lin_nnzj)
  Jlin_i, Jlin_j = NLPModels.jac_lin_structure(nlp)
  k = 0
  for (i, j) in zip(Jlin_i, Jlin_j)
    scaling_jac_lin[k += 1] = scaling_cons_lin[i]
  end
  scaling_jac_nln = zeros(T, nlp.meta.nln_nnzj)
  Jnln_i, Jnln_j = NLPModels.jac_nln_structure(nlp)
  k = 0
  for (i, j) in zip(Jnln_i, Jnln_j)
    scaling_jac_lin[k += 1] = scaling_cons_nln[i]
  end

  # Copy metadata from original problem, with some modifications.
  meta = NLPModels.NLPModelMeta(
    nlp.meta;
    y0 = NLPModels.get_y0(nlp) .* scaling_cons,
    lcon = NLPModels.get_lcon(nlp) .* scaling_cons,
    ucon = NLPModels.get_ucon(nlp) .* scaling_cons,
    name="scaled-" * nlp.meta.name,
  )

  return ScaledModel(
    nlp,
    meta,
    NLPModels.Counters(),
    scaling_obj,
    scaling_cons,
    scaling_cons_lin,
    scaling_cons_nln,
    scaling_jac,
    scaling_jac_lin,
    scaling_jac_nln,
    buffer_cons,
  )
end

function NLPModels.obj(nlp::ScaledModel{T, S}, x::AbstractVector) where {T, S <: AbstractVector{T}}
  @lencheck nlp.meta.nvar x
  return nlp.scaling_obj * NLPModels.obj(nlp.nlp, x)
end

function NLPModels.grad!(nlp::ScaledModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  NLPModels.grad!(nlp.nlp, x, g)
  g .*= nlp.scaling_obj
  return g
end

function NLPModels.cons!(nlp::ScaledModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  NLPModels.cons!(nlp.nlp, x, c)
  c .*= nlp.scaling_cons
  return c
end

function NLPModels.cons_lin!(nlp::ScaledModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nlin c
  NLPModels.cons_lin!(nlp.nlp, x, c)
  c .*= nlp.scaling_cons_lin
  return c
end

function NLPModels.cons_nln!(nlp::ScaledModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnln c
  NLPModels.cons_nln!(nlp.nlp, x, c)
  c .*= nlp.scaling_cons_nln
  return c
end

function NLPModels.jprod!(nlp::ScaledModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  NLPModels.jprod!(nlp.nlp, x, v, Jv)
  Jv .*= nlp.scaling_cons
  return Jv
end

function NLPModels.jprod_lin!(nlp::ScaledModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.nlin Jv
  NLPModels.jprod_lin!(nlp.nlp, x, v, Jv)
  Jv .*= nlp.scaling_cons_lin
  return Jv
end

function NLPModels.jprod_nln!(nlp::ScaledModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.nnln Jv
  NLPModels.jprod_lin!(nlp.nlp, x, v, Jv)
  Jv .*= nlp.scaling_cons_nln
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

function NLPModels.jtprod_lin!(nlp::ScaledModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nlin v
  v_scaled = view(nlp.buffer_cons, 1:nlp.meta.nlin)
  v_scaled .= v .* nlp.scaling_cons_lin
  NLPModels.jtprod_lin!(nlp.nlp, x, v_scaled, Jtv)
  return Jtv
end

function NLPModels.jtprod_nln!(nlp::ScaledModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.nnln v
  v_scaled = view(nlp.buffer_cons, 1:nlp.meta.nnln)
  v_scaled .= v .* nlp.scaling_cons_nln
  NLPModels.jtprod_nln!(nlp.nlp, x, v_scaled, Jtv)
  return Jtv
end

function NLPModels.jac_structure!(nlp::ScaledModel, jrows::AbstractVector, jcols::AbstractVector)
  @lencheck nlp.meta.nnzj jrows jcols
  NLPModels.jac_structure!(nlp.nlp, jrows, jcols)
  return jrows, jcols
end

function NLPModels.jac_lin_structure!(nlp::ScaledModel, jrows::AbstractVector, jcols::AbstractVector)
  NLPModels.jac_lin_structure!(nlp.nlp, jrows, jcols)
  return jrows, jcols
end

function NLPModels.jac_nln_structure!(nlp::ScaledModel, jrows::AbstractVector, jcols::AbstractVector)
  NLPModels.jac_nln_structure!(nlp.nlp, jrows, jcols)
  return jrows, jcols
end

function NLPModels.jac_coord!(nlp::ScaledModel, x::AbstractVector, jac::AbstractVector)
  NLPModels.jac_coord!(nlp.nlp, x, jac)
  jac .*= nlp.scaling_jac
  return jac
end

function NLPModels.jac_lin_coord!(nlp::ScaledModel, x::AbstractVector, jac::AbstractVector)
  NLPModels.jac_lin_coord!(nlp.nlp, x, jac)
  jac .*= nlp.scaling_jac_lin
  return jac
end

function NLPModels.jac_nln_coord!(nlp::ScaledModel, x::AbstractVector, jac::AbstractVector)
  NLPModels.jac_nln_coord!(nlp.nlp, x, jac)
  jac .*= nlp.scaling_jac_nln
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

