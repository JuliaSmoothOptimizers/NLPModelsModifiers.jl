"""
    SimpleNLPModel <: AbstractNLPModel

Simple model for testing purposes.
Modified problem 14 in the Hock-Schittkowski suite

     min   (x₁ - 2)² + (x₂ - 1)²
    s.to   x₁ - 2x₂ + 1 = 0
           -x₁² / 4 - x₂² + 1 ≥ 0
           0 ≤ x ≤ 1

x₀ = [2.0, 2.0].
"""
mutable struct SimpleNLPModel{T, S, M <: AbstractNLPModelMeta{T, S}} <: AbstractNLPModel{T, S}
  meta::M
  counters::Counters
end

mutable struct SimpleNLPMeta{T, S} <: AbstractNLPModelMeta{T, S}
  nvar::Int
  x0::S
  lvar::S
  uvar::S

  ifix::Vector{Int}
  ilow::Vector{Int}
  iupp::Vector{Int}
  irng::Vector{Int}
  ifree::Vector{Int}
  iinf::Vector{Int}

  nlvb::Int
  nlvo::Int
  nlvc::Int

  ncon::Int
  y0::S
  lcon::S
  ucon::S

  jfix::Vector{Int}
  jlow::Vector{Int}
  jupp::Vector{Int}
  jrng::Vector{Int}
  jfree::Vector{Int}
  jinf::Vector{Int}

  nnzo::Int
  nnzj::Int
  nnzh::Int

  nlin::Int
  nnln::Int

  lin::Vector{Int}
  nln::Vector{Int}

  minimize::Bool
  islp::Bool
  name::String
  function SimpleNLPMeta{T, S}(
    nvar::Int;
    x0::S = fill!(S(undef, nvar), zero(T)),
    lvar::S = fill!(S(undef, nvar), T(-Inf)),
    uvar::S = fill!(S(undef, nvar), T(Inf)),
    nlvb = nvar,
    nlvo = nvar,
    nlvc = nvar,
    ncon = 0,
    y0::S = fill!(S(undef, ncon), zero(T)),
    lcon::S = fill!(S(undef, ncon), T(-Inf)),
    ucon::S = fill!(S(undef, ncon), T(Inf)),
    nnzo = nvar,
    nnzj = nvar * ncon,
    nnzh = nvar * (nvar + 1) / 2,
    lin = Int[],
    nln = 1:ncon,
    nlin = length(lin),
    nnln = length(nln),
    minimize = true,
    islp = false,
    name = "Generic",
  ) where {T, S}
    if (nvar < 1) || (ncon < 0)
      error("Nonsensical dimensions")
    end

    @lencheck nvar x0 lvar uvar
    @lencheck ncon y0 lcon ucon
    @lencheck nlin lin
    @lencheck nnln nln
    @rangecheck 1 ncon lin nln

    ifix = findall(lvar .== uvar)
    ilow = findall((lvar .> T(-Inf)) .& (uvar .== T(Inf)))
    iupp = findall((lvar .== T(-Inf)) .& (uvar .< T(Inf)))
    irng = findall((lvar .> T(-Inf)) .& (uvar .< T(Inf)) .& (lvar .< uvar))
    ifree = findall((lvar .== T(-Inf)) .& (uvar .== T(Inf)))
    iinf = findall(lvar .> uvar)

    jfix = findall(lcon .== ucon)
    jlow = findall((lcon .> T(-Inf)) .& (ucon .== T(Inf)))
    jupp = findall((lcon .== T(-Inf)) .& (ucon .< T(Inf)))
    jrng = findall((lcon .> T(-Inf)) .& (ucon .< T(Inf)) .& (lcon .< ucon))
    jfree = findall((lcon .== T(-Inf)) .& (ucon .== T(Inf)))
    jinf = findall(lcon .> ucon)

    nnzj = max(0, nnzj)
    nnzh = max(0, nnzh)

    new{T, S}(
      nvar,
      x0,
      lvar,
      uvar,
      ifix,
      ilow,
      iupp,
      irng,
      ifree,
      iinf,
      nlvb,
      nlvo,
      nlvc,
      ncon,
      y0,
      lcon,
      ucon,
      jfix,
      jlow,
      jupp,
      jrng,
      jfree,
      jinf,
      nnzo,
      nnzj,
      nnzh,
      nlin,
      nnln,
      lin,
      nln,
      minimize,
      islp,
      name,
    )
  end
end
NLPModels.equality_constrained(meta::SimpleNLPMeta) = length(meta.jfix) == meta.ncon > 0
NLPModels.unconstrained(meta::SimpleNLPMeta) = meta.ncon == 0 && !has_bounds(meta)

function SimpleNLPModel(::Type{T}, ::Type{NLPModelMeta}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    nnzh = 2,
    ncon = 2,
    lvar = zeros(T, 2),
    uvar = ones(T, 2),
    x0 = T[2; 2],
    lcon = T[0; 0],
    ucon = T[0; Inf],
    name = "Simple NLP Model",
  )
  return SimpleNLPModel(meta, Counters())
end
function SimpleNLPModel(::Type{T}, ::Type{SimpleNLPMeta}) where {T}
  meta = SimpleNLPMeta{T, Vector{T}}(
    2,
    nnzh = 2,
    ncon = 2,
    lvar = zeros(T, 2),
    uvar = ones(T, 2),
    x0 = T[2; 2],
    lcon = T[0; 0],
    ucon = T[0; Inf],
    name = "Simple NLP Model",
  )
  return SimpleNLPModel(meta, Counters())
end
SimpleNLPModel() = SimpleNLPModel(Float64, NLPModelMeta)

function NLPModels.obj(nlp::SimpleNLPModel, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (x[1] - 2)^2 + (x[2] - 1)^2
end

function NLPModels.grad!(nlp::SimpleNLPModel, x::AbstractVector, gx::AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 2); 2 * (x[2] - 1)]
  return gx
end

function NLPModels.hess_structure!(
  nlp::SimpleNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y vals
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  vals[1] -= y[2] / 2
  vals[2] -= 2y[2]
  return vals
end

function NLPModels.hprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight * v
  Hv[1] -= y[2] * v[1] / 2
  Hv[2] -= 2y[2] * v[2]
  return Hv
end

function NLPModels.cons!(nlp::SimpleNLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x cx
  increment!(nlp, :neval_cons)
  cx .= [x[1] - 2 * x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1]
  return cx
end

function NLPModels.jac_structure!(
  nlp::SimpleNLPModel,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 4 rows cols
  rows .= [1, 2, 1, 2]
  cols .= [1, 1, 2, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp::SimpleNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 4 vals
  increment!(nlp, :neval_jac)
  vals .= [1, -x[1] / 2, -2, -2 * x[2]]
  return vals
end

function NLPModels.jprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v Jv
  increment!(nlp, :neval_jprod)
  Jv .= [v[1] - 2 * v[2]; -x[1] * v[1] / 2 - 2 * x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nlp, :neval_jtprod)
  Jtv .= [v[1] - x[1] * v[2] / 2; -2 * v[1] - 2 * x[2] * v[2]]
  return Jtv
end

function NLPModels.ghjvprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= [T(0); -g[1] * v[1] / 2 - 2 * g[2] * v[2]]
  return gHv
end

function NLPModels.jth_hess_coord!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 2 x
  @rangecheck 1 2 j
  @lencheck 2 vals
  if j == 1
    vals .= zero(T)
  else
    vals[1] = -T(1 / 2)
    vals[2] = -T(2)
  end
  return vals
end

function NLPModels.jth_hprod!(
  nlp::SimpleNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v hv
  @rangecheck 1 2 j
  if j == 1
    hv .= zero(T)
  else
    hv[1] = -v[1] / 2
    hv[2] = -2 * v[2]
  end
  return hv
end
