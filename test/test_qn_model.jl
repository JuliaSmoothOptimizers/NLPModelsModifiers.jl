function check_qn_model(qnmodel)
  rtol  = 1e-8
  model = qnmodel.model
  @assert typeof(qnmodel) <: QuasiNewtonModel
  @assert qnmodel.meta.nvar == model.meta.nvar
  @assert qnmodel.meta.ncon == model.meta.ncon

  x = [-(-1.0)^i for i = 1:qnmodel.meta.nvar]

  @assert isapprox(obj(model, x), obj(qnmodel, x), rtol=rtol)
  @assert neval_obj(model) == 2

  @assert isapprox(grad(model, x), grad(qnmodel, x), rtol=rtol)
  @assert neval_grad(model) == 2

  @assert isapprox(cons(model, x), cons(qnmodel, x), rtol=rtol)
  @assert neval_cons(model) == 2

  @assert isapprox(jac(model, x), jac(qnmodel, x), rtol=rtol)
  @assert neval_jac(model) == 2

  v = [-(-1.0)^i for i = 1:qnmodel.meta.nvar]
  u = [-(-1.0)^i for i = 1:qnmodel.meta.ncon]

  @assert isapprox(jprod(model, x, v), jprod(qnmodel, x, v), rtol=rtol)
  @assert neval_jprod(model) == 2

  @assert isapprox(jtprod(model, x, u), jtprod(qnmodel, x, u), rtol=rtol)
  @assert neval_jtprod(model) == 2

  H = hess_op(qnmodel, x)
  @assert typeof(H) <: LinearOperators.AbstractLinearOperator
  @assert size(H) == (model.meta.nvar, model.meta.nvar)
  @assert isapprox(H * v, hprod(qnmodel, x, v), rtol=rtol)

  g = grad(qnmodel, x)
  gp = grad(qnmodel, x - g)
  push!(qnmodel, -g, gp - g)  # only testing that the call succeeds, not that the update is valid
  # the quasi-Newton operator itself is tested in LinearOperators

  reset!(qnmodel)
end

nlp = SimpleNLPModel()
qn_model = LBFGSModel(nlp)
check_qn_model(qn_model)
qn_model = LBFGSModel(nlp, mem=2)
check_qn_model(qn_model)
qn_model = LSR1Model(nlp)
check_qn_model(qn_model)
qn_model = LSR1Model(nlp, mem=2)
check_qn_model(qn_model)

@testset "objgrad of a qnmodel" begin
  struct OnlyObjgradModel <: AbstractNLPModel
    meta :: NLPModelMeta
    counters :: Counters
  end

  function OnlyObjgradModel()
    meta = NLPModelMeta(2)
    OnlyObjgradModel(meta, Counters())
  end

  function NLPModels.objgrad!(:: OnlyObjgradModel, x :: AbstractVector, g :: AbstractVector)
    f = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2
    g[1] = 2 * (x[1] - 1) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    f, g
  end

  nlp = LBFGSModel(OnlyObjgradModel())

  @test objgrad!(nlp, nlp.meta.x0, zeros(2)) == objgrad!(nlp.model, nlp.meta.x0, zeros(2))
  @test objgrad(nlp, nlp.meta.x0) == objgrad(nlp.model, nlp.meta.x0)
end