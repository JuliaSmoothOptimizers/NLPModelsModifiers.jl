@testset "Check allocations for model modifiers of NLP" begin
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [hess]),
    map(x -> LBFGSModel(eval(Symbol(x))()), NLPModelsTest.nlp_problems),
  )
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [hess]),
    map(x -> LSR1Model(eval(Symbol(x))()), NLPModelsTest.nlp_problems),
  )
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [hess]),
    map(x -> DiagonalQNModel(eval(Symbol(x))()), NLPModelsTest.nlp_problems),
  )
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [hess]),
    map(x -> SpectralGradientModel(eval(Symbol(x))()), NLPModelsTest.nlp_problems),
  )
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [hess]),
    map(x -> SlackModel(eval(Symbol(x))()), NLPModelsTest.nlp_problems),
  )
  problems = setdiff(NLPModelsTest.nlp_problems, ["BROWNDEN", "HS5"])
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [hess]),
    map(x -> FeasibilityResidual(eval(Symbol(x))()), problems),
  )
end

@testset "Check allocations for model modifiers of NLS" begin
  problems = NLPModelsTest.nls_problems
  # `obj` and `grad` of an NLS model allocates a vector of size `nequ`.
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [obj, grad, hess]),
    map(x -> LBFGSModel(eval(Symbol(x))()), problems),
  )
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [obj, grad, hess]),
    map(x -> LSR1Model(eval(Symbol(x))()), problems),
  )
  map(
    nlp -> NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [obj, grad, hess]),
    map(x -> SlackModel(eval(Symbol(x))()), NLPModelsTest.nlp_problems),
  )
  map(
    nlp -> NLPModelsTest.test_zero_allocations(
      nlp,
      linear_api = true,
      exclude = [obj, grad, hess, hess_residual],
    ),
    map(
      x -> FeasibilityResidual(eval(Symbol(x))(), name = x * "feas"),
      setdiff(problems, ["MGH01", "BNDROSENBROCK"]),
    ),
  )
  # jtprod! https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl/issues/77
  map(
    nlp ->
      NLPModelsTest.test_zero_allocations(nlp, linear_api = true, exclude = [hess, jtprod, jac_op]),
    map(x -> FeasibilityFormNLS(eval(Symbol(x))()), problems),
  )
end
