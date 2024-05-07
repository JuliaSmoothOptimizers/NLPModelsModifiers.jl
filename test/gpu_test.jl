function nlp_gpu_tests(p, Model; exclude = [])
  @testset "NLP tests of problem $p" begin
    nlp_from_T = T -> Model(eval(Symbol(p))(T))
    @testset "GPU multiple precision support of problem $p" begin
      CUDA.allowscalar() do
        multiple_precision_nlp_array(nlp_from_T, CuArray, linear_api = true, exclude = exclude)
      end
    end
  end
end

function nls_gpu_tests(p, Model; exclude = [])
  @testset "NLP tests of problem $p" begin
    nls_from_T = T -> Model(eval(Symbol(p))(T))
    exclude = p == "LLS" ? union(exclude, [hess_coord, hess, hess_residual]) : exclude
    @testset "GPU multiple precision support of problem $p" begin
      CUDA.allowscalar() do
        multiple_precision_nls_array(nls_from_T, CuArray, linear_api = true, exclude = exclude)
      end
    end
  end
end

@testset "Check GPU multiprecision for quasi-Newton model modifiers $M of NLP" for M in [
  LBFGSModel,
  LSR1Model,
  DiagonalPSBModel,
  DiagonalAndreiModel,
  SpectralGradientModel,
]
  # for hprod, seehttps://github.com/JuliaSmoothOptimizers/LinearOperators.jl/issues/327
  map(
    p -> nlp_gpu_tests(
      p,
      M,
      exclude = [hprod, hess, hess_coord, ghjvprod, jth_hess, jth_hess_coord, jth_hprod],
    ),
    union(NLPModelsTest.nlp_problems, NLPModelsTest.nls_problems),
  )
end

@testset "Check GPU multiprecision for model modifiers $M of NLP" for M in [FeasibilityResidual]
  map(
    p -> nlp_gpu_tests(
      p,
      M,
      exclude = [hess, hess_coord, ghjvprod, jth_hess, jth_hess_coord, jth_hprod],
    ),
    setdiff(NLPModelsTest.nlp_problems, ["BROWNDEN", "HS5"]),
  )
end

@testset "Check GPU multiprecision for model modifiers $M of NLP" for M in [SlackModel]
  map(p -> nlp_gpu_tests(p, M, exclude = []), NLPModelsTest.nlp_problems)
end

@testset "Check GPU multiprecision for model modifiers $M of NLP" for M in [FeasibilityResidual]
  map(
    p -> nls_gpu_tests(
      p,
      M,
      exclude = [hess, hess_coord, ghjvprod, jth_hess, jth_hess_coord, jth_hprod],
    ),
    setdiff(NLPModelsTest.nls_problems, ["MGH01", "BNDROSENBROCK"]),
  )
end

@testset "Check GPU multiprecision for model modifiers $M of NLP" for M in [
  SlackNLSModel,
  FeasibilityFormNLS,
]
  map(p -> nls_gpu_tests(p, M, exclude = []), NLPModelsTest.nls_problems)
end
