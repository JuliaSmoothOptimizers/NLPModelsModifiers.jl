@testset "FixedVariableEliminationModel NLP tests" begin
  # Check consistency
  f(x) = (x[1] - x[2])^2 + x[2] * x[3] + (x[3] - x[4])^2 + x[1] * x[4]
  c(x) = [x[2]^2 + x[4]; x[1] * x[3] + x[4]^2]
  full_from_T =
    T -> ADNLPModel(
      f,
      T[1, 0, 2, 1 // 2],
      T[-10, 0, -10, 1 // 2],
      T[10, 0, 10, 1 // 2],
      [1, 1, 1],
      [1, 2, 3],
      T[1, 1, 1],
      c,
      T[0; 0; -1],
      T[1; 1; 1],
    )
  # Full model:
  # min f(x) = (x₁ - x₂)² + x₂ * x₃ + (x₃ - x₄)² + x₁ * x₄
  # s.t. c(x) = [x₂² + x₄; x₁ * x₃ + x₄²] = 0
  #      -10 ≤ x₁ ≤ 10, 0 ≤ x₂ ≤ 0, -10 ≤ x₃ ≤ 10, 1/2 ≤ x₄ ≤ 1/2

  # The reduced problem in x = (x₁, x₃)
  fr(x) = x[1]^2 + (x[2] - one(eltype(x)) / 2)^2 + x[1] / 2
  cr(x) = [one(eltype(x)) / 2 + 0 * x[1]; x[1] * x[2] + one(eltype(x)) / 4]
  reduced =
    ADNLPModel(fr, [1.0, 2.0], [-10.0, -10.0], [10.0, 10.0], [1, 1], [1, 2], [1.0, 1.0], cr, [0.0; 0.0; -1.0], [1.0; 1.0; 1.0])
  # Reduced model:
  # min fr(x) = x₁² + (x₂ - 1/2)² + x₁ / 2
  # s.t. cr(x) = [1/2; x₁ * x₂ + 1/4] = 0
  #     -10 ≤ x₁ ≤ 10, -10 ≤ x₂ ≤ 10

  @testset "Consistency with cross terms" begin
    nlp = FixedVariableEliminationModel(full_from_T(Float64))
    @test nlp.meta.nvar == 2
    @test nlp.meta.lin == [1]
    consistent_nlps([nlp, reduced], linear_api = true)
  end

  @testset "Consistency, bound constrained with x₁ fixed" begin
    g(x) = (x[1] - 1)^2 + x[1] * x[2] + (x[2] + x[3])^2 + x[3]^4
    full = ADNLPModel(g, [2.0, 1.0, 0.5], [2.0, -Inf, -1.0], [2.0, Inf, 1.0])
    nlp = FixedVariableEliminationModel(full)

    # x₁ = 2
    gr(x) = 1 + 2 * x[1] + (x[1] + x[2])^2 + x[2]^4
    reference = ADNLPModel(gr, [1.0, 0.5], [-Inf, -1.0], [Inf, 1.0])

    @test nlp.meta.nvar == 2
    consistent_nlps([nlp, reference])
  end

  @testset "NLPModelsTest checks" begin
    nlp_from_T = T -> FixedVariableEliminationModel(full_from_T(T))
    nlp = nlp_from_T(Float64)
    check_nlp_dimensions(nlp)
    view_subarray_nlp(nlp)
    multiple_precision_nlp(nlp_from_T, linear_api = true)
  end

  @testset "Constructor" begin
    model = ADNLPModel(f, ones(4))
    @test FixedVariableEliminationModel(model) === model # no fixed variables
    @test_throws ErrorException FixedVariableEliminationModel(
      ADNLPModel(f, ones(4), ones(4), ones(4)),
    )

    nlp = FixedVariableEliminationModel(full_from_T(Float64))
    @test nlp.meta.x0 == [1.0, 2.0]
    @test nlp.meta.lvar == [-10.0, -10.0]
    @test nlp.meta.uvar == [10.0, 10.0]
    @test nlp.meta.name == "Generic-nofixed"

    # minimal meta types work with the NLPModelMeta copier
    meta = NLPModelsModifiers.fixed_variable_elimination_meta(SimpleNLPModel(Float64, SimpleNLPMeta).meta)
    @test meta isa NLPModelMeta
    @test meta.nvar == 2
  end

  @testset "Utilities" begin
    nlp = FixedVariableEliminationModel(full_from_T(Float64))
    @test recover_full_solution(nlp, [3.0, 4.0]) == [3.0, 0.0, 4.0, 0.5]
    X = zeros(4)
    recover_full_solution!(nlp, [5.0, 6.0], X)
    @test X == [5.0, 0.0, 6.0, 0.5]

    @test get_model(nlp) isa ADNLPModel
    qn = LBFGSModel(nlp)
    @test find_model(FixedVariableEliminationModel, qn) === nlp
    @test find_model(ADNLPModel, qn) isa ADNLPModel

    io = IOBuffer()
    show(io, nlp)
    @test occursin("FixedVariableEliminationModel - Model without fixed variables", String(take!(io)))
  end
end
