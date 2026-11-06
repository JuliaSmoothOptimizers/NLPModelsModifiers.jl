@testset "FixedVariableEliminationNLSModel tests" begin
  # Check consistency
  F(x) = [x[1] - 1; 10 * (x[2] - x[1]^2); x[2] - x[3]]
  c(x) = [x[2]^2 + x[1] * x[3]]
  full_from_T =
    T -> ADNLSModel(
      F,
      T[1 // 2, 1, -1 // 2],
      3,
      T[1 // 2, -10, -1 // 2],
      T[1 // 2, 10, -1 // 2],
      [1, 1, 1],
      [1, 2, 3],
      T[1, 1, 1],
      c,
      T[-1; -1],
      T[1; 1],
    )

  @testset "Consistency" begin
    nls = FixedVariableEliminationNLSModel(full_from_T(Float64))

    # The reduced problem in x = x₂
    Fr(x) = [-one(eltype(x)) / 2 + 0 * x[1]; 10 * (x[1] - one(eltype(x)) / 4); x[1] + one(eltype(x)) / 2]
    cr(x) = [x[1]^2 - one(eltype(x)) / 4]
    reference = ADNLSModel(Fr, [1.0], 3, [-10.0], [10.0], [1], [1], [1.0], cr, [-1.0; -1.0], [1.0; 1.0])

    @test nls.meta.nvar == 1
    @test nls.nls_meta.nvar == 1
    consistent_nlss([nls, reference], exclude = [jth_hess, jth_hess_coord, jth_hprod, ghjvprod])
  end

  @testset "NLPModelsTest checks" begin
    nls_from_T = T -> FixedVariableEliminationNLSModel(full_from_T(T))
    nls = nls_from_T(Float64)
    check_nls_dimensions(nls)
    view_subarray_nls(nls)
    multiple_precision_nls(nls_from_T, linear_api = true)
  end

  @testset "Constructor and utilities" begin
    model = ADNLSModel(F, ones(3), 3)
    @test FixedVariableEliminationNLSModel(model) === model # no fixed variables

    nls = FixedVariableEliminationNLSModel(full_from_T(Float64))
    @test recover_full_solution(nls, [3.0]) == [0.5, 3.0, -0.5]

    io = IOBuffer()
    show(io, nls)
    @test occursin(
      "FixedVariableEliminationNLSModel - Nonlinear least-squares model without fixed variables",
      String(take!(io)),
    )
  end
end
