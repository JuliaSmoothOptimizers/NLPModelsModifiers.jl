function test_nls_to_cons()
  @testset "Test FeasibilityFormNLS consistency" begin
    F1(x) = [x[1] - 1; 10 * (x[2] - x[1]^2)]
    F2(x) = [x[1] * x[2] * x[3] * x[4] * x[5] - 1]
    F3(x) = [x[1] + x[2] - 1; x[1]^2 + x[2]^2 - 2; x[1]^3 + x[2]^3 - 3]
    c1(x) = [sum(x); x[1] * x[2] - 2]
    for (F, n, ne) in [(F1, 2, 2), (F2, 5, 1), (F3, 2, 3)], (c, m) in [(x -> zeros(0), 0), (c1, 2)]
      x0 = [-(1.0)^i for i = 1:n]
      nls = ADNLSModel(F, x0, ne, c, zeros(m), zeros(m))

      nlpcon = FeasibilityFormNLS(nls)
      adnlp = ADNLPModel(
        x -> sum(x[(n + 1):end] .^ 2) / 2,
        [x0; zeros(ne)],
        x -> [F(x[1:n]) - x[(n + 1):end]; c(x[1:n])],
        zeros(ne + m),
        zeros(ne + m),
      )
      TestUtils.consistent_functions([nlpcon; adnlp], exclude = [ghjvprod])

      adnls = ADNLSModel(
        x -> x[(n + 1):end],
        [x0; zeros(ne)],
        ne,
        x -> [F(x[1:n]) - x[(n + 1):end]; c(x[1:n])],
        zeros(ne + m),
        zeros(ne + m),
      )
      TestUtils.consistent_functions([nlpcon; adnls], exclude = [ghjvprod])
      TestUtils.consistent_nls_functions([nlpcon; adnls])
    end
  end

  @testset "Test FeasibilityFormNLS with LLSModel" begin
    for n in [10; 30], ne in [10; 20; 30], m in [0; 20]
      for T in [
        (rows, cols) -> Matrix(1.0I, rows, cols) .+ 1,
        (rows, cols) -> sparse(1.0I, rows, cols) .+ 1,
        (rows, cols) -> sparse(1.0I, rows, cols),
      ]
        A = T(ne, n)
        b = collect(1:ne)
        C = m > 0 ? T(m, n) : zeros(0, n)
        lls = LLSModel(A, b, C = C, lcon = zeros(m), ucon = zeros(m))
        nlpcon = FeasibilityFormNLS(lls)
        Ine = spdiagm(0 => ones(ne))
        lls2 = LLSModel(
          [spzeros(ne, n) Ine],
          zeros(ne),
          C = [A -Ine; C spzeros(m, ne)],
          lcon = [b; zeros(m)],
          ucon = [b; zeros(m)],
        )

        TestUtils.consistent_functions([nlpcon; lls2], exclude = [hess, hess_coord, ghjvprod])
        TestUtils.consistent_nls_functions([nlpcon; lls2])
      end
    end
  end

  @testset "Test FeasibilityFormNLS of a FeasibilityResidual" begin
    c(x) = [x[1]^2 + x[2]^2 - 5; x[1] * x[2] - 2; x[1] - 1; x[2] - 1]
    x0 = [0.5; 1.5]
    nlp = ADNLPModel(x -> 0, x0, c, zeros(4), zeros(4))
    ffnls = FeasibilityFormNLS(FeasibilityResidual(nlp))
    nlp2 = ADNLSModel(x -> x[3:6], [x0; zeros(4)], 4, x -> c(x[1:2]) - x[3:6], zeros(4), zeros(4))
    TestUtils.consistent_functions([ffnls; nlp2])
    TestUtils.consistent_nls_functions([ffnls; nlp2], exclude = [ghjvprod])

    # The test belows verifies that the nnzj and nnzh information are not lost
    n = 10
    m = 2n
    A = [spdiagm(0 => 2 * ones(n), 1 => -ones(n - 1), -1 => -ones(n - 1)); -I]
    b = zeros(m)
    nlp = LLSModel(spzeros(0, n), zeros(0), C = A, lcon = b, ucon = b)
    ffnls = FeasibilityFormNLS(FeasibilityResidual(nlp), name = "feas-of-feas")
    nlp2 = LLSModel([spzeros(m, n) I], zeros(m), C = [A -I], lcon = b, ucon = b)
    TestUtils.consistent_functions([ffnls; nlp2], exclude = [hess, hess_coord, ghjvprod])
    TestUtils.consistent_nls_functions([ffnls; nlp2])
  end

  @testset "FeasibilityFormNLS of an LLSModel should handle hess related function" begin
    lls = LLSModel(rand(10, 5), rand(10), C = rand(2, 5), lcon = zeros(2), ucon = zeros(2))
    nls = FeasibilityFormNLS(lls)
    @test hess_structure(nls) == (6:15, 6:15)
    @test hess_coord(nls, zeros(15)) == ones(10)
    @test hess_coord(nls, zeros(15), obj_weight = 0.3) == 0.3 * ones(10)
  end
end

test_nls_to_cons()
