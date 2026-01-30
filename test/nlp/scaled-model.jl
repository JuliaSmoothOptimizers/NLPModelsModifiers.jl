@testset "ScaledModel NLP tests" begin
  @testset "API" for T in [Float64, Float32], M in [NLPModelMeta, SimpleNLPMeta]
    nlp = ScaledModel(SimpleNLPModel(T, M))
    σ_obj, σ_cons = nlp.scaling_obj, nlp.scaling_cons

    f(x) = σ_obj * (x[1] - 2)^2 + (x[2] - 1)^2
    ∇f(x) = [σ_obj * 2 * (x[1] - 2); σ_obj * 2 * (x[2] - 1)]
    H(x) = T[(σ_obj * 2.0) 0; 0 (σ_obj * 2.0)]
    c(x) = [σ_cons[1] * (x[1] - 2x[2] + 1); σ_cons[2] * (-x[1]^2 / 4 - x[2]^2 + 1)]
    J(x) = [σ_cons[1] -2.0*σ_cons[1]; (-0.5 *σ_cons[1] * x[1]) (-2.0*σ_cons[2] * x[2])]
    H(x, y) = H(x) + σ_cons[2] * y[2] * T[-0.5 0; 0 -2.0]

    n = nlp.meta.nvar
    m = nlp.meta.ncon
    @test nlp.meta.x0 == T[2; 2]

    x = randn(T, n)
    y = randn(T, m)
    v = randn(T, n)
    w = randn(T, m)
    Jv = zeros(T, m)
    Jtw = zeros(T, n)
    Hv = zeros(T, n)
    Hvals = zeros(T, nlp.meta.nnzh)

    # Basic methods
    @test obj(nlp, x) ≈ f(x)
    @test grad(nlp, x) ≈ ∇f(x)
    @test hess(nlp, x) ≈ H(x)
    @test hprod(nlp, x, v) ≈ H(x) * v
    @test cons(nlp, x) ≈ c(x)
    @test jac(nlp, x) ≈ J(x)
    @test jprod(nlp, x, v) ≈ J(x) * v
    @test jtprod(nlp, x, w) ≈ J(x)' * w
    @test hess(nlp, x, y) ≈ H(x, y)
    @test hprod(nlp, x, y, v) ≈ H(x, y) * v
  end
end
