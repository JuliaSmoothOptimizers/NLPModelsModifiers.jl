@testset "ScaledModel NLP tests" begin
  @testset "API" for T in [Float64, Float32], M in [NLPModelMeta, SimpleNLPMeta]
    original_nlp = SimpleNLPModel(T, M)
    nlp = ScaledModel(original_nlp)
    σ_obj, σ_cons = nlp.scaling_obj, nlp.scaling_cons

    # Hand-code the scaled problem from the original NLP.
    f(x) = σ_obj * NLPModels.obj(original_nlp, x)
    ∇f(x) = σ_obj .* NLPModels.grad(original_nlp, x)
    H(x) = σ_obj .* NLPModels.hess(original_nlp, x)
    c(x) = σ_cons .* NLPModels.cons(original_nlp, x)
    J(x) = Diagonal(σ_cons) * NLPModels.jac(original_nlp, x)
    H(x, y) = NLPModels.hess(original_nlp, x, σ_cons .* y; obj_weight=σ_obj)

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
