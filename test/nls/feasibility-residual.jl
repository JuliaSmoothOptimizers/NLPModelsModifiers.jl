@testset "FeasibilityResidual tests" begin
  @testset "NLS API" for T in [Float64, Float32]
    F(x) = T[x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1 - x[3]]
    JF(x) = T[1.0 -2.0 0; -0.5x[1] -2.0x[2] -1]
    HF(x, w) = w[2] * diagm(0 => T[-0.5; -2.0; 0.0])

    nls = FeasibilityResidual(SimpleNLPModel(T))
    n = nls.meta.nvar
    ne = nls_meta(nls).nequ

    x = randn(T, n)
    v = randn(T, n)
    w = randn(T, ne)
    Jv = zeros(T, ne)
    Jtw = zeros(T, n)
    Hv = zeros(T, n)

    @test residual(nls, x) ≈ F(x)
    @test jac_residual(nls, x) ≈ JF(x)
    @test hess_residual(nls, x, w) ≈ HF(x, w)
    @test jprod_residual(nls, x, v) ≈ JF(x) * v
    @test jtprod_residual(nls, x, w) ≈ JF(x)' * w
    @test jprod_residual!(nls, jac_structure_residual(nls)..., jac_coord_residual(nls, x), v, Jv) ≈
          JF(x) * v
    @test jtprod_residual!(
      nls,
      jac_structure_residual(nls)...,
      jac_coord_residual(nls, x),
      w,
      Jtw,
    ) ≈ JF(x)' * w
    @test jprod_residual!(nls, x, jac_structure_residual(nls)..., v, Jv) ≈ JF(x) * v
    @test jtprod_residual!(nls, x, jac_structure_residual(nls)..., w, Jtw) ≈ JF(x)' * w
    Jop = jac_op_residual(nls, x)
    @test Jop * v ≈ JF(x) * v
    @test Jop' * w ≈ JF(x)' * w
    Jop = jac_op_residual!(nls, x, Jv, Jtw)
    @test Jop * v ≈ JF(x) * v
    @test Jop' * w ≈ JF(x)' * w
    Jop = jac_op_residual!(nls, jac_structure_residual(nls)..., jac_coord_residual(nls, x), Jv, Jtw)
    @test Jop * v ≈ JF(x) * v
    @test Jop' * w ≈ JF(x)' * w
    Jop = jac_op_residual!(nls, x, jac_structure_residual(nls)..., Jv, Jtw)
    @test Jop * v ≈ JF(x) * v
    @test Jop' * w ≈ JF(x)' * w
    I, J, V = findnz(sparse(HF(x, w)))
    @test hess_structure_residual(nls) == (I, J)
    @test hess_coord_residual(nls, x, w) ≈ V
    for j = 1:ne
      eⱼ = [i == j ? one(T) : zero(T) for i = 1:ne]
      @test jth_hess_residual(nls, x, j) ≈ HF(x, eⱼ)
      @test hprod_residual(nls, x, j, v) ≈ HF(x, eⱼ) * v
      Hop = hess_op_residual(nls, x, j)
      @test Hop * v ≈ HF(x, eⱼ) * v
      Hop = hess_op_residual!(nls, x, j, Hv)
      @test Hop * v ≈ HF(x, eⱼ) * v
    end
    fx, gx = objgrad!(nls, x, v)
    @test obj(nls, x) ≈ norm(F(x))^2 / 2 ≈ fx
    @test grad(nls, x) ≈ JF(x)' * F(x) ≈ gx
  end

  @testset "NLP API" for T in [Float64, Float32]
    F(x) = T[x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1 - x[3]]
    JF(x) = T[1.0 -2.0 0; -0.5x[1] -2.0x[2] -1]
    HF(x, w) = w[2] * diagm(0 => T[-0.5; -2.0; 0.0])
    f(x) = norm(F(x))^2 / 2
    ∇f(x) = JF(x)' * F(x)
    H(x) = JF(x)' * JF(x) + HF(x, F(x))

    nls = FeasibilityResidual(SimpleNLPModel(T))
    n = nls.meta.nvar

    x = randn(T, n)
    v = randn(T, n)
    Hv = zeros(T, n)
    Hvals = zeros(T, nls.meta.nnzh)

    fx, gx = objgrad!(nls, x, v)
    @test obj(nls, x) ≈ norm(F(x))^2 / 2 ≈ fx ≈ f(x)
    @test grad(nls, x) ≈ JF(x)' * F(x) ≈ gx ≈ ∇f(x)
    @test hess(nls, x) ≈ H(x)
    @test hprod(nls, x, v) ≈ H(x) * v
    fx, gx = objgrad(nls, x)
    @test fx ≈ f(x)
    @test gx ≈ ∇f(x)
    fx, _ = objgrad!(nls, x, gx)
    @test fx ≈ f(x)
    @test gx ≈ ∇f(x)
    Hop = hess_op(nls, x)
    @test Hop * v ≈ H(x) * v
    Hop = hess_op!(nls, x, Hv)
    @test Hop * v ≈ H(x) * v
  end

  @testset "Show" begin
    nls = FeasibilityResidual(SimpleNLSModel())
    @test typeof(nls.nlp) == SlackNLSModel{Float64, Vector{Float64}, SimpleNLSModel{Float64, Vector{Float64}}}
    io = IOBuffer()
    show(io, nls)
    showed = String(take!(io))
    expected = """FeasibilityResidual - Nonlinear least-squares defined from constraints of another problem
    Problem name: Simple NLS Model-feasres
   All variables: ████████████████████ 4      All constraints: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0        All residuals: ████████████████████ 3
            free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           lower: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 2                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0            nonlinear: ████████████████████ 3
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 nnzj: ( 33.33% sparsity)   8
         low/upp: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 2              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 nnzh: ( 70.00% sparsity)   3
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            nnzh: (  0.00% sparsity)   10              linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                                                    nonlinear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                                                         nnzj: (------% sparsity)

  Counters:
             obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
        residual: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0         jac_residual: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0       jprod_residual: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
 jtprod_residual: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0        hess_residual: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0       jhess_residual: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
  hprod_residual: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0"""

    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  end

  @testset "FeasibilityResidual of an unconstrained problem" begin
    mutable struct UncModel{T, S} <: AbstractNLPModel{T, S}
      meta::NLPModelMeta{T, S}
    end
    nlp = UncModel(NLPModelMeta(2))
    @test_throws ErrorException FeasibilityResidual(nlp)
  end
end
