@testset "SlackModel NLP tests" begin
  @testset "API" for T in [Float64, Float32]
    f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
    ∇f(x) = [2 * (x[1] - 2); 2 * (x[2] - 1); 0]
    H(x) = [2.0 0 0; 0 2.0 0; 0 0 0]
    c(x) = [x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1 - x[3]]
    J(x) = [1.0 -2.0 0; -0.5x[1] -2.0x[2] -1]
    H(x, y) = H(x) + y[2] * [-0.5 0 0; 0 -2.0 0; 0 0 0]

    nlp = SlackModel(SimpleNLPModel(T))
    n = nlp.meta.nvar
    m = nlp.meta.ncon

    x = randn(n)
    y = randn(m)
    v = randn(n)
    w = randn(m)
    Jv = zeros(m)
    Jtw = zeros(n)
    Hv = zeros(n)
    Hvals = zeros(nlp.meta.nnzh)

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

    # Increasing coverage
    fx, cx = objcons(nlp, x)
    @test fx ≈ f(x)
    @test cx ≈ c(x)
    fx, _ = objcons!(nlp, x, cx)
    @test fx ≈ f(x)
    @test cx ≈ c(x)
    fx, gx = objgrad(nlp, x)
    @test fx ≈ f(x)
    @test gx ≈ ∇f(x)
    fx, _ = objgrad!(nlp, x, gx)
    @test fx ≈ f(x)
    @test gx ≈ ∇f(x)
    @test jprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), v, Jv) ≈ J(x) * v
    @test jprod!(nlp, x, jac_structure(nlp)..., v, Jv) ≈ J(x) * v
    @test jtprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), w, Jtw) ≈ J(x)' * w
    @test jtprod!(nlp, x, jac_structure(nlp)..., w, Jtw) ≈ J(x)' * w
    Jop = jac_op!(nlp, x, Jv, Jtw)
    @test Jop * v ≈ J(x) * v
    @test Jop' * w ≈ J(x)' * w
    Jop = jac_op!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), Jv, Jtw)
    @test Jop * v ≈ J(x) * v
    @test Jop' * w ≈ J(x)' * w
    Jop = jac_op!(nlp, x, jac_structure(nlp)..., Jv, Jtw)
    @test Jop * v ≈ J(x) * v
    @test Jop' * w ≈ J(x)' * w
    ghjv = zeros(m)
    for j = 1:m
      eⱼ = [i == j ? 1.0 : 0.0 for i = 1:m]
      Cⱼ(x) = H(x, eⱼ) - H(x)
      ghjv[j] = dot(gx, Cⱼ(x) * v)
    end
    @test ghjvprod(nlp, x, gx, v) ≈ ghjv
    @test hess_coord!(nlp, x, Hvals) == hess_coord!(nlp, x, y * 0, Hvals)
    @test hprod!(nlp, hess_structure(nlp)..., hess_coord(nlp, x), v, Hv) ≈ H(x) * v
    @test hprod!(nlp, x, hess_structure(nlp)..., v, Hv) ≈ H(x) * v
    @test hprod!(nlp, x, y, hess_structure(nlp)..., v, Hv) ≈ H(x, y) * v
    Hop = hess_op(nlp, x)
    @test Hop * v ≈ H(x) * v
    Hop = hess_op!(nlp, x, Hv)
    @test Hop * v ≈ H(x) * v
    Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x), Hv)
    @test Hop * v ≈ H(x) * v
    Hop = hess_op!(nlp, x, hess_structure(nlp)..., Hv)
    @test Hop * v ≈ H(x) * v
    Hop = hess_op(nlp, x, y)
    @test Hop * v ≈ H(x, y) * v
    Hop = hess_op!(nlp, x, y, Hv)
    @test Hop * v ≈ H(x, y) * v
    Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x, y), Hv)
    @test Hop * v ≈ H(x, y) * v
    Hop = hess_op!(nlp, x, y, hess_structure(nlp)..., Hv)
    @test Hop * v ≈ H(x, y) * v
  end

  @testset "Show" begin
    nlp = SlackModel(SimpleNLPModel())
    io = IOBuffer()
    show(io, nlp)
    showed = String(take!(io))
    expected = """SlackModel - Model with slack variables
    Problem name: Simple NLP Model-slack
    All variables: ████████████████████ 3      All constraints: ████████████████████ 2
              free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            lower: ███████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
          low/upp: ██████████████⋅⋅⋅⋅⋅⋅ 2              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ████████████████████ 2
            infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              nnzh: ( 66.67% sparsity)   2               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                                                      nonlinear: ████████████████████ 2
                                                          nnzj: ( 16.67% sparsity)   5

    Counters:
              obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0"""

    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  end
end
