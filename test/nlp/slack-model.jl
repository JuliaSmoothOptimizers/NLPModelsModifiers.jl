@testset "SlackModel NLP tests" begin
  @testset "API" for T in [Float64, Float32], M in [NLPModelMeta, SimpleNLPMeta]
    f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
    ∇f(x) = T[2 * (x[1] - 2); 2 * (x[2] - 1); 0]
    H(x) = T[2.0 0 0; 0 2.0 0; 0 0 0]
    c(x) = T[x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1 - x[3]]
    J(x) = T[1.0 -2.0 0; -0.5x[1] -2.0x[2] -1]
    H(x, y) = H(x) + y[2] * T[-0.5 0 0; 0 -2.0 0; 0 0 0]

    nlp = SlackModel(SimpleNLPModel(T, M))
    n = nlp.meta.nvar
    m = nlp.meta.ncon
    @test nlp.meta.x0[1:2] == T[2; 2]

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
    @test jtprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), w, Jtw) ≈ J(x)' * w
    Jop = jac_op!(nlp, x, Jv, Jtw)
    @test Jop * v ≈ J(x) * v
    @test Jop' * w ≈ J(x)' * w
    res = J(x) * v - w
    @test mul!(w, Jop, v, one(T), -one(T)) ≈ res
    res = J(x)' * w - v
    @test mul!(v, Jop', w, one(T), -one(T)) ≈ res
    Jop = jac_op!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), Jv, Jtw)
    @test Jop * v ≈ J(x) * v
    @test Jop' * w ≈ J(x)' * w
    res = J(x) * v - w
    @test mul!(w, Jop, v, one(T), -one(T)) ≈ res
    res = J(x)' * w - v
    @test mul!(v, Jop', w, one(T), -one(T)) ≈ res
    ghjv = zeros(T, m)
    for j = 1:m
      eⱼ = [i == j ? one(T) : zero(T) for i = 1:m]
      Cⱼ(x) = H(x, eⱼ) - H(x)
      ghjv[j] = dot(gx, Cⱼ(x) * v)
      @test jth_hess(nlp, x, j) == Cⱼ(x)
      @test jth_hprod(nlp, x, v, j) == Cⱼ(x) * v
    end
    @test ghjvprod(nlp, x, gx, v) ≈ ghjv
    @test hess_coord!(nlp, x, Hvals) == hess_coord!(nlp, x, y * 0, Hvals)
    @test hprod!(nlp, hess_structure(nlp)..., hess_coord(nlp, x), v, Hv) ≈ H(x) * v
    Hop = hess_op(nlp, x)
    @test Hop * v ≈ H(x) * v
    Hop = hess_op!(nlp, x, Hv)
    @test Hop * v ≈ H(x) * v
    z = ones(T, nlp.meta.nvar)
    res = H(x) * v - z
    @test mul!(z, Hop, v, one(T), -one(T)) ≈ res
    Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x), Hv)
    @test Hop * v ≈ H(x) * v
    res = H(x) * v - z
    @test mul!(z, Hop, v, one(T), -one(T)) ≈ res
    Hop = hess_op(nlp, x, y)
    @test Hop * v ≈ H(x, y) * v
    Hop = hess_op!(nlp, x, y, Hv)
    @test Hop * v ≈ H(x, y) * v
    res = H(x, y) * v - z
    @test mul!(z, Hop, v, one(T), -one(T)) ≈ res
    Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x, y), Hv)
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
              nnzh: ( 66.67% sparsity)   2               linear: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
                                                      nonlinear: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
                                                          nnzj: ( 16.67% sparsity)   5

    Counters:
               obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
          cons_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0             cons_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              jac_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           jac_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0            jprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
         jprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0           jtprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
        jtprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0"""

    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  end
end
