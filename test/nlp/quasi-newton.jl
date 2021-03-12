@testset "QuasiNewtonModel NLP tests" begin
  @testset "API" begin
    f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
    ∇f(x) = [2 * (x[1] - 2); 2 * (x[2] - 1)]
    c(x) = [x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1]
    J(x) = [1.0  -2.0; -0.5x[1]  -2.0x[2]]

    for (QNM,QNO) in [(LSR1Model, LSR1Operator), (LBFGSModel, LBFGSOperator)]
      nlp = QNM(SimpleNLPModel())
      n = nlp.meta.nvar
      m = nlp.meta.ncon

      s, y = randn(n), randn(n)
      B = QNO(n)
      push!(B, s, y)
      push!(nlp, s, y)
      H(x) = B
      H(x,y) = B

      y = randn(m)
      x = randn(n)
      v = randn(n)
      w = randn(m)
      Jv = zeros(m)
      Jtw = zeros(n)
      Hv = zeros(n)
      Hvals = zeros(nlp.meta.nnzh)

      # Basic methods
      @test obj(nlp, x) ≈ f(x)
      @test grad(nlp, x) ≈ ∇f(x)
      @test hprod(nlp, x, v) ≈ H(x) * v
      @test cons(nlp, x) ≈ c(x)
      @test jac(nlp, x) ≈ J(x)
      @test jprod(nlp, x, v) ≈ J(x) * v
      @test jtprod(nlp, x, w) ≈ J(x)' * w

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
      Hop = hess_op(nlp, x)
      @test Hop * v ≈ H(x) * v
      Hop = hess_op!(nlp, x, Hv)
      @test Hop * v ≈ H(x) * v

      reset_data!(nlp)
      Hop = hess_op!(nlp, x, Hv)
      @test Hop * v == v
    end
  end

  @testset "Show" begin
    nlp = LSR1Model(SimpleNLPModel())
    io = IOBuffer()
    show(io, nlp)
    showed = String(take!(io))
    expected = """LSR1Model - A QuasiNewtonModel
    Problem name: Simple NLP Model
     All variables: ████████████████████ 2      All constraints: ████████████████████ 2
              free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
             upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           low/upp: ████████████████████ 2              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
            infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              nnzh: ( 33.33% sparsity)   2               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                                                      nonlinear: ████████████████████ 2
                                                           nnzj: (  0.00% sparsity)   4

    Counters:
               obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0"""

    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))

    nlp = LBFGSModel(SimpleNLPModel())
    io = IOBuffer()
    show(io, nlp)
    showed = String(take!(io))
    expected = """LBFGSModel - A QuasiNewtonModel
    Problem name: Simple NLP Model
     All variables: ████████████████████ 2      All constraints: ████████████████████ 2
              free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
             upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           low/upp: ████████████████████ 2              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
            infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              nnzh: ( 33.33% sparsity)   2               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                                                      nonlinear: ████████████████████ 2
                                                           nnzj: (  0.00% sparsity)   4

    Counters:
               obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
             hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0"""

    @test strip.(split(chomp(showed), "\n")) == strip.(split(chomp(expected), "\n"))
  end
end