@testset "QuasiNewtonModel NLP tests" begin
  @testset "API" begin
    f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
    ∇f(x) = [2 * (x[1] - 2); 2 * (x[2] - 1)]
    c(x) = [x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1]
    clin(x) = [x[1] - 2x[2] + 1]
    cnln(x) = [-x[1]^2 / 4 - x[2]^2 + 1]
    J(x) = [1.0 -2.0; -0.5x[1] -2.0x[2]]
    Jlin(x) = [1.0 -2.0]
    Jnln(x) = [-0.5x[1] -2.0x[2]]

    for (QNM, QNO) in [
        (LSR1Model, LSR1Operator),
        (LBFGSModel, LBFGSOperator),
        (DiagonalPSBModel, DiagonalPSB),
        (DiagonalAndreiModel, DiagonalAndrei),
        (SpectralGradientModel, SpectralGradient),
      ],
      T in [Float64, Float32],
      M in [NLPModelMeta, SimpleNLPMeta]

      nlp = QNM(SimpleNLPModel(T, M))
      n = nlp.meta.nvar
      m = nlp.meta.ncon

      s, y = randn(T, n), randn(T, n)
      if QNO ∈ (DiagonalPSB, DiagonalAndrei)
        B = QNO(ones(T, n))
      elseif QNO == SpectralGradient
        B = QNO(one(T), n)
      else
        B = QNO(T, n)
      end
      push!(B, s, y)
      push!(nlp, s, y)
      H(x) = B
      H(x, y) = B

      y = randn(T, m)
      x = randn(T, n)
      v = randn(T, n)
      w = randn(T, m)
      Jv = zeros(T, m)
      Jtw = zeros(T, n)
      Hv = zeros(T, n)
      Hvals = zeros(T, nlp.meta.nnzh)

      # Basic methods
      @test obj(nlp, x) ≈ f(x)
      @test grad(nlp, x) ≈ ∇f(x)
      @test hprod(nlp, x, v) ≈ H(x) * v
      @test neval_hprod(nlp.model) == 0
      (QNM == LSR1Model) ? (@test neval_hprod(nlp) == 2) : (@test neval_hprod(nlp) == 1)
      @test cons(nlp, x) ≈ c(x)
      @test jac(nlp, x) ≈ J(x)
      @test jprod(nlp, x, v) ≈ J(x) * v
      @test jtprod(nlp, x, w) ≈ J(x)' * w
      if nlp.meta.lin_nnzj > 0
        @test cons_lin(nlp, x) ≈ clin(x)
        @test jac_lin(nlp, x) ≈ Jlin(x)
        @test jprod_lin(nlp, x, v) ≈ Jlin(x) * v
        @test jtprod_lin(nlp, x, w[nlp.meta.lin]) ≈ Jlin(x)' * w[nlp.meta.lin]
      end
      if nlp.meta.nln_nnzj > 0
        @test cons_nln(nlp, x) ≈ cnln(x)
        @test jac_nln(nlp, x) ≈ Jnln(x)
        @test jprod_nln(nlp, x, v) ≈ Jnln(x) * v
        @test jtprod_nln(nlp, x, w[nlp.meta.nln]) ≈ Jnln(x)' * w[nlp.meta.nln]
      end

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
      @test jprod_lin!(nlp, jac_lin_structure(nlp)..., jac_lin_coord(nlp, x), v, Jv[nlp.meta.lin]) ≈
            Jlin(x) * v
      @test jtprod_lin!(
        nlp,
        jac_lin_structure(nlp)...,
        jac_lin_coord(nlp, x),
        w[nlp.meta.lin],
        Jtw,
      ) ≈ Jlin(x)' * w[nlp.meta.lin]
      @test jprod_nln!(nlp, jac_nln_structure(nlp)..., jac_nln_coord(nlp, x), v, Jv[nlp.meta.nln]) ≈
            Jnln(x) * v
      @test jtprod_nln!(
        nlp,
        jac_nln_structure(nlp)...,
        jac_nln_coord(nlp, x),
        w[nlp.meta.nln],
        Jtw,
      ) ≈ Jnln(x)' * w[nlp.meta.nln]
      Jop = jac_op!(nlp, x, Jv, Jtw)
      @test Jop * v ≈ J(x) * v
      @test Jop' * w ≈ J(x)' * w
      Jop = jac_op!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), Jv, Jtw)
      @test Jop * v ≈ J(x) * v
      @test Jop' * w ≈ J(x)' * w
      Jop = jac_lin_op!(nlp, x, Jv[nlp.meta.lin], Jtw)
      @test Jop * v ≈ Jlin(x) * v
      @test Jop' * w[nlp.meta.lin] ≈ Jlin(x)' * w[nlp.meta.lin]
      Jop =
        jac_lin_op!(nlp, jac_lin_structure(nlp)..., jac_lin_coord(nlp, x), Jv[nlp.meta.lin], Jtw)
      @test Jop * v ≈ Jlin(x) * v
      @test Jop' * w[nlp.meta.lin] ≈ Jlin(x)' * w[nlp.meta.lin]
      Jop = jac_nln_op!(nlp, x, Jv[nlp.meta.nln], Jtw)
      @test Jop * v ≈ Jnln(x) * v
      @test Jop' * w[nlp.meta.nln] ≈ Jnln(x)' * w[nlp.meta.nln]
      Jop =
        jac_nln_op!(nlp, jac_nln_structure(nlp)..., jac_nln_coord(nlp, x), Jv[nlp.meta.nln], Jtw)
      @test Jop * v ≈ Jnln(x) * v
      @test Jop' * w[nlp.meta.nln] ≈ Jnln(x)' * w[nlp.meta.nln]
      Hop = hess_op(nlp, x)
      @test Hop * v ≈ H(x) * v
      Hop = hess_op!(nlp, x, Hv)
      @test Hop * v ≈ H(x) * v
      @test nlp.counters == nlp.model.counters
      @test neval_obj(nlp) == neval_obj(nlp.model)
      @test neval_grad(nlp) == neval_grad(nlp.model)
      @test neval_hess(nlp) == neval_hess(nlp.model)
      @test neval_hprod(nlp) == neval_hprod(nlp.model)
      
      reset_data!(nlp)
      Hop = hess_op!(nlp, x, Hv)
      @test Hop * v == v
    end
  end

  @testset "Show" begin
    for QNM ∈ [LSR1Model, LBFGSModel, DiagonalPSBModel, DiagonalAndreiModel, SpectralGradientModel]
      nlp = QNM(SimpleNLPModel())
      io = IOBuffer()
      show(io, nlp)
      showed = String(take!(io))
      storage_type = typeof(nlp)
      expected = """$storage_type - A QuasiNewtonModel
      Problem name: Simple NLP Model
      All variables: ████████████████████ 2      All constraints: ████████████████████ 2
                free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
              upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            low/upp: ████████████████████ 2              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
              fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
              infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                nnzh: ( 33.33% sparsity)   2               linear: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
                                                        nonlinear: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 1
                                                            nnzj: (  0.00% sparsity)   4

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
end
