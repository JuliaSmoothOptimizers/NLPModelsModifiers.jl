function FeasibilityFormNLS(
  nls::FeasibilityResidual{T, S};
  name = "$(nls.meta.name)-ffnls",
) where {T, S}
  meta = nls.nlp.meta
  nequ = meta.ncon
  nvar = meta.nvar + nequ
  ncon = meta.ncon
  nnzj = meta.nnzj + nequ
  nnzh = meta.nnzh + nequ
  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = [meta.x0; zeros(T, nequ)],
    lvar = [meta.lvar; fill(T(-Inf), nequ)],
    uvar = [meta.uvar; fill(T(Inf), nequ)],
    ncon = ncon,
    lcon = meta.lcon,
    ucon = meta.ucon,
    y0 = meta.y0,
    lin = meta.lin,
    nln = meta.nln,
    nnzj = nnzj,
    nnzh = nnzh,
    name = name,
  )
  nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = [meta.x0; zeros(T, nequ)], nnzj = nequ, nnzh = 0)

  nlp = FeasibilityFormNLS{T, S, FeasibilityResidual{T, S}}(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end
