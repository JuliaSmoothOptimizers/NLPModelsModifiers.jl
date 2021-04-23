function FeasibilityFormNLS(nls::FeasibilityResidual; name = "$(nls.meta.name)-ffnls")
  meta = nls.nlp.meta
  nequ = meta.ncon
  nvar = meta.nvar + nequ
  ncon = meta.ncon
  nnzj = meta.nnzj + nequ
  nnzh = meta.nnzh + nequ
  meta = NLPModelMeta(
    nvar,
    x0 = [meta.x0; zeros(nequ)],
    lvar = [meta.lvar; fill(-Inf, nequ)],
    uvar = [meta.uvar; fill(Inf, nequ)],
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
  nls_meta = NLSMeta(nequ, nvar, x0 = [meta.x0; zeros(nequ)], nnzj = nequ, nnzh = 0)

  nlp = FeasibilityFormNLS{FeasibilityResidual}(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end
