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
  x0 = similar(meta.x0, nvar)
  x0[1 : meta.nvar] .= meta.x0
  x0[meta.nvar : end] .= zero(T)
  lvar = similar(meta.x0, nvar)
  lvar[1 : meta.nvar] .= meta.lvar
  lvar[meta.nvar + 1 : end] .= T(-Inf)
  uvar = similar(meta.x0, nvar)
  uvar[1 : meta.nvar] .= meta.uvar
  uvar[meta.nvar + 1 : end] .= T(Inf)
  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
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
  nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x0, nnzj = nequ, nnzh = 0)

  nlp = FeasibilityFormNLS{T, S, FeasibilityResidual{T, S}}(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end
