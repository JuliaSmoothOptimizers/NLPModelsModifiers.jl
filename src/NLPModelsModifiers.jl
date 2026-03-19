module NLPModelsModifiers

# stdlib
using LinearAlgebra, SparseArrays
# external
using FastClosures
# jso
using LinearOperators, NLPModels

macro notimplemented_use_nls(fun)
  :(error(
    $fun,
    " is not implemented for models of type FeasibilityResidual. Try converting the FeasibilityResidual model as a FeasbilityFormNLS.",
  ))
end

include("no-fixed.jl")
include("feasibility-form-nls.jl")
include("feasibility-residual.jl")
include("quasi-newton.jl")
include("slack-model.jl")
include("model-interaction.jl")

end # module
