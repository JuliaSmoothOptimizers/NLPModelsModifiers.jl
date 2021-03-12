module NLPModelsModifiers

# stdlib
using LinearAlgebra, SparseArrays
# external
using FastClosures
# jso
using LinearOperators, NLPModels

include("feasibility-form-nls.jl")
include("feasibility-residual.jl")
include("quasi-newton.jl")
include("slack-model.jl")
include("model-interaction.jl")

end # module
