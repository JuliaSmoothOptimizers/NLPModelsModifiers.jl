using LinearAlgebra, SparseArrays, Test
using LinearOperators, NLPModels, NLPModelsModifiers

include("nlp/simple-model.jl")
include("nlp/quasi-newton.jl")
include("nlp/slack-model.jl")

include("nls/simple-model.jl")
include("nls/feasibility-form-nls.jl")
include("nls/feasibility-residual.jl")
include("nls/slack-model.jl")

using CUDA, NLPModelsTest
using ADNLPModels

include("nlp/fixed-variable-elimination.jl")
include("nls/fixed-variable-elimination.jl")

if CUDA.functional()
  include("gpu_test.jl")
end

if (v"1.7" <= VERSION)
  include("allocs_test.jl")
end
