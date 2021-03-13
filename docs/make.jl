using Documenter, NLPModelsModifiers

makedocs(
  modules = [NLPModelsModifiers],
  doctest = true,
  linkcheck = false,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"], prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "NLPModelsModifiers.jl",
  pages = ["Home" => "index.md",
           "Models" => "models.md",
           "Reference" => "reference.md"
          ]
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl.git",
  push_preview = true
)
