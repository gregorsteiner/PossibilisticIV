
using LaTeXStrings

include("PossibilisticIV.jl")

## Data generating function ## 
function generate_data(n, ρ, α)
    β, γ_2 = 1.0, 1.0
    Z = rand(Normal(0, 1), n)

    u = rand(MvNormal(zeros(2), [1.0 ρ; ρ 1.0]), n)'
    X = Z * γ_2 + u[:, 1]
    Y = β * X + α * Z + u[:, 2]
    return Y, X, Z
end

## Write function to implement the simulation ##
function run_simulation(m; n = 100, ρ = 1/2, α = 0.0)
    # different methods
    methods = [L"PossIV (α = 0)", "PossIV (α ∈ [-1/2, 1/2])"]
    
    # storage objects
    coverage = Matrix{Bool}(undef, length(methods), m)

    # start iterating
    for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, ρ, α)

        # compute coverage
        # possibilistic contour at the true value must be > 0.05
        coverage[1, i] = possibilistic_contour(1.0, 0.0, 0.0, [Y X], Z) > 0.05
        coverage[2, i] = possibilistic_contour(1.0, -1/2, 1/2, [Y X], Z) > 0.05
    end

    return (Coverage = mean(coverage; dims = 2), Methods = methods)
end

res = run_simulation(1000; α = 1/2)


