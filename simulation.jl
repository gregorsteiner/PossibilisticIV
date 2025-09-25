
using LaTeXStrings, Random

include("PossibilisticIV.jl")
include("competing_methods.jl")

Y, X, Z = generate_data(100, 1/2, 0.0)
res = givbma(Y, X, Z[:, :])
plot(rbw(res))

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
    methods = [
        L"Possibilistic IV $(\alpha = 0)$",
        L"Possibilistic IV $(\alpha \in [-1/2, 1/2])$",
        L"Possibilistic IV $(\alpha \in [-1, 1])$",
        "TSLS",
        "gIVBMA"
        ]
    
    # storage objects
    coverage = Matrix{Bool}(undef, length(methods), m)

    # start iterating
    for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, ρ, α)

        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))

        # compute coverage
        # possibilistic contour at the true value must be > 0.05
        coverage[1, i] = possibilistic_contour(1.0, 0.0, 0.0, [Y X], Z) > 0.05
        coverage[2, i] = possibilistic_contour(1.0, -1/2, 1/2, [Y X], Z) > 0.05
        coverage[3, i] = possibilistic_contour(1.0, -1.0, 1.0, [Y X], Z) > 0.05
        coverage[4, i] = check_coverage(tsls(Y, X, Z), 1.0)
        coverage[5, i] = check_coverage(givbma(Y, X, Z[:, :]; g_prior = "hyper-g/n"), 1.0)
    end

    return (Coverage = mean(coverage; dims = 2), Methods = methods)
end

# Run simulation
m = 100
alphas = [0.0, 1/2, 1.0]

Random.seed!(42)
res = map(a -> run_simulation(m; α = a), alphas)


## Create a table displaying the results ##
function coverage_table_latex(res, alphas)
    methods = res[1].Methods
    # Create scenario labels dynamically from alphas
    scenarios = ["\\(\\alpha = $(a)\\)" for a in alphas]

    table_str = "\\begin{table}[ht]\n\\centering\n\\caption{Empirical coverage across \$1,000\$ iterations.}\n\\label{tab:coverage}\n"
    table_str *= "\\begin{tabular}{l" * "c"^length(scenarios) * "}\n"
    table_str *= "\\toprule\n"
    table_str *= "Method & " * join(scenarios, " & ") * " \\\\\n"
    table_str *= "\\midrule\n"

    for (i, method) in enumerate(methods)
        row_vals = []
        for j in 1:length(scenarios)
            coverage_val = res[j].Coverage[i]
            push!(row_vals, string(coverage_val))
        end
        table_str *= method * " & " * join(row_vals, " & ") * " \\\\\n"
    end

    table_str *= "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return println(table_str)
end


coverage_table_latex(res, alphas)