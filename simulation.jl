
using Random
using LaTeXStrings

include("PossibilisticIV.jl")
include("competing_methods.jl")


## Data generating function ## 
function generate_data(n, ρ, α; β = 1.0)
    γ_2 = 1.0
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
        L"Possibilistic IV $(A = \{0\})$",
        L"Possibilistic IV $(A = [-0.5, 0.5])$",
        L"Possibilistic IV $(A = [0.0, 0.5])$",
        "TSLS",
        "PGMM-g",
        L"BudgetIV ($\alpha = 0$)",
        L"BudgetIV ($\lvert \alpha \rvert \leq 0.5$)"
        ]
    
    # storage objects
    coverage = Matrix{Bool}(undef, length(methods), m)

    # true β
    true_β = 1.0

    # start iterating
    for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, ρ, α; β = true_β)

        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))

        # compute coverage
        # possibilistic contour at the true value must be > 0.05
        coverage[1, i] = possibilistic_contour(true_β, 0.0, 0.0, [Y X], Z) > 0.05
        coverage[2, i] = possibilistic_contour(true_β, -1/2, 1/2, [Y X], Z) > 0.05
        coverage[3, i] = possibilistic_contour(true_β, -0.0, 1/2, [Y X], Z) > 0.05
        coverage[4, i] = check_coverage(tsls(Y, X, Z), true_β) # Naive TSLS
        coverage[5, i] = check_coverage(pgmm(Y, X, Z, I), true_β) # PGMM
        coverage[6, i] =  check_coverage(budgetIV(Y, X, Z, 0.0, 1), true_β) # BudgetIV with budget 0
        coverage[7, i] =  check_coverage(budgetIV(Y, X, Z, 1/2, 1), true_β) # BudgetIV with budget 1/2
    end

    return (Coverage = mean(coverage; dims = 2), Methods = methods)
end


# Run simulation
m = 1000
alphas = [0.0, 0.25, 0.5]

Random.seed!(42)
res = map(a -> run_simulation(m; α = a), alphas)


## Create a table displaying the results ##
function coverage_table_latex(res, alphas)
    methods = res[1].Methods
    # Create scenario labels dynamically from alphas
    scenarios = ["\\(\\alpha = $(a)\\)" for a in alphas]

    # Find the index of the value closest to 0.95 for each scenario (column)
    best_indices = []
    for j in 1:length(scenarios)
        coverages = res[j].Coverage[:, 1]
        # Compute distances to 0.95
        distances = [abs(c - 0.95) for c in coverages]
        # Find the index of the min distance (i.e., closest to 0.95)
        push!(best_indices, argmin(distances))
    end

    table_str = "\\begin{table}[ht]\n\\centering\n\\caption{Empirical coverage of \$95\\%\$ uncertainty intervals across \$1,000\$ simulated datasets ot size \$n =100\$. The value closest to the nominal coverage in each column is printed in bold.}\n\\label{tab:coverage}\n"
    table_str *= "\\begin{tabular}{l" * "c"^length(scenarios) * "}\n"
    table_str *= "\\toprule\n"
    table_str *= "Method & " * join(scenarios, " & ") * " \\\\\n"
    table_str *= "\\midrule\n"

    for (i, method) in enumerate(methods)
        row_vals = []
        for j in 1:length(scenarios)
            coverage_val = res[j].Coverage[i]
            if i == best_indices[j]
                push!(row_vals, "\\textbf{$(coverage_val)}")
            else
                push!(row_vals, string(coverage_val))
            end
        end
        table_str *= method * " & " * join(row_vals, " & ") * " \\\\\n"
    end

    table_str *= "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return println(table_str)
end

coverage_table_latex(res, alphas)