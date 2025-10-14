
using Random
using LaTeXStrings

include("PossibilisticIV.jl")
include("competing_methods.jl")


## Data generating function ## 
function generate_data(n, s; ρ = 1/2, β = 1.0, p = 5)
    Z = rand(MvNormal(zeros(p), I), n)'

    γ_2   = ones(p) .* 1/4 # chosen s.t. the first-stage R^2 is approximately 0.25
    α = 0.1 .* [ones(s); zeros(p-s)]

    u = rand(MvNormal([0, 0], [1 ρ; ρ 1]), n)'
    x = Z * γ_2 + u[:,2]
    y = β * x .+ Z * α + u[:,1]

    return (y, x, Z)
end

## Write function to implement the simulation ##
function run_simulation(s; m = 100, n = 100, ρ = 1/2, p = 5)
    # different methods
    methods = [
        L"Possibilistic IV ($A = \{0\}, \chi^2$-Appr.)",
        L"Possibilistic IV ($A = [-0.1, 0.1]^p, \chi^2$-Appr.)",
        L"Possibilistic IV ($A = [-0.1, 0.1]^p$, MC)",
        L"Possibilistic IV ($A = [0.0, 0.2]^p, \chi^2$-Appr.)",
        L"Possibilistic IV ($A = [0.0, 0.2]^p$, MC)",
        "TSLS",
        "PGMM-g",
        "gIVBMA",
        L"BudgetIV ($\alpha = 0$)",
        L"BudgetIV ($\lvert \alpha_i \rvert \leq 0.2$)",
        "CIIV"
        ]
    
    # storage objects
    coverage = Matrix{Bool}(undef, length(methods), m)

    # true β
    true_β = 1.0

    # start iterating
    Threads.@threads for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, s; ρ = ρ, β = true_β, p = p)
        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))

        # compute coverage
        # possibilistic contour at the true value must be > 0.05
        coverage[1, i] = first(possibilistic_contour(true_β, zeros(p), zeros(p), [Y X], Z)) > 0.05
        coverage[2, i] = first(possibilistic_contour(true_β, -0.1 * ones(p), 0.1 * ones(p), [Y X], Z)) > 0.05
        coverage[3, i] = first(possibilistic_contour(true_β, -0.1 * ones(p), 0.1 * ones(p), [Y X], Z; type = "MC")) > 0.05
        coverage[4, i] = first(possibilistic_contour(true_β, -0.0 * ones(p), 0.2 * ones(p), [Y X], Z)) > 0.05
        coverage[5, i] = first(possibilistic_contour(true_β, -0.0 * ones(p), 0.2 * ones(p), [Y X], Z; type = "MC")) > 0.05

        # compute coverage for competing methods
        coverage[6, i] = check_coverage(tsls(Y, X, Z), true_β) # Naive TSLS
        coverage[7, i] = check_coverage(pgmm(Y, X, Z, I), true_β) # PGMM
        fit_givbma = givbma(Y, X, Z; g_prior = "hyper-g/n", iter = 1000, burn = 100) # gIVBMA
        coverage[8, i] = check_coverage(rbw(fit_givbma)[1], true_β)        
    end

    for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, s; ρ = ρ, β = true_β, p = p)
        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))
        # compute coverage
        coverage[9, i] =  check_coverage(budgetIV(Y, X, Z, 0.0, 1), true_β) # BudgetIV with budget 0
        coverage[10, i] =  check_coverage(budgetIV(Y, X, Z, 0.2, 1), true_β) # BudgetIV with budget 1/2
        coverage[11, i] = check_coverage(ciiv(Y, X, Z), true_β) # CIIV
    end

    return (Coverage = mean(coverage; dims = 2)[:, 1], Methods = methods, s = s)
end


## Run simulation ##
m = 500
ss = [0, 2, 3, 5]
Random.seed!(42)
res = map(s -> run_simulation(s; m = m), ss)


## Save results ##
using DataFrames, CSV
df = DataFrame(Method = res[1].Methods)
for scenario in res
    name = "s = " * string(scenario.s)
    df[!, name] = scenario.Coverage
end

CSV.write("Multiple_Instruments_Simulation_Results.csv", df)

## Latex table displaying the results ##
function coverage_table_latex(res, ss)
    methods = res[1].Methods
    # Create scenario labels dynamically from alphas
    scenarios = ["\\(s = $(s)\\)" for s in ss]

    # Find the index of the value closest to 0.95 for each scenario (column)
    best_indices = []
    for j in 1:length(scenarios)
        coverages = res[j].Coverage
        # Compute distances to 0.95
        distances = [abs(c - 0.95) for c in coverages]
        # Find the index of the min distance (i.e., closest to 0.95)
        push!(best_indices, argmin(distances))
    end

    table_str = "\\begin{table}[ht]\n\\centering\n\\caption{Empirical coverage of \$95\\%\$ uncertainty intervals across \$500\$ simulated datasets ot size \$n =100\$, where \$s\$ out of \$p=5\$ instruments are invalid with \$\\alpha_i = 0.1\$. The value closest to the nominal coverage in each column is printed in bold.}\n\\label{tab:coverage_multiple_instruments}\n"
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


coverage_table_latex(res, ss)