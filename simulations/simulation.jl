using Random
using LaTeXStrings
using Printf

include("../PossibilisticIV.jl")
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


## VIPER wrapper function ##
"""
    viper_ci_wrapper(lower_α, upper_α, W, Z, true_β; type = "Chisq")

Compute coverage and confidence interval for VIPER methods.
- Coverage: Pointwise evaluation at true_β
- Interval: Computed via root finding (returns NaN if it fails to converge)
"""
function viper_ci_wrapper(lower_α, upper_α, W, Z, true_β; type = "Chisq")
    # Compute pointwise coverage
    coverage = first(possibilistic_contour([true_β], lower_α, upper_α, W, Z; type = type)) > 0.05
    
    # Try to compute confidence interval via root finding
    interval_length = NaN
    if type == "Chisq"
        try
            ci = confidence_interval(lower_α, upper_α, W, Z; level = 0.05)
            interval_length = ci.upper - ci.lower
        catch e
            # Root finding failed to converge, return NaN
            interval_length = NaN
        end
    end
    # For MC type, we don't compute intervals (too expensive)
    
    return (coverage = coverage, interval_length = interval_length)
end

## Write function to implement the simulation ##
function run_simulation(m; n = 100, ρ = 1/2, α = 0.0)
    # different methods
    methods = [
        L"VIPER ($A = \{0\}, \chi^2$)",
        L"VIPER ($A = \{0\}$, MC)",
        L"VIPER ($A = [-0.5, 0.5], \chi^2$)",
        L"VIPER ($A = [-0.5, 0.5]$, MC)",
        L"VIPER ($A = [0.0, 0.5], \chi^2$)",
        L"VIPER ($A = [0.0, 0.5]$, MC)",
        "TSLS",
        "PGMM-g",
        L"BudgetIV ($\tau = 0$)",
        L"BudgetIV ($\tau = 0.5$)"
        ]
    
    # storage objects
    coverage = Matrix{Bool}(undef, length(methods), m)
    interval_lengths = Matrix{Float64}(undef, length(methods), m)

    # true β
    true_β = 1.0

    # use a parallel loop for all native julia methods
    Threads.@threads for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, ρ, α; β = true_β)
        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))
        W = [Y X]

        # Chi-square VIPER methods: compute coverage and confidence intervals
        # Method 1: VIPER (A = {0}, χ² approximation)
        res_1 = viper_ci_wrapper(0.0, 0.0, W, Z, true_β; type = "Chisq")
        coverage[1, i], interval_lengths[1, i] = res_1.coverage, res_1.interval_length

        # Method 3: VIPER (A = [-0.5, 0.5], χ² approximation)
        res_3 = viper_ci_wrapper(-1/2, 1/2, W, Z, true_β; type = "Chisq")
        coverage[3, i], interval_lengths[3, i] = res_3.coverage, res_3.interval_length

        # Method 5: VIPER (A = [0.0, 0.5], χ² approximation)
        res_5 = viper_ci_wrapper(0.0, 1/2, W, Z, true_β; type = "Chisq")
        coverage[5, i], interval_lengths[5, i] = res_5.coverage, res_5.interval_length

        # MC VIPER methods: pointwise evaluation only
        res_2 = viper_ci_wrapper(0.0, 0.0, W, Z, true_β; type = "MC")
        coverage[2, i], interval_lengths[2, i] = res_2.coverage, res_2.interval_length
        
        res_4 = viper_ci_wrapper(-1/2, 1/2, W, Z, true_β; type = "MC")
        coverage[4, i], interval_lengths[4, i] = res_4.coverage, res_4.interval_length
        
        res_6 = viper_ci_wrapper(0.0, 1/2, W, Z, true_β; type = "MC")
        coverage[6, i], interval_lengths[6, i] = res_6.coverage, res_6.interval_length

        # Compute coverage for all competing methods
        tsls_res = tsls(Y, X, Z)
        coverage[7, i], interval_lengths[7, i] = check_coverage(tsls_res, true_β), tsls_res.ci[2] - tsls_res.ci[1]

        pgmm_dist = pgmm(Y, X, Z, I)  # PGMM
        coverage[8, i], interval_lengths[8, i] = check_coverage(pgmm_dist, true_β), 2 * 1.96 * std(pgmm_dist)
    end

    # separate loop for the R-based methods
    # they are not multi-thread compatible
    for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, ρ, α; β = true_β)
        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))
        
        # compute coverage
        budgetiv_0 = budgetIV(Y, X, Z, 0.0, 1)  # BudgetIV with budget 0
        coverage[9, i], interval_lengths[9, i] = check_coverage(budgetiv_0, true_β), budgetiv_0.ci[2] - budgetiv_0.ci[1]

        budgetiv_half = budgetIV(Y, X, Z, 1/2, 1)  # BudgetIV with budget 1/2
        coverage[10, i], interval_lengths[10, i] = check_coverage(budgetiv_half, true_β), budgetiv_half.ci[2] - budgetiv_half.ci[1]
    end

    # Compute coverage rates and median interval lengths
    coverage_rates = mean(coverage; dims = 2)[:, 1]
    
    # For MC methods, compute MIL only from non-NaN values (other methods)
    mil = zeros(length(methods))
    for j in 1:length(methods)
        valid_lengths = filter(!isnan, interval_lengths[j, :])
        mil[j] = length(valid_lengths) > 0 ? median(valid_lengths) : NaN
    end

    return (Coverage = coverage_rates, MIL = mil, Methods = methods, alpha = α)
end


## Run simulation ##
m = 500
alphas = [0.0, 0.25, 0.5]

Random.seed!(42)
res = map(a -> run_simulation(m; α = a), alphas)

## Save results ##
using DataFrames, CSV

# Build dataframe with method, scenario, coverage, and MIL
all_results = []
for (idx, scenario) in enumerate(res)
    scenario_name = "α = " * string(scenario.alpha)
    for (method_idx, method) in enumerate(scenario.Methods)
        push!(all_results, (
            Method = method,
            Scenario = scenario_name,
            Coverage = scenario.Coverage[method_idx],
            MIL = scenario.MIL[method_idx]
        ))
    end
end

df = DataFrame(all_results)
CSV.write("Simulation_Results.csv", df)

println("Results DataFrame:")
println(df)

## Create a table displaying the results ##
function coverage_table_latex(res, alphas)
    methods = res[1].Methods
    # Create scenario labels dynamically from alphas
    scenarios = ["\\(\\alpha = $(a)\\)" for a in alphas]

    # Find the index of the value closest to 0.95 for each scenario (column)
    best_indices = []
    for j in 1:length(scenarios)
        coverages = res[j].Coverage
        # Compute distances to 0.95
        distances = [abs(c - 0.95) for c in coverages]
        # Find the index of the min distance (i.e., closest to 0.95)
        push!(best_indices, argmin(distances))
    end

    table_str = "\\begin{table}[ht]\n\\centering\n\\caption{Empirical coverage of \$95\\%\$ uncertainty intervals across \$500\$ simulated datasets ot size \$n =100\$. The value closest to the nominal coverage in each column is printed in bold. Median interval lengths (MIL) are reported in brackets.}\n\\label{tab:coverage}\n"
    table_str *= "\\begin{tabular}{l" * "c"^length(scenarios) * "}\n"
    table_str *= "\\toprule\n"
    table_str *= "Method & " * join(scenarios, " & ") * " \\\\\n"
    table_str *= "\\midrule\n"

    for (i, method) in enumerate(methods)
        row_vals = []
        for j in 1:length(scenarios)
            coverage_val = res[j].Coverage[i]
            mil_val = res[j].MIL[i]
            
            # Format MIL value (handle NaN for MC methods)
            mil_str = isnan(mil_val) ? "---" : @sprintf("%.3f", mil_val)
            cell_content = @sprintf("%.3f", coverage_val) * " [" * mil_str * "]"
            
            if i == best_indices[j]
                push!(row_vals, "\\textbf{" * cell_content * "}")
            else
                push!(row_vals, cell_content)
            end
        end
        table_str *= method * " & " * join(row_vals, " & ") * " \\\\\n"
    end

    table_str *= "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return println(table_str)
end

coverage_table_latex(res, alphas)
