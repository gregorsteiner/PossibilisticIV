using Random
using LaTeXStrings, CSV, DataFrames
using Distributions
using StatsBase
using Printf

include("../PossibilisticIV.jl")
include("competing_methods.jl")


## Data generating functions with alternative error distributions ##

# Student-t errors with ν degrees of freedom
function generate_data_t(n, ρ, α; β = 1.0, ν = 3)
    γ_2 = 1.0
    Z = rand(Normal(0, 1), n)
    
    # Generate correlated t-distributed errors
    # Use t-distribution scaled to have unit variance (approximately)
    t_scale = sqrt((ν - 2) / ν)
    u_1 = rand(TDist(ν), n) / t_scale
    u_2 = rand(TDist(ν), n) / t_scale
    
    # Create correlation between errors
    u_2_corr = ρ * u_1 + sqrt(1 - ρ^2) * u_2
    
    X = Z * γ_2 + u_1
    Y = β * X + α * Z + u_2_corr
    return Y, X, Z
end

# Skewed-normal errors (using moment-based approach)
# Positively skewed: more mass on the left, tail on the right
function generate_data_skewnormal(n, ρ, α; β = 1.0, skewness = 1.0)
    γ_2 = 1.0
    Z = rand(Normal(0, 1), n)
    
    # Generate skewed errors using mixture approach
    # Combine normal and chi-squared to induce skewness
    u_1 = randn(n)
    chi_component = rand(Chisq(1), n) .- 1  # Center to have mean 0
    u_1_skew = u_1 + skewness * chi_component / sqrt(8)  # Scale by skewness parameter
    
    u_2 = randn(n)
    chi_component_2 = rand(Chisq(1), n) .- 1
    u_2_skew = u_2 + skewness * chi_component_2 / sqrt(8)
    
    # Create correlation
    u_2_corr = ρ * u_1_skew + sqrt(max(0, 1 - ρ^2)) * u_2_skew
    
    # Standardize to have unit variance
    u_1_skew = (u_1_skew .- mean(u_1_skew)) / std(u_1_skew)
    u_2_corr = (u_2_corr .- mean(u_2_corr)) / std(u_2_corr)
    
    X = Z * γ_2 + u_1_skew
    Y = β * X + α * Z + u_2_corr
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


## Simulation function with custom error distribution ##
function run_simulation_custom(m, data_gen_func; n = 100, ρ = 1/2, α = 0.0)
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

    # start iterating with parallel loop for Julia methods
    Threads.@threads for i in 1:m
        # simulate data using custom function
        Y, X, Z = data_gen_func(n, ρ, α; β = true_β)
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

    # separate loop for R-based methods
    for i in 1:m
        Y, X, Z = data_gen_func(n, ρ, α; β = true_β)
        Y, X = (Y .- mean(Y), X .- mean(X))
        
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


## Run simulations with different error distributions ##
println("Running simulations with alternative error distributions...")

m = 200
alphas = [0.0, 0.25, 0.5]

Random.seed!(42)

# Student-t errors (ν = 3, heavy-tailed)
println("Running Student-t error simulations (ν=3)...")
res_t = map(a -> run_simulation_custom(m, (n, ρ, α; β=1.0) -> generate_data_t(n, ρ, α; β=β, ν=3); α = a), alphas)

# Skewed-normal errors
println("Running skewed-normal error simulations...")
res_skewnorm = map(a -> run_simulation_custom(m, (n, ρ, α; β=1.0) -> generate_data_skewnormal(n, ρ, α; β=β, skewness=1.0); α = a), alphas)


## Save results ##
using DataFrames, CSV

# Helper function to create results dataframe
function create_results_df(res_list, alphas)
    df = DataFrame(Method = res_list[1].Methods)
    for (idx, scenario) in enumerate(res_list)
        col_name = "α = $(alphas[idx])"
        df[!, col_name] = scenario.Coverage
    end
    return df
end

# Helper function to create results dataframe with both Coverage and MIL
function create_results_df_with_mil(res_list, alphas, metric="Coverage")
    results = []
    for (idx, scenario) in enumerate(res_list)
        scenario_name = "α = " * string(alphas[idx])
        for (method_idx, method) in enumerate(scenario.Methods)
            metric_val = metric == "Coverage" ? scenario.Coverage[method_idx] : scenario.MIL[method_idx]
            push!(results, (
                Method = method,
                Scenario = scenario_name,
                Metric = metric_val
            ))
        end
    end
    return DataFrame(results)
end

# Create results dataframes with Coverage and MIL
all_results_t = []
for (idx, scenario) in enumerate(res_t)
    scenario_name = "α = " * string(alphas[idx])
    for (method_idx, method) in enumerate(scenario.Methods)
        push!(all_results_t, (
            Method = method,
            Scenario = scenario_name,
            Coverage = scenario.Coverage[method_idx],
            MIL = scenario.MIL[method_idx]
        ))
    end
end
df_t = DataFrame(all_results_t)

all_results_skewnorm = []
for (idx, scenario) in enumerate(res_skewnorm)
    scenario_name = "α = " * string(alphas[idx])
    for (method_idx, method) in enumerate(scenario.Methods)
        push!(all_results_skewnorm, (
            Method = method,
            Scenario = scenario_name,
            Coverage = scenario.Coverage[method_idx],
            MIL = scenario.MIL[method_idx]
        ))
    end
end
df_skewnorm = DataFrame(all_results_skewnorm)

# Save results
CSV.write("Simulation_Results_StudentT.csv", df_t)
CSV.write("Simulation_Results_SkewNormal.csv", df_skewnorm)


## Create comparison table ##
function coverage_table_latex_alt_errors(results_dict, alphas)
    methods = unique(results_dict["t(3)"].Method)
    distribution_names = sort(collect(keys(results_dict)))
    
    table_str = "\\begin{table}[ht]\n\\centering\n"
    table_str *= "\\caption{Empirical coverage of \$95\\%\$ uncertainty intervals across \$200\$ simulated datasets of size \$n=100\$ under alternative error distributions. The values closest to the nominal coverage are printed in bold. Median interval lengths are reported in brackets.}\n"
    table_str *= "\\label{tab:coverage_alt_errors}\n"
    table_str *= "\\begin{tabular}{l" * "c"^(length(distribution_names) * length(alphas)) * "}\n"
    table_str *= "\\toprule\n"
    
    # Header: distribution names
    table_str *= "Method"
    for dist in distribution_names
        table_str *= " & \\multicolumn{$(length(alphas))}{c}{$dist}"
    end
    table_str *= " \\\\\n"
    
    # Sub-header: alpha values
    table_str *= "& " * join([join(["\\(\\alpha=$(a)\\)" for a in alphas], " & ") for _ in distribution_names], " & ")
    table_str *= " \\\\\n\\midrule\n"
    
    # Methods and values
    for (i, method) in enumerate(methods)
        table_str *= method
        for dist in distribution_names
            for alpha in alphas
                scenario_name = "α = $alpha"
                row = filter(row -> row.Method == method && row.Scenario == scenario_name, results_dict[dist])[1, :]
                coverage_val = row.Coverage
                mil_val = row.MIL
                
                # Format MIL value (handle NaN for MC methods)
                mil_str = isnan(mil_val) ? "---" : @sprintf("%.3f", mil_val)
                cell_content = @sprintf("%.3f", coverage_val) * " [" * mil_str * "]"
                
                table_str *= " & " * cell_content
            end
        end
        table_str *= " \\\\\n"
    end
    
    table_str *= "\\bottomrule\n\\end{tabular}\n\\end{table}"
    return println(table_str)
end


res_t = CSV.read("Simulation_Results_StudentT.csv", DataFrame)
res_skewnorm = CSV.read("Simulation_Results_SkewNormal.csv", DataFrame)

# Create results dictionary
results_dict = Dict(
    "t(3)" => res_t,
    "SkewNormal" => res_skewnorm
)

println("\n\nLaTeX Table:\n")
coverage_table_latex_alt_errors(results_dict, alphas)
