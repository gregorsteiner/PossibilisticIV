using Random
using LaTeXStrings
using Distributions
using StatsBase

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

    # true β
    true_β = 1.0

    # start iterating with parallel loop for Julia methods
    Threads.@threads for i in 1:m
        # simulate data using custom function
        Y, X, Z = data_gen_func(n, ρ, α; β = true_β)
        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))

        # compute coverage
        coverage[1, i] = first(possibilistic_contour([true_β], 0.0, 0.0, [Y X], Z)) > 0.05
        coverage[2, i] = first(possibilistic_contour([true_β], 0.0, 0.0, [Y X], Z; type = "MC")) > 0.05
        coverage[3, i] = first(possibilistic_contour([true_β], -1/2, 1/2, [Y X], Z)) > 0.05
        coverage[4, i] = first(possibilistic_contour([true_β], -1/2, 1/2, [Y X], Z; type = "MC")) > 0.05
        coverage[5, i] = first(possibilistic_contour([true_β], -0.0, 1/2, [Y X], Z)) > 0.05
        coverage[6, i] = first(possibilistic_contour([true_β], -0.0, 1/2, [Y X], Z; type = "MC")) > 0.05

        # Compute coverage for competing methods
        coverage[7, i] = check_coverage(tsls(Y, X, Z), true_β)
        coverage[8, i] = check_coverage(pgmm(Y, X, Z, I), true_β)
    end

    # separate loop for R-based methods
    for i in 1:m
        Y, X, Z = data_gen_func(n, ρ, α; β = true_β)
        Y, X = (Y .- mean(Y), X .- mean(X))
        
        coverage[9, i] =  check_coverage(budgetIV(Y, X, Z, 0.0, 1), true_β)
        coverage[10, i] =  check_coverage(budgetIV(Y, X, Z, 1/2, 1), true_β)
    end

    return (Coverage = mean(coverage; dims = 2)[:, 1], Methods = methods, alpha = α)
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

# Create results dataframes
df_t = create_results_df(res_t, alphas)
df_skewnorm = create_results_df(res_skewnorm, alphas)

# Save results
CSV.write("Simulation_Results_StudentT.csv", df_t)
CSV.write("Simulation_Results_SkewNormal.csv", df_skewnorm)

println("Simulations complete! Results saved to:")
println("  - Simulation_Results_StudentT.csv")
println("  - Simulation_Results_SkewNormal.csv")


## Create comparison table ##
function coverage_table_latex_alt_errors(results_dict, alphas)
    methods = results_dict["t(3)"][1].Methods
    distribution_names = sort(collect(keys(results_dict)))
    
    table_str = "\\begin{table}[ht]\n\\centering\n"
    table_str *= "\\caption{Empirical coverage of \$95\\%\$ uncertainty intervals across \$1,000\$ simulated datasets of size \$n=100\$ under alternative error distributions. The value closest to nominal coverage in each column is printed in bold.}\n"
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
                coverage_val = round(results_dict[dist][findfirst(x -> x.alpha == alpha, results_dict[dist])].Coverage[i]; digits=3)
                table_str *= " & $coverage_val"
            end
        end
        table_str *= " \\\\\n"
    end
    
    table_str *= "\\bottomrule\n\\end{tabular}\n\\end{table}"
    return println(table_str)
end

# Create results dictionary
results_dict = Dict(
    "t(3)" => res_t,
    "SkewNormal" => res_skewnorm
)

println("\n\nLaTeX Table:\n")
coverage_table_latex_alt_errors(results_dict, alphas)
