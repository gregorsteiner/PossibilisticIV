using Random, DataFrames, CSV
using LaTeXStrings
using Printf

include("../PossibilisticIV.jl")
include("competing_methods.jl")


## Data generating function ## 
function generate_data(n, R2_fs, s; ρ = 1/2, β = 1.0, p = 5)
    Z = rand(MvNormal(zeros(p), I), n)'

    c_instr = sqrt(R2_fs / ((1-R2_fs) * p))
    γ_2   = ones(p) .* c_instr
    α = 0.1 .* [ones(s); zeros(p-s)]

    u = rand(MvNormal([0, 0], [1 ρ; ρ 1]), n)'
    x = Z * γ_2 + u[:,2]
    y = β * x .+ Z * α + u[:,1]

    return (y, x, Z)
end


## VIPER wrapper function ##
"""
    viper_ci_wrapper(lower_α, upper_α, W, Z, true_β; type = "Chisq")

Compute coverage and confidence interval for VIPER methods (multi-dimensional case).
- Coverage: Pointwise evaluation at true_β
- Interval: Computed via root finding (returns NaN if it fails to converge)
"""
function viper_ci_wrapper(lower_α, upper_α, W, Z, true_β; type = "Chisq")
    # Compute pointwise coverage
    coverage = first(possibilistic_contour(true_β, lower_α, upper_α, W, Z; type = type)) > 0.05
    
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
function run_simulation(m, n, s, R2_fs; ρ = 1/2, p = 5)
    # different methods
    methods = [
        L"VIPER ($A = \{0\}, \chi^2$)",
        L"VIPER ($A = \{0\}$, MC)",
        L"VIPER ($A = [-0.1, 0.1]^p, \chi^2$)",
        L"VIPER ($A = [-0.1, 0.1]^p$, MC)",
        L"VIPER ($A = [0.0, 0.2]^p, \chi^2$)",
        L"VIPER ($A = [0.0, 0.2]^p$, MC)",
        "TSLS",
        "PGMM-g",
        "gIVBMA",
        L"LeakyIV ($\tau = 0.2$)",
        L"BudgetIV ($b = 1, \tau = 0$)",
        L"BudgetIV ($b = 1, \tau = 0.2$)",
        "CIIV"
        ]
    
    # storage objects
    coverage = Matrix{Bool}(undef, length(methods), m)
    interval_lengths = Matrix{Float64}(undef, length(methods), m)

    # true β
    true_β = 1.0

    # start iterating
    Threads.@threads for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, R2_fs, s; ρ = ρ, β = true_β, p = p)
        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))
        W = [Y X]

        # compute coverage based on the wrapper function above
        res_1 = viper_ci_wrapper(zeros(p), zeros(p), W, Z, true_β; type = "Chisq")
        coverage[1, i], interval_lengths[1, i] = res_1.coverage, res_1.interval_length
        
        res_2 = viper_ci_wrapper(zeros(p), zeros(p), W, Z, true_β; type = "MC")
        coverage[2, i], interval_lengths[2, i] = res_2.coverage, NaN
        
        res_3 = viper_ci_wrapper(-0.1 * ones(p), 0.1 * ones(p), W, Z, true_β; type = "Chisq")
        coverage[3, i], interval_lengths[3, i] = res_3.coverage, res_3.interval_length
        
        res_4 = viper_ci_wrapper(-0.1 * ones(p), 0.1 * ones(p), W, Z, true_β; type = "MC")
        coverage[4, i], interval_lengths[4, i] = res_4.coverage, NaN
        
        res_5 = viper_ci_wrapper(0.0 * ones(p), 0.2 * ones(p), W, Z, true_β; type = "Chisq")
        coverage[5, i], interval_lengths[5, i] = res_5.coverage, res_5.interval_length
        
        res_6 = viper_ci_wrapper(0.0 * ones(p), 0.2 * ones(p), W, Z, true_β; type = "MC")
        coverage[6, i], interval_lengths[6, i] = res_6.coverage, NaN

        # compute coverage for competing methods
        tsls_res = tsls(Y, X, Z)
        coverage[7, i], interval_lengths[7, i] = check_coverage(tsls_res, true_β), tsls_res.ci[2] - tsls_res.ci[1]
        
        pgmm_dist = pgmm(Y, X, Z, I)
        coverage[8, i], interval_lengths[8, i] = check_coverage(pgmm_dist, true_β), 2 * 1.96 * std(pgmm_dist)
        
        fit_givbma = givbma(Y, X, Z; g_prior = "hyper-g/n", iter = 1000, burn = 100)
        givbma_dist = rbw(fit_givbma)[1]
        coverage[9, i], interval_lengths[9, i] = check_coverage(givbma_dist, true_β), 2 * 1.96 * std(givbma_dist)
    end

    for i in 1:m
        # simulate data
        Y, X, Z = generate_data(n, R2_fs, s; ρ = ρ, β = true_β, p = p)
        # centre data
        Y, X = (Y .- mean(Y), X .- mean(X))
        # compute coverage
        leaky_res = leaky_iv(Y, X, Z, 0.2)
        coverage[10, i], interval_lengths[10, i] = check_coverage(leaky_res, true_β), leaky_res.ci[2] - leaky_res.ci[1]
        
        budget_0 = budgetIV(Y, X, Z, 0.0, 1)
        coverage[11, i], interval_lengths[11, i] = check_coverage(budget_0, true_β), budget_0.ci[2] - budget_0.ci[1]
        
        budget_half = budgetIV(Y, X, Z, 0.2, 1)
        coverage[12, i], interval_lengths[12, i] = check_coverage(budget_half, true_β), budget_half.ci[2] - budget_half.ci[1]
        
        ciiv_res = ciiv(Y, X, Z)
        coverage[13, i], interval_lengths[13, i] = check_coverage(ciiv_res, true_β), ciiv_res[1][2] - ciiv_res[1][1]
    end

    # Compute coverage rates and median interval lengths
    cover_rates = mean(coverage; dims = 2)[:, 1]
    
    mil = zeros(length(methods))
    for j in 1:length(methods)
        valid_lengths = filter(!isnan, interval_lengths[j, :])
        mil[j] = length(valid_lengths) > 0 ? median(valid_lengths) : NaN
    end

    return DataFrame(
        method = methods,
        coverage = cover_rates,
        mil = mil,
        s = fill(s, length(methods)),
        n = fill(n, length(methods)),
        R2_fs = fill(R2_fs, length(methods))
    )
end


## Run simulation ##

Random.seed!(42)
results = DataFrame()
for s in [0, 2, 3, 5]
    println("Running n=200, R2=0.15, s=$s")
    df = run_simulation(200, 200, s, 0.15)
    append!(results, df)
end

# run additional scenarios
m = 100 # number of iterations in each scenario
s_vals = [0, 1, 2, 3, 4, 5] # number of invalid instruments
n_vals = [50, 500] # sample sizes
R2_vals = [0.1, 0.25] # first-stage R^2 values
for n in n_vals
    for R2 in R2_vals
        for s in s_vals
            println("Running n=$n, R2=$R2, s=$s")
            df = run_simulation(m, n, s, R2)
            append!(results, df)
        end
    end
end



## Save results ##
CSV.write("Multiple_Instruments_Simulation_Results.csv", results)


# plot additional results
results = CSV.read("Multiple_Instruments_Simulation_Results.csv", DataFrame)

# Table of main text results
function df_to_latex_table(
    df::DataFrame;
    caption::String = "Caption here.",
    label::String = "tab:coverage"
)
    s_vals = sort(unique(df.s))
    methods = unique(df.method)

    lookup_cov = Dict((row.method, row.s) => row.coverage for row in eachrow(df))
    lookup_mil = Dict((row.method, row.s) => row.mil for row in eachrow(df))

    io = IOBuffer()

    col_spec = "l" * repeat("c", length(s_vals))
    println(io, "\\begin{tabular}{$col_spec}")
    println(io, "\\toprule")

    header_cols = join(["\\(s = $s\\)" for s in s_vals], " & ")
    println(io, "Method & $header_cols \\\\")
    println(io, "\\midrule")

    for method in methods
        cells = []
        for s in s_vals
            cov = get(lookup_cov, (method, s), NaN)
            mil = get(lookup_mil, (method, s), NaN)
            
            if isnan(mil)
                cell_str = @sprintf("%.3f", cov) * " [---]"
            else
                cell_str = @sprintf("%.3f", cov) * " [" * @sprintf("%.3f", mil) * "]"
            end
            push!(cells, cell_str)
        end
        println(io, method * " & " * join(cells, " & ") * " \\\\")
    end

    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    return String(take!(io))
end


bool_main = results.R2_fs .== 0.15
df_to_latex_table(
    results[bool_main, :];
    caption = ""
) |> println


# Plot of additional results
using Plots, Measures, LaTeXStrings

df = deepcopy(results[.!bool_main,:])
# Ensure we know the grid dimensions for labeling logic
n_vals = sort(unique(df.n))      # Rows? (Depends on your preference)
R2_vals = sort(unique(df.R2_fs)) # Columns?
methods = unique(df.method)


my_palette = palette(:turbo, length(methods)) 
gr()
default(
    linewidth = 2,
    markersize = 5,
    legendfontsize = 9,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 13,
    grid = false,
    framestyle = :axes,
    fontfamily = "Computer Modern" 
)

plots = []

# Assuming a 2x2 grid based on your layout
# We'll track indices to decide where to put labels
for (row_idx, n_val) in enumerate(n_vals)
    for (col_idx, R2_val) in enumerate(R2_vals)
        subdf = df[(df.n .== n_val) .& (df.R2_fs .== R2_val), :]
        
        # Only bottom row (row 2) gets xlabel
        # Only left column (col 1) gets ylabel
        show_x = (row_idx == 2)
        show_y = (col_idx == 1)

        p = plot(
            xlabel = show_x ? "Invalid Instruments (s)" : "",
            ylabel = show_y ? "|Coverage - 0.95|" : "",
            # Using L"" for LaTeX rendering in titles
            title = L"n = %$n_val, R^2 = %$R2_val",
            legend = false
        )

        #hline!(p, [0.95], linestyle = :dash, color = :black, alpha=0.4, label="")

        for (i, m) in enumerate(methods)
            data_m = subdf[subdf.method .== m, :]
            if !isempty(data_m)
                sort!(data_m, :s)
                plot!(p, data_m.s, abs.(data_m.coverage .- 0.95),
                      linestyle = (i <= 6) ? :solid : :dot,
                      marker = :circle, 
                      color = my_palette[i])
            end
        end
        push!(plots, p)
    end
end

# 4. Create the Legend-Only Plot
legend_plot = plot(
    grid = false, 
    showaxis = false, 
    ticks = false, 
    legend = :top,
    legend_columns = 3, # Grouped into rows for 11 methods
    background_color_subplot = :transparent,
    margins = 0mm
)

for (i, m) in enumerate(methods)
    # Wrap method names in LaTeXStrings wrapper
    # This handles both plain text and $...$ math strings
    plot!(legend_plot, [NaN], [NaN], 
          label = string(m),
          linestyle = (i <= 6) ? :solid : :dot,
          marker = :circle, 
          color = my_palette[i])
end

# 5. Final Assembly
l = @layout [
    grid(2, 2)
    legend_area{0.22h} 
]

final_plot = plot(
    plots..., legend_plot,
    layout = l,
    size = (900, 600),
    ylims = (-0.02, 0.31),
    # link=:both helps align axes when interior labels are hidden
    link = :both, 
    margin = 2mm
)

display(final_plot)
savefig("Multiple_Instruments_Additional_Results.pdf")
