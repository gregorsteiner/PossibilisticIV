using CSV, DataFrames
using Plots, LaTeXStrings, Measures

# import method function
include("../PossibilisticIV.jl")
include("../simulations/competing_methods.jl")

# load data
d = CSV.read(joinpath(@__DIR__, "AJR_Data.csv"), DataFrame)
y_raw, x_raw, z_raw = (d.GDP, d.Exprop, d.logMort)
W = [ones(length(y_raw)) Matrix(d[:, ["Latitude"]])]

# project out covariates
M_W = I - W * inv(W'W) * W'
y, x, z = map(vec -> M_W * vec, (y_raw, x_raw, z_raw))

# plot results
my_palette = [
    "#E69F00", # Orange
    "#56B4E9", # Sky Blue
    "#009E73", # Bluish Green
]
default(
    fontfamily="Computer Modern",
    titlefontsize=11, 
    guidefontsize=13, 
    tickfontsize=11, 
    legendfontsize=9,
    frame=:axes,
    tick_direction = :out,
    grid=false,
    lw=2.0,
    palette=my_palette,
    dpi=300
)

intervals = [([-0.0], [0.0]), ([-0.1], [0.1]), ([-0.4], [0.4])]
labels = [L"A = \{0\}", L"A = [-0.1, 0.1]", L"A = [-0.4, 0.4]"]

p = plot(xlabel = L"\beta", ylabel = L"\pi_w(\beta \mid A)", xlims=(-0.1, 2.8), ylims=(0, 1.05))
xx, xx_exact = -0.1:0.01:2.8, -0.1:0.02:2.8
for i in 1:length(intervals)
    low, high = intervals[i]
    
    # Approximation (Solid)
    plot!(xx, possibilistic_contour(xx, low, high, [y x], z), 
          label = labels[i], color = i)
    
    # Exact MC (Dashed)
    plot!(xx_exact, possibilistic_contour(xx_exact, low, high, [y x], z; type = "MC"),
          ls = :dash, color = i, label = "")
end

# Add threshold line with annotation
hline!([0.05], ls = :dot, lc = :black, alpha=0.5, label = "")

# Final Layout
plot!(
    p, 
    size=(550, 275),
    margin = 1.5mm,
    legend = :topright,
    legend_foreground_color = :transparent,
    legend_column = 1
)
savefig(p, "AJR_Possibility_Contour.pdf")


# find uncertainty intervals
using Roots

println(
    "95%-Interval for α = 0: " *
    string(confidence_interval([-0.0], [0.0], [y x], z; level = 0.05))
)
println(
    "95%-Interval for A = [-0.1, 0.1]: " *
    string(confidence_interval([-0.1], [0.1], [y x], z; level = 0.05))
)
println(
    "95%-Interval for A = [-0.4, 0.4]: " *
    string(confidence_interval([-0.4], [0.4], [y x], z; level = 0.05))
)


# Compute upper and lower probabilities
# for the hypothesis β > 0
function compute_upper_lower(lower_α, upper_α, W, Z; type = "Chisq", M = 1000)
    lower, upper = (
        1 - upper_probability(-1e10, 0.0, lower_α, upper_α, W, Z; type = type, M = M),
        upper_probability(0.0, 1e10, lower_α, upper_α, W, Z; type = type, M = M)
        )
    return [lower, upper]
end

probs_res = map(
    (b) -> compute_upper_lower([-b], [b], [y x], z; type = "MC", M = 10000),
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
) 
probs_res = reduce(hcat, probs_res)'
round.(probs_res; digits = 3)

using LaTeXStrings, RCall
row_names = [L"\{0\}", L"[-0.1, 0.1]", L"[-0.2, 0.2]", L"[-0.3, 0.3]", L"[-0.4, 0.4]", L"[-0.5, 0.5]"]
@rput probs_res row_names
R"""
rownames(probs_res) = row_names
knitr::kable(
    probs_res, 'latex', booktabs = TRUE,
    digits = 3,
    col.names = c("A", "Lower", "Upper"),
    escape = FALSE
    )
"""
