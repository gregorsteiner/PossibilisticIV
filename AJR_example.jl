using CSV, DataFrames
using StatsPlots, LaTeXStrings

# import method function
include("PossibilisticIV.jl")
include("competing_methods.jl")

# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y_raw, x_raw, z_raw = (d.GDP, d.Exprop, d.logMort)
W = [ones(length(y_raw)) Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]])]

# project out covariates
M_W = I - W * inv(W'W) * W'
y, x, z = map(vec -> M_W * vec, (y_raw, x_raw, z_raw))

# Create plot
xx = -10:0.01:10
p1 = plot(
    xx, log.(possibilistic_contour(xx, [0.0], [0.0], [y x], z)),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = L"\log \pi_w(\beta \mid A)",
    label = L"\alpha = 0",
    legend = :topleft,
    size=(400, 300)
)
plot!(xx, log.(possibilistic_contour(xx, [-0.1], [0.1], [y x], z)), linewidth = 1.5, label = L"\alpha \in [-0.1, 0.1]")
plot!(xx, log.(possibilistic_contour(xx, [-0.18], [0.18], [y x], z)), linewidth = 1.5, label = L"\alpha \in [-0.18, 0.18]")

hline!([log(0.05)], linestyle = :dash, label = "", colour = :grey)

savefig(p1, "AJR_Possibility_Contour.pdf")


# Compute upper and lower probabilities
# for the hypothesis β > 0
function compute_upper_lower(lower_α, upper_α, W, Z)
    lower, upper = (1 - upper_probability(-1e10, 0.0, lower_α, upper_α, W, Z), upper_probability(0.0, 1e10, lower_α, upper_α, W, Z))
    return [lower, upper]
end

# benchmark the computational time
using BenchmarkTools
@btime probs_res = map(
    (b) -> compute_upper_lower([-b], [b], [y x], z),
    [0.0, 0.1, 0.18, 0.25, 0.35]
) 
probs_res = reduce(hcat, probs_res)'
round.(probs_res; digits = 3)

using LaTeXStrings, RCall
row_names = [L"\{0\}", L"[-0.1, 0.1]", L"[-0.18, 0.18]", L"[-0.25, 0.25]", L"[-0.35, 0.35]"]
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


# Compare approximation and more exact sampling scheme
xx = -0.1:0.01:3
p2 = plot(
    xx, possibilistic_contour(xx, [-0.0], [0.0], [y x], z),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = L"\pi_w(\beta \mid A)",
    label = L"\alpha = 0 \quad (\chi^2)",
    legend = false,
    size=(400, 300)
)
plot!(
    xx, possibilistic_contour(xx, [-0.1], [0.1], [y x], z),
    linewidth = 1.5, label = L"\alpha \in [-0.1, 0.1] \quad (\chi^2)"
)
plot!(
    xx, possibilistic_contour(xx, [-0.18], [0.18], [y x], z),
    linewidth = 1.5, label = L"\alpha \in [-0.18, 0.18] \quad (\chi^2)"
)

xx_exact = -0.1:0.05:3 # use less fine grid to save computation
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.0], [0.0], [y x], z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 1,
    label = L"\alpha = 0 \quad (MC)"
)
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.1], [0.1], [y x], z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 2,
    label = L"\alpha \in [-0.1, 0.1] \quad (MC)"
)
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.18], [0.18], [y x], z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 3,
    label = L"\alpha \in [-0.18, 0.18] \quad (MC)"
)
hline!([0.05], linestyle = :dash, label = "", colour = :grey)


savefig(p2, "Possibility_Contour_Comparison.pdf")
