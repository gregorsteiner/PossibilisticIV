using CSV, DataFrames
using StatsPlots, LaTeXStrings

# import method function
include("PossibilisticIV.jl")
include("competing_methods.jl")

# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y_raw, x_raw, z_raw = (d.GDP, d.Exprop, d.logMort)
W = [ones(length(y_raw)) Matrix(d[:, ["Latitude"]])]

# project out covariates
M_W = I - W * inv(W'W) * W'
y, x, z = map(vec -> M_W * vec, (y_raw, x_raw, z_raw))


# Create plot with results (both approximation and more exact sampling scheme)
xx = -0.1:0.01:2.2
p = plot(
    xx, possibilistic_contour(xx, [-0.0], [0.0], [y x], z),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = L"\pi_w(\beta \mid A)",
    label = L"A = \{0\}",
    legend = true
)
plot!(
    xx, possibilistic_contour(xx, [-0.1], [0.1], [y x], z),
    linewidth = 1.5, label = L"A = [-0.1, 0.1]"
)

xx_exact = -0.1:0.02:2.2 # use less fine grid to save computation
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.0], [0.0], [y x], z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 1,
    label = ""
)
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.1], [0.1], [y x], z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 2,
    label = ""
)
hline!([0.05], linestyle = :dash, label = "", colour = :grey)

# add PGMM-g for comparison
#d_pgmm = pgmm(y, x, z, 0.05 * mean(z_raw.^2) * I)
#plot!(xx, pdf(d_pgmm, xx), label = L"\mathrm{PGMM\textnormal{-}g}", linewidth = 1.5)



# save both plots
p = plot(p, size=(500, 250))
savefig(p, "AJR_Possibility_Contour.pdf")


# find uncertainty intervals
using Roots
f(r; a = 0.0) = first(possibilistic_contour([r], [-a], [a], [y x], z)) - 0.05
println(
    "95%-Interval for α = 0: " *
    string(round.((find_zero(f, 0.6), find_zero(f, 1.4)), digits = 2))
)
println(
    "95%-Interval for A = [-0.1, 0.1]: " *
    string(round.((find_zero(r -> f(r, a = 0.1), 0.6), find_zero(r -> f(r, a = 0.1), 1.4)), digits = 2))
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


