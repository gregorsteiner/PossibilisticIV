using CSV, DataFrames
using Plots, LaTeXStrings


include("PossibilisticIV.jl")
include("competing_methods.jl")

# load and prepare data
d = CSV.read("card.csv", DataFrame, missingstring = "NA", types = Union{Float64, Missing})[:, Not(1)]

d.agesq = d.age .^ 2
d.expersq = d.exper .^ 2
d.fatheduc_missing = ismissing.(d.fatheduc)
d.fatheduc[d.fatheduc_missing] .= mean(d.fatheduc[.!d.fatheduc_missing])

d.motheduc_missing = ismissing.(d.motheduc)
d.motheduc[d.motheduc_missing] .= mean(d.motheduc[.!d.motheduc_missing])

covs = ["exper", "expersq", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married", "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669", "fatheduc", "motheduc", "fatheduc_missing", "motheduc_missing"]

d = d[:, [["lwage", "educ", "nearc4"]; covs]]
dropmissing!(d)

# partial out covariates
U = [ones(nrow(d)) Matrix(d[:, covs])]
M_U = I - U * inv(U'U) * U'
y = M_U * d.lwage
X = M_U * d.educ
Z = M_U * d.nearc4

# calibrate violation set
Z'Z \ Z'y .- Z'Z \ Z'X * 0.0


# Run analysis
xx = -0.1:0.001:0.3
p = plot(
    xx, possibilistic_contour(xx, [-0.0], [0.0], [y X], Z),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = L"\pi_w(\beta \mid A)",
    label = L"A = \{0\}",
    legend = true
)
plot!(
    xx, possibilistic_contour(xx, [0.0], [0.02], [y X], Z),
    linewidth = 1.5, label = L"A = [0, 0.02]"
)
plot!(
    xx, possibilistic_contour(xx, [0.0], [0.04], [y X], Z),
    linewidth = 1.5, label = L"A = [0, 0.04]"
)

xx_exact = -0.1:0.005:0.3
plot!(
    xx_exact, possibilistic_contour(xx_exact, [0.0], [0.0], [y X], Z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 1, label = ""
)
plot!(
    xx_exact, possibilistic_contour(xx_exact, [0.0], [0.02], [y X], Z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 2, label = ""
)
plot!(
    xx_exact, possibilistic_contour(xx_exact, [0.0], [0.04], [y X], Z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 3, label = ""
)

hline!([0.05], linestyle = :dash, label = "", colour = :grey)
p = plot(p, size=(600, 300), legend = :outerbottom, legend_column = 3, bottom_margin=-4Plots.mm)
savefig(p, "Schooling_Possibility_Contour.pdf")


# find uncertainty intervals
using Roots
f(r; a = 0.0) = first(possibilistic_contour([r], [0.0], [a], [y X], Z)) - 0.05
println(
    "95%-Interval for α = 0: " *
    string(round.([find_zero(f, 0.1), find_zero(f, 0.2)], digits = 2))
)
println(
    "95%-Interval for A = [0.0, 0.02]: " *
    string(round.([find_zero(r -> f(r, a = 0.02), 0.0), find_zero(r -> f(r, a = 0.02), 0.2)], digits = 2))
)
println(
    "95%-Interval for A = [0.0, 0.04]: " *
    string(round.([find_zero(r -> f(r, a = 0.04), 0.0), find_zero(r -> f(r, a = 0.04), 0.2)], digits = 2))
)

println("95%-Interval for TSLS: ", round.(tsls(y, X, Z).ci, digits = 2))
println("95%-Interval for PGMM: ", round.(quantile(pgmm(y, X, Z, 0.02 * sqrt(mean(Z.^2))), [0.025, 0.975]), digits = 2))


# Compute upper and lower probabilities
# for the hypothesis β > 0
function compute_upper_lower(lower_α, upper_α, W, Z; type = "Chisq", M = 1000)
    lower, upper = (
        1 - upper_probability(-1e1, 0.0, lower_α, upper_α, W, Z; type = type, M = M),
        upper_probability(0.0, 1e1, lower_α, upper_α, W, Z; type = type, M = M)
        )
    return [lower, upper]
end

probs_res = map(
    b -> compute_upper_lower([0.0], [b], [y X], Z; type = "MC", M = 5000),
    [0.0, 0.01, 0.02, 0.03, 0.04]
) 
probs_res = reduce(hcat, probs_res)'
round.(probs_res; digits = 3)

using LaTeXStrings, RCall
row_names = [L"\{0\}", L"[0, 0.01]", L"[0, 0.02]", L"[0, 0.03]", L"[0, 0.04]"]
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
