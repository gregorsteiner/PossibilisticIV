using CSV, DataFrames
using Plots, LaTeXStrings


include("PossibilisticIV.jl")

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
Z'Z \ Z'y .- Z'Z \ Z'X * [0.05, 0.2]


# Run analysis
xx = -0.05:0.001:0.35
p = plot(
    xx, possibilistic_contour(xx, [-0.0], [0.0], [y X], Z),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = L"\pi_w(\beta \mid A)",
    label = L"A = \{0\}",
    legend = true
)
plot!(
    xx, possibilistic_contour(xx, [-0.002], [0.012], [y X], Z),
    linewidth = 1.5, label = L"A = [-0.002, 0.012]"
)
plot!(
    xx, possibilistic_contour(xx, [-0.004], [0.024], [y X], Z),
    linewidth = 1.5, label = L"A = [-0.004, 0.024]"
)

xx_exact = -0.05:0.005:0.35
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.0], [0.0], [y X], Z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 1, label = ""
)
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.002], [0.012], [y X], Z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 2, label = ""
)
plot!(
    xx_exact, possibilistic_contour(xx_exact, [-0.004], [0.024], [y X], Z; type = "MC"),
    linewidth = 1.5, linestyle = :dash, colour = 3, label = ""
)

hline!([0.05], linestyle = :dash, label = "", colour = :grey)
p = plot(p, size=(600, 300), legend = :outerbottom, legend_column = 3, bottom_margin=-4Plots.mm)
savefig(p, "Schooling_Possibility_Contour.pdf")

