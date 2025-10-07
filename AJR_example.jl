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
xx = -1:0.005:5
p = plot(
    xx, possibilistic_contour(xx, [0.0], [0.0], [y x], z; x0 = [1.0]),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = "",
    label = L"\alpha = 0",
    size=(600, 300)
)
plot!(xx, possibilistic_contour(xx, [-0.1], [0.1], [y x], z; x0 = [1.0]), linewidth = 1.5, label = L"\alpha \in [-0.1, 0.1]")
plot!(xx, possibilistic_contour(xx, [-0.18], [0.18], [y x], z; x0 = [1.0]), linewidth = 1.5, label = L"\alpha \in [-0.18, 0.18]")
plot!(xx, exp.(conditional_possibility(xx, [0.0], [0.0], [y x], z; x0 = [1.0])), linestyle = :dash, linewidth = 1.5, color = 1, label = "")
plot!(xx, exp.(conditional_possibility(xx, [-0.1], [0.1], [y x], z; x0 = [1.0])), linestyle = :dash, linewidth = 1.5, color = 2, label = "")
plot!(xx, exp.(conditional_possibility(xx, [-0.18], [0.18], [y x], z; x0 = [1.0])), linestyle = :dash, linewidth = 1.5, color = 3, label = "")


hline!([0.05], linestyle = :dash, label = "", colour = :grey)

savefig(p, "AJR_Possibility_Contour.pdf")



