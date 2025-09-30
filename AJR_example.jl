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

# raw possibilistic posterior
posterior(β, lower, upper) = exp(f_β_given_α(β, [lower], [upper], [y x], z))

# possibilistic contour
pi_w(β, lower, upper) = possibilistic_contour(β, [lower], [upper], [y x], z)


# Create plot
xx = -3:0.005:6
p = plot(
    xx, pi_w.(xx, 0.0, 0.0),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = "",
    label = L"\alpha = 0"
)
plot!(xx, pi_w.(xx, -1/10, 1/10), linewidth = 1.5, label = L"\alpha \in [-0.1, 0.1]")
plot!(xx, posterior.(xx, 0.0, 0.0), linestyle = :dash, linewidth = 1.5, color = 1, label = "")
plot!(xx, posterior.(xx, -1/10, 1/10), linestyle = :dash, linewidth = 1.5, color = 2, label = "")

hline!([0.1], linestyle = :dash, label = "", colour = :grey)

savefig(p, "AJR_Possibility_Contour.pdf")




