using CSV, DataFrames
using Plots, LaTeXStrings

# import method function
include("PossibilisticIV.jl")

# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y_raw, x_raw, z_raw = (d.GDP, d.Exprop, d.logMort)
W = [ones(length(y_raw)) Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]])]

# project out covariates
M_W = I - W * inv(W'W) * W'
y, x, z = map(vec -> M_W * vec, (y_raw, x_raw, z_raw))

# try out method
posterior(β, bound) = exp(f_β_given_α(β, -bound, bound, [y x], z))
xx = -20:0.1:20
plot(
    xx, posterior.(xx, 0.0),
    xlabel = L"\beta", ylabel = "Posterior Possibility",
    label = L"\alpha = 0"
)
plot!(xx, posterior.(xx, 1/4), label = L"\alpha \in [-1/4, 1/4]")
plot!(xx, posterior.(xx, 1.0), label = L"\alpha \in [-1, 1]")




pi(β, lower, upper) = possibilistic_contour(β, [lower], [upper], [y x], z)
pi(1.0, 0.0, 0.0)

xx = -10:0.01:10
plot(
    xx, pi.(xx, 0.0, 0.0),
    xlabel = L"\beta", ylabel = "Possibilistic Contour",
    label = L"\alpha = 0"
)
plot!(xx, pi.(xx, -1/4, 1/4), label = L"\alpha \in [-1/4, 1/4]")
plot!(xx, pi.(xx, -1.0, 1.0), label = L"\alpha \in [-1, 1]")
