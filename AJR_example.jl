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

# plot raw possibilistic posterior
posterior(β, bound) = exp(f_β_given_α(β, -bound, bound, [y x], z))
xx = -20:0.1:20
plot(
    xx, posterior.(xx, 0.0),
    xlabel = L"\beta", ylabel = "Posterior Possibility",
    label = L"\alpha = 0"
)
plot!(xx, posterior.(xx, 1/10), label = L"\alpha \in [-0.1, 0.1]")



# plot possibilistic contour
pi(β, lower, upper) = possibilistic_contour(β, [lower], [upper], [y x], z)

xx = -4:0.005:8
p = plot(
    xx, pi.(xx, 0.0, 0.0),
    linewidth = 1.5,
    xlabel = L"\beta", ylabel = "Possibilistic Contour",
    label = L"\alpha = 0"
)
plot!(xx, pi.(xx, -1/4, 1/4), linewidth = 1.5, label = L"\alpha \in [-0.1, 0.1]")
hline!([0.1], linestyle = :dash, label = "")

savefig(p, "AJR_Possibility_Contour.pdf")
