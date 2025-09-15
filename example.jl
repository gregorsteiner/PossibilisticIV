using Distributions, LinearAlgebra, Random
using Optim
using Plots

# reduced-form likelihood function (on log scale)
function ll_rf(Γ, Ψ, W, Z)
    ll = -1/2 * tr( Ψ * (W - Z * Γ)' * (W - Z * Γ))
    return ll
end

# reduced-form posterior possibility
function pp_rf(Γ, Ψ, W, Z)
    Γ_ml = inv(Z'Z) * Z'W
    return ll_rf(Γ, Ψ, W, Z) - ll_rf(Γ_ml, Ψ, W, Z)
end


# optimise constrained function
function g(γ_2, α, β, W, Z)
    γ_1 = β * γ_2 + α
    return -pp_rf([γ_1 γ_2], Ψ, W, Z)
end

# compute posterior possibility for the structural parameters
function f_s(α, β, W, Z)
    res = optimize(x -> g(x, α, β, W, Z), α)
    γ_2_min = Optim.minimizer(res)
    γ_1_min = β * γ_2_min + α
    return pp_rf([γ_1_min γ_2_min], Ψ, W, Z)
end

# simulate data
α, β = ([0.0], 0.1)
Σ = [1.0 0.5; 0.5 1.0]

Ψ = inv([1.0 β; 0.0 1.0] * Σ * [1.0 β; 0.0 1.0]')
γ_2 = [1.0]
γ_1 = β * γ_2 + α

n = 100
Random.seed!(42)
Z = rand(n)
W = rand(MatrixNormal(Z[:, :] * [γ_1 γ_2], I(n), inv(Ψ)))


# plot marginal posterior of β for diferent values of α
plot(β -> exp(f_s([0.0], β, W, Z)), label = "α = 0")
plot!(β -> exp(f_s([0.5], β, W, Z)), label = "α = 1/2")
plot!(β -> exp(f_s([-0.5], β, W, Z)), label = "α = -1/2")
xlims!(-1.2, 1.2)

# plot joint posterior possibility of α and β
h(a, b) = f_s([a], b, W, Z)
aa = -1:0.1:1
bb = -1:0.1:1
zz = @. h(aa', bb)

# 3d plot
plot(aa, bb, zz, st = :surface, xlabel = "α", ylabel = "β", zlabel ="Posterior Possibility (log)")

# contour plot
plot(aa, bb, zz, xlabel = "α", ylabel = "β", zlabel ="Posterior Possibility (log)")
