using Distributions, LinearAlgebra, Random
using Optim
using Plots

# reduced-form likelihood function (on log scale)
function ll_rf(Γ, Ψ, W, Z)
    ll = -1/2 * tr( inv(Ψ) * (W - Z * Γ)' * (W - Z * Γ))
    return ll
end


# analytic posterior possibility
function f_str(α, β, W, Z)
    # Compute ML estimates
    Γ_ml = inv(Z'Z) * Z'W
    # Plug in ML estimate for Ψ
    # We could also explicitly model Σ and then Ψ is determinitic given Σ and β
    Ψ_ml = (W - Z * Γ_ml)' * (W - Z * Γ_ml) / size(W, 1) 

    # Compute optimal Γ given the constraint
    σ11 = dot([1.0 -β], Ψ_ml, [1.0 ; -β])
    Γ = Γ_ml + (1/σ11) * (α - [Γ_ml * [1.0; -β]]) * [1.0 -β] * Ψ_ml

    # Return relative likelihood at this point
    return ll_rf(Γ, Ψ_ml, W, Z) - ll_rf(Γ_ml, Ψ_ml, W, Z)
end

# simulate data
α, β = ([0.0], 1)
Σ = [1.0 0.5; 0.5 1.0]

Ψ = [1.0 β; 0.0 1.0] * Σ * [1.0 β; 0.0 1.0]'
γ_2 = [1.0]
γ_1 = β * γ_2 + α


n = 100
Random.seed!(42)
Z = rand(n)
W = rand(MatrixNormal(Z[:, :] * [γ_1 γ_2], I(n), Ψ))


# plot marginal posterior of β for diferent values of α
plot(β -> exp(f_str([0.0], β, W, Z)), label = "α = 0")
plot!(β -> exp(f_str([0.1], β, W, Z)), label = "α = 0.1")
plot!(β -> exp(f_str([-0.1], β, W, Z)), label = "α = -0.1")
xlims!(0.5, 1.5)


# plot joint posterior possibility of α and β
h(a, b) = f_str([a], b, W, Z)
aa = -1:0.1:1
bb = 0:0.1:2
zz = @. h(aa', bb)

# 3d plot
plot(aa, bb, zz, st = :surface, xlabel = "α", ylabel = "β", zlabel ="Posterior Possibility (log)")

# contour plot
plot(aa, bb, zz, xlabel = "α", ylabel = "β", zlabel ="Posterior Possibility (log)")



