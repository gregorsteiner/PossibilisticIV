using Distributions, LinearAlgebra, Random
using Optim


# reduced-form likelihood function (on log scale)
function ll_rf(Γ, Ψ, W, Z)
    ll = -1/2 * tr( inv(Ψ) * (W - Z * Γ)' * (W - Z * Γ))
    return ll
end


# analytic posterior possibility (marginalising over the covariance)
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



## conditional possibility of β ##
function f_β_given_α(β, lower, upper, W, Z) 
    f(a) = -f_str([a], β, W, Z)
    res = optimize(f, lower, upper, Brent())
    α_opt = Optim.minimizer(res)
    return round(f_str([α_opt], β, W, Z), digits = 10)
end



