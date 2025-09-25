using Distributions, LinearAlgebra, Random
using Optim


# reduced-form likelihood function (on log scale)
function ll_rf(Γ, Ψ, W, Z)
    n = size(W, 1)
    ll = -n/2 * log(det(Ψ)) -1/2 * tr( inv(Ψ) * (W - Z * Γ)' * (W - Z * Γ))
    return ll
end


# analytic posterior possibility (marginalising over the covariance)
function f_str(α, β, W, Z)
    # Compute ML estimates
    Γ_ml = inv(Z'Z) * Z'W
    # Plug in ML estimate for Ψ
    # We could also explicitly model Σ and then Ψ is deterministic given Σ and β
    Ψ_ml = (W - Z * Γ_ml)' * (W - Z * Γ_ml) / size(W, 1) 

    # Compute optimal Γ given the constraint
    σ11 = dot([1.0 -β], Ψ_ml, [1.0 ; -β])
    Γ = Γ_ml + (1/σ11) * (α - [Γ_ml * [1.0; -β]]) * [1.0 -β] * Ψ_ml
    Ψ = (W - Z * Γ)' * (W - Z * Γ) / size(W, 1) 

    # Return relative likelihood at this point (in logs)
    return ll_rf(Γ, Ψ, W, Z) - ll_rf(Γ_ml, Ψ_ml, W, Z)
end



## conditional possibility of β ##
function f_β_given_α(β, lower, upper, W, Z) 
    p = size(Z, 2)

    ## if p = 1 we can use Brent's method which is much faster
    ## for p > 1 we use Gradient based optimisation
    if p == 1
        lower, upper = lower[1], upper[1]
        res = optimize(a -> -f_str([a], β, W, Z), lower, upper, Brent())
        α_opt = [Optim.minimizer(res)]
    else
        function g!(G, α) # define gradient w.r.t. α
            Γ_ml = inv(Z'Z) * Z'W
            G[:] = -Z'Z * (α - [Γ_ml * [1.0; -β]])
        end
        inner_optimizer = BFGS()
        res = optimize(
            a -> -f_str(a, β, W, Z), g!, lower, upper, zeros(p), 
            Fminbox(inner_optimizer), Optim.Options(iterations=20)
        )
        α_opt = Optim.minimizer(res)
    end
    return round(f_str(α_opt, β, W, Z), digits=10)
end



