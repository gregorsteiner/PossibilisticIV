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
    Γ_ml = Z'Z \ Z'W
    # Plug in ML estimate for Ψ
    # We could also explicitly model Σ and then Ψ is deterministic given Σ and β
    Ψ_ml = (W - Z * Γ_ml)' * (W - Z * Γ_ml) / size(W, 1) 
    
    # Compute optimal Γ given the constraint
    σ11 = dot([1.0 -β], Ψ_ml, [1.0 ; -β])
    Γ = Γ_ml + (1/σ11) * (α .- Γ_ml * [1.0; -β]) * [1.0 -β] * Ψ_ml
    #Ψ = (W - Z * Γ)' * (W - Z * Γ) / size(W, 1) 

    # Return relative likelihood at this point (in logs)
    return ll_rf(Γ, Ψ_ml, W, Z) - ll_rf(Γ_ml, Ψ_ml, W, Z)
end



## conditional possibility of β ##
using JuMP, OSQP, Optim 

# perform constrained optimisation over α
function optimise_α(β, lower, upper, W, Z)
    p = size(Z, 2)
    Γ_ml = Z'Z \ Z'W
    t = p == 1 ? [Γ_ml * [1.0; -β]] : Γ_ml * [1.0; -β] # make sure t is a vector even for p=1

    # If t is in the constraint set return t
    # Else use quadratic programming to find optimal α
    if all(lower .< t .< upper)
        return t
    else
        model = Model(OSQP.Optimizer)
        set_silent(model) # suppress any output
        @variable(model, lower[i] <= α[i=1:p] <= upper[i])
        @objective(model, Min, dot(α .- t, Z'Z, α .- t))
        optimize!(model)
        return value(α)
    end
end

# unnormalised possibilistic conditional posterior (on log-scale)
function conditional_possibility_unnormalised(β, lower, upper, W, Z) 
    α_opt = optimise_α(β, lower, upper, W, Z)
    return f_str(α_opt, β, W, Z)
end

# optimise β given α
# we use this function to iteratively optimise across both α and β
function optimise_β(α, W, Z)
    h(β) = -f_str(α, β, W, Z)
    res = optimize(x -> h(first(x)), [1.0])
    return first(Optim.minimizer(res))
end

# find normalising constant
# by iterating across α and β
function normalising_constant(lower, upper, W, Z; tol=1e-6, max_iter=100)
    # Initialize β and α
    β = 0.0
    α = optimise_α(β, lower, upper, W, Z)

    for iter in 1:max_iter
        β_prev, α_prev = β, copy(α)

        # Optimize β given α
        β = optimise_β(α, W, Z)

        # Optimize α given updated β
        α = optimise_α(β, lower, upper, W, Z)

        # Check convergence on both α and β
        if norm(α - α_prev) < tol && abs(β - β_prev) < tol
            break
        end
    end

    # Return f_str evaluated at the optimal (α, β)
    return f_str(α, β, W, Z)
end


# conditional possibilistic posterior (normalised on log-scale)
function conditional_possibility(β_vec, lower, upper, W, Z)
    norm_const = normalising_constant(lower, upper, W, Z)
    cond_poss_β = [conditional_possibility_unnormalised(β, lower, upper, W, Z) - norm_const for β in β_vec]
    return cond_poss_β
end


## Validification (Martin, 2025)
## We use the Wilk's style approximation
chi_sq_approximation(x) = 1 - cdf(Chisq(1), -2 * x)
function possibilistic_contour(β_vec, lower, upper, W, Z)
    cond_poss_β = conditional_possibility(β_vec, lower, upper, W, Z)
    return map(chi_sq_approximation, cond_poss_β)
end



## Upper and lower probabilities
function upper_probability(lower_β, upper_β, lower_α, upper_α, W, Z)
    norm_const = normalising_constant(lower_α, upper_α, W, Z)
    f(b) = -conditional_possibility_unnormalised(b, lower_α, upper_α, W, Z)
    res = optimize(f, lower_β, upper_β)
    β_opt = Optim.minimizer(res)
    return chi_sq_approximation(-f(β_opt) - norm_const)
end

