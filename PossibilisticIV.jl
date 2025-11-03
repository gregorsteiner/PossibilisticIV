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
# Based on the Wilk's style χ^2 approximation
chi_sq_approximation(x) = 1 - cdf(Chisq(1), -2 * x)

# We also try the more expensive direct sampling
function simulate_W(β, γ_2_ml, Ψ_ml, α_opt, Z)
    n = size(Z, 1)

    Σ = [1.0 -β; 0.0 1.0] * Ψ_ml * [1.0 -β; 0.0 1.0]'

    # We add a small diagonal matrix for numerical stability
    # otherwise the matrix becomes singular for ver large β
    u = rand(MvNormal(zeros(2), Hermitian(Σ + 1e-8 * I)), n)'

    X = Z[:, :] * γ_2_ml + u[:, 2]
    Y = β * X + Z[:, :] * α_opt + u[:, 1]
    return [Y X]
end

function mc_exact(β, lower, upper, W, Z; M = 1000)
    # compute parameters needed
    Γ_ml = Z'Z \ Z'W
    γ_2_ml = Γ_ml[:, 2]
    Ψ_ml = (W - Z * Γ_ml)' * (W - Z * Γ_ml) / size(W, 1)
    α = optimise_α(β, lower, upper, W, Z)

    # compute possibility under the actually observed data
    actual_possibility = conditional_possibility_unnormalised(β, lower, upper, W, Z)

    # initialise storage vector and run loop
    bool_check = Vector{Bool}(undef, M)
    for i in 1:M
        W_new = simulate_W(β, γ_2_ml, Ψ_ml, α, Z)
        new_possibility = conditional_possibility_unnormalised(β, lower, upper, W_new, Z)
        bool_check[i] = new_possibility <= actual_possibility
    end
    return mean(bool_check)
end

# Evaluate possibilistic contour at vector of input values
function possibilistic_contour(β_vec, lower, upper, W, Z; type = "Chisq", M = 1000)
    if type == "Chisq"
        cond_poss_β = conditional_possibility(β_vec, lower, upper, W, Z)
        return map(chi_sq_approximation, cond_poss_β)
    elseif type == "MC"
        res = map(b -> mc_exact(b, lower, upper, W, Z; M = M), β_vec)
        return res
    end
    error("Invalid type argument.")
end

## Upper and lower probabilities
function upper_probability(lower_β, upper_β, lower_α, upper_α, W, Z; type = "Chisq", M = 1000)
    norm_const = normalising_constant(lower_α, upper_α, W, Z)
    f(b) = -conditional_possibility_unnormalised(b, lower_α, upper_α, W, Z)
    res = optimize(f, lower_β, upper_β)
    β_opt = Optim.minimizer(res)
    if type == "Chisq"
        return chi_sq_approximation(-f(β_opt) - norm_const)
    elseif type == "MC"
        return mc_exact(β_opt, lower_α, upper_α, W, Z; M = M)
    end
    error("Invalid type argument.")
end

