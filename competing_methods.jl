using LinearAlgebra
using gIVBMA

## TSLS ##
function tsls(Y, X, Z; level = 0.05)
    P_Z = Z * inv(Z'Z) * Z'
    β_hat = (X' * P_Z * X) \ (X' * P_Z * Y)

    residuals = Y - X * β_hat
    σ2_hat = sum(residuals.^2) / (size(X, 1) - size(X, 2))
    cov = σ2_hat * inv(X' * P_Z * X)
    ci = β_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov)
    return (beta_hat = β_hat, ci = ci)
end

check_coverage(res, true_value) = res.ci[1] < true_value < res.ci[2]


## Check coverage for distribution objects
function check_coverage(d::Distribution, true_value; level = 0.05)
    ci = quantile(d, [level/2, 1 - level/2])
    return ci[1] < true_value < ci[2]
end


## Plausible GMM (PGMM) by Chernozhukov et al (2025)
function pgmm(Y, X, Z, Λ)
    n = length(Y)

    g(y, x, z, β) = z * (y - x*β)
    β_hat = tsls(Y, X, Z).beta_hat[1]

    G = -mean([Z[i, :] * X[i] for i in eachindex(X)])
    m_hat = mean([g(Y[i], X[i], Z[i], β_hat) for i in eachindex(Y)])
    Ω_hat = mean([(g(Y[i], X[i], Z[i], β_hat) - m_hat) * (g(Y[i], X[i], Z[i], β_hat) - m_hat)' for i in eachindex(Y)])

    A = inv(Ω_hat) - inv(Ω_hat) * inv(inv(Λ) + inv(Ω_hat)) * inv(Ω_hat)
    
    cov = 1 / (n * dot(G, A, G))
    
    return Normal(β_hat, sqrt(cov))
end


## Budget IV (Penn et al, 2025) ##
using RCall

function budgetIV(Y, X, Z, tau, b)
    @rput Y X Z tau b
    R"""
    p = ncol(Z)
    beta_phi = solve(t(Z) %*% Z, t(Z) %*% X)
    beta_y = solve(t(Z) %*% Z, t(Z) %*% Y)

    ssr = sum((Y - Z %*% beta_y)^2)
    cov = (ssr / (length(Y)-1)) * solve(t(Z) %*% Z)
    se = sqrt(diag(cov))

    delta_beta_y = 1.96 * se
    res = budgetIVr::budgetIV_scalar(
        t(beta_y), t(beta_phi),
        b_vec = c(b),
        tau_vec = c(tau),
        delta_beta_y = delta_beta_y
    )
    ci = c(res$lower_bound, res$upper_bound)
    """
    @rget ci
    return (ci = ci, tau = tau)
end

## The confidence interval IV method (CIIV) by Windmeijer et al (2021) ##
function ciiv(Y, X, Z)
    p = size(Z, 2)
    @rput Y X Z
    R"""
    library(sandwich)
    res = CIIV::CIIV(Y, X, Z)
    ci = unname(res$ci_CIM)
    """
    @rget ci
    return (ci, p = p)
end

## Leaky IV by Watson et al (2024)
function leaky_iv(Y, X, Z, tau; level = 0.05, B = 10)
    p = size(Z, 2)
    @rput Y X Z tau B
    R"""
    library(leakyIV)
    res = leakyIV::leakyIV(cbind(Y, X, Z), tau = tau, n_boot = B, parallel = FALSE, method = "shrink")
    """
    @rget res
    
    # Construct confidence interval from bootstrap bounds (Theorem 4 in the LeakyIV paper)
    # Calculate indices
    l = max(1, ceil(Int, (B + 1) * (level / 2)))
    u = min(B, ceil(Int, (B + 1) * (1 - level / 2)))
    
    # Get l-th and u-th smallest values from bootstrap distributions
    lower_bounds = sort(res[:, 1])
    upper_bounds = sort(res[:, 2])
    
    qˆl = lower_bounds[l]
    qˆu = upper_bounds[u]
    
    return (ci = [qˆl, qˆu], tau = tau)
end

