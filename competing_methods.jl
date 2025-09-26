using LinearAlgebra


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


## Check coverage for gIVBMA
using gIVBMA

function check_coverage(res::gIVBMA.GIVBMA, true_value; level = 0.05)
    ci = quantile(rbw(res)[1], [level/2, 1 - level/2])
    return ci[1] < true_value < ci[2]
end
