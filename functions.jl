using Distributions, LinearAlgebra

# helper function to extract variances
function variances(Σ)
    Σ_xx = Σ[2:end, 2:end]
    Σ_yx = Σ[2:end, 1]
    σ_y_x = Σ[1,1] - Σ_yx' * inv(Σ_xx) * Σ_yx

    return (σ_y_x, Σ_yx, Σ_xx)
end

# likelihood function (on log scale)
function ll(y, X, Z, β, Δ, Σ)
    n = length(y)
    σ_y_x, Σ_yx, Σ_xx = variances(Σ)

    outcome = logpdf(MvNormal(X * β + (X - Z * Δ) * inv(Σ_xx) * Σ_yx, σ_y_x * I), y)
    treatment = logpdf(MatrixNormal(Z * Δ, I(n), Σ_xx), X)

    return outcome + treatment
end
