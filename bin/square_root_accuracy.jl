using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Distributions, Plots, LaTeXStrings
include("../src/normal_schur.jl")
include("../src/normal_jacobi.jl")

@views function create_Q_with_real(θs::AbstractVector, λs::AbstractVector, κ::AbstractVector)
    p = length(θs) 
    r = length(λs)
    n = 2p + r
    QR = qr(randn(n, n))
    Q = Matrix(QR.Q) * Diagonal(sign.(diag(QR.R)))
    M = similar(Q, n, n)
    for (i, θ) ∈  enumerate(θs)
        c = κ[i] * cos(θ); s = κ[i] * sin(θ)
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    for (i, λ) ∈ enumerate(λs)
        M[:, 2p + i] .= (λ .* Q[:, 2p+i])
    end
    return M * Q'
end


#Test 1
print("Test 1 \n \n")
n = 100
r = 0
p = (n - r) ÷ 2
vc = π * sqrt(eps(Float64)) .* (1 .+ randn(p))
κ = 2 .* rand(p)
vr = randn(r)
A = create_Q_with_real(vc, vr, κ)

print("Is A normal ? ", norm(A * A' - A' * A),"\n")
T, V = nrmschur(A)
print("Original accuracy: ", norm(A * V - V * T) / norm(A), "\n")
M = copy(V'*A*V)
print("Is V'AV normal ? ", norm(M * M' - M' * M),"\n")
T = normal_jacobi!(M, V, 1)
print("Corrected accuracy: ", norm(A * V - V * T) / norm(A), "\n \n")

#Test 2
print("Test 2 \n \n")
n = 100
E = randn(n, n)
E = E + E'
E ./= opnorm(E)
ε = eps(Float64)
δ = sqrt(ε)
C = cos(δ .* E)
S = sin(δ .* E)
A = [S -C; C S] 

print("Is A normal ? ", norm(A * A' - A' * A),"\n")
T, V = nrmschur(A, :H, false,  0)
print("Original accuracy: ", norm(A * V - V * T) / norm(A), "\n")
M = copy(V'*A*V)
print("Is V'AV normal ? ", norm(M * M' - M' * M),"\n")
T = normal_jacobi!(M, V, 10)
print("Corrected accuracy: ", norm(A * V - V * T) / norm(A), "\n")


