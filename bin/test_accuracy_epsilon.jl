using LinearAlgebra, SkewLinearAlgebra, Distributions
include("normal_schur.jl")

@views function create_Q(θ::AbstractVector, odd::Bool)
    N = length(θ) 
    n = (odd ? 2N + 1 : 2N)
    QR = qr(randn(n, n))
    Q = Matrix(QR.Q) * Diagonal(sign.(diag(QR.R)))
    M = similar(Q, n, n)
    for i ∈ 1:N
        κ = rand(Uniform(0, 2))
        c = κ *  cos(θ[i]); s = κ *  sin(θ[i])
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    if odd
        M[:, end] .= Q[:, end]
    end
    return M * Q'
end

ε = 1#1000000000 * eps(Float64)
n = 100
Θ = rand(Uniform(0, π/4), n ÷ 2) #rand(Uniform(π / 2 - √ε, π / 2 + √ε), n ÷ 2)
A = create_Q(Θ, isodd(n))  #Worst accuracy scenario
Ω = (A - A') / 2
S = schur(Ω)
V = S.Z
display(diag(V'*A*V))
display(eigvals(A))
display(norm(sort(diag(V'*A*V)) - sort(real.(eigvals(A))))/ norm(real.(eigvals(A))))
display(norm(real.(eigvals(A))))
M = V'*A*V
T= Tridiagonal(diag(M[2:n, 1:n-1]), diag(M), diag(M[1:n-1, 2:n]))
display(norm(A * V - V * T)/ norm(A))
display(norm(M - T)/norm(M))







