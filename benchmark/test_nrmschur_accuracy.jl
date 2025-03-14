using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Distributions
include("../src/normal_schur.jl")

"""
This file evaluates the accuracy of nrmschur(A::AbstractMatrix).
"""

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

BLAS.set_num_threads(1)
data = zeros(5, 3)
N = 100
@views(for (i, n) ∈ enumerate([10, 32, 100, 316, 1000])
    global data
    for k ∈ 1:N
        
        #Test 1
        r = 0
        p = (n - r) ÷ 2
        vc = π/4 .* rand(p)
        κ = ones(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
    
        #=
        #Test 2
        r = 0
        p = (n - r) ÷ 2
        vc = π .* rand(p)
        κ = 2 .* rand(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
        #=
        #Test 3
        r = floor(Int, n * 0.2)
        p = (n - r) ÷ 2
        vc = π .* rand(p)
        κ = 2 .* rand(p)
        vr = rand(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
        #=
        #Test 4
        r = 0
        m = floor(Int, n * 0.1)
        p = (n - 2m) ÷ 2
        vc = [π .* rand(p); (π* rand()) .* ones(m)]
        κ = ones(p + m)
        vr = rand(r)
        Q = create_Q_with_real(vc, vr, κ)
        
        #Test 5
        r = 0
        p = (n - r) ÷ 2
        vc = π * sqrt(eps(Float64)) .* (1 .+ randn(p))
        κ = 2 .* rand(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
        T, V = nrmschur(Q)
        data[i, 1] += log2(norm(Q * V - V * T ) / norm(Q))
        data[i, 2] += log2(norm(V'V - I) / sqrt(n))
        if iszero(r)
            data[i, 3] += log2(norm(sort!(diag(T), rev = true) .- sort!([κ .* cos.(vc); κ .* cos.(vc)], rev = true)) / (1 + norm(diag(T))))
        else
            data[i, 3] += log2(norm(sort!(diag(T), rev = true) .- sort!([κ .* cos.(vc);κ .* cos.(vc); vr], rev = true)) / (1 + norm([κ .* cos.(vc); κ .* cos.(vc); vr])))
        end
    end
    data[i, :] ./= N  
end)
data[:, 1] .= 2 .^(data[:, 1])
data[:, 2] .= 2 .^(data[:, 2])
data[:, 3] .= 2 .^(data[:, 3])
display(data)

