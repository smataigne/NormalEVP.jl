using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Distributions
include("../src/normal_schur.jl")
include("../src/normal_jacobi.jl")

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
data_nrmschur = zeros(5, 3)
data_gees = zeros(5, 3)
data_special = zeros(5, 3)
N = 10
special = false
@views(for (i, n) ∈ enumerate([10, 32, 100, 316, 1000])
    global data_nrmschur, data_gees, data_special, special
    for k ∈ 1:N
        #=
        #Experiment E1
        r = 0
        p = (n - r) ÷ 2
        vc = π/4 .* rand(p)
        κ = ones(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
        #=
        #Experiment E2
        r = 0
        p = (n - r) ÷ 2
        vc = π .* rand(p)
        κ = 2 .* rand(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
        #=
        #Experiment E3
        
        r = floor(Int, n * 0.2)
        p = (n - r) ÷ 2
        vc = π .* rand(p)
        κ = 2 .* rand(p)
        vr = rand(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
        #=
        #Experiment E4
        r = 0
        m = floor(Int, n * 0.1)
        p = (n - 2m) ÷ 2
        vc = [π .* rand(p); (π* rand()) .* ones(m)]
        κ = ones(p + m)
        vr = rand(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
        #Experiment E5
        special = true
        r = 0
        p = (n - r) ÷ 2
        vc = π * sqrt(eps(Float64)) .* (1 .+ randn(p))
        κ = 2 .* rand(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)

        T, V = nrmschur(Q)
        data_nrmschur[i, 1] += log2(norm(Q * V - V * T ) / norm(Q))
        data_nrmschur[i, 2] += log2(norm(V'V - I) / sqrt(n))
        if iszero(r)
            data_nrmschur[i, 3] += log2(norm(sort!(diag(T), rev = true) .- sort!([κ .* cos.(vc); κ .* cos.(vc)], rev = true)) / (1 + norm(diag(T))))
        else
            data_nrmschur[i, 3] += log2(norm(sort!(diag(T), rev = true) .- sort!([κ .* cos.(vc);κ .* cos.(vc); vr], rev = true)) / (1 + norm([κ .* cos.(vc); κ .* cos.(vc); vr])))
        end
        S = schur(Q)
        data_gees[i, 1] += log2(norm(Q * S.vectors - S.vectors * S.T ) / norm(Q))
        data_gees[i, 2] += log2(norm(S.vectors'* S.vectors - I) / sqrt(n))
        if iszero(r)
            data_gees[i, 3] += log2(norm(sort!(diag(S.T), rev = true) .- sort!([κ .* cos.(vc); κ .* cos.(vc)], rev = true)) / (1 + norm(diag(S.T))))
        else
            data_gees[i, 3] += log2(norm(sort!(diag(S.T), rev = true) .- sort!([κ .* cos.(vc);κ .* cos.(vc); vr], rev = true)) / (1 + norm([κ .* cos.(vc); κ .* cos.(vc); vr])))
        end
        if special
            M = copy(V' * Q * V)
            T = normal_jacobi!(M, V, 10)
            data_special[i, 1] += log2(norm(Q * V - V * T) / norm(Q))
            data_special[i, 2] += log2(norm(V'V - I) / sqrt(n))
            if iszero(r)
                data_special[i, 3] += log2(norm(sort!(diag(T), rev = true) .- sort!([κ .* cos.(vc); κ .* cos.(vc)], rev = true)) / (1 + norm(diag(T))))
            else
                data_special[i, 3] += log2(norm(sort!(diag(T), rev = true) .- sort!([κ .* cos.(vc);κ .* cos.(vc); vr], rev = true)) / (1 + norm([κ .* cos.(vc); κ .* cos.(vc); vr])))
            end
        end
    end
    data_nrmschur[i, :] ./= N  
    data_gees[i, :] ./= N
    if special
        data_special[i, :] ./= N
    end
end)
data_nrmschur[:, 1] .= 2 .^(data_nrmschur[:, 1])
data_nrmschur[:, 2] .= 2 .^(data_nrmschur[:, 2])
data_nrmschur[:, 3] .= 2 .^(data_nrmschur[:, 3])
data_gees[:, 1] .= 2 .^(data_gees[:, 1])
data_gees[:, 2] .= 2 .^(data_gees[:, 2])
data_gees[:, 3] .= 2 .^(data_gees[:, 3])
if special
    data_special[:, 1] .= 2 .^(data_special[:, 1])
    data_special[:, 2] .= 2 .^(data_special[:, 2])
    data_special[:, 3] .= 2 .^(data_special[:, 3])
end

display(round.(data_nrmschur; sigdigits=2))
#display(round.(data_gees; sigdigits=2))
datamerge = zeros(5, 6)
datamerge[:,1:2:end] = data_nrmschur
datamerge[:,2:2:end] = data_gees
display(round.(datamerge; sigdigits=2))

datamerge2 = zeros(5, 7)
datamerge2[:, 1] .= [10, 32, 100, 316, 1000]
datamerge2[:, 2:end] = round.(datamerge; sigdigits=2)
for i ∈ 1:size(datamerge2, 1)
    println(join(datamerge2[i, :], " & ") * " \\\\")
end