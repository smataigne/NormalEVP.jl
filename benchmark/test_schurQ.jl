using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Distributions
include("../src/normal_schur.jl")

@views function create_Q(θ::AbstractVector, odd::Bool)
    N = length(θ) 
    n = (odd ? 2N + 1 : 2N)
    Q = Matrix(qr(randn(n, n)).Q)
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

e = 100 * √eps(Float64)
BLAS.set_num_threads(1)
data = zeros(5, 3)
N = 10
for (i, n) ∈ enumerate([10, 33, 100, 333, 1000])
    global data
    for k ∈ 1:N
        Q = create_Q(rand(Uniform(π/2-e, π/2+e), n ÷ 2), isodd(n))
        T, V = schurQ(Q)
        S = schur(Q)
        data[i, 1] += norm(Q * V - V * T ) / norm(Q)
        data[i, 2] += norm(V'V - I) / √n
        data[i, 3] += norm(sort!(diag(T), rev = true)[1:2:end] .- sort!(real.(S.values), rev = true)[1:2:end]) / norm(diag(T)[1:2:end])
    end
    data[i, :] /= N  
end
display(data)

