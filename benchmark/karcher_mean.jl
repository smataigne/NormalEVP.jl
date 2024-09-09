using LinearAlgebra, SkewLinearAlgebra, Statistics, BenchmarkTools
include("../src/skewlog.jl")
include("../src/normal_schur.jl")

@views function karcher(Q::AbstractArray{T}, mylog::Function) where T
    n = size(Q, 1)
    N = size(Q, 3)
    ε = sqrt(eps(T))
    α = 1 / (N * n)
    Qc = Matrix(qr(dropdims(mean(Q, dims = 3), dims = 3)).Q)
    Log = randn(T, n, n)
    Log .-= Log'
    QtQc = similar(Q, n, n)
    itermax = 100
    if det(Qc) < 0
        Qc[:, end] .*= -1
    end
    i = 0
    while  i < itermax
        Log .= 0
        for j ∈ 1:N
            mul!(QtQc, Q[: , :, j]', Qc, 1, 0)
            Log .-= mylog(QtQc)
        end
        QtQc.= Qc
        mul!(Qc, QtQc, exp(skewhermitian!(Log .* α)), 1, 0)
        i += 1
    end 
    return Qc, norm(Log)
end
BLAS.set_num_threads(1)

N1 = 2
n1 = 10
Q1 = randn(n1, n1, N1)
for k ∈ 1:N1
    Q1[: ,:, k] .= Matrix(qr(Q1[: ,:, k]).Q)
    if det(Q1[: ,:, k]) < 0
        Q1[: , end, k] .*= -1
    end
end
Q1[: ,:, 1] .= Matrix(qr(Q1[: ,:, 1]).Q)
karcher(Q1, myskewlog)
myskewlog(Q1[:, :, 1])

times = zeros(3, 3, 2)
for (i, N) ∈ enumerate([16, 32, 64])
    for (j, n) ∈ enumerate([10, 33, 100])
        Q = randn(n, n, N)
        for k ∈ 1:N
            Q[: ,:, k] .= Matrix(qr(Q[: ,:, k]).Q)
            if det(Q[: ,:, k]) < 0
                Q[: , end, k] .*= -1
            end
        end
        times[i, j, 1] = @belapsed karcher($Q, skewlog) 
        times[i, j, 2] = @belapsed karcher($Q, myskewlog) 
    end
end
display(times)



