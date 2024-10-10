using LinearAlgebra, SkewLinearAlgebra, Statistics, BenchmarkTools
include("../src/skewlog.jl")
include("../src/normal_schur.jl")

@views function karcher(Q::AbstractArray{T}, mylog::Function) where T
    n = size(Q, 1)
    N = size(Q, 3)
    itermax = 100
    α = 1 / N
    Log = zeros(T, n, n)
    QtQc = similar(Q, n, n)
    # Initial center of mass as the Q factor of the average of the dataset.
    Qc = Matrix(qr(dropdims(mean(Q, dims = 3), dims = 3)).Q) 
    #Check that Qc is on SO(n)
    if logabsdet(Qc)[2] < 0 #logabsdet has increased accuracy compared to det;
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
    return Qc
end

BLAS.set_num_threads(1)
N1 = 2
n1 = 20
Q1 = randn(n1, n1, N1)
@views for k ∈ 1:N1
    global Q1
    QR = qr(Q1[: ,:, k])
    mul!(Q1[: ,:, k], Matrix(QR.Q), Diagonal(sign.(diag(QR.R))), 1, 0)
    if det(Q1[: ,:, k]) < 0
        Q1[: , end, k] .*= -1
    end
end
l = karcher(Q1, myskewlog)[2]
myskewlog(Q1[:, :, 1])

T = 4
times = 1000 * ones(3, 4, 2)
@views(for (i, N) ∈ enumerate([16, 32, 64])
    for (j, n) ∈ enumerate([25, 50, 100, 200])
        Q = randn(n, n, N)
        for k ∈ 1:N
            QR = qr(Q[: ,:, k])
            mul!(Q[: ,:, k], Matrix(QR.Q), Diagonal(sign.(diag(QR.R))), 1, 0)
            if det(Q[: ,:, k]) < 0
                Q[: , end, k] .*= -1
            end
        end
        for k ∈ 1:T
            times[i, j, 1] = min(@elapsed karcher(Q, skewlog), times[i, j, 1]) 
            times[i, j, 2] = min(@elapsed karcher(Q, myskewlog), times[i, j, 2])
        end
        display(times[i, j , :])
        print("Done: n = "*string(n)*"\n")
    end
    print("Done: N = "*string(N)*"\n")
end)
display(times)




