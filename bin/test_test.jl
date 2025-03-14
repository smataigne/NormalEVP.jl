using LinearAlgebra, BenchmarkTools, Plots, LaTeXStrings
include("../src/wxeigen.jl")

#=
BLAS.set_num_threads(1)
m = 10
W = randn(m, m); W = (W + W')/2
X = randn(m, m); X = (X - X')/2
A = [W -X; X W]
Y = Symmetric(A)
E = wxeigen(A, :Lfull)
M = eigen(A)
display(norm(A * E.vectors - E.vectors * Diagonal(E.values)) / norm(A))
Q = E.vectors
J = [zeros(m, m) -Matrix(1.0I, m, m);Matrix(1.0I, m, m) zeros(m, m)]
display(norm(Q'Q - I) / √(2m))
display(norm(Q' * J * Q - J)  / √(2m))
=#
function max_sep(v, t)
    Σ = 0
    n = size(v, 1)
    p = size(t, 1)
    for i ∈ 1:n
        for j ∈ 1:p
            Σ = max(Σ, abs(v[i]-t[j]))#/(v[i] + v[j]))
        end
    end
    return Σ
end
function avg_sep(v, t)
    Σ = 0
    n = size(v, 1)
    p = size(t, 1)
    for i ∈ 1:n
        for j ∈ 1:p
            Σ += abs(v[i]-t[j])#/(v[i] + v[j]))
        end
    end
    return Σ / (n*p)
end
n = 8; p = 4
X = Matrix(qr(randn(n,p)).Q)[:, 1:p]
D1 = Diagonal(randn(n))
D2 = Diagonal(randn(p))
display(opnorm(D1*X-X*D2))
display(max_sep(diag(D1), diag(D2)))
display(avg_sep(diag(D1), diag(D2)))