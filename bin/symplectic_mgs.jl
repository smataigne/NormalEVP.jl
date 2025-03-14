using LinearAlgebra

@views function symplectic_mgs!(A::AbstractMatrix)
    m = size(A, 1) ÷ 2
    A[:, 1] ./= norm(A[:, 1])
    A[1:m, m + 1] .= -A[(m+1):2m, 1]
    A[(m+1):2m, m + 1] .= A[1:m, 1]
    for i ∈ 2:m
        for j ∈ 1:i-1
            A[:, i] .-= dot(A[:, i], A[:, j]) .* A[:, j] .+ dot(A[:, i], A[:, j + m]) .* A[:, j + m]
        end
        A[:, i] ./= norm(A[:, i])
        A[1:m, m + i] .= -A[(m+1):2m, i]
        A[(m+1):2m, m + i] .= A[1:m, i]
    end
end

m =4
A = randn(2m, 2m)
c = opnorm(A)
J = [zeros(m, m) -Matrix(1.0I, m, m); Matrix(1.0I, m, m) zeros(m, m)]
symplectic_mgs!(A)
display(opnorm(A'A - I)/c)
display(norm(J'A*J - A))

U = randn(6, 1)
V = randn(6, 1)
M = U * V'
H = hessenberg!(M)
display(M)
M = Symmetric(U * U')
H = hessenberg!(M)
display(H.Q.factors)
