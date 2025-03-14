using LinearAlgebra

"""
```wxlanczos!(A::Symmetric{T})```

Implementation of Lanczos algorithm for A = [W -X; X W] with W symmetric and X is skew-symmetric.\\
No re-orthogonalization is performed. The algorithm is unstable.
A is assumed full-rank with eignvalues of multiplicity 2 exactly.\\
Only n/2 iterations are performed and the orthogonalized (Krylov) basis is then extended using the symplectic structure of the eigenvectors.\\
This procedure fully exploits the structure of A.

Input: A symmetric matrix A = [W -X; X W] with W symmetric and X is skew-symmetric.

Output: - A symplectic orthogonal basis K that tridiagonalizes A. (K stands for "Krylov")\\
        - The tridiagonal form T (A = K * T * K')
"""
@views function wxlanczos!(A::Symmetric{T}) where T
    n = size(A, 1); n2 =  n ÷ 2
    K = zeros(T, n, n)
    mul!(K[:, 1], A, randn(n), 1, 0)
    K[:, 1] ./= norm(K[:, 1])
    β = zeros(T, n - 1) 
    α = zeros(T, n)
    @inbounds(for i ∈ 1:n2-1  #Only half Lanczos iterations are needed
        mul!(K[:, i + 1], A, K[:, i], 1, 0)
        
        α[i] = dot(K[:, i], K[:, i + 1])
        if i > 1
            K[:, i + 1] .-= β[i - 1] * K[:, i - 1]
        end 
        K[:, i + 1] .-= α[i] * K[:, i]
        β[i] = norm(K[:, i + 1])
        K[:, i + 1] ./= β[i]
    end)
    α[n2] = dot(K[:, n2], A,  K[:, n2])
    #Use symplectic structure to extrapolate n/2 last iterations
    K[1:n2, (n2+1):n] .= -K[(n2+1):n, 1:n2] 
    K[(n2+1):n, (n2+1):n] .= K[1:n2, 1:n2] 
    α[(n2+1):n] .= α[1:n2]
    β[(n2+1):(n-1)] .= β[1:(n2-1)]
    return K, SymTridiagonal(α, β)
end

wxlanczos(A::Symmetric)  = wxlanczos!(copy(A)) 

"""
```wxlanczosfull!(A::Symmetric{T})```

Implementation of Lanczos algorithm for A = [W -X; X W] with W symmetric and X is skew-symmetric.\\
A full re-orthogonalization is performed. The algorithm is stable.
A is assumed full-rank with eignvalues of multiplicity 2 exactly.\\
Only n/2 iterations are performed and the orthogonalized (Krylov) basis is then extended using the symplectic structure of the eigenvectors.\\
This procedure fully exploits the structure of A.

Input: A symmetric matrix A = [W -X; X W] with W symmetric and X is skew-symmetric.

Output: - A symplectic orthogonal basis K that tridiagonalizes A. (K stands for "Krylov")\\
        - The tridiagonal form T (A = K * T * K')
"""
@views function wxlanczosfull!(A::Symmetric{T}) where T
    n = size(A, 1); n2 =  n ÷ 2
    K = zeros(T, n, n)
    mul!(K[:, 1], A, randn(n), 1, 0)
    K[:, 1] ./= norm(K[:, 1])
    β = zeros(T, n - 1) 
    α = zeros(T, n)
    @inbounds(for i ∈ 1:n2-1  #Only half Lanczos iterations are needed
        #### Use symplectic structure to extrapolate n/2 last iterations
        K[1:n2, n2 + i] .= -K[(n2+1):n, i] 
        K[(n2+1):n, n2 + i] .= K[1:n2, i] 

        mul!(K[:, i + 1], A, K[:, i], 1, 0)
        K[:, i + 1] .-= dot(K[:, i + 1] , K[:, i + n2]) .* K[:, i +n2]
        α[i] = dot(K[:, i], K[:, i + 1])
        if i > 1
            @. K[:, i + 1] -= α[i] * K[:, i] + β[i-1] * K[:, i - 1]
            K[:, i + 1] .-= dot(K[:, i + 1] , K[:, i + n2]) .* K[:, i + n2]
            K[:, i + 1] .-= dot(K[:, i + 1] , K[:, i - 1 + n2]) .* K[:, i - 1 + n2]
        else
            @. K[:, i + 1] -= α[i] * K[:, i]
            K[:, i + 1] .-= dot(K[:, i + 1] , K[:, i + n2]) .* K[:, i + n2]
        end
        ### Full re-orthogonalization
        for j ∈ 1:i
            K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j]) .* K[:, j]
            K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j + n2]) .* K[:, j + n2]
        end
        #=
        K[:, i + 1] .-= dot(K[:, i + 1] , K[:, i + n2]) .* K[:, i +n2]
        α[i] = dot(K[:, i], K[:, i + 1])
        K[:, i + 1] .-= α[i] .* K[:, i]
        =#
        β[i] = norm(K[:, i + 1])
        K[:, i + 1] ./= β[i]
    end)
    K[1:n2, n] .= -K[(n2+1):n, n2] 
    K[(n2+1):n, n] .= K[1:n2, n2] 
    α[n2] = dot(K[:, n2], A,  K[:, n2])
    α[(n2+1):n] .= α[1:n2]
    β[(n2+1):(n-1)] .= β[1:(n2-1)]
    return K, SymTridiagonal(α, β)
end

wxlanczosfull(A::Symmetric)  = wxlanczosfull!(copy(A)) 

"""
```wxhessenberg!(A::AbstractMatrix{T})```

"Dummy" implementation of Householder tridiagonalization for A = [W -X; X W] with W symmetric and X is skew-symmetric.\\
A is assumed full-rank with eignvalues of multiplicity 2 exactly.\\
Only n/2 - 1 iterations are performed and the orthogonal basis is then extended using the symplectic structure of the eigenvectors.\\
This procedure fully exploits the structure of A.

Input: A symmetric matrix A = [W -X; X W] with W symmetric and X is skew-symmetric.

Output: - A symplectic orthogonal basis Q that tridiagonalizes A.\\
        - The tridiagonal form T (A = Q * T * Q')
"""
@views function wxhessenberg!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    x = zeros(T, n)
    u = zeros(T, n - 1)
    t = zeros(T, n)
    α = zeros(T, n)
    β = zeros(T, n - 1)
    Q = Matrix(one(T) * I, n, n)
    n2 = n ÷ 2
    @inbounds(for i ∈ 1:n2-1
        #Build Householder reflector
        v = x[1:n-i]
        v .= A[(i+1):n, i]
        β[i] = - sign(v[1]) * norm(v);
        v[1] -= β[i]
        v ./= norm(v)
        #Apply two sided reflection
        mul!(u[1:n-i], A[(i+1):n, (i+1):n], v, 2, 0)
        u[1:n-i] .-= dot(v, u[1:n-i]) * v
        BLAS.ger!(-1., v, u[1:n-i], A[(i+1):n, (i+1):n])  #"Symmetric oblivious" version of syr2k!
        BLAS.ger!(-1., u[1:n-i], v, A[(i+1):n, (i+1):n])
        #Assemble Householder reflectors
        mul!(t, Q[:, (i+1):end], v, 1, 0 )
        BLAS.ger!(-2. , t, v, Q[:, (i+1):end])
    end)
    Q[1:n2, (n2+1):n] .= -Q[(n2+1):n, 1:n2] 
    Q[(n2+1):n, (n2+1):n] .= Q[1:n2, 1:n2]
    α[1:n2] = diag(A[1:n2, 1:n2])
    α[(n2+1):n] .= α[1:n2]
    β[(n2+1):(n-1)] .= β[1:(n2-1)]
    return Q, SymTridiagonal(α, β)
end

wxhessenberg(A::AbstractMatrix) = wxhessenberg!(copy(A))

"""
```wxeigen!(A::AbstractMatrix{T}, method::Symbol)```

Eigenvalue decomposition of A = [W -X; X W] with W symmetric and X is skew-symmetric.\\
A is assumed full-rank with eignvalues of multiplicity 2 exactly.\\

Input: - A symmetric matrix A = [W -X; X W] with W symmetric and X is skew-symmetric.\\
       - method = :L for Lanczos tridiag. and :H for Householder tridiagonalization.

Output: The eigenvalue decomposition of A using Eigen structure. 
"""
@views function wxeigen!(A::AbstractMatrix{T}, method::Symbol) where T
    n = size(A, 1)
    n2 = n ÷ 2
    Λ = similar(A, n)
    if method == :Lfull #Full re-orthogonalization, stable
        Q, Tr = wxlanczosfull!(Symmetric(A))
    elseif method ==:L #No re-orthogonalization, unstable
        Q, Tr = wxlanczos!(Symmetric(A))
    else
        Q, Tr = wxhessenberg!(A)
    end
    T₂ = SymTridiagonal(Tr.dv[1:n2], Tr.ev[1:n2-1])
    E = eigen(T₂)
    mul!(Q[:, (n2+1):n], Q[:, 1:n2], E.vectors, 1, 0)
    Q[:, 1:n2] .= Q[:, (n2+1):n]
    Q[1:n2, (n2+1):n] .= - Q[(n2+1):n, 1:n2] 
    Q[(n2+1):n, (n2+1):n] .= Q[1:n2, 1:n2]
    Λ[1:n2] .= E.values ; Λ[(n2+1):n] .= E.values
    return Eigen(Λ, Q)
end

wxeigen(A::AbstractMatrix, method::Symbol) = wxeigen!(copy(A), method)
wxeigen(A::AbstractMatrix) = wxeigen!(copy(A), :Lfull)
