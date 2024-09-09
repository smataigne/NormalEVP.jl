using LinearAlgebra, SkewLinearAlgebra
include("../src/normal_schur.jl")
#This file contains a routine to obtain the skew-symmetric logarithm of a real orthogonal matrix with determinant 1.
#This method is notably described in 
# R. Zimmermann and K. Hüper, Computing the Riemannian logarithm on the Stiefel manifold: Metrics, methods, and performance, SIAM Journal on Matrix Analysis and Applications, 43 (2022), pp. 953–980.

@views function myangle(c::Real, s::Real)
    """
    Computes the angle of a Givens rotation.
    Input: cosine c and sine s of an angle θ
    Output: θ ∈ (-π, π]
    """
    if s > 1
        #Numerical errors may induce s = 1 + ϵ. In such case, set s = 1 and throw a warning.
        @warn "WARNING: sine expected < 1. sine was set to 1"
        s = 1
    end

    if s < -1
        @warn "WARNING: sine expected > -1. sine was set to -1"
        s = -1
    end

    if c < 0
        return (s < 0 ? -π - asin(s) : π - asin(s))
    else
        return asin(s)
    end
end

@views function multiplybylog(V::AbstractMatrix{T}, S::AbstractMatrix{T}) where T
    """
    Performs a sparse matrix-matrix multiplication between V and log(S).
    Input: Square matrix V and block diagonal matrix S from a Real Schur form.
    Output: V * log(S) where log(S) is the principal logarithm of S.
    """
    n = size(V, 1)
    M = zeros(T, n, n)
    i = 1; mem = 0 #mem allows to treat the -1 eigenvalues
    while i <= n
        if i < n && !iszero(S[i+1, i])
            θ = myangle(S[i,i], S[i+1, i])
            M[:, i]   .=  θ * V[:, i+1]
            M[:, i+1] .= -θ * V[:, i]
            i += 2
        elseif 1 + S[i,i] < eps(T)
            if !iszero(mem)
                M[:, i]   .=   π * V[:, mem]
                M[:, mem] .= - π * V[:, i]
                mem = 0
            else
                mem = i
            end
            i += 1
        else
            i += 1
        end
    end
    return M
end

@views function skewlog(Q::AbstractMatrix{T}) where T
    """
    Computes the real skew-symmetric logarithm of a real orthogonal matrix Q.
    Input: Real Orthogonal matrix Q.
    Output: Real Skew-symmetric logarithm of Q.
    """
    S = schur(Q)
    #The log of S.T is sparse, multpiplybylog takes advantage of the sparsity
    return skewhermitian!(multiplybylog(S.Z, S.T) * S.Z')
end

@views function myskewlog(Q::AbstractMatrix{T}) where T
    """
    Computes the real skew-symmetric logarithm of a real orthogonal matrix Q.
    Input: Real Orthogonal matrix Q.
    Output: Real Skew-symmetric logarithm of Q.
    """
    ε = 10 * eps(T)
    S, V = schurQ(Q, :H, false, ε)
    #The log of S.T is sparse, multpiplybylog takes advantage of the sparsity
    return skewhermitian!(multiplybylog(V, Matrix(S)) * V')
end