using LinearAlgebra, SkewLinearAlgebra
include("chase_zeros.jl")
include("normal_schur.jl")
include("wxeigen.jl")

"""
nrmschur2(A::AbstractMatrix, param::Symbol, check_zeros::Bool, ε::Number)

!!!Prefer `nrmschur`for reliable computations.!!! 

Input: - a normal matrix A from which the real Schur decomposition is desired.\\
        - a param (:H or :L) to decide if skew-symmetric tridiagonalization is performed with Householder reflectors or with Lanczos.\\
        - a boolean check_zeros that specifies if the zeros of teh bidiagonal matrix are isolated or not (true or false).\\
        - a precision ε to decide multiplicity of the singular values (σ₁ ≈ σ₂ ⟺ |σ₁ - σ₂| < ε ⋅ σₘₐₓ)

Output: the tridiagonal Schur form S and the Schur vectors Q.

Description: Computes the real Schur decomposition of the matrix A. nrmschur2 assume that clusters of eigenvalues induce a symmetric skew-Hamiltonian subproblem.
"""
@views function nrmschur2(A::AbstractMatrix{T}, param::Symbol, check_zeros::Bool, ε::Number) where T
    n = size(A, 1)
    n2 = n ÷ 2; n2b = n2 + Int(isodd(n)) 
    Σ = zeros(n2)
    #First memory allocations
    multiples = ones(Int, n2)
    V = similar(A, n, n)
    symA = hermitianpart(A)
    #Compute the Schur decomposition of the skew-symmetric part
    Ω = skewhermitian(A)
    if param == :H
        H = hessenberg(Ω)
        K = Matrix(H.Q) #"K" for "Krylov" basis
        β = H.H.ev
    else
        K, β = skewlanczos(Ω)
    end
    # If n is odd, isolate one zero eigenvalue of the skew-symmetric part (particular interest on SO(n)) 
    #'Bidiagonal' type only admits square matrices so that it is necessary to perform this step.
    if isodd(n)
        Ginit = similar(A, n - 1)
        SkewLinearAlgebra.reducetozero(β, Ginit, n - 1)
        update_vectors!(K[:, 1:2:n], Ginit, n - 1)
        V[:, n2b]  .= K[:, n]
    end
    even_odd_perm!(K)
    β[2:2:end] .*= (-1)               #Abstract even-odd permutation of β to bidiagonal form.
    if check_zeros
        chase_zeros!(β, K)            #Chase all zero eigenvalues
        lz, nz = find_zeros(β)
        for i ∈ 0:nz                  #Find the SVD of each block
            if i == 0      #Block before first zero
                js = 1; je = (nz > 0 ? lz[1] - 2 : length(β) - Int(isodd(n)))
            elseif i == nz #Block after last zero
                js = lz[i] + 2; je = length(β) - Int(isodd(n)) 
            else
                js = lz[i] + 2; je = lz[i] - 2
            end
            if je > js
                B = Bidiagonal(Vector(β[js:2:je]), Vector(β[js+1:2:je]), :U)
                SVD = svd!(B)
                ks = (js + 1) ÷ 2; ke = (je + 1) ÷ 2
                mul!(V[:, ks:ke], K[:, ks:ke], SVD.Vt', 1, 0)
                mul!(V[:, (n2b+ks):(n2b+ke)], K[:, (n2b+ks):(n2b+ke)], SVD.U, 1, 0)
                Σ[ks:ke] .= SVD.S
            end
        end
        lz2 = (lz.+ 1) .÷ 2
        V[:, lz2] .= K[:, lz2]              #Copy vectors from zero singular values
        V[:, n2b .+ lz2] .= K[:, n2b .+ lz2]
        p = sortperm(Σ, rev = true)         #Sorting Σ puts the zeros at the center of the matrix
        Base.permute!(Σ, copy(p))
        Base.permutecols!!(V[:, 1:n2], copy(p))
        Base.permutecols!!(V[:, n2b+1:end], copy(p))
    else
        l = length(β) - Int(isodd(n))
        B = Bidiagonal(Vector(β[1:2:l]), Vector(β[2:2:l]), :U)
        SVD = svd!(B)
        mul!(V[:, 1:n2], K[:, 1:n2], SVD.Vt', 1, 0)
        mul!(V[:, (n2b+1):end], K[:, (n2b+1):end], SVD.U, 1, 0)
        Σ .= SVD.S
    end
    r, r2 = find_multiplicity!(Σ, multiples, n2, isodd(n), ε)
    complex_real_perm!(V, r2)
    m = n2 - r2
    smax = maximum(multiples[1:m])
    #Second memory allocation
    k = max(2smax, r)
    C = similar(A, m)
    M = similar(A, k, k)
    R = similar(A, n, k)
    temp = similar(A, n, max(m, r))
    
    if isone(smax)
        #No multiplicity of any Λsin(θ) = Σ
        mul!(temp[:, 1:m], symA, V[:, 1:m], 1, 0)
        for i ∈ 1:m
            C[i] = dot(V[:, i], temp[:, i])
        end
    else
        mul!(temp[:, 1:m], symA, V[:, 1:m], 1, 0)
        #Some sines have multiplicity > 1
        j = 1
        while j ≤ m
            ss2 = multiples[j]
            if ss2 > 1
                ss = 2ss2; istart = j ; iend = istart + ss2 - 1
                j = istart + ss2
                indices = vcat(istart:iend, (m + istart):(m + iend))
                mul!(M[1:ss, 1:ss2], V[:, indices]', temp[:, istart:iend], 1, 0)
                M[1:ss2, (ss2+1):ss] .= -M[(ss2+1):ss, 1:ss2]
                M[(ss2+1):ss, (ss2+1):ss] .= M[1:ss2, 1:ss2]
                E = wxeigen!(Symmetric(M[1:ss, 1:ss]), :Lfull)
                R[:, 1:ss] .= V[:, indices]
                mul!(V[:, indices], R[:,1:ss], E.vectors, 1, 0)
                C[istart:(j-1)] .= real.(E.values[1:ss2])
                #Σ[istart:(j-1)] .= abs.(imag.(E.values[1:2:end]))
            else
                C[j] = dot(V[:, j], temp[:, j])
                j += 1
            end
        end
    end
    Λᵣ = 0
    if r > 1
        #Compute real eigenvalues and real eigenvectors
        mul!(temp[:, 1:r], symA, V[:, (n - r + 1):n], 1, 0)
        mul!(M[1:r, 1:r],  V[:, (n - r + 1):n]', temp[:, 1:r], 1, 0)
        Y = Symmetric(M)
        #d&d eigensolver
        #E = Eigen(LinearAlgebra.sorteig!(LAPACK.syevd!('V', Y.uplo, Y.data)..., nothing)...) 
        #MRRR eigensolver
        E = Eigen(LinearAlgebra.sorteig!(LAPACK.syevr!('V', 'A', Y.uplo, Y.data, 0.0, 0.0, 0, 0, -1.0)..., nothing)...) 
        Λᵣ = E.values
        R[:, 1:r] .= V[:, (n - r + 1):n]
        mul!(V[:, (n - r + 1):n], R[:, 1:r], E.vectors, 1, 0)
    elseif isone(r)
        Λᵣ = [dot(V[:, n], A, V[:, n])]
    end
    #Provide results in Tridiagonal Schur form
    d = zeros(n)
    dl = zeros(n - 1)
    d[1:2:(2m)] .= C
    d[2:2:(2m)] .= C
    d[(n - r + 1):n] .= Λᵣ
    dl[1:2:2m] .= Σ[1:m]
    p = zeros(Int, 2m)
    p[1:2:2m] .= 1:m
    p[2:2:2m] .= (m+1):2m
    Base.permutecols!!(V[:, 1:2m], p)
    return Tridiagonal(dl, d, -dl), V
end

"""
nrmschur2(A::AbstractMatrix, param::Symbol, check_zeros::Bool, ε::Number)

Input: - a normal matrix A from which the real Schur decomposition is desired.\\
        - a param (:H or :L) to decide if skew-symmetric tridiagonalization is performed with Householder reflectors or with Lanczos.\\
        - a boolean check_zeros that specifies if the zeros of teh bidiagonal matrix are isolated or not (true or false).\\
        - a precision ε to decide multiplicity of the singular values (σ₁ ≈ σ₂ ⟺ |σ₁ - σ₂| < ε ⋅ σₘₐₓ)

Output: the tridiagonal Schur form S and the Schur vectors Q.

Description: Computes the real Schur decomposition of the matrix A.
"""
nrmschur2(A::AbstractMatrix{T}) where T = nrmschur2(A, :H, false, 10 * eps(T))