using LinearAlgebra, SkewLinearAlgebra
include("chase_zeros.jl")

@views function skewlanczos(A::SkewHermitian)
    n = size(A, 1)
    β = similar(A.data, n - 1)
    Q = similar(A.data, n, n)
    v = randn(n)
    v ./= norm(v)
    Q[:, 1] .= v
    mul!(Q[:, 2], A, Q[:, 1], 1, 0)
    β[1] = norm(Q[:, 2])
    Q[:, 2] ./= β[1]
    for i ∈ 2:n-1
        Q[:, i + 1] .= Q[:, i - 1]
        mul!(Q[:, i+1], A.data, Q[:, i], 1, β[i - 1])
        β[i] = norm(Q[:, i + 1])
        Q[:, i + 1] ./= β[i]
    end
    return Q, β
end

@views function even_odd_perm!(K::AbstractMatrix)
    n = size(K, 1)
    Base.permutecols!!(K, [1:2:n;2:2:n])
end

@views function complex_real_perm!(V::AbstractMatrix, r::Integer, r2::Integer)
    n = size(V, 1)
    n2 = n ÷ 2; odd = Int(isodd(n)) 
    r2b = r2 + odd
    init = n2 - r2
    Base.permutecols!!(V[:, (init + 1):(n - r2)], [(r2b + 1):(n - n2); 1:r2b])
end

@views function update_vectors!(Q::AbstractMatrix, G::AbstractVector, n::Integer)
    nn = size(Q, 2)
    n2 = n ÷ 2
    @inbounds(for i ∈ n2:-1:1
        c = G[2i - 1]
        s = G[2i]
        for j ∈ 1:(n+1)
            σ = Q[j, i]
            ω = Q[j, nn]
            Q[j, i]  =  c * σ + -s * ω
            Q[j, nn] =  s * σ + c * ω
        end
    end)
end



@views function find_multiplicity!(Σ::AbstractVector{T}, multiples::AbstractVector{Int}, n2::Integer, odd::Bool, ε₂::Number) where T
    ε =  10 * eps(T) #Discriminate from singular values from each other and from zero
    j = 1; i = 2
    r = Int(odd)
    while i ≤ n2 
        if Σ[i] > ε
            if abs(Σ[i] - 1) > ε₂ #ε * 100
                while i ≤ n2 && abs(Σ[i - 1] - Σ[i]) < ε && Σ[i] > ε
                    i += 1
                end
                multiples[j] = i - j
                j = i; i += 1
            else
                while i ≤ n2 && abs(Σ[i] - 1) < ε₂ #ε * 100
                    i += 1
                end
                multiples[j] = i - j
                j = i; i += 1
            end
            
        else
            r += 2 * (n2 - i + 1)
            return r, (r - Int(odd)) ÷ 2
        end
    end
    return r, (r - Int(odd)) ÷ 2
end

function find_zeros(v::AbstractVector{T}) where T
    n = length(v)
    lz = zeros(Int, n)
    ε = 10 * eps(T)
    nz = 0
    for i ∈ 1:2:n
        if abs(v[i]) < ε
            nz += 1
             lz[nz] = i 
        end
    end
    return lz[1:nz], nz
end

@views function schurQ(Q::AbstractMatrix{T}, param::Symbol, check_zeros::Bool, ε::Number) where T
    n = size(Q, 1)
    n2 = n ÷ 2; n2b = n2 + Int(isodd(n)) 
    Σ = zeros(n2)
    #First memory allocations
    multiples = ones(Int, n2)
    V = similar(Q, n, n)

    #Compute the Schur decomposition of the skew-symmetric part
    A = skewhermitian(Q)
    if param == :H
        H = hessenberg(A)
        K = Matrix(H.Q)
        β = H.H.ev
    else
        K, β = skewlanczos(A)
    end
    # If n is odd, isolate one zero eigenvalue of the skew-symmetric part (particular interest on SO(n)) 
    #'Bidiagonal' type only admits square matrices so that it is necessary to perform this step.
    if isodd(n)
        Ginit = similar(Q, n - 1)
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
    complex_real_perm!(V, r, r2)
    
    m = n2 - r2
    smax = maximum(multiples[1:m])
    
    #Second memory allocation
    k = max(2smax, r)
    c = similar(Q, m)
    M = similar(Q, k, k)
    R = similar(Q, n, k)
    temp = similar(Q, n, max(m, r))
    mul!(temp[:, 1:m], Q, V[:, 1:m], 1, 0)
    if isone(smax)
        #No multiplicity of any Λsin(θ)
        for i ∈ 1:m
            c[i] = dot(V[:, i], temp[:, i])
        end
    else
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
                E = schur(M[1:ss, 1:ss])
                R[:, 1:ss] .= V[:, indices]
                for i ∈ 1:2:ss
                    E.T[i + 1, i] < 0 && Base.permutecols!!(E.Z[:, i:(i+1)], [2, 1])
                end
                even_odd_perm!(E.Z)
                mul!(V[:, indices], R[:,1:ss], E.Z, 1, 0)
                c[istart:(j-1)] .= real.(E.values[1:2:end])
                Σ[istart:(j-1)] .= abs.(imag.(E.values[1:2:end]))
            else
                c[j] = dot(V[:, j], temp[:, j])
                j += 1
            end
        end
    end
    Λᵣ = 0
    if r > 1
        mul!(temp[:, 1:r], Q, V[:, (n-r+1):n], 1, 0)
        mul!(M[1:r, 1:r],  V[:, (n-r+1):n]', temp[:, 1:r], 1, 0)
        E = eigen(Symmetric(M[1:r,1:r]))
        Λᵣ = E.values
        R[:, 1:r] .= V[:, (n-r+1):n]
        mul!(V[:, (n-r+1):n], R[:, 1:r], E.vectors, 1, 0)
    elseif isone(r)
        Λᵣ = [dot(V[:, n], Q, V[:, n])]
    end

    #Provide results in Tridiagonal Schur form
    d = zeros(n)
    dl = zeros(n - 1)
    d[1:2:(2m)] .= c
    d[2:2:(2m)] .= c
    d[(n-r+1):n] .= Λᵣ
    dl[1:2:2m] .= Σ[1:m]
    p = zeros(Int, 2m)
    p[1:2:2m] .= 1:m
    p[2:2:2m] .= (m+1):2m
    Base.permutecols!!(V[:, 1:2m], p)
    return Tridiagonal(dl, d, -dl), V
end

schurQ(Q::AbstractMatrix{T}) where T = schurQ(Q, :H, false, 10 * eps(T))

@views function create_Q(θ::AbstractVector, odd::Bool)
    N = length(θ) 
    n = (odd ? 2N + 1 : 2N)
    Q = exp(skewhermitian!(randn(n, n)))
    M = similar(Q, n, n)
    for i ∈ 1:N
        κ = randn()
        c = κ * cos(θ[i]); s = κ * sin(θ[i])
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    if odd
        M[:, end] .= Q[:, end]
    end
    return M * Q'
end

@views function func(Q)
    H = hessenberg(Q)
    M = Matrix(H.Q)
    return
end
