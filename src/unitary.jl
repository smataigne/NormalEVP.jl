using LinearAlgebra, BenchmarkTools 

"""
This file implements the qr algorithm for unitary and orthogonal matrices as described, first, in

W. B. Gragg, The QR algorithm for unitary Hessenberg matrices, J. Comput. Appl. Math.,16 (1986), pp. 1–8.

but also, notably in

Stewart, Michael, 'An Error Analysis of a Unitary Hessenberg QR Algorithm', SIAM Journal on Matrix Analysis and Applications, 28(1), pp. 40-67, 2006.  https://doi.org/10.1137/04061948X
"""
function assembleH(γ::AbstractVector{T}, σ::AbstractVector{T}) where T
    o = one(T)
    n = size(γ, 1)
    H = Matrix(one(T)*I, n, n)
    temp = similar(H, n, 2)
    for k ∈ 1:n-1
        temp[1:k+1, :] .= H[1:k+1, k:k+1]
        @. H[1:k+1, k] = -γ[k] * temp[1:k+1, 1] + σ[k] * temp[1:k+1, 2]
        @. H[1:k+1, k+1] = σ[k] * temp[1:k+1, 1] + γ[k]' * temp[1:k+1, 2]
    end
    H[:, n] .*= -γ[n]
    return H
end

function chooseshift(α::AbstractVector, β::Number)
    t = α[1]' * α[2]- α[2]' * α[3]
    w = α[1]' * α[3]
    v = sqrt(t * t - 4 * w)
    λ = (- t - v) / 2
    if abs(λ + α[2]' * α[3]) ≤ abs(sqrt(abs(α[1])) * β) ≤ abs(λ + α[1]' * α[2])
        return λ
    end
    return (- t + v) / 2
end

function chooseshift2(α::AbstractVector, β::Number)
    return -α[3] * α[2] - β * 1im
end

"""
W. B. Gragg, The QR algorithm for unitary Hessenberg matrices, J. Comput. Appl. Math.,16 (1986), pp. 1–8.

In view of:
Stewart, Michael, 'An Error Analysis of a Unitary Hessenberg QR Algorithm', SIAM Journal on Matrix Analysis and Applications, 28(1), pp. 40-67, 2006.  https://doi.org/10.1137/04061948X
This implementation is not the most stable.
"""
@views function qrstep!(γ::AbstractVector,σ::AbstractVector, α::AbstractVector, β::AbstractVector, ρ::AbstractVector,
    newγ::AbstractVector, newσ::AbstractVector, αs::AbstractVector, τ::Number, n::Integer)
    T = typeof(τ)
    α[1] = 1; β[1] = 1; αs[1] = 1
    @inbounds(for k ∈ 1:n-1
        ρ[k] = sqrt(σ[k]^2 + abs(τ * α[k] + γ[k] * αs[k])^2)
        newσ[k] = β[k] * ρ[k]
        β[k + 1] = σ[k] / ρ[k]
        α[k + 1]  = (τ * α[k] + γ[k] * αs[k]) / ρ[k]
        αs[k + 1] = (αs[k] + γ[k]' * τ * α[k]) / ρ[k]
        newγ[k] = α[k + 1] * αs[k + 1]' - β[k + 1]^2 * γ[k + 1] * τ'
    end)
    ρ[n] = abs(τ * α[n] + γ[n] * αs[n])
    newσ[n] = β[n] * ρ[n]
    m = randn(T)
    m /= abs(m)
    α[n + 1] = (ρ[n] > 0 ? (τ * α[n] + γ[n] * αs[n]) / ρ[n] : m)
    αs[n + 1] = γ[n]' * α[n + 1]
    newγ[n] = γ[n]
    σ[1:n] .= newσ[2:n+1]
    γ[1:n] .= newγ[1:n]
    newσ .= 0; newγ .= 0  
end

@views function qrstep_novec!(γ::AbstractVector,σ::AbstractVector, τ::Number, n::Integer)
    α = 1; β = 1; αs = 1
    #initial iteration
    ρ = sqrt(σ[1]^2 + abs(τ * α + γ[1] * αs)^2)
    β = σ[1] / ρ
    oldα = α
    α  = (τ * α + γ[1] * αs) / ρ
    αs = (αs + γ[1]' * τ * oldα) / ρ
    γ[1] = α * αs' - β^2 * γ[2] * τ'
    #regular iterations
    @inbounds(for k ∈ 2:n-1
        ρ = sqrt(σ[k]^2 + abs(τ * α + γ[k] * αs)^2)
        σ[k - 1] = β[k] * ρ
        β = σ[k] / ρ
        oldα = α
        α  = (τ * α + γ[k] * αs) / ρ
        αs = (αs + γ[k]' * τ * oldα) / ρ
        γ[k] = α * αs' - β^2 * γ[k + 1] * τ'
    end)
    ρ = abs(τ * α + γ[n] * αs)
    σ[n - 1] = β * ρ
    σ[n] = 0
end

@views function updatevectors!(V::AbstractMatrix, temp::AbstractMatrix, G::AbstractMatrix, α::AbstractVector, β::AbstractVector, n::Integer)
    @inbounds(for k ∈ 1:n-1
        temp[:, :] .= V[:, k:k+1]
        G[1, 1] = -α[k]; G[2, 1] = β[k] 
        G[1, 2] =  β[k]; G[2, 2] = α[k]'
        mul!(V[:, k:k+1], temp, G, 1, 0)
        #=
        @. V[:, k]  = -α[k] * temp[:, 1] + β[k] * temp[:, 2]
        @. V[:, k+1] = β[k] * temp[:, 1] + α[k]' * temp[:, 2]
        =#
    end)
    V[:, n] .*= -α[n]
    return  
end

"""
```unitaryeigvals!(A::AbstractMatrix)```

UHQR algorithm for unitary matrices. Only eigenvalues are computed.
"""
@views function unitaryeigvals!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    N = n
    ε = eps(real(T))
    H = hessenberg!(A)
    TC = complex(T)
    γ = zeros(TC, n)
    σ = zeros(TC, n)
    itermax = 10n
    #Initialization
    @inbounds(for k ∈ 1:(n - 1)
        γ[k] = -H.H[k, k] ; σ[k] = H.H[k + 1, k]
        for j ∈ (k + 1):n
            H.H[k + 1, j] = σ[k] * H.H[k, j] + γ[k] * H.H[k + 1, j]
        end
    end)
    γ[n] = -H.H[n, n]; σ[n] = 0
    iter = 0
    tr = 0
    τ = 1
    tb =0
    @inbounds(while iter < itermax && n > 1
        #One QR iteration
        if tr < -5 || (1e4 * abs(σ[n-2]) ≥ abs(σ[n - 1]) && tr < 5)
            τ = - γ[n - 1]' * γ[n]
            tr += 1
        else
            tb +=1
            if n > 2
                τ = chooseshift(γ[n-2:n] , σ[n-1])
            else
                τ = chooseshift([1, γ[n-1], γ[n]] , σ[n-1])
            end 
            tr -= 1
        end
        if abs(τ) < ε * √N
            τ = one(T)
        end
        qrstep_novec!(γ, σ, τ, n)
        while n > 1 && abs(σ[n - 1]) < ε * √N
            n -= 1; tr= 0
        end
        iter += 1
    end)
    return γ, σ
end

"""
```unitaryeigvals(A::AbstractMatrix)```

UHQR algorithm for unitary matrices. Only eigenvalues are computed.
"""
unitaryeigvals(A::AbstractMatrix) = unitaryeigvals!(copy(A))

"""
```unitaryeigen!(A::AbstractMatrix)```

UHQR algorithm for unitary matrices. Eigenvalues and eigenvectors are computed.
"""
@views function unitaryeigen!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    N = n
    ε = eps(real(T))
    H = hessenberg(A)
    Q = complex.(Matrix(H.Q))
    TC = complex(T)
    temp = zeros(TC, n, 2)
    G = zeros(TC, 2, 2)
    γ = zeros(TC, n)
    σ = zeros(TC, n)
    newγ = zeros(TC, n)
    newσ = zeros(TC, n + 1)
    α = zeros(TC, n + 1)
    αs = zeros(TC, n + 1)
    β = zeros(TC, n + 1)
    ρ = zeros(real(T), n)
    itermax = 10n
    #Initialization
    @inbounds(for k ∈ 1:(n - 1)
        γ[k] = -H.H[k, k] ; σ[k] = H.H[k + 1, k]
        for j ∈ (k + 1):n
            H.H[k + 1, j] = σ[k] * H.H[k, j] + γ[k] * H.H[k + 1, j]
        end
    end)
    γ[n] = -H.H[n, n]; σ[n] = 0
    iter = 0
    tr = 0
    τ = 1
    tb =0
    @inbounds(while iter < itermax && n > 1
        #One QR iteration
        if tr < -5 || (1e4 * abs(σ[n-2]) ≥ abs(σ[n - 1]) && tr < 5)
            τ = - γ[n - 1]' * γ[n]
            tr += 1
        else
            tb +=1
            if n > 2
                τ = chooseshift(γ[n-2:n] , σ[n-1])
            else
                τ = chooseshift([1, γ[n-1], γ[n]] , σ[n-1])
            end 
            tr -= 1
        end
        if abs(τ) < ε * √N
            τ = one(T)
        end
        qrstep!(γ, σ, α, β, ρ, newγ, newσ, αs, τ, n)
        updatevectors!(Q, temp, G, α[2:end], β[2:end], n)
        while n > 1 && abs(σ[n - 1]) < ε * √N
            n -= 1; tr= 0
        end
        iter += 1
    end)
    return γ, σ, Q
end

"""
```unitaryeigen(A::AbstractMatrix)```

UHQR algorithm for unitary matrices. Only eigenvalues are computed.
"""
unitaryeigen(A::AbstractMatrix) = unitaryeigen!(copy(A))

"""
```orthoeigvals!(A::AbstractMatrix)```

UHQR algorithm for orthogonal matrices. Only eigenvalues are computed. Uses conjugate shifts and converges to the real Schur form.
"""
@views function orthoeigvals!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    N = n
    ε = eps(real(T))
    H = hessenberg!(A)
    TC = complex(T)
    γ = zeros(TC, n)
    σ = zeros(TC, n)
    itermax = 10n
    #Compute SChur parametric form
    @inbounds(for k ∈ 1:(n - 1)
        γ[k] = -H.H[k, k] ; σ[k] = H.H[k + 1, k]
        @simd(for j ∈ (k + 1):n
            H.H[k + 1, j] = σ[k] * H.H[k, j] + γ[k] * H.H[k + 1, j]
        end)
    end)
    γ[n] = -H.H[n, n]; σ[n] = 0
    iter = 0
    tr = 0
    τ = 1
    @inbounds(while iter < itermax && n > 1
        #One QR iteration
        if  tr < -10 || (abs(σ[n-2]) ≥ abs(σ[n - 1]) && tr < 10)
            τ = - real(γ[n - 1]' * γ[n])
            if abs(τ) < ε * √N || tr == 9
                τ = one(T)
            end
            tr += 1
            qrstep_novec!(γ, σ, τ, n)
        else
            τ = (n > 2 ? chooseshift2(γ[n-2:n] , σ[n-1]) : chooseshift2([1, γ[n-1], γ[n]] , σ[n-1]))
            if abs(τ) < ε * √N || tr == -9
                τ = one(T)
                qrstep_novec!(γ, σ, τ, n)
            else
                qrstep_novec!(γ, σ, τ, n)
                qrstep_novec!(γ, σ, τ', n)
            end
            tr -= 1
            
        end
        while n > 1 && abs(σ[n - 1]) < ε * √N 
            n -= 1; tr = 0
        end
        while n > 2 && abs(σ[n - 2]) < ε * √N 
            n -= 2; tr = 0
        end
        iter += 1
    end)
    return γ, σ
end

"""
```orthoeigvals(A::AbstractMatrix)```

UHQR algorithm for orthogonal matrices. Only eigenvalues are computed. Uses conjugate shifts and converges to the real Schur form.
"""
orthoeigvals(A::AbstractMatrix) = orthoeigvals!(copy(A))

"""
```orthoeigen!(A::AbstractMatrix)```

UHQR algorithm for orthogonal matrices. Eigenvalues and eigenvectors are computed. Uses conjugate shifts and converges to the real Schur form.
"""
@views function orthoeigen!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    N = n
    ε = eps(real(T))
    H = hessenberg!(A)
    TC = complex(T)
    temp = zeros(TC, n, 2)
    G = zeros(TC, 2, 2)
    Q = complex.(Matrix(H.Q))
    γ = zeros(TC, n)
    σ = zeros(TC, n)
    newγ = zeros(TC, n)
    newσ = zeros(TC, n + 1)
    α = zeros(TC, n + 1)
    αs = zeros(TC, n + 1)
    β = zeros(TC, n + 1)
    ρ = zeros(real(T), n)
    itermax = 10n
    #Initialization
    @inbounds(for k ∈ 1:(n - 1)
        γ[k] = -H.H[k, k] ; σ[k] = H.H[k + 1, k]
        for j ∈ (k + 1):n
            H.H[k + 1, j] = σ[k] * H.H[k, j] + γ[k] * H.H[k + 1, j]
        end
    end)
    γ[n] = -H.H[n, n]; σ[n] = 0
    iter = 0
    tr = 0
    τ = 1
    tb =0
    @inbounds(while iter < itermax && n > 1
        #One QR iteration
        if  tr < -10 || (abs(σ[n-2]) ≥ abs(σ[n - 1]) && tr < 10)
            τ = - real(γ[n - 1]' * γ[n])
            if abs(τ) < ε * √N || tr == 9
                τ = 1
            end
            tr += 1
            qrstep!(γ, σ, α, β, ρ, newγ, newσ, αs, τ, n)
            updatevectors!(Q, temp, G, c, s, n)
        else
            tb +=1
            τ = (n > 2 ? chooseshift2(γ[n-2:n] , σ[n-1]) : chooseshift2([1, γ[n-1], γ[n]] , σ[n-1]))
            if abs(τ) < ε * √N || tr == -9
                τ = 1
                qrstep!(γ, σ, α, β, ρ, newγ, newσ, αs, τ, n)
                updatevectors!(Q, temp, G, c, s, n)
            else
                qrstep!(γ, σ, α, β, ρ, newγ, newσ, αs, τ, n)
                updatevectors!(Q, temp, G, c, s, n)
                qrstep!(γ, σ, α, β, ρ, newγ, newσ, αs, τ', n)
                updatevectors!(Q, temp, G, c, s, n)
            end
            tr -= 1
        end
        while n > 1 && abs(σ[n - 1]) < ε * √N 
            n -= 1; tr = 0
        end
        while n > 2 && abs(σ[n - 2]) < ε * √N 
            n -= 2; tr = 0
        end
        iter += 1
    end)
    return γ, σ, Q
end

"""
```orthoeigen(A::AbstractMatrix)```

UHQR algorithm for orthogonal matrices. Eigenvalues and eigenvectors are computed. Uses conjugate shifts and converges to the real Schur form.
"""
orthoeigen(A::AbstractMatrix) = orthoeigen!(copy(A))






