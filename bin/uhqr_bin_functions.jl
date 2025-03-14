@views function updatevectors3!(V::AbstractMatrix{T}, H::UpperHessenberg, temp::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix, Ns::AbstractVector, nb::Integer) where T
    #BLAS 3 accumulation of Givens rotations
    nk = (Ns[1] - 1) ÷ nb
    nsweep = size(α, 2)
    @inbounds(for k ∈ 1:nk
        H.data .= 0
        for j ∈ 1:nb
            H.data[j, j] = 1
        end
        nkb = (k - 1) * nb
        for j ∈ 1:nb
            kk = nkb + j
            temp[1:(j+1), 1] .= H.data[1:(j+1), j]; temp[j + 1, 1] = 0
            temp[1:(j+1), 2] .= H.data[1:(j+1), j + 1]  
            @. H.data[1:(j+1), j]    = -α[kk] * temp[1:(j+1), 1] + β[kk] * temp[1:(j+1), 2]
            @. H.data[1:(j+1), j + 1]  = β[kk] * temp[1:(j+1), 1] + α[kk]' * temp[1:(j+1), 2]
        end
        temp .= V[:, ((k-1) * nb + 1): (k * nb + 1)]
        #mul!(V[:, ((k-1) * nb + 1):(k * nb + 1)], temp, H.data, 1, 0)
        BLAS.gemm!('N', 'N', one(T), temp, H.data, zero(T), V[:, ((k-1) * nb + 1):(k * nb + 1)])
        #=Attempt to take advantage of Hessenberg structure: not as fast as gemm!
        temp .= V[:, ((k-1) * nb + 1): (k * nb + 1)]
        BLAS.trmm!('R', 'U', 'N', 'N', 1, H.data, V[:, ((k-1) * nb + 1): (k * nb + 1)])
        nkb = (k-1) * nb 
        @simd(for j ∈ 1:nb
            @. V[:, nkb + j] += temp[:, j + 1] * H[j + 1, j]
        end)
        =#
        
    end)
    tk = max(nb * nk, 1)
    @inbounds(for k ∈ tk:n-1
        temp[:, 1:2] .= V[:, k:k+1]
        @. V[:, k]    = -α[k] * temp[:, 1] + β[k] * temp[:, 2]
        @. V[:, k + 1] = β[k] * temp[:, 1] + α[k]' * temp[:, 2]
    end)
    V[:, n] .*= -α[n]
    return  
end

@views function unitaryeigen2!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    N = n
    ε = eps(real(T))
    TC = complex(T)
    H = hessenberg(A)
    nb = min(n, 6)
    Hb = UpperHessenberg(zeros(TC, nb + 1, nb + 1))
    Q = complex.(Matrix(H.Q))
    
    temp = zeros(TC, n, nb + 1)
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
        ksweep += 1
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
        updatevectors2!(Q, Hb, temp, α[2:end], β[2:end], n, nb)
        while n > 1 && abs(σ[n - 1]) < ε * √N
            n -= 1; tr= 0
        end
        iter += 1
    end)
    return γ, σ, Q
end

unitaryeigen2(A::AbstractMatrix) = unitaryeigen2!(copy(A))

@views function updatevectors2!(V::AbstractMatrix{T}, H::UpperHessenberg, temp::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix, n::Integer, nb::Integer) where T
    #BLAS 3 accumulation of Givens rotations
    nk = (n - 1) ÷ nb
    @inbounds(for k ∈ 1:nk
        H.data .= 0
        for j ∈ 1:nb
            H.data[j, j] = 1
        end
        nkb = (k - 1) * nb
        for j ∈ 1:nb
            kk = nkb + j
            temp[1:(j+1), 1] .= H.data[1:(j+1), j]; temp[j + 1, 1] = 0
            temp[1:(j+1), 2] .= H.data[1:(j+1), j + 1]  
            @. H.data[1:(j+1), j]    = -α[kk] * temp[1:(j+1), 1] + β[kk] * temp[1:(j+1), 2]
            @. H.data[1:(j+1), j + 1]  = β[kk] * temp[1:(j+1), 1] + α[kk]' * temp[1:(j+1), 2]
        end
        temp .= V[:, ((k-1) * nb + 1): (k * nb + 1)]
        #mul!(V[:, ((k-1) * nb + 1):(k * nb + 1)], temp, H.data, 1, 0)
        BLAS.gemm!('N', 'N', one(T), temp, H.data, zero(T), V[:, ((k-1) * nb + 1):(k * nb + 1)])
        #=Attempt to take advantage of Hessenberg structure: not as fast as gemm!
        temp .= V[:, ((k-1) * nb + 1): (k * nb + 1)]
        BLAS.trmm!('R', 'U', 'N', 'N', 1, H.data, V[:, ((k-1) * nb + 1): (k * nb + 1)])
        nkb = (k-1) * nb 
        @simd(for j ∈ 1:nb
            @. V[:, nkb + j] += temp[:, j + 1] * H[j + 1, j]
        end)
        =#
        
    end)
    tk = max(nb * nk, 1)
    @inbounds(for k ∈ tk:n-1
        temp[:, 1:2] .= V[:, k:k+1]
        @. V[:, k]    = -α[k] * temp[:, 1] + β[k] * temp[:, 2]
        @. V[:, k + 1] = β[k] * temp[:, 1] + α[k]' * temp[:, 2]
    end)
    V[:, n] .*= -α[n]
    return  
end

"""
Stewart, Michael, 'An Error Analysis of a Unitary Hessenberg QR Algorithm', SIAM Journal on Matrix Analysis and Applications, 28(1), pp. 40-67, 2006.  https://doi.org/10.1137/04061948X
"""
@views function stableqrstep!(γ::AbstractVector,σ::AbstractVector, α::AbstractVector, β::AbstractVector, ρ::Number,
    newγ::AbstractVector, newσ::AbstractVector, αs::Number, τ::Number, n::Integer)
    T = typeof(τ)
    α[1] = 1; β[1] = 1; αs = 1
    @inbounds(for k ∈ 1:n-1
        ρ = sqrt(σ[k]^2 + abs(τ * α[k] + γ[k] * αs)^2)
        newσ[k] = β[k] * ρ
        β[k + 1] = σ[k] / ρ
        α[k + 1]  = (τ * α[k] + γ[k] * αs) / ρ
        αs = (αs[k] + γ[k]' * τ * α[k]) / ρ
        newγ[k] = α[k + 1] * αs[k + 1]' - β[k + 1]^2 * γ[k + 1] * τ'
    end)
    ρ = abs(τ * α[n] + γ[n] * αs)
    newσ[n] = β[n] * ρ[n]
    m = randn(T)
    m /= abs(m)
    α[n + 1] = (ρ > 0 ? (τ * α[n] + γ[n] * αs) / ρ : m)
    αs = γ[n]' * α[n + 1]
    newγ[n] = γ[n]
    σ[1:n] .= newσ[2:n+1]
    γ[1:n] .= newγ[1:n]
    newσ .= 0; newγ .= 0  
end