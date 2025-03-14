using LinearAlgebra
include("wxeigen.jl")

@views function householder!(x::AbstractVector{T}, n::Integer) where T
    if n == 1 && T <:Real
        return T(0), real(x[1]) # no final 1x1 reflection for the real case
    end
    xnorm = norm(x[2:end]) 
    α = x[1]
    if !iszero(xnorm) || (!iszero(α))
        β = (real(α) > 0 ? -1 : +1) * hypot(α,xnorm)
        τ = 1 - α / β
        α = 1 / (α - β)
        x[1] = 1
        x[2:n] .*= α
    else 
        τ = T(0)
        x .= 0
        β = real(T)(0)
    end
    return τ, β
end

@views function apply_householder!(H::AbstractMatrix, τ::AbstractVector, v::AbstractVector, reverse = false)
    k = size(H, 2)
    if !(reverse)
        for i ∈ 1:k
            v[i:end] .-= τ[i] * dot(v[i:end], H[i:end, i]) * H[i:end, i]
        end
    else
        for i ∈ k:-1:1
            v[i:end] .-= τ[i] * dot(v[i:end], H[i:end, i]) * H[i:end, i]
        end
    end
end

@views function apply_wx_householder!(H::AbstractMatrix, τ::AbstractVector, v::AbstractVector, reverse = false)
    k = size(H, 2)
    n = size(H, 1); n2 = n ÷ 2
    p = zeros(n)
    if !(reverse)
        
        for i ∈ 1:k
            v[i:end] .-= τ[i] * dot(v[i:end], H[i:end, i]) * H[i:end, i]
            
        end
        for i ∈ 1:k
            p[1:n2] .= -H[(n2+1):n, i] 
            p[(n2+1):n] .= H[1:n2, i] 
            v .-= τ[i] * dot(v, p) * p
        end
    else
        
        for i ∈ k:-1:1
            p[1:n2] .= -H[(n2+1):n, i] 
            p[(n2+1):n] .= H[1:n2, i] 
            v .-= τ[i] * dot(v, p) * p 
        end
        for i ∈ k:-1:1
            v[i:end] .-= τ[i] * dot(v[i:end], H[i:end, i]) * H[i:end, i] 
        end
    end
end

@views function safelanczos!(A::Symmetric{T}) where T
    n = size(A, 1)
    K = zeros(T, n, n)
    H = zeros(T, n, n - 1)
    v = zeros(T, n)
    τ = zeros(T, n)
    β = zeros(T, n) 
    α = zeros(T, n)
    
    mul!(K[:, 1], A, randn(n), 1, 0)
    K[:, 1] ./= norm(K[:, 1])

    H[:, 1] .= K[:, 1]
    a,  = householder!(H[:, 1], n)
    τ[1] = a
    
    mul!(v, A, K[:, 1], 1, 0)
    α[1] = dot(K[:, 1], v)  
   
    @inbounds(for i ∈ 1:n-1
        v .-= α[i] * K[:, i] + β[i] * K[:, i - 1]
        apply_householder!(H[:, 1:i], τ[1:i], v, false)
        H[(i+1):n, i + 1] .= v[(i+1):n]
        a, b = householder!(H[(i+1):n, i+1], n - i)
        τ[i + 1] = a
        β[i + 1] = b
        K[i + 1, i + 1] = 1
        apply_householder!(H[:, 1:(i+1)], τ[1:(i+1)], K[:, i + 1], true)    
        mul!(v, A, K[:, i + 1], 1, 0)
        α[i + 1] = dot(K[:, i + 1], v)   
    end)
    return K, SymTridiagonal(α, copy(β[2:n]))
end

@views function gslanczosfull!(A::Symmetric{T}) where T
    tol  = 10 * eps(T) * norm(A)
    n = size(A, 1);
    K = zeros(T, n, n)
    mul!(K[:, 1], A, randn(n), 1, 0)
    K[:, 1] ./= norm(K[:, 1])
    β = zeros(T, n - 1) 
    α = zeros(T, n)
    @inbounds(for i ∈ 1:n-1 #Only half Lanczos iterations are needed

        mul!(K[:, i + 1], A, K[:, i], 1, 0)
        ### Full re-orthogonalization
        for j ∈ 1:i-1
            K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j]) .* K[:, j]
        end
        α[i] = dot(K[:, i], K[:, i + 1])
        K[:, i + 1] .-= α[i] .* K[:, i]
        
        β[i] = norm(K[:, i + 1])
        if β[i] < tol
            K[:, i + 1] = randn(n)
            ### Full re-orthogonalization
            for j ∈ 1:i
                K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j]) .* K[:, j]
            end
            K[:, i + 1] ./= norm(K[:, i + 1])
        else
            K[:, i + 1] ./= β[i]
        end
    end)
    α[n] = dot(K[:, n], A,  K[:, n])
    return K, SymTridiagonal(α, β)
end

@views function gswxlanczosfull!(A::Symmetric{T}) where T
        tol = 10 * eps(T) * norm(A)
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
                K[:, i + 1] .-= α[i] * K[:, i]
                K[:, i + 1] .-= dot(K[:, i + 1] , K[:, i + n2]) .* K[:, i + n2]
            end

            ### Full re-orthogonalization
            for j ∈ 1:i-1
                K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j]) .* K[:, j]
                K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j + n2]) .* K[:, j + n2]
            end
            
            β[i] = norm(K[:, i + 1])
            if β[i] < tol
                K[:, i + 1] = randn(n)
                ### Full re-orthogonalization
                for j ∈ 1:i
                    K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j]) .* K[:, j]
                    K[:, i + 1] .-= dot(K[:, i + 1] , K[:, j + n2]) .* K[:, j + n2]
                end
                K[:, i + 1] ./= norm(K[:, i + 1])
            else
                K[:, i + 1] ./= β[i]
            end
        end)
        K[1:n2, n] .= -K[(n2+1):n, n2] 
        K[(n2+1):n, n] .= K[1:n2, n2] 
        α[n2] = dot(K[:, n2], A,  K[:, n2])
        α[(n2+1):n] .= α[1:n2]
        β[(n2+1):(n-1)] .= β[1:(n2-1)]
        return K, SymTridiagonal(α, β)
    end

@views function safewxlanczos!(A::Symmetric{T}) where T
    n = size(A, 1); n2 = n ÷ 2
    K = zeros(T, n, n)
    H = zeros(T, n, n2)
    v = zeros(T, n)
    τ = zeros(T, n2)
    β = zeros(T, n) 
    α = zeros(T, n)
    
    mul!(K[:, 1], A, randn(n), 1, 0)
    K[:, 1] ./= norm(K[:, 1])
    K[1:n2, n2 + 1] .= -K[(n2+1):n, 1] 
    K[(n2+1):n, n2 + 1] .= K[1:n2, 1] 

    H[:, 1] .= K[:, 1]
    a,  = householder!(H[:, 1], n)
    τ[1] = a
    
    mul!(v, A, K[:, 1], 1, 0)
    α[1] = dot(K[:, 1], v)  
    @inbounds(for i ∈ 1:n2-1
        v .-= α[i] * K[:, i] + β[i] * K[:, i - 1]
        apply_wx_householder!(H[:, 1:i], τ[1:i], v, false)
        H[(i+1):n, i + 1] .= v[(i+1):n]
        a, b = householder!(H[(i+1):n, i+1], n - i)
        τ[i + 1] = a
        β[i + 1] = b
        K[i + 1, i + 1] = 1
        apply_wx_householder!(H[:, 1:(i+1)], τ[1:(i+1)], K[:, i + 1], true)
        K[1:n2, n2 + i + 1] .= -K[(n2+1):n, i + 1] 
        K[(n2+1):n, n2 + i + 1] .= K[1:n2, i + 1]
        mul!(v, A, K[:, i + 1], 1, 0)
        α[i + 1] = dot(K[:, i + 1], v)   
    end)
    α[(n2+1):n] .= α[1:n2]
    β[(n2+2):n] .= β[2:n2]
    return K, SymTridiagonal(α, copy(β[2:n]))
end

n = 6
A = randn(n, n); A = (A + A') / 2
Y = Symmetric(A)
E = eigen(A)
E.values[1:2] .= 2
Y = Symmetric(E.vectors * Diagonal(E.values) * E.vectors')

K, T = gslanczosfull!(Y)
#=
display(K'K)
display(K'Y*K)
display(Y * K - K *T)
=#

m = 10
W = randn(m, m); W = (W + W')/2
X = randn(m, m); X = (X - X')/2
A = [W -X; X W]
E = wxeigen(A)
E.values[1:2] .= 2
E.values[m+1:m+2] .= 2
#A = Symmetric(E.vectors * Diagonal(E.values) * E.vectors')
#K, T = gswxlanczosfull!(A)
K, T = safewxlanczos!(Symmetric(A))
display(K'K)
#display(K'A*K)
#display(A * K - K * T)
