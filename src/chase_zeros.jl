using LinearAlgebra, SkewLinearAlgebra

function getgivens(a,b)
    nm = hypot(a, b)
    iszero(nm) && return 1, 0, 0
    return a / nm , b / nm, nm
end

@views function chase_zero!(β::AbstractVector{T}, loczero::Integer, Grow::AbstractMatrix{T}, Gcol::AbstractMatrix{T}) where T
    n = length(β)
    σ = β[loczero - 1]
    β[loczero - 1] = 0
    N = max(1, loczero - 2)
    #Chase backward
    kcol = 1
    for i ∈ N:-2:3
        iszero(σ) && break
        c, s, nm = getgivens(β[i], σ)
        Gcol[1, kcol] = c; Gcol[2, kcol] = s; kcol += 1
        β[i] = nm
        σ = - s * β[i - 1]
        β[i - 1] *= c
    end
    #Eliminate the bulge
    if !iszero(σ)
        c, s, nm = getgivens(β[1], σ)
        Gcol[1, kcol] = c; Gcol[2, kcol] = s
        β[1] = nm
    else
        kcol -= 1
    end
    #Chase forward
    σ = β[loczero + 1]
    β[loczero + 1] = 0
    N = min(n, loczero + 2)
    krow = 1
    for i ∈ N:2:(n-2)
        iszero(σ) && break
        c, s, nm = getgivens(β[i], σ)
        Grow[1, krow] = c; Grow[2, krow] = s; krow += 1
        β[i] = nm
        σ = - s * β[i + 1]
        β[i + 1] *= c
    end
    #Eliminate the bulge
    if !iszero(σ)
        c, s, nm = getgivens(β[n], σ)
        Grow[1, krow] = c; Grow[2, krow] = s
        β[n] = nm
    else
        krow -= 1
    end
    return krow, kcol
end



@views function chase_zeros!(β::AbstractVector{T}, Q::AbstractMatrix{T}) where T
    n = length(β)
    N = size(Q, 1)
    ε = 10 * eps(T)
    start = 1
    Gcol = zeros(T, 2, n)
    Grow = zeros(T, 2, n)
    for k ∈ 3:2:(n-2)
        if abs(β[k]) < ε
            if k - start > 0
                krow, kcol = chase_zero!(β[start:end], k - start + 1, Grow, Gcol)
                q2 = (k + 1) ÷  2; q1 = q2
                @inbounds(for i ∈ 1:kcol
                    c, s = Gcol[1, i], Gcol[2, i]
                    q1 -= 1 
                    for j ∈ 1:N
                        α, δ = Q[j, q1], Q[j, q2]
                        Q[j, q1] =  c * α + s * δ
                        Q[j, q2] =  -s * α +  c * δ
                    end
                end)
                q2 = ceil(Int, N / 2) + (k + 1) ÷  2
                q1 = q2
                @inbounds(for i ∈ 1:krow
                    c, s = Grow[1, i], Grow[2, i]
                    q1 += 1 
                    for j ∈ 1:N
                        α, δ = Q[j, q1], Q[j, q2]
                        Q[j, q1] =  c * α + s * δ
                        Q[j, q2] = -s * α + c * δ
                    end
                end)
            end
            start = k + 2
        end
    end
end
