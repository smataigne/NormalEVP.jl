using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Distributions, Plots, LaTeXStrings
include("../src/normal_schur.jl")

"""
This file evaluates the accuracy of nrmschur(A::AbstractMatrix) for eigenvalues concentrated around a mean value θ with dispersion Δθ.
"""

function bound(d, σ)
    Σ = 1
    p = size(d,1)
    for i ∈ 1:p
        for j ∈ i+1:p
            Σ =max(Σ,1+abs((d[i]-d[j])/(σ[i]-σ[j])))#max(Σ, 1 + tan((θ[i] + θ[j])/2)^2)
        end
    end
    return Σ
end
function avg_bound(d, σ)
    Σ = 1
    p = size(d,1)
    for i ∈ 1:p
        for j ∈ i+1:p
            Σ += 1+abs((d[i]-d[j])/(σ[i]-σ[j]))#max(Σ, 1 + tan((θ[i] + θ[j])/2)^2)
        end
    end
    return Σ *2/(p * (p - 1))
end

@views function create_Q_with_real(θs::AbstractVector, λs::AbstractVector)
    p = length(θs) 
    r = length(λs)
    n = 2p + r
    QR = qr(randn(n, n))
    Q = Matrix(QR.Q) * Diagonal(sign.(diag(QR.R)))
    M = similar(Q, n, n)
    for (i, θ) ∈  enumerate(θs)
        κ = 2 * rand()
        c = κ * cos(θ); s = κ * sin(θ)
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    for (i, λ) ∈ enumerate(λs)
        M[:, 2p + i] .= (λ .* Q[:, 2p+i])
    end
    return M * Q'
end

e = √eps(Float64)
BLAS.set_num_threads(1)
l = 40
data = zeros(l, 4)
N = 50
n = 100
@views(for (i, α) ∈ enumerate(LinRange(0, π/2, l))
    global data,n
    for k ∈ 1:N
        r = 0
        p = (n - r) ÷ 2
        θ = α .+ (0.005 .* rand(p))
        Q = create_Q_with_real(θ, randn(r))
        T, V = nrmschur(Q)
        S = schur(Q)
        τ = max(norm((Q-Q') * V - V * (T-T')) / (2 * norm(Q)), 10eps(Float64)) / eps(Float64)
        data[i, 1] += log2(norm(Q * V - V * T ) / norm(Q))
        #data[i, 2] += log2(norm(V'V - I))
        #data[i, 3] += log2(avg_bound(T.d[1:2:end],T.dl[1:2:end]))
        data[i, 4] += log2(bound(T.d[1:2:end],T.dl[1:2:end]) * τ) 
    end
    data[i, :] ./= N  
end)
data[:, 1] .= 2 .^(data[:, 1])
#data[:, 2] .= 2 .^(data[:, 2])
#data[:, 3] .= 2 .^(data[:, 3]) #Geometric mean
data[:, 4] .= 2 .^(data[:, 4])

P= plot(framestyle=:box,legend=:top,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = false, minorgridalpha=0.01, yscale=:log10,
xticks =([0,π/4,π/2], [ L"0",L"\frac{\pi}{4}", L"\frac{\pi}{2}"]),
yticks =([1e-9,1e-11, 1e-13, 1e-15],[L"10^{-9}",L"10^{-11}",L"10^{-13}", L"10^{-15}"]), ylims= (1e-15*0.5, 4*1e-9))
plot!(LinRange(0,π/2, l), data[:, 1], label="Samples", color=:blue, linestyle=:dash, markersize=3, markershape=:circle, markerstrokewidth=0.1)
x = LinRange(0,π/2-0.00001, 1000)
plot!(LinRange(0,π/2, l), (data[:, 4]).*eps(Float64), color=:green, label=L" $\varepsilon_m \tau \ \max_{i,j}\frac{|\lambda_ic_i-\lambda_j c_j\ |}{|\lambda_is_i-\lambda_j s_j\ |}$", linestyle=:solid, markersize=3, markershape=:diamond, markerstrokewidth=0.5)
xlabel!("Average phase " * L"\theta")
ylabel!(L"\Vert A\widehat{Q}- \widehat{Q}\widehat{S}\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/worst_angle_normal.pdf")
savefig(P, path)
display(P)