using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Distributions, Plots, LaTeXStrings
include("../src/normal_schur.jl")

"""
This file evaluates the accuracy of nrmschur(A::AbstractMatrix).
"""

@views function create_Q_with_real(θs::AbstractVector, λs::AbstractVector, κ::AbstractVector)
    p = length(θs) 
    r = length(λs)
    n = 2p + r
    QR = qr(randn(n, n))
    Q = Matrix(QR.Q) * Diagonal(sign.(diag(QR.R)))
    M = similar(Q, n, n)
    for (i, θ) ∈  enumerate(θs)
        c = κ[i] * cos(θ); s = κ[i] * sin(θ)
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    for (i, λ) ∈ enumerate(λs)
        M[:, 2p + i] .= (λ .* Q[:, 2p+i])
    end
    return M * Q'
end

BLAS.set_num_threads(1)
data1 = zeros(5)
data2 = zeros(5)
data3 = zeros(5)
data4 = zeros(5)
N = 100  #Number of samples
@views(for (i, n) ∈ enumerate([10, 32, 100, 316, 1000])
    global data1
    for k ∈ 1:N
        
        #Test 1
        r = 0
        p = (n - r) ÷ 2
        vc = π/4 .* rand(p)
        κ = ones(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
        T, V = nrmschur(Q)
        data1[i] += log2(norm(Q * V - V * T ) / norm(Q))
    
        
        #Test 2
        r = 0
        p = (n - r) ÷ 2
        vc = π .* rand(p)
        κ = 2 .* rand(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
        T, V = nrmschur(Q)
        data2[i] += log2(norm(Q * V - V * T ) / norm(Q))
        
        #Test 3
        r = floor(Int, n * 0.2)
        p = (n - r) ÷ 2
        vc = π .* rand(p)
        κ = 2 .* rand(p)
        vr = rand(r)
        Q = create_Q_with_real(vc, vr, κ)
        T, V = nrmschur(Q)
        data3[i] += log2(norm(Q * V - V * T ) / norm(Q))
        
        #Test 4
        r = 0
        m = floor(Int, n * 0.1)
        p = (n - 2m) ÷ 2
        vc = [π .* rand(p); (π* rand()) .* ones(m)]
        κ = ones(p + m)
        vr = rand(r)
        Q = create_Q_with_real(vc, vr, κ)
        T, V = nrmschur(Q)
        data4[i] += log2(norm(Q * V - V * T ) / norm(Q))
        #=
        #Test 5
        r = 0
        p = (n - r) ÷ 2
        vc = π * sqrt(eps(Float64)) .* (1 .+ randn(p))
        κ = 2 .* rand(p)
        vr = randn(r)
        Q = create_Q_with_real(vc, vr, κ)
        =#
    end 
end)
data1 ./= N 
data2 ./= N
data3 ./= N
data4 ./= N
data1 .= 2 .^(data1)
data2 .= 2 .^(data2)
data3 .= 2 .^(data3)
data4 .= 2 .^(data4)

P = plot(;
    xscale = :log10,
    yscale = :log10,
    xlabel = "Matrix size " * L"n",
    ylabel = L"\Vert A\widehat{Q} - \widehat{Q}\widehat{S}\ \Vert_\mathrm{F}\ /\ \Vert A\ \Vert_\mathrm{F}",
    legend = :topleft,
    grid = :both,
    ylims = (5e-16, 2e-11),
    framestyle = :box,
    font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=15,xguidefontsize=15, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = false,
yticks =([10^(-15),10^(-13),10^(-11)], [ L"10^{-15}",L"10^{-13}", L"10^{-11}"])
)
plot!(P, [10, 32, 100, 316, 1000], data1, label = "Experiment E1", marker = :circle, linestyle = :solid)
plot!(P, [10, 32, 100, 316, 1000], data2, label = "Experiment E2", marker = :diamond, linestyle = :dash)
plot!(P, [10, 32, 100, 316, 1000], data3, label = "Experiment E3", marker = :utriangle, linestyle = :dot)
plot!(P, [10, 32, 100, 316, 1000], data4, label = "Experiment E4", marker = :star5, linestyle = :dashdot)
plot!(P, [10, 1000], eps(Float64) * [10, 1000] * 10, label = L"\mathcal{O}(n)", color = :black, linestyle = :solid)
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/nrmschur_accuracy.pdf")
savefig(P, path)
display(P)