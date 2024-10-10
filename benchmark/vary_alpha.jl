using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Random, Plots, LaTeXStrings
include("../src/normal_schur.jl")

@views function create_Q(θs::AbstractVector, λs::AbstractVector)
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
        M[:, 2p + i] .= (λ .* Q[:, end])
    end
    return M * Q'
end

BLAS.set_num_threads(1)
n = 1000
αs = Vector(0:0.02:0.6)
N = length(αs)
K = 100
ε = 10 * eps(Float64)
means = zeros(N, K)
for j ∈ 0:K
    for (i, α) ∈ enumerate(αs)
        local Q
        r = floor(Int, n * α)
        p = (n - r) ÷ 2
        Q = create_Q(randn(p), randn(r))
        bench = @elapsed nrmschur(Q, :H, true, ε)# setup = (Q = copy($Q))
        if j > 0
           means[i, j] = bench * 1e3
        end
    end
end

αt = LinRange(0, 0.6, 300)
M = mean(means, dims = 2)
Min = minimum(means, dims=2)
P =  plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = true)
plot!(αs, M, ribbon = std(means, dims= 2), fillalpha = 0.15, label = "Avg. runtime " *L"\pm\ \sigma", lw = 1.5)#, ribbon = stds, fillalpha = 0.2)
plot!(αs, Min, label = "Min. runtime", lw = 1.5, ls = :dashdot)
plot!(αt,    (8/3 .* αt.^3 .+ 5 .* αt .^2 .- αt .+ 14/3)./(14/3) .* M[1], label = L"\propto \frac{8}{3}\alpha^3 + 5\alpha^2-\alpha + \frac{14}{3}", lw = 1.5, color=:darkgreen, ls =:dot) 
plot!(αt,    (8/3 .* αt.^3 .+ 5 .* αt .^2 .- αt .+ 14/3)./(14/3) .* Min[1], label = false, lw = 1.5, color=:darkgreen, ls =:dot) 
#plot!(αt,    (5 .* αt .^2 .- αt .+ 14/3)./(14/3) .* M[1], label = L"\propto 5\alpha^2-\alpha + \frac{14}{3}", lw = 1.5, color=:darkgreen, ls =:dot) 
xlabel!(L"α")
ylabel!("Time [ms]")
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/vary_alpha_n_" * string(n) * "_K_" * string(K) * ".pdf")
savefig(P, path)
display(P)