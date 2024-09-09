using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Random, Plots, LaTeXStrings
include("../src/normal_schur.jl")

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

BLAS.set_num_threads(1)

n = 1000
αs = Vector(0:0.05:0.5)
N = length(αs)
K = 50
means = zeros(N, K - 1)
for j ∈ 1:K
    for i ∈ 1:N
        α = αs[i]
        r = floor(Int, n * α / 2)
        p = (n - 2r) ÷ 2
        Q = create_Q(shuffle!(vcat(π * ones(r), randn(p))), false)
        bench = @elapsed schurQ(Q, :H, true)
        if j > 1
            means[i, j - 1] = bench * 1e3
        end
    end
end

αt = LinRange(0, 0.5, 200)
M = mean(means, dims = 2)
P =  plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = true)
plot!(αs, M, ribbon = std(means, dims= 2), fillalpha = 0.2, label = "Avg. runtime " *L"\pm\ \sigma", lw = 1.5)#, ribbon = stds, fillalpha = 0.2)
plot!(αt,    (8/3 .* αt.^3 .+ 5 .* αt .^2 .- αt .+ 14/3)./(14/3) .* M[1], label = L"\propto \frac{8}{3}\alpha^3 + 5\alpha^2-\alpha + \frac{14}{3}", lw = 1.5, color=:darkgreen, ls =:dot) 
xlabel!(L"α")
ylabel!("Time [ms]")
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/vary_alpha_n_"*string(n)*"_K_"*string(K)*".pdf")
savefig(P, path)
display(P)