using LinearAlgebra, BenchmarkTools, Plots, LaTeXStrings
include("../src/wxeigen.jl")

BLAS.set_num_threads(1)
J = 100
ms = [8, 16, 32, 64, 128, 256]

timesWX  = zeros(length(ms)) .+1
timesWXfull= zeros(length(ms)) .+1
timesSym = zeros(length(ms)) .+1

errorsWX   = zeros(length(ms))
errorsWXfull   = zeros(length(ms))
errorsSym   = zeros(length(ms))
nn= []
for (i, m) ∈ enumerate(ms)
    for j ∈ 1:J
        W = randn(m, m); W = (W + W')/2
        X = randn(m, m); X = (X - X')/2
        A = [W -X; X W]
        Y = Symmetric(A)
        E = wxeigen(A, :L)
        Efull = wxeigen(A, :Lfull)
        M = eigen(A)
        errorsWX[i]     = (errorsWX[i] * (j - 1) + opnorm(A * E.vectors - E.vectors * Diagonal(E.values)) / opnorm(A)) / j
        errorsWXfull[i] = (errorsWXfull[i] * (j - 1) + opnorm(A * Efull.vectors - Efull.vectors * Diagonal(Efull.values)) / opnorm(A)) / j
        errorsSym[i]    = (errorsSym[i] * (j - 1) + opnorm(A * M.vectors - M.vectors * Diagonal(M.values)) / opnorm(A)) / j
    end
    W = randn(m, m); W = (W + W')/2
    X = randn(m, m); X = (X - X')/2
    A = [W -X; X W]
    append!(nn, opnorm(A))
    Y = Symmetric(A)
    timesSym[i]    = @belapsed eigen($Y)
    timesWX[i]     = @belapsed wxeigen($A, :L)
    timesWXfull[i] = @belapsed wxeigen($A, :Lfull)
end

Ptimes =  plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = false, yscale=:log, xscale=:log2, xticks =([16,32,64,128, 256, 512], [ L"2^4",L"2^5", L"2^6", L"2^7", L"2^8", L"2^9"]),
yticks =([1e-5, 1e-4, 1e-3, 1e-2, 1e-1],[L"10^{-5}", L"10^{-4}",L"10^{-3}", L"10^{-2}",L"10^{-1}", L"10^{-0}"]), ylims=(0.5*1e-5, 4*0.1))
plot!(2ms, timesWX, label = L"\texttt{wxeigen}" * " - No re-orth.", color = :blue, linestyle =:solid, markershape=:circle)
plot!(2ms, timesWXfull, label = L"\texttt{wxeigen}" * " - Full re-orth.", color = :red, linestyle =:dashdot, markershape=:diamond)
plot!(2ms, timesSym, label = L"\texttt{eigen}\ (\texttt{syevr})", color = :green, linestyle =:dash, markershape=:utriangle)
xlabel!(L"2m")
ylabel!("Running time [s]")

Perrors =  plot(framestyle=:box,legend=:right,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = false, minorgridalpha=0.01, yscale=:log10, xscale=:log2, xticks =([16,32,64,128,256,512], [ L"2^4",L"2^5", L"2^6", L"2^7", L"2^8", L"2^9"]),
yticks =([1, 1e-5, 1e-10, 1e-15],[ L"10^0",L"10^{-5}", L"10^{-10}", L"10^{-15}"]), ylims= (1e-16, 10))
plot!(2ms, errorsWX, label = L"\texttt{wxeigen}" * " - No re-orth.", color = :blue, linestyle =:solid, markershape=:circle)
plot!(2ms, errorsWXfull, label = L"\texttt{wxeigen}" * " - Full re-orth.", color = :red, linestyle =:dashdot, markershape=:diamond)
plot!(2ms, errorsSym, label = L"\texttt{eigen}\ (\texttt{syevr})", color = :green, linestyle =:dash, markershape=:utriangle)
xlabel!(L"2m")
ylabel!(L"\Vert A\breve{Q} - \breve{Q} \tilde{\Lambda}\Vert_\mathrm{2} / \Vert A \Vert_\mathrm{2} ")
display(Ptimes)
display(Perrors)

script_dir = @__DIR__
path1 = joinpath(script_dir, "../figures/WX_times_2.pdf")
path2 = joinpath(script_dir, "../figures/WX_errors_2.pdf")
savefig(Ptimes, path1)
savefig(Perrors, path2)




