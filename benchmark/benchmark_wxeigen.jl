using LinearAlgebra, BenchmarkTools, Plots, LaTeXStrings
include("../src/wxeigen.jl")

BLAS.set_num_threads(1)

ms = [8, 16, 32, 64, 128]
timesWX  = zeros(length(ms))
timesSym = zeros(length(ms))
errorsWX   = zeros(length(ms))
errorsSym   = zeros(length(ms))
for (i, m) ∈ enumerate(ms)
    W = randn(m, m); W = (W + W')/2
    X = randn(m, m); X = (X - X')/2
    A = [W -X; X W]
    Y = Symmetric(A)
    E = wxeigen(A)
    M = eigen(A)
    errorsWX[i] = norm(A * E.vectors - E.vectors * Diagonal(E.values))
    errorsSym[i] = norm(A * M.vectors - M.vectors * Diagonal(M.values))
    timesSym[i] = @belapsed eigen($Y)
    timesWX[i]  = @belapsed wxeigen($A)
end

Ptimes =  plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = true, yscale=:log, xscale=:log2, xticks =([16,32,64,128, 256],[ L"2^4",L"2^5", L"2^6", L"2^7", L"2^8"]))
plot!(2ms, timesWX, label = L"\texttt{wxeigen}", color = :blue, linestyle =:solid)
plot!(2ms, timesSym, label = L"\texttt{eigen}\ (\texttt{syevr})", color = :green, linestyle =:dash)
scatter!(2ms, timesWX, label = false, color=:blue)
scatter!(2ms, timesSym, label = false, color=:green)
xlabel!(L"2m")
ylabel!("Running time [s]")

Perrors =  plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = true, yscale=:log, xscale=:log2, xticks =([16,32,64,128,256],[ L"2^4",L"2^5", L"2^6", L"2^7", L"2^8"]))
plot!(2ms, errorsWX, label = L"\texttt{wxeigen}", color = :blue, linestyle =:solid)
plot!(2ms, errorsSym, label = L"\texttt{eigen}\ (\texttt{syevr})", color = :green, linestyle =:dash)
scatter!(2ms, errorsWX, label = false, color=:blue)
scatter!(2ms, errorsSym, label = false, color=:green)
xlabel!(L"2m")
ylabel!(L"\Vert A\breve{Q} - \breve{Q} \tilde{\Lambda}\Vert_\mathrm{F}")
display(Ptimes)
display(Perrors)
script_dir = @__DIR__
path1 = joinpath(script_dir, "../figures/WX_times.pdf")
path2 = joinpath(script_dir, "../figures/WX_errors.pdf")
savefig(Ptimes, path1)
savefig(Perrors, path2)



