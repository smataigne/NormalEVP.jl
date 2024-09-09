using LinearAlgebra, SkewLinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("../src/normal_schur.jl")

@views function func(Q)
    H = hessenberg(Q)
    M = Matrix(H.Q)
    return
end

@views function create_worstQ(n, odd::Bool)
    N = n ÷ 2
    Q = Matrix(qr(randn(n, n)).Q)
    M = similar(Q, n, n)
    for i ∈ 1:N
        κ = 
        c = rand([-π, π]); s = sqrt(2)
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    if odd
        M[:, end] .= Q[:, end]
    end
    return M * Q'
end

sizes = [10, 34, 100,334, 1000]
ns = length(sizes)
min_myschurH  = zeros(ns); med_myschurH  = zeros(ns); mean_myschurH = zeros(ns)
min_myschurL  = zeros(ns); med_myschurL  = zeros(ns); mean_myschurL = zeros(ns)
min_LAschur  = zeros(ns); med_LAschur  = zeros(ns); mean_LAschur = zeros(ns)
min_LAeig    = zeros(ns); med_LAeig    = zeros(ns); mean_LAeig    = zeros(ns)
min_LAlog    = zeros(ns); med_LAlog    = zeros(ns); mean_LAlog    = zeros(ns)
min_LAhes    = zeros(ns); med_LAhes    = zeros(ns); mean_LAhes    = zeros(ns)

BLAS.set_num_threads(1)
for (i, n) ∈ enumerate(sizes)
    global Q = create_worstQ(n, false)
    res_myschurH = @benchmark schurQ(Q, :H, false)
    res_myschurL = @benchmark schurQ(Q, :L, false)
    res_LAschur  = @benchmark schur(Q)
    #res_LAeig    = @benchmark eigvals(Q)
    res_LAlog    = @benchmark log(Q)
    res_LAhes    = @benchmark func(Q)
    min_myschurH[i]  = minimum(res_myschurH).time / 1000; med_myschurH[i]   = median(res_myschurH).time / 1000; mean_myschurH[i] = mean(res_myschurH).time / 1000
    min_myschurL[i]  = minimum(res_myschurL).time / 1000; med_myschurL[i]   = median(res_myschurL).time / 1000; mean_myschurL[i] = mean(res_myschurL).time / 1000
    min_LAschur[i]  = minimum(res_LAschur).time / 1000; med_LAschur[i]   = median(res_LAschur).time / 1000; mean_LAschur[i] = mean(res_LAschur).time / 1000
    min_LAhes[i]  = minimum(res_LAhes).time / 1000; med_LAhes[i]   = median(res_LAhes).time / 1000; mean_LAhes[i] = mean(res_LAhes).time / 1000
    #min_LAeig[i]  = minimum(res_LAeig).time / 1000; med_LAeig[i]   = median(res_LAeig).time / 1000; mean_LAeig[i] = mean(res_LAeig).time / 1000
    min_LAlog[i]  = minimum(res_LAlog).time / 1000; med_LAlog[i]   = median(res_LAlog).time / 1000; mean_LAlog[i] = mean(res_LAlog).time / 1000
    
    print("Done: " * string(n))
end

Pmin = plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, yscale=:log, xscale=:log, minorgrid=true, yticks =[10^2,10^4,10^6, 10^8])
plot!(sizes, min_myschurH, label = L"\texttt{myschur}" *" (Householder)", color = :blue, linestyle =:solid); scatter!(sizes, min_myschurH, label = false, color = :blue)
plot!(sizes, min_myschurL, label = L"\texttt{myschur}" * " (Lanczos)", color =:red, linestyle=:dash ); scatter!(sizes, min_myschurL, label = false, color =:red )
plot!(sizes, min_LAschur,  label  = L"\texttt{schur}", color = :green, linestyle =:dot);  scatter!(sizes, min_LAschur, label = false, color = :green)
plot!(sizes, min_LAhes,  label  = L"\texttt{hessenberg}+\texttt{dorghr}", color = :purple, linestyle =:dashdot);  scatter!(sizes, min_LAhes, label = false, color = :purple)
#plot!(sizes, min_LAeig,    label = L"\texttt{eigvals}", color =:purple, linestyle=:dashdot); scatter!(sizes, min_LAeig, label = false, color =:purple)
plot!(sizes, min_LAlog,    label = L"\texttt{log}", color =:black, linestyle=:dashdot); scatter!(sizes, min_LAlog, label = false, color=:black)
xlabel!(L"n")
ylabel!("Minimum time [μs]")

Pmed = plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, yscale=:log, xscale=:log, minorgrid=true, yticks =[10^2,10^4,10^6, 10^8])
plot!(sizes, med_myschurH, label = L"\texttt{myschur}" *" (Householder)", color = :blue, linestyle =:solid); scatter!(sizes, med_myschurH, label = false, color = :blue)
plot!(sizes, med_myschurL, label = L"\texttt{myschur}"*" (Lanczos)", color =:red, linestyle=:dash ); scatter!(sizes, med_myschurL, label = false, color =:red )
plot!(sizes, med_LAschur,  label  = L"\texttt{schur}", color = :green, linestyle =:dot);  scatter!(sizes, med_LAschur, label = false, color = :green)
plot!(sizes, med_LAhes,  label  = L"\texttt{hessenberg}+\texttt{dorghr}", color = :purple, linestyle =:dashdot);  scatter!(sizes, med_LAhes, label = false, color = :purple)
#plot!(sizes, med_LAeig,    label = L"\texttt{eigvals}", color =:purple, linestyle=:dashdot); scatter!(sizes, med_LAeig, label = false, color =:purple)
plot!(sizes, med_LAlog,    label = L"\texttt{log}", color =:black, linestyle=:dashdot); scatter!(sizes, med_LAlog, label = false, color=:black)
xlabel!(L"n")
ylabel!("Median time [μs]")

Pmean = plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, yscale=:log, xscale=:log, minorgrid=true, yticks =[10^2,10^4,10^6, 10^8])
plot!(sizes, mean_myschurH, label = L"\texttt{myschur}" *" (Householder)", color = :blue, linestyle =:solid); scatter!(sizes, mean_myschurH, label = false, color = :blue)
plot!(sizes, mean_myschurL, label = L"\texttt{myschur}"*" (Lanczos)", color =:red, linestyle=:dash ); scatter!(sizes, mean_myschurL, label = false, color =:red )
plot!(sizes, mean_LAschur,  label  = L"\texttt{schur}", color = :green, linestyle =:dot);  scatter!(sizes, mean_LAschur, label = false, color = :green)
plot!(sizes, mean_LAhes,  label  = L"\texttt{hessenberg}+\texttt{dorghr}", color = :purple, linestyle =:dashdot);  scatter!(sizes, mean_LAhes, label = false, color = :purple)
#plot!(sizes, mean_LAeig,    label = L"\texttt{eigvals}", color =:purple, linestyle=:dashdot); scatter!(sizes, mean_LAeig, label = false, color =:purple)
plot!(sizes, mean_LAlog,    label = L"\texttt{log}", color =:black, linestyle=:dashdot); scatter!(sizes, mean_LAlog, label = false, color=:black)
xlabel!(L"n")
ylabel!("Mean time [μs]")

script_dir = @__DIR__
pathmin = joinpath(script_dir, "./figures/min_worst2_benchmark.pdf")
pathmed = joinpath(script_dir, "./figures/med_worst2_benchmark.pdf")
pathmean = joinpath(script_dir, "./figures/mean_worst2_benchmark.pdf")
savefig(Pmin, pathmin)
savefig(Pmed, pathmed)
savefig(Pmean, pathmean)
display(Pmean)