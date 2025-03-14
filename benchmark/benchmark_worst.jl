using LinearAlgebra, SkewLinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("../src/normal_schur.jl")
include("../src/bold_normal_schur.jl")
"""
This file implements a benchmark of nrmschur(A::AbstractMatrix) where A yields the worst-case complexity.
"""

@views function func(Q)
    H = hessenberg(Q)
    M = Matrix(H.Q)
    return
end

@views function create_worstQ(n, odd::Bool)
    N = n ÷ 2
    QR = qr(randn(n,n))
    Q = Matrix(QR.Q) * Diagonal(sign.(diag(QR.R)))
    M = similar(Q, n, n)
    for i ∈ 1:N
        c = randn(); s = sqrt(2)
        #c = rand([-1, 1]) * cos(θ); s = sin(θ)
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    if odd
        M[:, end] .= Q[:, end]
    end
    return M * Q'
end

sizes = [10, 32, 100, 316, 1000]
ns = length(sizes)
min_nrmschurH  = zeros(ns); med_nrmschurH  = zeros(ns); mean_nrmschurH = zeros(ns)
min_ortho  = zeros(ns); med_ortho = zeros(ns); mean_ortho = zeros(ns)
min_LAschur  = zeros(ns); med_LAschur  = zeros(ns); mean_LAschur = zeros(ns)
min_LAhes    = zeros(ns); med_LAhes    = zeros(ns); mean_LAhes    = zeros(ns)

BLAS.set_num_threads(1)
for (i, n) ∈ enumerate(sizes)
    global Q = create_worstQ(n, false)
    ε = 10 * eps(Float64)
    res_nrmschurH = @benchmark nrmschur(Q, :H, false)
    res_ortho    = @benchmark nrmschur2(Q)
    res_LAschur  = @benchmark schur(Q)
    res_LAhes    = @benchmark func(Q)
    min_nrmschurH[i] = minimum(res_nrmschurH).time / 1000; med_nrmschurH[i]   = median(res_nrmschurH).time / 1000; mean_nrmschurH[i] = mean(res_nrmschurH).time / 1000
    min_ortho[i] = minimum(res_ortho).time / 1000; med_ortho[i]   = median(res_ortho).time / 1000; mean_ortho[i] = mean(res_ortho).time / 1000
    min_LAschur[i]  = minimum(res_LAschur).time / 1000; med_LAschur[i]   = median(res_LAschur).time / 1000; mean_LAschur[i] = mean(res_LAschur).time / 1000
    min_LAhes[i]    = minimum(res_LAhes).time / 1000; med_LAhes[i]   = median(res_LAhes).time / 1000; mean_LAhes[i] = mean(res_LAhes).time / 1000
   print("Done: n = " * string(n) * "\n")
end

Pmin = plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, yscale=:log, xscale=:log, minorgrid=false, yticks =[10^2,10^4,10^6, 10^8])
plot!(sizes, min_nrmschurH, label = L"\texttt{nrmschur}" *" (Householder)", color = :blue, linestyle =:solid); scatter!(sizes, min_nrmschurH, label = false, color = :blue)
#plot!(sizes, min_ortho, label = L"QR [44] (eigenvalues)", color =:red, linestyle=:dash ); scatter!(sizes, min_ortho, label = false, color =:red )
#plot!(sizes, min_nrmschurL, label = L"\texttt{nrmschur}" * " (Lanczos)", color =:red, linestyle=:dash ); scatter!(sizes, min_nrmschurL, label = false, color =:red )
plot!(sizes, min_LAschur,  label  = L"\texttt{schur}", color = :green, linestyle =:dot);  scatter!(sizes, min_LAschur, label = false, color = :green)
plot!(sizes, min_LAhes,  label  = L"\texttt{hessenberg}+\texttt{orghr}", color = :purple, linestyle =:dashdot);  scatter!(sizes, min_LAhes, label = false, color = :purple)
#plot!(sizes, min_LAeig,    label = L"\texttt{eigvals}", color =:purple, linestyle=:dashdot); scatter!(sizes, min_LAeig, label = false, color =:purple)
#plot!(sizes, min_LAlog,    label = L"\texttt{log}", color =:black, linestyle=:dashdot); scatter!(sizes, min_LAlog, label = false, color=:black)
xlabel!(L"n")
ylabel!("Minimum time [μs]")

Pmed = plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, yscale=:log, xscale=:log, minorgrid=true, yticks =[10^2,10^4,10^6, 10^8])
plot!(sizes, med_nrmschurH, label = L"\texttt{nrmschur}" *" (Householder)", color = :blue, linestyle =:solid); scatter!(sizes, med_nrmschurH, label = false, color = :blue)
#plot!(sizes, med_ortho, label = L"QR [44] (eigenvalues)", color =:red, linestyle=:dash ); scatter!(sizes, med_ortho, label = false, color =:red )
#plot!(sizes, med_nrmschurL, label = L"\texttt{nrmschur}"*" (Lanczos)", color =:red, linestyle=:dash ); scatter!(sizes, med_nrmschurL, label = false, color =:red )
plot!(sizes, med_LAschur,  label  = L"\texttt{schur}", color = :green, linestyle =:dot);  scatter!(sizes, med_LAschur, label = false, color = :green)
plot!(sizes, med_LAhes,  label  = L"\texttt{hessenberg}+\texttt{orghr}", color = :purple, linestyle =:dashdot);  scatter!(sizes, med_LAhes, label = false, color = :purple)
#plot!(sizes, med_LAeig,    label = L"\texttt{eigvals}", color =:purple, linestyle=:dashdot); scatter!(sizes, med_LAeig, label = false, color =:purple)
#plot!(sizes, med_LAlog,    label = L"\texttt{log}", color =:black, linestyle=:dashdot); scatter!(sizes, med_LAlog, label = false, color=:black)
xlabel!(L"n")
ylabel!("Median time [μs]")

Pmean = plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=15,xguidefontsize=15, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, yscale=:log, xscale=:log, minorgrid=false, yticks =[10^2,10^4,10^6, 10^8])
plot!(sizes, mean_LAhes,  label  = L"\texttt{hessenberg}\ (\texttt{gehrd})+\texttt{orghr}", color = :purple, linestyle =:dashdot);  scatter!(sizes, mean_LAhes, label = false, color = :purple)
plot!(sizes, mean_nrmschurH, label = L"\texttt{nrmschur}" * " (Algorithm 3.1)", color = :blue, linestyle =:solid); scatter!(sizes, mean_nrmschurH, label = false, color = :blue)
plot!(sizes, mean_ortho, label = L"\texttt{nrmschur2}" * " (with skew-Ham. solver)", color =:red, linestyle=:dash ); scatter!(sizes, mean_ortho, label = false, color =:red )
#plot!(sizes, mean_nrmschurL, label = L"\texttt{nrmschur}"*" (Lanczos)", color =:red, linestyle=:dash ); scatter!(sizes, mean_nrmschurL, label = false, color =:red )
plot!(sizes, mean_LAschur,  label  = L"\texttt{schur}\ (\texttt{gees})", color = :green, linestyle =:dot);  scatter!(sizes, mean_LAschur, label = false, color = :green)

#plot!(sizes, mean_LAeig,    label = L"\texttt{eigvals}", color =:purple, linestyle=:dashdot); scatter!(sizes, mean_LAeig, label = false, color =:purple)
#plot!(sizes, mean_LAlog,    label = L"\texttt{log}", color =:black, linestyle=:dashdot); scatter!(sizes, mean_LAlog, label = false, color=:black)
plot!(sizes[2:end], 14/3 .* (sizes[2:end]).^3 .* 0.00005, label =L"\mathcal{O}(n^3)", color=:black, linestyle=:solid)
xlabel!(L"n")
ylabel!("Average time [μs]")

script_dir = @__DIR__
pathmin = joinpath(script_dir, "../figures/min_worst2_benchmark.pdf")
pathmed = joinpath(script_dir, "../figures/med_worst2_benchmark.pdf")
pathmean = joinpath(script_dir, "../figures/mean_worst2_benchmark.pdf")
#savefig(Pmin, pathmin)
#savefig(Pmed, pathmed)
#savefig(Pmean, pathmean)
display(Pmean)