using Plots, LaTeXStrings

b(x::Number) = abs(x - 1)

f(x::Number) = abs(x * x - 1)

B(x::Number) = 2b(x) + b(x)^2

x = LinRange(0, 2, 1000)

P = plot(framestyle=:box, legend=:top,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=13,yguidefontsize=15,xguidefontsize=15, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm)
plot!(x, b.(x), label = L"|x-1|",color = :purple, linestyle =:dashdot, linewidth = 2)
plot!(x, f.(x), label = L"|x^2-1|", color = :blue, linestyle =:solid, linewidth = 1.5)
plot!(x, B.(x), label = L"2|x-1| + |x-1|^2", color = :green4, linestyle =:dash, linewidth = 2)




xlabel!(L"x")
display(P)