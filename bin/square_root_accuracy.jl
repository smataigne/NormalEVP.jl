using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools, Distributions, Plots, LaTeXStrings
include("../src/normal_schur.jl")

n = 100
E = randn(n, n)
E = E + E'
E ./= opnorm(E)
ε = 10*eps(Float64)
δ = sqrt(ε)
C = cos(δ .* E)
S = sin(δ .* E)
A = [S -C; C S]
#A = [C -S; S C]
T, V = nrmschur(A)
display(opnorm(A * V - V * T) / opnorm(A))
display(opnorm(A - A') / 2 / opnorm(A))