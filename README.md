# NormalEVP.jl
This repository contains routines for the eigenvalue decomposition of real normal matrices. It is associated with the paper :"Mataigne S., Gallivan K., *The eigenvalue decomposition of normal matrices by the decomposition of the skew-symmetric part with applications to orthogonal matrices, (2024).*

## Requirements
To use this repository, the user must have a Julia installation and have installed the following packages for computations: `LinearAlgebra`, `SkewLinearAlgebra`. And also the following packages to make the plots: `Plots`, `Colors`, `LaTeXStrings`, `XSLX`, `Distributions`, `BenchmarkTools`, `LaTeXStrings`. These packages are easily obtained from the package installation environment as follows. In Julia REPL, press `]` to access the installation environment and for each package, do
```julia
(@v1.6) pkg> add Name_of_Package
```

## Use
* The folder `src` contains the files:
    * `normal_schur.jl` contains the main routines to compute the real Schur decomposition of a normal matrix, notably `nrmschur(A::AbstractMatrix)`.
    * `skewlog.jl` contains the routine `nrmskewlog(A::AbstractMatrix)` to compute the matrix logarithm of an orthogonal matrix with determinant 1.
    * `chase_zeros.jl`contains a routine to isolate the zero diagonal entries of an upper bidiagonal matrix, as described in the folder `Note on zero chasing`.
    * `wxeigen.jl` contains routines to obtain the eigenvalue decomposition of a symmetric matrix $A = [W -X; X W]$ where $W$ is symmetric and $X$ is skew-symmetric.
* The folder `benchmark` contains files to perform all experiments presented in the paper.
* The fodler `figures`contains all figures produced by the files from the `benchmark` folder.
* The folder `Note on zero chasing` briefly explains the routines from `chase_zeros.jl`.