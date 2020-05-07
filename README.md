# ThreadedSparseArrays.jl

[![Build Status](https://travis-ci.com/jagot/ThreadedSparseArrays.jl.svg?branch=master)](https://travis-ci.com/jagot/ThreadedSparseArrays.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jagot/ThreadedSparseArrays.jl?svg=true)](https://ci.appveyor.com/project/jagot/ThreadedSparseArrays-jl)
[![Codecov](https://codecov.io/gh/jagot/ThreadedSparseArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jagot/ThreadedSparseArrays.jl)

Simple package providing a wrapper type enabling threaded sparse
matrix–dense matrix multiplication. Based on [this
PR](https://github.com/JuliaLang/julia/pull/29525).

## Installation
ThreadedSparseArrays.jl is not yet a registered package, but you can
install it with:
```
] add git@github.com:jagot/ThreadedSparseArrays.jl.git
```

Note that you *must* enable threading in Julia for
ThreadedSparseArrays to work. You can do so by setting the
[JULIA_NUM_THREADS](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1)
environment variable. To test that it is set properly, run
```julia
Threads.nthreads()
```
and make sure it returns the number of threads you wanted.


## Example usage
To use ThreadedSparseArrays, all you need to do is to wrap your sparse
matrix using the ThreadedSparseMatrixCSC type, like this:
```julia
using SparseArrays
using ThreadedSparseArrays

A = sprand(10000, 100, 0.05); # sparse
X1 = randn(100, 100); # dense
X2 = randn(10000, 100); # dense

At = ThreadedSparseMatrixCSC(A); # threaded version

# threaded sparse matrix–dense matrix multiplication
B1 = At*X1;
B2 = At'X2;
```

## Notes
* If the right hand side `X` is a `Vector`, you need to use `At'X` to
get threading. `At*X` will not work.
* You might only get speedups for large matrices. Use `@btime` from
the [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
package to check if your use case is improved.
