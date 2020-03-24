using ThreadedSparseArrays
using SparseArrays
using LinearAlgebra
using Test

@testset "ThreadedSparseArrays.jl" begin
    M,N = 5000,4000
    n = 200
    T = ComplexF64

    C = sprand(T, N, n, 0.05)
    @testset "$(Mat)" for Mat in [ThreadedSparseMatrixCSC, ThreadedColumnizedSparseMatrix]
        Ct = Mat(C)

        eye = Matrix(one(T)*I, N, N)
        out = zeros(T, N, n)
        LinearAlgebra.mul!(out, eye, Ct)
        ref = eye*C
        @test norm(ref-out) == 0
    end
end
