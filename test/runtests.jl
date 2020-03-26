using ThreadedSparseArrays
using SparseArrays
using LinearAlgebra
using Test

@testset "ThreadedSparseArrays.jl" begin
    N = 4000
    n = 200
    T = ComplexF64

    C = sprand(T, N, n, 0.05)
    @testset "$(Mat)_R" for Mat in [ThreadedSparseMatrixCSC, ThreadedColumnizedSparseMatrix]
        Ct = Mat(C)

        eye = Matrix(one(T)*I, N, N)
        out = zeros(T, N, n)
        LinearAlgebra.mul!(out, eye, Ct)
        ref = eye*C
        @test norm(ref-out) == 0
    end

    @testset "$(Mat)_L" for Mat in [ThreadedSparseMatrixCSC]
        Ct = Mat(C)

        eye = Matrix(one(T)*I, n, n)
        out = zeros(T, N, n)
        LinearAlgebra.mul!(out, Ct, eye)
        ref = C*eye
        @test norm(ref-out) == 0
    end

    @testset "$(Mat)_L_$(op)" for op in [adjoint,transpose], Mat in [ThreadedSparseMatrixCSC]
        Ct = Mat(C)

        eye = Matrix(one(T)*I, N, N)
        out = zeros(T, n, N)
        LinearAlgebra.mul!(out, op(Ct), eye)
        ref = op(C)*eye
        @test norm(ref-out) == 0
    end

end
