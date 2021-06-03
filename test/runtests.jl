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

    x = rand(0:1,N)
    @testset "$(Mat)_L_$(op)_vec" for op in [adjoint,transpose], Mat in [ThreadedSparseMatrixCSC]
        Ct = Mat(C)

        out = zeros(T, n)
        LinearAlgebra.mul!(out, op(Ct), x)
        ref = op(C)*x
        @test norm(ref-out) == 0
    end


    # These test below are here to ensure we don't hit ambiguity warnings.
    # The implementations are not (currently) threaded.
    sx = sprand(Bool,n,0.05)
    @testset "$(Mat)_L_sparsevec" for Mat in [ThreadedSparseMatrixCSC]
        Ct = Mat(C)

        out = Ct*sx
        ref = C*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end

    sx = sprand(Bool,N,0.05)
    @testset "$(Mat)_L_$(op)_sparsevec" for op in [adjoint,transpose], Mat in [ThreadedSparseMatrixCSC]
        Ct = Mat(C)

        out = op(Ct)*sx
        ref = op(C)*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end

    sx = sparse(rand(1:n,10),1:10,true,n,10)
    @testset "$(Mat)_L_sparse" for Mat in [ThreadedSparseMatrixCSC]
        Ct = Mat(C)

        out = Ct*sx
        ref = C*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end

    sx = sparse(rand(1:N,10),1:10,true,N,10)
    @testset "$(Mat)_L_$(op)_sparse" for op in [adjoint,transpose], Mat in [ThreadedSparseMatrixCSC]
        Ct = Mat(C)

        out = op(Ct)*sx
        ref = op(C)*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end

end
