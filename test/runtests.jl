using ThreadedSparseArrays
using SparseArrays
using LinearAlgebra
using Test


function match_exception(f, ::Type{T}=DimensionMismatch, func=:mul!, path="ThreadedSparseArrays.jl") where T
    try
        f()
    catch ex
        st = stacktrace(catch_backtrace())[1]
        p = splitpath(path)
        p2 = splitpath(string(st.file))
        return ex isa T && st.func == func && p==p2[max(1,end-length(p)+1):end]
    end
    false
end


@testset "ThreadedSparseArrays.jl" begin
    N = 4000
    n = 200
    T = ComplexF64

    C = sprand(T, N, n, 0.05)
    @testset "R" begin
        Ct = ThreadedSparseMatrixCSC(C)

        eye = Matrix(one(T)*I, N, N)
        out = zeros(T, N, n)
        LinearAlgebra.mul!(out, eye, Ct)
        ref = eye*C
        @test norm(ref-out) == 0
    end

    @testset "L" begin
        Ct = ThreadedSparseMatrixCSC(C)

        eye = Matrix(one(T)*I, n, n)
        out = zeros(T, N, n)
        LinearAlgebra.mul!(out, Ct, eye)
        ref = C*eye
        @test norm(ref-out) == 0
    end

    @testset "L_$(op)" for op in [adjoint,transpose]
        Ct = ThreadedSparseMatrixCSC(C)

        eye = Matrix(one(T)*I, N, N)
        out = zeros(T, n, N)
        LinearAlgebra.mul!(out, op(Ct), eye)
        ref = op(C)*eye
        @test norm(ref-out) == 0
    end

    x = rand(0:1,N)
    @testset "L_$(op)_vec" for op in [adjoint,transpose]
        Ct = ThreadedSparseMatrixCSC(C)

        out = zeros(T, n)
        LinearAlgebra.mul!(out, op(Ct), x)
        ref = op(C)*x
        @test norm(ref-out) == 0
    end


    # Test that all combinations of dense*sparse multiplication are threaded
    @testset "Threading" begin
        A = ThreadedSparseMatrixCSC(spzeros(2,3))
        B = zeros(4,1)
        @test match_exception(()->A*B)
        @test match_exception(()->A'B)
        @test match_exception(()->A*B')
        @test match_exception(()->A'B')
        @test match_exception(()->B*A)
        @test match_exception(()->B'A)
        # @test match_exception(()->B*A') # TODO: implement!
        # @test match_exception(()->B'A') # TODO: implement!
    end


    # These test below are here to ensure we don't hit ambiguity warnings.
    # The implementations are not (currently) threaded.
    sx = sprand(Bool,n,0.05)
    @testset "L_sparsevec" begin
        Ct = ThreadedSparseMatrixCSC(C)

        out = Ct*sx
        ref = C*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end

    sx = sprand(Bool,N,0.05)
    @testset "L_$(op)_sparsevec" for op in [adjoint,transpose]
        Ct = ThreadedSparseMatrixCSC(C)

        out = op(Ct)*sx
        ref = op(C)*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end

    sx = sparse(rand(1:n,10),1:10,true,n,10)
    @testset "L_sparse" begin
        Ct = ThreadedSparseMatrixCSC(C)

        out = Ct*sx
        ref = C*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end

    sx = sparse(rand(1:N,10),1:10,true,N,10)
    @testset "L_$(op)_sparse" for op in [adjoint,transpose]
        Ct = ThreadedSparseMatrixCSC(C)

        out = op(Ct)*sx
        ref = op(C)*sx
        @test norm(ref-out) == 0
        @test typeof(ref)==typeof(out)
    end
end
