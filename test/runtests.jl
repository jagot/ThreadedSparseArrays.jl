using ThreadedSparseArrays
using SparseArrays
using LinearAlgebra
using Random
using StableRNGs
using Test

@static if VERSION < v"1.6.0"
    parentmodule(x) = x.linfo.def.module
end

function match_exception(f, ::Type{T}=DimensionMismatch, func=r"^mul!$", m=ThreadedSparseArrays) where T
    try
        f()
    catch ex
        st = stacktrace(catch_backtrace())[1]
        return ex isa T && match(func,string(st.func))!==nothing && parentmodule(st) === m
    end
    false
end



function rand_dense(rng,::Type{ComplexF64}, N, n)
    M = max(N,n)
    Matrix(sparse(randperm(rng,M),1:M,one(ComplexF64)))[1:N,1:n]
end
rand_dense(rng,::Type{ComplexF64}, N) = (x=zeros(ComplexF64,N); x[rand(rng,1:N)] = one(ComplexF64); x)
rand_dense(rng,::Type{Complex{Int}}, sz...) = rand(rng,0:5,sz...) .+ im*rand(rng,0:5,sz...)

rand_sparse(rng,::Type{ComplexF64}, N, n, p)   = sprand(rng,ComplexF64,N,n,p)
rand_sparse(rng,::Type{Complex{Int}}, N, n, p) = sprand(N,n,p,x->Complex{Int}.(rand(rng,0:5,x),rand(rng,0:5,x)))

rand_scalar(rng,::Type{T}) where T<:Complex = T(rand(rng,2 .^ (1:5)) + im*rand(rng,2 .^ (1:5)))



@testset "ThreadedSparseArrays.jl" begin
    # Test that all combinations of dense*sparse multiplication are threaded
    @testset "Threading" begin
        A = ThreadedSparseMatrixCSC(spzeros(2,3))
        B = zeros(4,5)
        @test match_exception(()->A*B)
        @test match_exception(()->A'B)
        @test match_exception(()->A*B')
        @test match_exception(()->A'B')
        @test match_exception(()->B*A)
        @test match_exception(()->B'A)
        @test match_exception(()->B*A')
        @test match_exception(()->B'A')
    end

    # These test below are here to get the right fallback for sparse times sparse.
    # The implementations are not (currently) threaded.
    @testset "Fallbacks" begin
        A = ThreadedSparseMatrixCSC(spzeros(2,3))
        B = spzeros(4,5)
        @test match_exception(()->A*B,  DimensionMismatch, r"", SparseArrays)
        @test match_exception(()->A'B,  DimensionMismatch, r"", SparseArrays)
        @test match_exception(()->A*B', DimensionMismatch, r"", SparseArrays)
        @test match_exception(()->A'B', DimensionMismatch, r"", SparseArrays)
        @test match_exception(()->B*A,  DimensionMismatch, r"", SparseArrays)
        @test match_exception(()->B'A,  DimensionMismatch, r"", SparseArrays)
        @test match_exception(()->B*A', DimensionMismatch, r"", SparseArrays)
        @test match_exception(()->B'A', DimensionMismatch, r"", SparseArrays)
    end


    @testset "ReturnType_$(op1)_$(op2)" for op1 in [identity,adjoint,transpose], op2 in [identity,adjoint,transpose]
        rng = StableRNG(1234)
        A = rand_sparse(rng,Complex{Int64},10,10,0.4)
        B = rand_sparse(rng,Complex{Int64},10,10,0.4)
        ref = op1(A)*op2(B)
        out = op1(ThreadedSparseMatrixCSC(A))*op2(ThreadedSparseMatrixCSC(B))
        @test out isa ThreadedSparseMatrixCSC
        @test out == ref
        out = op1(ThreadedSparseMatrixCSC(A))*op2(B)
        @test out isa SparseMatrixCSC
        @test out == ref
        out = op1(A)*op2(ThreadedSparseMatrixCSC(B))
        @test out isa SparseMatrixCSC
        @test out == ref
    end

    @testset "copy_$op" for op in [identity,adjoint,transpose]
        rng = StableRNG(1234)
        A = rand_sparse(rng,Complex{Int64},8,10,0.4)
        out = copy(op(ThreadedSparseMatrixCSC(A)))
        @test out isa ThreadedSparseMatrixCSC
        @test out == op(A)
        out = permutedims(op(ThreadedSparseMatrixCSC(A)))
        @test out isa ThreadedSparseMatrixCSC
        @test out == permutedims(op(A))
    end

    N = 1000
    n = 200
    @testset "$T" for T in (ComplexF64, Complex{Int64})
        rng = StableRNG(1234)
        C = rand_sparse(rng,T,N,n,0.05)

        @testset "(α,β)=$αβ" for αβ in ((), (rand_scalar(rng,T), zero(T)), (zero(T),rand_scalar(rng,T)), (rand_scalar(rng,T),rand_scalar(rng,T)))
            @testset "R_$(op)" for op in [identity,adjoint,transpose]
                Ct = op(ThreadedSparseMatrixCSC(C))
                M = size(Ct,1)

                X = rand_dense(rng,T,M,M)

                out = zeros(T, size(Ct))
                LinearAlgebra.mul!(out, X, Ct, αβ...)
                ref = zeros(T, size(op(C)))
                LinearAlgebra.mul!(ref, X, op(C), αβ...)
                @test out == ref
            end

            @testset "L_$(op)" for op in [identity,adjoint,transpose]
                Ct = op(ThreadedSparseMatrixCSC(C))
                m = size(Ct,2)

                X = rand_dense(rng,T,m,m)

                out = zeros(T, size(Ct))
                LinearAlgebra.mul!(out, Ct, X, αβ...)
                ref = zeros(T, size(op(C)))
                LinearAlgebra.mul!(ref, op(C), X, αβ...)
                @test out == ref
            end

            x = rand_dense(rng,T,N)
            @testset "L_$(op)_vec" for op in [adjoint,transpose]
                Ct = op(ThreadedSparseMatrixCSC(C))

                out = zeros(T, n)
                LinearAlgebra.mul!(out, Ct, x, αβ...)
                ref = zeros(T, n)
                LinearAlgebra.mul!(ref, op(C), x, αβ...)
                @test out == ref
            end
        end
    end
end
