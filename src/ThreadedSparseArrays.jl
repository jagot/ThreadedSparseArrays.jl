module ThreadedSparseArrays


export ThreadedSparseMatrixCSC


using LinearAlgebra
import LinearAlgebra: mul!
using SparseArrays
import SparseArrays: getcolptr, AbstractSparseMatrixCSC
const AdjOrTransDenseMatrix = if VERSION < v"1.6.0-rc2"
    SparseArrays.AdjOrTransStridedOrTriangularMatrix
else
    SparseArrays.AdjOrTransDenseMatrix
end

# * Threading utilities
struct RangeIterator
    k::Int
    d::Int
    r::Int
end

"""
    RangeIterator(n::Int,k::Int)

Returns an iterator splitting the range `1:n` into `min(k,n)` parts of (almost) equal size.
"""
RangeIterator(n::Int, k::Int) = RangeIterator(min(n,k),divrem(n,k)...)
Base.length(it::RangeIterator) = it.k
endpos(it::RangeIterator, i::Int) = i*it.d+min(i,it.r)
Base.iterate(it::RangeIterator, i::Int=1) = i>it.k ? nothing : (endpos(it,i-1)+1:endpos(it,i), i+1)


# * ThreadedSparseMatrixCSC

"""
    ThreadedSparseMatrixCSC(A)

Thin container around `A::SparseMatrixCSC` that will enable certain
threaded multiplications of `A` with dense matrices.
"""
struct ThreadedSparseMatrixCSC{Tv,Ti,At} <: AbstractSparseMatrixCSC{Tv,Ti}
    A::At
    ThreadedSparseMatrixCSC(A::At) where {Tv,Ti,At<:AbstractSparseMatrixCSC{Tv,Ti}} =
        new{Tv,Ti,At}(A)
end

Base.size(A::ThreadedSparseMatrixCSC, args...) = size(A.A, args...)

for f in [:rowvals, :nonzeros, :getcolptr]
    @eval SparseArrays.$(f)(A::ThreadedSparseMatrixCSC) = SparseArrays.$(f)(A.A)
end


@static if v"1.7.0" <= VERSION < v"1.8.0-"
    SparseArrays._goodbuffers(A::ThreadedSparseMatrixCSC) = SparseArrays._goodbuffers(A.A)
    SparseArrays._checkbuffers(A::ThreadedSparseMatrixCSC) = SparseArrays._checkbuffers(A.A)
end

Base.copy(A::Adjoint{<:Any,<:ThreadedSparseMatrixCSC}) = ThreadedSparseMatrixCSC(copy(A.parent.A'))
Base.copy(A::Transpose{<:Any,<:ThreadedSparseMatrixCSC}) = ThreadedSparseMatrixCSC(copy(transpose(A.parent.A)))
Base.permutedims(A::ThreadedSparseMatrixCSC, (a,b))  = ThreadedSparseMatrixCSC(permutedims(A.A, (a,b)))


# sparse * sparse multiplications are not (currently) threaded, but we want to keep the return type
for (T1,t1) in ((ThreadedSparseMatrixCSC,identity), (Adjoint{<:Any,<:ThreadedSparseMatrixCSC},adjoint), (Transpose{<:Any,<:ThreadedSparseMatrixCSC},transpose))
    for (T2,t2) in ((ThreadedSparseMatrixCSC,identity), (Adjoint{<:Any,<:ThreadedSparseMatrixCSC},adjoint), (Transpose{<:Any,<:ThreadedSparseMatrixCSC},transpose))
        @eval Base.:(*)(A::$T1, B::$T2) = ThreadedSparseMatrixCSC($t1($t1(A).A)*$t2($t2(B).A))
    end
end


function mul!(C::StridedVecOrMat, A::ThreadedSparseMatrixCSC, B::Union{StridedVector,AdjOrTransDenseMatrix}, α::Number, β::Number)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = nonzeros(A)
    rv = rowvals(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    @sync for r in RangeIterator(size(C,2), Threads.nthreads())
        Threads.@spawn for k in r
            @inbounds for col in 1:size(A, 2)
                αxj = B[col,k] * α
                for j in nzrange(A, col)
                    C[rv[j], k] += nzv[j]*αxj
                end
            end
        end
    end
    C
end

for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
    @eval function mul!(C::StridedVecOrMat, xA::$T{<:Any,<:ThreadedSparseMatrixCSC}, B::AdjOrTransDenseMatrix, α::Number, β::Number)
        A = xA.parent
        size(A, 2) == size(C, 1) || throw(DimensionMismatch())
        size(A, 1) == size(B, 1) || throw(DimensionMismatch())
        size(B, 2) == size(C, 2) || throw(DimensionMismatch())
        nzv = nonzeros(A)
        rv = rowvals(A)
        if β != 1
            β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
        end
        @sync for r in RangeIterator(size(C,2), Threads.nthreads())
            Threads.@spawn for k in r
                @inbounds for col in 1:size(A, 2)
                    tmp = zero(eltype(C))
                    for j in nzrange(A, col)
                        tmp += $t(nzv[j])*B[rv[j],k]
                    end
                    C[col,k] += tmp * α
                end
            end
        end
        C
    end

    @eval function mul!(C::StridedVecOrMat, xA::$T{<:Any,<:ThreadedSparseMatrixCSC}, B::StridedVector, α::Number, β::Number)
        A = xA.parent
        size(A, 2) == size(C, 1) || throw(DimensionMismatch())
        size(A, 1) == size(B, 1) || throw(DimensionMismatch())
        size(B, 2) == size(C, 2) || throw(DimensionMismatch())
        @assert size(B,2)==1
        nzv = nonzeros(A)
        rv = rowvals(A)
        if β != 1
            β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
        end
        @sync for r in RangeIterator(size(A,2), Threads.nthreads())
            Threads.@spawn @inbounds for col in r
                tmp = zero(eltype(C))
                for j in nzrange(A, col)
                    tmp += $t(nzv[j])*B[rv[j]]
                end
                C[col] += tmp * α
            end
        end
        C
    end
end

function mul!(C::StridedVecOrMat, X::AdjOrTransDenseMatrix, A::ThreadedSparseMatrixCSC, α::Number, β::Number)
    mX, nX = size(X)
    nX == size(A, 1) || throw(DimensionMismatch())
    mX == size(C, 1) || throw(DimensionMismatch())
    size(A, 2) == size(C, 2) || throw(DimensionMismatch())
    rv = rowvals(A)
    nzv = nonzeros(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    # TODO: split in X isa DenseMatrixUnion and X isa Adjoint/Transpose so we can use @simd in the first case (see original code in SparseArrays)
    @sync for r in RangeIterator(size(A,2), Threads.nthreads())
        Threads.@spawn for col in r
            @inbounds for k in nzrange(A, col)
                Aiα = nzv[k] * α
                rvk = rv[k]
                for multivec_row in 1:mX
                    C[multivec_row, col] += X[multivec_row, rvk] * Aiα
                end
            end
        end
    end
    C
end

for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
    @eval function mul!(C::StridedVecOrMat, X::AdjOrTransDenseMatrix, xA::$T{<:Any,<:ThreadedSparseMatrixCSC}, α::Number, β::Number)
        A = xA.parent
        mX, nX = size(X)
        nX == size(A, 2) || throw(DimensionMismatch())
        mX == size(C, 1) || throw(DimensionMismatch())
        size(A, 1) == size(C, 2) || throw(DimensionMismatch())
        rv = rowvals(A)
        nzv = nonzeros(A)
        if β != 1
            β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
        end

        # transpose of Threaded * Dense algorithm above
        @sync for r in RangeIterator(size(C,1), Threads.nthreads())
            Threads.@spawn for k in r
                @inbounds for col in 1:size(A, 2)
                    αxj = X[k,col] * α
                    for j in nzrange(A, col)
                        C[k, rv[j]] += $t(nzv[j])*αxj
                    end
                end
            end
        end
        C
    end
end

end # module
