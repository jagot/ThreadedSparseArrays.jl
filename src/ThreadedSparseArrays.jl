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
            @inbounds for col = 1:size(A, 2)
                αxj = B[col,k] * α
                for j = getcolptr(A)[col]:(getcolptr(A)[col + 1] - 1)
                    C[rv[j], k] += nzv[j]*αxj
                end
            end
        end
    end
    C
end

function mul!(C::StridedVecOrMat, adjA::Adjoint{<:Any,<:ThreadedSparseMatrixCSC}, B::AdjOrTransDenseMatrix, α::Number, β::Number)
    A = adjA.parent
    size(A, 2) == size(C, 1) || throw(DimensionMismatch())
    size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    colptrA = getcolptr(A)
    nzv = nonzeros(A)
    rv = rowvals(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    @sync for r in RangeIterator(size(C,2), Threads.nthreads())
        Threads.@spawn for k in r
            @inbounds for col = 1:size(A, 2)
                tmp = zero(eltype(C))
                for j = getcolptr(A)[col]:(getcolptr(A)[col + 1] - 1)
                    tmp += adjoint(nzv[j])*B[rv[j],k]
                end
                C[col,k] += tmp * α
            end
        end
    end
    C
end
function mul!(C::StridedVecOrMat, adjA::Adjoint{<:Any,<:ThreadedSparseMatrixCSC}, B::StridedVector, α::Number, β::Number)
    A = adjA.parent
    size(A, 2) == size(C, 1) || throw(DimensionMismatch())
    size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    @assert size(B,2)==1
    colptrA = getcolptr(A)
    nzv = nonzeros(A)
    rv = rowvals(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    @sync for r in RangeIterator(size(A,2), Threads.nthreads())
        Threads.@spawn @inbounds for col = r
            tmp = zero(eltype(C))
            for j = getcolptr(A)[col]:(getcolptr(A)[col + 1] - 1)
                tmp += adjoint(nzv[j])*B[rv[j]]
            end
            C[col] += tmp * α
        end
    end
    C
end

function mul!(C::StridedVecOrMat, transA::Transpose{<:Any,<:ThreadedSparseMatrixCSC}, B::AdjOrTransDenseMatrix, α::Number, β::Number)
    A = transA.parent
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
            @inbounds for col = 1:size(A, 2)
                tmp = zero(eltype(C))
                for j = getcolptr(A)[col]:(getcolptr(A)[col + 1] - 1)
                    tmp += transpose(nzv[j])*B[rv[j],k]
                end
                C[col,k] += tmp * α
            end
        end
    end
    C
end
function mul!(C::StridedVecOrMat, transA::Transpose{<:Any,<:ThreadedSparseMatrixCSC}, B::StridedVector, α::Number, β::Number)
    A = transA.parent
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
        Threads.@spawn @inbounds for col = r
            tmp = zero(eltype(C))
            for j = getcolptr(A)[col]:(getcolptr(A)[col + 1] - 1)
                tmp += transpose(nzv[j])*B[rv[j]]
            end
            C[col] += tmp * α
        end
    end
    C
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
    @sync for r in RangeIterator(size(A,2), Threads.nthreads())
        Threads.@spawn for col in r
            @inbounds for k=getcolptr(A)[col]:(getcolptr(A)[col+1]-1)
                j = rv[k]
                αv = nzv[k]*α
                for multivec_row=1:mX
                    C[multivec_row, col] += X[multivec_row, j] * αv
                end
            end
        end
    end
    C
end

end # module
