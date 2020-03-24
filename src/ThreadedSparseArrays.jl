module ThreadedSparseArrays

using LinearAlgebra
import LinearAlgebra: mul!
using SparseArrays
import SparseArrays: AdjOrTransStridedOrTriangularMatrix, getcolptr

# * ThreadedSparseMatrixCSC

"""
    ThreadedSparseMatrixCSC(A)

Thin container around `A::SparseMatrixCSC` that will enable certain
threaded multiplications of `A` with dense matrices.
"""
struct ThreadedSparseMatrixCSC{Tv,Ti,At} <: AbstractSparseMatrix{Tv,Ti}
    A::At
    ThreadedSparseMatrixCSC(A::At) where {Tv,Ti,At<:AbstractSparseMatrix{Tv,Ti}} =
        new{Tv,Ti,At}(A)
end

Base.size(A::ThreadedSparseMatrixCSC, args...) = size(A.A, args...)
Base.eltype(A::ThreadedSparseMatrixCSC) = eltype(A.A)
Base.getindex(A::ThreadedSparseMatrixCSC, args...) = getindex(A.A, args...)

# Need to override printing
# Need to forward findnz, etc

for f in [:rowvals, :nonzeros, :getcolptr]
    @eval SparseArrays.$(f)(A::ThreadedSparseMatrixCSC) = SparseArrays.$(f)(A.A)
end

function mul!(C::StridedVecOrMat, adjA::Adjoint{<:Any,<:ThreadedSparseMatrixCSC}, B::Union{StridedVector,AdjOrTransStridedOrTriangularMatrix}, α::Number, β::Number)
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
    for k = 1:size(C, 2)
        Threads.@threads for col = 1:size(A, 2)
            @inbounds begin
                tmp = zero(eltype(C))
                for j = colptrA[col]:(colptrA[col+1] - 1)
                    tmp += adjoint(nzv[j])*B[rv[j],k]
                end
                C[col,k] += α*tmp
            end
        end
    end
    C
end

function mul!(C::StridedVecOrMat, transA::Transpose{<:Any,<:ThreadedSparseMatrixCSC}, B::Union{StridedVector,AdjOrTransStridedOrTriangularMatrix}, α::Number, β::Number)
    A = transA.parent
    size(A, 2) == size(C, 1) || throw(DimensionMismatch())
    size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = nonzeros(A)
    rv = rowvals(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    Threads.@threads for k = 1:size(C, 2)
        @inbounds for col = 1:size(A, 2)
            tmp = zero(eltype(C))
            for j = getcolptr(A)[col]:(getcolptr(A)[col + 1] - 1)
                tmp += transpose(nzv[j])*B[rv[j],k]
            end
            C[col,k] += tmp * α
        end
    end
    C
end

function mul!(C::StridedVecOrMat, X::AdjOrTransStridedOrTriangularMatrix, A::ThreadedSparseMatrixCSC, α::Number, β::Number)
    mX, nX = size(X)
    nX == size(A, 1) || throw(DimensionMismatch())
    mX == size(C, 1) || throw(DimensionMismatch())
    size(A, 2) == size(C, 2) || throw(DimensionMismatch())
    rv = rowvals(A)
    nzv = nonzeros(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    # Threads.@threads for col = 1:size(A, 2)
    #     @inbounds for multivec_row=1:mX, k=getcolptr(A)[col]:(getcolptr(A)[col+1]-1)
    #         C[multivec_row, col] += α * X[multivec_row, rv[k]] * nzv[k] # perhaps suboptimal position of α?
    #     end
    # end
    Threads.@threads for col = 1:size(A, 2)
        @inbounds for k=getcolptr(A)[col]:(getcolptr(A)[col+1]-1)
            j = rv[k]
            αv = nzv[k]*α
            for multivec_row=1:mX
                C[multivec_row, col] += X[multivec_row, j] * αv
            end
        end
    end

    C
end

# * ThreadedColumnizedSparseMatrix

"""
    ThreadedColumnizedSparseMatrix(columns, m, n)

Sparse matrix of size `m×n` where the `columns` are stored separately,
enabling threaded multiplication. Seems faster than
[`ThreadedSparseMatrixCSC`](@ref) for some use cases.
"""
struct ThreadedColumnizedSparseMatrix{Tv,Ti,Columns} <: AbstractSparseMatrix{Tv,Ti}
    columns::Columns
    m::Int
    n::Int
    ThreadedColumnizedSparseMatrix(::Type{Tv}, ::Type{Ti}, columns::Columns, m, n) where {Tv,Ti,Columns} =
        new{Tv,Ti,Columns}(columns, m, n)
end

function ThreadedColumnizedSparseMatrix(A::AbstractSparseMatrix{Tv,Ti}) where {Tv,Ti}
    m,n = size(A)
    Column = typeof(A[:,1])
    columns = Column[A[:,j] for j = 1:n]
    ThreadedColumnizedSparseMatrix(Tv, Ti, columns, m, n)
end

Base.size(A::ThreadedColumnizedSparseMatrix) = (A.m,A.n)
Base.size(A::ThreadedColumnizedSparseMatrix,i) = size(A)[i]
Base.getindex(A::ThreadedColumnizedSparseMatrix, i, j) = A.columns[j][i]

function LinearAlgebra.mul!(A::AbstractMatrix, B::AbstractMatrix, C::ThreadedColumnizedSparseMatrix,
              α::Number=true, β::Number=false)
    Threads.@threads for j = 1:C.n
        LinearAlgebra.mul!(view(A, :, j), B, C.columns[j], α, β)
    end
    A
end

export ThreadedSparseMatrixCSC, ThreadedColumnizedSparseMatrix

end # module
