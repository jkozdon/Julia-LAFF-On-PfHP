using SIMD

function mygemm!(C, A, B, ::Val{MB}, ::Val{NB}) where {NB, MB}
  (m, n) = size(C)
  (~, k) = size(A)
  @assert mod(m, MB) == 0
  @assert mod(n, NB) == 0
  @inbounds for J = 1:NB:n
    for I = 1:MB:m
      @views k_mygemm(k,
                      pointer(C, I + (J-1) * m),
                      pointer(A, I),
                      B,
                      m,
                      m,
                      J, Val(MB), Val(NB))
    end
  end
end

function k_mygemm(k, C, A, B, ldc, lda, j, ::Val{MB}, ::Val{NB}) where {MB, NB}
  T = eltype(A)
  NB > 4 && error("NB too big")

  NB>0 && (c1 = vload(Vec{MB, T}, C + 0ldc*sizeof(T)))
  NB>1 && (c2 = vload(Vec{MB, T}, C + 1ldc*sizeof(T)))
  NB>2 && (c3 = vload(Vec{MB, T}, C + 2ldc*sizeof(T)))
  NB>3 && (c4 = vload(Vec{MB, T}, C + 3ldc*sizeof(T)))

  @inbounds for p = 1:k
    αv = vload(Vec{MB, T}, A + (p-1)*lda*sizeof(T))

    β = B[p, j + 0]
    c1 = muladd(β, αv, c1)

    if NB > 1
      β = B[p, j + 1]
      c2 = muladd(β, αv, c2)
    end

    if NB > 2
      β = B[p, j + 2]
      c3 = muladd(β, αv, c3)
    end

    if NB > 3
      β = B[p, j + 3]
      c4 = muladd(β, αv, c4)
    end
  end
  NB>0 && vstore(c1, C + 0ldc*sizeof(T))
  NB>1 && vstore(c2, C + 1ldc*sizeof(T))
  NB>2 && vstore(c3, C + 2ldc*sizeof(T))
  NB>3 && vstore(c4, C + 3ldc*sizeof(T))
end

