using SIMD

# Rank one update
#   C := C + ap * bp^T
# with rank-1 updates computed using axpy with vector ap
#   (ap * bp^T)_j = bpj * ap
function mygemm_pji(C, A, B)
  # C = C + A * B
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for p = 1:k
    for j = 1:n
      for i = 1:m
        C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

function mygemm_jpi(C, A, B)
  # C = C + A * B
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for j = 1:n
    for p = 1:k
      for i = 1:m
        C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# block Matrix times block column vector
#   Cj := A * Bj + Cj
# with inner A * Cj computed using block dot products:
#   (A * Bj)_i = Ai^T Bj
function mygemm_jip_pji(C, A, B, NB=4, MB=4)
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for J = 1:NB:n
    J2 = min(n, J+NB-1)
    for I = 1:MB:m
      I2 = min(m, I+MB-1)
      @views mygemm_pji(C[I:I2, J:J2], A[I:I2, :], B[:, J:J2])
    end
  end
end


# block Matrix times block column vector
#   Cj := A * Bj + Cj
# with inner A * Cj computed using block dot products:
#   (A * Bj)_i = Ai^T Bj
function mygemm_ji_4x4(C, A, B)
  (m, n) = size(C)
  (~, k) = size(A)
  @assert mod(m, 4) == 0
  @assert mod(n, 4) == 0
  @inbounds for J = 1:4:n
    for I = 1:4:m
      @views mygemm_pji_4x4(k,
                            pointer(C, I + (J-1) * m),
                            pointer(A, I),
                            B,
                            m,
                            m,
                            J)
      # @views mygemm_pji_4x4(k,
      #                       pointer(C) + I-1 + (J-1)*m,
      #                       pointer(A) + I-1,
      #                       pointer(V) + B[:, J:J2], m, m, k)
    end
  end
end

function mygemm_pji_4x4(k, C, A, B, ldc, lda, j)
  T = eltype(A)

  c1 = vload(Vec{4, T}, C + 0ldc*sizeof(T))
  c2 = vload(Vec{4, T}, C + 1ldc*sizeof(T))
  c3 = vload(Vec{4, T}, C + 2ldc*sizeof(T))
  c4 = vload(Vec{4, T}, C + 3ldc*sizeof(T))

  @inbounds for p = 1:k
    αv = vload(Vec{4, T}, A + (p-1)*lda*sizeof(T))

    # β = Vec{4, T}(B[p, 1])
    β = B[p, j + 0]
    c1 += β * αv

    # β = Vec{4, T}(B[p, 2])
    β = B[p, j + 1]
    c2 += β * αv

    # β = Vec{4, T}(B[p, 3])
    β = B[p, j + 2]
    c3 += β * αv

    # β = Vec{4, T}(B[p, 4])
    β = B[p, j + 3]
    c4 += β * αv
  end
  vstore(c1, C + 0ldc*sizeof(T))
  vstore(c2, C + 1ldc*sizeof(T))
  vstore(c3, C + 2ldc*sizeof(T))
  vstore(c4, C + 3ldc*sizeof(T))
end

function mygemm_ji(C, A, B, ::Val{MB}, ::Val{NB}) where {NB, MB}
  (m, n) = size(C)
  (~, k) = size(A)
  @assert mod(m, MB) == 0
  @assert mod(n, NB) == 0
  @inbounds for J = 1:NB:n
    for I = 1:MB:m
      @views mygemm_p(k,
                      pointer(C, I + (J-1) * m),
                      pointer(A, I),
                      B,
                      m,
                      m,
                      J, Val(MB), Val(NB))
    end
  end
end

function mygemm_p(k, C, A, B, ldc, lda, j, ::Val{MB}, ::Val{NB}) where {MB, NB}
  T = eltype(A)

  c1 = vload(Vec{MB, T}, C + 0ldc*sizeof(T))
  c2 = vload(Vec{MB, T}, C + 1ldc*sizeof(T))
  c3 = vload(Vec{MB, T}, C + 2ldc*sizeof(T))
  c4 = vload(Vec{MB, T}, C + 3ldc*sizeof(T))

  @inbounds for p = 1:k
    αv = vload(Vec{MB, T}, A + (p-1)*lda*sizeof(T))

    β = B[p, j + 0]
    # c1 += β * αv
    c1 = muladd(c1, β, αv)

    if NB > 2
      β = B[p, j + 1]
      # c2 += β * αv
      c2 = muladd(c2, β, αv)
    end

    if NB > 3
      β = B[p, j + 2]
      # c3 += β * αv
      c3 = muladd(c3, β, αv)
    end

    if NB > 4
      β = B[p, j + 3]
      # c4 += β * αv
      c4 = muladd(c4, β, αv)
    end

    if NB > 5
      error("NB too big")
    end
  end
  vstore(c1, C + 0ldc*sizeof(T))
  vstore(c2, C + 1ldc*sizeof(T))
  vstore(c3, C + 2ldc*sizeof(T))
  vstore(c4, C + 3ldc*sizeof(T))
end

