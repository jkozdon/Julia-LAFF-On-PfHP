using SIMD

function mygemm!(C, A, B,
                 ::Val{MC}, ::Val{NC}, ::Val{KC},
                 ::Val{MR}, ::Val{NR},
                ) where {MC, NC, KC, MR, NR}
  (m, n) = size(C)
  (~, k) = size(A)
  @assert mod(m, MR) == 0
  @assert mod(n, NR) == 0
  @inbounds begin
    for J = 1:NC:n
      J2 = min(n, J-1+NC)
      for P = 1:KC:k
        P2 = min(k, P-1+KC)
        for I = 1:MC:m
          I2 = min(n, I-1+MC)
          mygemm_NRxMR!(view(C, I:I2, J:J2), m,
                        view(A, I:I2, P:P2), m,
                        view(B, P:P2, J:J2), k,
                        Val(MR), Val(NR))
        end
      end
    end
  end
end


function mygemm_NRxMR!(C, ldc, A, lda, B, ldb, ::Val{MR}, ::Val{NR}) where {MR, NR}
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for J = 1:NR:n
    for I = 1:MR:m
      @views k_mygemm(k,
                      pointer(C, I + (J-1) * m), ldc,
                      pointer(A, I), lda,
                      pointer(B, 1 + (J-1) * k), ldb,
                      Val(MR), Val(NR))
    end
  end
end

function k_mygemm(k,
                  C, ldc,
                  A, lda,
                  B, ldb,
                  ::Val{MR}, ::Val{NR}) where {MR, NR}
  T = eltype(A)
  NR > 4 && error("NR too big")

  NR>0 && (c1 = vload(Vec{MR, T}, C + 0ldc*sizeof(T)))
  NR>1 && (c2 = vload(Vec{MR, T}, C + 1ldc*sizeof(T)))
  NR>2 && (c3 = vload(Vec{MR, T}, C + 2ldc*sizeof(T)))
  NR>3 && (c4 = vload(Vec{MR, T}, C + 3ldc*sizeof(T)))

  @inbounds for p = 1:k
    αv = vload(Vec{MR, T}, A + (p-1)*lda*sizeof(T))

    β = unsafe_load(B + (p-1 + 0ldb)*sizeof(T))
    c1 = muladd(β, αv, c1)

    if NR > 1
      β = unsafe_load(B + (p-1 + 1ldb)*sizeof(T))
      c2 = muladd(β, αv, c2)
    end

    if NR > 2
      β = unsafe_load(B + (p-1 + 2ldb)*sizeof(T))
      c3 = muladd(β, αv, c3)
    end

    if NR > 3
      β = unsafe_load(B + (p-1 + 3ldb)*sizeof(T))
      c4 = muladd(β, αv, c4)
    end
  end
  NR>0 && vstore(c1, C + 0ldc*sizeof(T))
  NR>1 && vstore(c2, C + 1ldc*sizeof(T))
  NR>2 && vstore(c3, C + 2ldc*sizeof(T))
  NR>3 && vstore(c4, C + 3ldc*sizeof(T))
end

