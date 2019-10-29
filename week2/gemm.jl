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

# block Matrix times block column vector
#   Cj := A * Bj + Cj
# with inner A * Cj computed using block dot products:
#   (A * Bj)_i = Ai^T Bj
function mygemm_jip_pji(C, A, B, NB=4, MB=4, KB=4)
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for J = 1:NB:n
    J2 = min(n, J+NB-1)
    for I = 1:MB:m
      I2 = min(m, I+MB-1)
      for p = 1:KB:k
        # @views C[I:I2, J:J2] += A[I:I2, :] * B[:, J:J2]
        @views mygemm_pji(C[I:I2, J:J2], A[I:I2, :], B[:, J:J2])
      end
    end
  end
end
