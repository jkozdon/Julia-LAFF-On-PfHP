# Matrix times row vector
#   cj^T := ai^T * B + cj^T
# with inner ai^T * B computed using dot products:
#   (ai^T * B)_j = ai^T * bj
function mygemm_ijp(C, A, B)
  # C = C + A * B
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for i = 1:m
    for j = 1:n
      for p = 1:k
        C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# Matrix times row vector
#   cj^T := ai^T * B + cj^T
# with inner ai^T * B computed using axpy:
#   ai^T * B = [ai1 * b1^T;
#               ai2 * b2^T;
#               ...]
function mygemm_ipj(C, A, B)
  # C = C + A * B
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for i = 1:m
    for p = 1:k
      for j = 1:n
        C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# Rank one update
#   C := C + ap * bp^T
# with rank-1 updates computed using axpy with vector bp
#   (ap * bp^T)_i^T = aip * bp^T
function mygemm_pij(C, A, B)
  # C = C + A * B
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for p = 1:k
    for i = 1:m
      for j = 1:n
        C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

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

# Matrix times column vector
#   cj := A * bj + cj
# with inner A * bj computed using axpy:
#   A * bj = [a1 * b1j | (a2 * b2j | ... ]
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

# Matrix times column vector
#   cj := A * bj + cj
# with inner A * bj computed using dot products:
#   (A * bj)_i = ai^T bj
function mygemm_jip(C, A, B)
  # C = C + A * B
  (m, n) = size(C)
  (~, k) = size(A)
  @inbounds for j = 1:n
    for i = 1:m
      for p = 1:k
        C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

