using LinearAlgebra
using UnicodePlots
include("gemm.jl")

function plot_data(ns, data)
  scaling = 2 * ns.^3

  minv = floatmax()
  maxv = floatmin()
  for (s, v) = pairs(data)
    # if s != :ref
    minv = min(minv, minimum(scaling ./ v))
    maxv = max(maxv, maximum(scaling ./ v))
  # end
  end

  plt = lineplot(ns, scaling ./ data.ref, xlim = (ns[1], ns[end]),
                 ylim = (minv, maxv), title="max GFLOPS", xlabel="n",
                 ylabel="time", name = "ref")

  for (s, v) = pairs(data)
    if s != :ref
      lineplot!(plt, ns, scaling ./ v, name = string(s))
    end
  end
  plt
end

using BenchmarkTools

function run(ns, nsamples = 10)
  ref = zeros(length(ns))
  ji_reg = zeros(length(ns))
  ji = zeros(length(ns))

  for (i, n) = enumerate(ns)
    println()
    @show i,n
    m = k = n
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    C2 = copy(C)
    C1 = copy(C)

    t = @elapsed for n = 1:nsamples
      mul!(C2, A, B, 1, 1)
    end
    ref[i] = t / nsamples
    @show ref[i], 2*m^3 / ref[i]

    C .= C1
    t = @elapsed for n = 1:nsamples
      mygemm_NRxMR!(C, m, A, m, B, k, Val(8), Val(4))
    end
    ji_reg[i] = t / nsamples
    @show ji_reg[i], 2m^3/ji_reg[i]
    @show extrema(C - C2)

    C .= C1
    t = @elapsed for n = 1:nsamples
      mygemm!(C, A, B, Val(48), Val(256), Val(256), Val(8), Val(4))
    end
    ji[i] = t / nsamples
    @show ji[i], 2m^3/ji[i]
    @show extrema(C - C2)
  end

  data = (ref = ref,
          ji_reg = ji_reg,
          ji = ji,
         )

  return data
end

ns = 48:48:1000
data = run(ns)
plt = plot_data(ns, data)
