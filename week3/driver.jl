using LinearAlgebra
using UnicodePlots
include("gemm.jl")

function plot_data(ns, data)
  scaling = 2 * ns.^3

  minv = floatmax()
  maxv = floatmin()
  for (s, v) = pairs(data)
    # if s != :ref
      minv = min(minv, minimum(scaling ./ v.min))
      maxv = max(maxv, maximum(scaling ./ v.min))
    # end
  end

  plt = lineplot(ns, scaling ./ data.ref.min, xlim = (ns[1], ns[end]),
                 ylim = (minv, maxv), title="max GFLOPS", xlabel="n",
                 ylabel="time", name = "ref")

  for (s, v) = pairs(data)
    if s != :ref
      lineplot!(plt, ns, scaling ./ v.min, name = string(s))
    end
  end
  plt
end

using BenchmarkTools

function run(ns, nsamples = 10)

  ref = (min=zeros(length(ns)), avg = zeros(length(ns)))
  jip_pji = (min=zeros(length(ns)), avg = zeros(length(ns)))
  pji = (min=zeros(length(ns)), avg = zeros(length(ns)))
  jpi = (min=zeros(length(ns)), avg = zeros(length(ns)))
  ji_4x4 = (min=zeros(length(ns)), avg = zeros(length(ns)))
  ji = (min=zeros(length(ns)), avg = zeros(length(ns)))

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
    ref.min[i] = ref.avg[i] = t / nsamples
    @show ref.min[i], 2*m^3 / ref.min[i]

    #=
    C .= C1
    t = @elapsed for n = 1:nsamples
      mygemm_jpi(C, A, B)
    end
    jpi.min[i] = jpi.avg[i] = t / nsamples
    @show jpi.min[i], 2m^3/jpi.min[i]
    @show extrema(C - C2)
    =#

    #=
    t = @elapsed for n = 1:nsamples
      mygemm_pji(C, A, B)
    end
    pji.min[i] = pji.avg[i] = t / nsamples
    @show pji.min[i], 2m^3/pji.min[i]
    =#

    #=
    t = @elapsed for n = 1:nsamples
      mygemm_jip_pji(C, A, B, 4, 4)
    end
    jip_pji.min[i] = jip_pji.avg[i] = t / nsamples
    @show jip_pji.min[i], 2m^3/jip_pji.min[i]
    =#

    #=
    C .= C1
    t = @elapsed for n = 1:nsamples
      mygemm_ji_4x4(C, A, B)
    end
    ji_4x4.min[i] = ji_4x4.avg[i] = t / nsamples
    @show ji_4x4.min[i], 2m^3/ji_4x4.min[i]
    @show extrema(C - C2)
    =#

    C .= C1
    t = @elapsed for n = 1:nsamples
      mygemm_ji(C, A, B, Val(8), Val(4))
    end
    ji.min[i] = ji.avg[i] = t / nsamples
    @show ji.min[i], 2m^3/ji.min[i]
    @show extrema(C - C2)
  end

  data = (ref = ref,
          # jpi = jpi,
          # pji = pji,
          # jip_pji = jip_pji,
          # ji_4x4 = ji_4x4,
          ji = ji,
         )

  return data
end

ns = 48:48:1500
# data = run(ns)
plt = plot_data(ns, data)
