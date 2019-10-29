include("gemm.jl")

using UnicodePlots

function plot_data(ns, data)
  scaling = 2 * ns.^3

  minv = floatmax()
  maxv = floatmin()
  for (s, v) = pairs(data)
    if s == :ref
      continue
    end
    minv = min(minv, minimum(scaling ./ v.min))
    maxv = max(maxv, maximum(scaling ./ v.min))
  end

  plt = lineplot(ns, scaling ./ data.ref.min, xlim = (ns[1], ns[end]),
                 ylim = (minv, maxv), title="max GFLOPS", xlabel="n",
                 ylabel="time", name = "ref")

  for (s, v) = pairs(data)
    if s == :ref
      continue
    end
    lineplot!(plt, ns, scaling ./ v.min, name = string(s))
  end
  plt
end

using BenchmarkTools

function run(ns, nsamples = 2)

  ref = (min=zeros(length(ns)), avg = zeros(length(ns)))
  jip_pji = (min=zeros(length(ns)), avg = zeros(length(ns)))
  jip = (min=zeros(length(ns)), avg = zeros(length(ns)))

  for (i, n) = enumerate(ns)
    @show i,n
    m = k = n
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    Cref = C + A * B

    bref = @benchmark $C += $A * $B samples=nsamples
    ref.min[i] = minimum(bref.times)
    ref.avg[i] = sum(bref.times) / length(bref.times)

    bjip = @benchmark mygemm_jip($C, $A, $B) samples=nsamples
    jip.min[i] = minimum(bjip.times)
    jip.avg[i] = sum(bjip.times) / length(bjip.times)

    bjip_pji = @benchmark mygemm_jip_pji($C, $A, $B) samples=nsamples
    jip_pji.min[i] = minimum(bjip_pji.times)
    jip_pji.avg[i] = sum(bjip_pji.times) / length(bjip_pji.times)
  end

  data = (ref = ref, jip_pji = jip_pji, jip = jip)

  return data
end

ns = 50:50:1000
data = run(ns)
plt = plot_data(ns, data)
